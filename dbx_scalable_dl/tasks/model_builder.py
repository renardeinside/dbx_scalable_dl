import json
from typing import Callable

import mlflow
import tensorflow as tf
from py4j.protocol import Py4JJavaError
from pyspark.sql import DataFrame

from dbx_scalable_dl.callbacks import MLflowLoggingCallback
from dbx_scalable_dl.common import Job
from dbx_scalable_dl.data_provider import DataProvider
from dbx_scalable_dl.models import BasicModel
from dbx_scalable_dl.serialization import (
    DatabricksApiInfo,
    MlflowInfo,
    RunnerFunctionInfo,
    SerializableFunctionProvider,
)


class ModelBuilderTask(Job):
    def _get_parallelism_level(self) -> int:  # pragma: no cover
        """
        There are 2 types of parallelism available:
        - single node with one or multiple nodes - in this case we return -1
        - multiple executors with one GPU
        :return:
        """
        try:
            raw_tags = self.spark.conf.get(
                "spark.databricks.clusterUsageTags.clusterAllTags"
            )
        except Py4JJavaError:
            self.logger.info(
                "Clusters tags not found, this launch is not on the Databricks environment"
            )
            return -1

        sn_finder = [
            item
            for item in json.loads(raw_tags)
            if item["key"] == "ResourceClass" and item["value"] == "SingleNode"
        ]

        if sn_finder:
            return -1
        else:
            tracker = self.spark.sparkContext._jsc.sc().statusTracker()  # noqa
            return len(tracker.getExecutorInfos()) - 1

    def get_runner(self):  # pragma: no cover
        from sparkdl import HorovodRunner  # noqa

        parallelism_level = self._get_parallelism_level()
        runner = HorovodRunner(np=parallelism_level, driver_log_verbosity="all")
        return runner

    def get_training_function(self, info: RunnerFunctionInfo) -> Callable:
        self.logger.info("Preparing the training function")

        def runner():
            """
            Important - this function shall have no class-based dependencies that might be unserializable
            :return:
            """
            import horovod.tensorflow as hvd
            from horovod.keras.callbacks import (
                BroadcastGlobalVariablesCallback,
                MetricAverageCallback,
            )

            hvd.init()
            SerializableFunctionProvider.setup_gpu_properties()
            SerializableFunctionProvider.setup_mlflow_properties(info.mlflow_info)

            optimizer = hvd.DistributedOptimizer(
                tf.keras.optimizers.Adagrad(0.01 * hvd.size())
            )
            model = BasicModel(
                rating_weight=1.0,
                retrieval_weight=1.0,
                product_ids=info.product_ids,
                user_ids=info.user_ids,
            )

            model.compile(optimizer=optimizer, experimental_run_tf_function=False)
            callbacks = [BroadcastGlobalVariablesCallback(0), MetricAverageCallback()]

            with info.train_converter.make_tf_dataset(
                batch_size=info.batch_size,
                cur_shard=hvd.rank(),
                shard_count=hvd.size(),
            ) as train_reader, info.validation_converter.make_tf_dataset(
                batch_size=info.batch_size,
                cur_shard=hvd.rank(),
                shard_count=hvd.size(),
            ) as validation_reader:
                train_ds: tf.data.Dataset = train_reader.map(
                    SerializableFunctionProvider.convert_to_row
                )
                validation_ds: tf.data.Dataset = validation_reader.map(
                    SerializableFunctionProvider.convert_to_row
                )

                train_steps_per_epoch = len(info.train_converter) // (
                    info.batch_size * hvd.size()
                )
                validation_steps_per_epoch = max(
                    1,
                    len(info.validation_converter) // (info.batch_size * hvd.size()),
                )

                if hvd.rank() == 0:
                    logging_callback = MLflowLoggingCallback(info.model_name)
                    callbacks.append(logging_callback)

                model.fit(
                    train_ds,
                    epochs=info.num_epochs,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_steps=validation_steps_per_epoch,
                    validation_data=validation_ds,
                    callbacks=callbacks,
                )

        return runner

    def get_ratings(self) -> DataFrame:
        return self.spark.table(f"{self.conf['database']}.{self.conf['table']}")

    def get_provider(self, ratings: DataFrame) -> DataProvider:
        return DataProvider(self.spark, ratings, self.conf["cache_dir"])

    def _get_databricks_api_info(self) -> DatabricksApiInfo:  # pragma: no cover
        host = (
            self.dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .apiUrl()
            .getOrElse(None)
        )
        token = (
            self.dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .apiToken()
            .getOrElse(None)
        )
        return DatabricksApiInfo(host=host, token=token)

    def prepare_info(self, provider: DataProvider) -> RunnerFunctionInfo:
        train_converter, validation_converter = provider.get_train_test_converters()
        _info = RunnerFunctionInfo(
            batch_size=self.conf["batch_size"],
            model_name=self.conf["model_name"],
            num_epochs=self.conf["num_epochs"],
            product_ids=provider.dataset_to_numpy_array(
                provider.get_unique_product_ids()
            ),
            user_ids=provider.dataset_to_numpy_array(provider.get_unique_user_ids()),
            train_converter=train_converter,
            validation_converter=validation_converter,
            mlflow_info=MlflowInfo(
                registry_uri=self.conf["mlflow"].get("registry_uri", "databricks"),
                tracking_uri=self.conf["mlflow"].get("tracking_uri", "databricks"),
                experiment=self.conf["mlflow"]["experiment"],
                databricks_api_info=self._get_databricks_api_info(),
            ),
        )
        return _info

    def launch(self):
        self.logger.info("Starting the model building job")
        ratings = self.get_ratings()
        provider = self.get_provider(ratings)
        runner = self.get_runner()
        runner_info = self.prepare_info(provider)
        training_function = self.get_training_function(runner_info)
        mlflow.set_experiment(runner_info.mlflow_info.experiment)
        runner.run(training_function)


if __name__ == "__main__":
    ModelBuilderTask().launch()
