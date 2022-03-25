import json
from typing import Callable, Optional, Any

import mlflow
import tensorflow as tf
from pyspark import StorageLevel
from pyspark.sql import DataFrame

from dbx_scalable_dl.callbacks import MLflowLoggingCallback
from dbx_scalable_dl.common import Job, MetricsWrapper
from dbx_scalable_dl.data_provider import DataProvider
from dbx_scalable_dl.serialization import (
    DatabricksApiInfo,
    MlflowInfo,
    RunnerFunctionInfo,
    SerializableFunctionProvider,
)
from dbx_scalable_dl.utils import LaunchEnvironment


class ModelBuilderTask(Job):
    def _get_launch_environment(self) -> LaunchEnvironment:  # pragma: no cover
        raw_tags = self.spark.conf.get(
            "spark.databricks.clusterUsageTags.clusterAllTags"
        )

        sn_tag_search = [
            item
            for item in json.loads(raw_tags)
            if item["key"] == "ResourceClass" and item["value"] == "SingleNode"
        ]
        launch_environment = (
            LaunchEnvironment.SINGLE_NODE
            if sn_tag_search
            else LaunchEnvironment.MULTI_NODE
        )
        return launch_environment

    def _get_parallelism_level(self) -> int:  # pragma: no cover
        """
        There are 2 types of parallelism available:
        - single node with one or multiple nodes - in this case we return -1
        - multiple executors with one GPU
        :return:
        """
        tracker = self.spark.sparkContext._jsc.sc().statusTracker()  # noqa
        return len(tracker.getExecutorInfos()) - 1

    def get_runner(self) -> Optional[Any]:
        if self._get_launch_environment() == LaunchEnvironment.SINGLE_NODE:
            return None
        else:  # pragma: no cover
            from sparkdl import HorovodRunner  # noqa

            parallelism_level = self._get_parallelism_level()
            self.logger.info(f"Defined parallelism level {parallelism_level}")
            runner = HorovodRunner(np=parallelism_level, driver_log_verbosity="all")
            return runner

    def get_training_function(self, info: RunnerFunctionInfo) -> Callable:
        self.logger.info("Preparing the training function")

        def single_node_runner():
            self.logger.info("Starting execution in a single node mode")
            optimizer = tf.keras.optimizers.Adagrad(0.01)
            model = SerializableFunctionProvider.get_model(
                info.product_ids, info.user_ids
            )
            model.compile(optimizer=optimizer, experimental_run_tf_function=False)

            with info.train_converter.make_tf_dataset(
                batch_size=info.batch_size,
                shuffling_queue_capacity=0,
            ) as train_reader, info.validation_converter.make_tf_dataset(
                batch_size=info.batch_size,
                shuffling_queue_capacity=0,
            ) as validation_reader:
                (
                    train_ds,
                    validation_ds,
                ) = SerializableFunctionProvider.prepare_datasets(
                    train_reader, validation_reader
                )

                train_steps_per_epoch = (
                    SerializableFunctionProvider.get_steps_per_epoch(
                        len(info.train_converter), info.batch_size
                    )
                )
                validation_steps_per_epoch = (
                    SerializableFunctionProvider.get_steps_per_epoch(
                        len(info.validation_converter), info.batch_size
                    )
                )

                model.fit(
                    train_ds,
                    epochs=info.num_epochs,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_steps=validation_steps_per_epoch,
                    validation_data=validation_ds,
                    callbacks=[MLflowLoggingCallback(info.model_name)],
                    verbose=True,
                )

        def distributed_runner():  # pragma: no cover
            """
            Important - this function shall have no class-based dependencies that might be unserializable
            """
            import horovod.tensorflow as hvd
            from horovod.keras.callbacks import (
                BroadcastGlobalVariablesCallback,
                MetricAverageCallback,
            )
            import mlflow

            mlflow.autolog(disable=True)

            info.logger.info("Initializing horovod")
            hvd.init()
            info.logger.info("Initializing horovod - done")

            SerializableFunctionProvider.setup_gpu_properties()
            SerializableFunctionProvider.setup_mlflow_properties(info.mlflow_info)

            optimizer = hvd.DistributedOptimizer(
                tf.keras.optimizers.Adagrad(0.01 * hvd.size())
            )

            info.logger.info("Compiling the model")
            model = SerializableFunctionProvider.get_model(
                info.product_ids, info.user_ids
            )
            model.compile(optimizer=optimizer, experimental_run_tf_function=False)
            info.logger.info("Compiling the model - done")

            callbacks = [BroadcastGlobalVariablesCallback(0), MetricAverageCallback()]

            with info.train_converter.make_tf_dataset(
                batch_size=info.batch_size,
                cur_shard=hvd.rank(),
                shard_count=hvd.size(),
                shuffling_queue_capacity=0,
            ) as train_reader, info.validation_converter.make_tf_dataset(
                batch_size=info.batch_size,
                cur_shard=hvd.rank(),
                shard_count=hvd.size(),
                shuffling_queue_capacity=0,
            ) as validation_reader:
                (
                    train_ds,
                    validation_ds,
                ) = SerializableFunctionProvider.prepare_datasets(
                    train_reader, validation_reader
                )

                train_steps_per_epoch = (
                    SerializableFunctionProvider.get_steps_per_epoch(
                        len(info.train_converter), info.batch_size, hvd.size()
                    )
                )
                validation_steps_per_epoch = (
                    SerializableFunctionProvider.get_steps_per_epoch(
                        len(info.validation_converter), info.batch_size, hvd.size()
                    )
                )

                info.logger.info(
                    f"""Train information:
                        total size: {len(info.train_converter)}
                        steps per epoch: {train_steps_per_epoch}
                    """
                )

                info.logger.info(
                    f"""Validation information:
                        total size: {len(info.validation_converter)}
                        steps per epoch: {validation_steps_per_epoch}
                    """
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
                    verbose=True,
                )

        runner = (
            single_node_runner
            if self._get_launch_environment() == LaunchEnvironment.SINGLE_NODE
            else distributed_runner
        )

        return runner

    def get_ratings(self) -> DataFrame:
        _df = self.spark.table(f"{self.conf['database']}.{self.conf['table']}")

        limit = self.conf.get("dataset_size_limit")
        if limit:
            total_size = _df.count()
            fraction = round(float(limit) / float(total_size), 10)
            sampling_seed = 42
            _df = _df.sample(fraction=fraction, seed=sampling_seed)
            _df = _df.persist(StorageLevel.DISK_ONLY_2)
            self.logger.info(
                f"Persisting limit-based dataset, total records: {_df.count()}"
            )
        return _df

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

        if (
            self._get_launch_environment() == LaunchEnvironment.MULTI_NODE
        ):  # pragma: no cover
            num_partitions = (
                provider.DEFAULT_NUM_PETASTORM_PARTITIONS
                * self._get_parallelism_level()
            )
        else:
            num_partitions = provider.DEFAULT_NUM_PETASTORM_PARTITIONS

        self.logger.info(
            f"Provided number of partitions for petastorm: {num_partitions}"
        )
        train_converter, validation_converter = provider.get_train_test_converters(
            num_partitions=num_partitions
        )

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
        mlflow.autolog(
            disable=True
        )  # we're disable autologger since we're collecting details in a callback
        if not runner:
            self.logger.info("Model builder is launched in a single-node context")
            training_function()
        else:  # pragma: no cover
            self.logger.info(
                "Model builder is launched in a multi-node context, using horovod runner"
            )
            runner.run(training_function)


if __name__ == "__main__":
    job = ModelBuilderTask()
    wrapper = MetricsWrapper(job)
    wrapper.launch()
