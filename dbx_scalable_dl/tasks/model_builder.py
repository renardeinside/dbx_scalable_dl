from typing import Callable, Dict, Optional

import mlflow
from pyspark.sql import DataFrame

from dbx_scalable_dl.callbacks import MLflowLoggingCallback
from dbx_scalable_dl.data_provider import DataProvider
from dbx_scalable_dl.common import Job
from dbx_scalable_dl.model import BasicModel
import horovod.tensorflow as hvd
import tensorflow as tf
from tensorflow_recommenders.models import Model
from horovod.keras.callbacks import (
    BroadcastGlobalVariablesCallback,
    MetricAverageCallback,
)


class ModelBuilderTask(Job):
    def _get_parallelism_level(self) -> int:
        tracker = self.spark.sparkContext._jsc.sc().statusTracker()  # noqa
        return len(tracker.getExecutorInfos()) - 1

    @staticmethod
    def _convert_to_row(data) -> Dict:
        return {"user_id": data[0], "product_id": data[1], "rating": data[2]}

    def get_runner(self):
        from sparkdl import HorovodRunner  # noqa

        parallelism_level = self._get_parallelism_level()
        runner = HorovodRunner(np=parallelism_level, driver_log_verbosity="all")
        return runner

    def _setup_mlflow(self):
        mlflow.set_registry_uri(self.conf["mlflow"].get("registry_uri", "databricks"))
        mlflow.set_tracking_uri(self.conf["mlflow"].get("tracking_uri", "databricks"))
        mlflow.set_experiment(self.conf["mlflow"]["experiment"])

    @staticmethod
    def _setup_gpu_properties():
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    @staticmethod
    def get_model(
        provider: DataProvider, optimizer_multiplier: Optional[int] = 1
    ) -> Model:
        model = BasicModel(
            rating_weight=1.0, retrieval_weight=1.0, data_provider=provider
        )
        optimizer = hvd.DistributedOptimizer(
            tf.keras.optimizers.Adagrad(0.01 * optimizer_multiplier)
        )
        model.compile(optimizer=optimizer, experimental_run_tf_function=False)

        return model

    def get_training_function(self, provider: DataProvider) -> Callable:
        self.logger.info("Preparing the training function")
        train_converter, validation_converter = provider.get_train_test_converters()

        def runner():
            hvd.init()

            self._setup_gpu_properties()
            self._setup_mlflow()

            model = self.get_model(provider, hvd.size())

            callbacks = [BroadcastGlobalVariablesCallback(0), MetricAverageCallback()]

            with train_converter.make_tf_dataset(
                batch_size=self.conf["batch_size"],
                cur_shard=hvd.rank(),
                shard_count=hvd.size(),
            ) as train_reader, validation_converter.make_tf_dataset(
                batch_size=self.conf["batch_size"],
                cur_shard=hvd.rank(),
                shard_count=hvd.size(),
            ) as validation_reader:
                train_ds: tf.data.Dataset = train_reader.map(self._convert_to_row)
                validation_ds: tf.data.Dataset = validation_reader.map(
                    self._convert_to_row
                )

                train_steps_per_epoch = len(train_converter) // (
                    self.conf["batch_size"] * hvd.size()
                )
                validation_steps_per_epoch = max(
                    1,
                    len(validation_converter) // (self.conf["batch_size"] * hvd.size()),
                )

                if hvd.rank() == 0:
                    logging_callback = MLflowLoggingCallback(self.conf["model_name"])
                    callbacks.append(logging_callback)

                model.fit(
                    train_ds,
                    epochs=self.conf["num_epochs"],
                    steps_per_epoch=train_steps_per_epoch,
                    validation_steps=validation_steps_per_epoch,
                    validation_data=validation_ds,
                    callbacks=callbacks,
                )

        return runner

    def get_ratings(self) -> DataFrame:
        return self.spark.table(f"{self.conf['database']}.{self.conf['table']}")

    def launch(self):
        self.logger.info("Starting the model building job")
        ratings = self.get_ratings()
        provider = DataProvider(self.spark, ratings, self.conf["cache_dir"])
        runner = self.get_runner()
        training_function = self.get_training_function(provider)
        self._setup_mlflow()
        with mlflow.start_run():
            runner.run(training_function)


if __name__ == "__main__":
    ModelBuilderTask().launch()
