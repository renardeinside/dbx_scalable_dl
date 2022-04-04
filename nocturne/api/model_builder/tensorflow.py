from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import tensorflow as tf
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Optimizer
from pyspark.sql import SparkSession

from nocturne.api.model_builder.base import ModelBuilder, ModelBuilderInfo


@dataclass
class TensorflowModelBuilderInfo(ModelBuilderInfo):
    """
    dataclass for providing information for the Tensorflow model builder
    model is a non-compiled instance of the tf.keras.Model
    global_callbacks: callbacks that will be used across all workers
    unique_callbacks: callbacks that will be used only for the worker with hvd.rank() == 0
    compile_arguments: additional arguments to the model.compile command
    train_converter_context_arguments: additional arguments for horovod converter for train dataset
    validation_converter_context_arguments: additional arguments for horovod converter for validation dataset
    model_fit_arguments: additional arguments for model fit
    """
    model: Model
    global_callbacks: Optional[List[Callback]] = field(default_factory=list)
    unique_callbacks: Optional[List[Callback]] = field(default_factory=list)
    compile_arguments: Optional[Dict] = field(default_factory=dict)
    train_converter_context_arguments: Optional[Dict] = field(default_factory=dict)
    validation_converter_context_arguments: Optional[Dict] = field(default_factory=dict)
    model_fit_arguments: Optional[Dict] = field(default_factory=dict)


class TensorflowModelBuilder(ModelBuilder, ABC):
    """
    Abstract class for model builder implementations.
    """
    def __init__(self, spark: SparkSession, info: TensorflowModelBuilderInfo):
        self._info = info
        super().__init__(spark, info)

    @staticmethod
    def setup_gpu_properties():  # pragma: no cover
        import horovod.tensorflow as hvd
        import tensorflow as tf  # noqa

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    @abstractmethod
    def get_optimizer(self, size_multiplier: Optional[int] = 1) -> Optimizer:
        """
        this method shall return an optimizer instance.
        Since hvd.size() might be used as a variable in the optimizer initialization, it's passed as a size_multiplier.
        :param size_multiplier:
        :return: keras.optimizers.Optimizer instance
        """

    def train_dataset_preprocessor(  # noqa pragma: no cover
        self, dataset: tf.data.Dataset
    ) -> tf.data.Dataset:  # pragma: no cover
        return dataset

    def validation_dataset_preprocessor(self, dataset: tf.data.Dataset) -> tf.data.Dataset:  # noqa  # pragma: no cover
        return dataset

    def __call__(self, *args, **kwargs):
        import horovod.tensorflow as hvd
        from horovod.keras.callbacks import (
            BroadcastGlobalVariablesCallback,
        )

        self.setup_gpu_properties()

        self.logger.info("Initializing horovod")
        hvd.init()
        self.logger.info("Initializing horovod - done")
        self.logger.info(f"Horovod size: {hvd.size()}")

        distributed_optimizer = hvd.DistributedOptimizer(self.get_optimizer(hvd.size()))
        self._info.model.compile(optimizer=distributed_optimizer, **self._info.compile_arguments)

        callbacks = self._info.global_callbacks + [BroadcastGlobalVariablesCallback(0)]

        with self.train_converter.make_tf_dataset(
            batch_size=self.batch_size,
            cur_shard=hvd.rank(),
            shard_count=hvd.size(),
            **self._info.train_converter_context_arguments,
        ) as train_reader, self.validation_converter.make_tf_dataset(
            batch_size=self.batch_size,
            cur_shard=hvd.rank(),
            shard_count=hvd.size(),
            **self._info.validation_converter_context_arguments,
        ) as validation_reader:
            train_steps_per_epoch = self.get_steps_per_epoch(len(self.train_converter), hvd.size())
            validation_steps_per_epoch = self.get_steps_per_epoch(len(self.validation_converter), hvd.size())

            self.logger.info(
                f"""Train information:
                        total size: {len(self.train_converter)}
                        steps per epoch: {train_steps_per_epoch}
                    """
            )

            self.logger.info(
                f"""Validation information:
                        total size: {len(self.validation_converter)}
                        steps per epoch: {validation_steps_per_epoch}
                    """
            )

            train_ds = self.train_dataset_preprocessor(train_reader)
            validation_ds = self.validation_dataset_preprocessor(validation_reader)

            if (hvd.rank() == 0) and self._info.unique_callbacks:
                self.logger.info(
                    f"Appending the following callbacks to the zero-rank worker: {self._info.unique_callbacks}"
                )
                callbacks.extend(self._info.unique_callbacks)

            self._info.model.fit(
                train_ds,
                epochs=self.num_epochs,
                steps_per_epoch=train_steps_per_epoch,
                validation_steps=validation_steps_per_epoch,
                validation_data=validation_ds,
                callbacks=callbacks,
                verbose=True,
                **self._info.model_fit_arguments
            )
