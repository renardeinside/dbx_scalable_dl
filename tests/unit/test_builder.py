from typing import Optional

import pytest
from keras.optimizer_v1 import Optimizer
from pyspark.sql import SparkSession
import tensorflow as tf
from nocturne.api.model_builder.tensorflow import (
    TensorflowModelBuilder,
    TensorflowModelBuilderInfo,
)
from nocturne.api.providers import TrainValidationProvider
from keras.callbacks import LambdaCallback
import logging


class SampleTensorflowModelBuilder(TensorflowModelBuilder):
    def get_optimizer(self, size_multiplier: Optional[int] = 1) -> Optimizer:
        return tf.keras.optimizers.Adagrad(0.01 * size_multiplier)

    def train_dataset_preprocessor(  # noqa
        self, dataset: tf.data.Dataset
    ) -> tf.data.Dataset:
        return dataset.map(lambda element: (element[0], element[1]))

    def validation_dataset_preprocessor(  # noqa
        self, dataset: tf.data.Dataset
    ) -> tf.data.Dataset:
        return self.train_dataset_preprocessor(dataset)


class UnserializableModelBuilder(SampleTensorflowModelBuilder):
    def __init__(self, spark: SparkSession, info: TensorflowModelBuilderInfo):
        self.spark = spark
        super().__init__(spark, info)


def test_builder(spark: SparkSession, petastorm_cache_dir: str):
    sample_df = spark.range(2000).toDF("X").selectExpr("X", "X * 1000 as y")

    provider = TrainValidationProvider(
        spark, sample_df, petastorm_cache_dir=f"file://{petastorm_cache_dir}"
    )

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8, input_shape=(1,)))
    model.add(tf.keras.layers.Dense(1))

    train_converter, validation_converter = provider.get_train_validation_converters()

    basic_info = TensorflowModelBuilderInfo(
        model=model,
        batch_size=100,
        num_epochs=1,
        train_converter=train_converter,
        validation_converter=validation_converter,
        compile_arguments={"loss": "mean_absolute_error"},
        unique_callbacks=[
            LambdaCallback(
                on_batch_begin=lambda batch, logs: logging.info(f"On batch {batch}")
            )
        ],
    )
    train_function = SampleTensorflowModelBuilder(spark, basic_info)
    train_function()

    no_unique_callbacks = TensorflowModelBuilderInfo(
        model=model,
        batch_size=100,
        num_epochs=1,
        train_converter=train_converter,
        validation_converter=validation_converter,
        compile_arguments={"loss": "mean_absolute_error"},
    )
    no_callbacks_func = SampleTensorflowModelBuilder(spark, no_unique_callbacks)
    no_callbacks_func()

    with pytest.raises(Exception):
        _ = UnserializableModelBuilder(spark, basic_info)
