from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.sql import DataFrame, SparkSession


class DataProvider:

    DEFAULT_NUM_PETASTORM_PARTITIONS = 16

    def __init__(
        self,
        spark: SparkSession,
        ratings: DataFrame,
        cache_dir: Optional[str] = "file:///tmp/petastorm/cache",
    ):
        self._ratings = ratings
        spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, cache_dir)

    def _get_unique_vector(self, column_name: str) -> tf.data.Dataset:
        _values = (
            self._ratings.select(column_name)
            .distinct()
            .toPandas()[column_name]
            .map(lambda i: i.encode())
            .sort_values()
            .values
        )
        return tf.data.Dataset.from_tensor_slices(_values)

    def get_unique_product_ids(self) -> tf.data.Dataset:
        return self._get_unique_vector("product_id").cache()

    def get_unique_user_ids(self) -> tf.data.Dataset:
        return self._get_unique_vector("user_id").cache()

    @staticmethod
    def dataset_to_numpy_array(dataset) -> np.array:
        return np.array(list(dataset.as_numpy_iterator()))

    def get_train_test_converters(
        self,
        weights: Optional[Tuple[float, float]] = (0.8, 0.2),
        seed: Optional[int] = 42,
        num_partitions: Optional[int] = DEFAULT_NUM_PETASTORM_PARTITIONS,
    ) -> Tuple[SparkDatasetConverter, SparkDatasetConverter]:
        train_df, validation_df = self._ratings.select(
            "product_id", "user_id", "rating"
        ).randomSplit(list(weights), seed)
        train_converter = make_spark_converter(train_df.repartition(num_partitions))
        validation_converter = make_spark_converter(
            validation_df.repartition(num_partitions)
        )
        return train_converter, validation_converter
