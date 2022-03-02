from typing import Tuple, Optional

import tensorflow as tf
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.sql import DataFrame, SparkSession


class DataProvider:
    def __init__(
        self,
        spark: SparkSession,
        ratings: DataFrame,
        cache_dir: Optional[str] = "file://tmp/petastorm/cache",
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

    def get_products(self) -> tf.data.Dataset:
        return self._get_unique_vector("product_id")

    def get_users(self) -> tf.data.Dataset:
        return self._get_unique_vector("user_id")

    def get_train_test_readers(
        self, weights=None, seed: Optional[int] = 42
    ) -> Tuple[SparkDatasetConverter, SparkDatasetConverter]:
        if weights is None:
            weights = [0.8, 0.2]
        train_df, validation_df = self._ratings.select(
            "product_id", "user_id", "rating"
        ).randomSplit(weights, seed)
        train_converter = make_spark_converter(train_df)
        validation_converter = make_spark_converter(validation_df)
        return train_converter, validation_converter
