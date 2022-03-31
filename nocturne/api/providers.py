from pathlib import Path
from typing import Optional, Tuple

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.sql import DataFrame, SparkSession

from nocturne.api.utils import get_num_executors


def default_cache_dir():  # pragma: no cover
    """
    Provides default petastorm cache dir with if switch for Databricks environment
    :return:
    """
    if Path("/dbfs/").exists():
        return "file:///dbfs/tmp/petastorm/cache"
    else:
        return "file:///tmp/petastorm/cache"


class ConverterProvider:
    DEFAULT_NUM_REPARTITIONS = 8

    def __init__(
        self,
        spark: SparkSession,
        cache_dir: str,
    ):
        self._spark = spark
        self._spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, cache_dir)

    def get_dataset_converter(self, df: DataFrame, with_repartition: Optional[bool] = True) -> SparkDatasetConverter:

        # sometimes provided dataset is not properly or equally distributed across executor nodes
        # such cases will lead to runtime failures in horovod
        # therefore it's better to ensure that distribution of data is appropriate
        if with_repartition:
            num_partitions = self.DEFAULT_NUM_REPARTITIONS * get_num_executors(self._spark)
            _df = df.repartition(num_partitions)
        else:
            _df = df

        converter = make_spark_converter(_df)

        return converter


class TrainValidationProvider:
    def __init__(
        self,
        spark: SparkSession,
        df: DataFrame,
        petastorm_cache_dir: Optional[str] = default_cache_dir(),
    ):
        self._df = df
        self._provider = ConverterProvider(spark, petastorm_cache_dir)

    def get_train_validation_converters(
        self, weights: Optional[tuple] = (0.7, 0.3), seed: Optional[int] = 42
    ) -> Tuple[SparkDatasetConverter, SparkDatasetConverter]:
        train_df, validation_df = self._df.randomSplit(list(weights), seed)
        train_converter = self._provider.get_dataset_converter(train_df)
        validation_converter = self._provider.get_dataset_converter(validation_df)
        return train_converter, validation_converter
