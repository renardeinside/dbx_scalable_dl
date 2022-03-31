import logging
import tempfile
from abc import ABC, abstractmethod
from typing import Optional

from petastorm.spark import SparkDatasetConverter
from pyspark.sql import SparkSession
from pyspark.cloudpickle.cloudpickle_fast import dump
from nocturne.api.utils import get_num_executors
from dataclasses import dataclass


@dataclass
class ModelBuilderInfo:
    train_converter: SparkDatasetConverter
    validation_converter: SparkDatasetConverter
    batch_size: int
    num_epochs: int


class ModelBuilder(ABC):
    def __init__(self, spark: SparkSession, info: ModelBuilderInfo):
        """
        Important - super init call shall always be the LAST command of the subclass initialization
        so it can verify serializability of the whole code
        :param spark: SparkSession, active Spark session
        :param info:  ModelBuilderInfo, runtime model builder information
        """
        self.num_executors: int = get_num_executors(spark)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._info = info
        self.verify_serialization()

    @property
    def train_converter(self) -> SparkDatasetConverter:
        """
        Train dataset converter from horovod. Required
        :return:
        """
        return self._info.train_converter

    @property
    def validation_converter(self) -> SparkDatasetConverter:
        """
        Validation dataset converter from horovod. Optional
        """
        return self._info.validation_converter

    def get_steps_per_epoch(self, total_elements: int, multiplier: Optional[int] = 1) -> int:
        return max(1, total_elements // (self.batch_size * multiplier))

    @property
    def batch_size(self) -> int:
        """
        Batch size for horovod reader and model iterations
        :return:
        """
        return self._info.batch_size

    @property
    def num_epochs(self) -> int:
        """
        Num epochs for model iterations
        :return:
        """
        return self._info.num_epochs

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        This is the main method that describes which actions shall be performed during the distributed training
        """

    def verify_serialization(self):
        try:
            with tempfile.TemporaryFile() as t_file:
                dump(self, t_file)
        except Exception as e:
            self.logger.error(
                f"""
            Failed to serialize model builder functional class {self.__class__.__name__}.
            This typically means that functional class contains dependencies that cannot be serialized, for instance:
                - SparkSession
                - any other runtime-dependent objects
            Please check that these objects are not defined as class or object properties.
            """
            )
            raise e
