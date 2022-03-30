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
        self.num_executors: int = get_num_executors(spark)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._info = info

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

    def get_steps_per_epoch(
        self, total_elements: int, multiplier: Optional[int] = 1
    ) -> int:
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
        pass

    def verify_serialization(self):
        with tempfile.TemporaryFile() as t_file:
            dump(self, t_file)
