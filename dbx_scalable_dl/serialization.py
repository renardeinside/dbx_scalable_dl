import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import mlflow
import numpy as np
import tensorflow as tf
from petastorm.spark import SparkDatasetConverter

from dbx_scalable_dl.models import BasicModel


@dataclass
class DatabricksApiInfo:
    host: str
    token: str


@dataclass
class MlflowInfo:
    registry_uri: str
    tracking_uri: str
    experiment: str
    databricks_api_info: Optional[DatabricksApiInfo] = None


@dataclass
class RunnerFunctionInfo:
    batch_size: int
    model_name: str
    num_epochs: int
    user_ids: np.array
    product_ids: np.array
    train_converter: SparkDatasetConverter
    validation_converter: SparkDatasetConverter
    mlflow_info: MlflowInfo
    logger: Optional[logging.Logger] = logging.Logger(__name__)


class SerializableFunctionProvider:
    """
    This class provides static methods for converting different objects into serializable representations
    It's required by horovod/sparkdl serialization logic
    """

    @staticmethod
    def setup_gpu_properties():  # pragma: no cover
        import horovod.tensorflow as hvd
        import tensorflow as tf  # noqa

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    @staticmethod
    def setup_mlflow_properties(mlflow_info: MlflowInfo):  # pragma: no cover
        import os

        # this is required for a proper mlflow setup on the worker nodes
        if mlflow_info.databricks_api_info:
            os.environ["DATABRICKS_HOST"] = mlflow_info.databricks_api_info.host
            os.environ["DATABRICKS_TOKEN"] = mlflow_info.databricks_api_info.token

        mlflow.set_registry_uri(mlflow_info.registry_uri)
        mlflow.set_tracking_uri(mlflow_info.tracking_uri)
        mlflow.set_experiment(mlflow_info.experiment)

    @staticmethod
    def convert_to_row(data) -> Dict:
        return {"user_id": data[0], "product_id": data[1], "rating": data[2]}

    @staticmethod
    def prepare_datasets(
        train_reader, validation_reader
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        train_ds: tf.data.Dataset = train_reader.map(
            SerializableFunctionProvider.convert_to_row
        )
        validation_ds: tf.data.Dataset = validation_reader.map(
            SerializableFunctionProvider.convert_to_row
        )
        return train_ds, validation_ds

    @staticmethod
    def get_model(product_ids: np.array, user_ids: np.array) -> BasicModel:
        _m = BasicModel(
            rating_weight=5.0,
            retrieval_weight=1.0,
            product_ids=product_ids,
            user_ids=user_ids,
        )
        return _m

    @staticmethod
    def get_steps_per_epoch(
        total_elements: int, batch_size: int, multiplier: Optional[int] = 1
    ) -> int:
        return max(1, total_elements // (batch_size * multiplier))
