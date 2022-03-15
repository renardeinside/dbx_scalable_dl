import logging
from dataclasses import dataclass
from typing import Optional, Dict

import mlflow
import numpy as np
from petastorm.spark import SparkDatasetConverter


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
    product_ids: np.array
    user_ids: np.array
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
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    @staticmethod
    def setup_mlflow_properties(mlflow_info: MlflowInfo):
        import os

        # this is required for a proper mlflow setup on the worker nodes
        if mlflow_info.databricks_api_info:  # pragma: no cover
            os.environ["DATABRICKS_HOST"] = mlflow_info.databricks_api_info.host
            os.environ["DATABRICKS_TOKEN"] = mlflow_info.databricks_api_info.token

        mlflow.set_registry_uri(mlflow_info.registry_uri)
        mlflow.set_tracking_uri(mlflow_info.tracking_uri)
        mlflow.set_experiment(mlflow_info.experiment)

    @staticmethod
    def convert_to_row(data) -> Dict:
        return {"user_id": data[0], "product_id": data[1], "rating": data[2]}
