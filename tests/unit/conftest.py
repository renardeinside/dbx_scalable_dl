import pathlib
import shutil
import tempfile
from typing import Callable

import mlflow
import numpy as np
import pandas as pd
import pytest
from delta import configure_spark_with_delta_pip
from flask import Flask
from flask.testing import FlaskClient
from mlflow.pyfunc.scoring_server import init
from pyspark.sql import SparkSession
from dataclasses import dataclass

from unittest.mock import MagicMock

from dbx_scalable_dl.controller import ModelController
from dbx_scalable_dl.data_provider import DataProvider
from dbx_scalable_dl.tasks.model_builder import ModelBuilderTask


class LocalRunner:
    def run(self, main: Callable):
        main()


@dataclass
class MlflowInfo:
    tracking_uri: str
    registry_uri: str


@dataclass
class RegisteredModelInfo:
    model_name: str
    sample_uid: str


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    warehouse_dir = tempfile.TemporaryDirectory().name
    _builder = (
        SparkSession.builder.master("local[1]")
        .config("spark.default.parallelism", 1)
        .config("spark.sql.shuffle.partitions", 1)
        .config("spark.sql.adaptive.enabled", False)
        .config("spark.hive.metastore.warehouse.dir", warehouse_dir)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    spark: SparkSession = configure_spark_with_delta_pip(_builder).getOrCreate()
    yield spark
    spark.stop()
    if pathlib.Path(warehouse_dir).exists():
        shutil.rmtree(warehouse_dir)


@pytest.fixture(scope="session")
def mlflow_info() -> MlflowInfo:
    tracking_uri = tempfile.TemporaryDirectory().name
    registry_uri = f"sqlite:///{tempfile.TemporaryDirectory().name}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    yield MlflowInfo(tracking_uri, registry_uri)

    if pathlib.Path(tracking_uri).exists():
        shutil.rmtree(tracking_uri)

    if pathlib.Path(registry_uri).exists():
        pathlib.Path(registry_uri).unlink()


@pytest.fixture(scope="session")
def petastorm_cache_dir() -> str:
    cache_dir = tempfile.TemporaryDirectory().name
    yield cache_dir
    if pathlib.Path(cache_dir).exists():
        shutil.rmtree(cache_dir)


@pytest.fixture(scope="session")
def user_ids() -> pd.Series:
    user_ids = [f"UID_{i:03}" for i in range(10)]
    user_ids = pd.Series(user_ids)
    return user_ids


@pytest.fixture(scope="session")
def data_provider(
    spark: SparkSession, user_ids: pd.Series, petastorm_cache_dir: str
) -> DataProvider:
    product_ids = pd.Series([f"PID_{i:03}" for i in range(10)])
    ratings_size = 1000

    _pdf = pd.DataFrame().from_dict(
        {
            "user_id": user_ids.sample(n=ratings_size, replace=True).tolist(),
            "product_id": product_ids.sample(n=ratings_size, replace=True).tolist(),
            "rating": np.random.randint(1, 5, size=ratings_size).astype(np.float),
        }
    )

    _sdf = spark.createDataFrame(_pdf)

    provider = DataProvider(
        spark, ratings=_sdf, cache_dir=f"file://{petastorm_cache_dir}"
    )
    return provider


@pytest.fixture(scope="class")
def registered_model_info(
    spark: SparkSession,
    mlflow_info: MlflowInfo,
    data_provider: DataProvider,
    user_ids: pd.Series,
):
    experiment = "dbx_test_experiment"
    model_name = "dbx_scalable_ml_test"

    builder_task = ModelBuilderTask(
        spark=spark,
        init_conf={
            "batch_size": 100,
            "num_epochs": 1,
            "model_name": model_name,
            "mlflow": {
                "experiment": experiment,
                "registry_uri": mlflow_info.registry_uri,
                "tracking_uri": mlflow_info.tracking_uri,
            },
        },
    )

    builder_task.get_ratings = MagicMock()
    builder_task.get_provider = MagicMock(return_value=data_provider)
    builder_task.get_runner = MagicMock(return_value=LocalRunner())
    builder_task._get_databricks_api_info = MagicMock(return_value=None)
    builder_task.launch()

    yield RegisteredModelInfo(model_name, user_ids[0])


@pytest.fixture()
def serving_model(registered_model_info) -> Flask:
    _model_uri = ModelController(registered_model_info.model_name).get_latest_model_uri(
        stages=("None",)
    )
    pyfunc_model = mlflow.pyfunc.load_model(_model_uri)
    app = init(pyfunc_model)
    app.config.update(
        {
            "TESTING": True,
        }
    )
    yield app


@pytest.fixture(scope="function")
def model_client(serving_model: Flask) -> FlaskClient:
    return serving_model.test_client()
