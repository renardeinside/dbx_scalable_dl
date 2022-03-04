import pathlib
import shutil
import tempfile

import mlflow
import numpy as np
import pandas as pd
import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from dataclasses import dataclass

from dbx_scalable_dl.callbacks import MLflowLoggingCallback
from dbx_scalable_dl.data_provider import DataProvider
from dbx_scalable_dl.tasks.model_builder import ModelBuilderTask


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


@pytest.fixture(scope="class")
def registered_model_info(
    spark: SparkSession, mlflow_info: MlflowInfo, petastorm_cache_dir: str
):
    model_name = "dbx_scalable_dl_test_model"
    user_ids = [f"UID_{i:03}" for i in range(10)]
    user_ids = pd.Series(user_ids)
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

    _model = ModelBuilderTask.get_model(provider)

    train_converter, _ = provider.get_train_test_converters()

    with train_converter.make_tf_dataset(batch_size=100) as train_reader:
        train_ds = train_reader.map(ModelBuilderTask._convert_to_row)
        _model.fit(
            train_ds,
            epochs=1,
            steps_per_epoch=2,
            callbacks=[MLflowLoggingCallback(model_name)],
        )

    yield RegisteredModelInfo(model_name, user_ids[0])
