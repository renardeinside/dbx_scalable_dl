import pathlib
import shutil
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, DataFrame


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
def petastorm_cache_dir() -> str:
    cache_dir = tempfile.TemporaryDirectory().name
    yield cache_dir
    if pathlib.Path(cache_dir).exists():
        shutil.rmtree(cache_dir)


@pytest.fixture()
def sample_ratings_dataset(
    spark: SparkSession, ratings_size: Optional[int] = 1000
) -> DataFrame:
    product_ids = pd.Series([f"PID_{i:03}" for i in range(10)])
    user_ids = pd.Series([f"UID_{i:03}" for i in range(10)])
    _pdf = pd.DataFrame().from_dict(
        {
            "user_id": user_ids.sample(n=ratings_size, replace=True).tolist(),
            "product_id": product_ids.sample(n=ratings_size, replace=True).tolist(),
            "rating": np.random.randint(1, 5, size=ratings_size).astype(np.float),
        }
    )

    _sdf = spark.createDataFrame(_pdf)
    return _sdf
