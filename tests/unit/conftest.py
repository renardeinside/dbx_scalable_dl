import pathlib
import shutil
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import pytest
from delta import configure_spark_with_delta_pip
from flask import Flask
from flask.testing import FlaskClient
from mlflow.pyfunc.scoring_server import init
from pyspark.sql import SparkSession, DataFrame
from pytest_httpserver import HTTPServer

from nocturne.controller import ModelController


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


@pytest.fixture(scope="function")
def ganglia_server(httpserver: HTTPServer) -> HTTPServer:
    resp_text = """<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
    <!DOCTYPE GANGLIA_XML [
            <!ELEMENT GANGLIA_XML (GRID|CLUSTER|HOST)*>
            <!ATTLIST GANGLIA_XML VERSION CDATA #REQUIRED>
            <!ATTLIST GANGLIA_XML SOURCE CDATA #REQUIRED>
            <!ELEMENT GRID (CLUSTER | GRID | HOSTS | METRICS)*>
            <!ATTLIST GRID NAME CDATA #REQUIRED>
            <!ATTLIST GRID AUTHORITY CDATA #REQUIRED>
            <!ATTLIST GRID LOCALTIME CDATA #IMPLIED>
            <!ELEMENT CLUSTER (HOST | HOSTS | METRICS)*>
            <!ATTLIST CLUSTER NAME CDATA #REQUIRED>
            <!ATTLIST CLUSTER OWNER CDATA #IMPLIED>
            <!ATTLIST CLUSTER LATLONG CDATA #IMPLIED>
            <!ATTLIST CLUSTER URL CDATA #IMPLIED>
            <!ATTLIST CLUSTER LOCALTIME CDATA #REQUIRED>
            <!ELEMENT HOST (METRIC)*>
            <!ATTLIST HOST NAME CDATA #REQUIRED>
            <!ATTLIST HOST IP CDATA #REQUIRED>
            <!ATTLIST HOST LOCATION CDATA #IMPLIED>
            <!ATTLIST HOST TAGS CDATA #IMPLIED>
            <!ATTLIST HOST REPORTED CDATA #REQUIRED>
            <!ATTLIST HOST TN CDATA #IMPLIED>
            <!ATTLIST HOST TMAX CDATA #IMPLIED>
            <!ATTLIST HOST DMAX CDATA #IMPLIED>
            <!ATTLIST HOST GMOND_STARTED CDATA #IMPLIED>
            <!ELEMENT METRIC (EXTRA_DATA*)>
            <!ATTLIST METRIC NAME CDATA #REQUIRED>
            <!ATTLIST METRIC VAL CDATA #REQUIRED>
            <!ATTLIST METRIC UNITS CDATA #IMPLIED>
            <!ATTLIST METRIC TN CDATA #IMPLIED>
            <!ATTLIST METRIC TMAX CDATA #IMPLIED>
            <!ATTLIST METRIC DMAX CDATA #IMPLIED>
            <!ATTLIST METRIC SLOPE (zero | positive | negative | both | unspecified) #IMPLIED>
            <!ATTLIST METRIC SOURCE (gmond) 'gmond'>
            <!ELEMENT EXTRA_DATA (EXTRA_ELEMENT*)>
            <!ELEMENT EXTRA_ELEMENT EMPTY>
            <!ATTLIST EXTRA_ELEMENT NAME CDATA #REQUIRED>
            <!ATTLIST EXTRA_ELEMENT VAL CDATA #REQUIRED>
            <!ELEMENT HOSTS EMPTY>
            <!ATTLIST HOSTS UP CDATA #REQUIRED>
            <!ATTLIST HOSTS DOWN CDATA #REQUIRED>
            <!ATTLIST HOSTS SOURCE (gmond | gmetad) #REQUIRED>
            <!ELEMENT METRICS (EXTRA_DATA*)>
            <!ATTLIST METRICS NAME CDATA #REQUIRED>
            <!ATTLIST METRICS SUM CDATA #REQUIRED>
            <!ATTLIST METRICS NUM CDATA #REQUIRED>
            <!ATTLIST METRICS UNITS CDATA #IMPLIED>
            <!ATTLIST METRICS SLOPE (zero | positive | negative | both | unspecified) #IMPLIED>
            <!ATTLIST METRICS SOURCE (gmond) 'gmond'>
            ]>
    <GANGLIA_XML VERSION="3.6.0" SOURCE="gmetad">
        <GRID NAME="unspecified" AUTHORITY="" LOCALTIME="">
            <CLUSTER NAME="cluster" LOCALTIME="" OWNER="unspecified" LATLONG="unspecified" URL="unspecified">
                <HOST NAME="some-host" IP="some-ip" REPORTED="1647112064" TN="17"
                      TMAX="20" DMAX="0" LOCATION="unspecified" GMOND_STARTED="1647109757" TAGS="">
                    <METRIC NAME="some-metric"
                            VAL="0" TYPE="double" UNITS="milliseconds" TN="17" TMAX="60" DMAX="0" SLOPE="both"
                            SOURCE="gmond">
                    </METRIC>
                </HOST>
            </CLUSTER>
        </GRID>
    </GANGLIA_XML>
    """
    httpserver.expect_request("/", method="get").respond_with_data(
        response_data=resp_text, status=HTTPStatus.OK
    )
    return httpserver
