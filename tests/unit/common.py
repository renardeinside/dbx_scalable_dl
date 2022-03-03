import shutil
import tempfile
import unittest

import pathlib
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
import mlflow


class TestCaseWithEnvironment(unittest.TestCase):
    warehouse_dir = tempfile.TemporaryDirectory().name
    tracking_uri = tempfile.TemporaryDirectory().name
    registry_uri = f"sqlite:///{tempfile.TemporaryDirectory().name}"
    spark: SparkSession

    @classmethod
    def setUpClass(cls) -> None:
        _builder = (
            SparkSession.builder.master("local[1]")
            .config("spark.default.parallelism", 1)
            .config("spark.sql.shuffle.partitions", 1)
            .config("spark.sql.adaptive.enabled", False)
            .config("spark.hive.metastore.warehouse.dir", cls.warehouse_dir)
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
        )

        cls.spark = configure_spark_with_delta_pip(_builder).getOrCreate()
        cls.spark.sparkContext.setLogLevel("INFO")

        mlflow.set_tracking_uri(cls.tracking_uri)
        mlflow.set_registry_uri(cls.registry_uri)

    @classmethod
    def tearDownClass(cls) -> None:
        for _dir in [
            cls.tracking_uri,
            cls.warehouse_dir,
            cls.registry_uri.replace("sqlite:///", ""),
        ]:
            if pathlib.Path(_dir).exists():
                if pathlib.Path(_dir).is_dir():
                    shutil.rmtree(_dir)
                else:
                    pathlib.Path(_dir).unlink()
