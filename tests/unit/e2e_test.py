import pathlib
import shutil
import tempfile
import unittest

from unittest.mock import MagicMock

from typing import Callable

from dbx_scalable_dl.callbacks import MLflowLoggingCallback
from dbx_scalable_dl.tasks.data_loader import DataLoaderTask
from dbx_scalable_dl.tasks.model_builder import ModelBuilderTask
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

from dbx_scalable_dl.utils import FileLoadingContext
from dbx_scalable_dl.data_provider import DataProvider


class LocalRunner:
    def run(self, main: Callable):
        main()


class SampleJobUnitTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory().name
        self.source_url = "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Gift_Cards_5.json.gz"
        _builder = (
            SparkSession.builder.master("local[1]")
            .config("spark.default.parallelism", 1)
            .config("spark.sql.shuffle.partitions", 1)
            .config("spark.sql.adaptive.enabled", False)
            .config("spark.hive.metastore.warehouse.dir", self.test_dir)
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
        )
        self.spark: SparkSession = configure_spark_with_delta_pip(
            _builder
        ).getOrCreate()
        self.spark.sparkContext.setLogLevel("INFO")

    def test_data_loader(self):
        # feel free to add new methods to this magic mock to mock some particular functionality
        loader_conf = {
            "data_url": self.source_url,
            "database": "dbx_scalable_dl_demo",
            "table": "ratings",
        }

        self.loader_task = DataLoaderTask(
            spark=self.spark,
            init_conf=loader_conf,
        )
        self.loader_task.launch()

        self.assertIn(
            loader_conf["table"],
            [t.name for t in self.spark.catalog.listTables(loader_conf["database"])],
        )

        self.assertGreater(
            self.spark.table(
                f"{loader_conf['database']}.{loader_conf['table']}"
            ).count(),
            0,
        )

    def test_model_builder(self):
        # mlflow_uri = f"file://{tempfile.TemporaryDirectory().name}"
        mlflow_uri = None
        builder_task = ModelBuilderTask(
            spark=self.spark,
            init_conf={
                "batch_size": 100,
                "database": "dbx_scalable_dl_demo",
                "table": "ratings",
                "cache_dir": "file://" + self.test_dir,
                "num_epochs": 1,
                "model_name": "dbx_scalable_ml_test",
                "mlflow": {
                    "experiment": "dbx_test_experiment",
                    "registry_uri": mlflow_uri,
                    "tracking_uri": mlflow_uri,
                },
            },
        )

        with FileLoadingContext(self.source_url) as output_path:
            raw = DataLoaderTask._extract(self.spark, output_path)
            _df = DataLoaderTask._transform(raw)
            _df.count()

            builder_task.get_ratings = MagicMock(return_value=_df)
            builder_task.get_runner = MagicMock(return_value=LocalRunner())
            builder_task.launch()

    def tearDown(self):
        if pathlib.Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)


if __name__ == "__main__":
    unittest.main()
