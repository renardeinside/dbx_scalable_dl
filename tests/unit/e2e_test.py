import pathlib
import shutil
import tempfile
import unittest

from dbx_scalable_dl.tasks.data_loader import DataLoaderTask
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


class SampleJobUnitTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory().name
        _builder = (
            SparkSession.builder.master("local[1]")
            .config("spark.default.parallelism", 2)
            .config("spark.sql.shuffle.partitions", 2)
            .config("spark.hive.metastore.warehouse.dir", self.test_dir)
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
        )
        self.spark = configure_spark_with_delta_pip(_builder).getOrCreate()
        self.spark.sparkContext.setLogLevel("INFO")

    def test_data_loader(self):
        # feel free to add new methods to this magic mock to mock some particular functionality
        _conf = {
            "data_url": "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Gift_Cards_5.json.gz",
            "database": "dbx_scalable_dl_demo",
        }
        self.job = DataLoaderTask(
            spark=self.spark,
            init_conf=_conf,
        )
        self.job.launch()

        self.assertIn(
            "reviews",
            [t.name for t in self.spark.catalog.listTables(_conf["database"])],
        )
        self.assertGreater(self.spark.table(f"{_conf['database']}.reviews").count(), 0)
        self.spark.table(f"{_conf['database']}.reviews").show()

    def tearDown(self):
        if pathlib.Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)


if __name__ == "__main__":
    unittest.main()
