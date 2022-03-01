import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from dbx_scalable_dl.tasks.data_loader import DataLoaderTask
from pyspark.sql import SparkSession


class SampleJobUnitTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory().name
        self.spark = SparkSession.builder.master("local[1]").getOrCreate()
        self.spark.sparkContext.setLogLevel("INFO")

    def test_loader(self):
        # feel free to add new methods to this magic mock to mock some particular functionality
        self.job = DataLoaderTask(spark=self.spark, init_conf={})
        self.job.dbutils = MagicMock()

    def tearDown(self):
        if pathlib.Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)


if __name__ == "__main__":
    unittest.main()
