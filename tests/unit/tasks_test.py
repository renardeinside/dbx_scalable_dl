import tempfile
import unittest
from typing import Callable
from unittest.mock import MagicMock

import mlflow

from common import TestCaseWithEnvironment
from dbx_scalable_dl.controller import ModelController
from dbx_scalable_dl.tasks.data_loader import DataLoaderTask
from dbx_scalable_dl.tasks.model_builder import ModelBuilderTask
from dbx_scalable_dl.utils import FileLoadingContext


class LocalRunner:
    def run(self, main: Callable):
        main()


class SampleJobUnitTest(TestCaseWithEnvironment):
    def setUp(self):
        self.source_url = "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Gift_Cards_5.json.gz"

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
        experiment = "dbx_test_experiment"
        model_name = "dbx_scalable_ml_test"
        with tempfile.TemporaryDirectory() as cache_dir:
            builder_task = ModelBuilderTask(
                spark=self.spark,
                init_conf={
                    "batch_size": 100,
                    "database": "dbx_scalable_dl_demo",
                    "table": "ratings",
                    "cache_dir": "file://" + cache_dir,
                    "num_epochs": 1,
                    "model_name": model_name,
                    "mlflow": {
                        "experiment": experiment,
                        "registry_uri": self.registry_uri,
                        "tracking_uri": self.tracking_uri,
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

                self.assertIsNotNone(
                    mlflow.get_experiment_by_name(experiment), "experiment_exists"
                )

                _controller = ModelController(model_name)

                self.assertRaises(
                    Exception,
                    _controller.get_latest_model_uri,
                    stages=("Production",),
                    msg="model_not_in_production",
                )

                self.assertIsNotNone(
                    _controller.get_latest_model_uri(stages=("None",)),
                    msg="model_in_none",
                )


if __name__ == "__main__":
    unittest.main()
