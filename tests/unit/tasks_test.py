from unittest.mock import MagicMock

import mlflow
import pytest
from pyspark.sql import SparkSession

from conftest import MlflowInfo, LocalRunner
from dbx_scalable_dl.data_provider import DataProvider
from dbx_scalable_dl.controller import ModelController
from dbx_scalable_dl.tasks.data_loader import DataLoaderTask
from dbx_scalable_dl.tasks.model_builder import ModelBuilderTask


test_data_source_url = (
    "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Gift_Cards_5.json.gz"
)


def test_data_loader(spark: SparkSession):
    loader_conf = {
        "data_url": test_data_source_url,
        "database": "dbx_scalable_dl_demo",
        "table": "ratings",
        "temp_directory_prefix": None,
    }

    loader_task = DataLoaderTask(
        spark=spark,
        init_conf=loader_conf,
    )

    loader_task.launch()

    assert loader_conf["table"] in [
        t.name for t in spark.catalog.listTables(loader_conf["database"])
    ]

    assert spark.table(f"{loader_conf['database']}.{loader_conf['table']}").count() > 0


def test_model_builder(
    spark: SparkSession, mlflow_info: MlflowInfo, data_provider: DataProvider
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

    assert mlflow.get_experiment_by_name(experiment) is not None

    _controller = ModelController(model_name)

    assert _controller.get_latest_model_uri(stages=("Production",)) is None

    assert _controller.get_latest_model_uri(stages=("None",)) is not None

    with pytest.raises(Exception):
        _controller.deploy_model_to_sagemaker(
            image_url="fake",
            region_name="fake",
        )
