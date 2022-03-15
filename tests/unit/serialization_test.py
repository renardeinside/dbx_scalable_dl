import tempfile
from unittest.mock import Mock

import numpy as np
from pyspark.cloudpickle.cloudpickle_fast import dump

from conftest import MlflowInfo
from dbx_scalable_dl.data_provider import DataProvider
from dbx_scalable_dl.serialization import (
    MlflowInfo as SerializableMlflowInfo,
    RunnerFunctionInfo,
    SerializableFunctionProvider,
)
from dbx_scalable_dl.tasks.model_builder import ModelBuilderTask
from dbx_scalable_dl.utils import LaunchEnvironment


def test_serializable(data_provider: DataProvider, mlflow_info: MlflowInfo):
    task = ModelBuilderTask()
    task._get_launch_environment = Mock(return_value=LaunchEnvironment.SINGLE_NODE)
    tc, vc = data_provider.get_train_test_converters()
    mock_info = RunnerFunctionInfo(
        batch_size=100,
        model_name="whatever",
        num_epochs=1,
        product_ids=np.ones(shape=(100,)),
        user_ids=np.ones(shape=(100,)),
        train_converter=tc,
        validation_converter=vc,
        mlflow_info=SerializableMlflowInfo(
            tracking_uri=mlflow_info.tracking_uri,
            registry_uri=mlflow_info.registry_uri,
            experiment="some-exp",
        ),
    )

    with tempfile.TemporaryFile() as t_file:
        dump(SerializableFunctionProvider, t_file)
        dump(mock_info, t_file)
