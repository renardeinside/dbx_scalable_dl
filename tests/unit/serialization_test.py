import tempfile

from pyspark.cloudpickle.cloudpickle_fast import dump

from dbx_scalable_dl.data_provider import DataProvider
from dbx_scalable_dl.serialization import (
    MlflowInfo as SerializableMlflowInfo,
    RunnerFunctionInfo,
    SerializableFunctionProvider,
)
from dbx_scalable_dl.tasks.model_builder import ModelBuilderTask
from conftest import MlflowInfo


def test_serializable(data_provider: DataProvider, mlflow_info: MlflowInfo):
    task = ModelBuilderTask()
    tc, vc = data_provider.get_train_test_converters()
    mock_info = RunnerFunctionInfo(
        batch_size=100,
        model_name="whatever",
        num_epochs=1,
        product_ids=data_provider.dataset_to_numpy_array(
            data_provider.get_unique_product_ids()
        ),
        user_ids=data_provider.dataset_to_numpy_array(
            data_provider.get_unique_user_ids()
        ),
        train_converter=tc,
        validation_converter=vc,
        mlflow_info=SerializableMlflowInfo(
            tracking_uri=mlflow_info.tracking_uri,
            registry_uri=mlflow_info.registry_uri,
            experiment="some-exp",
        ),
    )

    runner = task.get_training_function(mock_info)

    with tempfile.TemporaryFile() as t_file:
        dump(SerializableFunctionProvider, t_file)
        dump(runner, t_file)
