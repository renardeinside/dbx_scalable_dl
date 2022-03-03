import tempfile

import mlflow
import tensorflow as tf
import tensorflow_recommenders as tfrs
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from typing import List, Optional, Tuple
import mlflow.sagemaker as mfs
from dbx_scalable_dl.model import BasicModel, InferenceModel
from dbx_scalable_dl.serving import ServingModel


class ModelController:
    def __init__(self, model_name: str):
        self._model_name = model_name

    @staticmethod
    def _get_conda_env() -> str:
        conda_env_definition = mlflow.pyfunc.get_default_conda_env()
        pip_deps = conda_env_definition.get("dependencies")[-1].get("pip")
        new_pip_deps = pip_deps + [
            f"tensorflow-recommenders=={tfrs.__version__}",
            f"tensorflow=={tf.__version__}",
        ]
        conda_env_definition["dependencies"][-1] = {"pip": new_pip_deps}
        return conda_env_definition

    def register_model(self, model: BasicModel):
        inference_model = InferenceModel(model)

        with tempfile.TemporaryDirectory() as temp_dir:
            tf.saved_model.save(inference_model, temp_dir)
            artifacts = {"model_instance": temp_dir}
            mlflow.pyfunc.log_model(
                "model",
                python_model=ServingModel(),
                registered_model_name=self._model_name,
                input_example={"user_id": "some_user_id", "num_recommendations": 10},
                artifacts=artifacts,
                conda_env=self._get_conda_env(),
            )

    def _get_latest_version(
        self, stages: Optional[List[str]] = None
    ) -> Optional[ModelVersion]:
        client = MlflowClient()
        _resp: List[ModelVersion] = client.get_latest_versions(self._model_name, stages)
        _latest = (
            None
            if not _resp
            else sorted(_resp, key=lambda mv: int(mv.version))[-1].version
        )
        return _latest

    def get_latest_model_uri(
        self, stages: Optional[Tuple[str]] = ("Production",)
    ) -> Optional[str]:
        latest_version = self._get_latest_version(list(stages))
        latest_model_uri = (
            None
            if not latest_version
            else f"models:/{self._model_name}/{latest_version}"
        )
        return latest_model_uri

    def deploy_model_to_sagemaker(
        self,
        image_url: str,
        region_name: str,
        stages: Optional[Tuple[str]] = ("Production",),
    ):
        model_uri = self.get_latest_model_uri(stages)
        if not model_uri:
            raise Exception(
                f"No model version found for model {self._model_name} with stages {stages}"
            )
        else:
            mfs.deploy(
                app_name=self._model_name,
                model_uri=model_uri,
                image_url=image_url,
                region_name=region_name,
                mode="create",
            )
