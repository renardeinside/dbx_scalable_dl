import logging
import tempfile
from typing import Optional

import mlflow
import tensorflow as tf
import tensorflow_recommenders as tfrs
from mlflow.pyfunc.model import PythonModel, PythonModelContext

from dbx_scalable_dl.model import InferenceModel, BasicModel


class ServingModel(PythonModel):
    def __init__(self):
        super(ServingModel, self).__init__()
        self._model: Optional[InferenceModel] = None

    def load_context(self, context: PythonModelContext):
        self._model: InferenceModel = tf.saved_model.load(
            context.artifacts["model_instance"]
        )

    def predict(self, context, model_input):
        logging.info(f"Received input vector: {model_input}")
        _prediction = self._model.call(model_input)
        logging.info(f"Predicted {_prediction}")
        return _prediction


class ModelRegistrator:
    def _get_conda_env(self) -> str:
        conda_env_definition = mlflow.pyfunc.get_default_conda_env()
        pip_deps = conda_env_definition.get("dependencies")[-1].get("pip")
        new_pip_deps = pip_deps + [
            f"tensorflow-recommenders=={tfrs.__version__}",
            f"tensorflow=={tf.__version__}",
        ]
        conda_env_definition["dependencies"][-1] = {"pip": new_pip_deps}
        return conda_env_definition

    def register_model(self, model: BasicModel, model_name: str):
        inference_model = InferenceModel(model)

        with tempfile.TemporaryDirectory() as temp_dir:
            tf.saved_model.save(inference_model, temp_dir)
            artifacts = {"model_instance": temp_dir}
            mlflow.pyfunc.log_model(
                "model",
                python_model=ServingModel(),
                registered_model_name=model_name,
                artifacts=artifacts,
                conda_env=self._get_conda_env(),
            )
