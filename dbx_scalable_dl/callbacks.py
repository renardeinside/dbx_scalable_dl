import mlflow
import tempfile
import tensorflow_recommenders as tfrs
import tensorflow as tf
import numpy as np
from dbx_scalable_dl.inference import convert_to_inference_model
from dbx_scalable_dl.model import BasicModel


class MLflowLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name: str, unique_products: np.array):
        self.model_name = model_name
        self.unique_products = unique_products

    def on_epoch_end(self, epoch, logs=None):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("epoch", epoch)
            if logs:
                mlflow.log_param("epoch", epoch)
                mlflow.log_metrics(logs)

    @staticmethod
    def get_conda_env() -> str:
        conda_env_definition = mlflow.pyfunc.get_default_conda_env()
        pip_deps = conda_env_definition.get("dependencies")[-1].get("pip")
        new_pip_deps = pip_deps + [
            f"tensorflow-recommenders=={tfrs.__version__}",
            f"tensorflow=={tf.__version__}",
        ]
        conda_env_definition["dependencies"][-1] = {"pip": new_pip_deps}
        return conda_env_definition

    def on_train_end(self, *_):
        with mlflow.start_run(nested=True):
            with tempfile.TemporaryDirectory() as temp_dir:
                trained_model: BasicModel = self.model
                inference_model = convert_to_inference_model(trained_model)

