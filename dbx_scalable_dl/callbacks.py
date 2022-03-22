import mlflow
import tensorflow as tf

from dbx_scalable_dl.controller import ModelController


class MLflowLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name: str):
        self._model_name = model_name
        with mlflow.start_run() as run:
            self._parent_run_id = run.info.run_id

    def on_epoch_end(self, epoch, logs=None):
        with mlflow.start_run(run_id=self._parent_run_id):
            with mlflow.start_run(nested=True):
                if logs:
                    mlflow.log_param("epoch", epoch)
                    mlflow.log_metrics(logs)

    def on_train_end(self, *_):
        with mlflow.start_run(run_id=self._parent_run_id):
            ModelController(self._model_name).register_model(self.model)
