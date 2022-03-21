import mlflow
import tensorflow as tf

from dbx_scalable_dl.controller import ModelController


class MLflowLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._active_run = mlflow.start_run()

    def on_epoch_end(self, epoch, logs=None):
        with self._active_run:
            with mlflow.start_run(nested=True):
                mlflow.set_tag("epoch", epoch)
                if logs:
                    mlflow.log_param("epoch", epoch)
                    mlflow.log_metrics(logs)

    def on_train_end(self, *_):
        with self._active_run:
            ModelController(self._model_name).register_model(self.model)
            mlflow.end_run()
