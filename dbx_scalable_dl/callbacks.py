import mlflow
import tensorflow as tf

from controller import ModelController


class MLflowLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("epoch", epoch)
            if logs:
                mlflow.log_param("epoch", epoch)
                mlflow.log_metrics(logs)

    def on_train_end(self, *_):
        with mlflow.start_run(nested=True):
            ModelController(self.model_name).register_model(self.model)
