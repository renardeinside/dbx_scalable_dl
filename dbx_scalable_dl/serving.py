import logging
from typing import Optional

import tensorflow as tf
from mlflow.pyfunc.model import PythonModel, PythonModelContext

from dbx_scalable_dl.model import InferenceModel


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
