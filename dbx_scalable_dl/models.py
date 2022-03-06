import logging
from typing import Dict, Text, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from keras.engine.sequential import Sequential
from keras.layers import StringLookup
from mlflow.pyfunc import PythonModel, PythonModelContext
from tensorflow_recommenders.metrics import FactorizedTopK
from tensorflow_recommenders.models import Model
from tensorflow_recommenders.tasks import Ranking, Retrieval

NUM_DEFAULT_INFERENCE_RECOMMENDATIONS = 10


class BasicModel(Model):
    def __init__(
        self,
        rating_weight: float,
        retrieval_weight: float,
        product_ids: np.array,
        user_ids: np.array,
    ) -> None:
        # We take the loss weights in the constructor: this allows us to instantiate
        # several model objects with different loss weights.

        super().__init__()

        embedding_dimension = 32

        self._product_ids_numpy = product_ids
        self._user_ids_numpy = user_ids

        self._product_ids_dataset = tf.data.Dataset.from_tensor_slices(
            self._product_ids_numpy
        )

        # User and movie models.
        self.product_model = Sequential(
            [
                StringLookup(vocabulary=self._product_ids_numpy, mask_token=None),
                tf.keras.layers.Embedding(
                    len(self._product_ids_numpy) + 1, embedding_dimension
                ),
            ]
        )

        self.user_lookup = StringLookup(
            vocabulary=self._user_ids_numpy, mask_token=None
        )

        self.user_model = Sequential(
            [
                self.user_lookup,
                tf.keras.layers.Embedding(
                    len(self._user_ids_numpy) + 1, embedding_dimension
                ),
            ]
        )

        # A small model to take in user and product embeddings and predict ratings.
        # We can make this as complicated as we want as long as we output a scalar
        # as our prediction.
        self.ranking_model = Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

        # The tasks.
        self.ranking_task: Ranking = Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task: Retrieval = Retrieval(
            metrics=FactorizedTopK(
                candidates=self._product_ids_dataset.batch(128).map(self.product_model)
            )
        )

        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the product features and pass them into the product model.
        product_embeddings = self.product_model(features["product_id"])

        return (
            user_embeddings,
            product_embeddings,
            # We apply the multi-layered rating model to a concatenation of
            # user and product embeddings.
            self.ranking_model(
                tf.concat([user_embeddings, product_embeddings], axis=1)
            ),
        )

    def _build_index(self) -> tfrs.layers.factorized_top_k.BruteForce:
        bf_index = tfrs.layers.factorized_top_k.BruteForce(
            self.user_model
        ).index_from_dataset(
            self._product_ids_dataset.batch(100).map(
                lambda product_id: (product_id, self.product_model(product_id))
            )
        )
        _ = bf_index(
            self._user_ids_numpy[[0]]
        )  # we need to trigger index at least once to make it serializable
        return bf_index

    @property
    def index(self) -> tfrs.layers.factorized_top_k.BruteForce:
        return self._build_index()

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        ratings = features.pop("rating")

        user_embeddings, product_embeddings, rating_predictions = self.call(features)

        # We compute the loss for each task.
        rating_loss = self.ranking_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, product_embeddings)

        # And combine them using the loss weights.
        return self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss


class InferenceModel(Model):
    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        """
        This model doesn't compute any losses
        """

    def __init__(self, trained_model: BasicModel):
        super().__init__()

        self.index = trained_model.index
        self.product_model = trained_model.product_model
        self.user_model = trained_model.user_model
        self.ranking_model = trained_model.ranking_model

    @staticmethod
    def preprocess_arguments(
        inputs: Dict[str, np.ndarray]
    ) -> Dict[Text, Union[tf.Tensor, int]]:
        return {
            "user_id": tf.reshape(tf.constant(inputs["user_id"]), (1, 1)),
            "num_recommendations": int(
                inputs.get("num_recommendations", NUM_DEFAULT_INFERENCE_RECOMMENDATIONS)
            ),
        }

    def call(self, inputs: Dict[str, tf.Tensor], training=None, mask=None):
        # retrieval - limit required amount for ranking to subset of the N retrieved recommendations
        user_id = inputs["user_id"]

        num_recommendations = inputs["num_recommendations"]
        _, product_ids = self.index(user_id, num_recommendations)

        # ranking - here we simply rank N best recommendations

        # we repeat user_embedding across user axis to make it compatible with product_embeddings
        user_embeddings = tf.repeat(
            self.user_model(user_id), num_recommendations, axis=0
        )

        # in this context squeeze will transform (1,N,E) to (N,E)
        # where N - number of recommendations
        # E - embedding length from the model
        product_embeddings = tf.squeeze(self.product_model(product_ids))

        ratings = self.ranking_model(
            tf.concat([user_embeddings, product_embeddings], axis=1)
        )

        return product_ids, ratings


class ServingModel(PythonModel):
    def __init__(self):
        super(ServingModel, self).__init__()
        self._model: Optional[InferenceModel] = None

    def load_context(self, context: PythonModelContext):
        self._model: InferenceModel = tf.saved_model.load(
            context.artifacts["model_instance"]
        )

    def predict(self, context, model_input: Dict[str, np.ndarray]):
        logging.info(f"Received input vector: {model_input}")
        product_ids, ratings = self._model(
            InferenceModel.preprocess_arguments(model_input)
        )
        _prediction = dict(
            zip(
                [
                    p.decode() for p in product_ids.numpy().squeeze()
                ],  # to decode product id from bytes to string
                ratings.numpy().squeeze(),
            )
        )
        logging.info(f"Predicted {_prediction}")
        return _prediction
