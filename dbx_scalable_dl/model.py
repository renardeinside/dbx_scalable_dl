from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from keras.engine.sequential import Sequential
from keras.layers import StringLookup
from tensorflow_recommenders.layers.factorized_top_k import TopK
from tensorflow_recommenders.metrics import FactorizedTopK
from tensorflow_recommenders.models import Model
from tensorflow_recommenders.tasks import Ranking, Retrieval

from dbx_scalable_dl.data_provider import DataProvider


class BasicModel(Model):
    def __init__(
            self,
            rating_weight: float,
            retrieval_weight: float,
            data_provider: DataProvider,
    ) -> None:
        # We take the loss weights in the constructor: this allows us to instantiate
        # several model objects with different loss weights.

        super().__init__()

        embedding_dimension = 32

        self.product_ids_as_dataset = data_provider.get_unique_product_ids()
        self.product_ids_as_numpy = data_provider.dataset_to_numpy_array(
            self.product_ids_as_dataset
        )

        self.users_ids_as_numpy = data_provider.dataset_to_numpy_array(
            data_provider.get_unique_user_ids()
        )

        # User and movie models.
        self.product_model = Sequential(
            [
                StringLookup(vocabulary=self.product_ids_as_numpy, mask_token=None),
                tf.keras.layers.Embedding(
                    len(self.product_ids_as_dataset) + 1, embedding_dimension
                ),
            ]
        )

        self.user_model = Sequential(
            [
                StringLookup(vocabulary=self.users_ids_as_numpy, mask_token=None),
                tf.keras.layers.Embedding(
                    len(self.users_ids_as_numpy) + 1, embedding_dimension
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
                candidates=self.product_ids_as_dataset.batch(128).map(
                    self.product_model
                )
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

    def build_index(self) -> TopK:
        bf_index = tfrs.layers.factorized_top_k.BruteForce(self.user_model)
        bf_index.index_from_dataset(
            self.product_ids_as_dataset.batch(100).map(
                lambda product_id: (product_id, self.product_model(product_id))
            )
        )
        _ = bf_index(
            self.users_ids_as_numpy[[0]]
        )  # we need to trigger index at least once to make it serializable
        return bf_index

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

        self.index = trained_model.build_index()
        self.product_model = trained_model.product_model
        self.user_model = trained_model.user_model
        self.ranking_model = trained_model.ranking_model

    def call(self, inputs: Dict[str, tf.Tensor], training=None, mask=None):
        # retrieval - limit required amount for ranking to subset of the N retrieved recommendations
        user_id = tf.constant(inputs["user_id"])
        num_recommendations = inputs.get("num_recommendations", 10)
        _, product_ids = self.index(np.array([user_id]), num_recommendations)

        # ranking - here we simply rank N best recommendations

        # we repeat user_embedding across user axis to make it compatible with product_embeddings
        user_embeddings = tf.repeat(self.user_model(user_id), num_recommendations, axis=0)

        # in this context squeeze will transform (1,N,E) to (N,E)
        # where N - number of recommendations
        # E - embedding length from the model
        product_embeddings = tf.squeeze(self.product_model(product_ids))

        ratings = self.ranking_model(tf.concat([user_embeddings, product_embeddings], axis=1))

        # we squeeze both vectors and map them into a dictionary
        _ratings = dict(zip(product_ids.numpy().squeeze(), ratings.numpy().squeeze()))

        return _ratings
