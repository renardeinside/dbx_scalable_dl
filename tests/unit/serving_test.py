import logging
import socket
import tempfile
from multiprocessing import Process, log_to_stderr
from typing import Optional, List

import mlflow.pyfunc
import numpy as np
import pandas as pd
import requests
from mlflow.pyfunc.scoring_server import _serve
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from common import TestCaseWithEnvironment
from dbx_scalable_dl.callbacks import MLflowLoggingCallback
from dbx_scalable_dl.controller import ModelController
from dbx_scalable_dl.data_provider import DataProvider
from dbx_scalable_dl.models import DEFAULT_INFERENCE_RECOMMENDATIONS
from dbx_scalable_dl.tasks.model_builder import ModelBuilderTask


def session(
    retries=10,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    sess=None,
):
    sess = sess or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


class ServerContext:
    def __init__(self, model_name: str, registry_uri: str, tracking_uri: str):
        self._port = self._get_next_free_port()
        self._registry_uri = registry_uri
        self._tracking_uri = tracking_uri
        self._host = "0.0.0.0"
        self._model_uri = ModelController(model_name).get_latest_model_uri(
            stages=("None",)
        )
        log_to_stderr(logging.INFO)
        self._proc: Optional[Process] = None

    @staticmethod
    def _get_next_free_port(port=4000, max_port=5000):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while port <= max_port:
            try:
                sock.bind(("", port))
                sock.close()
                return port
            except OSError:
                port += 1
        raise IOError("no free ports")

    def serve(self):
        mlflow.set_registry_uri(self._registry_uri)
        mlflow.set_tracking_uri(self._tracking_uri)
        _serve(self._model_uri, host=self._host, port=self._port)

    def __enter__(self) -> str:
        self._proc = Process(target=self.serve)
        self._proc.daemon = True
        self._proc.start()
        return f"http://{self._host}:{self._port}"

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._proc.kill()


class ServingTest(TestCaseWithEnvironment):
    model_name = "serving_model_test"
    sample_uid: str
    user_ids = List[str]

    @classmethod
    def _prepare_model_in_mlflow(cls):
        with tempfile.TemporaryDirectory() as cache_dir:
            cls.user_ids = [f"UID_{i:03}" for i in range(10)]

            user_ids = pd.Series(cls.user_ids)
            product_ids = pd.Series([f"PID_{i:03}" for i in range(10)])
            ratings_size = 1000

            cls.sample_uid = user_ids[1]

            _pdf = pd.DataFrame().from_dict(
                {
                    "user_id": user_ids.sample(n=ratings_size, replace=True).tolist(),
                    "product_id": product_ids.sample(
                        n=ratings_size, replace=True
                    ).tolist(),
                    "rating": np.random.randint(1, 5, size=ratings_size).astype(
                        np.float
                    ),
                }
            )

            _sdf = cls.spark.createDataFrame(_pdf)

            provider = DataProvider(
                cls.spark, ratings=_sdf, cache_dir=f"file://{cache_dir}"
            )

            _model = ModelBuilderTask.get_model(provider)

            train_converter, _ = provider.get_train_test_converters()

            with train_converter.make_tf_dataset(batch_size=100) as train_reader:
                train_ds = train_reader.map(ModelBuilderTask._convert_to_row)
                _model.fit(
                    train_ds,
                    epochs=1,
                    steps_per_epoch=2,
                    callbacks=[MLflowLoggingCallback(cls.model_name)],
                )

    @classmethod
    def setUpClass(cls) -> None:
        super(ServingTest, cls).setUpClass()
        cls._prepare_model_in_mlflow()

    def test_predict_call(self):
        model_uri = ModelController(self.model_name).get_latest_model_uri(
            stages=("None",)
        )

        self.assertIsNotNone(model_uri, msg="model registered in MLflow")

        loaded_model = mlflow.pyfunc.load_model(model_uri)

        default_prediction = loaded_model.predict({"user_id": self.sample_uid})
        self.assertEqual(
            len(default_prediction),
            10,
            msg="default call returns 10 recommendations",
        )

        custom_prediction = loaded_model.predict(
            {"user_id": self.sample_uid, "num_recommendations": 5}
        )
        self.assertEqual(
            len(custom_prediction),
            5,
            msg="parametrized call returns required amount of reqs",
        )

    def test_serving(self):
        with ServerContext(
            self.model_name, mlflow.get_registry_uri(), mlflow.get_tracking_uri()
        ) as endpoint_url:
            default_response = session().post(
                f"{endpoint_url}/invocations",
                json={"inputs": {"user_id": [self.sample_uid]}},
            )
            self.assertEqual(default_response.status_code, 200)
            self.assertEqual(
                len(default_response.json()), DEFAULT_INFERENCE_RECOMMENDATIONS
            )

            custom_response = session().post(
                f"{endpoint_url}/invocations",
                json={
                    "inputs": {"user_id": [self.sample_uid], "num_recommendations": [5]}
                },
            )

            self.assertEqual(custom_response.status_code, 200)
            self.assertEqual(len(custom_response.json()), 5)
