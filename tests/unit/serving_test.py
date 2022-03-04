import logging
import socket
from http import HTTPStatus
from multiprocessing import Process, log_to_stderr
from typing import Optional

import mlflow.pyfunc
import requests
from mlflow.pyfunc.scoring_server import _serve
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from conftest import RegisteredModelInfo, MlflowInfo
from dbx_scalable_dl.controller import ModelController
from dbx_scalable_dl.models import NUM_DEFAULT_INFERENCE_RECOMMENDATIONS


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
    def __init__(self, model_name: str, mlflow_info: MlflowInfo):
        self._port = self._get_next_free_port()
        self._mlflow_info = mlflow_info
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
        mlflow.set_registry_uri(self._mlflow_info.registry_uri)
        mlflow.set_tracking_uri(self._mlflow_info.tracking_uri)
        _serve(self._model_uri, host=self._host, port=self._port)

    def __enter__(self) -> str:
        self._proc = Process(target=self.serve)
        self._proc.daemon = True
        self._proc.start()
        return f"http://{self._host}:{self._port}"

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._proc.kill()


def test_predict_call(registered_model_info: RegisteredModelInfo):
    model_uri = ModelController(registered_model_info.model_name).get_latest_model_uri(
        stages=("None",)
    )

    assert model_uri is not None

    loaded_model = mlflow.pyfunc.load_model(model_uri)

    default_prediction = loaded_model.predict(
        {"user_id": registered_model_info.sample_uid}
    )
    assert len(default_prediction) == NUM_DEFAULT_INFERENCE_RECOMMENDATIONS

    custom_length = 5
    custom_prediction = loaded_model.predict(
        {
            "user_id": registered_model_info.sample_uid,
            "num_recommendations": custom_length,
        }
    )
    assert (len(custom_prediction)) == custom_length


def test_serving(mlflow_info: MlflowInfo, registered_model_info: RegisteredModelInfo):
    with ServerContext(registered_model_info.model_name, mlflow_info) as endpoint_url:
        default_response = session().post(
            f"{endpoint_url}/invocations",
            json={"inputs": {"user_id": [registered_model_info.sample_uid]}},
        )
        assert default_response.status_code == HTTPStatus.OK
        assert len(default_response.json()) == NUM_DEFAULT_INFERENCE_RECOMMENDATIONS

        custom_length = 5
        custom_response = session().post(
            f"{endpoint_url}/invocations",
            json={
                "inputs": {
                    "user_id": [registered_model_info.sample_uid],
                    "num_recommendations": [custom_length],
                }
            },
        )

        assert custom_response.status_code == HTTPStatus.OK
        assert len(custom_response.json()) == custom_length
