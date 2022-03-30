from http import HTTPStatus

import mlflow.pyfunc
from flask.testing import FlaskClient

from conftest import RegisteredModelInfo
from nocturne.controller import ModelController
from nocturne.models import NUM_DEFAULT_INFERENCE_RECOMMENDATIONS


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


def test_serving(model_client: FlaskClient, registered_model_info: RegisteredModelInfo):
    ping = model_client.get("/ping")
    assert ping.status_code == HTTPStatus.OK

    default_response = model_client.post(
        "/invocations",
        json={
            "inputs": {
                "user_id": [registered_model_info.sample_uid],
            }
        },
    )

    assert default_response.status_code == HTTPStatus.OK
    assert len(default_response.json) == NUM_DEFAULT_INFERENCE_RECOMMENDATIONS

    custom_length = 5
    custom_response = model_client.post(
        "/invocations",
        json={
            "inputs": {
                "user_id": [registered_model_info.sample_uid],
                "num_recommendations": [custom_length],
            }
        },
    )

    assert custom_response.status_code == HTTPStatus.OK
    assert len(custom_response.json) == custom_length
