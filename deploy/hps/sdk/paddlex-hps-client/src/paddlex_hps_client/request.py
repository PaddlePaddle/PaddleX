import json

import numpy as np
from tritonclient import grpc as triton_grpc

from . import constants


def _create_triton_input(data):
    data = json.dumps(data, separators=(",", ":"))
    data = data.encode("utf-8")
    data = [[data]]
    data = np.array(data, dtype=np.object_)
    return data


def _parse_triton_output(data):
    data = data[0, 0]
    data = data.decode("utf-8")
    data = json.loads(data)
    return data


def triton_request(client, model_name, data, *, request_kwargs=None):
    if request_kwargs is None:
        request_kwargs = {}
    input_ = triton_grpc.InferInput(constants.INPUT_NAME, [1, 1], "BYTES")
    input_.set_data_from_numpy(_create_triton_input(data))
    results = client.infer(model_name, inputs=[input_], **request_kwargs)
    output = results.as_numpy(constants.OUTPUT_NAME)
    return _parse_triton_output(output)
