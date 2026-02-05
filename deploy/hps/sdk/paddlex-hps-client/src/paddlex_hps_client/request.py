# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import numpy as np
from tritonclient import grpc as triton_grpc
from tritonclient.grpc import aio as triton_grpc_aio

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


def triton_request(client, model_name, data, *, timeout=None, request_kwargs=None):
    """
    Make a synchronous request to Triton Inference Server.

    Args:
        client: Triton gRPC client (tritonclient.grpc.InferenceServerClient)
        model_name: Name of the model to call
        data: Request payload dict
        timeout: Request timeout in seconds
        request_kwargs: Additional kwargs passed to client.infer()

    Returns:
        Response dict from Triton
    """
    if request_kwargs is None:
        request_kwargs = {}
    if timeout is not None:
        request_kwargs.setdefault("timeout", timeout)
        request_kwargs.setdefault("client_timeout", timeout)
    input_ = triton_grpc.InferInput(constants.INPUT_NAME, [1, 1], "BYTES")
    input_.set_data_from_numpy(_create_triton_input(data))
    results = client.infer(model_name, inputs=[input_], **request_kwargs)
    output = results.as_numpy(constants.OUTPUT_NAME)
    return _parse_triton_output(output)


async def triton_request_async(client, model_name, data, *, timeout=None, request_kwargs=None):
    """
    Make an async request to Triton Inference Server.

    Args:
        client: Async Triton gRPC client (tritonclient.grpc.aio.InferenceServerClient)
        model_name: Name of the model to call
        data: Request payload dict
        timeout: Request timeout in seconds
        request_kwargs: Additional kwargs passed to client.infer()

    Returns:
        Response dict from Triton
    """
    if request_kwargs is None:
        request_kwargs = {}
    if timeout is not None:
        request_kwargs.setdefault("timeout", timeout)
        request_kwargs.setdefault("client_timeout", timeout)
    input_ = triton_grpc_aio.InferInput(constants.INPUT_NAME, [1, 1], "BYTES")
    input_.set_data_from_numpy(_create_triton_input(data))
    results = await client.infer(model_name, inputs=[input_], **request_kwargs)
    output = results.as_numpy(constants.OUTPUT_NAME)
    return _parse_triton_output(output)
