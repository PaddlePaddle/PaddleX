#!/usr/bin/env python

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

import argparse
import pprint
import sys

from tritonclient import grpc as triton_grpc

from paddlex_hps_client import triton_request, utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", type=str, required=True)
    parser.add_argument("--url", type=str, default="localhost:8001")
    args = parser.parse_args()

    client = triton_grpc.InferenceServerClient(args.url)
    input_ = {"tar": utils.prepare_input_file(args.tar)}
    output = triton_request(client, "bev-3d-object-detection", input_)
    if output["errorCode"] != 0:
        print(f"Error code: {output['errorCode']}", file=sys.stderr)
        print(f"Error message: {output['errorMsg']}", file=sys.stderr)
        sys.exit(1)
    result = output["result"]
    print("Detected objects:")
    pprint.pp(result["detectedObjects"])


if __name__ == "__main__":
    main()
