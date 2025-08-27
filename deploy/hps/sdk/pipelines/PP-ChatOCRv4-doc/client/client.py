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
import sys

from paddlex_hps_client import triton_request, utils
from tritonclient import grpc as triton_grpc


def ensure_no_error(output, additional_msg):
    if output["errorCode"] != 0:
        print(additional_msg, file=sys.stderr)
        print(f"Error code: {output['errorCode']}", file=sys.stderr)
        print(f"Error message: {output['errorMsg']}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--key-list", type=str, nargs="+", required=True)
    parser.add_argument("--file-type", type=int, choices=[0, 1])
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--invoke-mllm", action="store_true")
    parser.add_argument("--url", type=str, default="localhost:8001")

    args = parser.parse_args()

    client = triton_grpc.InferenceServerClient(args.url)

    input_ = {"file": utils.prepare_input_file(args.file)}
    if args.file_type is not None:
        input_["fileType"] = args.file_type
    if args.no_visualization:
        input_["visualize"] = False
    output = triton_request(client, "chatocr-visual", input_)
    ensure_no_error(output, "Failed to analyze the images")
    result_visual = output["result"]

    for i, res in enumerate(result_visual["layoutParsingResults"]):
        print(res["prunedResult"])
        for img_name, img in res["outputImages"].items():
            img_path = f"{img_name}_{i}.jpg"
            utils.save_output_file(img, img_path)
            print(f"Output image saved at {img_path}")

    input_ = {
        "visualInfo": result_visual["visualInfo"],
    }
    output = triton_request(client, "chatocr-vector", input_)
    ensure_no_error(output, "Failed to build a vector store")
    result_vector = output["result"]

    if args.invoke_mllm:
        input_ = {
            "image": utils.prepare_input_file(args.file),
            "keyList": args.key_list,
        }
        output = triton_request(client, "chatocr-mllm", input_)
        ensure_no_error(output, "Failed to invoke the MLLM")
        result_mllm = output["result"]

    input_ = {
        "keyList": args.key_list,
        "visualInfo": result_visual["visualInfo"],
        "useVectorRetrieval": True,
        "vectorInfo": result_vector["vectorInfo"],
    }
    if args.invoke_mllm:
        input_["mllmPredictInfo"] = result_mllm["mllmPredictInfo"]
    output = triton_request(client, "chatocr-chat", input_)
    ensure_no_error(output, "Failed to chat with the LLM")
    result_chat = output["result"]
    print("Final result:")
    print(result_chat["chatResult"])


if __name__ == "__main__":
    main()
