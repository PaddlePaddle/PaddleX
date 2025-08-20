#!/usr/bin/env python

import argparse
import sys

from paddlex_hps_client import triton_request, utils
from tritonclient import grpc as triton_grpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--file-type", type=int, choices=[0, 1])
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--url", type=str, default="localhost:8001")

    args = parser.parse_args()

    client = triton_grpc.InferenceServerClient(args.url)
    input_ = {"file": utils.prepare_input_file(args.file)}
    if args.file_type is not None:
        input_["fileType"] = args.file_type
    if args.no_visualization:
        input_["visualize"] = False
    output = triton_request(client, "ocr", input_)
    if output["errorCode"] != 0:
        print(f"Error code: {output['errorCode']}", file=sys.stderr)
        print(f"Error message: {output['errorMsg']}", file=sys.stderr)
        sys.exit(1)
    result = output["result"]
    for i, res in enumerate(result["ocrResults"]):
        print(res["prunedResult"])
        ocr_img_path = f"ocr_{i}.jpg"
        utils.save_output_file(res["ocrImage"], ocr_img_path)
        print(f"Output image saved at {ocr_img_path}")


if __name__ == "__main__":
    main()
