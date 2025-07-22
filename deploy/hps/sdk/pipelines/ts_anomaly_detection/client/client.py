#!/usr/bin/env python

import argparse
import sys

from paddlex_hps_client import triton_request, utils
from tritonclient import grpc as triton_grpc

OUTPUT_IMAGE_PATH = "out.jpg"
OUTPUT_CSV_PATH = "out.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--url", type=str, default="localhost:8001")

    args = parser.parse_args()

    client = triton_grpc.InferenceServerClient(args.url)
    input_ = {"csv": utils.prepare_input_file(args.csv)}
    if args.no_visualization:
        input_["visualize"] = False
    output = triton_request(client, "time-series-anomaly-detection", input_)
    if output["errorCode"] != 0:
        print(f"Error code: {output['errorCode']}", file=sys.stderr)
        print(f"Error message: {output['errorMsg']}", file=sys.stderr)
        sys.exit(1)
    result = output["result"]
    utils.save_output_file(result["image"], OUTPUT_IMAGE_PATH)
    print(f"Output image saved at {OUTPUT_IMAGE_PATH}")
    utils.save_output_file(result["csv"], OUTPUT_CSV_PATH)
    print(f"Output time-series data saved at {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
