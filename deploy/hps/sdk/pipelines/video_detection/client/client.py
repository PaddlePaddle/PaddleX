#!/usr/bin/env python

import argparse
import pprint
import sys

from paddlex_hps_client import triton_request, utils
from tritonclient import grpc as triton_grpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--url", type=str, default="localhost:8001")
    args = parser.parse_args()

    client = triton_grpc.InferenceServerClient(args.url)
    input_ = {"video": utils.prepare_input_file(args.video)}
    output = triton_request(client, "video-detection", input_)
    if output["errorCode"] != 0:
        print(f"Error code: {output['errorCode']}", file=sys.stderr)
        print(f"Error message: {output['errorMsg']}", file=sys.stderr)
        sys.exit(1)
    result = output["result"]
    pprint.pp(result["frames"])


if __name__ == "__main__":
    main()
