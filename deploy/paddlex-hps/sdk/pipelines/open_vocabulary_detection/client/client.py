#!/usr/bin/env python

import argparse
import pprint
import sys

from paddlex_hps_client import triton_request, utils
from tritonclient import grpc as triton_grpc

OUTPUT_IMAGE_PATH = "out.jpg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--url", type=str, default="localhost:8001")

    args = parser.parse_args()

    client = triton_grpc.InferenceServerClient(args.url)
    input_ = {"image": utils.prepare_input_file(args.image), "prompt": args.prompt}
    if args.no_visualization:
        input_["visualize"] = False
    output = triton_request(client, "open-vocabulary-detection", input_)
    if output["errorCode"] != 0:
        print(f"Error code: {output['errorCode']}", file=sys.stderr)
        print(f"Error message: {output['errorMsg']}", file=sys.stderr)
        sys.exit(1)
    result = output["result"]
    utils.save_output_file(result["image"], OUTPUT_IMAGE_PATH)
    print(f"Output image saved at {OUTPUT_IMAGE_PATH}")
    print("\nDetected objects:")
    pprint.pp(result["detectedObjects"])


if __name__ == "__main__":
    main()
