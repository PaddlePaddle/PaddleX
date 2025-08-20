#!/usr/bin/env python

import argparse
import sys

from paddlex_hps_client import triton_request, utils
from tritonclient import grpc as triton_grpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--max-image-tokens", type=int, default=None)
    parser.add_argument("--url", type=str, default="localhost:8001")
    args = parser.parse_args()

    client = triton_grpc.InferenceServerClient(
        args.url,
        # HACK
        keepalive_options=triton_grpc.KeepAliveOptions(keepalive_timeout_ms=1000000),
    )
    image = utils.prepare_input_file(args.image, include_header=True)
    input_ = {
        "model": "pp-docbee",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.query},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            },
        ],
        "max_image_tokens": args.max_image_tokens,
    }
    output = triton_request(client, "document-understanding", input_)
    if output["errorCode"] != 0:
        print(f"Error code: {output['errorCode']}", file=sys.stderr)
        print(f"Error message: {output['errorMsg']}", file=sys.stderr)
        sys.exit(1)
    result = output["result"]
    print("Final result:")
    print(result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
