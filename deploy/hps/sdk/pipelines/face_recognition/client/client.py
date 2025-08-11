#!/usr/bin/env python

import argparse
import pprint
import sys

from paddlex_hps_client import triton_request, utils
from tritonclient import grpc as triton_grpc

OUTPUT_IMAGE_PATH = "out.jpg"


def parse_image_label_pairs(image_label_pairs):
    if len(image_label_pairs) % 2 != 0:
        raise ValueError("The number of image-label pairs must be even.")
    return [
        {"image": utils.prepare_input_file(img), "label": lab}
        for img, lab in zip(image_label_pairs[0::2], image_label_pairs[1::2])
    ]


def create_triton_client(url):
    return triton_grpc.InferenceServerClient(url)


def ensure_no_error(output):
    if output["errorCode"] != 0:
        print(f"Error code: {output['errorCode']}", file=sys.stderr)
        print(f"Error message: {output['errorMsg']}", file=sys.stderr)
        sys.exit(1)


def do_index_build(args):
    client = create_triton_client(args.url)
    if args.image_label_pairs:
        image_label_pairs = parse_image_label_pairs(args.image_label_pairs)
    else:
        image_label_pairs = []
    input_ = {"imageLabelPairs": image_label_pairs}
    output = triton_request(client, "face-recognition-index-build", input_)
    ensure_no_error(output)
    result = output["result"]
    pprint.pp(result)


def do_index_add(args):
    client = create_triton_client(args.url)
    image_label_pairs = parse_image_label_pairs(args.image_label_pairs)
    input_ = {"imageLabelPairs": image_label_pairs}
    if args.index_key is not None:
        input_["indexKey"] = args.index_key
    output = triton_request(client, "face-recognition-index-add", input_)
    ensure_no_error(output)
    result = output["result"]
    pprint.pp(result)


def do_index_remove(args):
    client = create_triton_client(args.url)
    input_ = {"ids": args.ids}
    if args.index_key is not None:
        input_["indexKey"] = args.index_key
    output = triton_request(client, "face-recognition-index-remove", input_)
    ensure_no_error(output)
    result = output["result"]
    pprint.pp(result)


def do_infer(args):
    client = create_triton_client(args.url)
    input_ = {"image": utils.prepare_input_file(args.image)}
    if args.index_key is not None:
        input_["indexKey"] = args.index_key
    if args.no_visualization:
        input_["visualize"] = False
    output = triton_request(client, "face-recognition-infer", input_)
    ensure_no_error(output)
    result = output["result"]
    utils.save_output_file(result["image"], OUTPUT_IMAGE_PATH)
    print(f"Output image saved at {OUTPUT_IMAGE_PATH}")
    print("\nDetected faces:")
    pprint.pp(result["faces"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="localhost:8001")

    subparsers = parser.add_subparsers(dest="cmd")

    parser_index_build = subparsers.add_parser("index-build")
    parser_index_build.add_argument("--image-label-pairs", type=str, nargs="+")
    parser_index_build.set_defaults(func=do_index_build)

    parser_index_add = subparsers.add_parser("index-add")
    parser_index_add.add_argument(
        "--image-label-pairs", type=str, nargs="+", required=True
    )
    parser_index_add.add_argument("--index-key", type=str, required=True)
    parser_index_add.set_defaults(func=do_index_add)

    parser_index_remove = subparsers.add_parser("index-remove")
    parser_index_remove.add_argument("--ids", type=int, nargs="+", required=True)
    parser_index_remove.add_argument("--index-key", type=str, required=True)
    parser_index_remove.set_defaults(func=do_index_remove)

    parser_infer = subparsers.add_parser("infer")
    parser_infer.add_argument("--image", type=str, required=True)
    parser_infer.add_argument("--index-key", type=str)
    parser.add_argument("--no-visualization", action="store_true")

    parser_infer.set_defaults(func=do_infer)

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
