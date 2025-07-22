#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

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
    parser.add_argument("--target-language", type=str, default="zh")
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

    output = triton_request(client, "doctrans-visual", input_)
    ensure_no_error(output, "Failed to analyze the images")
    result_visual = output["result"]

    markdown_list = []
    for i, res in enumerate(result_visual["layoutParsingResults"]):
        print(res["prunedResult"])
        md_dir = Path(f"markdown_{i}")
        md_dir.mkdir(exist_ok=True)
        (md_dir / "doc.md").write_text(res["markdown"]["text"])
        for img_path, img in res["markdown"]["images"].items():
            img_path = md_dir / img_path
            img_path.parent.mkdir(parents=True, exist_ok=True)
            utils.save_output_file(img, img_path)
        print(f"Markdown document to be translated is saved at {md_dir / 'doc.md'}")
        del res["markdown"]["images"]
        markdown_list.append(res["markdown"])
        for img_name, img in res["outputImages"].items():
            img_path = f"{img_name}_{i}.jpg"
            utils.save_output_file(img, img_path)
            print(f"Output image saved at {img_path}")

    input_ = {
        "markdownList": markdown_list,
        "targetLanguage": args.target_language,
    }
    output = triton_request(client, "doctrans-translate", input_)
    ensure_no_error(output, "Failed to translate the markdown")
    result_translate = output["result"]

    for i, res in enumerate(result_translate["translationResults"]):
        md_dir = Path(f"markdown_{i}")
        (md_dir / "doc_translated.md").write_text(res["markdown"]["text"])
        print(f"Translated markdown document saved at {md_dir / 'doc_translated.md'}")


if __name__ == "__main__":
    main()
