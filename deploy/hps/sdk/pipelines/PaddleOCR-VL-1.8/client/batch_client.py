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


"""Submit one image per request to a PaddleOCR-VL 1.8 HPS HTTP gateway."""

import argparse
import base64
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_PAGE_CONCURRENCY = 160
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def submit_page(endpoint, page, output_dir, timeout):
    data = json.dumps(
        {
            "file": base64.b64encode(page.read_bytes()).decode("ascii"),
            "fileType": 1,
            "visualize": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if result.get("errorCode") != 0:
        raise RuntimeError(result.get("errorMsg", "HPS request failed"))
    (output_dir / f"{page.stem}.json").write_text(
        json.dumps(result["result"], ensure_ascii=False), encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://localhost:8080/layout-parsing")
    parser.add_argument(
        "--concurrency", type=positive_int, default=DEFAULT_PAGE_CONCURRENCY
    )
    parser.add_argument("--timeout", type=positive_int, default=600)
    args = parser.parse_args(argv)
    if not args.images_dir.is_dir():
        parser.error("--images-dir must be a directory")
    pages = sorted(
        p
        for p in args.images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not pages or len({p.stem for p in pages}) != len(pages):
        parser.error("provide image files with unique stems")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def run(page):
        try:
            submit_page(args.url, page, args.output_dir, args.timeout)
            return True
        except Exception as exc:
            print(f"{page.name}: {exc}")
            return False

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        ok = sum(pool.map(run, pages))
    print(f"{ok}/{len(pages)} pages succeeded")
    return 0 if ok == len(pages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
