#!/usr/bin/env python

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import textwrap
from datetime import datetime
from pathlib import Path

DEP_SPECS = {
    "aiohttp": ">= 3.9",
    "bce-python-sdk": ">= 0.9",
    "chardet": "",
    "chinese-calendar": "",
    "colorlog": "",
    "decord": "== 0.6.0; (platform_machine == 'x86_64' or platform_machine == 'AMD64') and sys_platform != 'darwin'",
    "faiss-cpu": "",
    "fastapi": ">= 0.110",
    "filelock": "",
    "filetype": ">= 1.2",
    "ftfy": "",
    "GPUtil": ">= 1.4",
    "imagesize": "",
    "Jinja2": "",
    "joblib": "",
    "langchain": "== 0.2.17",
    "langchain-community": "== 0.2.17",
    "langchain-core": "",
    "langchain-openai": "== 0.1.25",
    "lxml": "",
    "matplotlib": "",
    "numpy": [
        "== 1.24.4; python_version < '3.12'",
        "== 1.26.4; python_version >= '3.12'",
    ],
    "openai": "== 1.63.2",
    "opencv-contrib-python": "== 4.10.0.84",
    "openpyxl": "",
    "packaging": "",
    "paddle2onnx": ">= 2",
    "pandas": "",
    "pillow": "",
    "premailer": "",
    "prettytable": "",
    "py-cpuinfo": "",
    "pyclipper": "",
    "pycocotools": "",
    "pydantic": ">= 2",
    "PyMuPDF": "",
    "PyYAML": "== 6.0.2",
    "regex": "",
    "requests": "",
    "ruamel.yaml": "",
    "scikit-image": "",
    "scikit-learn": "",
    "shapely": "",
    "six": "",
    "soundfile": "",
    "starlette": ">= 0.36",
    "tokenizers": "== 0.19.1",
    "tqdm": "",
    "typing-extensions": "",
    "ujson": "",
    "uvicorn": ">= 0.16",
    "yarl": ">= 1.9",
}

REQUIRED_DEPS = [
    "chardet",
    "colorlog",
    "filelock",
    "GPUtil",
    "numpy",
    "packaging",
    "pillow",
    "py-cpuinfo",
    "pydantic",
    "PyYAML",
    "requests",
    "ruamel.yaml",
    "typing-extensions",
    "ujson",
]

EXTRAS = {
    "base": {
        "cv": [
            "faiss-cpu",
            "matplotlib",
            "opencv-contrib-python",
            "pycocotools",
            "scikit-image",
        ],
        "multimodal": [
            "ftfy",
            "Jinja2",
            "regex",
            "six",
        ],
        "ie": [
            "ftfy",
            "imagesize",
            "langchain",
            "langchain-community",
            "langchain-core",
            "langchain-openai",
            "lxml",
            "openai",
            "opencv-contrib-python",
            "openpyxl",
            "premailer",
            "prettytable",
            "pyclipper",
            "PyMuPDF",
            "scikit-learn",
            "shapely",
            "tokenizers",
        ],
        "ocr": [
            "ftfy",
            "imagesize",
            "lxml",
            "opencv-contrib-python",
            "openpyxl",
            "premailer",
            "prettytable",
            "pyclipper",
            "PyMuPDF",
            "scikit-learn",
            "shapely",
            "tokenizers",
        ],
        "speech": [
            "ftfy",
            "Jinja2",
            "regex",
            "six",
            "soundfile",
            "tqdm",
        ],
        "ts": [
            "chinese-calendar",
            "joblib",
            "matplotlib",
            "pandas",
            "scikit-learn",
        ],
        "video": [
            "decord",
            "opencv-contrib-python",
        ],
    },
    "plugins": {
        "serving": [
            "aiohttp",
            "bce-python-sdk",
            "fastapi",
            "filetype",
            "starlette",
            "uvicorn",
            "yarl",
        ],
        "paddle2onnx": [
            "paddle2onnx",
        ],
    },
}

HEADER_TEMPLATE = """# Copyright (c) {year} PaddlePaddle Authors. All Rights Reserved.
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
"""


def get_header():
    return HEADER_TEMPLATE.format(year=datetime.today().year)


def get_script_path():
    script_path = Path(__file__)
    repo_root = script_path.parent.parent
    script_path = script_path.relative_to(repo_root)
    return script_path


def validate(required_deps, extras):
    known_deps = set(DEP_SPECS)
    required_deps = set(required_deps)
    diff = required_deps - known_deps
    if diff:
        print(
            f"Unknown dependencies in the required dependency list: {diff}",
            file=sys.stderr,
        )
        exit(1)
    optional_deps = set()
    for extra_type, extras_ in extras.items():
        for name, deps in extras_.items():
            name = extra_type + ":" + name
            deps = set(deps)
            diff = deps - known_deps
            if diff:
                print(f"Unknown dependencies for {repr(name)}: {diff}", file=sys.stderr)
                exit(1)
            diff = deps & required_deps
            if diff:
                print(
                    f"There should be no overlap between the required dependencies and the optional dependencies, but found {diff} for {repr(name)}",
                    file=sys.stderr,
                )
                exit(1)
            optional_deps |= deps
    diff = known_deps - (required_deps | optional_deps)
    if diff:
        print(f"These dependencies are not used: {diff}", file=sys.stderr)
        exit(1)


def create_package(path):
    print(f"Creating '{path}'")
    path.mkdir(parents=True)
    with (path / "__init__.py").open("w", encoding="utf-8") as f:
        header = get_header()
        f.write(header)
        f.write("\n")
        f.write(
            f"# This package was created by {repr(get_script_path().as_posix())}.\n"
        )
        f.write("# Modifications should be made carefully.\n")


def ensure_package(path):
    if not path.exists():
        create_package(path)
    elif not path.is_dir():
        sys.exit(f"{repr(path.as_posix())} is not a directory")


def write_deps(path, deps):
    if path.exists():
        print(f"Overwriting '{path}'")
    else:
        print(f"Creating '{path}'")
    with path.open("w", encoding="utf-8") as f:
        header = get_header()
        f.write(header)

        f.write("\n")
        f.write(
            f"# This file was generated by {repr(get_script_path().as_posix())}.\n# DO NOT edit it directly.\n"
        )
        f.write("\n")
        f.write("DEPS = {\n")
        items = []
        indent_prefix = " " * 4
        for dep in deps:
            dep_spec = DEP_SPECS[dep]
            if isinstance(dep_spec, list):
                val = (
                    "[\n"
                    + textwrap.indent(
                        "\n".join(f'"{item}",' for item in dep_spec), indent_prefix
                    )
                    + "\n]"
                )
            else:
                val = f'"{dep_spec}"'
            items.append(f'"{dep}": {val},')
        f.write(textwrap.indent("\n".join(items), indent_prefix))
        f.write("\n")
        f.write("}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("paddlex", "deps"))
    args = parser.parse_args()

    validate(REQUIRED_DEPS, EXTRAS)

    ensure_package(args.output_dir)
    write_deps(args.output_dir / "required.py", REQUIRED_DEPS)

    for extra_type, extras in EXTRAS.items():
        for name, deps in extras.items():
            group_path = args.output_dir / extra_type
            ensure_package(group_path)
            write_deps(group_path / f"{name}.py", deps)
