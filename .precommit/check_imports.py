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

# TODO: Less verbose

import ast
import pathlib
import re
import sys
import traceback
from collections import deque

from stdlib_list import stdlib_list

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from setup import DEP_SPECS, REQUIRED_DEPS

# NOTE: We do not use `importlib.metadata.packages_distributions` here because
# 1. It is supported only in Python 3.10+.
# 2. It requires the packages to be installed, but we are doing a static check.
MOD_TO_DEP = {
    "aiohttp": "aiohttp",
    "baidubce": "bce-python-sdk",
    "chardet": "chardet",
    "chinese_calendar": "chinese-calendar",
    "colorlog": "colorlog",
    "decord": "decord",
    "faiss": "faiss-cpu",
    "fastapi": "fastapi",
    "filelock": "filelock",
    "filetype": "filetype",
    "ftfy": "ftfy",
    "GPUtil": "GPUtil",
    "imagesize": "imagesize",
    "jinja2": "Jinja2",
    "joblib": "joblib",
    "langchain": "langchain",
    "langchain_community": "langchain-community",
    "langchain_core": "langchain-core",
    "langchain_openai": "langchain-openai",
    "lxml": "lxml",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "openai": "openai",
    "cv2": "opencv-contrib-python",
    "openpyxl": "openpyxl",
    "packaging": "packaging",
    "pandas": "pandas",
    "PIL": "pillow",
    "premailer": "premailer",
    "prettytable": "prettytable",
    "cpuinfo": "py-cpuinfo",
    "pyclipper": "pyclipper",
    "pycocotools": "pycocotools",
    "pydantic": "pydantic",
    "fitz": "PyMuPDF",
    "yaml": "PyYAML",
    "regex": "regex",
    "requests": "requests",
    "ruamel.yaml": "ruamel.yaml",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "shapely": "shapely",
    "soundfile": "soundfile",
    "starlette": "starlette",
    "tokenizers": "tokenizers",
    "tqdm": "tqdm",
    "typing_extensions": "typing-extensions",
    "ujson": "ujson",
    "uvicorn": "uvicorn",
    "yarl": "yarl",
}
assert (
    set(MOD_TO_DEP.values()) == DEP_SPECS.keys()
), f"`MOD_TO_DEP` should be updated to match `DEP_SPECS`. Symmetric difference: {set(MOD_TO_DEP.values()) ^ DEP_SPECS.keys()}"
MOD_PATTERN = re.compile(
    rf"^(?:{'|'.join([re.escape(mod) for mod in MOD_TO_DEP])})(?=\.|$)"
)
STDLIB_MODS = set(stdlib_list())
SPECIAL_KNOWN_MODS = {
    "paddleseg",
    "paddleclas",
    "paddledet",
    "paddlets",
    "paddlenlp",
    "paddlespeech",
    "parl",
    "paddlemix",
    "paddle3d",
    "paddlevideo",
}
MANUALLY_MANAGED_HEAVY_MODS = {"paddle", "paddle_custom_device", "ultra_infer"}


def check(file_path):
    # TODO:
    # 1. Handle more cases, e.g., `from ruamel import yaml`.
    # 2. Find unused dependencies.
    # 3. Better output format.

    with open(file_path, "r", encoding="utf-8") as f:
        file_contents = f.read()

    try:
        tree = ast.parse(file_contents)
    except Exception:
        print(
            f"Failed to parse the source code in `{file_path}` into an AST node:\n{traceback.format_exc()}"
        )
        return False

    # 1. Never import unknown modules
    # 2. Don't import optional third-party modules at the top level
    unknown_modules_found = False
    top_level_imports_found = False
    q = deque()
    for child in ast.iter_child_nodes(tree):
        q.append((child, 1))
    while q:
        node, level = q.popleft()
        mods = set()
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                mods.add(mod)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                mod = node.module
                mods.add(mod)
        for mod in mods:
            pos = f"{file_path}:{node.lineno}:{node.col_offset}"
            tl = mod.split(".")[0]
            if tl == "paddlex" or tl in SPECIAL_KNOWN_MODS or tl in STDLIB_MODS:
                continue
            elif tl in MANUALLY_MANAGED_HEAVY_MODS:
                if level == 1:
                    print(
                        f"{pos}: Module of a manually managed heavy dependency imported at the top level: {mod}"
                    )
                    top_level_imports_found = True
            elif match_ := MOD_PATTERN.match(mod):
                if level == 1:
                    dep = MOD_TO_DEP[match_.group(0)]
                    if dep not in REQUIRED_DEPS:
                        print(
                            f"{pos}: Module of an optional dependency imported at the top level: {mod}"
                        )
                        top_level_imports_found = True
            else:
                print(f"{pos}: Unknown module imported: {mod}")
                unknown_modules_found = True
        for child in ast.iter_child_nodes(node):
            q.append((child, level + 1))

    return unknown_modules_found | (top_level_imports_found << 1)


def main():
    files = sys.argv[1:]
    flag = 0
    for file in files:
        ret = check(file)
        flag |= ret
    if flag:
        if flag & 1:
            curr_script_path = pathlib.Path(__file__)
            curr_script_path = curr_script_path.relative_to(
                curr_script_path.parent.parent
            )
            print(
                f"If a new dependency should be added, please update `setup.py` and `{curr_script_path}`."
            )
        if (flag >> 1) & 1:
            print(
                "Please put the imports from optional dependencies and manually managed heavy dependencies inside a conditional body or within a function body."
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
