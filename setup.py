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


import glob
import itertools
import os
from pathlib import Path

from setuptools import find_packages, setup

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
    # Currently it is not easy to make `pandas` optional
    "pandas",
    "pillow",
    "prettytable",
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
            # Currently `PyMuPDF` is required by the image batch sampler
            "PyMuPDF",
            "scikit-image",
        ],
        "multimodal": [
            "ftfy",
            "Jinja2",
            "opencv-contrib-python",
            # For the same reason as in `cv`
            "PyMuPDF",
            "regex",
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
            "soundfile",
            "tqdm",
        ],
        "ts": [
            "chinese-calendar",
            "joblib",
            "matplotlib",
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
    },
}


def _get_dep_specs(deps):
    dep_specs = []
    for dep in deps:
        val = DEP_SPECS[dep]
        if not isinstance(val, list):
            val = [val]
        for v in val:
            if not v:
                dep_specs.append(dep)
            else:
                dep_specs.append(dep + " " + v)
    return dep_specs


def _sort_dep_specs(dep_specs):
    return sorted(dep_specs, key=str.lower)


def readme():
    """get readme"""
    with open("README.md", "r", encoding="utf-8") as file:
        return file.read()


def dependencies():
    dep_specs = _get_dep_specs(REQUIRED_DEPS)
    return _sort_dep_specs(dep_specs)


def extras():
    dic = {}
    all_dep_specs = set()
    for group_name, group in EXTRAS.items():
        group_dep_specs = set()
        for extra_name, extra_deps in group.items():
            extra_dep_specs = _get_dep_specs(extra_deps)
            dic[extra_name] = _sort_dep_specs(extra_dep_specs)
            group_dep_specs.update(extra_dep_specs)
            dic[group_name] = _sort_dep_specs(group_dep_specs)
            all_dep_specs.update(group_dep_specs)
    dic["all"] = _sort_dep_specs(all_dep_specs)
    return dic


def version():
    """get version"""
    with open(os.path.join("paddlex", ".version"), "r") as file:
        return file.read().rstrip()


def get_data_files(directory: str, filetypes: list = None):
    all_files = []
    filetypes = filetypes or []

    for root, _, files in os.walk(directory):
        rel_root = os.path.relpath(root, directory)
        for file in files:
            filepath = os.path.join(rel_root, file)
            filetype = os.path.splitext(file)[1][1:]
            if filetype in filetypes:
                all_files.append(filepath)

    return all_files


def packages_and_package_data():
    """get packages and package_data"""

    def _recursively_find(pattern, exts=None):
        for dir_ in glob.iglob(pattern):
            for root, _, files in os.walk(dir_):
                for f in files:
                    if exts is not None:
                        ext = os.path.splitext(f)[1]
                        if ext not in exts:
                            continue
                    yield os.path.join(root, f)

    pkgs = find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"])
    pkg_data = []
    for p in itertools.chain(
        _recursively_find("paddlex/configs/*", exts=[".yml", ".yaml"]),
    ):
        if Path(p).suffix in (".pyc", ".pyo"):
            continue
        pkg_data.append(Path(p).relative_to("paddlex").as_posix())
    pipeline_config = [
        Path(p).relative_to("paddlex").as_posix()
        for p in glob.glob("paddlex/pipelines/*.yaml")
    ]
    pkg_data.append("inference/pipelines/ppchatocrv3/ch_prompt.yaml")
    pkg_data.extend(pipeline_config)
    pkg_data.append(".version")
    pkg_data.append("hpip_links.html")
    pkg_data.append("inference/utils/hpi_model_info_collection.json")
    ops_file_dir = "paddlex/ops"
    ops_file_types = ["h", "hpp", "cpp", "cc", "cu"]
    return pkgs, {
        "paddlex.ops": get_data_files(ops_file_dir, ops_file_types),
        "paddlex": pkg_data,
    }


if __name__ == "__main__":
    pkgs, pkg_data = packages_and_package_data()

    s = setup(
        name="paddlex",
        version=version(),
        description=("Low-code development tool based on PaddlePaddle."),
        long_description=readme(),
        long_description_content_type="text/markdown",
        author="PaddlePaddle Authors",
        author_email="",
        install_requires=dependencies(),
        extras_require=extras(),
        packages=pkgs,
        package_data=pkg_data,
        entry_points={
            "console_scripts": [
                "paddlex = paddlex.__main__:console_entry",
            ],
        },
        # PyPI package information
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
        license="Apache 2.0",
        keywords=["paddlepaddle"],
    )
