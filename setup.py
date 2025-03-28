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
from runpy import run_path

from setuptools import find_packages, setup


def _get_deps_from_module(module_path):
    mod_globals = run_path(str(module_path))
    deps = []
    for name, spec in mod_globals["DEPS"].items():
        if not isinstance(spec, list):
            spec = [spec]
        for item in spec:
            if item:
                deps.append(name + " " + item)
            else:
                deps.append(name)
    return deps


def readme():
    """get readme"""
    with open("README.md", "r", encoding="utf-8") as file:
        return file.read()


def dependencies():
    """get dependencies"""
    return _get_deps_from_module(Path("paddlex", "deps", "required.py"))


def extras():
    dic = {}
    all_deps = set()
    for child in Path("paddlex", "deps").iterdir():
        if child.is_dir():
            group_name = child.stem
            group_deps = set()
            for mod_path in child.glob("*.py"):
                if mod_path.name == "__init__.py":
                    continue
                extra = mod_path.stem
                deps = _get_deps_from_module(mod_path)
                dic[extra] = deps
                group_deps.update(deps)
            dic[group_name] = sorted(group_deps, key=str.lower)
            all_deps.update(group_deps)
    dic["all"] = sorted(all_deps, key=str.lower)
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
    pkg_data.extend(get_data_files("requirements", "txt"))
    pkg_data.append("repo_manager/requirements.txt")
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
