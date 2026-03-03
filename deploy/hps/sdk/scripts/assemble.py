#!/usr/bin/env python

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

import argparse
import ast
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile

TARGET_NAME_PATTERN = "paddlex_hps_{pipeline_name}_sdk"
ARCHIVE_SUFFIX = ".tar.gz"

BASE_DIR = pathlib.Path.cwd()
PIPELINES_DIR = BASE_DIR / "pipelines"
COMMON_DIR = BASE_DIR / "common"
CLIENT_LIB_PATH = BASE_DIR / "paddlex-hps-client"
OUTPUT_DIR = BASE_DIR / "output"
NAME_MAPPINGS_PATH = BASE_DIR / "_name_mappings.py"


def _load_pipeline_app_router():
    """Parse PIPELINE_APP_ROUTER from the mounted name_mappings.py file."""
    if not NAME_MAPPINGS_PATH.exists():
        return {}
    source = NAME_MAPPINGS_PATH.read_text()
    # NOTE: We use `ast` to extract the dict value without importing the module,
    # because name_mappings.py may have dependencies that are not available in
    # the build environment. `ast.parse` + `ast.literal_eval` safely evaluates
    # the dict literal from the source code.
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PIPELINE_APP_ROUTER":
                    return ast.literal_eval(node.value)
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline_names", type=str, metavar="pipeline-names", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--no-server",
        action="store_true",
    )
    parser.add_argument(
        "--no-client",
        action="store_true",
    )
    args = parser.parse_args()

    if args.all and args.pipeline_names:
        print(
            "Cannot specify `--all` and `pipeline-names` at the same time",
            file=sys.stderr,
        )
        sys.exit(2)

    pipeline_app_router = _load_pipeline_app_router()

    if args.all:
        pipeline_names = [p.name for p in PIPELINES_DIR.iterdir()]
    else:
        pipeline_names = args.pipeline_names

    if not pipeline_names:
        sys.exit(0)

    with_server = not args.no_server
    with_client = not args.no_client

    OUTPUT_DIR.mkdir(exist_ok=True)

    if with_client:
        # HACK: Make a copy to avoid creating files in the source directory
        with tempfile.TemporaryDirectory() as td:
            tmp_client_lib_path = shutil.copytree(
                CLIENT_LIB_PATH, str(pathlib.Path(td, CLIENT_LIB_PATH.name))
            )
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--wheel-dir",
                    str(OUTPUT_DIR),
                    tmp_client_lib_path,
                ]
            )
            client_lib_whl_path = next(OUTPUT_DIR.glob("paddlex_hps_client*.whl"))

    for pipeline_name in pipeline_names:
        print("=" * 30)
        print(f"Pipeline: {pipeline_name}")
        pipeline_dir = PIPELINES_DIR / pipeline_name

        mapped_pipeline_dir = None
        if pipeline_name in pipeline_app_router:
            source_name = pipeline_app_router[pipeline_name]
            source_dir = PIPELINES_DIR / source_name
            if not source_dir.exists():
                sys.exit(
                    f"Source pipeline directory {source_dir} not found"
                    f" for mapped pipeline {pipeline_name}"
                )
            mapped_pipeline_dir = pipeline_dir
            pipeline_dir = source_dir
            print(f"Using source pipeline: {source_name}")
        elif not pipeline_dir.exists():
            sys.exit(f"{pipeline_dir} not found")

        tgt_name = TARGET_NAME_PATTERN.format(pipeline_name=pipeline_name)
        tgt_dir = OUTPUT_DIR / tgt_name

        if tgt_dir.exists():
            print(f"Removing existing target directory: {tgt_dir}")
            shutil.rmtree(tgt_dir)

        if with_server:
            shutil.copytree(pipeline_dir / "server", tgt_dir / "server")
            shutil.copy(COMMON_DIR / "server.sh", tgt_dir / "server")
            for dir_ in (tgt_dir / "server" / "model_repo").iterdir():
                if dir_.is_dir():
                    if (dir_ / "config.pbtxt").exists():
                        continue
                    for device_type in ("cpu", "gpu"):
                        config_path = dir_ / f"config_{device_type}.pbtxt"
                        if not config_path.exists():
                            shutil.copy(
                                COMMON_DIR / f"config_{device_type}.pbtxt", config_path
                            )

        if with_client:
            shutil.copytree(pipeline_dir / "client", tgt_dir / "client")
            shutil.copy(client_lib_whl_path, tgt_dir / "client")

        shutil.copy(pipeline_dir / "version.txt", tgt_dir / "version.txt")

        if mapped_pipeline_dir is not None:
            shutil.copytree(mapped_pipeline_dir, tgt_dir, dirs_exist_ok=True)

        arch_path = OUTPUT_DIR / (tgt_name + ARCHIVE_SUFFIX)
        print(f"Creating archive: {arch_path}")
        with tarfile.open(arch_path, "w:gz") as tar:
            tar.add(tgt_dir, arcname=tgt_dir.name)
        print("Done" + "\n" + "=" * 30)
