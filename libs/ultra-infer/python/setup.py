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
# This file refered to github.com/onnx/onnx.git

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import shutil

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
TOP_DIR = os.path.split(TOP_DIR)[0]
PACKAGE_NAME = os.getenv("PACKAGE_NAME", "ultra_infer")
wheel_name = os.getenv("WHEEL_NAME", "ultra-infer-python")

if not os.path.exists(PACKAGE_NAME):
    shutil.copytree("ultra_infer", PACKAGE_NAME)

import glob
import multiprocessing
import platform
import shlex
import subprocess
import sys
from collections import namedtuple
from contextlib import contextmanager
from distutils import log, sysconfig
from distutils.spawn import find_executable
from textwrap import dedent

import setuptools
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.develop

with open(os.path.join(TOP_DIR, "python", "requirements.txt")) as fin:
    REQUIRED_PACKAGES = fin.read()

if os.getenv("BUILD_ON_CPU", "OFF") == "ON":
    os.environ["ENABLE_PADDLE_BACKEND"] = "ON"
    os.environ["ENABLE_ORT_BACKEND"] = "ON"
    os.environ["ENABLE_OPENVINO_BACKEND"] = "ON"
    os.environ["ENABLE_VISION"] = "ON"
    os.environ["ENABLE_TEXT"] = "ON"
    os.environ["WITH_GPU"] = "OFF"

setup_configs = dict()
setup_configs["LIBRARY_NAME"] = PACKAGE_NAME
setup_configs["PY_LIBRARY_NAME"] = PACKAGE_NAME + "_main"
# Backend options
setup_configs["ENABLE_TVM_BACKEND"] = os.getenv("ENABLE_TVM_BACKEND", "OFF")
setup_configs["ENABLE_RKNPU2_BACKEND"] = os.getenv("ENABLE_RKNPU2_BACKEND", "OFF")
setup_configs["ENABLE_SOPHGO_BACKEND"] = os.getenv("ENABLE_SOPHGO_BACKEND", "OFF")
setup_configs["ENABLE_ORT_BACKEND"] = os.getenv("ENABLE_ORT_BACKEND", "OFF")
setup_configs["ENABLE_OPENVINO_BACKEND"] = os.getenv("ENABLE_OPENVINO_BACKEND", "OFF")
setup_configs["ENABLE_PADDLE_BACKEND"] = os.getenv("ENABLE_PADDLE_BACKEND", "OFF")
setup_configs["ENABLE_POROS_BACKEND"] = os.getenv("ENABLE_POROS_BACKEND", "OFF")
setup_configs["ENABLE_TRT_BACKEND"] = os.getenv("ENABLE_TRT_BACKEND", "OFF")
setup_configs["ENABLE_LITE_BACKEND"] = os.getenv("ENABLE_LITE_BACKEND", "OFF")
setup_configs["ENABLE_OM_BACKEND"] = os.getenv("ENABLE_OM_BACKEND", "OFF")
setup_configs["ENABLE_PADDLE2ONNX"] = os.getenv("ENABLE_PADDLE2ONNX", "OFF")
setup_configs["ENABLE_VISION"] = os.getenv("ENABLE_VISION", "OFF")
setup_configs["ENABLE_FLYCV"] = os.getenv("ENABLE_FLYCV", "OFF")
setup_configs["ENABLE_CVCUDA"] = os.getenv("ENABLE_CVCUDA", "OFF")
setup_configs["ENABLE_TEXT"] = os.getenv("ENABLE_TEXT", "OFF")
setup_configs["ENABLE_BENCHMARK"] = os.getenv("ENABLE_BENCHMARK", "OFF")
# Hardware options
setup_configs["WITH_GPU"] = os.getenv("WITH_GPU", "OFF")
setup_configs["WITH_IPU"] = os.getenv("WITH_IPU", "OFF")
setup_configs["WITH_OPENCL"] = os.getenv("WITH_OPENCL", "OFF")
setup_configs["WITH_TIMVX"] = os.getenv("WITH_TIMVX", "OFF")
setup_configs["WITH_DIRECTML"] = os.getenv("WITH_DIRECTML", "OFF")
setup_configs["WITH_ASCEND"] = os.getenv("WITH_ASCEND", "OFF")
setup_configs["WITH_KUNLUNXIN"] = os.getenv("WITH_KUNLUNXIN", "OFF")
setup_configs["RKNN2_TARGET_SOC"] = os.getenv("RKNN2_TARGET_SOC", "")
setup_configs["DEVICE_TYPE"] = os.getenv("DEVICE_TYPE", "")
# Custom deps settings
setup_configs["TRT_DIRECTORY"] = os.getenv("TRT_DIRECTORY", "UNDEFINED")
setup_configs["CUDA_DIRECTORY"] = os.getenv("CUDA_DIRECTORY", "/usr/local/cuda")
setup_configs["OPENCV_DIRECTORY"] = os.getenv("OPENCV_DIRECTORY", "")
setup_configs["ORT_DIRECTORY"] = os.getenv("ORT_DIRECTORY", "")
setup_configs["OPENVINO_DIRECTORY"] = os.getenv("OPENVINO_DIRECTORY", "")
setup_configs["PADDLEINFERENCE_DIRECTORY"] = os.getenv("PADDLEINFERENCE_DIRECTORY", "")
setup_configs["PADDLEINFERENCE_VERSION"] = os.getenv("PADDLEINFERENCE_VERSION", "")
setup_configs["PADDLEINFERENCE_URL"] = os.getenv("PADDLEINFERENCE_URL", "")
setup_configs["PADDLEINFERENCE_API_COMPAT_2_4_x"] = os.getenv(
    "PADDLEINFERENCE_API_COMPAT_2_4_x", "OFF"
)
setup_configs["PADDLEINFERENCE_API_COMPAT_2_5_x"] = os.getenv(
    "PADDLEINFERENCE_API_COMPAT_2_5_x", "OFF"
)
setup_configs["PADDLEINFERENCE_API_COMPAT_2_6_x"] = os.getenv(
    "PADDLEINFERENCE_API_COMPAT_2_6_x", "OFF"
)
setup_configs["PADDLEINFERENCE_API_COMPAT_DEV"] = os.getenv(
    "PADDLEINFERENCE_API_COMPAT_DEV", "OFF"
)
setup_configs["PADDLEINFERENCE_API_CUSTOM_OP"] = os.getenv(
    "PADDLEINFERENCE_API_CUSTOM_OP", "OFF"
)
setup_configs["PADDLE2ONNX_URL"] = os.getenv("PADDLE2ONNX_URL", "")
setup_configs["PADDLELITE_URL"] = os.getenv("PADDLELITE_URL", "")

# Other settings
setup_configs["BUILD_ON_JETSON"] = os.getenv("BUILD_ON_JETSON", "OFF")
setup_configs["BUILD_PADDLE2ONNX"] = os.getenv("BUILD_PADDLE2ONNX", "OFF")

if setup_configs["RKNN2_TARGET_SOC"] != "" or setup_configs["BUILD_ON_JETSON"] != "OFF":
    REQUIRED_PACKAGES = REQUIRED_PACKAGES.replace("opencv-contrib-python", "")

if wheel_name == "ultra-infer-python":
    device_type = setup_configs["DEVICE_TYPE"]
    if device_type:
        if device_type not in ["GPU", "IPU", "NPU"]:
            sys.exit(
                f"Invalid DEVICE_TYPE: '{device_type}'. Supported values are: GPU, IPU, NPU. "
                "Please update the DEVICE_TYPE environment variable accordingly."
            )
        wheel_name = f"ultra-infer-{device_type.lower()}-python"
    else:
        if (
            setup_configs["WITH_GPU"] == "ON"
            or setup_configs["BUILD_ON_JETSON"] == "ON"
        ):
            wheel_name = "ultra-infer-gpu-python"
        elif setup_configs["WITH_IPU"] == "ON":
            wheel_name = "ultra-infer-ipu-python"

if os.getenv("CMAKE_CXX_COMPILER", None) is not None:
    setup_configs["CMAKE_CXX_COMPILER"] = os.getenv("CMAKE_CXX_COMPILER")

SRC_DIR = os.path.join(TOP_DIR, PACKAGE_NAME)
PYTHON_SRC_DIR = os.path.join(TOP_DIR, "python", PACKAGE_NAME)
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, "python", ".setuptools-cmake-build")

WINDOWS = os.name == "nt"

CMAKE = find_executable("cmake3") or find_executable("cmake")
MAKE = find_executable("make")

setup_requires = []
extras_require = {}

################################################################################
# Global variables for controlling the build variant
################################################################################

# Default value is set to TRUE\1 to keep the settings same as the current ones.
# However going forward the recomemded way to is to set this to False\0
USE_MSVC_STATIC_RUNTIME = bool(os.getenv("USE_MSVC_STATIC_RUNTIME", "1") == "1")
ONNX_NAMESPACE = os.getenv("ONNX_NAMESPACE", "paddle2onnx")
################################################################################
# Version
################################################################################

try:
    git_version = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TOP_DIR)
        .decode("ascii")
        .strip()
    )
except (OSError, subprocess.CalledProcessError):
    git_version = None

extra_version_info = ""
if setup_configs["PADDLEINFERENCE_VERSION"] != "":
    extra_version_info += "." + setup_configs["PADDLEINFERENCE_VERSION"]

with open(os.path.join(TOP_DIR, "VERSION_NUMBER")) as version_file:
    VersionInfo = namedtuple(
        "VersionInfo",
        [
            "version",
            "git_version",
            "extra_version_info",
            "enable_trt_backend",
            "enable_paddle_backend",
            "with_gpu",
        ],
    )(
        version=version_file.read().strip(),
        git_version=git_version,
        extra_version_info=extra_version_info.strip("."),
        enable_trt_backend=setup_configs["ENABLE_TRT_BACKEND"],
        enable_paddle_backend=setup_configs["ENABLE_PADDLE_BACKEND"],
        with_gpu=setup_configs["WITH_GPU"],
    )

################################################################################
# Pre Check
################################################################################

assert CMAKE, 'Could not find "cmake" executable!'

################################################################################
# Utilities
################################################################################


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError("Can only cd to absolute path, got: {}".format(path))
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


################################################################################
# Customized commands
################################################################################


class NoOptionCommand(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


def get_all_files(dirname):
    files = list()
    for root, dirs, filenames in os.walk(dirname):
        for f in filenames:
            fullname = os.path.join(root, f)
            files.append(fullname)
    return files


class create_version(NoOptionCommand):
    def run(self):
        with open(os.path.join(PYTHON_SRC_DIR, "code_version.py"), "w") as f:
            f.write(
                dedent(
                    """\
            # This file is generated by setup.py. DO NOT EDIT!
            from __future__ import absolute_import
            from __future__ import division
            from __future__ import print_function
            from __future__ import unicode_literals
            version = '{version}'
            git_version = '{git_version}'
            extra_version_info = '{extra_version_info}'
            enable_trt_backend = '{enable_trt_backend}'
            enable_paddle_backend = '{enable_paddle_backend}'
            with_gpu = '{with_gpu}'
            """.format(
                        **dict(VersionInfo._asdict())
                    )
                )
            )


class cmake_build(setuptools.Command):
    """
    Compiles everything when `python setupmnm.py build` is run using cmake.
    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable.
    The number of CPUs used by `make` can be specified by passing `-j<ncpus>`
    to `setup.py build`.  By default all CPUs are used.
    """

    user_options = [
        (str("jobs="), str("j"), str("Specifies the number of jobs to use with make"))
    ]

    built = False

    def initialize_options(self):
        self.jobs = None

    def finalize_options(self):
        if sys.version_info[0] >= 3:
            self.set_undefined_options("build", ("parallel", "jobs"))
        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        self.jobs = multiprocessing.cpu_count() if self.jobs is None else int(self.jobs)

    def run(self):
        if cmake_build.built:
            return
        cmake_build.built = True
        if not os.path.exists(CMAKE_BUILD_DIR):
            os.makedirs(CMAKE_BUILD_DIR)

        with cd(CMAKE_BUILD_DIR):
            build_type = "Release"
            # configure
            cmake_args = [
                CMAKE,
                "-DPYTHON_INCLUDE_DIR={}".format(sysconfig.get_python_inc()),
                "-DPYTHON_EXECUTABLE={}".format(sys.executable),
                "-DBUILD_ULTRAINFER_PYTHON=ON",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                "-DONNX_NAMESPACE={}".format(ONNX_NAMESPACE),
                "-DPY_EXT_SUFFIX={}".format(
                    sysconfig.get_config_var("EXT_SUFFIX") or ""
                ),
            ]
            cmake_args.append("-DCMAKE_BUILD_TYPE=%s" % build_type)
            for k, v in setup_configs.items():
                cmake_args.append("-D{}={}".format(k, v))
            if WINDOWS:
                cmake_args.extend(
                    [
                        # we need to link with libpython on windows, so
                        # passing python version to window in order to
                        # find python in cmake
                        "-DPY_VERSION={}".format(
                            "{0}.{1}".format(*sys.version_info[:2])
                        ),
                    ]
                )
                if platform.architecture()[0] == "64bit":
                    cmake_args.extend(["-A", "x64", "-T", "host=x64"])
                else:
                    cmake_args.extend(["-A", "Win32", "-T", "host=x86"])
            if "CMAKE_ARGS" in os.environ:
                extra_cmake_args = shlex.split(os.environ["CMAKE_ARGS"])
                # prevent crossfire with downstream scripts
                del os.environ["CMAKE_ARGS"]
                log.info("Extra cmake args: {}".format(extra_cmake_args))
                cmake_args.extend(extra_cmake_args)
            cmake_args.append(TOP_DIR)
            subprocess.check_call(cmake_args)

            build_args = [CMAKE, "--build", os.curdir]
            if WINDOWS:
                build_args.extend(["--config", build_type])
                build_args.extend(["--", "/maxcpucount:{}".format(self.jobs)])
            else:
                build_args.extend(["--", "-j", str(self.jobs)])
            subprocess.check_call(build_args)


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command("create_version")
        self.run_command("cmake_build")

        generated_python_files = glob.glob(
            os.path.join(CMAKE_BUILD_DIR, PACKAGE_NAME, "*.py")
        ) + glob.glob(os.path.join(CMAKE_BUILD_DIR, PACKAGE_NAME, "*.pyi"))

        for src in generated_python_files:
            dst = os.path.join(TOP_DIR, os.path.relpath(src, CMAKE_BUILD_DIR))
            self.copy_file(src, dst)

        return setuptools.command.build_py.build_py.run(self)


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command("build_py")
        setuptools.command.develop.develop.run(self)


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command("cmake_build")
        setuptools.command.build_ext.build_ext.run(self)

    def build_extensions(self):
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = os.path.basename(self.get_ext_filename(fullname))

            lib_path = CMAKE_BUILD_DIR
            if os.name == "nt":
                debug_lib_dir = os.path.join(lib_path, "Debug")
                release_lib_dir = os.path.join(lib_path, "Release")
                if os.path.exists(debug_lib_dir):
                    lib_path = debug_lib_dir
                elif os.path.exists(release_lib_dir):
                    lib_path = release_lib_dir
            src = os.path.join(lib_path, filename)
            dst = os.path.join(os.path.realpath(self.build_lib), PACKAGE_NAME, filename)
            self.copy_file(src, dst)


cmdclass = {
    "create_version": create_version,
    "cmake_build": cmake_build,
    "build_py": build_py,
    "develop": develop,
    "build_ext": build_ext,
}

################################################################################
# Extensions
################################################################################

ext_modules = [
    setuptools.Extension(
        name=str(PACKAGE_NAME + "." + setup_configs["PY_LIBRARY_NAME"]), sources=[]
    ),
]

################################################################################
# Packages
################################################################################

# no need to do fancy stuff so far
if PACKAGE_NAME != "ultra_infer":
    packages = setuptools.find_packages(exclude=["ultra_infer*", "scripts"])
else:
    packages = setuptools.find_packages(exclude=["xencrypt*", "scripts"])

################################################################################
# Test
################################################################################

if sys.version_info[0] == 3:
    # Mypy doesn't work with Python 2
    extras_require["mypy"] = ["mypy==0.600"]

################################################################################
# Pyonly
################################################################################

extras_require["pyonly"] = [
    "pyyaml",
    "pillow<10.0.0",
    "pandas>=0.25.0,<=1.3.5",
    "pycocotools",
    "matplotlib",
    "chinese_calendar",
    "joblib",
    "scikit-image",
    "scikit-learn>=1.3.2",
    "tokenizers",
]

################################################################################
# Final
################################################################################

package_data = {PACKAGE_NAME: ["LICENSE", "ThirdPartyNotices.txt"]}

if sys.argv[1] == "install" or sys.argv[1] == "bdist_wheel":
    shutil.copy(
        os.path.join(TOP_DIR, "ThirdPartyNotices.txt"),
        os.path.join(TOP_DIR, "python", PACKAGE_NAME),
    )
    shutil.copy(
        os.path.join(TOP_DIR, "LICENSE"), os.path.join(TOP_DIR, "python", PACKAGE_NAME)
    )
    if not os.path.exists(
        os.path.join(TOP_DIR, "python", PACKAGE_NAME, "libs", "third_libs")
    ):
        print(
            f"Didn't detect path: {PACKAGE_NAME}/libs/third_libs exist, please execute `python setup.py build` first"
        )
        sys.exit(0)
    from scripts.process_libraries import process_libraries

    all_lib_data = process_libraries(os.path.split(os.path.abspath(__file__))[0])
    package_data[PACKAGE_NAME].extend(all_lib_data)
    setuptools.setup(
        name=wheel_name,
        version=VersionInfo.version + extra_version_info,
        ext_modules=ext_modules,
        description="Deploy Kit Tool For Deeplearning models.",
        packages=packages,
        package_data=package_data,
        include_package_data=True,
        setup_requires=setup_requires,
        extras_require=extras_require,
        author="ultra_infer",
        install_requires=REQUIRED_PACKAGES,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        license="Apache 2.0",
    )
else:
    setuptools.setup(
        name=wheel_name,
        version=VersionInfo.version + extra_version_info,
        description="Deploy Kit Tool For Deeplearning models.",
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        packages=packages,
        package_data=package_data,
        include_package_data=False,
        setup_requires=setup_requires,
        extras_require=extras_require,
        author="ultra_infer",
        install_requires=REQUIRED_PACKAGES,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        license="Apache 2.0",
    )
