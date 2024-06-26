# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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


import os
import glob
import itertools

from setuptools import find_packages
from setuptools import setup


def readme():
    """get readme
    """
    with open('README.md', 'r', encoding='utf-8') as file:
        return file.read()


def dependencies():
    """get dependencies
    """
    with open('requirements.txt', 'r') as file:
        return file.read()


def version():
    """get version
    """
    with open(os.path.join('paddlex', '.version'), 'r') as file:
        return file.read().rstrip()


def packages_and_package_data():
    """get packages and package_data
    """

    def _recursively_find(pattern, exts=None):
        for dir_ in glob.iglob(pattern):
            for root, _, files in os.walk(dir_):
                for f in files:
                    if exts is not None:
                        ext = os.path.splitext(f)[1]
                        if ext not in exts:
                            continue
                    yield os.path.join(root, f)

    pkgs = find_packages(include=[
        'paddlex',
        'paddlex.modules',
        'paddlex.modules.*',
        'paddlex.pipelines',
        'paddlex.pipelines.*',
        'paddlex.repo_manager',
        'paddlex.repo_apis',
        'paddlex.repo_apis.*',
        'paddlex.utils',
        'paddlex.utils.*',
    ])
    pkg_data = []
    for p in itertools.chain(
            # Configuration files
            _recursively_find(
                "paddlex/configs/*", exts=['.yml', '.yaml']), ):
        parts = os.path.normpath(p).split(os.sep)
        ext = os.path.splitext(p)[1]
        # Globally exclude Python bytecode files
        if ext in ('.pyc', '.pyo'):
            continue
        # According to https://setuptools.pypa.io/en/latest/userguide/datafiles.html
        rp = '/'.join(parts[1:])
        pkg_data.append(rp)
    pkg_data.append('.version')
    pkg_data.append('utils/fonts/PingFang-SC-Regular.ttf')
    pkg_data.append('repo_manager/requirements.txt')
    return pkgs, {'paddlex': pkg_data}


def check_paddle_version():
    """check paddle version
    """
    import paddle
    supported_versions = ['3.0', '0.0']
    version = paddle.__version__
    # Recognizable version number: major.minor.patch
    major, minor, patch = version.split('.')
    # Ignore patch
    version = f"{major}.{minor}"
    if version not in supported_versions:
        raise RuntimeError(
            f"The {version} version of PaddlePaddle is not supported. "
            f"Please install one of the following versions of PaddlePaddle: {supported_versions}."
        )


if __name__ == '__main__':
    check_paddle_version()

    pkgs, pkg_data = packages_and_package_data()

    s = setup(
        name='paddlex',
        version=version(),
        description=('Low-code development tool based on PaddlePaddle.'),
        long_description=readme(),
        author='PaddlePaddle Authors',
        author_email='',
        install_requires=dependencies(),
        packages=pkgs,
        package_data=pkg_data,
        entry_points={
            'console_scripts': ['paddlex = paddlex.paddlex_cli:main', ],
        },
        # PyPI package information
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='Apache 2.0',
        keywords=['paddlepaddle'])
