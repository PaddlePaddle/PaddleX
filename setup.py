# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import setuptools
import sys

long_description = "PaddleX. A end-to-end deeplearning model development toolkit base on PaddlePaddle\n\n"

setuptools.setup(
    name="paddlex",
    version='1.0.5',
    author="paddlex",
    author_email="paddlex@baidu.com",
    description=long_description,
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/PaddlePaddle/PaddleX",
    packages=setuptools.find_packages(),
    setup_requires=['cython', 'numpy'],
    install_requires=[
        "pycocotools;platform_system!='Windows'", 'pyyaml', 'colorama', 'tqdm',
        'paddleslim==1.0.1', 'visualdl>=2.0.0b', 'paddlehub>=1.6.2'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    entry_points={'console_scripts': ['paddlex=paddlex.command:main', ]})
