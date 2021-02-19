# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

long_description = "PaddlePaddle Entire Process Development Toolkit"

setuptools.setup(
    name="paddlex",
    version='1.3.6',
    author="paddlex",
    author_email="paddlex@baidu.com",
    description=long_description,
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/PaddlePaddle/PaddleX",
    packages=setuptools.find_packages(),
    package_data={
        'paddlex_restful/restful/templates':
        ['paddlex_restful/restful/templates/paddlex_restful_demo.html']
    },
    include_package_data=True,
    data_files=[('paddlex_restful/restful/templates', [
        'paddlex_restful/restful/templates/paddlex_restful_demo.html'
    ])],
    include_data_files=True,
    setup_requires=['cython', 'numpy'],
    install_requires=[
        "pycocotools;platform_system!='Windows'",
        'pyyaml',
        'colorama',
        'tqdm',
        'paddleslim==1.1.1',
        'visualdl>=2.0.0',  #'paddlehub==1.8.2',
        'shapely>=1.7.0',
        'opencv-python',
        'flask_cors',
        'sklearn',
        'psutil',
        'xlwt'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    entry_points={
        'console_scripts': [
            'paddlex=paddlex.command:main',
            'paddlex_restful=paddlex_restful.command:main'
        ]
    })
