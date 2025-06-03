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


def get_device_type():
    import paddle

    device_str = paddle.get_device()
    return device_str.split(":")[0]


def get_paddle_version():
    import paddle

    version = paddle.__version__
    if "-" in version:
        version, tag = version.split("-")
    else:
        tag = None
    version = version.split(".")
    assert len(version) == 3
    major_v, minor_v, patch_v = map(int, version)
    if tag:
        return major_v, minor_v, patch_v, tag
    else:
        return major_v, minor_v, patch_v, None


def get_paddle_cuda_version():
    import paddle.version

    cuda_version = paddle.version.cuda()
    if cuda_version == "False":
        return None
    return tuple(map(int, cuda_version.split(".")))


def get_paddle_cudnn_version():
    import paddle.version

    cudnn_version = paddle.version.cudnn()
    if cudnn_version == "False":
        return None
    return tuple(map(int, cudnn_version.split(".")))


# Should we also support getting the runtime versions of CUDA and cuDNN?
