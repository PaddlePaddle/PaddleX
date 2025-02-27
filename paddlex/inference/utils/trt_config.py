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

TRT_CFG = {
    "DETR-R50": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B0": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B1": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B2": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B3": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B4": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B5": {"optimization_level": 4, "workspace_size": 1 << 32},
    "LaTeX_OCR_rec": {"disable_ops": ["pd_op.slice"]},
}
