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


def get_config(backend):
    if backend == "fastdeploy":
        return {
            "gpu-memory-utilization": 0.7,
            "max-model-len": 16384,
            "max-num-batched-tokens": 16384,
            "max-num-seqs": 256,
            "workers": 4,
            "graph-optimization-config": '{"graph_opt_level":0, "use_cudagraph":true}',
        }
    elif backend == "vllm":
        return {
            "trust-remote-code": True,
            "gpu-memory-utilization": 0.5,
            "max-model-len": 16384,
            "max-num-batched-tokens": 131072,
            "api-server-count": 4,
        }
    elif backend == "sglang":
        return {
            "trust-remote-code": True,
            "mem-fraction-static": 0.5,
            "context-length": 16384,
            "max-prefill-tokens": 131072,
        }
    else:
        raise ValueError(f"Unsupported backend: {backend}")
