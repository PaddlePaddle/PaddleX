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


import numpy as np
import paddle


def fuse_param_func():
    def fn(fuse_params, is_qkv=False, num_heads=None, num_key_value_heads=None):
        concat_fn = np.concatenate
        split_fn = np.split
        if isinstance(fuse_params[0], paddle.Tensor):
            concat_fn = paddle.concat
            split_fn = paddle.split

        if is_qkv:
            assert (
                num_heads
            ), f"num_heads should be number of heads for Q, but got {num_heads}"
            assert (
                num_key_value_heads
            ), f"num_key_value_heads should be number of key_value_heads for K and V, but got {num_key_value_heads}"
            assert (
                len(fuse_params) == 3
            ), f"fuse_params length is not equal 3, it should be Q K V list. but got length {len(fuse_params)}"
            num_query_groups = num_heads // num_key_value_heads
            q_list = split_fn(fuse_params[0], num_heads, axis=-1)
            k_list = split_fn(fuse_params[1], num_key_value_heads, axis=-1)
            v_list = split_fn(fuse_params[2], num_key_value_heads, axis=-1)

            qkv_pairs = []
            for i in range(num_key_value_heads):
                qkv_pairs += q_list[i * num_query_groups : (i + 1) * num_query_groups]
                qkv_pairs.append(k_list[i])
                qkv_pairs.append(v_list[i])
            return concat_fn(qkv_pairs, axis=-1)
        else:
            return concat_fn(fuse_params, axis=-1)

    return fn


def split_param_func():
    def fn(
        fused_param,
        split_nums=2,
        is_qkv=False,
        num_heads=None,
        num_key_value_heads=None,
    ):
        concat_fn = np.concatenate
        split_fn = np.split
        if isinstance(fused_param, paddle.Tensor):
            concat_fn = paddle.concat
            split_fn = paddle.split

        if is_qkv:
            assert (
                num_heads
            ), f"num_heads should be number of heads for Q, but got {num_heads}"
            assert (
                num_key_value_heads
            ), f"num_key_value_heads should be number of key_value_heads for K and V, but got {num_key_value_heads}"
            num_query_groups = num_heads // num_key_value_heads
            q_list, k_list, v_list = [], [], []
            split_heads = split_fn(
                fused_param, num_heads + 2 * num_key_value_heads, axis=-1
            )
            for i in range(num_key_value_heads):
                q_list += split_heads[
                    i * (num_query_groups + 2) : (i + 1) * (num_query_groups + 2) - 2
                ]
                k_list.append(split_heads[(i + 1) * (num_query_groups + 2) - 2])
                v_list.append(split_heads[(i + 1) * (num_query_groups + 2) - 1])
            return (
                concat_fn(q_list, axis=-1),
                concat_fn(k_list, axis=-1),
                concat_fn(v_list, axis=-1),
            )
        else:
            return split_fn(fused_param, split_nums, axis=-1)

    return fn


def split_or_fuse_func(is_fuse=True):
    return fuse_param_func() if is_fuse else split_param_func()
