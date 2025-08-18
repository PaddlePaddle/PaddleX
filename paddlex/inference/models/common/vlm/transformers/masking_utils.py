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

# This file was adapted from https://github.com/huggingface/transformers/blob/05000aefe173bf7a10fa1d90e4c528585b45d3c7/src/transformers/masking_utils.py
# Original copyright notice below:
# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, List, Optional

import paddle


def and_masks(*mask_functions: List[Callable]) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(
            f"All inputs should be callable mask_functions: {mask_functions}"
        )

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = paddle.ones(q_idx.shape, dtype="bool")
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx)
        return result

    return and_mask


def or_masks(*mask_functions: List[Callable]) -> Callable:
    """Returns a mask function that is the union of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(
            f"All inputs should be callable mask_functions: {mask_functions}"
        )

    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_zeros((), dtype="bool")
        for mask in mask_functions:
            result = result | mask(batch_idx, head_idx, q_idx, kv_idx)
        return result

    return or_mask


def causal_mask_function(
    batch_idx: int, head_idx: int, q_idx: int, kv_idx: int
) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx


def prepare_padding_mask(
    attention_mask, kv_length: int, kv_offset: int, _slice: bool = True
):
    """
    From the 2D attention mask, prepare the correct padding mask to use by potentially padding it, and slicing
    according to the `kv_offset` if `_slice` is `True`.
    """
    local_padding_mask = attention_mask
    if attention_mask is not None:
        # Pad it if necesary
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = paddle.nn.functional.pad(
                attention_mask, (0, padding_length)
            )
        # For flex, we should not slice them, only use an offset
        if _slice:
            mask_indices = paddle.arange(kv_length)
            mask_indices += kv_offset
            local_padding_mask = local_padding_mask[:, mask_indices]
    return local_padding_mask


def _ignore_causal_mask_sdpa(
    padding_mask,
    query_length: int,
    kv_length: int,
    kv_offset: int,
    local_attention_size: Optional[int] = None,
) -> bool:
    """
    Detects whether the causal mask can be ignored in case PaddlePaddle's SDPA is used, rather relying on SDPA's `is_causal` argument.

    In case no token is masked in the 2D `padding_mask` argument, if `query_length == 1` or
    `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    if padding_mask is not None and padding_mask.shape[-1] > kv_length:
        mask_indices = paddle.arange(kv_length)
        mask_indices += kv_offset
        padding_mask = padding_mask[:, mask_indices]

    if (
        (query_length == 1 or (kv_length == query_length))
        and (local_attention_size is None or kv_length < local_attention_size)
        and (
            padding_mask is None
            or (
                padding_mask.all()
                if query_length == 1
                else padding_mask[:, :query_length].all()
            )
        )
    ):
        return True

    return False


def padding_mask_function(padding_mask: paddle.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D padding mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # Note that here the mask should ALWAYS be at least of the max `kv_index` size in the dimension 1. This is because
        # we cannot pad it here in the mask_function as we don't know the final size, and we cannot try/except, as it is not
        # vectorizable on accelerator devices
        return padding_mask[batch_idx, kv_idx]

    return inner_mask


def sdpa_mask(
    batch_size: int,
    cache_position,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask=None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
):
    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    padding_mask = prepare_padding_mask(
        attention_mask, kv_length, kv_offset, _slice=False
    )

    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(
        padding_mask, q_length, kv_length, kv_offset, local_size
    ):
        return None

    kv_arange = paddle.arange(kv_length)
    kv_arange += kv_offset

    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    batch_arange = paddle.arange(batch_size)
    batch_idx = batch_arange.reshape([-1, 1, 1, 1])
    head_arange = paddle.arange(1)
    head_idx = head_arange.reshape([1, -1, 1, 1])
    q_idx = cache_position.reshape([1, 1, -1, 1])
    kv_idx = kv_arange.reshape([1, 1, 1, -1])
    causal_mask = mask_function(batch_idx, head_idx, q_idx, kv_idx)

    return causal_mask


def eager_mask(
    batch_size: int,
    cache_position,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask=None,
    dtype=paddle.float32,
    **kwargs,
):
    # The masks for eager attention are simply boolean mask from sdpa, casted to 0 and -inf
    _ = kwargs.pop("allow_is_causal_skip", None)
    mask = sdpa_mask(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
        **kwargs,
    )
    min_dtype = paddle.finfo(dtype).min
    # we need 0s where the tokens should be taken into account, and -inf otherwise (mask is already of boolean type)
    mask = paddle.where(
        mask,
        paddle.to_tensor(0.0, dtype=dtype),
        paddle.to_tensor(min_dtype, dtype=dtype),
    )
    return mask


ALL_MASK_ATTENTION_FUNCTIONS = {
    "sdpa": sdpa_mask,
    "eager": eager_mask,
}


def _preprocess_mask_arguments(
    attn_implementation,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    layer_idx,
):
    """
    Perform some common pre-processing of the mask arguments we get from the modeling code. Mostly determine the
    key-value length and offsets, and if we should early exit or not.
    """
    # If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
    if paddle.is_tensor(attention_mask) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None

    if attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS:
        return True, None, None, None

    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.astype("bool")

    # If using a cache, it can give all informations about mask sizes based on seen tokens
    if past_key_values is not None:
        kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
    # Otherwise, the sizes are simply the input sizes
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0

    return False, attention_mask, kv_length, kv_offset


def create_causal_mask(
    attn_implementation,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids=None,
    or_mask_function=None,
    and_mask_function=None,
):
    # `position_ids` is currently not used
    # If we have an HybridCache structure, here we want to create the mask for the full layers
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    early_exit, attention_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        attn_implementation,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        layer_idx,
    )
    if early_exit:
        return attention_mask

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[attn_implementation]

    allow_is_causal_skip = (
        not past_key_values.is_compileable if past_key_values is not None else True
    )

    # Allow slight deviations from causal mask
    if or_mask_function is not None:
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
    )
    return causal_mask
