# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from .fusion import (
    fuse_v6_medium_det_state_dict,
    fuse_v6_rec_state_dict,
    fuse_v6_small_det_state_dict,
)
from .inference_meta import (
    CHART2TABLE_ADDED_TOKENS,
    CHART2TABLE_GENERATION_CONFIG,
    CHART2TABLE_SPECIAL_TOKENS_MAP,
    CHART2TABLE_TOKENIZER_CONFIG,
    PREPROCESSOR_CONFIGS,
    UNIMERNET_GENERATION_CONFIG,
    UNIMERNET_PROCESSOR_CONFIG,
    UNIMERNET_TOKENIZER_CONFIG,
    build_inference_meta,
    load_character_dict,
)
from .mapping import (
    MOBILE_DET_DROP_PREFIXES,
    PP_CHART2TABLE_DROP_PREFIXES,
    PP_CHART2TABLE_MAPPING,
    PP_DOCLAYOUTV2_DROP_PREFIXES,
    PP_DOCLAYOUTV2_MAPPING,
    PP_FORMULANET_MAPPING,
    PPLCNET_MAPPING,
    PPOCRV5_MOBILE_DET_MAPPING,
    PPOCRV5_MOBILE_REC_MAPPING,
    PPOCRV5_SERVER_DET_MAPPING,
    PPOCRV5_SERVER_REC_MAPPING,
    PPOCRV6_DET_DROP_PREFIXES,
    PPOCRV6_MEDIUM_DET_MAPPING,
    PPOCRV6_REC_DROP_PREFIXES,
    PPOCRV6_SMALL_DET_MAPPING,
    PPOCRV6_SMALL_REC_MAPPING,
    PPOCRV6_TINY_REC_MAPPING,
    REC_DROP_PREFIXES,
    RTDETR_MAPPING,
    SERVER_DET_DROP_PREFIXES,
    SERVER_REC_DROP_PREFIXES,
    SLANET_DROP_PREFIXES,
    SLANET_MAPPING,
    SLANEXT_DROP_PREFIXES,
    SLANEXT_MAPPING,
    UVDOC_DROP_PREFIXES,
    UVDOC_MAPPING,
    apply_key_mapping,
    rename_bn_keys,
)
from .model_config import MODEL_CONFIGS
