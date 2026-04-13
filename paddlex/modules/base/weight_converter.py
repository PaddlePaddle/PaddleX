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

"""Weight converter: pdparams -> safetensors.

Conversion flow:
  1. paddle.load() -> state dict with OLD PaddleOCR/PaddleDetection key names
  2. Rename BatchNorm keys: _mean -> running_mean, _variance -> running_var
  3. Apply per-architecture regex key mappings (old keys -> HF keys)
  4. Transpose linear weight keys (Paddle [in, out] -> HF [out, in])
  5. Save as safetensors via safetensors.numpy.save_file()
"""

import json
import os
from pathlib import Path

from ...utils import logging
from ...utils.config import AttrDict
from .utils.pdparams2safetensors import (
    CHART2TABLE_ADDED_TOKENS,
    CHART2TABLE_GENERATION_CONFIG,
    CHART2TABLE_SPECIAL_TOKENS_MAP,
    CHART2TABLE_TOKENIZER_CONFIG,
    MOBILE_DET_DROP_PREFIXES,
    PP_CHART2TABLE_DROP_PREFIXES,
    PP_CHART2TABLE_MAPPING,
    PP_DOCLAYOUTV2_DROP_PREFIXES,
    PP_DOCLAYOUTV2_MAPPING,
    PPLCNET_MAPPING,
    PPOCRV5_MOBILE_DET_MAPPING,
    PPOCRV5_MOBILE_REC_MAPPING,
    PPOCRV5_SERVER_DET_MAPPING,
    PPOCRV5_SERVER_REC_MAPPING,
    PREPROCESSOR_CONFIGS,
    REC_DROP_PREFIXES,
    RTDETR_MAPPING,
    SERVER_DET_DROP_PREFIXES,
    SERVER_REC_DROP_PREFIXES,
    SLANEXT_DROP_PREFIXES,
    SLANEXT_MAPPING,
    UVDOC_DROP_PREFIXES,
    UVDOC_MAPPING,
    apply_key_mapping,
    build_inference_meta,
    load_character_dict,
    rename_bn_keys,
)
from .utils.pdparams2safetensors.model_config import MODEL_CONFIGS


def build_weight_converter(config: AttrDict) -> "WeightConverter":
    """Build a weight converter from PaddleX config."""
    return WeightConverter(config)


# Model registry: model_name -> (key_mapping, drop_key_prefixes)
_MODEL_REGISTRY = {
    "PP-LCNet_x1_0_doc_ori": (PPLCNET_MAPPING, []),
    "PP-LCNet_x1_0_table_cls": (PPLCNET_MAPPING, []),
    "PP-LCNet_x0_25_textline_ori": (PPLCNET_MAPPING, []),
    "PP-LCNet_x1_0_textline_ori": (PPLCNET_MAPPING, []),
    "PP-OCRv5_mobile_det": (PPOCRV5_MOBILE_DET_MAPPING, MOBILE_DET_DROP_PREFIXES),
    "PP-OCRv5_server_det": (PPOCRV5_SERVER_DET_MAPPING, SERVER_DET_DROP_PREFIXES),
    "PP-OCRv5_mobile_rec": (PPOCRV5_MOBILE_REC_MAPPING, REC_DROP_PREFIXES),
    "PP-OCRv5_server_rec": (PPOCRV5_SERVER_REC_MAPPING, SERVER_REC_DROP_PREFIXES),
    "SLANeXt_wired": (SLANEXT_MAPPING, SLANEXT_DROP_PREFIXES),
    "SLANeXt_wireless": (SLANEXT_MAPPING, SLANEXT_DROP_PREFIXES),
    "PP-DocLayoutV2": (PP_DOCLAYOUTV2_MAPPING, PP_DOCLAYOUTV2_DROP_PREFIXES),
    "PP-DocLayoutV3": (RTDETR_MAPPING, []),
    "RT-DETR-L_wired_table_cell_det": (RTDETR_MAPPING, []),
    "RT-DETR-L_wireless_table_cell_det": (RTDETR_MAPPING, []),
    "PP-DocLayout_plus-L": (RTDETR_MAPPING, []),
    "PP-DocBlockLayout": (RTDETR_MAPPING, []),
    "UVDoc": (UVDOC_MAPPING, UVDOC_DROP_PREFIXES),
    "PP-Chart2Table": (PP_CHART2TABLE_MAPPING, PP_CHART2TABLE_DROP_PREFIXES),
}


_TRANSPOSE_SUBSTRINGS = [
    "fc",
    "channelwise",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
    "linear_1",
    "linear_2",
    "linear1",
    "linear2",
    "attn.qkv",
    "mlp.lin1",
    "mlp.lin2",
    "attn.proj",
    "mixer.qkv",
    "mixer.proj",
    "self_attn.qkv",
    "self_attn.projection",
    "mapper_crp",
    "mapper_sca",
    ".mapper.",
    "txt_mapper",
    "txt_pooled_mapper",
    "clip_img_mapper",
    "kv_mapper",
    "clip_mapper",
    "mm_projector_vary",
    "score_head",
    "enc_score_head",
    "dec_score_head",
    "bbox_head",
    "enc_bbox_head",
    "dec_bbox_head",
    "mask_query_head",
    "enc_output",
    "query_pos_head",
    "dec_global_pointer",
    "dec_order_head",
    "attention_weights",
    "sampling_offsets",
    "value_proj",
    "output_proj",
    "head.head",
    "ctc_head",
    "conv_reduce_channel",
    "structure_attention_cell.score",
    "structure_attention_cell.i2h",
    "structure_attention_cell.h2h",
    "structure_generator.0.",
    "structure_generator.1.",
    # Reading order (PP-DocLayoutV2) linear layers
    "spatial_proj",
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "intermediate.dense",
    ".output.dense",
    "relative_head",
    "label_features_projection",
    "pos_proj",
]


def _should_transpose(key):
    """Check if a 2D weight tensor should be transposed (linear layer)."""
    return any(sub in key for sub in _TRANSPOSE_SUBSTRINGS)


def _preprocess_tensors(state_dict):
    """Preprocess Paddle tensors for safetensors output.

    Converts to numpy, transposes linear weight tensors from Paddle [in, out]
    to HF [out, in] format, reshapes channelwise parameters, and splits
    fused in_proj weights into separate q/k/v.

    Applied on OLD key names before regex key mapping.
    Returns dict of {key: numpy_array}.
    """
    import numpy as np

    result = {}
    for key, tensor in state_dict.items():
        if hasattr(tensor, "numpy"):
            import paddle

            if tensor.dtype in (paddle.bfloat16, paddle.float16):
                tensor = tensor.astype(paddle.float32)
            np_weight = tensor.cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            np_weight = tensor
        else:
            np_weight = np.array(tensor)

        if "channelwise" in key and ("gamma" in key or "beta" in key):
            np_weight = np_weight.reshape((-1, 1, 1, 1))

        if "in_proj_weight" in key or "in_proj_bias" in key:
            if "weight" in key:
                np_weight = np_weight.transpose()
                split_size = np_weight.shape[0] // 3
                result[key.replace("in_proj_weight", "q_proj.weight")] = np_weight[
                    :split_size
                ]
                result[key.replace("in_proj_weight", "k_proj.weight")] = np_weight[
                    split_size : 2 * split_size
                ]
                result[key.replace("in_proj_weight", "v_proj.weight")] = np_weight[
                    2 * split_size :
                ]
            elif "bias" in key:
                split_size = np_weight.shape[0] // 3
                result[key.replace("in_proj_bias", "q_proj.bias")] = np_weight[
                    :split_size
                ]
                result[key.replace("in_proj_bias", "k_proj.bias")] = np_weight[
                    split_size : 2 * split_size
                ]
                result[key.replace("in_proj_bias", "v_proj.bias")] = np_weight[
                    2 * split_size :
                ]
            continue

        if np_weight.ndim == 2 and "bias" not in key and _should_transpose(key):
            np_weight = np_weight.transpose()

        result[key] = np_weight

    return result


def _resolve_input_path(input_path):
    """Resolve input_path to a concrete .pdparams file.

    Accepts a direct .pdparams file path or a directory containing one.
    Directory resolution checks: model_state.pdparams, inference.pdparams,
    best_model.pdparams, best_accuracy.pdparams, or the single .pdparams file.
    """
    p = Path(input_path)

    if p.is_file():
        if not p.name.endswith(".pdparams"):
            raise ValueError(f"input_path file must end with .pdparams, got: {p}")
        return str(p)

    if p.is_dir():
        candidates = [
            "model_state.pdparams",
            "inference.pdparams",
            "best_model.pdparams",
            "best_accuracy.pdparams",
        ]
        for name in candidates:
            candidate = p / name
            if candidate.exists():
                return str(candidate)

        pdparams_files = list(p.glob("*.pdparams"))
        if len(pdparams_files) == 1:
            return str(pdparams_files[0])
        elif len(pdparams_files) > 1:
            names = [f.name for f in pdparams_files]
            raise ValueError(
                f"Multiple .pdparams files found in {p}: {names}. "
                "Please specify the exact file path."
            )
        else:
            raise FileNotFoundError(f"No .pdparams files found in directory: {p}")

    raise FileNotFoundError(f"input_path does not exist: {p}")


# WeightConverter
class WeightConverter:
    """Converts Paddle .pdparams weights to safetensors format."""

    def __init__(self, config):
        self.model_name = config.Global.model

        convert_config = config.Pdparams2safetensors
        self._get = (
            convert_config.get
            if isinstance(convert_config, dict)
            else lambda k, d=None: getattr(convert_config, k, d)
        )

        self.input_path = self._get("input_path")
        self.output_dir = self._get("output_dir")
        if self.input_path is None:
            raise ValueError(
                "Pdparams2safetensors.input_path is required. "
                "Specify a .pdparams file or a directory containing one."
            )
        if self.output_dir is None:
            raise ValueError("Pdparams2safetensors.output_dir is required.")

        if self.model_name not in _MODEL_REGISTRY:
            supported = ", ".join(sorted(_MODEL_REGISTRY.keys()))
            raise ValueError(
                f"Model '{self.model_name}' is not supported for "
                f"pdparams2safetensors conversion. Supported models: {supported}"
            )

        self._input_is_dir = Path(self.input_path).is_dir()
        self._user_configs = self._load_user_configs()

    def _load_user_configs(self):
        """Load user-provided config files from input directory."""
        if not self._input_is_dir:
            logging.info(
                "Input is a single pdparams file. "
                "Using official config files for %s.",
                self.model_name,
            )
            return {}

        import yaml

        input_dir = Path(self.input_path)
        user_configs = {}

        for fname, loader in [
            ("config.json", lambda f: json.load(f)),
            ("preprocessor_config.json", lambda f: json.load(f)),
            ("inference.yml", lambda f: yaml.safe_load(f)),
        ]:
            fpath = input_dir / fname
            if fpath.exists():
                with open(fpath, encoding="utf-8") as f:
                    user_configs[fname] = loader(f)
                logging.info(f"Loaded user config: {fpath}")
            else:
                logging.warning(
                    f"{fname} not found in {input_dir}. "
                    f"Using official default for {self.model_name}."
                )

        return user_configs

    def convert(self):
        """Execute the pdparams -> safetensors conversion."""
        from ...inference.models.doc_vlm.constants import PP_CHART2TABLE_MODELS

        key_mapping, drop_prefixes = _MODEL_REGISTRY[self.model_name]

        numpy_sd = self._convert_weights(key_mapping, drop_prefixes)

        os.makedirs(self.output_dir, exist_ok=True)
        self._save_safetensors(numpy_sd)
        self._save_model_config()
        self._save_preprocessor_config()
        self._save_inference_yml()

        if self.model_name in PP_CHART2TABLE_MODELS:
            self._save_llm_config()

        logging.info(f"Conversion complete. Output saved to: {self.output_dir}")

    def _convert_weights(self, key_mapping, drop_prefixes):
        """Load pdparams and convert to numpy state dict with HF key names."""
        import paddle

        resolved_path = _resolve_input_path(self.input_path)
        logging.info(f"Loading weights from: {resolved_path}")
        state_dict = paddle.load(resolved_path)

        if drop_prefixes:
            dropped = [
                k for k in state_dict if any(k.startswith(p) for p in drop_prefixes)
            ]
            for k in dropped:
                del state_dict[k]
            if dropped:
                logging.info(f"Dropped {len(dropped)} keys not needed for inference")

        state_dict = rename_bn_keys(state_dict)
        numpy_sd = _preprocess_tensors(state_dict)

        if key_mapping:
            numpy_sd = apply_key_mapping(numpy_sd, key_mapping)
        else:
            logging.warning(
                f"No key mapping defined for {self.model_name}. "
                "Keys will be saved as-is from pdparams."
            )

        self._postprocess_weights(numpy_sd)
        return numpy_sd

    def _postprocess_weights(self, numpy_sd):
        """Apply model-specific post-processing to converted weights."""
        import numpy as np

        config = MODEL_CONFIGS.get(self.model_name, {})

        embed_key = "model.denoising_class_embed.weight"
        if embed_key in numpy_sd and config.get("model_type") == "rt_detr":
            id2label = config.get("id2label", {})
            expected_size = len(id2label) + 1
            current_size = numpy_sd[embed_key].shape[0]
            if current_size < expected_size:
                pad = np.zeros(
                    (expected_size - current_size, numpy_sd[embed_key].shape[1]),
                    dtype=numpy_sd[embed_key].dtype,
                )
                numpy_sd[embed_key] = np.concatenate(
                    [numpy_sd[embed_key], pad],
                    axis=0,
                )
                logging.info(
                    f"Padded {embed_key} from {current_size} to "
                    f"{expected_size} (added background class)"
                )

        # PP-Chart2Table: lm_head.weight is a tied embedding weight [vocab, hidden],
        # NOT a linear weight. _preprocess_tensors wrongly transposed it because
        # "lm_head" is in _TRANSPOSE_SUBSTRINGS. Undo the transpose.
        lm_head_key = "lm_head.weight"
        if lm_head_key in numpy_sd and config.get("model_type") == "pp_chart2table":
            vocab_size = config.get("vocab_size", 151860)
            if numpy_sd[lm_head_key].shape[0] != vocab_size:
                numpy_sd[lm_head_key] = numpy_sd[lm_head_key].transpose()
                logging.info(
                    f"Reverted transpose on {lm_head_key} "
                    f"(tied embedding, not linear)"
                )

        if config.get("model_type") == "rt_detr":
            nbt_keys = [
                k.replace(".running_mean", ".num_batches_tracked")
                for k in numpy_sd
                if k.endswith(".running_mean") and not k.startswith("model.backbone.")
            ]
            added = 0
            for k in nbt_keys:
                if k not in numpy_sd:
                    numpy_sd[k] = np.int64(0)
                    added += 1
            if added:
                logging.info(f"Added {added} num_batches_tracked keys")

    def _save_safetensors(self, numpy_sd):
        """Save numpy state dict as model.safetensors."""
        from safetensors.numpy import save_file

        out_path = os.path.join(self.output_dir, "model.safetensors")
        save_file(numpy_sd, out_path)
        logging.info(f"Saved model.safetensors to: {out_path}")

    def _save_model_config(self):
        """Save config.json — user-provided or official default."""
        data = self._user_configs.get(
            "config.json",
            MODEL_CONFIGS.get(self.model_name, {}),
        )
        out_path = os.path.join(self.output_dir, "config.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved config.json to: {out_path}")

    def _save_preprocessor_config(self):
        """Save preprocessor_config.json — user-provided or official default."""
        if "preprocessor_config.json" in self._user_configs:
            data = self._user_configs["preprocessor_config.json"]
        else:
            data = dict(PREPROCESSOR_CONFIGS.get(self.model_name, {}))
            if self.model_name in ("PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec"):
                data["character_list"] = ["blank"] + load_character_dict() + [" "]

        out_path = os.path.join(self.output_dir, "preprocessor_config.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved preprocessor_config.json to: {out_path}")

    def _save_inference_yml(self):
        """Save inference.yml — user-provided or official default."""
        import yaml

        if "inference.yml" in self._user_configs:
            data = self._user_configs["inference.yml"]
        else:
            data = {"Global": {"model_name": self.model_name}}
            data.update(build_inference_meta(self.model_name))
            if self.model_name in ("PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec"):
                data.setdefault("PostProcess", {})[
                    "character_dict"
                ] = load_character_dict()

        out_path = os.path.join(self.output_dir, "inference.yml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
            )
        logging.info(f"Saved inference.yml to: {out_path}")

    def _save_llm_config(self):
        """Save tokenizer and generation config for Chart2Table models.

        Outputs: qwen.tiktoken, added_tokens.json, generation_config.json,
        special_tokens_map.json, tokenizer_config.json.
        """
        import shutil

        # qwen.tiktoken: binary file, must be copied (not hardcoded)
        tiktoken_src = self._resolve_tiktoken_source()
        tiktoken_dst = os.path.join(self.output_dir, "qwen.tiktoken")
        shutil.copy2(tiktoken_src, tiktoken_dst)
        logging.info(f"Copied qwen.tiktoken to: {tiktoken_dst}")

        # JSON tokenizer assets: user-provided or hardcoded defaults
        _TOKENIZER_DEFAULTS = {
            "added_tokens.json": CHART2TABLE_ADDED_TOKENS,
            "generation_config.json": CHART2TABLE_GENERATION_CONFIG,
            "special_tokens_map.json": CHART2TABLE_SPECIAL_TOKENS_MAP,
            "tokenizer_config.json": CHART2TABLE_TOKENIZER_CONFIG,
        }
        for fname, default_data in _TOKENIZER_DEFAULTS.items():
            if self._input_is_dir:
                src = Path(self.input_path) / fname
                if src.exists():
                    data = json.load(open(src, encoding="utf-8"))
                    logging.info(f"Loaded user tokenizer config: {src}")
                else:
                    data = default_data
                    logging.warning(
                        f"{fname} not found in {self.input_path}. "
                        f"Using default for {self.model_name}."
                    )
            else:
                data = default_data

            out_path = os.path.join(self.output_dir, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved {fname} to: {out_path}")

    def _resolve_tiktoken_source(self):
        """Find qwen.tiktoken for Chart2Table conversion."""
        if self._input_is_dir:
            src = Path(self.input_path) / "qwen.tiktoken"
            if src.exists():
                return str(src)
            logging.warning(
                f"qwen.tiktoken not found in {self.input_path}. "
                "Falling back to official model cache."
            )

        # Try official HF cache
        from ...utils.cache import CACHE_DIR

        cache_path = (
            Path(CACHE_DIR)
            / "official_models"
            / f"{self.model_name}_safetensors"
            / "qwen.tiktoken"
        )
        if cache_path.exists():
            return str(cache_path)

        raise FileNotFoundError(
            f"qwen.tiktoken not found. For single-file input, ensure the official "
            f"model is cached at {cache_path} (run inference once to download). "
            f"For directory input, include qwen.tiktoken in the input directory."
        )
