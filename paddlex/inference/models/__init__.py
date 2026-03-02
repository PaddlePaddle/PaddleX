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


from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Type, Union

from pydantic import BaseModel, ValidationError

from paddlex.utils.deps import require_deps

from ...constants import MODEL_FILE_PREFIX
from ...utils import errors, logging
from ...utils.device import get_default_device, parse_device
from ..utils.hpi import HPIConfig
from ..utils.model_paths import resolve_paddle_engine_from_model_files
from ..utils.official_models import official_models
from ..utils.pp_option import PaddlePredictorOption
from .anomaly_detection import UadPredictor
from .base.predictor import (
    BasePredictor,
    FlexiblePredictor,
    GenAIClientPredictor,
    RunnerPredictor,
    TransformersPredictor,
)
from .base.predictor.transformers_predictor import TransformersEngineConfig
from .common.genai import SERVER_BACKENDS, GenAIConfig, need_local_model
from .common.runner.onnxruntime_runner import ONNXRuntimeRunnerConfig
from .common.runner.paddle_dynamic_runner import PaddleDynamicRunnerConfig
from .common.runner.paddle_static_runner import PaddleStaticRunnerConfig
from .doc_vlm import DocVLMPredictor
from .face_feature import FaceFeaturePredictor
from .formula_recognition import FormulaRecPredictor
from .image_classification import ClasPredictor
from .image_feature import ImageFeaturePredictor
from .image_multilabel_classification import MLClasPredictor
from .image_unwarping import WarpPredictor
from .instance_segmentation import InstanceSegPredictor
from .keypoint_detection import KptPredictor
from .layout_analysis import LayoutAnalysisPredictor
from .m_3d_bev_detection import BEVDet3DPredictor
from .multilingual_speech_recognition import WhisperPredictor
from .object_detection import DetPredictor
from .open_vocabulary_detection import OVDetPredictor
from .open_vocabulary_segmentation import OVSegPredictor
from .semantic_segmentation import SegPredictor
from .table_structure_recognition import TablePredictor
from .text_detection import TextDetPredictor
from .text_recognition import TextRecPredictor
from .text_to_pinyin import TextToPinyinPredictor
from .text_to_speech_acoustic import Fastspeech2Predictor
from .text_to_speech_vocoder import PwganPredictor
from .ts_anomaly_detection import TSAdPredictor
from .ts_classification import TSClsPredictor
from .ts_forecasting import TSFcPredictor
from .video_classification import VideoClasPredictor
from .video_detection import VideoDetPredictor


def _get_engine_base_predictor(engine: str) -> Type[BasePredictor]:
    if engine in {"paddle", "paddle_static", "paddle_dynamic", "hpi", "onnxruntime"}:
        return RunnerPredictor
    if engine == "flexible":
        return FlexiblePredictor
    if engine == "transformers":
        return TransformersPredictor
    if engine == "genai_client":
        return GenAIClientPredictor
    raise ValueError(f"Unsupported engine: {engine!r}.")


def _pick_predictor_cls(model_name: str, engine: str) -> Type[BasePredictor]:
    base_predictor = _get_engine_base_predictor(engine)
    try:
        return base_predictor.get(model_name)
    except errors.ClassNotFoundException as e:
        raise NotImplementedError(
            f"Model {model_name!r} has no predictor registered for engine {engine!r}."
        ) from e


def _is_flexible_only_model(model_name: str) -> bool:
    """True if model is registered with FlexiblePredictor but not RunnerPredictor."""
    try:
        RunnerPredictor.get(model_name)
        return False
    except errors.ClassNotFoundException:
        pass
    try:
        FlexiblePredictor.get(model_name)
        return True
    except errors.ClassNotFoundException:
        return False


def _pp_option_to_engine_config(pp_option: PaddlePredictorOption) -> Dict[str, Any]:
    """Convert PaddlePredictorOption to PaddleStaticRunnerConfig dict (backward compat)."""
    static_fields = set(PaddleStaticRunnerConfig.model_fields)
    cfg = {}
    for k in static_fields:
        if hasattr(pp_option, k):
            v = getattr(pp_option, k)
            if v is not None:
                cfg[k] = v
    return cfg


def _engine_config_to_dict(cfg: Any) -> Dict[str, Any]:
    """Convert Pydantic model or PaddlePredictorOption to dict."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if isinstance(cfg, PaddlePredictorOption):
        return _pp_option_to_engine_config(cfg)
    if hasattr(cfg, "model_dump"):
        dump_kw: Dict[str, Any] = {"exclude_none": True, "by_alias": True}
        return cfg.model_dump(**dump_kw)
    raise TypeError(
        f"`engine_config` must be dict, Pydantic model, or PaddlePredictorOption, "
        f"but got {type(cfg).__name__}."
    )


def normalize_engine_config(
    engine: str,
    cfg: Optional[Union[Dict[str, Any], BaseModel, PaddlePredictorOption]],
    *,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse, validate and normalize engine-specific config to a canonical dict."""
    cfg = cfg or {}

    raw = _engine_config_to_dict(cfg)

    # paddle_static
    if engine == "paddle_static":
        if device:
            device_type, device_ids = parse_device(device)
            raw["device_type"] = device_type
            raw["device_id"] = device_ids[0] if device_ids is not None else None
        try:
            return PaddleStaticRunnerConfig.model_validate(raw).model_dump(
                exclude_none=True
            )
        except ValidationError as e:
            raise ValueError(f"Invalid paddle_static engine_config: {e}") from e

    # paddle_dynamic
    if engine == "paddle_dynamic":
        if device:
            device_type, device_ids = parse_device(device)
            raw["device_type"] = device_type
            raw["device_id"] = device_ids[0] if device_ids is not None else None
        try:
            return PaddleDynamicRunnerConfig.model_validate(raw).model_dump(
                exclude_none=True
            )
        except ValidationError as e:
            raise ValueError(f"Invalid paddle_dynamic engine_config: {e}") from e

    # hpi
    if engine == "hpi":
        try:
            raw.setdefault("model_name", model_name or "")
            if device:
                device_type, device_ids = parse_device(device)
                raw["device_type"] = device_type
                raw["device_id"] = device_ids[0] if device_ids is not None else None
            elif "device_type" not in raw:
                raw["device_type"], _ = parse_device(get_default_device())
            validated = HPIConfig.model_validate(raw).model_dump(
                exclude_none=True, by_alias=True
            )
            return validated
        except ValidationError as e:
            raise ValueError(f"Invalid hpi engine_config: {e}") from e

    # flexible
    if engine == "flexible":
        if device:
            device_type, device_ids = parse_device(device)
            raw["device_type"] = device_type
            raw["device_id"] = device_ids[0] if device_ids is not None else None
        return raw

    # transformers
    if engine == "transformers":
        if device:
            device_type, device_ids = parse_device(device)
            if device_type == "gpu":
                raw["device_map"] = f"cuda:{device_ids[0]}" if device_ids else "cuda"
            elif device_type == "cpu":
                raw["device_map"] = "cpu"
            else:
                raw["device_map"] = (
                    f"{device_type}:{device_ids[0]}" if device_ids else device_type
                )
        try:
            return TransformersEngineConfig.model_validate(raw).model_dump(
                exclude_none=True
            )
        except ValidationError as e:
            raise ValueError(f"Invalid transformers engine_config: {e}") from e

    # onnxruntime
    if engine == "onnxruntime":
        if device:
            device_type, device_ids = parse_device(device)
            raw["device_type"] = device_type
            raw["device_id"] = device_ids[0] if device_ids is not None else None
        try:
            return ONNXRuntimeRunnerConfig.model_validate(raw).model_dump(
                exclude_none=True
            )
        except ValidationError as e:
            raise ValueError(f"Invalid onnxruntime engine_config: {e}") from e

    # genai_client
    if engine == "genai_client":
        try:
            validated = GenAIConfig.model_validate(raw).model_dump(exclude_none=True)
            if validated.get("backend") not in SERVER_BACKENDS:
                raise ValueError(
                    f"engine='genai_client' requires backend in {SERVER_BACKENDS!r}, "
                    f"got {validated.get('backend')!r}."
                )
            return validated
        except ValidationError as e:
            raise ValueError(f"Invalid genai_client engine_config: {e}") from e

    return raw


def create_predictor(
    model_name: str,
    *,
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    engine: Optional[str] = None,
    engine_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 1,
    pp_option=None,
    use_hpip: bool = False,
    hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    genai_config: Optional[Union[Dict[str, Any], GenAIConfig]] = None,
    **kwargs,
) -> BasePredictor:
    """Create a predictor for the given model and engine.

    Args:
        model_name: Model name.
        model_dir (Optional[str]): Path to model directory. Required for local engines
            when the model is not official. Ignored when a local model is not needed.
        device (Optional[str]): Device to run on (e.g. `'gpu'`, `'cpu'`). Used by local
            engines.
        engine (Optional[str]): Inference engine. One of `'paddle'` (resolved to
            `paddle_static` or `paddle_dynamic` per model), `'paddle_static'`,
            `'paddle_dynamic'`, `'hpi'`, `'flexible'`, `'transformers'`,
            `'onnxruntime'`, `'genai_client'`, or `None`.
            When `None`: if `genai_config.backend` is a server backend, engine
            becomes `'genai_client'`; else if `use_hpip=True` and model supports
            hpi, engine becomes `'hpi'`; else if model is flexible-only, engine
            becomes `'flexible'`; otherwise defaults to `'paddle'`.
        engine_config (Optional[Dict[str, Any]]): Engine-specific config.
        batch_size (int): Batch size for inference. Defaults to 1.
        pp_option (Optional[PaddlePredictorOption]): Paddle predictor options. Used when
            `engine='paddle_static'` and `engine_config` is not specified. Prefer
            `engine_config` for new code.
        use_hpip: When `engine` is `None`, if `True`, resolves to `engine='hpi'`.
            Ignored when `engine` is set.
        hpi_config (Optional[Union[Dict[str, Any], HPIConfig]]): HPI configuration.
            Used when `engine='hpi'` and `engine_config` is not specified. Prefer
            `engine_config` for new code.
        genai_config (Optional[Union[Dict[str, Any], GenAIConfig]]): GenAI configuration.
            Mainly used when `engine='genai_client'` and `engine_config` is not specified.
            Prefer `engine_config` for new code.

    Returns:
        A predictor instance.
    """
    if use_hpip and engine is not None:
        logging.warning(
            "`use_hpip` only takes effect when `engine` is None. Since engine=%r "
            "is explicitly set, use_hpip will be ignored. To use HPI, pass "
            "engine='hpi' instead.",
            engine,
        )

    if engine_config is not None:
        if pp_option is not None:
            logging.warning(
                "`pp_option` is ignored when `engine_config` is specified. "
                "Use `engine_config` for paddle_static configuration."
            )
        if hpi_config is not None:
            logging.warning(
                "`hpi_config` is ignored when `engine_config` is specified. "
                "Use `engine_config` for hpi configuration."
            )
        if genai_config is not None:
            logging.warning(
                "`genai_config` is ignored when `engine_config` is specified. "
                "Use `engine_config` for genai_client configuration."
            )

    if engine is None:
        engine = "paddle"
        if genai_config is not None:
            validated_genai = GenAIConfig.model_validate(genai_config)
            if not need_local_model(validated_genai):
                engine = "genai_client"
        elif use_hpip:
            engine = "hpi"
        elif _is_flexible_only_model(model_name):
            engine = "flexible"

    need_local = engine != "genai_client"
    model_dir_resolved: Optional[Path] = None
    if need_local:
        if model_dir is None:
            model_dir_resolved = Path(official_models[model_name])
        else:
            model_dir_resolved = Path(model_dir)
            if not model_dir_resolved.exists():
                raise FileNotFoundError(f"{model_dir} does not exist!")

    if engine == "paddle":
        resolved_engine = resolve_paddle_engine_from_model_files(
            model_dir_resolved,
            MODEL_FILE_PREFIX,
        )
        if resolved_engine is None:
            raise ValueError(
                f"Model {model_name!r} does not support the paddle engine. "
            )
        engine = resolved_engine

    if pp_option is not None and engine != "paddle_static":
        logging.warning(
            "`pp_option` only applies to engine='paddle_static'. "
            "For engine=%r, pp_option will be ignored.",
            engine,
        )
    if hpi_config is not None and engine != "hpi":
        logging.warning(
            "`hpi_config` only applies to engine='hpi'. "
            "For engine=%r, hpi_config will be ignored.",
            engine,
        )

    config_to_validate: Optional[Union[Dict[str, Any], HPIConfig, GenAIConfig]] = None
    if engine_config is None:
        if engine == "paddle_static":
            config_to_validate = pp_option
        elif engine == "hpi":
            config_to_validate = hpi_config
        elif engine == "genai_client":
            config_to_validate = genai_config
        else:
            config_to_validate = None
    else:
        config_to_validate = engine_config

    validated_engine_config = normalize_engine_config(
        engine,
        config_to_validate,
        model_name=model_name,
        device=device,
    )

    if need_local:
        config = BasePredictor.load_config(model_dir_resolved)
        if model_name != config["Global"]["model_name"]:
            raise ValueError(
                f"Model name mismatch，please input the correct model dir."
            )
    else:
        config = None

    predictor_cls = _pick_predictor_cls(model_name, engine)
    create_kwargs = dict(model_name=model_name, batch_size=batch_size)
    if engine:
        create_kwargs["engine"] = engine
    if validated_engine_config:
        create_kwargs["engine_config"] = validated_engine_config
    if need_local and engine in (
        "paddle_static",
        "paddle_dynamic",
        "hpi",
        "flexible",
        "onnxruntime",
    ):
        create_kwargs["model_dir"] = model_dir_resolved
        create_kwargs["model_config"] = config

    return predictor_cls(**create_kwargs, **kwargs)
