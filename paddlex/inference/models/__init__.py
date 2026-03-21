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


from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from ...utils import errors, logging
from ..utils.official_models import official_models
from .anomaly_detection import UadPredictor
from .bindings import Binding, UnknownModelError, get_binding, get_supported_engines
from .common.genai import GenAIConfig, need_local_model, uses_server_backend
from .doc_vlm import DocVLMPredictor
from .engines import InferenceEngine
from .face_feature import FaceFeaturePredictor
from .formula_recognition import FormulaRecPredictor
from .hpi import HPIConfig
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
from .predictors import BasePredictor, LocalModelPredictor, RunnerPredictor
from .runners.paddle_static.config import PaddlePredictorOption
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
from .utils.model_paths import resolve_paddle_engine_from_model_files
from .video_classification import VideoClasPredictor
from .video_detection import VideoDetPredictor


def _get_supported_engines(model_name: str) -> Tuple[str, ...]:
    try:
        return get_supported_engines(model_name)
    except UnknownModelError as e:
        raise ValueError(
            f"No engine bindings registered for model {model_name!r}."
        ) from e


from .utils.model_resolver import resolve_model_name


def _resolve_model_dir(
    model_name: str,
    model_dir: Optional[str],
    *,
    model_formats=None,
) -> Path:
    if model_dir is None:
        return Path(
            official_models.get_model_path(
                model_name,
                model_formats=model_formats,
            )
        )
    resolved = Path(model_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"{model_dir} does not exist!")
    return resolved


def _resolve_default_paddle_engine(model_name: str) -> str:
    supported = _get_supported_engines(model_name)
    for engine in ("paddle_static", "paddle_dynamic"):
        if engine in supported:
            return engine
    raise ValueError(
        f"Model {model_name!r} does not support engine 'paddle'. "
        f"Supported engines: {list(supported)!r}."
    )


def _resolve_requested_engine(
    model_name: str,
    engine: str,
    model_dir: Optional[str],
) -> tuple[str, Optional[Path]]:
    if engine != "paddle":
        return engine, None

    if model_dir is None:
        return _resolve_default_paddle_engine(model_name), None

    model_dir_resolved = _resolve_model_dir(model_name, model_dir)
    resolved_engine = resolve_paddle_engine_from_model_files(model_dir_resolved)
    if resolved_engine is None:
        raise ValueError(f"No Paddle model files were found in {model_dir!r}.")
    return resolved_engine, model_dir_resolved


def _is_flexible_only_model(model_name: str) -> bool:
    try:
        return _get_supported_engines(model_name) == ("flexible",)
    except ValueError:
        return False


@lru_cache(None)
def _get_inference_engine(engine: str) -> InferenceEngine:
    try:
        return InferenceEngine.get(engine)()
    except errors.ClassNotFoundException as e:
        raise ValueError(f"Unsupported engine: {engine!r}.") from e


def normalize_engine_config(
    engine: str,
    cfg: Optional[Union[Dict[str, Any], PaddlePredictorOption, Any]],
    *,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse, validate and normalize engine-specific config to a canonical dict."""
    return _get_inference_engine(engine).normalize_config(
        cfg,
        model_name=model_name,
        device=device,
    )


def _warn_if_legacy_configs_are_overridden(
    *,
    engine_config,
    pp_option,
    hpi_config,
    genai_config,
) -> None:
    if engine_config is None:
        return
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


def _resolve_effective_engine(
    model_name: str,
    engine: Optional[str],
    use_hpip: bool,
    genai_config: Optional[Union[Dict[str, Any], GenAIConfig]],
) -> str:
    if engine is not None:
        return engine
    if uses_server_backend(genai_config):
        return "genai_client"
    if use_hpip:
        return "hpi"
    if _is_flexible_only_model(model_name):
        return "flexible"
    return "paddle"


def _select_engine_config_source(
    *,
    engine: str,
    engine_config,
    pp_option,
    hpi_config,
    genai_config,
) -> Optional[Union[Dict[str, Any], HPIConfig, GenAIConfig]]:
    if engine_config is not None:
        return engine_config
    if engine == "paddle_static":
        return pp_option
    if engine == "hpi":
        return hpi_config
    if engine == "genai_client":
        return genai_config
    return None


def _resolve_local_model_dir(
    *,
    inference_engine: InferenceEngine,
    model_name: str,
    model_dir: Optional[str],
    model_dir_resolved: Optional[Path],
) -> Optional[Path]:
    if model_dir_resolved is not None or not inference_engine.needs_local_model:
        return model_dir_resolved
    return _resolve_model_dir(
        model_name,
        model_dir,
        model_formats=inference_engine.get_supported_model_formats(),
    )


def _load_local_model_config(
    *,
    needs_local_model: bool,
    model_name: str,
    model_dir: Optional[Path],
) -> Optional[Dict[str, Any]]:
    if not needs_local_model:
        return None
    config = BasePredictor.load_config(model_dir)
    if model_name != config["Global"]["model_name"]:
        raise ValueError(f"Model name mismatch，please input the correct model dir.")
    return config


def _build_predictor_runner(
    *,
    predictor_cls,
    inference_engine: InferenceEngine,
    binding: Binding,
    model_name: str,
    model_dir: Optional[Path],
    model_config: Optional[Dict[str, Any]],
    engine_config: Dict[str, Any],
):
    if not issubclass(predictor_cls, RunnerPredictor):
        return None
    build_runner = getattr(inference_engine, "build_runner", None)
    if not callable(build_runner):
        raise RuntimeError(
            f"InferenceEngine {type(inference_engine).__name__!r} does not support runner construction."
        )
    return build_runner(
        model_name=model_name,
        model_dir=model_dir,
        model_config=model_config,
        engine_config=engine_config,
        binding=binding,
    )


def _build_predictor_kwargs(
    *,
    binding: Binding,
    inference_engine: InferenceEngine,
    model_name: str,
    batch_size: int,
    normalized_engine_config: Dict[str, Any],
    predictor_engine_config: Dict[str, Any],
    model_dir: Optional[Path],
    model_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    predictor_cls = binding.predictor
    create_kwargs = {
        "model_name": model_name,
        "batch_size": batch_size,
    }
    if predictor_engine_config:
        create_kwargs["engine_config"] = predictor_engine_config
    runner = _build_predictor_runner(
        predictor_cls=predictor_cls,
        inference_engine=inference_engine,
        binding=binding,
        model_name=model_name,
        model_dir=model_dir,
        model_config=model_config,
        engine_config=normalized_engine_config,
    )
    if runner is not None:
        create_kwargs["runner"] = runner
    if issubclass(predictor_cls, LocalModelPredictor):
        create_kwargs["model_dir"] = model_dir
        create_kwargs["model_config"] = model_config
    return create_kwargs


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
            `paddle_static` or `paddle_dynamic` from local model files when
            `model_dir` is provided; otherwise resolved from predictor support,
            preferring `paddle_static`), `'paddle_static'`,
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

    _warn_if_legacy_configs_are_overridden(
        engine_config=engine_config,
        pp_option=pp_option,
        hpi_config=hpi_config,
        genai_config=genai_config,
    )

    model_name, model_dir_resolved, resolved_config = resolve_model_name(
        model_name=model_name,
        model_dir=model_dir,
    )

    engine = _resolve_effective_engine(model_name, engine, use_hpip, genai_config)

    engine, model_dir_resolved = _resolve_requested_engine(
        model_name, engine, model_dir
    )

    requested_engine = _get_inference_engine(engine)

    need_local = requested_engine.needs_local_model
    model_dir_resolved = _resolve_local_model_dir(
        inference_engine=requested_engine,
        model_name=model_name,
        model_dir=model_dir,
        model_dir_resolved=model_dir_resolved,
    )

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

    config_to_validate = _select_engine_config_source(
        engine=engine,
        engine_config=engine_config,
        pp_option=pp_option,
        hpi_config=hpi_config,
        genai_config=genai_config,
    )
    normalized_engine_config = normalize_engine_config(
        engine,
        config_to_validate,
        model_name=model_name,
        device=device,
    )
    if need_local:
        requested_engine.ensure_model_files(model_dir_resolved)
    requested_engine.ensure_environment(
        device=device,
        engine_config=normalized_engine_config,
    )
    if resolved_config and need_local:
        config = resolved_config
    else:
        config = _load_local_model_config(
            needs_local_model=need_local,
            model_name=model_name,
            model_dir=model_dir_resolved,
        )

    predictor_binding = get_binding(model_name, engine)
    predictor_engine_config = requested_engine.to_predictor_config(
        normalized_engine_config
    )
    create_kwargs = _build_predictor_kwargs(
        binding=predictor_binding,
        model_name=model_name,
        batch_size=batch_size,
        normalized_engine_config=normalized_engine_config,
        predictor_engine_config=predictor_engine_config,
        model_dir=model_dir_resolved,
        model_config=config,
        inference_engine=requested_engine,
    )

    return predictor_binding.predictor(**create_kwargs, **kwargs)
