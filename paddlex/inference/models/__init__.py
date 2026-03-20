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
from typing import Any, Dict, Optional, Tuple, Type, Union

from ...utils import errors, logging
from ..utils.hpi import HPIConfig
from ..utils.model_paths import resolve_paddle_engine_from_model_files
from ..utils.official_models import official_models
from ..utils.pp_option import PaddlePredictorOption
from .anomaly_detection import UadPredictor
from .common.genai import GenAIConfig, need_local_model, uses_server_backend
from .doc_vlm import DocVLMPredictor
from .engines import EngineSpec
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
from .predictors import BasePredictor
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


def _pick_predictor_cls(model_name: str, engine: str) -> Type[BasePredictor]:
    return _get_engine_spec_instance(engine).get_predictor_cls(model_name)


@lru_cache(None)
def _get_supported_engines(model_name: str) -> Tuple[str, ...]:
    supported = []
    for engine_name in EngineSpec.all():
        try:
            engine_spec = _get_engine_spec_instance(engine_name)
            for engine in engine_spec.get_supported_engines(model_name):
                if engine not in supported:
                    supported.append(engine)
        except NotImplementedError:
            continue
    if not supported:
        raise ValueError(f"No predictor registered for model {model_name!r}.")
    return tuple(supported)


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
def _get_engine_spec_instance(engine: str) -> EngineSpec:
    try:
        return EngineSpec.get(engine)()
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
    return _get_engine_spec_instance(engine).normalize_config(
        cfg,
        model_name=model_name,
        device=device,
    )


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
        if uses_server_backend(genai_config):
            engine = "genai_client"
        elif use_hpip:
            engine = "hpi"
        elif _is_flexible_only_model(model_name):
            engine = "flexible"

    engine, model_dir_resolved = _resolve_requested_engine(
        model_name, engine, model_dir
    )

    requested_spec = _get_engine_spec_instance(engine)

    need_local = requested_spec.needs_local_model
    if model_dir_resolved is None and need_local:
        model_dir_resolved = _resolve_model_dir(
            model_name,
            model_dir,
            model_formats=requested_spec.get_supported_model_formats(),
        )

    requested_spec.ensure_predictor_support(model_name)

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
    predictor_engine_config = requested_spec.to_predictor_config(
        validated_engine_config
    )

    if need_local:
        requested_spec.ensure_model_files(model_dir_resolved)
    requested_spec.ensure_environment(
        device=device,
        engine_config=validated_engine_config,
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
    if predictor_engine_config:
        create_kwargs["engine_config"] = predictor_engine_config
    if need_local and engine in (
        "paddle_static",
        "paddle_dynamic",
        "hpi",
        "flexible",
        "onnxruntime",
        "transformers",
    ):
        create_kwargs["model_dir"] = model_dir_resolved
        create_kwargs["model_config"] = config

    return predictor_cls(**create_kwargs, **kwargs)
