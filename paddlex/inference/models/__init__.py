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
from typing import Any, Dict, Optional, Type, Union

from ...utils import errors, logging
from ..utils.hpi import HPIConfig
from ..utils.official_models import official_models
from ..utils.pp_option import PaddlePredictorOption
from .anomaly_detection import UadPredictor
from .base.predictor import BasePredictor, FlexiblePredictor, RunnerPredictor
from .common.genai import GenAIConfig, need_local_model
from .doc_vlm import DocVLMPredictor
from .engine_specs import EngineSpec
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


def _pick_predictor_cls(model_name: str, engine: str) -> Type[BasePredictor]:
    return _get_engine_spec_instance(engine).get_predictor_cls(model_name)


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

    requested_spec = _get_engine_spec_instance(engine)
    if engine != "paddle":
        requested_spec.ensure_predictor_support(model_name)

    need_local = requested_spec.needs_local_model
    model_dir_resolved: Optional[Path] = None
    if need_local:
        if model_dir is None:
            supported_engines = requested_spec.get_supported_engines(model_name)
            model_dir_resolved = Path(
                official_models.get_model_path(
                    model_name,
                    engine=engine,
                    supported_engines=supported_engines,
                )
            )
        else:
            model_dir_resolved = Path(model_dir)
            if not model_dir_resolved.exists():
                raise FileNotFoundError(f"{model_dir} does not exist!")

    if engine == "paddle":
        engine = requested_spec.resolve_engine_from_model_dir(model_dir_resolved)
        requested_spec = _get_engine_spec_instance(engine)
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
