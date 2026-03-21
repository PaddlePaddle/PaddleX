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

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ...utils import logging
from ...utils.config import parse_config
from ..models import HPIConfig, PaddlePredictorOption
from .anomaly_detection import AnomalyDetectionPipeline
from .attribute_recognition import (
    PedestrianAttributeRecPipeline,
    VehicleAttributeRecPipeline,
)
from .base import BasePipeline
from .components import BaseChat, BaseGeneratePrompt, BaseRetriever
from .doc_preprocessor import DocPreprocessorPipeline
from .doc_understanding import DocUnderstandingPipeline
from .face_recognition import FaceRecPipeline
from .formula_recognition import FormulaRecognitionPipeline
from .image_classification import ImageClassificationPipeline
from .image_multilabel_classification import ImageMultiLabelClassificationPipeline
from .instance_segmentation import InstanceSegmentationPipeline
from .keypoint_detection import KeypointDetectionPipeline
from .layout_parsing import LayoutParsingPipeline
from .m_3d_bev_detection import BEVDet3DPipeline
from .multilingual_speech_recognition import MultilingualSpeechRecognitionPipeline
from .object_detection import ObjectDetectionPipeline
from .ocr import OCRPipeline
from .open_vocabulary_detection import OpenVocabularyDetectionPipeline
from .open_vocabulary_segmentation import OpenVocabularySegmentationPipeline
from .paddleocr_vl import PaddleOCRVLPipeline
from .pp_chatocr import PP_ChatOCRv3_Pipeline, PP_ChatOCRv4_Pipeline
from .pp_doctranslation import PP_DocTranslation_Pipeline
from .pp_shitu_v2 import ShiTuV2Pipeline
from .rotated_object_detection import RotatedObjectDetectionPipeline
from .seal_recognition import SealRecognitionPipeline
from .semantic_segmentation import SemanticSegmentationPipeline
from .small_object_detection import SmallObjectDetectionPipeline
from .table_recognition import TableRecognitionPipeline, TableRecognitionPipelineV2
from .text_to_speech import TextToSpeechPipeline
from .ts_anomaly_detection import TSAnomalyDetPipeline
from .ts_classification import TSClsPipeline
from .ts_forecasting import TSFcPipeline
from .video_classification import VideoClassificationPipeline
from .video_detection import VideoDetectionPipeline


def get_pipeline_path(pipeline_name: str) -> str:
    """
    Get the full path of the pipeline configuration file based on the provided pipeline name.

    Args:
        pipeline_name (str): The name of the pipeline.

    Returns:
        str: The full path to the pipeline configuration file or None if not found.
    """
    pipeline_path = (
        Path(__file__).parent.parent.parent
        / "configs/pipelines"
        / f"{pipeline_name}.yaml"
    ).resolve()
    if not Path(pipeline_path).exists():
        return None
    return pipeline_path


def load_pipeline_config(pipeline: str) -> Dict[str, Any]:
    """
    Load the pipeline configuration.

    Args:
        pipeline (str): The name of the pipeline or the path to the config file.

    Returns:
        Dict[str, Any]: The parsed pipeline configuration.

    Raises:
        Exception: If the config file of pipeline does not exist.
    """
    if not (pipeline.endswith(".yml") or pipeline.endswith(".yaml")):
        pipeline_path = get_pipeline_path(pipeline)
        if pipeline_path is None:
            raise Exception(
                f"The pipeline ({pipeline}) does not exist! Please use a pipeline name or a config file path!"
            )
    else:
        pipeline_path = pipeline
    config = parse_config(pipeline_path)
    return config


def create_pipeline(
    pipeline: Optional[str] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    engine: Optional[str] = None,
    engine_config: Optional[Dict[str, Any]] = None,
    pp_option: Optional[PaddlePredictorOption] = None,
    use_hpip: Optional[bool] = None,
    hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    **kwargs: Any,
) -> BasePipeline:
    """Create a pipeline instance from a pipeline name or config.


    Args:
        pipeline (Optional[str], optional): Pipeline name (e.g. "OCR", "object_detection") or
            path to a YAML config file. Required if `config` is None.
        config (Optional[Dict[str, Any]], optional): Pipeline configuration dict. If provided,
            `pipeline` name is optional and used only for validation. If both are provided
            and pipeline names differ, `config` takes precedence.
        device (Optional[str], optional): Device to run on (e.g. "gpu", "cpu", "npu:0").
        engine (Optional[str], optional): Inference engine for models in the pipeline.
        engine_config (Optional[Dict[str, Any]], optional): Engine-specific config.
        pp_option (Optional[PaddlePredictorOption], optional): Paddle predictor options.
        use_hpip (Optional[bool], optional): Whether to use HPIP.
        hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional): HPIP configuration.

    Returns:
        BasePipeline: The created pipeline instance.
    """
    if pipeline is None and config is None:
        raise ValueError(
            "Both `pipeline` and `config` cannot be None at the same time."
        )
    if config is None:
        config = load_pipeline_config(pipeline)
    else:
        if pipeline is not None and config["pipeline_name"] != pipeline:
            logging.warning(
                "The pipeline name in the config (%r) is different from the specified pipeline name (%r). %r will be used.",
                config["pipeline_name"],
                pipeline,
                config["pipeline_name"],
            )
        config = config.copy()
    pipeline_name = config["pipeline_name"]
    if use_hpip is None:
        use_hpip = config.pop("use_hpip", False)
    else:
        config.pop("use_hpip", None)
    if hpi_config is None:
        hpi_config = config.pop("hpi_config", None)
    else:
        config.pop("hpi_config", None)
    if engine is None:
        engine = config.get("engine", None)
    else:
        config["engine"] = engine
    if engine_config is None:
        engine_config = config.get("engine_config", None)
    else:
        config["engine_config"] = engine_config

    pipeline = BasePipeline.get(pipeline_name)(
        config=config,
        device=device,
        engine=engine,
        engine_config=engine_config,
        pp_option=pp_option,
        use_hpip=use_hpip,
        hpi_config=hpi_config,
        **kwargs,
    )
    return pipeline


def create_chat_bot(config: Dict, *args, **kwargs) -> BaseChat:
    """Creates an instance of a chat bot based on the provided configuration.

    Args:
        config (Dict): Configuration settings, expected to be a dictionary with at least a 'model_name' key.
        *args: Additional positional arguments. Not used in this function but allowed for future compatibility.
        **kwargs: Additional keyword arguments. Not used in this function but allowed for future compatibility.

    Returns:
        BaseChat: An instance of the chat bot class corresponding to the 'model_name' in the config.
    """
    if "chat_bot_config_error" in config:
        raise ValueError(config["chat_bot_config_error"])

    api_type = config["api_type"]
    chat_bot = BaseChat.get(api_type)(config)
    return chat_bot


def create_retriever(
    config: Dict,
    *args,
    **kwargs,
) -> BaseRetriever:
    """
    Creates a retriever instance based on the provided configuration.

    Args:
        config (Dict): Configuration settings, expected to be a dictionary with at least a 'model_name' key.
        *args: Additional positional arguments. Not used in this function but allowed for future compatibility.
        **kwargs: Additional keyword arguments. Not used in this function but allowed for future compatibility.

    Returns:
        BaseRetriever: An instance of a retriever class corresponding to the 'model_name' in the config.
    """
    if "retriever_config_error" in config:
        raise ValueError(config["retriever_config_error"])
    api_type = config["api_type"]
    retriever = BaseRetriever.get(api_type)(config)
    return retriever


def create_prompt_engineering(
    config: Dict,
    *args,
    **kwargs,
) -> BaseGeneratePrompt:
    """
    Creates a prompt engineering instance based on the provided configuration.

    Args:
        config (Dict): Configuration settings, expected to be a dictionary with at least a 'task_type' key.
        *args: Variable length argument list for additional positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        BaseGeneratePrompt: An instance of a prompt engineering class corresponding to the 'task_type' in the config.
    """
    if "pe_config_error" in config:
        raise ValueError(config["pe_config_error"])
    task_type = config["task_type"]
    pe = BaseGeneratePrompt.get(task_type)(config)
    return pe
