# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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
from typing import Any, Dict, Optional
from importlib import import_module
from .base import BasePipeline
from ..utils.pp_option import PaddlePredictorOption
from .components import BaseChat, BaseRetriever, BaseGeneratePrompt
from ...utils import logging
from ...utils.config import parse_config
from .ocr import OCRPipeline
from .doc_preprocessor import DocPreprocessorPipeline
from .layout_parsing import LayoutParsingPipeline
from .pp_chatocr import PP_ChatOCRv3_Pipeline, PP_ChatOCRv4_Pipeline
from .image_classification import ImageClassificationPipeline
from .object_detection import ObjectDetectionPipeline
from .seal_recognition import SealRecognitionPipeline
from .table_recognition import TableRecognitionPipeline
from .table_recognition import TableRecognitionPipelineV2
from .multilingual_speech_recognition import MultilingualSpeechRecognitionPipeline
from .formula_recognition import FormulaRecognitionPipeline
from .image_multilabel_classification import ImageMultiLabelClassificationPipeline
from .video_classification import VideoClassificationPipeline
from .video_detection import VideoDetectionPipeline
from .anomaly_detection import AnomalyDetectionPipeline
from .ts_forecasting import TSFcPipeline
from .ts_anomaly_detection import TSAnomalyDetPipeline
from .ts_classification import TSClsPipeline
from .pp_shitu_v2 import ShiTuV2Pipeline
from .face_recognition import FaceRecPipeline
from .attribute_recognition import (
    PedestrianAttributeRecPipeline,
    VehicleAttributeRecPipeline,
)

from .semantic_segmentation import SemanticSegmentationPipeline
from .instance_segmentation import InstanceSegmentationPipeline
from .small_object_detection import SmallObjectDetectionPipeline
from .rotated_object_detection import RotatedObjectDetectionPipeline
from .keypoint_detection import KeypointDetectionPipeline
from .open_vocabulary_detection import OpenVocabularyDetectionPipeline
from .open_vocabulary_segmentation import OpenVocabularySegmentationPipeline

module_3d_bev_detection = import_module(
    ".3d_bev_detection", "paddlex.inference.pipelines"
)
BEVDet3DPipeline = getattr(module_3d_bev_detection, "BEVDet3DPipeline")


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
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    pp_option: Optional[PaddlePredictorOption] = None,
    use_hpip: bool = False,
    *args: Any,
    **kwargs: Any,
) -> BasePipeline:
    """
    Create a pipeline instance based on the provided parameters.

    If the input parameter config is not provided, it is obtained from the
    default config corresponding to the pipeline name.

    Args:
        pipeline (Optional[str], optional): The name of the pipeline to
            create, or the path to the config file. Defaults to None.
        config (Optional[Dict[str, Any]], optional): The pipeline configuration.
            Defaults to None.
        device (Optional[str], optional): The device to run the pipeline on.
            Defaults to None.
        pp_option (Optional[PaddlePredictorOption], optional): The options for
            the PaddlePredictor. Defaults to None.
        use_hpip (bool, optional): Whether to use high-performance inference
            plugin (HPIP) for prediction. Defaults to False.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

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
    pipeline_name = config["pipeline_name"]

    pipeline = BasePipeline.get(pipeline_name)(
        config=config,
        device=device,
        pp_option=pp_option,
        use_hpip=use_hpip,
        *args,
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
