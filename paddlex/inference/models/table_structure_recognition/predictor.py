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

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from ....utils.func_register import FuncRegister
from ....utils.import_guard import import_paddle
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ...utils.official_models import official_models
from ..common import Normalize, ResizeByLong, ToBatch, ToCHWImage
from ..predictors import RunnerPredictor, TransformersPredictor
from .processors import Pad, TableLabelDecode
from .result import TableRecResult

TABLE_REC_TRANSFORMERS_MODELS = ["SLANeXt_wired", "SLANeXt_wireless"]


class TableRunnerPredictor(RunnerPredictor):

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        super().__init__(*args, **kwargs)
        self.preprocessors, self.postprocessors = self._build()

    def _build_batch_sampler(self) -> ImageBatchSampler:
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        return TableRecResult

    def _build(self) -> Tuple:
        preprocessors = []
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            op = func(self, **args) if args else func(self)
            if op:
                preprocessors.append(op)
        preprocessors.append(ToBatch())
        postprocessors = TableLabelDecode(
            model_name=self.config["Global"]["model_name"],
            merge_no_span_structure=self.config["PreProcess"]["transform_ops"][1][
                "TableLabelEncode"
            ]["merge_no_span_structure"],
            dict_character=self.config["PostProcess"]["character_dict"],
        )
        return preprocessors, postprocessors

    def process(self, batch_data: List[Union[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        batch_raw_imgs = self.preprocessors[0](imgs=batch_data.instances)  # ReadImage
        ori_shapes = []
        for s in range(len(batch_raw_imgs)):
            ori_shapes.append([batch_raw_imgs[s].shape[1], batch_raw_imgs[s].shape[0]])
        batch_imgs = self.preprocessors[1](imgs=batch_raw_imgs)  # ResizeByLong
        batch_imgs = self.preprocessors[2](imgs=batch_imgs)  # Normalize
        pad_results = self.preprocessors[3](imgs=batch_imgs)  # Pad
        pad_imgs = []
        padding_sizes = []
        for pad_img, padding_size in pad_results:
            pad_imgs.append(pad_img)
            padding_sizes.append(padding_size)
        batch_imgs = self.preprocessors[4](imgs=pad_imgs)  # ToCHWImage
        x = self.preprocessors[5](imgs=batch_imgs)  # ToBatch

        batch_preds = self.runner(x=x)

        table_result = self.postprocessors(
            pred=batch_preds,
            img_size=padding_sizes,
            ori_img_size=ori_shapes,
        )

        table_result_bbox = []
        table_result_structure = []
        table_result_structure_score = []
        for i in range(len(table_result)):
            table_result_bbox.append(table_result[i]["bbox"])
            table_result_structure.append(table_result[i]["structure"])
            table_result_structure_score.append(table_result[i]["structure_score"])

        final_result = {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "bbox": table_result_bbox,
            "structure": table_result_structure,
            "structure_score": table_result_structure_score,
        }

        return final_result

    @register("DecodeImage")
    def build_readimg(self, channel_first=False, img_mode="BGR"):
        assert channel_first is False
        assert img_mode == "BGR"
        return ReadImage(format=img_mode)

    @register("TableLabelEncode")
    def foo(self, *args, **kwargs):
        return None

    @register("TableBoxEncode")
    def foo(self, *args, **kwargs):
        return None

    @register("ResizeTableImage")
    def build_resize_table(self, max_len=488, resize_bboxes=True):
        return ResizeByLong(target_long_edge=max_len)

    @register("NormalizeImage")
    def build_normalize(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        scale=1 / 255,
        order="hwc",
    ):
        return Normalize(mean=mean, std=std)

    @register("PaddingTableImage")
    def build_padding(self, size=[488, 448], pad_value=0):
        return Pad(target_size=size[0], val=pad_value)

    @register("ToCHWImage")
    def build_to_chw(self):
        return ToCHWImage()

    @register("KeepKeys")
    def foo(self, *args, **kwargs):
        return None

    def _pack_res(self, single):
        keys = ["input_path", "bbox", "structure"]
        return TableRecResult({key: single[key] for key in keys})


class TableTransformersPredictor(TransformersPredictor):

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        super().__init__(*args, **kwargs)
        self.read_op = ReadImage(format="BGR")
        self.image_processor, self.infer, self.postprocessor = self._build()
        self.padding_size = self._resolve_padding_size()

    def _build_batch_sampler(self) -> ImageBatchSampler:
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        return TableRecResult

    def _build(self) -> Tuple:
        from transformers import AutoImageProcessor, SLANeXtForTableRecognition

        image_processor = self._load_pretrained_processor(AutoImageProcessor)
        model = self._load_pretrained_model(SLANeXtForTableRecognition)
        self._ensure_slanext_loc_generator(model)
        postprocessor = TableLabelDecode(
            model_name=self.config["Global"]["model_name"],
            merge_no_span_structure=self.config["PreProcess"]["transform_ops"][1][
                "TableLabelEncode"
            ]["merge_no_span_structure"],
            dict_character=self.config["PostProcess"]["character_dict"],
        )
        return image_processor, model, postprocessor

    def _resolve_padding_size(self) -> List[int]:
        for cfg in self.config["PreProcess"]["transform_ops"]:
            if "PaddingTableImage" not in cfg:
                continue
            size = cfg["PaddingTableImage"]["size"]
            if isinstance(size, int):
                return [size, size]
            if isinstance(size, (list, tuple)) and len(size) == 2:
                return [int(size[0]), int(size[1])]
        pad_size = getattr(self.image_processor, "pad_size", None) or {}
        return [int(pad_size.get("width", 512)), int(pad_size.get("height", 512))]

    def _resolve_paddle_model_prefix(self) -> str:
        paddle_model_dir = Path(
            official_models.get_model_path(self.model_name, model_formats=["paddle"])
        )
        return str((paddle_model_dir / "inference").resolve())

    def _ensure_slanext_loc_generator(self, model) -> None:
        import torch.nn as nn

        head = model.head
        if not hasattr(head, "loc_generator"):
            hidden_size = model.config.hidden_size
            head.loc_generator = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, model.config.loc_reg_num),
                nn.Sigmoid(),
            )

        self._load_slanext_loc_generator_weights(head.loc_generator, model.config)

        infer_device = self._get_infer_device(model=model)
        head.loc_generator = head.loc_generator.to(device=infer_device)

    def _load_slanext_loc_generator_weights(self, loc_generator, config) -> None:
        paddle = import_paddle()
        import torch

        state_items = list(
            paddle.jit.load(self._resolve_paddle_model_prefix()).state_dict().items()
        )
        hidden_size = int(config.hidden_size)
        loc_reg_num = int(config.loc_reg_num)
        expected_shapes = [
            [hidden_size],
            [hidden_size, hidden_size],
            [loc_reg_num],
            [hidden_size, loc_reg_num],
        ]

        start_idx = None
        for idx in range(len(state_items) - len(expected_shapes) + 1):
            window_shapes = [
                list(tensor.shape) for _, tensor in state_items[idx : idx + 4]
            ]
            if window_shapes == expected_shapes:
                start_idx = idx
        if start_idx is None:
            raise RuntimeError(
                f"Failed to locate SLANeXt loc_generator weights for {self.model_name!r}."
            )

        fc1_bias = state_items[start_idx][1].numpy()
        fc1_weight = state_items[start_idx + 1][1].numpy().T
        fc2_bias = state_items[start_idx + 2][1].numpy()
        fc2_weight = state_items[start_idx + 3][1].numpy().T

        with torch.no_grad():
            loc_generator[0].bias.copy_(torch.from_numpy(fc1_bias))
            loc_generator[0].weight.copy_(torch.from_numpy(fc1_weight))
            loc_generator[1].bias.copy_(torch.from_numpy(fc2_bias))
            loc_generator[1].weight.copy_(torch.from_numpy(fc2_weight))

    def _run_slanext_decoder(self, pixel_values):
        import torch
        import torch.nn.functional as F

        backbone_outputs = self.infer.backbone(pixel_values)
        hidden_states = backbone_outputs.last_hidden_state
        head = self.infer.head

        features = torch.zeros(
            (hidden_states.shape[0], self.infer.config.hidden_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        predicted_chars = torch.zeros(
            size=[hidden_states.shape[0]],
            dtype=torch.long,
            device=hidden_states.device,
        )

        structure_preds_list = []
        structure_ids_list = []
        loc_preds_list = []
        for _ in range(self.infer.config.max_text_length + 1):
            embedding_feature = F.one_hot(
                predicted_chars, self.infer.config.out_channels
            ).float()
            features, _ = head.structure_attention_cell(
                features,
                hidden_states.float(),
                embedding_feature,
            )
            structure_step = head.structure_generator(features)
            loc_step = head.loc_generator(features.float())
            predicted_chars = structure_step.argmax(dim=1)

            structure_preds_list.append(structure_step)
            structure_ids_list.append(predicted_chars)
            loc_preds_list.append(loc_step)
            if (
                torch.stack(structure_ids_list, dim=1)
                .eq(self.infer.config.out_channels - 1)
                .any(-1)
                .all()
            ):
                break

        structure_preds = F.softmax(
            torch.stack(structure_preds_list, dim=1),
            dim=-1,
            dtype=torch.float32,
        ).to(hidden_states.dtype)
        loc_preds = torch.stack(loc_preds_list, dim=1).to(hidden_states.dtype)
        return loc_preds, structure_preds

    def process(self, batch_data: List[Union[str, np.ndarray]]) -> Dict[str, Any]:
        import torch

        batch_raw_imgs = self.read_op(imgs=batch_data.instances)
        images = [Image.fromarray(img[..., ::-1]) for img in batch_raw_imgs]
        ori_shapes = [[img.shape[1], img.shape[0]] for img in batch_raw_imgs]
        padding_sizes = [list(self.padding_size) for _ in batch_raw_imgs]

        model_inputs = self.image_processor(images=images, return_tensors="pt")
        model_inputs = self._move_to_infer_device(model_inputs)

        with torch.inference_mode():
            loc_preds, structure_probs = self._run_slanext_decoder(
                model_inputs["pixel_values"]
            )

        predictions = self.postprocessor(
            pred=[
                loc_preds.detach().cpu().numpy(),
                structure_probs.detach().cpu().numpy(),
            ],
            img_size=padding_sizes,
            ori_img_size=ori_shapes,
        )

        table_result_bbox = []
        table_result_structure = []
        table_result_structure_score = []
        for pred in predictions:
            table_result_bbox.append(pred["bbox"])
            table_result_structure.append(pred["structure"])
            table_result_structure_score.append(pred["structure_score"])

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "bbox": table_result_bbox,
            "structure": table_result_structure,
            "structure_score": table_result_structure_score,
        }

    def _pack_res(self, single):
        keys = ["input_path", "bbox", "structure"]
        return TableRecResult({key: single[key] for key in keys})
