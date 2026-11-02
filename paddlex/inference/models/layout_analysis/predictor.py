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

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from ..object_detection.predictor import DetRunnerPredictor, DetTransformersPredictor
from ..object_detection.processors import Resize, ToBatch
from .processors import LayoutAnalysisProcess
from .result import LayoutAnalysisResult
from .utils import STATIC_SHAPE_MODEL_LIST

LAYOUT_ANALYSIS_TRANSFORMERS_MODELS = [
    "PP-DocLayoutV2",
    "PP-DocLayoutV3",
    "PP-DocLayoutV4",
]


def _split_quad(bbox_pred):
    """Split the quad off 4-point rows [label, score, x1, y1, ..., x4, y4, ...].

    Returns the rows with the quad replaced by its axis-aligned box, keeping any
    trailing columns (e.g. a pre-decoded reading order), and the quads [N, 8].
    Rows with fewer than 10 columns carry no quad and are returned unchanged.
    """
    if bbox_pred.ndim != 2 or bbox_pred.shape[1] < 10:
        return bbox_pred, None
    quad = bbox_pred[:, 2:10]
    points = quad.reshape(-1, 4, 2)
    boxes = np.column_stack(
        [bbox_pred[:, :2], points.min(axis=1), points.max(axis=1), bbox_pred[:, 10:]]
    )
    return boxes, quad


class LayoutAnalysisRunnerPredictor(DetRunnerPredictor):
    """Layout analysis predictor."""

    def __init__(
        self,
        *args,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs,
    ):
        """Initializes LayoutAnalysisPredictor.
        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            img_size (Optional[Union[int, Tuple[int, int]]], optional): The input image size (w, h). Defaults to None.
            threshold (Optional[float], optional): The threshold for filtering out low-confidence predictions.
                Defaults to None.
            layout_nms (bool, optional): Whether to use layout-aware NMS. Defaults to False.
            layout_unclip_ratio (Optional[Union[float, Tuple[float, float]]], optional): The ratio of unclipping the bounding box.
                Defaults to None.
                If it's a single number, then both width and height are used.
                If it's a tuple of two numbers, then they are used separately for width and height respectively.
                If it's None, then no unclipping will be performed.
            layout_merge_bboxes_mode (Optional[Union[str, dict]], optional): The mode for merging bounding boxes. Defaults to None.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        if img_size is not None:
            assert (
                self.model_name not in STATIC_SHAPE_MODEL_LIST
            ), f"The model {self.model_name} is not supported set input shape"
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            elif isinstance(img_size, (tuple, list)):
                assert len(img_size) == 2, f"The length of `img_size` should be 2."
            else:
                raise ValueError(
                    f"The type of `img_size` must be int or Tuple[int, int], but got {type(img_size)}."
                )
        super().__init__(*args, **kwargs)

    def _get_result_class(self):
        return LayoutAnalysisResult

    def _format_output(self, pred: Sequence[Any]) -> List[dict]:
        """
        Transform batch outputs into a list of single image output.

        Args:
            pred (Sequence[Any]): The model outputs. PP-DocLayoutV4 returns
                [bbox_pred, bbox_num, qi, rel_logits, roor_logits], where each row of
                bbox_pred is [label, score, x1, y1, x2, y2, x3, y3, x4, y4] and qi
                holds the query index of each row. Models that predict the page
                deformation append warp_params [batch, 5] = (roll_deg,
                quadrant_conf, shear_deg, kx, ky). Models that also predict
                masks return [bbox_pred, bbox_num, masks, qi, rel_logits,
                roor_logits], where masks is [N, H, W]. In every layout, the
                rows of bbox_pred may instead be [label, score, xmin, ymin,
                xmax, ymax], which carry no quad. Plain detection outputs
                [bbox_pred, bbox_num] may also carry a quad at columns 2:10,
                followed by optional columns such as a pre-decoded reading
                order; outputs of any other length are read as plain
                detection outputs too. Instance segmentation outputs
                [bbox_pred, bbox_num, masks] are handled by
                ``DetRunnerPredictor``.

        Returns:
            List[dict]: A list of dictionaries, one per image.
        """
        masks = None
        warp_params = None
        # The 3-D mask batch tells this layout from V4 with warp_params, whose
        # third output is the 2-D qi.
        if len(pred) == 6 and pred[2].ndim == 3:
            bbox_pred, bbox_num, masks, qi, rel_logits, roor_logits = pred
        elif len(pred) in (5, 6):
            bbox_pred, bbox_num, qi, rel_logits, roor_logits = pred[:5]
            if len(pred) == 6:
                warp_params = pred[5]
        elif len(pred) == 3:
            return super()._format_output(pred)
        else:
            boxes, quad = _split_quad(pred[0])
            results = super()._format_output([boxes, pred[1]])
            if quad is not None:
                splits = np.cumsum(pred[1], dtype=np.int64)[:-1]
                for result, image_quad in zip(results, np.split(quad, splits)):
                    result["quad"] = image_quad
            return results

        if bbox_pred.ndim != 2 or bbox_pred.shape[1] not in (6, 10):
            raise ValueError(
                "Expected bbox_pred of shape [N, 6] or [N, 10], "
                f"got {bbox_pred.shape}."
            )
        if warp_params is not None and warp_params.shape != (len(bbox_num), 5):
            raise ValueError(
                f"Expected warp_params of shape [batch, 5], got {warp_params.shape}."
            )

        # Detection uses the axis-aligned box of each quad.
        boxes, quad = _split_quad(bbox_pred)

        results = []
        box_idx_start = 0
        for idx, num in enumerate(bbox_num):
            box_idx_end = box_idx_start + int(num)
            result = {
                "boxes": boxes[box_idx_start:box_idx_end],
                "qi": qi[idx, : int(num)],
                "rel_logits": rel_logits[idx],
                "roor_logits": roor_logits[idx],
            }
            if masks is not None:
                result["masks"] = masks[box_idx_start:box_idx_end]
            if quad is not None:
                result["quad"] = quad[box_idx_start:box_idx_end]
            if warp_params is not None:
                result["warp_params"] = warp_params[idx].tolist()
            results.append(result)
            box_idx_start = box_idx_end
        return results

    def process(
        self,
        batch_data: List[Any],
        threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, dict]] = None,
        layout_shape_mode: Optional[str] = "auto",
        filter_overlap_boxes: Optional[bool] = True,
        skip_order_labels: Optional[List[str]] = None,
    ):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).
            threshold (Optional[float, dict], optional): The threshold for filtering out low-confidence predictions.
            layout_nms (bool, optional): Whether to use layout-aware NMS. Defaults to None.
            layout_unclip_ratio (Optional[Union[float, Tuple[float, float]]], optional): The ratio of unclipping the bounding box.
            layout_merge_bboxes_mode (Optional[Union[str, dict]], optional): The mode for merging bounding boxes. Defaults to None.
            layout_shape_mode (Optional[str], optional): The mode for layout shape. Defaults to "auto", [ "rect", "quad","poly", "auto"]. are supported.
            filter_overlap_boxes (Optional[bool], optional): Whether to filter out overlap boxes. Defaults to True.
            skip_order_labels (Optional[List[str]], optional): The labels to skip order. Defaults to None.

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names
                for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        datas = batch_data.instances
        # preprocess
        for pre_op in self.pre_ops[:-1]:
            datas = pre_op(datas)

        # use `ToBatch` format batch inputs
        batch_inputs = self.pre_ops[-1](datas)

        # do infer
        batch_preds = self.runner(batch_inputs)

        # process a batch of predictions into a list of single image result
        preds_list = self._format_output(batch_preds)
        # postprocess
        boxes = self.post_op(
            preds_list,
            datas,
            threshold=threshold if threshold is not None else self.threshold,
            layout_nms=self.layout_nms if layout_nms is None else layout_nms,
            layout_unclip_ratio=layout_unclip_ratio or self.layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode
            or self.layout_merge_bboxes_mode,
            layout_shape_mode=layout_shape_mode,
            filter_overlap_boxes=filter_overlap_boxes,
            skip_order_labels=skip_order_labels,
        )

        result = {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": [data["ori_img"] for data in datas],
            "boxes": boxes,
        }
        # Image-level, so it never goes through the box post-processor.  Added
        # only for models that predict it, to leave every other model's result
        # (and its serialized JSON) unchanged.  `BasePredictor.apply` splits this
        # dict of lists by index, so each single-image result gets its own row.
        if any("warp_params" in pred for pred in preds_list):
            result["warp_params"] = [pred.get("warp_params") for pred in preds_list]
        return result

    @DetRunnerPredictor.register("Resize")
    def build_resize(self, target_size, keep_ratio=False, interp=2):
        assert target_size
        self.target_size = target_size
        if isinstance(interp, int):
            interp = {
                0: "NEAREST",
                1: "LINEAR",
                2: "BICUBIC",
                3: "AREA",
                4: "LANCZOS4",
            }[interp]
        op = Resize(target_size=target_size[::-1], keep_ratio=keep_ratio, interp=interp)
        return op

    def build_to_batch(self):
        models_required_imgsize = [
            "PP-DocLayoutV2",
            "PP-DocLayoutV3",
            "PP-DocLayoutV4",
        ]
        if any(name in self.model_name for name in models_required_imgsize):
            ordered_required_keys = (
                "img_size",
                "img",
                "scale_factors",
            )
        else:
            ordered_required_keys = ("img", "scale_factors")

        return ToBatch(ordered_required_keys=ordered_required_keys)

    def build_postprocess(self):
        if self.threshold is None:
            self.threshold = self.config.get("draw_threshold", 0.5)
        if self.layout_nms is None:
            self.layout_nms = self.config.get("layout_nms", None)
        if self.layout_unclip_ratio is None:
            self.layout_unclip_ratio = self.config.get("layout_unclip_ratio", None)
        if self.layout_merge_bboxes_mode is None:
            self.layout_merge_bboxes_mode = self.config.get(
                "layout_merge_bboxes_mode", None
            )
        scale_size = getattr(self, "target_size", [800, 800])
        return LayoutAnalysisProcess(
            labels=self.config["label_list"], scale_size=scale_size
        )


class LayoutAnalysisTransformersPredictor(DetTransformersPredictor):
    """Layout analysis predictor backed by HuggingFace transformers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout_postprocess = LayoutAnalysisProcess(
            labels=self.labels,
            scale_size=self._resolve_scale_size(),
        )

    def _get_result_class(self):
        return LayoutAnalysisResult

    def _resolve_scale_size(self) -> List[int]:
        for cfg in self.model_config.get("Preprocess", []):
            if cfg.get("type") != "Resize":
                continue
            target_size = cfg.get("target_size")
            if isinstance(target_size, int):
                return [target_size, target_size]
            if isinstance(target_size, (tuple, list)) and len(target_size) == 2:
                return list(target_size)
        return [800, 800]

    def preprocess_images(self, images, **kwargs):
        if self.model_name == "PP-DocLayoutV4" and kwargs.get("do_resize", True):
            # Match the Paddle engines exactly: tensor bicubic interpolation has
            # different uint8 rounding, which changes the page geometry output.
            resize = Resize(
                target_size=self._resolve_scale_size()[::-1], interp="BICUBIC"
            )
            images = [Image.fromarray(resize.resize(np.asarray(im))) for im in images]
            kwargs["do_resize"] = False
        return super().preprocess_images(images, **kwargs)

    def _format_layout_transformers_output(self, prediction):
        formatted = self._format_transformers_output(prediction)
        if "order_seq" in prediction:
            order_seq = (
                prediction["order_seq"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            if len(formatted) == len(order_seq):
                formatted = np.concatenate([formatted, order_seq[:, None]], axis=1)

        output = {"boxes": formatted}
        if "polygon_points" in prediction:
            points = prediction["polygon_points"]
            if self.model_name == "PP-DocLayoutV4":
                # V4 predicts one quad [4, 2] per box, in original-image
                # coordinates, like the Paddle engines.
                points = points.detach().cpu().numpy()
                if points.shape != (len(formatted), 4, 2):
                    raise ValueError(
                        f"Expected polygon_points of shape [{len(formatted)}, 4, 2], "
                        f"got {points.shape}."
                    )
                output["quad"] = points.reshape(len(formatted), 8)
            else:
                output["polygon_points"] = [np.asarray(p) for p in points]
        if "warp_params" in prediction:
            warp_params = prediction["warp_params"].detach().cpu().numpy()
            if warp_params.shape != (5,):
                raise ValueError(
                    f"Expected warp_params of shape [5], got {warp_params.shape}."
                )
            output["warp_params"] = warp_params
        return output

    def process(
        self,
        batch_data: List[Any],
        threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, dict]] = None,
        layout_shape_mode: Optional[str] = "auto",
        filter_overlap_boxes: Optional[bool] = True,
        skip_order_labels: Optional[List[str]] = None,
    ):
        if not hasattr(self.image_processor, "post_process_object_detection"):
            raise RuntimeError(
                f"{type(self.image_processor).__name__} does not support "
                "`post_process_object_detection`."
            )

        datas = self.read_op(batch_data.instances)
        images = [Image.fromarray(data["img"]) for data in datas]
        effective_threshold, hf_threshold = self._get_hf_threshold(threshold)

        model_inputs = self.preprocess_images(images=images)
        outputs = self.forward(model_inputs)
        predictions = self.postprocess(outputs, datas=datas, threshold=hf_threshold)

        batch_outputs = [
            self._format_layout_transformers_output(prediction)
            for prediction in predictions
        ]
        boxes = self.layout_postprocess(
            batch_outputs,
            datas,
            threshold=effective_threshold,
            layout_nms=self.layout_nms if layout_nms is None else layout_nms,
            layout_unclip_ratio=layout_unclip_ratio or self.layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode
            or self.layout_merge_bboxes_mode,
            layout_shape_mode=layout_shape_mode,
            filter_overlap_boxes=filter_overlap_boxes,
            skip_order_labels=skip_order_labels,
        )

        result = {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": [data["ori_img"] for data in datas],
            "boxes": boxes,
        }
        if any("warp_params" in output for output in batch_outputs):
            result["warp_params"] = [
                output.get("warp_params") for output in batch_outputs
            ]
        return result

    def postprocess(self, outputs, *, datas, threshold, **kwargs):
        predictions = self.image_processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=self._get_target_sizes(datas),
        )

        return predictions
