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

from typing import Any, Union, Dict, List, Tuple, Iterator
import shutil
import tempfile
from importlib import import_module
import lazy_paddle

from ....utils import logging
from ....utils.func_register import FuncRegister

module_3d_bev_detection = import_module(".3d_bev_detection", "paddlex.modules")
module_3d_model_list = getattr(module_3d_bev_detection, "model_list")
MODELS = getattr(module_3d_model_list, "MODELS")
from ...common.batch_sampler import Det3DBatchSampler
from ...common.reader import ReadNuscenesData
from ..common import StaticInfer
from ..base import BasicPredictor
from ..base.predictor.base_predictor import PredictionWrap
from .processors import (
    LoadPointsFromFile,
    LoadPointsFromMultiSweeps,
    LoadMultiViewImageFromFiles,
    ResizeImage,
    NormalizeImage,
    PadImage,
    SampleFilterByKey,
    GetInferInput,
)
from .result import BEV3DDetResult


class BEVDet3DPredictor(BasicPredictor):
    """BEVDet3DPredictor that inherits from BasicPredictor."""

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """Initializes BEVDet3DPredictor.

        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        self.temp_dir = tempfile.mkdtemp()
        logging.info(
            f"infer data will be stored in temporary directory {self.temp_dir}"
        )
        super().__init__(*args, **kwargs)
        self.pre_tfs, self.infer = self._build()

    def _build_batch_sampler(self) -> Det3DBatchSampler:
        """Builds and returns an Det3DBatchSampler instance.

        Returns:
            Det3DBatchSampler: An instance of Det3DBatchSampler.
        """
        return Det3DBatchSampler(temp_dir=self.temp_dir)

    def _get_result_class(self) -> type:
        """Returns the result class, BEV3DDetResult.

        Returns:
            type: The BEV3DDetResult class.
        """
        return BEV3DDetResult

    def _build(self) -> Tuple:
        """Build the preprocessors and inference engine based on the configuration.

        Returns:
            tuple: A tuple containing the preprocessors and inference engine.
        """
        if (
            lazy_paddle.is_compiled_with_cuda()
            and not lazy_paddle.is_compiled_with_rocm()
        ):
            from ....ops.voxelize import hard_voxelize
            from ....ops.iou3d_nms import nms_gpu
        else:
            logging.error("3D BEVFusion custom ops only support GPU platform!")

        pre_tfs = {"Read": ReadNuscenesData()}
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            name, op = func(self, **args) if args else func(self)
            if op:
                pre_tfs[name] = op
        pre_tfs["GetInferInput"] = GetInferInput()

        infer = StaticInfer(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )

        return pre_tfs, infer

    def _format_output(
        self, infer_input: List[Any], outs: List[Any], img_metas: Dict[str, Any]
    ) -> Dict[str, Any]:
        """format inference input and output into predict result

        Args:
            infer_input(List): Model infer inputs with list containing images, points and lidar2img matrix.
            outs(List): Model infer output containing bboxes, scores, labels result.
            img_metas(Dict): Image metas info of input sample.

        Returns:
            Dict: A Dict containing formatted inference output results.
        """
        input_lidar_path = img_metas["input_lidar_path"]
        input_img_paths = img_metas["input_img_paths"]
        sample_id = img_metas["sample_id"]
        results = {}
        out_bboxes_3d = []
        out_scores_3d = []
        out_labels_3d = []
        input_imgs = []
        input_points = []
        input_lidar2imgs = []
        input_ids = []
        input_lidar_path_list = []
        input_img_paths_list = []
        out_bboxes_3d.append(outs[0])
        out_labels_3d.append(outs[1])
        out_scores_3d.append(outs[2])
        input_imgs.append(infer_input[1])
        input_points.append(infer_input[0])
        input_lidar2imgs.append(infer_input[2])
        input_ids.append(sample_id)
        input_lidar_path_list.append(input_lidar_path)
        input_img_paths_list.append(input_img_paths)
        results["input_path"] = input_lidar_path_list
        results["input_img_paths"] = input_img_paths_list
        results["sample_id"] = input_ids
        results["boxes_3d"] = out_bboxes_3d
        results["labels_3d"] = out_labels_3d
        results["scores_3d"] = out_scores_3d
        return results

    def process(self, batch_data: List[str]) -> Dict[str, Any]:
        """
        Process a batch of data through the preprocessing and inference.

        Args:
            batch_data (List[str]): A batch of input data (e.g., sample anno file paths).

        Returns:
            dict: A dictionary containing the input path, input img, input points, input lidar2img, output bboxes, output labels, output scores and label names. Keys include 'input_path', 'input_img', 'input_points', 'input_lidar2img', 'boxes_3d', 'labels_3d' and 'scores_3d'.
        """
        sample = self.pre_tfs["Read"](batch_data=batch_data)
        sample = self.pre_tfs["LoadPointsFromFile"](results=sample[0])
        sample = self.pre_tfs["LoadPointsFromMultiSweeps"](results=sample)
        sample = self.pre_tfs["LoadMultiViewImageFromFiles"](sample=sample)
        sample = self.pre_tfs["ResizeImage"](results=sample)
        sample = self.pre_tfs["NormalizeImage"](results=sample)
        sample = self.pre_tfs["PadImage"](results=sample)
        sample = self.pre_tfs["SampleFilterByKey"](sample=sample)
        infer_input, img_metas = self.pre_tfs["GetInferInput"](sample=sample)
        infer_output = self.infer(x=infer_input)
        results = self._format_output(infer_input, infer_output, img_metas)
        return results

    @register("LoadPointsFromFile")
    def build_load_img_from_file(
        self, load_dim=6, use_dim=[0, 1, 2], shift_height=False, use_color=False
    ):
        return "LoadPointsFromFile", LoadPointsFromFile(
            load_dim=load_dim,
            use_dim=use_dim,
            shift_height=shift_height,
            use_color=use_color,
        )

    @register("LoadPointsFromMultiSweeps")
    def build_load_points_from_multi_sweeps(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        point_cloud_angle_range=None,
    ):
        return "LoadPointsFromMultiSweeps", LoadPointsFromMultiSweeps(
            sweeps_num=sweeps_num,
            load_dim=load_dim,
            use_dim=use_dim,
            pad_empty_sweeps=pad_empty_sweeps,
            remove_close=remove_close,
            test_mode=test_mode,
            point_cloud_angle_range=point_cloud_angle_range,
        )

    @register("LoadMultiViewImageFromFiles")
    def build_load_multi_view_image_from_files(
        self,
        to_float32=False,
        project_pts_to_img_depth=False,
        cam_depth_range=[4.0, 45.0, 1.0],
        constant_std=0.5,
        imread_flag=-1,
    ):
        return "LoadMultiViewImageFromFiles", LoadMultiViewImageFromFiles(
            to_float32=to_float32,
            project_pts_to_img_depth=project_pts_to_img_depth,
            cam_depth_range=cam_depth_range,
            constant_std=constant_std,
            imread_flag=imread_flag,
        )

    @register("ResizeImage")
    def build_resize_image(
        self,
        img_scale=None,
        multiscale_mode="range",
        ratio_range=None,
        keep_ratio=True,
        bbox_clip_border=True,
        backend="cv2",
        override=False,
    ):
        return "ResizeImage", ResizeImage(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range,
            keep_ratio=keep_ratio,
            bbox_clip_border=bbox_clip_border,
            backend=backend,
            override=override,
        )

    @register("NormalizeImage")
    def build_normalize_image(self, mean, std, to_rgb=True):
        return "NormalizeImage", NormalizeImage(mean=mean, std=std, to_rgb=to_rgb)

    @register("PadImage")
    def build_pad_image(self, size=None, size_divisor=None, pad_val=0):
        return "PadImage", PadImage(
            size=size, size_divisor=size_divisor, pad_val=pad_val
        )

    @register("SampleFilterByKey")
    def build_sample_filter_by_key(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "sample_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pts_filename",
            "transformation_3d_flow",
        ),
    ):
        return "SampleFilterByKey", SampleFilterByKey(keys=keys, meta_keys=meta_keys)

    @register("GetInferInput")
    def build_get_infer_input(self):
        return "GetInferInput", GetInferInput()

    def apply(self, input: Any, **kwargs) -> Iterator[Any]:
        """
        Do predicting with the input data and yields predictions.

        Args:
            input (Any): The input data to be predicted.

        Yields:
            Iterator[Any]: An iterator yielding prediction results.
        """

        try:
            for batch_data in self.batch_sampler(input):
                prediction = self.process(batch_data, **kwargs)
                prediction = PredictionWrap(prediction, len(batch_data))
                for idx in range(len(batch_data)):
                    yield self.result_class(prediction.get_by_idx(idx))
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(self.temp_dir)
