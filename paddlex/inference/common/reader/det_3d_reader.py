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

import os
from typing import Generic

import numpy as np

from ...utils.benchmark import benchmark


class _EasyDict(dict):
    def __getattr__(self, key: str):
        if key in self:
            return self[key]
        return super().__getattr__(self, key)

    def __setattr__(self, key: str, value: Generic):
        self[key] = value


class SampleMeta(_EasyDict):

    # yapf: disable
    __slots__ = [
        "camera_intrinsic",
        # bgr or rgb
        "image_format",
        # pillow or cv2
        "image_reader",
        # chw or hwc
        "channel_order",
        # Unique ID of the sample
        "id",
        "time_lag",
        "ref_from_curr"
    ]
    # yapf: enable

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Sample(_EasyDict):
    """Data structure containing sample data information"""

    _VALID_MODALITIES = ["image", "lidar", "radar", "multimodal", "multiview"]

    def __init__(self, path: str, modality: str):
        if modality not in self._VALID_MODALITIES:
            raise ValueError(
                "Only modality {} is supported, but got {}".format(
                    self._VALID_MODALITIES, modality
                )
            )

        self.meta = SampleMeta()

        self.path = path
        self.data = None
        self.modality = modality.lower()

        self.bboxes_2d = None
        self.bboxes_3d = None
        self.labels = None

        self.sweeps = []
        self.attrs = None


@benchmark.timeit_with_options(name=None, is_read_operation=True)
class ReadNuscenesData:

    def __init__(
        self,
        dataset_root="",
        load_interval=1,
        noise_sensor_type="camera",
        drop_frames=False,
        drop_set=[0, "discrete"],
        modality="multimodal",
        extrinsics_noise=False,
        extrinsics_noise_type="single",
    ):

        self.load_interval = load_interval
        self.noise_data = None
        self.noise_sensor_type = noise_sensor_type
        self.drop_frames = drop_frames
        self.drop_ratio = drop_set[0]
        self.drop_type = drop_set[1]
        self.modality = modality
        self.extrinsics_noise = extrinsics_noise
        self.extrinsics_noise_type = extrinsics_noise_type
        self.dataset_root = dataset_root

    def get_data_info(self, info):
        """Get data info.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        sample = Sample(path=None, modality=self.modality)
        sample.sample_idx = info["token"]
        sample.meta.id = info["token"]
        sample.pts_filename = os.path.join(self.dataset_root, info["lidar_path"])
        sample.sweeps = info["sweeps"]
        sample.timestamp = info["timestamp"] / 1e6

        if self.noise_sensor_type == "lidar":
            if self.drop_frames:
                pts_filename = sample.pts_filename
                file_name = pts_filename.split("/")[-1]

                if self.noise_data[file_name]["noise"]["drop_frames"][self.drop_ratio][
                    self.drop_type
                ]["stuck"]:
                    replace_file = self.noise_data[file_name]["noise"]["drop_frames"][
                        self.drop_ratio
                    ][self.drop_type]["replace"]
                    if replace_file != "":
                        pts_filename = pts_filename.replace(file_name, replace_file)

                        sample.pts_filename = pts_filename
                        sample.sweeps = self.noise_data[replace_file]["mmdet_info"][
                            "sweeps"
                        ]
                        sample.timestamp = (
                            self.noise_data[replace_file]["mmdet_info"]["timestamp"]
                            / 1e6
                        )

        cam_orders = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
        ]
        if self.modality == "multiview" or self.modality == "multimodal":
            image_paths = []
            lidar2img_rts = []
            caminfos = []
            for cam_type in cam_orders:
                cam_info = info["cams"][cam_type]

                cam_data_path = cam_info["data_path"]
                cam_data_path = os.path.join(self.dataset_root, cam_data_path)
                file_name = cam_data_path.split("/")[-1]
                if self.noise_sensor_type == "camera":
                    if self.drop_frames:
                        if self.noise_data[file_name]["noise"]["drop_frames"][
                            self.drop_ratio
                        ][self.drop_type]["stuck"]:
                            replace_file = self.noise_data[file_name]["noise"][
                                "drop_frames"
                            ][self.drop_ratio][self.drop_type]["replace"]
                            if replace_file != "":
                                cam_data_path = cam_data_path.replace(
                                    file_name, replace_file
                                )

                image_paths.append(cam_data_path)
                # obtain lidar to image transformation matrix
                if self.extrinsics_noise:
                    sensor2lidar_rotation = self.noise_data[file_name]["noise"][
                        "extrinsics_noise"
                    ][f"{self.extrinsics_noise_type}_noise_sensor2lidar_rotation"]
                    sensor2lidar_translation = self.noise_data[file_name]["noise"][
                        "extrinsics_noise"
                    ][f"{self.extrinsics_noise_type}_noise_sensor2lidar_translation"]
                else:
                    sensor2lidar_rotation = cam_info["sensor2lidar_rotation"]
                    sensor2lidar_translation = cam_info["sensor2lidar_translation"]

                lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
                lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)
                caminfos.append(
                    {
                        "sensor2lidar_translation": sensor2lidar_translation,
                        "sensor2lidar_rotation": sensor2lidar_rotation,
                        "cam_intrinsic": cam_info["cam_intrinsic"],
                    }
                )

            sample.update(
                dict(
                    img_filename=image_paths, lidar2img=lidar2img_rts, caminfo=caminfos
                )
            )

        return sample

    def prepare_test_data(self, info):
        sample = self.get_data_info(info)
        sample = self.add_new_fields(sample)
        return sample

    def add_new_fields(self, sample):
        sample["img_fields"] = []
        sample["bbox3d_fields"] = []
        sample["pts_mask_fields"] = []
        sample["pts_seg_fields"] = []
        sample["bbox_fields"] = []
        sample["mask_fields"] = []
        sample["seg_fields"] = []
        return sample

    def __call__(self, batch_data):
        return [self.prepare_test_data(data_info) for data_info in batch_data]
