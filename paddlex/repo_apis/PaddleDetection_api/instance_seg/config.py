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

from typing import List

from ....utils.misc import abspath
from ..object_det.config import DetConfig


class InstanceSegConfig(DetConfig):
    """InstanceSegConfig"""

    def load(self, config_path: str):
        """load the config from config file

        Args:
            config_path (str): the config file path.
        """
        dict_ = self.load_config_literally(config_path)
        self.reset_from_dict(dict_)

    def dump(self, config_path: str):
        """dump the config

        Args:
            config_path (str): the path to save dumped config.
        """
        self.dump_literal_config(config_path, self._dict)

    def update(self, dict_like_obj: list):
        """update self from dict

        Args:
            dict_like_obj (list): the list of pairs that contain key and value.
        """
        self.update_from_dict(dict_like_obj, self._dict)

    def update_dataset(
        self,
        dataset_path: str,
        dataset_type: str = None,
        *,
        data_fields: List[str] = None,
        image_dir: str = "images",
        train_anno_path: str = "annotations/instance_train.json",
        val_anno_path: str = "annotations/instance_val.json",
        test_anno_path: str = "annotations/instance_val.json",
    ):
        """update dataset settings

        Args:
            dataset_path (str): the root path fo dataset.
            dataset_type (str, optional): the dataset type. Defaults to None.
            data_fields (list[str], optional): the data fields in dataset. Defaults to None.
            image_dir (str, optional): the images file directory that relative to `dataset_path`. Defaults to "images".
            train_anno_path (str, optional): the train annotations file that relative to `dataset_path`.
                Defaults to "annotations/instance_train.json".
            val_anno_path (str, optional): the validation annotations file that relative to `dataset_path`.
                Defaults to "annotations/instance_val.json".
            test_anno_path (str, optional): the test annotations file that relative to `dataset_path`.
                Defaults to "annotations/instance_val.json".

        Raises:
            ValueError: the `dataset_type` error.
        """
        dataset_path = abspath(dataset_path)

        if dataset_type is None:
            dataset_type = "COCOInstSegDataset"
        if dataset_type == "COCOInstSegDataset":
            ds_cfg = self._make_dataset_config(
                dataset_path,
                data_fields,
                image_dir,
                train_anno_path,
                val_anno_path,
                test_anno_path,
            )
            self.set_val("metric", "COCO")
        else:
            raise ValueError(f"{repr(dataset_type)} is not supported.")
        self.update(ds_cfg)

    def _make_dataset_config(
        self,
        dataset_root_path: str,
        data_fields: List[str,] = None,
        image_dir: str = "images",
        train_anno_path: str = "annotations/instance_train.json",
        val_anno_path: str = "annotations/instance_val.json",
        test_anno_path: str = "annotations/instance_val.json",
    ) -> dict:
        """construct the dataset config that meets the format requirements

        Args:
            dataset_root_path (str): the root directory of dataset.
            data_fields (list[str,], optional): the data field. Defaults to None.
            image_dir (str, optional): _description_. Defaults to "images".
            train_anno_path (str, optional): _description_. Defaults to "annotations/instance_train.json".
            val_anno_path (str, optional): _description_. Defaults to "annotations/instance_val.json".
            test_anno_path (str, optional): _description_. Defaults to "annotations/instance_val.json".

        Returns:
            dict: the dataset config.
        """
        data_fields = (
            ["image", "gt_bbox", "gt_poly", "gt_class", "is_crowd"]
            if data_fields is None
            else data_fields
        )

        return {
            "TrainDataset": {
                "name": "COCOInstSegDataset",
                "image_dir": image_dir,
                "anno_path": train_anno_path,
                "dataset_dir": dataset_root_path,
                "data_fields": data_fields,
            },
            "EvalDataset": {
                "name": "COCOInstSegDataset",
                "image_dir": image_dir,
                "anno_path": val_anno_path,
                "dataset_dir": dataset_root_path,
            },
            "TestDataset": {
                "name": "ImageFolder",
                "anno_path": test_anno_path,
                "dataset_dir": dataset_root_path,
            },
        }

    def update_ema(
        self,
        use_ema: bool,
        ema_decay: float = 0.9999,
        ema_decay_type: str = "exponential",
        ema_filter_no_grad: bool = True,
    ):
        """update EMA setting

        Args:
            use_ema (bool): whether or not to use EMA
            ema_decay (float, optional): value of EMA decay. Defaults to 0.9999.
            ema_decay_type (str, optional): type of EMA decay. Defaults to "exponential".
            ema_filter_no_grad (bool, optional): whether or not to filter the parameters that been set to stop gradient
                and are not batch norm parameters. Defaults to True.
        """
        self.update(
            {
                "use_ema": use_ema,
                "ema_decay": ema_decay,
                "ema_decay_type": ema_decay_type,
                "ema_filter_no_grad": ema_filter_no_grad,
            }
        )

    def update_learning_rate(self, learning_rate: float):
        """update learning rate

        Args:
            learning_rate (float): the learning rate value to set.
        """
        self.LearningRate["base_lr"] = learning_rate

    def update_warmup_steps(self, warmup_steps: int):
        """update warmup steps

        Args:
            warmup_steps (int): the warmup steps value to set.
        """
        schedulers = self.LearningRate["schedulers"]
        for sch in schedulers:
            key = "name" if "name" in sch else "_type_"
            if sch[key] == "LinearWarmup":
                sch["steps"] = warmup_steps
                sch["epochs_first"] = False

    def update_warmup_enable(self, use_warmup: bool):
        """whether or not to enable learning rate warmup

        Args:
            use_warmup (bool): `True` is enable learning rate warmup and `False` is disable.
        """
        schedulers = self.LearningRate["schedulers"]
        for sch in schedulers:
            if "use_warmup" in sch:
                sch["use_warmup"] = use_warmup

    def update_cossch_epoch(self, max_epochs: int):
        """update max epochs of cosine learning rate scheduler

        Args:
            max_epochs (int): the max epochs value.
        """
        schedulers = self.LearningRate["schedulers"]
        for sch in schedulers:
            key = "name" if "name" in sch else "_type_"
            if sch[key] == "CosineDecay":
                sch["max_epochs"] = max_epochs

    def update_milestone(self, milestones: List[int]):
        """update milstone of `PiecewiseDecay` learning scheduler

        Args:
            milestones (list[int]): the list of milestone values of `PiecewiseDecay` learning scheduler.
        """
        schedulers = self.LearningRate["schedulers"]
        for sch in schedulers:
            key = "name" if "name" in sch else "_type_"
            if sch[key] == "PiecewiseDecay":
                sch["milestones"] = milestones

    def update_batch_size(self, batch_size: int, mode: str = "train"):
        """update batch size setting

        Args:
            batch_size (int): the batch size number to set.
            mode (str, optional): the mode that to be set batch size, must be one of 'train', 'eval', 'test'.
                Defaults to 'train'.

        Raises:
            ValueError: mode error.
        """
        assert mode in (
            "train",
            "eval",
            "test",
        ), "mode ({}) should be train, eval or test".format(mode)
        if mode == "train":
            self.TrainReader["batch_size"] = batch_size
        elif mode == "eval":
            self.EvalReader["batch_size"] = batch_size
        else:
            self.TestReader["batch_size"] = batch_size

    def update_epochs(self, epochs: int):
        """update epochs setting

        Args:
            epochs (int): the epochs number value to set
        """
        self.update({"epoch": epochs})

    def update_device(self, device_type: str):
        """update device setting

        Args:
            device (str): the running device to set
        """
        if device_type.lower() == "gpu":
            self["use_gpu"] = True
        elif device_type.lower() == "xpu":
            self["use_xpu"] = True
            self["use_gpu"] = False
        elif device_type.lower() == "npu":
            self["use_npu"] = True
            self["use_gpu"] = False
        elif device_type.lower() == "mlu":
            self["use_mlu"] = True
            self["use_gpu"] = False
        elif device_type.lower() == "gcu":
            self["use_gcu"] = True
            self["use_gpu"] = False
        else:
            assert device_type.lower() == "cpu"
            self["use_gpu"] = False

    def update_save_dir(self, save_dir: str):
        """update directory to save outputs

        Args:
            save_dir (str): the directory to save outputs.
        """
        self["save_dir"] = abspath(save_dir)

    def update_log_interval(self, log_interval: int):
        """update log interval(steps)

        Args:
            log_interval (int): the log interval value to set.
        """
        self.update({"log_iter": log_interval})

    def update_eval_interval(self, eval_interval: int):
        """update eval interval(epochs)

        Args:
            eval_interval (int): the eval interval value to set.
        """
        self.update({"snapshot_epoch": eval_interval})

    def update_save_interval(self, save_interval: int):
        """update eval interval(epochs)

        Args:
            save_interval (int): the save interval value to set.
        """
        self.update({"snapshot_epoch": save_interval})

    def update_log_ranks(self, device):
        """update log ranks

        Args:
            device (str): the running device to set
        """
        log_ranks = device.split(":")[1]
        self.update({"log_ranks": log_ranks})

    def update_weights(self, weight_path: str):
        """update model weight

        Args:
            weight_path (str): the path to weight file of model.
        """
        self["weights"] = weight_path

    def update_pretrained_weights(self, pretrain_weights: str):
        """update pretrained weight path

        Args:
            pretrained_model (str): the local path or url of pretrained weight file to set.
        """
        if not pretrain_weights.startswith(
            "http://"
        ) and not pretrain_weights.startswith("https://"):
            pretrain_weights = abspath(pretrain_weights)
        self["pretrain_weights"] = pretrain_weights

    def update_num_class(self, num_classes: int):
        """update classes number

        Args:
            num_classes (int): the classes number value to set.
        """
        self["num_classes"] = num_classes

    def update_random_size(self, randomsize):
        """update `target_size` of `BatchRandomResize` op in TestReader

        Args:
            randomsize (list[list[int, int]]): the list of different size scales.
        """
        self.TestReader["batch_transforms"]["BatchRandomResize"][
            "target_size"
        ] = randomsize

    def update_num_workers(self, num_workers: int):
        """update workers number of train and eval dataloader

        Args:
            num_workers (int): the value of train and eval dataloader workers number to set.
        """
        self["worker_num"] = num_workers

    def update_static_assigner_epochs(self, static_assigner_epochs: int):
        """update static assigner epochs value

        Args:
            static_assigner_epochs (int): the value of static assigner epochs
        """
        assert "PicoHeadV2" in self
        self.PicoHeadV2["static_assigner_epoch"] = static_assigner_epochs

    def get_epochs_iters(self) -> int:
        """get epochs

        Returns:
            int: the epochs value, i.e., `Global.epochs` in config.
        """
        return self.epoch

    def get_log_interval(self) -> int:
        """get log interval(steps)

        Returns:
            int: the log interval value, i.e., `Global.print_batch_step` in config.
        """
        self.log_iter

    def get_eval_interval(self) -> int:
        """get eval interval(epochs)

        Returns:
            int: the eval interval value, i.e., `Global.eval_interval` in config.
        """
        self.snapshot_epoch

    def get_save_interval(self) -> int:
        """get save interval(epochs)

        Returns:
            int: the save interval value, i.e., `Global.save_interval` in config.
        """
        self.snapshot_epoch

    def get_learning_rate(self) -> float:
        """get learning rate

        Returns:
            float: the learning rate value, i.e., `Optimizer.lr.learning_rate` in config.
        """
        return self.LearningRate["base_lr"]

    def get_batch_size(self, mode="train") -> int:
        """get batch size

        Args:
            mode (str, optional): the mode that to be get batch size value, must be one of 'train', 'eval', 'test'.
                Defaults to 'train'.

        Returns:
            int: the batch size value of `mode`, i.e., `DataLoader.{mode}.sampler.batch_size` in config.
        """
        if mode == "train":
            return self.TrainReader["batch_size"]
        elif mode == "eval":
            return self.EvalReader["batch_size"]
        elif mode == "test":
            return self.TestReader["batch_size"]
        else:
            raise (f"Unknown mode: {repr(mode)}")

    def get_qat_epochs_iters(self) -> int:
        """get qat epochs

        Returns:
            int: the epochs value.
        """
        return self.epoch // 2.0

    def get_qat_learning_rate(self) -> float:
        """get qat learning rate

        Returns:
            float: the learning rate value.
        """
        return self.LearningRate["base_lr"] // 2.0

    def get_train_save_dir(self) -> str:
        """get the directory to save output

        Returns:
            str: the directory to save output
        """
        return self.save_dir
