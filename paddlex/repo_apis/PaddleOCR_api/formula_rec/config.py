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
from typing import Union

import yaml

from ....utils.misc import abspath
from ...base import BaseConfig
from ..config_utils import load_config, merge_config


class FormulaRecConfig(BaseConfig):
    """Formula Recognition Config"""

    def update(self, dict_like_obj: list):
        """update self

        Args:
            dict_like_obj (dict): dict of pairs(key0.key1.idx.key2=value).
        """
        dict_ = merge_config(self.dict, dict_like_obj)
        self.reset_from_dict(dict_)

    def load(self, config_file_path: str):
        """load config from yaml file

        Args:
            config_file_path (str): the path of yaml file.

        Raises:
            TypeError: the content of yaml file `config_file_path` error.
        """
        dict_ = load_config(config_file_path)
        if not isinstance(dict_, dict):
            raise TypeError
        self.reset_from_dict(dict_)

    def dump(self, config_file_path: str):
        """dump self to yaml file

        Args:
            config_file_path (str): the path to save self as yaml file.
        """
        with open(config_file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.dict, f, default_flow_style=False, sort_keys=False)

    def update_dataset(
        self,
        dataset_path: str,
        dataset_type: str = None,
        *,
        train_list_path: str = None,
    ):
        """update dataset settings

        Args:
            dataset_path (str): the root path of dataset.
            dataset_type (str, optional): dataset type. Defaults to None.
            train_list_path (str, optional): the path of train dataset annotation file . Defaults to None.

        Raises:
            ValueError: the dataset_type error.
        """
        dataset_path = abspath(dataset_path)
        if dataset_type is None:
            dataset_type = "SimpleDataSet"
        if train_list_path:
            train_list_path = f"{train_list_path}"
        else:
            train_list_path = os.path.join(dataset_path, "train.txt")

        if dataset_type == "SimpleDataSet":
            _cfg = {
                "Train.dataset.name": dataset_type,
                "Train.dataset.data_dir": dataset_path,
                "Train.dataset.label_file_list": [train_list_path],
                "Eval.dataset.name": "SimpleDataSet",
                "Eval.dataset.data_dir": dataset_path,
                "Eval.dataset.label_file_list": [os.path.join(dataset_path, "val.txt")],
            }
            self.update(_cfg)
        elif dataset_type == "LaTeXOCRDataSet":
            _cfg = {
                "Train.dataset.name": dataset_type,
                "Train.dataset.data_dir": dataset_path,
                "Train.dataset.data": os.path.join(dataset_path, "latexocr_train.pkl"),
                "Train.dataset.label_file_list": [train_list_path],
                "Eval.dataset.name": dataset_type,
                "Eval.dataset.data_dir": dataset_path,
                "Eval.dataset.data": os.path.join(dataset_path, "latexocr_val.pkl"),
                "Eval.dataset.label_file_list": [os.path.join(dataset_path, "val.txt")],
                "Global.character_dict_path": os.path.join(dataset_path, "dict.txt"),
            }
            self.update(_cfg)
        else:
            raise ValueError(f"{repr(dataset_type)} is not supported.")

    def update_batch_size(self, batch_size: int, mode: str = "train"):
        """update batch size setting for SimpleDataSet

        Args:
            batch_size (int): the batch size number to set.
            mode (str, optional): the mode that to be set batch size, must be one of 'train', 'eval'
                Defaults to 'train'.

        Raises:
            ValueError: `mode` error.
        """

        if mode == "train":
            _cfg = {
                "Train.loader.batch_size_per_card": batch_size,
            }
        elif mode == "eval":
            _cfg = {
                "Eval.loader.batch_size_per_card": batch_size,
            }
        else:
            raise ValueError("The input `mode` should be train or eval.")
        self.update(_cfg)

    def update_batch_size_pair(self, batch_size: int, mode: str = "train"):
        """update batch size setting for LaTeXOCRDataSet

        Args:
            batch_size (int): the batch size number to set.
            mode (str, optional): the mode that to be set batch size, must be one of 'train', 'eval'
                Defaults to 'train'.

        Raises:
            ValueError: `mode` error.
        """

        if mode == "train":
            _cfg = {
                "Train.dataset.batch_size_per_pair": batch_size,
            }
        elif mode == "eval":
            _cfg = {"Eval.dataset.batch_size_per_pair": batch_size}
        else:
            raise ValueError("The input `mode` should be train or eval.")
        self.update(_cfg)

    def update_learning_rate(self, learning_rate: float):
        """update learning rate

        Args:
            learning_rate (float): the learning rate value to set.
        """
        _cfg = {
            "Optimizer.lr.learning_rate": learning_rate,
        }
        self.update(_cfg)

    def update_label_dict_path(self, dict_path: str):
        """update label dict file path

        Args:
            dict_path (str): the path to label dict file.
        """
        _cfg = {
            "Global.character_dict_path": abspath(dict_path),
        }
        self.update(_cfg)

    def update_warmup_epochs(self, warmup_epochs: int):
        """update warmup epochs

        Args:
            warmup_epochs (int): the warmup epochs value to set.
        """
        _cfg = {"Optimizer.lr.warmup_epoch": warmup_epochs}
        self.update(_cfg)

    def update_pretrained_weights(self, pretrained_model: str):
        """update pretrained weight path

        Args:
            pretrained_model (str): the local path or url of pretrained weight file to set.
        """
        if pretrained_model:
            if not pretrained_model.startswith(
                "http://"
            ) and not pretrained_model.startswith("https://"):
                pretrained_model = abspath(pretrained_model)
        self.update(
            {"Global.pretrained_model": pretrained_model, "Global.checkpoints": ""}
        )

    # TODO
    def update_class_path(self, class_path: str):
        """_summary_

        Args:
            class_path (str): _description_
        """
        self.update(
            {
                "PostProcess.class_path": class_path,
            }
        )

    def _update_amp(self, amp: Union[None, str]):
        """update AMP settings

        Args:
            amp (None | str): the AMP level if it is not None or `OFF`.
        """
        _cfg = {
            "Global.use_amp": amp is not None and amp != "OFF",
            "Global.amp_level": amp,
        }
        self.update(_cfg)

    def update_device(self, device: str):
        """update device setting

        Args:
            device (str): the running device to set
        """
        device = device.split(":")[0]
        default_cfg = {
            "Global.use_gpu": False,
            "Global.use_xpu": False,
            "Global.use_npu": False,
            "Global.use_mlu": False,
            "Global.use_gcu": False,
        }

        device_cfg = {
            "cpu": {},
            "gpu": {"Global.use_gpu": True},
            "xpu": {"Global.use_xpu": True},
            "mlu": {"Global.use_mlu": True},
            "npu": {"Global.use_npu": True},
            "gcu": {"Global.use_gcu": True},
        }
        default_cfg.update(device_cfg[device])
        self.update(default_cfg)

    def _update_epochs(self, epochs: int):
        """update epochs setting

        Args:
            epochs (int): the epochs number value to set
        """
        self.update({"Global.epoch_num": epochs})

    def _update_checkpoints(self, resume_path: Union[None, str]):
        """update checkpoint setting

        Args:
            resume_path (None | str): the resume training setting. if is `None`, train from scratch, otherwise,
                train from checkpoint file that path is `.pdparams` file.
        """
        self.update(
            {"Global.checkpoints": abspath(resume_path), "Global.pretrained_model": ""}
        )

    def _update_to_static(self, dy2st: bool):
        """update config to set dynamic to static mode

        Args:
            dy2st (bool): whether or not to use the dynamic to static mode.
        """
        self.update({"Global.to_static": dy2st})

    def _update_use_vdl(self, use_vdl: bool):
        """update config to set VisualDL

        Args:
            use_vdl (bool): whether or not to use VisualDL.
        """
        self.update({"Global.use_visualdl": use_vdl})

    def _update_output_dir(self, save_dir: str):
        """update output directory

        Args:
            save_dir (str): the path to save output.
        """
        self.update({"Global.save_model_dir": abspath(save_dir)})

    # TODO
    # def _update_log_interval(self, log_interval):
    #     self.update({'Global.print_batch_step': log_interval})

    def update_log_interval(self, log_interval: int):
        """update log interval(by steps)

        Args:
            log_interval (int): the log interval value to set.
        """
        self.update({"Global.print_batch_step": log_interval})

    # def _update_eval_interval(self, eval_start_step, eval_interval):
    #     self.update({
    #         'Global.eval_batch_step': [eval_start_step, eval_interval]
    #     })

    def update_log_ranks(self, device):
        """update log ranks

        Args:
            device (str): the running device to set
        """
        log_ranks = device.split(":")[1]
        self.update({"Global.log_ranks": log_ranks})

    def update_print_mem_info(self, print_mem_info: bool):
        """setting print memory info"""
        assert isinstance(print_mem_info, bool), "print_mem_info should be a bool"
        self.update({"Global.print_mem_info": f"{print_mem_info}"})

    def update_shared_memory(self, shared_memeory: bool):
        """update shared memory setting of train and eval dataloader

        Args:
            shared_memeory (bool): whether or not to use shared memory
        """
        assert isinstance(shared_memeory, bool), "shared_memeory should be a bool"
        _cfg = {
            "Train.loader.use_shared_memory": f"{shared_memeory}",
            "Train.loader.use_shared_memory": f"{shared_memeory}",
        }
        self.update(_cfg)

    def update_shuffle(self, shuffle: bool):
        """update shuffle setting of train and eval dataloader

        Args:
            shuffle (bool): whether or not to shuffle the data
        """
        assert isinstance(shuffle, bool), "shuffle should be a bool"
        _cfg = {
            f"Train.loader.shuffle": shuffle,
            f"Train.loader.shuffle": shuffle,
        }
        self.update(_cfg)

    def update_cal_metrics(self, cal_metrics: bool):
        """update calculate metrics setting
        Args:
            cal_metrics (bool): whether or not to calculate metrics during train
        """
        assert isinstance(cal_metrics, bool), "cal_metrics should be a bool"
        self.update({"Global.cal_metric_during_train": cal_metrics})

    def update_seed(self, seed: int):
        """update seed

        Args:
            seed (int): the random seed value to set
        """
        assert isinstance(seed, int), "seed should be an int"
        self.update({"Global.seed": seed})

    def _update_eval_interval_by_epoch(self, eval_interval):
        """update eval interval(by epoch)

        Args:
            eval_interval (int): the eval interval value to set.
        """
        self.update({"Global.eval_batch_epoch": eval_interval})

    def update_eval_interval(self, eval_interval: int, eval_start_step: int = 0):
        """update eval interval(by steps)

        Args:
            eval_interval (int): the eval interval value to set.
            eval_start_step (int, optional): step number from which the evaluation is enabled. Defaults to 0.
        """
        self._update_eval_interval(eval_start_step, eval_interval)

    def update_delimiter(self, delimiter: str, mode: str = "train"):
        """update_delimiter

        Args:
            delimiter (str): the dataset delimiter value to set.
            mode (str, optional): the mode that to be set batch size, must be one of 'train', 'eval'
                Defaults to 'train'.
        """
        delimiter = delimiter.encode().decode("unicode_escape")

        if mode == "train":
            _cfg = {"Train.dataset.delimiter": delimiter}
        elif mode == "eval":
            _cfg = {"Eval.dataset.delimiter": delimiter}
        else:
            raise ValueError("The input `mode` should be train or eval.")
        self.update(_cfg)

    def _update_save_interval(self, save_interval: int):
        """update save interval(by steps)

        Args:
            save_interval (int): the save interval value to set.
        """
        self.update({"Global.save_epoch_step": save_interval})

    def update_save_interval(self, save_interval: int):
        """update save interval(by steps)

        Args:
            save_interval (int): the save interval value to set.
        """
        self._update_save_interval(save_interval)

    def _update_infer_img(self, infer_img: str, infer_list: str = None):
        """update image list to be inferred

        Args:
            infer_img (str): path to the image file to be inferred. It would be ignored when `infer_list` is be set.
            infer_list (str, optional): path to the .txt file containing the paths to image to be inferred.
                Defaults to None.
        """
        if infer_list:
            self.update({"Global.infer_list": infer_list})
        self.update({"Global.infer_img": infer_img})

    def _update_save_inference_dir(self, save_inference_dir: str):
        """update the directory saving infer outputs

        Args:
            save_inference_dir (str): the directory saving infer outputs.
        """
        self.update({"Global.save_inference_dir": abspath(save_inference_dir)})

    def _update_save_res_path(self, save_res_path: str):
        """update the .txt file path saving OCR model inference result

        Args:
            save_res_path (str): the .txt file path saving OCR model inference result.
        """
        self.update({"Global.save_res_path": abspath(save_res_path)})

    def update_num_workers(
        self, num_workers: int, modes: Union[str, list] = ["train", "eval"]
    ):
        """update workers number of train or eval dataloader

        Args:
            num_workers (int): the value of train and eval dataloader workers number to set.
            modes (str | [list], optional): mode. Defaults to ['train', 'eval'].

        Raises:
            ValueError: mode error. The `mode` should be `train`, `eval` or `['train', 'eval']`.
        """
        if not isinstance(modes, list):
            modes = [modes]
        for mode in modes:
            if not mode in ("train", "eval"):
                raise ValueError
            if mode == "train":
                self["Train"]["loader"]["num_workers"] = num_workers
            else:
                self["Eval"]["loader"]["num_workers"] = num_workers

    def _get_model_type(self) -> str:
        """get model type

        Returns:
            str: model type, i.e. `Architecture.algorithm` or `Architecture.Models.Student.algorithm`.
        """
        if "Models" in self.dict["Architecture"]:
            return self.dict["Architecture"]["Models"]["Student"]["algorithm"]

        return self.dict["Architecture"]["algorithm"]

    def get_epochs_iters(self) -> int:
        """get epochs

        Returns:
            int: the epochs value, i.e., `Global.epochs` in config.
        """
        return self.dict["Global"]["epoch_num"]

    def get_learning_rate(self) -> float:
        """get learning rate

        Returns:
            float: the learning rate value, i.e., `Optimizer.lr.learning_rate` in config.
        """
        return self.dict["Optimizer"]["lr"]["learning_rate"]

    def get_batch_size(self, mode="train") -> int:
        """get batch size

        Args:
            mode (str, optional): the mode that to be get batch size value, must be one of 'train', 'eval', 'test'.
                Defaults to 'train'.

        Returns:
            int: the batch size value of `mode`, i.e., `DataLoader.{mode}.sampler.batch_size` in config.
        """
        return self.dict["Train"]["loader"]["batch_size_per_card"]

    def get_qat_epochs_iters(self) -> int:
        """get qat epochs

        Returns:
            int: the epochs value.
        """
        return self.get_epochs_iters()

    def get_qat_learning_rate(self) -> float:
        """get qat learning rate

        Returns:
            float: the learning rate value.
        """
        return self.get_learning_rate()

    def get_label_dict_path(self) -> str:
        """get label dict file path

        Returns:
            str: the label dict file path, i.e., `Global.character_dict_path` in config.
        """
        return self.dict["Global"]["character_dict_path"]

    def _get_dataset_root(self) -> str:
        """get root directory of dataset, i.e. `DataLoader.Train.dataset.data_dir`

        Returns:
            str: the root directory of dataset
        """
        return self.dict["Train"]["dataset"]["data_dir"]

    def _get_infer_shape(self) -> str:
        """get resize scale of ResizeImg operation in the evaluation

        Returns:
            str: resize scale, i.e. `Eval.dataset.transforms.ResizeImg.image_shape`
        """
        size = None
        transforms = self.dict["Eval"]["dataset"]["transforms"]
        for op in transforms:
            op_name = list(op)[0]
            if "ResizeImg" in op_name:
                size = op[op_name]["image_shape"]
        return ",".join([str(x) for x in size])

    def get_train_save_dir(self) -> str:
        """get the directory to save output

        Returns:
            str: the directory to save output
        """
        return self["Global"]["save_model_dir"]

    def get_predict_save_dir(self) -> str:
        """get the directory to save output in predicting

        Returns:
            str: the directory to save output
        """
        return os.path.dirname(self["Global"]["save_res_path"])
