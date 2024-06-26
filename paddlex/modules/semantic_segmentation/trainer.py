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


import os
import glob
from pathlib import Path
import paddle

from ..base import BaseTrainer, BaseTrainDeamon
from ...utils.config import AttrDict
from .model_list import MODELS


class SegTrainer(BaseTrainer):
    """ Semantic Segmentation Model Trainer """
    entities = MODELS

    def build_deamon(self, config: AttrDict) -> "SegTrainDeamon":
        """build deamon thread for saving training outputs timely

        Args:
            config (AttrDict): PaddleX pipeline config, which is loaded from pipeline yaml file.

        Returns:
            SegTrainDeamon: the training deamon thread object for saving training outputs timely.
        """
        return SegTrainDeamon(config)

    def update_config(self):
        """update training config
        """
        self.pdx_config.update_dataset(self.global_config.dataset_dir,
                                       "SegDataset")
        if self.train_config.num_classes is not None:
            self.pdx_config.update_num_classes(self.train_config.num_classes)
        if self.train_config.pretrain_weight_path and self.train_config.pretrain_weight_path != "":
            self.pdx_config.update_pretrained_weights(
                self.train_config.pretrain_weight_path, is_backbone=True)

    def get_train_kwargs(self) -> dict:
        """get key-value arguments of model training function

        Returns:
            dict: the arguments of training function.
        """
        train_args = {"device": self.get_device()}
        # XXX:
        os.environ.pop("FLAGS_npu_jit_compile", None)
        if self.train_config.batch_size is not None:
            train_args["batch_size"] = self.train_config.batch_size
        if self.train_config.learning_rate is not None:
            train_args["learning_rate"] = self.train_config.learning_rate
        if self.train_config.epochs_iters is not None:
            train_args["epochs_iters"] = self.train_config.epochs_iters
        if self.train_config.resume_path is not None and self.train_config.resume_path != "":
            train_args["resume_path"] = self.train_config.resume_path
        if self.global_config.output is not None:
            train_args["save_dir"] = self.global_config.output
        if self.train_config.log_interval:
            train_args["log_iters"] = self.train_config.log_interval
        if self.train_config.eval_interval:
            train_args["do_eval"] = True
            train_args["save_interval"] = self.train_config.eval_interval
        return train_args


class SegTrainDeamon(BaseTrainDeamon):
    """ SegTrainResultDemon """
    last_k = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_the_pdparams_suffix(self):
        """ get the suffix of pdparams file """
        return "pdparams"

    def get_the_pdema_suffix(self):
        """ get the suffix of pdema file """
        return "pdema"

    def get_the_pdopt_suffix(self):
        """ get the suffix of pdopt file """
        return "pdopt"

    def get_the_pdstates_suffix(self):
        """ get the suffix of pdstates file """
        return "pdstates"

    def get_ith_ckp_prefix(self, epoch_id):
        """ get the prefix of the epoch_id checkpoint file """
        return f"iter_{epoch_id}/model"

    def get_best_ckp_prefix(self):
        """ get the prefix of the best checkpoint file """
        return "best_model/model"

    def get_score(self, pdstates_path):
        """ get the score by pdstates file """
        if not Path(pdstates_path).exists():
            return 0
        return paddle.load(pdstates_path)["mIoU"]

    def get_epoch_id_by_pdparams_prefix(self, pdparams_dir):
        """ get the epoch_id by pdparams file """
        return int(pdparams_dir.parent.name.split("_")[-1])

    def update_result(self, result, train_output):
        """ update every result """
        train_output = Path(train_output).resolve()
        config_path = train_output.joinpath("config.yaml").resolve()
        if not config_path.exists():
            return result

        model_name = result["model_name"]
        if model_name in self.config_recorder and self.config_recorder[
                model_name] != config_path:
            result["models"] = self.init_model_pkg()
        result["config"] = config_path
        self.config_recorder[model_name] = config_path

        result["visualdl_log"] = self.update_vdl_log(train_output)
        result["label_dict"] = self.update_label_dict(train_output)

        model = self.get_model(result["model_name"], config_path)

        params_path_list = list(
            train_output.glob(".".join([
                self.get_ith_ckp_prefix("[0-9]*"), self.get_the_pdparams_suffix(
                )
            ])))
        iter_ids = []
        for params_path in params_path_list:
            iter_id = self.get_epoch_id_by_pdparams_prefix(params_path)
            iter_ids.append(iter_id)
        iter_ids.sort()
        # TODO(gaotingquan): how to avoid that the latest ckp files is being saved
        # epoch_ids = epoch_ids[:-1]
        for i in range(1, self.last_k + 1):
            if len(iter_ids) < i:
                break
            self.update_models(result, model, train_output, f"last_{i}",
                               self.get_ith_ckp_prefix(iter_ids[-i]))
        self.update_models(result, model, train_output, "best",
                           self.get_best_ckp_prefix())
        return result

    def update_models(self, result, model, train_output, model_key, ckp_prefix):
        """ update info of the models to be saved """
        pdparams = train_output.joinpath(".".join(
            [ckp_prefix, self.get_the_pdparams_suffix()]))
        if pdparams.exists():
            recorder_key = f"{train_output.name}_{model_key}"
            if model_key != "best" and recorder_key in self.model_recorder and self.model_recorder[
                    recorder_key] == pdparams:
                return

            self.model_recorder[recorder_key] = pdparams

            pdema = ""
            pdema_suffix = self.get_the_pdema_suffix()
            if pdema_suffix:
                pdema = pdparams.parents[1].joinpath(".".join(
                    [ckp_prefix, pdema_suffix]))
                if not pdema.exists():
                    pdema = ""

            pdopt = ""
            pdopt_suffix = self.get_the_pdopt_suffix()
            if pdopt_suffix:
                pdopt = pdparams.parents[1].joinpath(".".join(
                    [ckp_prefix, pdopt_suffix]))
                if not pdopt.exists():
                    pdopt = ""

            pdstates = ""
            pdstates_suffix = self.get_the_pdstates_suffix()
            if pdstates_suffix:
                pdstates = pdparams.parents[1].joinpath(".".join(
                    [ckp_prefix, pdstates_suffix]))
                if not pdstates.exists():
                    pdstates = ""

            score = self.get_score(Path(pdstates).resolve().as_posix())

            result["models"][model_key] = {
                "score": score,
                "pdparams": pdparams,
                "pdema": pdema,
                "pdopt": pdopt,
                "pdstates": pdstates
            }

            self.update_inference_model(model, pdparams,
                                        train_output.joinpath(f"{ckp_prefix}"),
                                        result["models"][model_key])
