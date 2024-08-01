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
import json
import time
from pathlib import Path
import tarfile
import paddle

from ..base import BaseTrainer, BaseTrainDeamon
from ...utils.config import AttrDict
from .model_list import MODELS


class TSADTrainer(BaseTrainer):
    """ TS Anomaly Detection Model Trainer """
    entities = MODELS

    def build_deamon(self, config: AttrDict) -> "TSADTrainDeamon":
        """build deamon thread for saving training outputs timely

        Args:
            config (AttrDict): PaddleX pipeline config, which is loaded from pipeline yaml file.

        Returns:
            TSADTrainDeamon: the training deamon thread object for saving training outputs timely.
        """
        return TSADTrainDeamon(config)

    def train(self):
        """firstly, update and dump train config, then train model
        """
        # XXX: using super().train() instead when the train_hook() is supported.
        os.makedirs(self.global_config.output, exist_ok=True)
        self.update_config()
        self.dump_config()
        train_result = self.pdx_model.train(**self.get_train_kwargs())
        assert train_result.returncode == 0, f"Encountered an unexpected error({train_result.returncode}) in \
training!"

        self.make_tar_file()
        self.deamon.stop()

    def make_tar_file(self):
        """make tar file to package the training outputs
        """
        tar_path = Path(
            self.global_config.output) / "best_accuracy.pdparams.tar"
        with tarfile.open(tar_path, 'w') as tar:
            tar.add(self.global_config.output, arcname='best_accuracy.pdparams')

    def update_config(self):
        """update training config
        """
        self.pdx_config.update_dataset(self.global_config.dataset_dir,
                                       "TSADDataset")
        if self.train_config.input_len is not None:
            self.pdx_config.update_input_len(self.train_config.input_len)

        if self.train_config.time_col is not None:
            self.pdx_config.update_basic_info({
                'time_col': self.train_config.time_col
            })
        if self.train_config.feature_cols is not None:
            if isinstance(self.train_config.feature_cols, tuple):
                feature_cols = [str(item) for item in self.train_config.feature_cols] 
                self.pdx_config.update_basic_info({
                    'feature_cols': feature_cols
                })
            else:
                self.pdx_config.update_basic_info({
                    'feature_cols': self.train_config.feature_cols.split(',')
                })
        if self.train_config.label_col is not None:
            self.pdx_config.update_basic_info({
                'label_col': self.train_config.label_col
            })
        if self.train_config.freq is not None:
            try:
                self.train_config.freq = int(self.train_config.freq)
            except ValueError:
                pass
            self.pdx_config.update_basic_info({'freq': self.train_config.freq})
        if self.train_config.batch_size is not None:
            self.pdx_config.update_batch_size(self.train_config.batch_size)
        if self.train_config.learning_rate is not None:
            self.pdx_config.update_learning_rate(
                self.train_config.learning_rate)
        if self.train_config.epochs_iters is not None:
            self.pdx_config.update_epochs(self.train_config.epochs_iters)
        if self.global_config.output is not None:
            self.pdx_config.update_save_dir(self.global_config.output)

    def get_train_kwargs(self) -> dict:
        """get key-value arguments of model training function

        Returns:
            dict: the arguments of training function.
        """
        train_args = {"device": self.get_device()}
        if self.global_config.output is not None:
            train_args["save_dir"] = self.global_config.output
        return train_args


class TSADTrainDeamon(BaseTrainDeamon):
    """ DetTrainResultDemon """

    def get_watched_model(self):
        """ get the models needed to be watched """
        watched_models = []
        watched_models.append("best")
        return watched_models

    def update(self):
        """ update train result json """
        self.processing = True
        for i, result in enumerate(self.results):
            self.results[i] = self.update_result(result, self.train_outputs[i])
        self.save_json()
        self.processing = False

    def update_train_log(self, train_output):
        """ update train log """
        train_log_path = train_output / "train_ct.log"
        with open(train_log_path, 'w') as f:
            seconds = time.time()
            f.write('current training time: ' + time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(seconds)))
        f.close()
        return train_log_path

    def update_result(self, result, train_output):
        """ update every result """
        config = Path(train_output).joinpath("config.yaml")
        if not config.exists():
            return result

        result["config"] = config
        result["train_log"] = self.update_train_log(train_output)
        result["visualdl_log"] = self.update_vdl_log(train_output)
        result["label_dict"] = self.update_label_dict(train_output)
        self.update_models(result, train_output, "best")
        return result

    def update_models(self, result, train_output, model_key):
        """ update info of the models to be saved """
        pdparams = Path(train_output).joinpath("best_accuracy.pdparams.tar")
        if pdparams.exists():

            score = self.get_score(Path(train_output).joinpath("score.json"))
            result["models"][model_key] = {
                "score": "%.3f" % score,
                "pdparams": pdparams,
                "pdema": "",
                "pdopt": "",
                "pdstates": "",
                "inference_config": "",
                "pdmodel": "",
                "pdiparams": pdparams,
                "pdiparams.info": ""
            }

    def get_score(self, score_path):
        """ get the score by pdstates file """
        if not Path(score_path).exists():
            return 0
        return json.load(open(score_path, 'r'))["metric"]

    def get_best_ckp_prefix(self):
        """ get the prefix of the best checkpoint file """
        pass

    def get_epoch_id_by_pdparams_prefix(self):
        """ get the epoch_id by pdparams file """
        pass

    def get_ith_ckp_prefix(self):
        """ get the prefix of the epoch_id checkpoint file """
        pass

    def get_the_pdema_suffix(self):
        """ get the suffix of pdema file """
        pass

    def get_the_pdopt_suffix(self):
        """ get the suffix of pdopt file """
        pass

    def get_the_pdparams_suffix(self):
        """ get the suffix of pdparams file """
        pass

    def get_the_pdstates_suffix(self):
        """ get the suffix of pdstates file """
        pass
