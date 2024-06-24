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
import sys
import time
import json
import traceback
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from ..build_model import build_model
from ....utils.file_interface import write_json_file
from ....utils import logging


def try_except_decorator(func):
    """ try-except """

    def wrap(self, *args, **kwargs):
        try:
            func(self, *args, **kwargs)
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            self.save_json()
            traceback.print_exception(exc_type, exc_value, exc_tb)
        finally:
            self.processing = False

    return wrap


class BaseTrainDeamon(ABC):
    """ BaseTrainResultDemon """
    update_interval = 600
    last_k = 5

    def __init__(self, global_config):
        """ init """
        self.global_config = global_config
        self.init_pre_hook()
        self.output_dir = global_config.output
        self.train_outputs = self.get_train_outputs()
        self.save_paths = self.get_save_paths()
        self.results = self.init_train_result()
        self.save_json()
        self.models = {}
        self.init_post_hook()

        self.config_recorder = {}
        self.model_recorder = {}
        self.processing = False
        self.start()

    def init_train_result(self):
        """ init train result structure """
        model_names = self.init_model_names()
        configs = self.init_configs()
        train_log = self.init_train_log()
        vdl = self.init_vdl_log()

        results = []
        for i, model_name in enumerate(model_names):
            results.append({
                "model_name": model_name,
                "done_flag": False,
                "config": configs[i],
                "label_dict": "",
                "train_log": train_log,
                "visualdl_log": vdl,
                "models": self.init_model_pkg()
            })
        return results

    def get_save_names(self):
        """ get names to save """
        return ["train_result.json"]

    def get_train_outputs(self):
        """ get training outputs dir """
        return [Path(self.output_dir)]

    def init_model_names(self):
        """ get models name """
        return [self.global_config.model]

    def get_save_paths(self):
        """ get the path to save train_result.json """
        return [
            Path(self.output_dir, save_name)
            for save_name in self.get_save_names()
        ]

    def init_configs(self):
        """ get the init value of config field in result """
        return [""] * len(self.init_model_names())

    def init_train_log(self):
        """ get train log """
        return ""

    def init_vdl_log(self):
        """ get visualdl log """
        return ""

    def init_model_pkg(self):
        """ get model package """
        init_content = self.init_model_content()
        model_pkg = {}

        for pkg in self.get_watched_model():
            model_pkg[pkg] = init_content
        return model_pkg

    def normlize_path(self, dict_obj, relative_to):
        """ normlize path to string type path relative to the output_dir """
        for key in dict_obj:
            if isinstance(dict_obj[key], dict):
                self.normlize_path(dict_obj[key], relative_to)
            if isinstance(dict_obj[key], Path):
                dict_obj[key] = dict_obj[key].resolve().relative_to(
                    relative_to.resolve()).as_posix()

    def save_json(self):
        """ save result to json """
        for i, result in enumerate(self.results):
            self.save_paths[i].parent.mkdir(parents=True, exist_ok=True)
            self.normlize_path(result, relative_to=self.save_paths[i].parent)
            write_json_file(result, self.save_paths[i], indent=2)

    def start(self):
        """ start deamon thread """
        self.exit = False
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def stop_hook(self):
        """ hook befor stop """
        for result in self.results:
            result["done_flag"] = True
            self.update()

    def stop(self):
        """ stop self """
        self.exit = True
        while True:
            if not self.processing:
                self.stop_hook()
                break
            time.sleep(60)

    def run(self):
        """ main function """
        while not self.exit:
            self.update()
            if self.exit:
                break
            time.sleep(self.update_interval)

    def update_train_log(self, train_output):
        """ update train log """
        train_log_path = train_output / "train.log"
        if train_log_path.exists():
            return train_log_path

    def update_vdl_log(self, train_output):
        """ update visualdl log """
        vdl_path = list(train_output.glob("vdlrecords*log"))
        if len(vdl_path) >= 1:
            return vdl_path[0]

    def update_label_dict(self, train_output):
        """ update label dict """
        dict_path = train_output.joinpath("label_dict.txt")
        if not dict_path.exists():
            return ""
        return dict_path

    @try_except_decorator
    def update(self):
        """ update train result json """
        self.processing = True
        for i in range(len(self.results)):
            self.results[i] = self.update_result(self.results[i],
                                                 self.train_outputs[i])
        self.save_json()
        self.processing = False

    def get_model(self, model_name, config_path):
        """ initialize the model """
        if model_name not in self.models:
            config, model = build_model(
                model_name,
                device=self.global_config.device,
                config_path=config_path)
            self.models[model_name] = model
        return self.models[model_name]

    def get_watched_model(self):
        """ get the models needed to be watched """
        watched_models = [f"last_{i}" for i in range(1, self.last_k + 1)]
        watched_models.append("best")
        return watched_models

    def init_model_content(self):
        """ get model content structure """
        return {
            "score": "",
            "pdparams": "",
            "pdema": "",
            "pdopt": "",
            "pdstates": "",
            "inference_config": "",
            "pdmodel": "",
            "pdiparams": "",
            "pdiparams.info": ""
        }

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

        result["train_log"] = self.update_train_log(train_output)
        result["visualdl_log"] = self.update_vdl_log(train_output)
        result["label_dict"] = self.update_label_dict(train_output)

        model = self.get_model(result["model_name"], config_path)

        params_path_list = list(
            train_output.glob(".".join([
                self.get_ith_ckp_prefix("[0-9]*"), self.get_the_pdparams_suffix(
                )
            ])))
        epoch_ids = []
        for params_path in params_path_list:
            epoch_id = self.get_epoch_id_by_pdparams_prefix(params_path.stem)
            epoch_ids.append(epoch_id)
        epoch_ids.sort()
        # TODO(gaotingquan): how to avoid that the latest ckp files is being saved
        # epoch_ids = epoch_ids[:-1]
        for i in range(1, self.last_k + 1):
            if len(epoch_ids) < i:
                break
            self.update_models(result, model, train_output, f"last_{i}",
                               self.get_ith_ckp_prefix(epoch_ids[-i]))
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
                pdema = pdparams.parent.joinpath(".".join(
                    [ckp_prefix, pdema_suffix]))
                if not pdema.exists():
                    pdema = ""

            pdopt = ""
            pdopt_suffix = self.get_the_pdopt_suffix()
            if pdopt_suffix:
                pdopt = pdparams.parent.joinpath(".".join(
                    [ckp_prefix, pdopt_suffix]))
                if not pdopt.exists():
                    pdopt = ""

            pdstates = ""
            pdstates_suffix = self.get_the_pdstates_suffix()
            if pdstates_suffix:
                pdstates = pdparams.parent.joinpath(".".join(
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

    def update_inference_model(self, model, weight_path, export_save_dir,
                               result_the_model):
        """ update inference model """
        export_save_dir.mkdir(parents=True, exist_ok=True)
        export_result = model.export(
            weight_path=weight_path, save_dir=export_save_dir)

        if export_result.returncode == 0:
            inference_config = export_save_dir.joinpath("inference.yml")
            if not inference_config.exists():
                inference_config = ""
            pdmodel = export_save_dir.joinpath("inference.pdmodel")
            pdiparams = export_save_dir.joinpath("inference.pdiparams")
            pdiparams_info = export_save_dir.joinpath(
                "inference.pdiparams.info")
        else:
            inference_config = ""
            pdmodel = ""
            pdiparams = ""
            pdiparams_info = ""

        result_the_model["inference_config"] = inference_config
        result_the_model["pdmodel"] = pdmodel
        result_the_model["pdiparams"] = pdiparams
        result_the_model["pdiparams.info"] = pdiparams_info

    def init_pre_hook(self):
        """ hook func that would be called befor init """
        pass

    def init_post_hook(self):
        """ hook func that would be called after init """
        pass

    @abstractmethod
    def get_the_pdparams_suffix(self):
        """ get the suffix of pdparams file """
        raise NotImplementedError

    @abstractmethod
    def get_the_pdema_suffix(self):
        """ get the suffix of pdema file """
        raise NotImplementedError

    @abstractmethod
    def get_the_pdopt_suffix(self):
        """ get the suffix of pdopt file """
        raise NotImplementedError

    @abstractmethod
    def get_the_pdstates_suffix(self):
        """ get the suffix of pdstates file """
        raise NotImplementedError

    @abstractmethod
    def get_ith_ckp_prefix(self, epoch_id):
        """ get the prefix of the epoch_id checkpoint file """
        raise NotImplementedError

    @abstractmethod
    def get_best_ckp_prefix(self):
        """ get the prefix of the best checkpoint file """
        raise NotImplementedError

    @abstractmethod
    def get_score(self, pdstates_path):
        """ get the score by pdstates file """
        raise NotImplementedError

    @abstractmethod
    def get_epoch_id_by_pdparams_prefix(self, pdparams_prefix):
        """ get the epoch_id by pdparams file """
        raise NotImplementedError
