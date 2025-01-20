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

from abc import abstractmethod
from copy import deepcopy

from .inference import create_predictor, PaddlePredictorOption
from .modules import (
    build_dataset_checker,
    build_trainer,
    build_evaluater,
    build_exportor,
)
from .utils.flags import NEW_PREDICTOR


# TODO(gaotingquan): support _ModelBasedConfig
def create_model(model_name, model_dir=None, *args, **kwargs):
    return _ModelBasedInference(
        model_name=model_name, model_dir=model_dir, *args, **kwargs
    )


class _BaseModel:
    def check_dataset(self, *args, **kwargs):
        raise Exception("check_dataset is not supported!")

    def train(self, *args, **kwargs):
        raise Exception("train is not supported!")

    def evaluate(self, *args, **kwargs):
        raise Exception("evaluate is not supported!")

    def export(self, *args, **kwargs):
        raise Exception("export is not supported!")

    def predict(self, *args, **kwargs):
        raise Exception("predict is not supported!")

    def set_predict(self, *args, **kwargs):
        raise Exception("set_predict is not supported!")

    def __call__(self, *args, **kwargs):
        yield from self.predict(*args, **kwargs)


class _ModelBasedInference(_BaseModel):
    def __init__(self, *args, **kwargs):
        self._predictor = create_predictor(*args, **kwargs)

    def predict(self, *args, **kwargs):
        yield from self._predictor(*args, **kwargs)

    def set_predictor(self, **kwargs):
        self._predictor.set_predictor(**kwargs)

    def __getattr__(self, name):
        if hasattr(self._predictor, name):
            return getattr(self._predictor, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


class _ModelBasedConfig(_BaseModel):
    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        self._config = config
        self._model_name = config.Global.model

    def _build_predictor(self):
        predict_kwargs = deepcopy(self._config.Predict)

        model_dir = predict_kwargs.pop("model_dir", None)

        device = self._config.Global.get("device")
        kernel_option = predict_kwargs.pop("kernel_option", {})
        kernel_option.update({"device": device})

        pp_option = PaddlePredictorOption(self._model_name, **kernel_option)
        if NEW_PREDICTOR:
            predictor = create_predictor(
                self._model_name, model_dir, pp_option=pp_option
            )
        else:
            model = self._model_name if model_dir is None else model_dir
            predictor = create_predictor(model, pp_option=pp_option)
        assert "input" in predict_kwargs
        return predict_kwargs, predictor

    def check_dataset(self):
        dataset_checker = build_dataset_checker(self._config)
        return dataset_checker.check()

    def train(self):
        trainer = build_trainer(self._config)
        trainer.train()

    def evaluate(self):
        evaluator = build_evaluater(self._config)
        return evaluator.evaluate()

    def export(self):
        exportor = build_exportor(self._config)
        return exportor.export()

    def predict(self):
        predict_kwargs, predictor = self._build_predictor()
        yield from predictor(**predict_kwargs)
