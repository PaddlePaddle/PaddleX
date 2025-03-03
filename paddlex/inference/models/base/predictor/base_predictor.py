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

from typing import List, Dict, Any, Iterator
from pathlib import Path
from abc import abstractmethod, ABC

from .....utils.flags import INFER_BENCHMARK
from ....utils.io import YAMLReader
from ....common.batch_sampler import BaseBatchSampler


class PredictionWrap:
    """Wraps the prediction data and supports get by index."""

    def __init__(self, data: Dict[str, List[Any]], num: int) -> None:
        """Initializes the PredictionWrap with prediction data.

        Args:
            data (Dict[str, List[Any]]): A dictionary where keys are string identifiers and values are lists of predictions.
            num (int): The number of predictions, that is length of values per key in the data dictionary.

        Raises:
            AssertionError: If the length of any list in data does not match num.
        """
        assert isinstance(data, dict), "data must be a dictionary"
        for k in data:
            assert len(data[k]) == num, f"{len(data[k])} != {num} for key {k}!"
        self._data = data
        self._keys = data.keys()

    def get_by_idx(self, idx: int) -> Dict[str, Any]:
        """Get the prediction by specified index.

        Args:
            idx (int): The index to get predictions from.

        Returns:
            Dict[str, Any]: A dictionary with the same keys as the input data, but with the values at the specified index.
        """
        return {key: self._data[key][idx] for key in self._keys}


class BasePredictor(ABC):
    """BasePredictor."""

    MODEL_FILE_PREFIX = "inference"

    def __init__(self, model_dir: str, config: Dict = None) -> None:
        """Initializes the BasePredictor.

        Args:
            model_dir (str): The directory where the static model files is stored.
            config (dict, optional): The configuration of model to infer. Defaults to None.
        """
        super().__init__()
        self.model_dir = Path(model_dir)
        self.config = config if config else self.load_config(self.model_dir)
        self.batch_sampler = self._build_batch_sampler()
        self.result_class = self._get_result_class()

        # alias predict() to the __call__()
        self.predict = self.__call__

    @property
    def config_path(self) -> str:
        """
        Get the path to the configuration file.

        Returns:
            str: The path to the configuration file.
        """
        return self.get_config_path(self.model_dir)

    @property
    def model_name(self) -> str:
        """
        Get the model name.

        Returns:
            str: The model name.
        """
        return self.config["Global"]["model_name"]

    @classmethod
    def get_config_path(cls, model_dir) -> str:
        """Get the path to the configuration file for the given model directory.

        Args:
            model_dir (Path): The directory where the static model files is stored.

        Returns:
            Path: The path to the configuration file.
        """
        return model_dir / f"{cls.MODEL_FILE_PREFIX}.yml"

    @classmethod
    def load_config(cls, model_dir) -> Dict:
        """Load the configuration from the specified model directory.

        Args:
            model_dir (Path): The where the static model files is stored.

        Returns:
            dict: The loaded configuration dictionary.
        """
        yaml_reader = YAMLReader()
        return yaml_reader.read(cls.get_config_path(model_dir))

    @abstractmethod
    def __call__(self, input: Any, **kwargs: Dict[str, Any]) -> Iterator[Any]:
        """Predict with the given input and additional keyword arguments."""
        raise NotImplementedError

    @abstractmethod
    def set_predictor(self, batch_size: int = None, device: str = None, *args) -> None:
        """Sets up the predictor."""
        raise NotImplementedError

    def apply(self, input: Any, **kwargs) -> Iterator[Any]:
        """
        Do predicting with the input data and yields predictions.

        Args:
            input (Any): The input data to be predicted.

        Yields:
            Iterator[Any]: An iterator yielding prediction results.
        """
        if INFER_BENCHMARK:
            if not isinstance(input, list):
                raise TypeError("In benchmark mode, `input` must be a list")
            batches = list(self.batch_sampler(input))
            if len(batches) != 1 or len(batches[0]) != len(input):
                raise ValueError("Unexpected number of instances")
        else:
            batches = self.batch_sampler(input)
        for batch_data in batches:
            prediction = self.process(batch_data, **kwargs)
            prediction = PredictionWrap(prediction, len(batch_data))
            for idx in range(len(batch_data)):
                yield self.result_class(prediction.get_by_idx(idx))

    @abstractmethod
    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        """process the batch data sampled from BatchSampler and return the prediction result.

        Args:
            batch_data (List[Any]): The batch data sampled from BatchSampler.

        Returns:
            Dict[str, List[Any]]: The prediction result.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_batch_sampler(self) -> BaseBatchSampler:
        """Build batch sampler.

        Returns:
            BaseBatchSampler: batch sampler object.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_result_class(self) -> type:
        """Get the result class.

        Returns:
            type: The result class.
        """
        raise NotImplementedError
