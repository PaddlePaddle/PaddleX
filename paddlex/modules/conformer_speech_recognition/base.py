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


from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDatasetChecker(ABC):
    @abstractmethod
    def check(self, dataset_path: str) -> bool:
        """Validate dataset format and integrity"""
        pass

    @abstractmethod
    def get_error_report(self) -> str:
        """Return formatted error report for invalid datasets"""
        pass


class BasePipeline(ABC):
    """Abstract base class for speech recognition pipelines"""

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate pipeline configuration parameters"""
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load pretrained model weights"""
        pass

    @abstractmethod
    def process(self, input_data: Any) -> str:
        """Execute end-to-end speech recognition processing"""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Clean up pipeline resources"""
        pass
