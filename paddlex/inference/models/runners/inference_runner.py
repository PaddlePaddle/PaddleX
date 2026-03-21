# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Interface for inference runners."""

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class InferenceRunner(Protocol):
    """Loosely typed protocol for executable inference runners."""

    def __call__(self, x: Optional[Any] = None, **kwargs: Any) -> Any:
        """Run inference with positional or keyword inputs."""
        ...

    def close(self) -> None:
        """Release any runner resources if necessary."""
        ...
