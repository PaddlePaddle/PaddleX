# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .interpretation_algorithms import CAM, LIME, NormLIME
from .normlime_base import precompute_normlime_weights


class Interpretation(object):
    """
    Base class for all interpretation algorithms.
    """

    def __init__(self, interpretation_algorithm_name, predict_fn, label_names,
                 **kwargs):
        supported_algorithms = {'cam': CAM, 'lime': LIME, 'normlime': NormLIME}

        self.algorithm_name = interpretation_algorithm_name.lower()
        assert self.algorithm_name in supported_algorithms.keys()
        self.predict_fn = predict_fn

        # initialization for the interpretation algorithm.
        self.algorithm = supported_algorithms[self.algorithm_name](
            self.predict_fn, label_names, **kwargs)

    def interpret(self, data_, visualization=True, save_dir='./'):
        """

        Args:
            data_: data_ can be a path or numpy.ndarray.
            visualization: whether to show using matplotlib.
            save_dir: dir to save figure if save_to_disk is True.

        Returns:

        """
        return self.algorithm.interpret(data_, visualization, save_dir)
