#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from .explanation_algorithms import CAM, LIME, NormLIME


class Explanation(object):
    """
    Base class for all explanation algorithms.
    """
    def __init__(self, explanation_algorithm_name, predict_fn, **kwargs):
        supported_algorithms = {
            'cam': CAM,
            'lime': LIME,
            'normlime': NormLIME
        }

        self.algorithm_name = explanation_algorithm_name.lower()
        assert self.algorithm_name in supported_algorithms.keys()
        self.predict_fn = predict_fn

        # initialization for the explanation algorithm.
        self.explain_algorithm = supported_algorithms[self.algorithm_name](
            self.predict_fn, **kwargs
        )

    def explain(self, data_, visualization=True, save_to_disk=True, save_dir='./tmp'):
        """

        Args:
            data_: data_ can be a path or numpy.ndarray.
            visualization: whether to show using matplotlib.
            save_to_disk: whether to save the figure in local disk.
            save_dir: dir to save figure if save_to_disk is True.

        Returns:

        """
        return self.explain_algorithm.explain(data_, visualization, save_to_disk, save_dir)

