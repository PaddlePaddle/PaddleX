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


import codecs
import yaml

from ....utils import logging
from ...base.predictor.transforms import image_common


class InnerConfig(object):
    """ Inner Config
    """

    def __init__(self, config_path):
        self.inner_cfg = self.load(config_path)

    def load(self, config_path):
        """load config
        """
        with codecs.open(config_path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        return dic

    @property
    def pre_transforms(self):
        """ read preprocess transforms from  config file """

        def _process_incompct_args(cfg, arg_names, action):
            for name in arg_names:
                if name in cfg:
                    if action == 'ignore':
                        logging.warning(
                            f"Ignoring incompatible argument: {name}")
                    elif action == 'raise':
                        raise RuntimeError(
                            f"Incompatible argument detected: {name}")
                    else:
                        raise ValueError(f"Unknown action: {action}")

        tfs_cfg = self.inner_cfg['Deploy']['transforms']
        tfs = []
        for cfg in tfs_cfg:
            if cfg['type'] == 'Normalize':
                tf = image_common.Normalize(
                    mean=cfg.get('mean', 0.5), std=cfg.get('std', 0.5))
            elif cfg['type'] == 'Resize':
                tf = image_common.Resize(
                    target_size=cfg.get('target_size', (512, 512)),
                    keep_ratio=cfg.get('keep_ratio', False),
                    size_divisor=cfg.get('size_divisor', None),
                    interp=cfg.get('interp', 'LINEAR'))
            elif cfg['type'] == 'ResizeByLong':
                tf = image_common.ResizeByLong(
                    target_long_edge=cfg['long_size'],
                    size_divisor=None,
                    interp='LINEAR')
            elif cfg['type'] == 'ResizeByShort':
                _process_incompct_args(cfg, ['max_size'], action='raise')
                tf = image_common.ResizeByShort(
                    target_short_edge=cfg['short_size'],
                    size_divisor=None,
                    interp='LINEAR')
            elif cfg['type'] == 'Padding':
                _process_incompct_args(
                    cfg, ['label_padding_value'], action='ignore')
                tf = image_common.Pad(target_size=cfg['target_size'],
                                      val=cfg.get('im_padding_value', 127.5))
            else:
                raise RuntimeError(f"Unsupported type: {cfg['type']}")
            tfs.append(tf)
        return tfs
