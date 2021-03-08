# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import math
import chardet
import paddlex


def seconds_to_hms(seconds):
    h = math.floor(seconds / 3600)
    m = math.floor((seconds - h * 3600) / 60)
    s = int(seconds - h * 3600 - m * 60)
    hms_str = "{}:{}:{}".format(h, m, s)
    return hms_str


def get_encoding(path):
    f = open(path, 'rb')
    data = f.read()
    file_encoding = chardet.detect(data).get('encoding')
    f.close()
    return file_encoding


def get_single_card_bs(batch_size):
    card_num = paddlex.env_info['num']
    place = paddlex.env_info['place']
    if batch_size % card_num == 0:
        return int(batch_size // card_num)
    else:
        raise Exception("Please support correct batch_size, \
                        which can be divided by available cards({}) in {}"
                        .format(card_num, place))


def dict2str(dict_input):
    out = ''
    for k, v in dict_input.items():
        try:
            v = round(float(v), 6)
        except:
            pass
        out = out + '{}={}, '.format(k, v)
    return out.strip(', ')
