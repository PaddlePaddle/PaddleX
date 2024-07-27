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




class FormulaRecKeys(object):
    """
    This class defines a set of keys used for communication of TextRec predictors
    and transforms. Both predictors and transforms accept a dict or a list of
    dicts as input, and they get the objects of their interest from the dict, or
    put the generated objects into the dict, all based on these keys.
    """

    # Common keys
    IMAGE = 'image'
    IM_SIZE = 'image_size'
    IM_PATH = 'input_path'
    ORI_IM = 'original_image'
    ORI_IM_SIZE = 'original_image_size'
    # Suite-specific keys
    REC_PROBS = 'probs'
    REC_TEXT = 'rec_text'
