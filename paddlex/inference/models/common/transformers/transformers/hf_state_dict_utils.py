# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class BatchNormHFStateDictMixin:
    def _get_forward_key_rules(self):
        return [
            ("_mean", "_mean", "running_mean"),
            ("_variance", "_variance", "running_var"),
        ]

    def _get_reverse_key_rules(self):
        return [
            ("running_mean", "running_mean", "_mean"),
            ("running_var", "running_var", "_variance"),
        ]

    def get_hf_state_dict(self, *args, **kwargs):

        try:
            super().get_hf_state_dict(*args, **kwargs)
        except NotImplementedError:
            pass

        model_state_dict = self.state_dict(*args, **kwargs)
        hf_state_dict = {}
        rules = self._get_forward_key_rules()
        for old_key, value in model_state_dict.items():
            new_key = old_key
            for match_key, old_sub, new_sub in rules:
                if match_key in old_key:
                    new_key = old_key.replace(old_sub, new_sub)
                    break
            hf_state_dict[new_key] = value
        return hf_state_dict

    def set_hf_state_dict(self, state_dict, *args, **kwargs):

        try:
            super().set_hf_state_dict(state_dict, *args, **kwargs)
        except NotImplementedError:
            pass

        key_mapping = {}
        rules = self._get_reverse_key_rules()
        for old_key in list(state_dict.keys()):
            for match_key, old_sub, new_sub in rules:
                if match_key in old_key:
                    key_mapping[old_key] = old_key.replace(old_sub, new_sub)
                    break
        for old_key, new_key in key_mapping.items():
            state_dict[new_key] = state_dict.pop(old_key)
        return self.set_state_dict(state_dict, *args, **kwargs)
