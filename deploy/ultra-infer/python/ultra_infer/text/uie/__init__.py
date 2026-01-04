# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import

import logging

from ... import ModelFormat, RuntimeOption, UltraInferModel
from ... import c_lib_wrap as C


class SchemaLanguage(object):
    ZH = 0
    EN = 1


class SchemaNode(object):
    def __init__(self, name, children=[]):
        schema_node_children = []
        if isinstance(children, str):
            children = [children]
        for child in children:
            if isinstance(child, str):
                schema_node_children += [C.text.SchemaNode(child, [])]
            elif isinstance(child, dict):
                for key, val in child.items():
                    schema_node_child = SchemaNode(key, val)
                    schema_node_children += [schema_node_child._schema_node]
            else:
                assert "The type of child of SchemaNode should be str or dict."
        self._schema_node = C.text.SchemaNode(name, schema_node_children)
        self._schema_node_children = schema_node_children


class UIEModel(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file,
        vocab_file,
        position_prob=0.5,
        max_length=128,
        schema=[],
        batch_size=64,
        runtime_option=RuntimeOption(),
        model_format=ModelFormat.PADDLE,
        schema_language=SchemaLanguage.ZH,
    ):
        if isinstance(schema, list):
            schema = SchemaNode("", schema)._schema_node_children
        elif isinstance(schema, dict):
            schema_tmp = []
            for key, val in schema.items():
                schema_tmp += [SchemaNode(key, val)._schema_node]
            schema = schema_tmp
        else:
            assert "The type of schema should be list or dict."
        schema_language = C.text.SchemaLanguage(schema_language)
        self._model = C.text.UIEModel(
            model_file,
            params_file,
            vocab_file,
            position_prob,
            max_length,
            schema,
            batch_size,
            runtime_option._option,
            model_format,
            schema_language,
        )
        assert self.initialized, "UIEModel initialize failed."

    def set_schema(self, schema):
        if isinstance(schema, list):
            schema = SchemaNode("", schema)._schema_node_children
        elif isinstance(schema, dict):
            schema_tmp = []
            for key, val in schema.items():
                schema_tmp += [SchemaNode(key, val)._schema_node]
            schema = schema_tmp
        self._model.set_schema(schema)

    def predict(self, texts, return_dict=False):
        results = self._model.predict(texts)
        if not return_dict:
            return results
        new_results = []
        for result in results:
            uie_result = dict()
            for key, uie_results in result.items():
                uie_result[key] = list()
                for uie_res in uie_results:
                    uie_result[key].append(uie_res.get_dict())
            new_results += [uie_result]
        return new_results
