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


import collections.abc
import copy
import os

import yaml


class PPDetConfigMixin(object):
    """PPDetConfigMixin"""

    def load_config_literally(self, config_path):
        """load_config_literally"""
        # Adapted from
        # https://github.com/PaddlePaddle/PaddleDetection/blob/e3f8dd16bffca04060ec1edc388c5a618e15bbf8/ppdet/core/workspace.py#L77
        # XXX: This function relies on implementation details of PaddleDetection.

        BASE_KEY = "_BASE_"
        with open(config_path, "r", encoding="utf-8") as f:
            dic = yaml.load(f, Loader=_PPDetSerializableLoader)
        if not isinstance(dic, dict):
            print(dic)
            raise TypeError

        if BASE_KEY in dic:
            all_base_cfg = dict()
            base_ymls = list(dic[BASE_KEY])
            for base_yml in base_ymls:
                if base_yml.startswith("~"):
                    base_yml = os.path.expanduser(base_yml)
                if not base_yml.startswith("/"):
                    base_yml = os.path.join(os.path.dirname(config_path), base_yml)

                with open(base_yml, "r", encoding="utf-8") as f:
                    base_cfg = self.load_config_literally(base_yml)
                    all_base_cfg = merge_dicts(base_cfg, all_base_cfg)

            del dic[BASE_KEY]
            return merge_dicts(dic, all_base_cfg)

        return dic

    def dump_literal_config(self, config_path, dic):
        """dump_literal_config"""
        with open(config_path, "w", encoding="utf-8") as f:
            # XXX: We make an extra copy here by calling `dict()`
            # to ensure that `dic` can be represented.
            yaml.dump(dict(dic), f, Dumper=_PPDetSerializableDumper)

    def update_from_dict(self, src_dic, dst_dic):
        """update_from_dict"""
        return merge_dicts(src_dic, dst_dic)


class _PPDetSerializableHandler(collections.abc.MutableMapping):
    """_PPDetSerializableHandler"""

    TYPE_KEY = "_type_"
    EMPTY_TAG = object()

    def __init__(self, tag=None, dic=None):
        super().__init__()
        if tag is None:
            tag = self.EMPTY_TAG
        if dic is None:
            dic = dict()
        self.tag = tag
        self.dic = dic

    def __repr__(self):
        # TODO: Prettier format
        return repr({self.TYPE_KEY: self.tag, **self.dic})

    def __getitem__(self, key):
        if key == self.TYPE_KEY:
            return self.tag
        else:
            return self.dic[key]

    def __setitem__(self, key, val):
        if key == self.TYPE_KEY:
            self.tag = val
        else:
            self.dic[key] = val

    def __delitem__(self, key):
        if key == self.TYPE_KEY:
            self.tag = self.EMPTY_TAG
        else:
            del self.dic[key]

    def __len__(self):
        return len(self.dic) + 1

    def __iter__(self):
        if self.has_nonempty_tag():
            yield self.TYPE_KEY
        yield from self.dic

    def has_nonempty_tag(self):
        """has_nonempty_tag"""
        return self.tag != self.EMPTY_TAG

    @classmethod
    def is_convertible(cls, obj):
        """is_convertible"""
        if isinstance(obj, cls):
            return False
        elif isinstance(obj, collections.abc.Mapping):
            return cls.TYPE_KEY in obj
        else:
            return False

    @classmethod
    def build_from_dict(cls, dic):
        """build_from_dict"""
        dic = copy.deepcopy(dic)
        tag = dic.pop(cls.TYPE_KEY)
        return cls(tag=tag, dic=dic)


def merge_dicts(src_dic, dst_dic):
    """merge_dicts"""

    # Refer to
    # https://github.com/PaddlePaddle/PaddleDetection/blob/e3f8dd16bffca04060ec1edc388c5a618e15bbf8/ppdet/core/workspace.py#L121
    # Additionally, this function deals with the case when `src_dic`
    # or `dst_dic` contains `_PPDetSerializableHandler` objects.

    def _update_sohandler(src_handler, dst_handler):
        """_update_sohandler"""
        dst_handler.update(src_handler)

    def _convert_to_sohandler_if_possible(obj):
        """_convert_to_sohandler_if_possible"""
        if _PPDetSerializableHandler.is_convertible(obj):
            return _PPDetSerializableHandler.build_from_dict(obj)
        else:
            return obj

    def _convert_dict_to_sohandler_with_tag(dic, tag):
        """_convert_dict_to_sohandler_with_tag"""
        return _PPDetSerializableHandler(tag, dic)

    for k, v in src_dic.items():
        v = _convert_to_sohandler_if_possible(v)
        if k not in dst_dic:
            dst_dic[k] = v
        else:
            dst_dic[k] = _convert_to_sohandler_if_possible(dst_dic[k])
            if isinstance(dst_dic[k], _PPDetSerializableHandler):
                if isinstance(v, _PPDetSerializableHandler):
                    _update_sohandler(v, dst_dic[k])
                elif isinstance(v, collections.abc.Mapping):
                    v = _convert_dict_to_sohandler_with_tag(v, dst_dic[k].tag)
                    _update_sohandler(v, dst_dic[k])
                else:
                    dst_dic[k] = v
            elif isinstance(dst_dic[k], collections.abc.Mapping):
                if isinstance(v, _PPDetSerializableHandler):
                    dst_dic[k] = _convert_dict_to_sohandler_with_tag(dst_dic[k], v.tag)
                    _update_sohandler(v, dst_dic[k])
                elif isinstance(v, collections.abc.Mapping):
                    merge_dicts(v, dst_dic[k])
                else:
                    dst_dic[k] = v
            else:
                dst_dic[k] = v

    return dst_dic


class _PPDetSerializableConstructor(yaml.constructor.SafeConstructor):
    """_PPDetSerializableConstructor"""

    def construct_sohandler(self, tag_suffix, node):
        """construct_sohandler"""
        if not isinstance(node, yaml.nodes.MappingNode):
            raise TypeError("Currently, we can only handle a MappingNode.")
        mapping = self.construct_mapping(node)
        return _PPDetSerializableHandler(tag_suffix, mapping)


class _PPDetSerializableLoader(_PPDetSerializableConstructor, yaml.loader.SafeLoader):
    """_PPDetSerializableLoader"""

    def __init__(self, stream):
        _PPDetSerializableConstructor.__init__(self)
        yaml.loader.SafeLoader.__init__(self, stream)


class _PPDetSerializableRepresenter(yaml.representer.SafeRepresenter):
    """_PPDetSerializableRepresenter"""

    def represent_sohandler(self, data):
        """represent_sohandler"""
        # If `data` has empty tag, we represent `data.dic` as a dict
        if not data.has_nonempty_tag:
            return self.represent_dict(data.dic)
        else:
            # XXX: Manually represent a serializable object according to the rules defined in
            # https://github.com/PaddlePaddle/PaddleDetection/blob/e3f8dd16bffca04060ec1edc388c5a618e15bbf8/ppdet/core/config/yaml_helpers.py#L80
            # We prepend a '!' to reconstruct the complete tag
            tag = "!" + data.tag
            return self.represent_mapping(tag, data.dic)


class _PPDetSerializableDumper(_PPDetSerializableRepresenter, yaml.dumper.SafeDumper):
    """_PPDetSerializableDumper"""

    def __init__(
        self,
        stream,
        default_style=None,
        default_flow_style=False,
        canonical=None,
        indent=None,
        width=None,
        allow_unicode=None,
        line_break=None,
        encoding=None,
        explicit_start=None,
        explicit_end=None,
        version=None,
        tags=None,
        sort_keys=True,
    ):

        _PPDetSerializableRepresenter.__init__(
            self,
            default_style=default_style,
            default_flow_style=default_flow_style,
            sort_keys=sort_keys,
        )

        yaml.dumper.SafeDumper.__init__(
            self,
            stream,
            default_style=default_style,
            default_flow_style=default_flow_style,
            canonical=canonical,
            indent=indent,
            width=width,
            allow_unicode=allow_unicode,
            line_break=line_break,
            encoding=encoding,
            explicit_start=explicit_start,
            explicit_end=explicit_end,
            version=version,
            tags=tags,
            sort_keys=sort_keys,
        )

    def ignore_aliases(self, data):
        """ignore_aliases"""
        return True


# We note that all custom tags defined in ppdet starts with a '!'.
# We assume that all unknown tags in the config file corresponds to a serializable class defined in ppdet.
_PPDetSerializableLoader.add_multi_constructor(
    "!", _PPDetSerializableConstructor.construct_sohandler
)
_PPDetSerializableDumper.add_representer(
    _PPDetSerializableHandler, _PPDetSerializableRepresenter.represent_sohandler
)
