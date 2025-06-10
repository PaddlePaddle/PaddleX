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

from abc import ABCMeta

from . import logging
from .errors import (
    DuplicateRegistrationError,
    raise_class_not_found_error,
    raise_no_entity_registered_error,
)


class AutoRegisterMetaClass(type):
    """meta class that automatically registry subclass to its baseclass

    Args:
        type (class): type

    Returns:
        class: meta class
    """

    __model_type_attr_name = "entities"
    __base_class_flag = "__is_base"
    __registered_map_name = "__registered_map"

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        mcs.__register_model_entity(bases, cls, attrs)
        return cls

    @classmethod
    def __register_model_entity(mcs, bases, cls, attrs):
        if bases:
            for base in bases:
                base_cls = mcs.__find_base_class(base)
                if base_cls and hasattr(cls, mcs.__model_type_attr_name):
                    mcs.__register_to_base_class(base_cls, cls)

    @classmethod
    def __find_base_class(mcs, cls):
        is_base_flag = mcs.__base_class_flag
        if is_base_flag.startswith("__"):
            is_base_flag = f"_{cls.__name__}" + is_base_flag
        if getattr(cls, is_base_flag, False):
            return cls
        for base in cls.__bases__:
            base_cls = mcs.__find_base_class(base)
            if base_cls:
                return base_cls
        return None

    @classmethod
    def __register_to_base_class(mcs, base, cls):
        cls_entity_name = getattr(cls, mcs.__model_type_attr_name)
        if isinstance(cls_entity_name, str):
            cls_entity_name = [cls_entity_name]

        records = getattr(base, mcs.__registered_map_name, {})
        for name in cls_entity_name:
            if name in records and records[name] is not cls:
                raise DuplicateRegistrationError(
                    f"The name(`{name}`) duplicated registration! The class entities are: `{cls.__name__}` and \
`{records[name].__name__}`."
                )
            records[name] = cls
            logging.debug(
                f"The class entity({cls.__name__}) has been register as name(`{name}`)."
            )
        setattr(base, mcs.__registered_map_name, records)

    def all(cls):
        """get all subclass"""
        if not hasattr(cls, type(cls).__registered_map_name):
            raise_no_entity_registered_error(cls)
        return getattr(cls, type(cls).__registered_map_name)

    def get(cls, name: str):
        """get the registried class by name"""
        all_entities = cls.all()
        if name not in all_entities:
            raise_class_not_found_error(name, cls, all_entities)
        return all_entities[name]


class AutoRegisterABCMetaClass(ABCMeta, AutoRegisterMetaClass):
    """AutoRegisterABCMetaClass"""
