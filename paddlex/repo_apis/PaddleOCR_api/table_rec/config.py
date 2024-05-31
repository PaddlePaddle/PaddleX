# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Author: PaddlePaddle Authors
"""
import os

from ....utils.misc import abspath
from ..text_rec.config import TextRecConfig


class TableRecConfig(TextRecConfig):
    """ Table Recognition Config """

    def update_dataset(self, dataset_path: str, dataset_type: str=None):
        """update dataset settings

        Args:
            dataset_path (str): the root path of dataset.
            dataset_type (str, optional): dataset type. Defaults to None.

        Raises:
            ValueError: the dataset_type error.
        """
        dataset_path = abspath(dataset_path)
        if dataset_type is None:
            dataset_type = 'PubTabTableRecDataset'
        if dataset_type == 'PubTabTableRecDataset':
            _cfg = {
                'Train.dataset.name': dataset_type,
                'Train.dataset.data_dir': dataset_path,
                'Train.dataset.label_file_list':
                [os.path.join(dataset_path, 'train.txt')],
                'Eval.dataset.name': dataset_type,
                'Eval.dataset.data_dir': dataset_path,
                'Eval.dataset.label_file_list':
                [os.path.join(dataset_path, 'val.txt')]
            }
            self.update(_cfg)
        else:
            raise ValueError(f"{repr(dataset_type)} is not supported.")

    def _get_infer_shape(self):
        """get resize scale of ResizeImg operation in the evaluation

        Returns:
            str: resize scale, i.e. `Eval.dataset.transforms.ResizeTableImage.max_len`
        """
        size = None
        transforms = self.dict['Eval']['dataset']['transforms']
        for op in transforms:
            if list(op)[0] == 'ResizeTableImage':
                size = op['ResizeTableImage']['max_len']
        return size
