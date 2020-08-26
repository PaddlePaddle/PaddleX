# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
import paddle.fluid as fluid
import os
import sys
import paddlex as pdx
import paddlex.utils.logging as logging

__all__ = ['export_onnx']


def export_onnx(model_dir, save_dir, fixed_input_shape):
    assert len(fixed_input_shape) == 2, "len of fixed input shape must == 2"
    model = pdx.load_model(model_dir, fixed_input_shape)
    model_name = os.path.basename(model_dir.strip('/')).split('/')[-1]
    export_onnx_model(model, save_dir)


def export_onnx_model(model, save_dir, opset_version=10):
    if model.__class__.__name__ == "FastSCNN" or (
            model.model_type == "detector" and
            model.__class__.__name__ != "YOLOv3"):
        logging.error(
            "Only image classifier models, detection models(YOLOv3) and semantic segmentation models(except FastSCNN) are supported to export to ONNX"
        )
    try:
        import x2paddle
        if x2paddle.__version__ < '0.7.4':
            logging.error("You need to upgrade x2paddle >= 0.7.4")
    except:
        logging.error(
            "You need to install x2paddle first, pip install x2paddle>=0.7.4")
    if opset_version == 10 and model.__class__.__name__ == "YOLOv3":
        logging.warning(
            "Export for openVINO by default, the output of multiclass_nms exported to onnx will contains background. If you need onnx completely consistent with paddle, please use X2Paddle to export"
        )
        x2paddle.op_mapper.paddle2onnx.opset10.paddle_custom_layer.multiclass_nms.multiclass_nms = multiclass_nms_for_openvino
    from x2paddle.op_mapper.paddle2onnx.paddle_op_mapper import PaddleOpMapper
    mapper = PaddleOpMapper()
    mapper.convert(
        model.test_prog,
        save_dir,
        scope=model.scope,
        opset_version=opset_version)


def multiclass_nms_for_openvino(op, block):
    """
    Convert the paddle multiclass_nms to onnx op.
    This op is get the select boxes from origin boxes.
    This op is for OpenVINO, which donn't support dynamic shape).
    """
    import math
    import sys
    import numpy as np
    import paddle.fluid.core as core
    import paddle.fluid as fluid
    import onnx
    import warnings
    from onnx import helper, onnx_pb
    inputs = dict()
    outputs = dict()
    attrs = dict()
    for name in op.input_names:
        inputs[name] = op.input(name)
    for name in op.output_names:
        outputs[name] = op.output(name)
    for name in op.attr_names:
        attrs[name] = op.attr(name)

    result_name = outputs['Out'][0]
    background = attrs['background_label']
    normalized = attrs['normalized']
    if normalized == False:
        warnings.warn(
            'The parameter normalized of multiclass_nms OP of Paddle is False, which has diff with ONNX. \
                         Please set normalized=True in multiclass_nms of Paddle'
        )

    #convert the paddle attribute to onnx tensor
    name_score_threshold = [outputs['Out'][0] + "@score_threshold"]
    name_iou_threshold = [outputs['Out'][0] + "@iou_threshold"]
    name_keep_top_k = [outputs['Out'][0] + '@keep_top_k']
    name_keep_top_k_2D = [outputs['Out'][0] + '@keep_top_k_1D']

    node_score_threshold = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_score_threshold,
        value=onnx.helper.make_tensor(
            name=name_score_threshold[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[float(attrs['score_threshold'])]))

    node_iou_threshold = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_iou_threshold,
        value=onnx.helper.make_tensor(
            name=name_iou_threshold[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[float(attrs['nms_threshold'])]))

    node_keep_top_k = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_keep_top_k,
        value=onnx.helper.make_tensor(
            name=name_keep_top_k[0] + "@const",
            data_type=onnx.TensorProto.INT64,
            dims=(),
            vals=[np.int64(attrs['keep_top_k'])]))

    node_keep_top_k_2D = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_keep_top_k_2D,
        value=onnx.helper.make_tensor(
            name=name_keep_top_k_2D[0] + "@const",
            data_type=onnx.TensorProto.INT64,
            dims=[1, 1],
            vals=[np.int64(attrs['keep_top_k'])]))

    # the paddle data format is x1,y1,x2,y2
    kwargs = {'center_point_box': 0}

    name_select_nms = [outputs['Out'][0] + "@select_index"]
    node_select_nms= onnx.helper.make_node(
        'NonMaxSuppression',
        inputs=inputs['BBoxes'] + inputs['Scores'] + name_keep_top_k +\
            name_iou_threshold + name_score_threshold,
        outputs=name_select_nms)
    # step 1 nodes select the nms class
    node_list = [
        node_score_threshold, node_iou_threshold, node_keep_top_k,
        node_keep_top_k_2D, node_select_nms
    ]

    # create some const value to use
    name_const_value = [result_name+"@const_0",
        result_name+"@const_1",\
        result_name+"@const_2",\
        result_name+"@const_-1"]
    value_const_value = [0, 1, 2, -1]
    for name, value in zip(name_const_value, value_const_value):
        node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[name],
            value=onnx.helper.make_tensor(
                name=name + "@const",
                data_type=onnx.TensorProto.INT64,
                dims=[1],
                vals=[value]))
        node_list.append(node)

    # In this code block, we will deocde the raw score data, reshape N * C * M to 1 * N*C*M
    # and the same time, decode the select indices to 1 * D, gather the select_indices
    outputs_gather_1_ = [result_name + "@gather_1_"]
    node_gather_1_ = onnx.helper.make_node(
        'Gather',
        inputs=name_select_nms + [result_name + "@const_1"],
        outputs=outputs_gather_1_,
        axis=1)
    node_list.append(node_gather_1_)
    outputs_gather_1 = [result_name + "@gather_1"]
    node_gather_1 = onnx.helper.make_node(
        'Unsqueeze',
        inputs=outputs_gather_1_,
        outputs=outputs_gather_1,
        axes=[0])
    node_list.append(node_gather_1)

    outputs_gather_2_ = [result_name + "@gather_2_"]
    node_gather_2_ = onnx.helper.make_node(
        'Gather',
        inputs=name_select_nms + [result_name + "@const_2"],
        outputs=outputs_gather_2_,
        axis=1)
    node_list.append(node_gather_2_)

    outputs_gather_2 = [result_name + "@gather_2"]
    node_gather_2 = onnx.helper.make_node(
        'Unsqueeze',
        inputs=outputs_gather_2_,
        outputs=outputs_gather_2,
        axes=[0])
    node_list.append(node_gather_2)

    # reshape scores N * C * M to (N*C*M) * 1
    outputs_reshape_scores_rank1 = [result_name + "@reshape_scores_rank1"]
    node_reshape_scores_rank1 = onnx.helper.make_node(
        "Reshape",
        inputs=inputs['Scores'] + [result_name + "@const_-1"],
        outputs=outputs_reshape_scores_rank1)
    node_list.append(node_reshape_scores_rank1)

    # get the shape of scores
    outputs_shape_scores = [result_name + "@shape_scores"]
    node_shape_scores = onnx.helper.make_node(
        'Shape', inputs=inputs['Scores'], outputs=outputs_shape_scores)
    node_list.append(node_shape_scores)

    # gather the index: 2 shape of scores
    outputs_gather_scores_dim1 = [result_name + "@gather_scores_dim1"]
    node_gather_scores_dim1 = onnx.helper.make_node(
        'Gather',
        inputs=outputs_shape_scores + [result_name + "@const_2"],
        outputs=outputs_gather_scores_dim1,
        axis=0)
    node_list.append(node_gather_scores_dim1)

    # mul class * M
    outputs_mul_classnum_boxnum = [result_name + "@mul_classnum_boxnum"]
    node_mul_classnum_boxnum = onnx.helper.make_node(
        'Mul',
        inputs=outputs_gather_1 + outputs_gather_scores_dim1,
        outputs=outputs_mul_classnum_boxnum)
    node_list.append(node_mul_classnum_boxnum)

    # add class * M * index
    outputs_add_class_M_index = [result_name + "@add_class_M_index"]
    node_add_class_M_index = onnx.helper.make_node(
        'Add',
        inputs=outputs_mul_classnum_boxnum + outputs_gather_2,
        outputs=outputs_add_class_M_index)
    node_list.append(node_add_class_M_index)

    # Squeeze the indices to 1 dim
    outputs_squeeze_select_index = [result_name + "@squeeze_select_index"]
    node_squeeze_select_index = onnx.helper.make_node(
        'Squeeze',
        inputs=outputs_add_class_M_index,
        outputs=outputs_squeeze_select_index,
        axes=[0, 2])
    node_list.append(node_squeeze_select_index)

    # gather the data from flatten scores
    outputs_gather_select_scores = [result_name + "@gather_select_scores"]
    node_gather_select_scores = onnx.helper.make_node('Gather',
        inputs=outputs_reshape_scores_rank1 + \
            outputs_squeeze_select_index,
        outputs=outputs_gather_select_scores,
        axis=0)
    node_list.append(node_gather_select_scores)

    # get nums to input TopK
    outputs_shape_select_num = [result_name + "@shape_select_num"]
    node_shape_select_num = onnx.helper.make_node(
        'Shape',
        inputs=outputs_gather_select_scores,
        outputs=outputs_shape_select_num)
    node_list.append(node_shape_select_num)

    outputs_gather_select_num = [result_name + "@gather_select_num"]
    node_gather_select_num = onnx.helper.make_node(
        'Gather',
        inputs=outputs_shape_select_num + [result_name + "@const_0"],
        outputs=outputs_gather_select_num,
        axis=0)
    node_list.append(node_gather_select_num)

    outputs_unsqueeze_select_num = [result_name + "@unsqueeze_select_num"]
    node_unsqueeze_select_num = onnx.helper.make_node(
        'Unsqueeze',
        inputs=outputs_gather_select_num,
        outputs=outputs_unsqueeze_select_num,
        axes=[0])
    node_list.append(node_unsqueeze_select_num)

    outputs_concat_topK_select_num = [result_name + "@conat_topK_select_num"]
    node_conat_topK_select_num = onnx.helper.make_node(
        'Concat',
        inputs=outputs_unsqueeze_select_num + name_keep_top_k_2D,
        outputs=outputs_concat_topK_select_num,
        axis=0)
    node_list.append(node_conat_topK_select_num)

    outputs_cast_concat_topK_select_num = [
        result_name + "@concat_topK_select_num"
    ]
    node_outputs_cast_concat_topK_select_num = onnx.helper.make_node(
        'Cast',
        inputs=outputs_concat_topK_select_num,
        outputs=outputs_cast_concat_topK_select_num,
        to=6)
    node_list.append(node_outputs_cast_concat_topK_select_num)
    # get min(topK, num_select)
    outputs_compare_topk_num_select = [
        result_name + "@compare_topk_num_select"
    ]
    node_compare_topk_num_select = onnx.helper.make_node(
        'ReduceMin',
        inputs=outputs_cast_concat_topK_select_num,
        outputs=outputs_compare_topk_num_select,
        keepdims=0)
    node_list.append(node_compare_topk_num_select)

    # unsqueeze the indices to 1D tensor
    outputs_unsqueeze_topk_select_indices = [
        result_name + "@unsqueeze_topk_select_indices"
    ]
    node_unsqueeze_topk_select_indices = onnx.helper.make_node(
        'Unsqueeze',
        inputs=outputs_compare_topk_num_select,
        outputs=outputs_unsqueeze_topk_select_indices,
        axes=[0])
    node_list.append(node_unsqueeze_topk_select_indices)

    # cast the indices to INT64
    outputs_cast_topk_indices = [result_name + "@cast_topk_indices"]
    node_cast_topk_indices = onnx.helper.make_node(
        'Cast',
        inputs=outputs_unsqueeze_topk_select_indices,
        outputs=outputs_cast_topk_indices,
        to=7)
    node_list.append(node_cast_topk_indices)

    # select topk scores  indices
    outputs_topk_select_topk_indices = [result_name + "@topk_select_topk_values",\
        result_name + "@topk_select_topk_indices"]
    node_topk_select_topk_indices = onnx.helper.make_node(
        'TopK',
        inputs=outputs_gather_select_scores + outputs_cast_topk_indices,
        outputs=outputs_topk_select_topk_indices)
    node_list.append(node_topk_select_topk_indices)

    # gather topk label, scores, boxes
    outputs_gather_topk_scores = [result_name + "@gather_topk_scores"]
    node_gather_topk_scores = onnx.helper.make_node(
        'Gather',
        inputs=outputs_gather_select_scores +
        [outputs_topk_select_topk_indices[1]],
        outputs=outputs_gather_topk_scores,
        axis=0)
    node_list.append(node_gather_topk_scores)

    outputs_gather_topk_class = [result_name + "@gather_topk_class"]
    node_gather_topk_class = onnx.helper.make_node(
        'Gather',
        inputs=outputs_gather_1 + [outputs_topk_select_topk_indices[1]],
        outputs=outputs_gather_topk_class,
        axis=1)
    node_list.append(node_gather_topk_class)

    # gather the boxes need to gather the boxes id, then get boxes
    outputs_gather_topk_boxes_id = [result_name + "@gather_topk_boxes_id"]
    node_gather_topk_boxes_id = onnx.helper.make_node(
        'Gather',
        inputs=outputs_gather_2 + [outputs_topk_select_topk_indices[1]],
        outputs=outputs_gather_topk_boxes_id,
        axis=1)
    node_list.append(node_gather_topk_boxes_id)

    # squeeze the gather_topk_boxes_id to 1 dim
    outputs_squeeze_topk_boxes_id = [result_name + "@squeeze_topk_boxes_id"]
    node_squeeze_topk_boxes_id = onnx.helper.make_node(
        'Squeeze',
        inputs=outputs_gather_topk_boxes_id,
        outputs=outputs_squeeze_topk_boxes_id,
        axes=[0, 2])
    node_list.append(node_squeeze_topk_boxes_id)

    outputs_gather_select_boxes = [result_name + "@gather_select_boxes"]
    node_gather_select_boxes = onnx.helper.make_node(
        'Gather',
        inputs=inputs['BBoxes'] + outputs_squeeze_topk_boxes_id,
        outputs=outputs_gather_select_boxes,
        axis=1)
    node_list.append(node_gather_select_boxes)

    # concat the final result
    # before concat need to cast the class to float
    outputs_cast_topk_class = [result_name + "@cast_topk_class"]
    node_cast_topk_class = onnx.helper.make_node(
        'Cast',
        inputs=outputs_gather_topk_class,
        outputs=outputs_cast_topk_class,
        to=1)
    node_list.append(node_cast_topk_class)

    outputs_unsqueeze_topk_scores = [result_name + "@unsqueeze_topk_scores"]
    node_unsqueeze_topk_scores = onnx.helper.make_node(
        'Unsqueeze',
        inputs=outputs_gather_topk_scores,
        outputs=outputs_unsqueeze_topk_scores,
        axes=[0, 2])
    node_list.append(node_unsqueeze_topk_scores)

    inputs_concat_final_results = outputs_cast_topk_class + outputs_unsqueeze_topk_scores +\
        outputs_gather_select_boxes
    outputs_sort_by_socre_results = [result_name + "@concat_topk_scores"]
    node_sort_by_socre_results = onnx.helper.make_node(
        'Concat',
        inputs=inputs_concat_final_results,
        outputs=outputs_sort_by_socre_results,
        axis=2)
    node_list.append(node_sort_by_socre_results)

    # select topk classes indices
    outputs_squeeze_cast_topk_class = [
        result_name + "@squeeze_cast_topk_class"
    ]
    node_squeeze_cast_topk_class = onnx.helper.make_node(
        'Squeeze',
        inputs=outputs_cast_topk_class,
        outputs=outputs_squeeze_cast_topk_class,
        axes=[0, 2])
    node_list.append(node_squeeze_cast_topk_class)
    outputs_neg_squeeze_cast_topk_class = [
        result_name + "@neg_squeeze_cast_topk_class"
    ]
    node_neg_squeeze_cast_topk_class = onnx.helper.make_node(
        'Neg',
        inputs=outputs_squeeze_cast_topk_class,
        outputs=outputs_neg_squeeze_cast_topk_class)
    node_list.append(node_neg_squeeze_cast_topk_class)
    outputs_topk_select_classes_indices = [result_name + "@topk_select_topk_classes_scores",\
        result_name + "@topk_select_topk_classes_indices"]
    node_topk_select_topk_indices = onnx.helper.make_node(
        'TopK',
        inputs=outputs_neg_squeeze_cast_topk_class + outputs_cast_topk_indices,
        outputs=outputs_topk_select_classes_indices)
    node_list.append(node_topk_select_topk_indices)
    outputs_concat_final_results = outputs['Out']
    node_concat_final_results = onnx.helper.make_node(
        'Gather',
        inputs=outputs_sort_by_socre_results +
        [outputs_topk_select_classes_indices[1]],
        outputs=outputs_concat_final_results,
        axis=1)
    node_list.append(node_concat_final_results)
    return node_list
