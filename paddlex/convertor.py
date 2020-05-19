# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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


def export_onnx_model(model, save_dir):
    support_list = [
        'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet50_vd',
        'ResNet101_vd', 'ResNet50_vd_ssld', 'ResNet101_vd_ssld', 'DarkNet53',
        'MobileNetV1', 'MobileNetV2', 'DenseNet121', 'DenseNet161',
        'DenseNet201'
    ]
    if model.__class__.__name__ not in support_list:
        raise Exception("Model: {} unsupport export to ONNX".format(
            model.__class__.__name__))
    try:
        from fluid.utils import op_io_info, init_name_prefix
        from onnx import helper, checker
        import fluid_onnx.ops as ops
        from fluid_onnx.variables import paddle_variable_to_onnx_tensor, paddle_onnx_weight
        from debug.model_check import debug_model, Tracker
    except Exception as e:
        logging.error(
            "Import Module Failed! Please install paddle2onnx. Related requirements see https://github.com/PaddlePaddle/paddle2onnx."
        )
        raise e
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.global_scope()
    with fluid.scope_guard(inference_scope):
        test_input_names = [
            var.name for var in list(model.test_inputs.values())
        ]
        inputs_outputs_list = ["fetch", "feed"]
        weights, weights_value_info = [], []
        global_block = model.test_prog.global_block()
        for var_name in global_block.vars:
            var = global_block.var(var_name)
            if var_name not in test_input_names\
                and var.persistable:
                weight, val_info = paddle_onnx_weight(
                    var=var, scope=inference_scope)
                weights.append(weight)
                weights_value_info.append(val_info)

        # Create inputs
        inputs = [
            paddle_variable_to_onnx_tensor(v, global_block)
            for v in test_input_names
        ]
        logging.INFO("load the model parameter done.")
        onnx_nodes = []
        op_check_list = []
        op_trackers = []
        nms_first_index = -1
        nms_outputs = []
        for block in model.test_prog.blocks:
            for op in block.ops:
                if op.type in ops.node_maker:
                    # TODO: deal with the corner case that vars in
                    #     different blocks have the same name
                    node_proto = ops.node_maker[str(op.type)](
                        operator=op, block=block)
                    op_outputs = []
                    last_node = None
                    if isinstance(node_proto, tuple):
                        onnx_nodes.extend(list(node_proto))
                        last_node = list(node_proto)
                    else:
                        onnx_nodes.append(node_proto)
                        last_node = [node_proto]
                    tracker = Tracker(str(op.type), last_node)
                    op_trackers.append(tracker)
                    op_check_list.append(str(op.type))
                    if op.type == "multiclass_nms" and nms_first_index < 0:
                        nms_first_index = 0
                    if nms_first_index >= 0:
                        _, _, output_op = op_io_info(op)
                        for output in output_op:
                            nms_outputs.extend(output_op[output])
                else:
                    if op.type not in ['feed', 'fetch']:
                        op_check_list.append(op.type)
        logging.info('The operator sets to run test case.')
        logging.info(set(op_check_list))

        # Create outputs
        # Get the new names for outputs if they've been renamed in nodes' making
        renamed_outputs = op_io_info.get_all_renamed_outputs()
        test_outputs = list(model.test_outputs.values())
        test_outputs_names = [var.name for var in model.test_outputs.values()]
        test_outputs_names = [
            name if name not in renamed_outputs else renamed_outputs[name]
            for name in test_outputs_names
        ]
        outputs = [
            paddle_variable_to_onnx_tensor(v, global_block)
            for v in test_outputs_names
        ]

        # Make graph
        onnx_name = 'paddlex.onnx'
        onnx_graph = helper.make_graph(
            nodes=onnx_nodes,
            name=onnx_name,
            initializer=weights,
            inputs=inputs + weights_value_info,
            outputs=outputs)

        # Make model
        onnx_model = helper.make_model(
            onnx_graph, producer_name='PaddlePaddle')

        # Model check
        checker.check_model(onnx_model)
        if onnx_model is not None:
            onnx_model_file = os.path.join(save_dir, onnx_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open(onnx_model_file, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            logging.info("Saved converted model to path: %s" % onnx_model_file)
