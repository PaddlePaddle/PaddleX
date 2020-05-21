#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import os.path as osp
import paddle.fluid as fluid
import paddlex as pdx
import numpy as np
from paddle.fluid.param_attr import ParamAttr
from paddlex.interpret.as_data_reader.readers import preprocess_image

def gen_user_home():
    if "HOME" in os.environ:
        home_path = os.environ["HOME"]
        if os.path.exists(home_path) and os.path.isdir(home_path):
            return home_path
    return os.path.expanduser('~')


def paddle_get_fc_weights(var_name="fc_0.w_0"):
    fc_weights = fluid.global_scope().find_var(var_name).get_tensor()
    return np.array(fc_weights)


def paddle_resize(extracted_features, outsize):
    resized_features = fluid.layers.resize_bilinear(extracted_features, outsize)
    return resized_features


def compute_features_for_kmeans(data_content):
    root_path = gen_user_home()
    root_path = osp.join(root_path, '.paddlex')
    h_pre_models = osp.join(root_path, "pre_models")
    if not osp.exists(h_pre_models):
        if not osp.exists(root_path):
            os.makedirs(root_path)
        url = "https://bj.bcebos.com/paddlex/interpret/pre_models.tar.gz"
        pdx.utils.download_and_decompress(url, path=root_path)
    def conv_bn_layer(input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None,
                      is_test=True,
                      global_name=''):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=global_name + name + "_weights"),
            bias_attr=False,
            name=global_name + name + '.conv2d.output.1')
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=global_name + bn_name + '.output.1',
            param_attr=ParamAttr(global_name + bn_name + '_scale'),
            bias_attr=ParamAttr(global_name + bn_name + '_offset'),
            moving_mean_name=global_name + bn_name + '_mean',
            moving_variance_name=global_name + bn_name + '_variance',
            use_global_stats=is_test
        )

    startup_prog = fluid.default_startup_program().clone(for_test=True)
    prog = fluid.Program()
    with fluid.program_guard(prog, startup_prog):
        with fluid.unique_name.guard():
            image_op = fluid.data(name='image', shape=[None, 3, 224, 224], dtype='float32')

            conv = conv_bn_layer(
                input=image_op,
                num_filters=32,
                filter_size=3,
                stride=2,
                act='relu',
                name='conv1_1')
            conv = conv_bn_layer(
                input=conv,
                num_filters=32,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv1_2')
            conv = conv_bn_layer(
                input=conv,
                num_filters=64,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv1_3')
            extracted_features = conv
            resized_features = fluid.layers.resize_bilinear(extracted_features, image_op.shape[2:])

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id)
    # place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    fluid.io.load_persistables(exe, h_pre_models, prog)

    images = preprocess_image(data_content)  # transpose to [N, 3, H, W], scaled to [0.0, 1.0]
    result = exe.run(prog, fetch_list=[resized_features], feed={'image': images})

    return result[0][0]
