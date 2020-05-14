import os
import paddle.fluid as fluid
import numpy as np


def paddle_get_fc_weights(var_name="fc_0.w_0"):
    fc_weights = fluid.global_scope().find_var(var_name).get_tensor()
    return np.array(fc_weights)


def paddle_resize(extracted_features, outsize):
    resized_features = fluid.layers.resize_bilinear(extracted_features, outsize)
    return resized_features