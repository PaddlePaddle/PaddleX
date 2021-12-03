from abc import ABC
from paddle import nn
import re


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class TheseusLayer(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(TheseusLayer, self).__init__()
        self.res_dict = {}
        self.res_name = self.full_name()

    # stop doesn't work when stop layer has a parallel branch.
    def stop_after(self, stop_layer_name: str):
        after_stop = False
        for layer_i in self._sub_layers:
            if after_stop:
                self._sub_layers[layer_i] = Identity()
                continue
            layer_name = self._sub_layers[layer_i].full_name()
            if layer_name == stop_layer_name:
                after_stop = True
                continue
            if isinstance(self._sub_layers[layer_i], TheseusLayer):
                after_stop = self._sub_layers[layer_i].stop_after(
                    stop_layer_name)
        return after_stop

    def update_res(self, return_patterns):
        for return_pattern in return_patterns:
            pattern_list = return_pattern.split(".")
            if not pattern_list:
                continue
            sub_layer_parent = self
            while len(pattern_list) > 1:
                if '[' in pattern_list[0]:
                    sub_layer_name = pattern_list[0].split('[')[0]
                    sub_layer_index = pattern_list[0].split('[')[1].split(']')[
                        0]
                    sub_layer_parent = getattr(sub_layer_parent,
                                               sub_layer_name)[sub_layer_index]
                else:
                    sub_layer_parent = getattr(sub_layer_parent,
                                               pattern_list[0], None)
                    if sub_layer_parent is None:
                        break
                if isinstance(sub_layer_parent, WrapLayer):
                    sub_layer_parent = sub_layer_parent.sub_layer
                pattern_list = pattern_list[1:]
            if sub_layer_parent is None:
                continue
            if '[' in pattern_list[0]:
                sub_layer_name = pattern_list[0].split('[')[0]
                sub_layer_index = pattern_list[0].split('[')[1].split(']')[0]
                sub_layer = getattr(sub_layer_parent,
                                    sub_layer_name)[sub_layer_index]
                if not isinstance(sub_layer, TheseusLayer):
                    sub_layer = wrap_theseus(sub_layer)
                getattr(sub_layer_parent,
                        sub_layer_name)[sub_layer_index] = sub_layer
            else:
                sub_layer = getattr(sub_layer_parent, pattern_list[0])
                if not isinstance(sub_layer, TheseusLayer):
                    sub_layer = wrap_theseus(sub_layer)
                setattr(sub_layer_parent, pattern_list[0], sub_layer)

            sub_layer.res_dict = self.res_dict
            sub_layer.res_name = return_pattern
            sub_layer.register_forward_post_hook(sub_layer._save_sub_res_hook)

    def _save_sub_res_hook(self, layer, input, output):
        self.res_dict[self.res_name] = output

    def _return_dict_hook(self, layer, input, output):
        res_dict = {"output": output}
        for res_key in list(self.res_dict):
            res_dict[res_key] = self.res_dict.pop(res_key)
        return res_dict

    def replace_sub(self, layer_name_pattern, replace_function,
                    recursive=True):
        for layer_i in self._sub_layers:
            layer_name = self._sub_layers[layer_i].full_name()
            if re.match(layer_name_pattern, layer_name):
                self._sub_layers[layer_i] = replace_function(self._sub_layers[
                    layer_i])
            if recursive:
                if isinstance(self._sub_layers[layer_i], TheseusLayer):
                    self._sub_layers[layer_i].replace_sub(
                        layer_name_pattern, replace_function, recursive)
                elif isinstance(self._sub_layers[layer_i],
                                (nn.Sequential, nn.LayerList)):
                    for layer_j in self._sub_layers[layer_i]._sub_layers:
                        self._sub_layers[layer_i]._sub_layers[
                            layer_j].replace_sub(layer_name_pattern,
                                                 replace_function, recursive)

    '''
    example of replace function:
    def replace_conv(origin_conv: nn.Conv2D):
        new_conv = nn.Conv2D(
            in_channels=origin_conv._in_channels,
            out_channels=origin_conv._out_channels,
            kernel_size=origin_conv._kernel_size,
            stride=2
        )
        return new_conv

        '''


class WrapLayer(TheseusLayer):
    def __init__(self, sub_layer):
        super(WrapLayer, self).__init__()
        self.sub_layer = sub_layer

    def forward(self, *inputs, **kwargs):
        return self.sub_layer(*inputs, **kwargs)


def wrap_theseus(sub_layer):
    wrapped_layer = WrapLayer(sub_layer)
    return wrapped_layer
