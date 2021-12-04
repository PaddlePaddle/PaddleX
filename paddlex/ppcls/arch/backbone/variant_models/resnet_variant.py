from paddle.nn import Conv2D
from paddlex.ppcls.arch.backbone.legendary_models.resnet import ResNet50, MODEL_URLS, _load_pretrained

__all__ = ["ResNet50_last_stage_stride1"]


def ResNet50_last_stage_stride1(pretrained=False, use_ssld=False, **kwargs):
    def replace_function(conv):
        new_conv = Conv2D(
            in_channels=conv._in_channels,
            out_channels=conv._out_channels,
            kernel_size=conv._kernel_size,
            stride=1,
            padding=conv._padding,
            groups=conv._groups,
            bias_attr=conv._bias_attr)
        return new_conv

    match_re = "conv2d_4[4|6]"
    model = ResNet50(pretrained=False, use_ssld=use_ssld, **kwargs)
    model.replace_sub(match_re, replace_function, True)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet50"], use_ssld)
    return model
