---
comments: true
---

# PaddleX模型列表（昇腾 NPU）

PaddleX 内置了多条产线，每条产线都包含了若干模块，每个模块包含若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型存储大小，请选择存储大小较小的模型。

## [图像分类模块](../module_usage/tutorials/cv_modules/image_classification.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>Top1 Acc（%）</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>CLIP_vit_base_patch16_224</td>
<td>85.36</td>
<td>306.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>CLIP_vit_large_patch14_224</td>
<td>88.1</td>
<td>1.04 G</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_large_patch14_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_large_patch14_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ConvNeXt_base_224</td>
<td>83.84</td>
<td>313.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_base_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_base_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ConvNeXt_base_384</td>
<td>84.90</td>
<td>313.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_base_384_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_base_384_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ConvNeXt_large_224</td>
<td>84.26</td>
<td>700.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_large_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_large_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ConvNeXt_large_384</td>
<td>85.27</td>
<td>700.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_large_384_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_large_384_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ConvNeXt_small</td>
<td>83.13</td>
<td>178.0 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_small_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_small_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ConvNeXt_tiny</td>
<td>82.03</td>
<td>101.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_tiny_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_tiny_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV1_x0_5</td>
<td>63.5</td>
<td>4.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV1_x0_25</td>
<td>51.4</td>
<td>1.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_25_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV1_x0_75</td>
<td>68.8</td>
<td>9.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_75_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_75_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV1_x1_0</td>
<td>71.0</td>
<td>15.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x1_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV2_x0_5</td>
<td>65.0</td>
<td>7.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x0_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV2_x0_25</td>
<td>53.2</td>
<td>5.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x0_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x0_25_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV2_x1_0</td>
<td>72.2</td>
<td>12.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x1_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV2_x1_5</td>
<td>74.1</td>
<td>25.0 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x1_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x1_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV2_x2_0</td>
<td>75.2</td>
<td>41.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x2_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x2_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_large_x0_5</td>
<td>69.2</td>
<td>9.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_large_x0_35</td>
<td>64.3</td>
<td>7.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_35_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_35_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_large_x0_75</td>
<td>73.1</td>
<td>14.0 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_75_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_75_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_large_x1_0</td>
<td>75.3</td>
<td>19.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x1_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_large_x1_25</td>
<td>76.4</td>
<td>26.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x1_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x1_25_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_small_x0_5</td>
<td>59.2</td>
<td>6.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_small_x0_35</td>
<td>53.0</td>
<td>6.0 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_35_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_35_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_small_x0_75</td>
<td>66.0</td>
<td>8.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_75_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_75_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_small_x1_0</td>
<td>68.2</td>
<td>10.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_small_x1_25</td>
<td>70.7</td>
<td>13.0 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x1_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_25_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV4_conv_large</td>
<td>83.4</td>
<td>125.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_large_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_large_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV4_conv_medium</td>
<td>79.9</td>
<td>37.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_medium_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_medium_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV4_conv_small</td>
<td>74.6</td>
<td>14.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_small_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_small_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV4_hybrid_large</td>
<td>83.8</td>
<td>145.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_hybrid_large_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_hybrid_large_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV4_hybrid_medium</td>
<td>80.5</td>
<td>42.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_hybrid_medium_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_hybrid_medium_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNet_base</td>
<td>85.0</td>
<td>249.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_base_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_base_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNet_small</td>
<td>81.51</td>
<td>86.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_small_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_small_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNet_tiny</td>
<td>79.83</td>
<td>52.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_tiny_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_tiny_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B0</td>
<td>77.77</td>
<td>21.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B1</td>
<td>79.18</td>
<td>22.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B1_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B1_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B2</td>
<td>81.74</td>
<td>39.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B2_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B2_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B3</td>
<td>82.98</td>
<td>57.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B3_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B3_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B4</td>
<td>83.57</td>
<td>70.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B5</td>
<td>84.75</td>
<td>140.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B6</td>
<td>86.30</td>
<td>268.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x0_5</td>
<td>63.14</td>
<td>6.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x0_25</td>
<td>51.86</td>
<td>5.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x0_35</td>
<td>58.09</td>
<td>5.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_35_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_35_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x0_75</td>
<td>68.18</td>
<td>8.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_75_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_75_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x1_0</td>
<td>71.32</td>
<td>10.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x1_5</td>
<td>73.71</td>
<td>16.0 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x2_0</td>
<td>75.18</td>
<td>23.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x2_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x2_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x2_5</td>
<td>76.60</td>
<td>32.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x2_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x2_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNetV2_base</td>
<td>77.05</td>
<td>23.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_base_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_base_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNetV2_large</td>
<td>78.51</td>
<td>37.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_large_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_large_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNetV2_small</td>
<td>73.97</td>
<td>14.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_small_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_small_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet18_vd</td>
<td>72.3</td>
<td>41.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet18_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet18_vd_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet18</td>
<td>71.0</td>
<td>41.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet18_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet18_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet34_vd</td>
<td>76.0</td>
<td>77.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet34_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet34_vd_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet34</td>
<td>74.6</td>
<td>77.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet34_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet34_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet50_vd</td>
<td>79.1</td>
<td>90.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_vd_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet50</td>
<td>76.5</td>
<td>90.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet101_vd</td>
<td>80.2</td>
<td>158.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet101_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet101_vd_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet101</td>
<td>77.6</td>
<td>158.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet101_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet101_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet152_vd</td>
<td>80.6</td>
<td>214.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet152_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet152_vd_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet152</td>
<td>78.3</td>
<td>214.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet152_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet152_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet200_vd</td>
<td>80.9</td>
<td>266.0 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet200_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet200_vd_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SwinTransformer_base_patch4_window7_224</td>
<td>83.37</td>
<td>310.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_base_patch4_window7_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_base_patch4_window7_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SwinTransformer_base_patch4_window12_384</td>
<td>84.17</td>
<td>311.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_base_patch4_window12_384_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_base_patch4_window12_384_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SwinTransformer_large_patch4_window7_224</td>
<td>86.19</td>
<td>694.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_large_patch4_window7_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_large_patch4_window7_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SwinTransformer_large_patch4_window12_384</td>
<td>87.06</td>
<td>696.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_large_patch4_window12_384_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_large_patch4_window12_384_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SwinTransformer_small_patch4_window7_224</td>
<td>83.21</td>
<td>175.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_small_patch4_window7_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_small_patch4_window7_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SwinTransformer_tiny_patch4_window7_224</td>
<td>81.10</td>
<td>100.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_tiny_patch4_window7_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>StarNet-S1</td>
<td>73.6</td>
<td>11.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S1_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S1_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>StarNet-S2</td>
<td>74.8</td>
<td>14.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S2_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S2_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>StarNet-S3</td>
<td>77.0</td>
<td>22.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S3_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S3_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>StarNet-S4</td>
<td>79.0</td>
<td>28.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S4_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S4_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterNet-L</td>
<td>83.5</td>
<td>357.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterNet-M</td>
<td>83.0</td>
<td>204.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-M_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterNet-S</td>
<td>81.3</td>
<td>119.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-S_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterNet-T0</td>
<td>71.9</td>
<td>15.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterNet-T1</td>
<td>75.9</td>
<td>29.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T1_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T1_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterNet-T2</td>
<td>79.1</td>
<td>57.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T2_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T2_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为</b>[ImageNet-1k](https://www.image-net.org/index.php)<b>验证集 Top1 Acc。</b>

## [图像多标签分类模块](../module_usage/tutorials/cv_modules/image_multilabel_classification.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>CLIP_vit_base_patch16_448_ML</td>
<td>89.15</td>
<td>325.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_448_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_448_ML_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B0_ML</td>
<td>80.98</td>
<td>39.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_ML_pretrained.pdparams">训练模型</a></td>
<tr>
<td>PP-HGNetV2-B4_ML</td>
<td>87.96</td>
<td>88.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_ML_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B6_ML</td>
<td>91.06</td>
<td>286.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_ML_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x1_0_ML</td>
<td>77.96</td>
<td>29.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_ML_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet50_ML</td>
<td>83.42</td>
<td>108.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_ML_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 [COCO2017](https://cocodataset.org/#home) 的多标签分类任务mAP。</b>

## [行人属性模块](../module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mA（%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_pedestrian_attribute</td>
<td>92.2</td>
<td>6.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_pedestrian_attribute_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pedestrian_attribute_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 PaddleX 内部自建数据集mA。</b>

## [车辆属性模块](../module_usage/tutorials/cv_modules/vehicle_attribute_recognition.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mA（%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_vehicle_attribute</td>
<td>91.7</td>
<td>6.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_vehicle_attribute_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_vehicle_attribute_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 VeRi 数据集 mA。</b>

## [目标检测模块](../module_usage/tutorials/cv_modules/object_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>Cascade-FasterRCNN-ResNet50-FPN</td>
<td>41.1</td>
<td>245.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-FasterRCNN-ResNet50-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-FasterRCNN-ResNet50-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
<td>45.0</td>
<td>246.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>CenterNet-DLA-34</td>
<td>37.6</td>
<td>75.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CenterNet-DLA-34_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CenterNet-DLA-34_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>CenterNet-ResNet50</td>
<td>38.9</td>
<td>319.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CenterNet-ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CenterNet-ResNet50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>DETR-R50</td>
<td>42.3</td>
<td>159.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DETR-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DETR-R50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet34-FPN</td>
<td>37.8</td>
<td>137.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet34-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet34-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet50</td>
<td>36.7</td>
<td>120.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet50-FPN</td>
<td>38.4</td>
<td>148.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet50-vd-FPN</td>
<td>39.5</td>
<td>148.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-vd-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-vd-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
<td>41.4</td>
<td>148.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet101</td>
<td>39.0</td>
<td>188.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet101_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet101_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet101-FPN</td>
<td>41.4</td>
<td>216.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet101-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet101-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNeXt101-vd-FPN</td>
<td>43.4</td>
<td>360.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNeXt101-vd-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNeXt101-vd-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-Swin-Tiny-FPN</td>
<td>42.6</td>
<td>159.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-Swin-Tiny-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-Swin-Tiny-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FCOS-ResNet50</td>
<td>39.6</td>
<td>124.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FCOS-ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FCOS-ResNet50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PicoDet-L</td>
<td>42.6</td>
<td>20.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PicoDet-M</td>
<td>37.5</td>
<td>16.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-M_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PicoDet-S</td>
<td>29.1</td>
<td>4.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PicoDet-XS</td>
<td>26.2</td>
<td>5.7M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-XS_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-XS_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_plus-L</td>
<td>52.9</td>
<td>185.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_plus-M</td>
<td>49.8</td>
<td>83.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-M_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_plus-S</td>
<td>43.7</td>
<td>28.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_plus-X</td>
<td>54.7</td>
<td>349.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-X_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-X_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RT-DETR-H</td>
<td>56.3</td>
<td>435.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RT-DETR-L</td>
<td>53.0</td>
<td>113.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RT-DETR-R18</td>
<td>46.5</td>
<td>70.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-R18_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-R18_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RT-DETR-R50</td>
<td>53.1</td>
<td>149.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-R50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RT-DETR-X</td>
<td>54.8</td>
<td>232.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-X_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-X_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOv3-DarkNet53</td>
<td>39.1</td>
<td>219.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-DarkNet53_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-DarkNet53_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOv3-MobileNetV3</td>
<td>31.4</td>
<td>83.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-MobileNetV3_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-MobileNetV3_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOv3-ResNet50_vd_DCN</td>
<td>40.6</td>
<td>163.0 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-ResNet50_vd_DCN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-ResNet50_vd_DCN_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为</b>[COCO2017](https://cocodataset.org/#home)<b>验证集 mAP(0.5:0.95)。</b>

## [小目标检测模块](../module_usage/tutorials/cv_modules/small_object_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-YOLOE_plus_SOD-S</td>
<td>25.1</td>
<td>77.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-S_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_plus_SOD-L</td>
<td>31.9</td>
<td>325.0 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_plus_SOD-largesize-L</td>
<td>42.7</td>
<td>340.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-largesize-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-largesize-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOX-S</td>
<td>40.4</td>
<td>32.0 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-S_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOX-T</td>
<td>32.9</td>
<td>18.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-T_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-T_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOX-M</td>
<td>46.9</td>
<td>90.0 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-M_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOX-N</td>
<td>26.1</td>
<td>3.4M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-N_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-N_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset)<b> 验证集 mAP(0.5:0.95)。</b>

## [行人检测模块](../module_usage/tutorials/cv_modules/human_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-YOLOE-L_human</td>
<td>48.0</td>
<td>196.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-L_human_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-L_human_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE-S_human</td>
<td>42.5</td>
<td>28.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-S_human_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-S_human_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[CrowdHuman](https://bj.bcebos.com/v1/paddledet/data/crowdhuman.zip)<b> 验证集 mAP(0.5:0.95)。</b>

## [语义分割模块](../module_usage/tutorials/cv_modules/semantic_segmentation.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mloU（%）</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>Deeplabv3_Plus-R50</td>
<td>80.36</td>
<td>94.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Deeplabv3_Plus-R101</td>
<td>81.10</td>
<td>162.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R101_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R101_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Deeplabv3-R50</td>
<td>79.90</td>
<td>138.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3-R50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Deeplabv3-R101</td>
<td>80.85</td>
<td>205.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R101_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3-R101_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>OCRNet_HRNet-W18</td>
<td>80.67</td>
<td>43.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W18_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W18_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>OCRNet_HRNet-W48</td>
<td>82.15</td>
<td>249.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W48_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W48_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LiteSeg-T</td>
<td>73.10</td>
<td>28.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-T_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-T_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SegFormer-B0 (slice)</td>
<td>76.73</td>
<td>13.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B0 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B0 (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SegFormer-B1 (slice)</td>
<td>78.35</td>
<td>48.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B1 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B1 (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SegFormer-B2 (slice)</td>
<td>81.60</td>
<td>96.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B2 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B2 (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SegFormer-B3 (slice)</td>
<td>82.47</td>
<td>167.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B3 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B3 (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SegFormer-B4 (slice)</td>
<td>82.38</td>
<td>226.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B4 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B4 (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SegFormer-B5 (slice)</td>
<td>82.58</td>
<td>229.7 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B5 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B5 (slice)_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为</b>[Cityscapes](https://www.cityscapes-dataset.com/)<b>数据集 mloU。</b>
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mloU（%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>SeaFormer_base(slice)</td>
<td>40.92</td>
<td>30.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_base(slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_base(slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SeaFormer_large (slice)</td>
<td>43.66</td>
<td>49.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_large (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_large (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SeaFormer_small (slice)</td>
<td>38.73</td>
<td>14.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_small (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_small (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SeaFormer_tiny (slice)</td>
<td>34.58</td>
<td>6.1M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_tiny (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_tiny (slice)_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)<b> 数据集, slice 表示对输入图像进行了切图操作。</b>

## 实例分割模块
<table>
<thead>
<tr>
<th>模型名称</th>
<th>Mask AP</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>Mask-RT-DETR-H</td>
<td>50.6</td>
<td>449.9</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-H_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-H_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Mask-RT-DETR-L</td>
<td>45.7</td>
<td>113.6</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Mask-RT-DETR-M</td>
<td>42.7</td>
<td>66.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-M_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Mask-RT-DETR-S</td>
<td>41.0</td>
<td>51.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-S_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Mask-RT-DETR-X</td>
<td>47.5</td>
<td>237.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-X_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-X_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Cascade-MaskRCNN-ResNet50-FPN</td>
<td>36.3</td>
<td>254.8</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-MaskRCNN-ResNet50-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-MaskRCNN-ResNet50-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN</td>
<td>39.1</td>
<td>254.7</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MaskRCNN-ResNet50-FPN</td>
<td>35.6</td>
<td>157.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MaskRCNN-ResNet50-vd-FPN</td>
<td>36.4</td>
<td>157.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50-vd-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50-vd-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MaskRCNN-ResNet50</td>
<td>32.8</td>
<td>127.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MaskRCNN-ResNet101-FPN</td>
<td>36.6</td>
<td>225.4 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet101-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet101-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MaskRCNN-ResNet101-vd-FPN</td>
<td>38.1</td>
<td>225.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet101-vd-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet101-vd-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MaskRCNN-ResNeXt101-vd-FPN</td>
<td>39.5</td>
<td>370.0 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNeXt101-vd-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNeXt101-vd-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_seg-S</td>
<td>32.5</td>
<td>31.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_seg-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_seg-S_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为</b>[COCO2017](https://cocodataset.org/#home)<b>验证集 Mask AP(0.5:0.95)。</b>

## [图像特征模块](../module_usage/tutorials/cv_modules/image_feature.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>recall@1（%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-ShiTuV2_rec</td>
<td>84.2</td>
<td>16.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_base</td>
<td>88.69</td>
<td>306.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_CLIP_vit_base_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_base_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_large</td>
<td>91.03</td>
<td>1.05 G</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_CLIP_vit_large_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_large_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 AliProducts recall@1。</b>

## [主体检测模块](../module_usage/tutorials/cv_modules/mainbody_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-ShiTuV2_det</td>
<td>41.5</td>
<td>27.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_det_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 [PaddleClas主体检测数据集](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/training/PP-ShiTu/mainbody_detection.md) mAP(0.5:0.95)。</b>

## [车辆检测模块](../module_usage/tutorials/cv_modules/vehicle_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-YOLOE-L_vehicle</td>
<td>63.9</td>
<td>196.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-L_vehicle_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-L_vehicle_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE-S_vehicle</td>
<td>61.3</td>
<td>28.8 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-S_vehicle_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-S_vehicle_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[PPVehicle](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppvehicle)<b> 验证集 mAP(0.5:0.95)。</b>

## [人脸检测模块](../module_usage/tutorials/cv_modules/face_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th style="text-align: center;">AP (%)<br/>Easy/Medium/Hard</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PicoDet_LCNet_x2_5_face</td>
<td style="text-align: center;">93.7/90.7/68.1</td>
<td>28.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_LCNet_x2_5_face_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_LCNet_x2_5_face_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
**注：以上精度指标是在WIDER-FACE验证集上，以640
\*640作为输入尺寸评估得到的。**

## [异常检测模块](../module_usage/tutorials/cv_modules/anomaly_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>Avg（%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>STFPM</td>
<td>96.2</td>
<td>21.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/STFPM_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/STFPM_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)<b> 验证集 平均异常分数。</b>

## [文本检测模块](../module_usage/tutorials/ocr_modules/text_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>检测Hmean（%）</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td>77.79</td>
<td>4.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-OCRv4_server_det</td>
<td>82.69</td>
<td>100.1M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。</b>

## [文本识别模块](../module_usage/tutorials/ocr_modules/text_recognition.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>识别Avg Accuracy(%)</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td>78.20</td>
<td>10.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-OCRv4_server_rec</td>
<td>79.20</td>
<td>71.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。</b>
<table>
<thead>
<tr>
<th>模型名称</th>
<th>识别Avg Accuracy(%)</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>ch_SVTRv2_rec</td>
<td>68.81</td>
<td>73.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_SVTRv2_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 [PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务](https://aistudio.baidu.com/competition/detail/1131/0/introduction)A榜。</b>
<table>
<thead>
<tr>
<th>模型名称</th>
<th>识别Avg Accuracy(%)</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>ch_RepSVTR_rec</td>
<td>65.07</td>
<td>22.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_RepSVTR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 [PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务](https://aistudio.baidu.com/competition/detail/1131/0/introduction)B榜。</b>

## [表格结构识别模块](../module_usage/tutorials/ocr_modules/table_structure_recognition.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>精度（%）</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>SLANet</td>
<td>76.31</td>
<td>6.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SLANet_plus</td>
<td>63.69</td>
<td>6.9 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_plus_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标测量自PubtabNet英文表格识别数据集。</b>

## [图像矫正模块](../module_usage/tutorials/ocr_modules/text_image_unwarping.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>MS-SSIM （%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td>
<td>54.40</td>
<td>30.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UVDoc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标测量自 </b><b>PaddleX自建的图像矫正数据集</b><b>。</b>

## [印章文本检测模块](../module_usage/tutorials/ocr_modules/seal_text_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>检测Hmean（%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_seal_det</td>
<td>96.47</td>
<td>4.7M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-OCRv4_server_seal_det</td>
<td>98.21</td>
<td>108.3 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 PaddleX 自建的印章数据集，包含500印章图像。</b>

## [文档方向分类模块](../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>Top-1 Acc（%）</th>
<th>模型存储大小</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td>99.26</td>
<td>7.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 PaddleX 内部自建数据集 Top-1 Acc 。</b>

## [版面区域检测模块](../module_usage/tutorials/ocr_modules/layout_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td>
<td>86.8</td>
<td>7.4M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PicoDet-L_layout_3cls</td>
<td>89.3</td>
<td>22.6 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RT-DETR-H_layout_3cls</td>
<td>95.9</td>
<td>470.1 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RT-DETR-H_layout_17cls</td>
<td>92.6</td>
<td>470.2 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 PaddleOCR 自建的版面区域分析数据集，包含 1w 张图片。</b>

## [时序预测模块](../module_usage/tutorials/time_series_modules/time_series_forecasting.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mse</th>
<th>mae</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>DLinear</td>
<td>0.382</td>
<td>0.394</td>
<td>72K</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DLinear_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DLinear_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>NLinear</td>
<td>0.386</td>
<td>0.392</td>
<td>40K</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/NLinear_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/NLinear_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Nonstationary</td>
<td>0.600</td>
<td>0.515</td>
<td>55.5 M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Nonstationary_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Nonstationary_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PatchTST</td>
<td>0.385</td>
<td>0.397</td>
<td>2.0M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PatchTST_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PatchTST_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RLinear</td>
<td>0.384</td>
<td>0.392</td>
<td>40K</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RLinear_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RLinear_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>TiDE</td>
<td>0.405</td>
<td>0.412</td>
<td>31.7M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TiDE_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TiDE_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>TimesNet</td>
<td>0.417</td>
<td>0.431</td>
<td>4.9M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标测量自</b>[ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar)<b>数据集 </b><b>（在测试集test.csv上的评测结果）</b><b>。</b>

## 时序异常检测模块
<table>
<thead>
<tr>
<th>模型名称</th>
<th>precison</th>
<th>recall</th>
<th>f1_score</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>AutoEncoder_ad</td>
<td>99.36</td>
<td>84.36</td>
<td>91.25</td>
<td>52K</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/AutoEncoder_ad_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/AutoEncoder_ad_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>DLinear_ad</td>
<td>98.98</td>
<td>93.96</td>
<td>96.41</td>
<td>112K</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DLinear_ad_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DLinear_ad_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Nonstationary_ad</td>
<td>98.55</td>
<td>88.95</td>
<td>93.51</td>
<td>1.8M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Nonstationary_ad_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Nonstationary_ad_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PatchTST_ad</td>
<td>98.78</td>
<td>90.70</td>
<td>94.57</td>
<td>320K</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PatchTST_ad_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PatchTST_ad_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>TimesNet_ad</td>
<td>98.37</td>
<td>94.80</td>
<td>96.56</td>
<td>1.3M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_ad_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_ad_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标测量自</b>[PSM](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar)<b>数据集。</b>

## 时序分类模块
<table>
<thead>
<tr>
<th>模型名称</th>
<th>acc(%)</th>
<th>模型存储大小（M)</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>TimesNet_cls</td>
<td>87.5</td>
<td>792K</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_cls_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标测量自UWaveGestureLibrary：[训练](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TRAIN.csv)、[评测](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TEST.csv)数据集。</b>
