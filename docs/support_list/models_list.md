---
comments: true
---

# PaddleX模型列表（CPU/GPU）

PaddleX 内置了多条产线，每条产线都包含了若干模块，每个模块包含若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型。

## [图像分类模块](../module_usage/tutorials/cv_modules/image_classification.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>Top1 Acc（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>CLIP_vit_base_patch16_224</td>
<td>85.36</td>
<td>13.1957</td>
<td>285.493</td>
<td>306.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/CLIP_vit_base_patch16_224.yaml">CLIP_vit_base_patch16_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>CLIP_vit_large_patch14_224</td>
<td>88.1</td>
<td>51.1284</td>
<td>1131.28</td>
<td>1.04 G</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/CLIP_vit_large_patch14_224.yaml">CLIP_vit_large_patch14_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_large_patch14_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_large_patch14_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ConvNeXt_base_224</td>
<td>83.84</td>
<td>12.8473</td>
<td>1513.87</td>
<td>313.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ConvNeXt_base_224.yaml">ConvNeXt_base_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_base_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_base_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ConvNeXt_base_384</td>
<td>84.90</td>
<td>31.7607</td>
<td>3967.05</td>
<td>313.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ConvNeXt_base_384.yaml">ConvNeXt_base_384.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_base_384_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_base_384_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ConvNeXt_large_224</td>
<td>84.26</td>
<td>26.8103</td>
<td>2463.56</td>
<td>700.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ConvNeXt_large_224.yaml">ConvNeXt_large_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_large_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_large_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ConvNeXt_large_384</td>
<td>85.27</td>
<td>66.4058</td>
<td>6598.92</td>
<td>700.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ConvNeXt_large_384.yaml">ConvNeXt_large_384.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_large_384_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_large_384_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ConvNeXt_small</td>
<td>83.13</td>
<td>9.74075</td>
<td>1127.6</td>
<td>178.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ConvNeXt_small.yaml">ConvNeXt_small.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_small_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_small_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ConvNeXt_tiny</td>
<td>82.03</td>
<td>5.48923</td>
<td>672.559</td>
<td>101.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ConvNeXt_tiny.yaml">ConvNeXt_tiny.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_tiny_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_tiny_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterNet-L</td>
<td>83.5</td>
<td>23.4415</td>
<td>-</td>
<td>357.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/FasterNet-L.yaml">FasterNet-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterNet-M</td>
<td>83.0</td>
<td>21.8936</td>
<td>-</td>
<td>204.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/FasterNet-M.yaml">FasterNet-M.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-M_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterNet-S</td>
<td>81.3</td>
<td>13.0409</td>
<td>-</td>
<td>119.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/FasterNet-S.yaml">FasterNet-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-S_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterNet-T0</td>
<td>71.9</td>
<td>12.2432</td>
<td>-</td>
<td>15.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/FasterNet-T0.yaml">FasterNet-T0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterNet-T1</td>
<td>75.9</td>
<td>11.3562</td>
<td>-</td>
<td>29.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/FasterNet-T1.yaml">FasterNet-T1.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T1_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T1_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterNet-T2</td>
<td>79.1</td>
<td>10.703</td>
<td>-</td>
<td>57.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/FasterNet-T2.yaml">FasterNet-T2.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T2_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T2_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV1_x0_5</td>
<td>63.5</td>
<td>1.86754</td>
<td>7.48297</td>
<td>4.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV1_x0_5.yaml">MobileNetV1_x0_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV1_x0_25</td>
<td>51.4</td>
<td>1.83478</td>
<td>4.83674</td>
<td>1.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV1_x0_25.yaml">MobileNetV1_x0_25.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_25_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV1_x0_75</td>
<td>68.8</td>
<td>2.57903</td>
<td>10.6343</td>
<td>9.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV1_x0_75.yaml">MobileNetV1_x0_75.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_75_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_75_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV1_x1_0</td>
<td>71.0</td>
<td>2.78781</td>
<td>13.98</td>
<td>15.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV1_x1_0.yaml">MobileNetV1_x1_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x1_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV2_x0_5</td>
<td>65.0</td>
<td>4.94234</td>
<td>11.1629</td>
<td>7.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV2_x0_5.yaml">MobileNetV2_x0_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x0_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV2_x0_25</td>
<td>53.2</td>
<td>4.50856</td>
<td>9.40991</td>
<td>5.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV2_x0_25.yaml">MobileNetV2_x0_25.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x0_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x0_25_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV2_x1_0</td>
<td>72.2</td>
<td>6.12159</td>
<td>16.0442</td>
<td>12.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV2_x1_0.yaml">MobileNetV2_x1_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x1_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV2_x1_5</td>
<td>74.1</td>
<td>6.28385</td>
<td>22.5129</td>
<td>25.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV2_x1_5.yaml">MobileNetV2_x1_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x1_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x1_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV2_x2_0</td>
<td>75.2</td>
<td>6.12888</td>
<td>30.8612</td>
<td>41.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV2_x2_0.yaml">MobileNetV2_x2_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x2_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x2_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_large_x0_5</td>
<td>69.2</td>
<td>6.31302</td>
<td>14.5588</td>
<td>9.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_large_x0_5.yaml">MobileNetV3_large_x0_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_large_x0_35</td>
<td>64.3</td>
<td>5.76207</td>
<td>13.9041</td>
<td>7.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_large_x0_35.yaml">MobileNetV3_large_x0_35.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_35_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_35_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_large_x0_75</td>
<td>73.1</td>
<td>8.41737</td>
<td>16.9506</td>
<td>14.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_large_x0_75.yaml">MobileNetV3_large_x0_75.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_75_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_75_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_large_x1_0</td>
<td>75.3</td>
<td>8.64112</td>
<td>19.1614</td>
<td>19.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_large_x1_0.yaml">MobileNetV3_large_x1_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x1_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_large_x1_25</td>
<td>76.4</td>
<td>8.73358</td>
<td>22.1296</td>
<td>26.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_large_x1_25.yaml">MobileNetV3_large_x1_25.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x1_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x1_25_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_small_x0_5</td>
<td>59.2</td>
<td>5.16721</td>
<td>11.2688</td>
<td>6.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_small_x0_5.yaml">MobileNetV3_small_x0_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_small_x0_35</td>
<td>53.0</td>
<td>5.22053</td>
<td>11.0055</td>
<td>6.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_small_x0_35.yaml">MobileNetV3_small_x0_35.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_35_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_35_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_small_x0_75</td>
<td>66.0</td>
<td>5.39831</td>
<td>12.8313</td>
<td>8.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_small_x0_75.yaml">MobileNetV3_small_x0_75.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_75_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_75_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_small_x1_0</td>
<td>68.2</td>
<td>6.00993</td>
<td>12.9598</td>
<td>10.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_small_x1_0.yaml">MobileNetV3_small_x1_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV3_small_x1_25</td>
<td>70.7</td>
<td>6.9589</td>
<td>14.3995</td>
<td>13.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_small_x1_25.yaml">MobileNetV3_small_x1_25.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x1_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_25_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV4_conv_large</td>
<td>83.4</td>
<td>12.5485</td>
<td>51.6453</td>
<td>125.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV4_conv_large.yaml">MobileNetV4_conv_large.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_large_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_large_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV4_conv_medium</td>
<td>79.9</td>
<td>9.65509</td>
<td>26.6157</td>
<td>37.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV4_conv_medium.yaml">MobileNetV4_conv_medium.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_medium_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_medium_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV4_conv_small</td>
<td>74.6</td>
<td>5.24172</td>
<td>11.0893</td>
<td>14.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV4_conv_small.yaml">MobileNetV4_conv_small.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_small_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_small_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV4_hybrid_large</td>
<td>83.8</td>
<td>20.0726</td>
<td>213.769</td>
<td>145.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV4_hybrid_large.yaml">MobileNetV4_hybrid_large.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_hybrid_large_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_hybrid_large_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MobileNetV4_hybrid_medium</td>
<td>80.5</td>
<td>19.7543</td>
<td>62.2624</td>
<td>42.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV4_hybrid_medium.yaml">MobileNetV4_hybrid_medium.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_hybrid_medium_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_hybrid_medium_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNet_base</td>
<td>85.0</td>
<td>14.2969</td>
<td>327.114</td>
<td>249.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNet_base.yaml">PP-HGNet_base.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_base_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_base_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNet_small</td>
<td>81.51</td>
<td>5.50661</td>
<td>119.041</td>
<td>86.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNet_small.yaml">PP-HGNet_small.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_small_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_small_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNet_tiny</td>
<td>79.83</td>
<td>5.22006</td>
<td>69.396</td>
<td>52.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNet_tiny.yaml">PP-HGNet_tiny.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_tiny_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_tiny_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B0</td>
<td>77.77</td>
<td>6.53694</td>
<td>23.352</td>
<td>21.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B0.yaml">PP-HGNetV2-B0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B1</td>
<td>79.18</td>
<td>6.56034</td>
<td>27.3099</td>
<td>22.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B1.yaml">PP-HGNetV2-B1.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B1_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B1_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B2</td>
<td>81.74</td>
<td>9.60494</td>
<td>43.1219</td>
<td>39.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B2.yaml">PP-HGNetV2-B2.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B2_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B2_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B3</td>
<td>82.98</td>
<td>11.0042</td>
<td>55.1367</td>
<td>57.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B3.yaml">PP-HGNetV2-B3.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B3_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B3_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B4</td>
<td>83.57</td>
<td>9.66407</td>
<td>54.2462</td>
<td>70.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B4.yaml">PP-HGNetV2-B4.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B5</td>
<td>84.75</td>
<td>15.7091</td>
<td>115.926</td>
<td>140.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B5.yaml">PP-HGNetV2-B5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B6</td>
<td>86.30</td>
<td>21.226</td>
<td>255.279</td>
<td>268.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B6.yaml">PP-HGNetV2-B6.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x0_5</td>
<td>63.14</td>
<td>3.67722</td>
<td>6.66857</td>
<td>6.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x0_5.yaml">PP-LCNet_x0_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x0_25</td>
<td>51.86</td>
<td>2.65341</td>
<td>5.81357</td>
<td>5.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x0_25.yaml">PP-LCNet_x0_25.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x0_35</td>
<td>58.09</td>
<td>2.7212</td>
<td>6.28944</td>
<td>5.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x0_35.yaml">PP-LCNet_x0_35.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_35_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_35_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x0_75</td>
<td>68.18</td>
<td>3.91032</td>
<td>8.06953</td>
<td>8.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x0_75.yaml">PP-LCNet_x0_75.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_75_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_75_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x1_0</td>
<td>71.32</td>
<td>3.84845</td>
<td>9.23735</td>
<td>10.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml">PP-LCNet_x1_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x1_5</td>
<td>73.71</td>
<td>3.97666</td>
<td>12.3457</td>
<td>16.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x1_5.yaml">PP-LCNet_x1_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x2_0</td>
<td>75.18</td>
<td>4.07556</td>
<td>16.2752</td>
<td>23.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x2_0.yaml">PP-LCNet_x2_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x2_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x2_0_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x2_5</td>
<td>76.60</td>
<td>4.06028</td>
<td>21.5063</td>
<td>32.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x2_5.yaml">PP-LCNet_x2_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x2_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x2_5_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNetV2_base</td>
<td>77.05</td>
<td>5.23428</td>
<td>19.6005</td>
<td>23.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNetV2_base.yaml">PP-LCNetV2_base.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_base_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_base_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNetV2_large</td>
<td>78.51</td>
<td>6.78335</td>
<td>30.4378</td>
<td>37.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNetV2_large.yaml">PP-LCNetV2_large.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_large_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_large_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNetV2_small</td>
<td>73.97</td>
<td>3.89762</td>
<td>13.0273</td>
<td>14.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNetV2_small.yaml">PP-LCNetV2_small.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_small_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_small_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet18_vd</td>
<td>72.3</td>
<td>3.53048</td>
<td>31.3014</td>
<td>41.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet18_vd.yaml">ResNet18_vd.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet18_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet18_vd_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet18</td>
<td>71.0</td>
<td>2.4868</td>
<td>27.4601</td>
<td>41.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet18.yaml">ResNet18.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet18_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet18_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet34_vd</td>
<td>76.0</td>
<td>5.60675</td>
<td>56.0653</td>
<td>77.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet34_vd.yaml">ResNet34_vd.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet34_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet34_vd_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet34</td>
<td>74.6</td>
<td>4.16902</td>
<td>51.925</td>
<td>77.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet34.yaml">ResNet34.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet34_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet34_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet50_vd</td>
<td>79.1</td>
<td>10.1885</td>
<td>68.446</td>
<td>90.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet50_vd.yaml">ResNet50_vd.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_vd_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet50</td>
<td>76.5</td>
<td>9.62383</td>
<td>64.8135</td>
<td>90.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet50.yaml">ResNet50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet101_vd</td>
<td>80.2</td>
<td>20.0563</td>
<td>124.85</td>
<td>158.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet101_vd.yaml">ResNet101_vd.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet101_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet101_vd_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet101</td>
<td>77.6</td>
<td>19.2297</td>
<td>121.006</td>
<td>158.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet101.yaml">ResNet101.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet101_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet101_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet152_vd</td>
<td>80.6</td>
<td>29.6439</td>
<td>181.678</td>
<td>214.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet152_vd.yaml">ResNet152_vd.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet152_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet152_vd_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet152</td>
<td>78.3</td>
<td>30.0461</td>
<td>177.707</td>
<td>214.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet152.yaml">ResNet152.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet152_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet152_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet200_vd</td>
<td>80.9</td>
<td>39.1628</td>
<td>235.185</td>
<td>266.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet200_vd.yaml">ResNet200_vd.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet200_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet200_vd_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>StarNet-S1</td>
<td>73.6</td>
<td>9.895</td>
<td>23.0465</td>
<td>11.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/StarNet-S1.yaml">StarNet-S1.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S1_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S1_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>StarNet-S2</td>
<td>74.8</td>
<td>7.91279</td>
<td>21.9571</td>
<td>14.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/StarNet-S2.yaml">StarNet-S2.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S2_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S2_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>StarNet-S3</td>
<td>77.0</td>
<td>10.7531</td>
<td>30.7656</td>
<td>22.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/StarNet-S3.yaml">StarNet-S3.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S3_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S3_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>StarNet-S4</td>
<td>79.0</td>
<td>15.2868</td>
<td>43.2497</td>
<td>28.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/StarNet-S4.yaml">StarNet-S4.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S4_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S4_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SwinTransformer_base_patch4_window7_224</td>
<td>83.37</td>
<td>16.9848</td>
<td>383.83</td>
<td>310.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_base_patch4_window7_224.yaml">SwinTransformer_base_patch4_window7_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_base_patch4_window7_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_base_patch4_window7_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SwinTransformer_base_patch4_window12_384</td>
<td>84.17</td>
<td>37.2855</td>
<td>1178.63</td>
<td>311.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_base_patch4_window12_384.yaml">SwinTransformer_base_patch4_window12_384.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_base_patch4_window12_384_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_base_patch4_window12_384_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SwinTransformer_large_patch4_window7_224</td>
<td>86.19</td>
<td>27.5498</td>
<td>689.729</td>
<td>694.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_large_patch4_window7_224.yaml">SwinTransformer_large_patch4_window7_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_large_patch4_window7_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_large_patch4_window7_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SwinTransformer_large_patch4_window12_384</td>
<td>87.06</td>
<td>74.1768</td>
<td>2105.22</td>
<td>696.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_large_patch4_window12_384.yaml">SwinTransformer_large_patch4_window12_384.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_large_patch4_window12_384_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_large_patch4_window12_384_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SwinTransformer_small_patch4_window7_224</td>
<td>83.21</td>
<td>16.3982</td>
<td>285.56</td>
<td>175.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_small_patch4_window7_224.yaml">SwinTransformer_small_patch4_window7_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_small_patch4_window7_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_small_patch4_window7_224_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SwinTransformer_tiny_patch4_window7_224</td>
<td>81.10</td>
<td>8.54846</td>
<td>156.306</td>
<td>100.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_tiny_patch4_window7_224.yaml">SwinTransformer_tiny_patch4_window7_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_tiny_patch4_window7_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[ImageNet-1k](https://www.image-net.org/index.php)<b> 验证集 Top1 Acc。</b>

## [图像多标签分类模块](../module_usage/tutorials/cv_modules/image_multilabel_classification.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>CLIP_vit_base_patch16_448_ML</td>
<td>89.15</td>
<td>-</td>
<td>-</td>
<td>325.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_multilabel_classification/CLIP_vit_base_patch16_448_ML.yaml">CLIP_vit_base_patch16_448_ML.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_448_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_448_ML_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B0_ML</td>
<td>80.98</td>
<td>-</td>
<td>-</td>
<td>39.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_multilabel_classification/PP-HGNetV2-B0_ML.yaml">PP-HGNetV2-B0_ML.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_ML_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B4_ML</td>
<td>87.96</td>
<td>-</td>
<td>-</td>
<td>88.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_multilabel_classification/PP-HGNetV2-B4_ML.yaml">PP-HGNetV2-B4_ML.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_ML_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-HGNetV2-B6_ML</td>
<td>91.06</td>
<td>-</td>
<td>-</td>
<td>286.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_multilabel_classification/PP-HGNetV2-B6_ML.yaml">PP-HGNetV2-B6_ML.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_ML_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LCNet_x1_0_ML</td>
<td>77.96</td>
<td>-</td>
<td>-</td>
<td>29.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_multilabel_classification/PP-LCNet_x1_0_ML.yaml">PP-LCNet_x1_0_ML.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_ML_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet50_ML</td>
<td>83.42</td>
<td>-</td>
<td>-</td>
<td>108.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_multilabel_classification/ResNet50_ML.yaml">ResNet50_ML.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_ML_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为[COCO2017](https://cocodataset.org/#home)的多标签分类任务mAP。</b>

## [行人属性模块](../module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mA（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_pedestrian_attribute</td>
<td>92.2</td>
<td>3.84845</td>
<td>9.23735</td>
<td>6.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/pedestrian_attribute_recognition/PP-LCNet_x1_0_pedestrian_attribute.yaml">PP-LCNet_x1_0_pedestrian_attribute.yaml</a></td>
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
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_vehicle_attribute</td>
<td>91.7</td>
<td>3.84845</td>
<td>9.23735</td>
<td>6.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/vehicle_attribute_recognition/PP-LCNet_x1_0_vehicle_attribute.yaml">PP-LCNet_x1_0_vehicle_attribute.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_vehicle_attribute_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_vehicle_attribute_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 VeRi 数据集 mA。</b>

## [图像特征模块](../module_usage/tutorials/cv_modules/image_feature.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>recall@1（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-ShiTuV2_rec</td>
<td>84.2</td>
<td>5.23428</td>
<td>19.6005</td>
<td>16.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml">PP-ShiTuV2_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_base</td>
<td>88.69</td>
<td>13.1957</td>
<td>285.493</td>
<td>306.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_feature/PP-ShiTuV2_rec_CLIP_vit_base.yaml">PP-ShiTuV2_rec_CLIP_vit_base.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_CLIP_vit_base_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_base_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_large</td>
<td>91.03</td>
<td>51.1284</td>
<td>1131.28</td>
<td>1.05 G</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_feature/PP-ShiTuV2_rec_CLIP_vit_large.yaml">PP-ShiTuV2_rec_CLIP_vit_large.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_CLIP_vit_large_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_large_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 AliProducts recall@1。</b>

## [文档方向分类模块](../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td>99.06</td>
<td>3.84845</td>
<td>9.23735</td>
<td>7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml">PP-LCNet_x1_0_doc_ori.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 PaddleX 内部自建数据集 Top-1 Acc 。</b>

## [人脸特征模块](../module_usage/tutorials/cv_modules/face_feature.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>输出特征维度</th>
<th>Acc (%)<br/>AgeDB-30/CFP-FP/LFW</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时</th>
<th>模型存储大小 (M)</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>MobileFaceNet</td>
<td>128</td>
<td>96.28/96.71/99.58</td>
<td>5.7</td>
<td>101.6</td>
<td>4.1</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/face_recognition/MobileFaceNet.yaml">MobileFaceNet.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileFaceNet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileFaceNet_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>ResNet50_face</td>
<td>512</td>
<td>98.12/98.56/99.77</td>
<td>8.7</td>
<td>200.7</td>
<td>87.2</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/face_recognition/ResNet50_face.yaml">ResNet50_face.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_face_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_face_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标是分别在AgeDB-30、CFP-FP和LFW数据集上测得的Accuracy。</b>

## [主体检测模块](../module_usage/tutorials/cv_modules/mainbody_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-ShiTuV2_det</td>
<td>41.5</td>
<td>33.7</td>
<td>537.0</td>
<td>27.54</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/mainbody_detection/PP-ShiTuV2_det.yaml">PP-ShiTuV2_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_det_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 [PaddleClas主体检测数据集](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/training/PP-ShiTu/mainbody_detection.md) mAP(0.5:0.95)。</b>

## [目标检测模块](../module_usage/tutorials/cv_modules/object_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>Cascade-FasterRCNN-ResNet50-FPN</td>
<td>41.1</td>
<td>-</td>
<td>-</td>
<td>245.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/Cascade-FasterRCNN-ResNet50-FPN.yaml">Cascade-FasterRCNN-ResNet50-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-FasterRCNN-ResNet50-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-FasterRCNN-ResNet50-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
<td>45.0</td>
<td>-</td>
<td>-</td>
<td>246.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml">Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>CenterNet-DLA-34</td>
<td>37.6</td>
<td>-</td>
<td>-</td>
<td>75.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/CenterNet-DLA-34.yaml">CenterNet-DLA-34.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CenterNet-DLA-34_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CenterNet-DLA-34_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>CenterNet-ResNet50</td>
<td>38.9</td>
<td>-</td>
<td>-</td>
<td>319.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/CenterNet-ResNet50.yaml">CenterNet-ResNet50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CenterNet-ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CenterNet-ResNet50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>DETR-R50</td>
<td>42.3</td>
<td>59.2132</td>
<td>5334.52</td>
<td>159.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/DETR-R50.yaml">DETR-R50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DETR-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DETR-R50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet34-FPN</td>
<td>37.8</td>
<td>-</td>
<td>-</td>
<td>137.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet34-FPN.yaml">FasterRCNN-ResNet34-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet34-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet34-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet50-FPN</td>
<td>38.4</td>
<td>-</td>
<td>-</td>
<td>148.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet50-FPN.yaml">FasterRCNN-ResNet50-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet50-vd-FPN</td>
<td>39.5</td>
<td>-</td>
<td>-</td>
<td>148.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet50-vd-FPN.yaml">FasterRCNN-ResNet50-vd-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-vd-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-vd-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
<td>41.4</td>
<td>-</td>
<td>-</td>
<td>148.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml">FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet50</td>
<td>36.7</td>
<td>-</td>
<td>-</td>
<td>120.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet50.yaml">FasterRCNN-ResNet50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet101-FPN</td>
<td>41.4</td>
<td>-</td>
<td>-</td>
<td>216.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet101-FPN.yaml">FasterRCNN-ResNet101-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet101-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet101-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNet101</td>
<td>39.0</td>
<td>-</td>
<td>-</td>
<td>188.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet101.yaml">FasterRCNN-ResNet101.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet101_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet101_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-ResNeXt101-vd-FPN</td>
<td>43.4</td>
<td>-</td>
<td>-</td>
<td>360.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNeXt101-vd-FPN.yaml">FasterRCNN-ResNeXt101-vd-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNeXt101-vd-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNeXt101-vd-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FasterRCNN-Swin-Tiny-FPN</td>
<td>42.6</td>
<td>-</td>
<td>-</td>
<td>159.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-Swin-Tiny-FPN.yaml">FasterRCNN-Swin-Tiny-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-Swin-Tiny-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-Swin-Tiny-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>FCOS-ResNet50</td>
<td>39.6</td>
<td>103.367</td>
<td>3424.91</td>
<td>124.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FCOS-ResNet50.yaml">FCOS-ResNet50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FCOS-ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FCOS-ResNet50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PicoDet-L</td>
<td>42.6</td>
<td>16.6715</td>
<td>169.904</td>
<td>20.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PicoDet-L.yaml">PicoDet-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PicoDet-M</td>
<td>37.5</td>
<td>16.2311</td>
<td>71.7257</td>
<td>16.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PicoDet-M.yaml">PicoDet-M.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-M_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PicoDet-S</td>
<td>29.1</td>
<td>14.097</td>
<td>37.6563</td>
<td>4.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PicoDet-S.yaml">PicoDet-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PicoDet-XS</td>
<td>26.2</td>
<td>13.8102</td>
<td>48.3139</td>
<td>5.7M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PicoDet-XS.yaml">PicoDet-XS.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-XS_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-XS_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_plus-L</td>
<td>52.9</td>
<td>33.5644</td>
<td>814.825</td>
<td>185.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PP-YOLOE_plus-L.yaml">PP-YOLOE_plus-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_plus-M</td>
<td>49.8</td>
<td>19.843</td>
<td>449.261</td>
<td>83.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PP-YOLOE_plus-M.yaml">PP-YOLOE_plus-M.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-M_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_plus-S</td>
<td>43.7</td>
<td>16.8884</td>
<td>223.059</td>
<td>28.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PP-YOLOE_plus-S.yaml">PP-YOLOE_plus-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_plus-X</td>
<td>54.7</td>
<td>57.8995</td>
<td>1439.93</td>
<td>349.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PP-YOLOE_plus-X.yaml">PP-YOLOE_plus-X.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-X_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-X_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RT-DETR-H</td>
<td>56.3</td>
<td>114.814</td>
<td>3933.39</td>
<td>435.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/RT-DETR-H.yaml">RT-DETR-H.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RT-DETR-L</td>
<td>53.0</td>
<td>34.5252</td>
<td>1454.27</td>
<td>113.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/RT-DETR-L.yaml">RT-DETR-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RT-DETR-R18</td>
<td>46.5</td>
<td>19.89</td>
<td>784.824</td>
<td>70.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/RT-DETR-R18.yaml">RT-DETR-R18.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-R18_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-R18_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RT-DETR-R50</td>
<td>53.1</td>
<td>41.9327</td>
<td>1625.95</td>
<td>149.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/RT-DETR-R50.yaml">RT-DETR-R50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-R50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RT-DETR-X</td>
<td>54.8</td>
<td>61.8042</td>
<td>2246.64</td>
<td>232.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/RT-DETR-X.yaml">RT-DETR-X.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-X_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-X_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOv3-DarkNet53</td>
<td>39.1</td>
<td>40.1055</td>
<td>883.041</td>
<td>219.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOv3-DarkNet53.yaml">YOLOv3-DarkNet53.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-DarkNet53_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-DarkNet53_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOv3-MobileNetV3</td>
<td>31.4</td>
<td>18.6692</td>
<td>267.214</td>
<td>83.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOv3-MobileNetV3.yaml">YOLOv3-MobileNetV3.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-MobileNetV3_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-MobileNetV3_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOv3-ResNet50_vd_DCN</td>
<td>40.6</td>
<td>31.6276</td>
<td>856.047</td>
<td>163.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOv3-ResNet50_vd_DCN.yaml">YOLOv3-ResNet50_vd_DCN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-ResNet50_vd_DCN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-ResNet50_vd_DCN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOX-L</td>
<td>50.1</td>
<td>185.691</td>
<td>1250.58</td>
<td>192.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOX-L.yaml">YOLOX-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOX-M</td>
<td>46.9</td>
<td>123.324</td>
<td>688.071</td>
<td>90.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOX-M.yaml">YOLOX-M.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-M_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOX-N</td>
<td>26.1</td>
<td>79.1665</td>
<td>155.59</td>
<td>3.4M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOX-N.yaml">YOLOX-N.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-N_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-N_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOX-S</td>
<td>40.4</td>
<td>184.828</td>
<td>474.446</td>
<td>32.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOX-S.yaml">YOLOX-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-S_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOX-T</td>
<td>32.9</td>
<td>102.748</td>
<td>212.52</td>
<td>18.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOX-T.yaml">YOLOX-T.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-T_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-T_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>YOLOX-X</td>
<td>51.8</td>
<td>227.361</td>
<td>2067.84</td>
<td>351.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOX-X.yaml">YOLOX-X.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-X_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-X_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[COCO2017](https://cocodataset.org/#home)<b> 验证集 mAP(0.5:0.95)。</b>

## [小目标检测模块](../module_usage/tutorials/cv_modules/small_object_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-YOLOE_plus_SOD-S</td>
<td>25.1</td>
<td>65.4608</td>
<td>324.37</td>
<td>77.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/small_object_detection/PP-YOLOE_plus_SOD-S.yaml">PP-YOLOE_plus_SOD-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-S_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_plus_SOD-L</td>
<td>31.9</td>
<td>57.1448</td>
<td>1006.98</td>
<td>325.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/small_object_detection/PP-YOLOE_plus_SOD-L.yaml">PP-YOLOE_plus_SOD-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_plus_SOD-largesize-L</td>
<td>42.7</td>
<td>458.521</td>
<td>11172.7</td>
<td>340.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/small_object_detection/PP-YOLOE_plus_SOD-largesize-L.yaml">PP-YOLOE_plus_SOD-largesize-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-largesize-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-largesize-L_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset)<b> 验证集 mAP(0.5:0.95)。</b>

## [开放词汇目标检测](../module_usage/tutorials/cv_modules/open_vocabulary_detection.md)

<table>
<tr>
<th>模型</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>GroundingDINO-T</td>
<td>49.4</td>
<td>64.4</td>
<td>253.72</td>
<td>1807.4</td>
<td>658.3</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/GroundingDINO-T_infer.tar">推理模型</a></td>
</tr>
</table>

<b>注：以上精度指标为 COCO val2017 验证集 mAP(0.5:0.95)。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32</b>。

## [开放词汇分割](../module_usage/tutorials/cv_modules/open_vocabulary_segmentation.md)

<table>
<tr>
<th>模型</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>SAM-H_box</td>
<td>144.9</td>
<td>33920.7</td>
<td>2433.7</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SAM-H_box_infer.tar">推理模型</a></td>
</tr>
<tr>
<td>SAM-H_point</td>
<td>144.9</td>
<td>33920.7</td>
<td>2433.7</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SAM-H_point_infer.tar">推理模型</a></td>
</tr>
</table>

<b>注：所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32</b>。

## [旋转目标检测](../module_usage/tutorials/cv_modules/rotated_object_detection.md)

<table>
<tr>
<th>模型</th>
<th>mAP(%)</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小 (M)</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>PP-YOLOE-R-L</td>
<td>78.14</td>
<td>20.7039</td>
<td>157.942</td>
<td>211.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yamll">PP-YOLOE-R.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-R-L_infer.tar">推理模型</a>/<a href="https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams">训练模型</a></td>
</tr>
</table>

<p><b>注：以上精度指标为<a href="https://captain-whu.github.io/DOTA/">DOTA</a>验证集 mAP(0.5:0.95)。所有模型 GPU 推理耗时基于 NVIDIA TRX2080 Ti 机器，精度类型为 F16， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>

## [行人检测模块](../module_usage/tutorials/cv_modules/human_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-YOLOE-L_human</td>
<td>48.0</td>
<td>32.7754</td>
<td>777.691</td>
<td>196.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/human_detection/PP-YOLOE-L_human.yaml">PP-YOLOE-L_human.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-L_human_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-L_human_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE-S_human</td>
<td>42.5</td>
<td>15.0118</td>
<td>179.317</td>
<td>28.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/human_detection/PP-YOLOE-S_human.yaml">PP-YOLOE-S_human.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-S_human_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-S_human_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[CrowdHuman](https://bj.bcebos.com/v1/paddledet/data/crowdhuman.zip)<b> 验证集 mAP(0.5:0.95)。</b>

## [车辆检测模块](../module_usage/tutorials/cv_modules/vehicle_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mAP（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-YOLOE-L_vehicle</td>
<td>63.9</td>
<td>32.5619</td>
<td>775.633</td>
<td>196.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/vehicle_detection/PP-YOLOE-L_vehicle.yaml">PP-YOLOE-L_vehicle.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-L_vehicle_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-L_vehicle_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE-S_vehicle</td>
<td>61.3</td>
<td>15.3787</td>
<td>178.441</td>
<td>28.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/vehicle_detection/PP-YOLOE-S_vehicle.yaml">PP-YOLOE-S_vehicle.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-S_vehicle_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-S_vehicle_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[PPVehicle](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/modules/ppvehicle)<b> 验证集 mAP(0.5:0.95)。</b>

## [人脸检测模块](../module_usage/tutorials/cv_modules/face_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th style="text-align: center;">AP (%)<br/>Easy/Medium/Hard</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>BlazeFace</td>
<td style="text-align: center;">77.7/73.4/49.5</td>
<td>49.9</td>
<td>68.2</td>
<td>0.447 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/face_detection/BlazeFace.yaml">BlazeFace.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BlazeFace_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>BlazeFace-FPN-SSH</td>
<td style="text-align: center;">83.2/80.5/60.5</td>
<td>52.4</td>
<td>73.2</td>
<td>0.606 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/face_detection/BlazeFace-FPN-SSH.yaml">BlazeFace-FPN-SSH.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BlazeFace-FPN-SSH_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace-FPN-SSH_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PicoDet_LCNet_x2_5_face</td>
<td style="text-align: center;">93.7/90.7/68.1</td>
<td>33.7</td>
<td>185.1</td>
<td>28.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml">PicoDet_LCNet_x2_5_face.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_LCNet_x2_5_face_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_LCNet_x2_5_face_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_plus-S_face</td>
<td style="text-align: center;">93.9/91.8/79.8</td>
<td>25.8</td>
<td>159.9</td>
<td>26.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/face_detection/PP-YOLOE_plus-S_face.yaml">PP-YOLOE_plus-S_face</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_face_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_face_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>

**注：以上精度指标是在WIDER-FACE验证集上，以640*640作为输入尺寸评估得到的。**

## [异常检测模块](../module_usage/tutorials/cv_modules/anomaly_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mIoU</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>STFPM</td>
<td>0.9901</td>
<td>-</td>
<td>-</td>
<td>22.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/anomaly_detection/STFPM.yaml">STFPM.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/STFPM_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/STFPM_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)<b> 验证集 平均异常分数。</b>

## [人体关键点检测模块](../module_usage/tutorials//cv_modules/human_keypoint_detection.md)

<table>
  <tr>
    <th >模型</th>
    <th >方案</th>
    <th >输入尺寸</th>
    <th >AP(0.5:0.95)</th>
    <th >GPU推理耗时（ms）</th>
    <th >CPU推理耗时 (ms)</th>
    <th >模型存储大小（M）</th>
    <th >yaml文件</th>
    <th >模型下载链接</th>
  </tr>
  <tr>
    <td>PP-TinyPose_128x96</td>
    <td>Top-Down</td>
    <td>128*96</td>
    <td>58.4</td>
    <td></td>
    <td></td>
    <td>4.9</td>
    <td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/keypoint_detection/PP-TinyPose_128x96.yaml">PP-TinyPose_128x96.yaml</a></td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TinyPose_128x96_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TinyPose_128x96_pretrained.pdparams">训练模型</a></td>
  </tr>
  <tr>
    <td>PP-TinyPose_256x192</td>
    <td>Top-Down</td>
    <td>256*192</td>
    <td>68.3</td>
    <td></td>
    <td></td>
    <td>4.9</td>
    <td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/keypoint_detection/PP-TinyPose_256x192.yaml">PP-TinyPose_256x192.yaml</a></td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TinyPose_256x192_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TinyPose_256x192_pretrained.pdparams">训练模型</a></td>
  </tr>
</table>

**注：以上精度指标为COCO数据集 AP(0.5:0.95)，所依赖的检测框为ground truth标注得到。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

## [3D多模态融合检测模块](../module_usage/tutorials//cv_modules/3d_bev_detection.md)

<table>
<tr>
<th>模型</th>
<th>mAP(%)</th>
<th>NDS</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>BEVFusion</td>
<td>53.9</td>
<td>60.9</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/3d_bev_detection/BEVFusion.yaml">BEVFusion.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BEVFusion_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BEVFusion_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
</table>

<p><b>注：以上精度指标为<a href="https://www.nuscenes.org/nuscenes">nuscenes</a>验证集 mAP(0.5:0.95), NDS 60.9, 精度类型为 FP32。</b></p>

## [语义分割模块](../module_usage/tutorials/cv_modules/semantic_segmentation.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mloU（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>Deeplabv3_Plus-R50</td>
<td>80.36</td>
<td>61.0531</td>
<td>1513.58</td>
<td>94.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/Deeplabv3_Plus-R50.yaml">Deeplabv3_Plus-R50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Deeplabv3_Plus-R101</td>
<td>81.10</td>
<td>100.026</td>
<td>2460.71</td>
<td>162.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/Deeplabv3_Plus-R101.yaml">Deeplabv3_Plus-R101.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R101_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R101_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Deeplabv3-R50</td>
<td>79.90</td>
<td>82.2631</td>
<td>1735.83</td>
<td>138.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/Deeplabv3-R50.yaml">Deeplabv3-R50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3-R50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Deeplabv3-R101</td>
<td>80.85</td>
<td>121.492</td>
<td>2685.51</td>
<td>205.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/Deeplabv3-R101.yaml">Deeplabv3-R101.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R101_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3-R101_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>OCRNet_HRNet-W18</td>
<td>80.67</td>
<td>48.2335</td>
<td>906.385</td>
<td>43.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/OCRNet_HRNet-W18.yaml">OCRNet_HRNet-W18.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W18_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W18_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>OCRNet_HRNet-W48</td>
<td>82.15</td>
<td>78.9976</td>
<td>2226.95</td>
<td>249.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/OCRNet_HRNet-W48.yaml">OCRNet_HRNet-W48.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W48_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W48_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LiteSeg-T</td>
<td>73.10</td>
<td>7.6827</td>
<td>138.683</td>
<td>28.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml">PP-LiteSeg-T.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-T_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-T_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-LiteSeg-B</td>
<td>75.25</td>
<td>10.9935</td>
<td>194.727</td>
<td>47.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-B.yaml">PP-LiteSeg-B.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-B_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-B_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SegFormer-B0 (slice)</td>
<td>76.73</td>
<td>11.1946</td>
<td>268.929</td>
<td>13.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SegFormer-B0.yaml">SegFormer-B0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B0 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B0 (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SegFormer-B1 (slice)</td>
<td>78.35</td>
<td>17.9998</td>
<td>403.393</td>
<td>48.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SegFormer-B1.yaml">SegFormer-B1.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B1 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B1 (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SegFormer-B2 (slice)</td>
<td>81.60</td>
<td>48.0371</td>
<td>1248.52</td>
<td>96.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SegFormer-B2.yaml">SegFormer-B2.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B2 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B2 (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SegFormer-B3 (slice)</td>
<td>82.47</td>
<td>64.341</td>
<td>1666.35</td>
<td>167.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SegFormer-B3.yaml">SegFormer-B3.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B3 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B3 (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SegFormer-B4 (slice)</td>
<td>82.38</td>
<td>82.4336</td>
<td>1995.42</td>
<td>226.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SegFormer-B4.yaml">SegFormer-B4.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B4 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B4 (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SegFormer-B5 (slice)</td>
<td>82.58</td>
<td>97.3717</td>
<td>2420.19</td>
<td>229.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SegFormer-B5.yaml">SegFormer-B5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B5 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B5 (slice)_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[Cityscapes](https://www.cityscapes-dataset.com/)<b> 数据集 mloU。</b>
<table>
<thead>
<tr>
<th>模型名称</th>
<th>mloU（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>SeaFormer_base(slice)</td>
<td>40.92</td>
<td>24.4073</td>
<td>397.574</td>
<td>30.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SeaFormer_base.yaml">SeaFormer_base.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_base(slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_base(slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SeaFormer_large (slice)</td>
<td>43.66</td>
<td>27.8123</td>
<td>550.464</td>
<td>49.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SeaFormer_large.yaml">SeaFormer_large.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_large (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_large (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SeaFormer_small (slice)</td>
<td>38.73</td>
<td>19.2295</td>
<td>358.343</td>
<td>14.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SeaFormer_small.yaml">SeaFormer_small.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_small (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_small (slice)_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SeaFormer_tiny (slice)</td>
<td>34.58</td>
<td>13.9496</td>
<td>330.132</td>
<td>6.1M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SeaFormer_tiny.yaml">SeaFormer_tiny.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_tiny (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_tiny (slice)_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)<b> 数据集, slice 表示对输入图像进行了切图操作。</b>

## [实例分割模块](../module_usage/tutorials/cv_modules/instance_segmentation.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>Mask AP</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>Mask-RT-DETR-H</td>
<td>50.6</td>
<td>132.693</td>
<td>4896.17</td>
<td>449.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-H.yaml">Mask-RT-DETR-H.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-H_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-H_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Mask-RT-DETR-L</td>
<td>45.7</td>
<td>46.5059</td>
<td>2575.92</td>
<td>113.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-L.yaml">Mask-RT-DETR-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-L_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Mask-RT-DETR-M</td>
<td>42.7</td>
<td>36.8329</td>
<td>-</td>
<td>66.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-M.yaml">Mask-RT-DETR-M.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-M_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Mask-RT-DETR-S</td>
<td>41.0</td>
<td>33.5007</td>
<td>-</td>
<td>51.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-S.yaml">Mask-RT-DETR-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-S_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Mask-RT-DETR-X</td>
<td>47.5</td>
<td>75.755</td>
<td>3358.04</td>
<td>237.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-X.yaml">Mask-RT-DETR-X.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-X_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-X_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Cascade-MaskRCNN-ResNet50-FPN</td>
<td>36.3</td>
<td>-</td>
<td>-</td>
<td>254.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Cascade-MaskRCNN-ResNet50-FPN.yaml">Cascade-MaskRCNN-ResNet50-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-MaskRCNN-ResNet50-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-MaskRCNN-ResNet50-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN</td>
<td>39.1</td>
<td>-</td>
<td>-</td>
<td>254.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN.yaml">Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MaskRCNN-ResNet50-FPN</td>
<td>35.6</td>
<td>-</td>
<td>-</td>
<td>157.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/MaskRCNN-ResNet50-FPN.yaml">MaskRCNN-ResNet50-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MaskRCNN-ResNet50-vd-FPN</td>
<td>36.4</td>
<td>-</td>
<td>-</td>
<td>157.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/MaskRCNN-ResNet50-vd-FPN.yaml">MaskRCNN-ResNet50-vd-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50-vd-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50-vd-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MaskRCNN-ResNet50</td>
<td>32.8</td>
<td>-</td>
<td>-</td>
<td>127.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/MaskRCNN-ResNet50.yaml">MaskRCNN-ResNet50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MaskRCNN-ResNet101-FPN</td>
<td>36.6</td>
<td>-</td>
<td>-</td>
<td>225.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/MaskRCNN-ResNet101-FPN.yaml">MaskRCNN-ResNet101-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet101-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet101-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MaskRCNN-ResNet101-vd-FPN</td>
<td>38.1</td>
<td>-</td>
<td>-</td>
<td>225.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/MaskRCNN-ResNet101-vd-FPN.yaml">MaskRCNN-ResNet101-vd-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet101-vd-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet101-vd-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>MaskRCNN-ResNeXt101-vd-FPN</td>
<td>39.5</td>
<td>-</td>
<td>-</td>
<td>370.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/MaskRCNN-ResNeXt101-vd-FPN.yaml">MaskRCNN-ResNeXt101-vd-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNeXt101-vd-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNeXt101-vd-FPN_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-YOLOE_seg-S</td>
<td>32.5</td>
<td>-</td>
<td>-</td>
<td>31.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/PP-YOLOE_seg-S.yaml">PP-YOLOE_seg-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_seg-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_seg-S_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>SOLOv2</td>
<td>35.5</td>
<td>-</td>
<td>-</td>
<td>179.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/paddlex/configs/modules/instance_segmentation/SOLOv2.yaml">SOLOv2.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SOLOv2_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SOLOv2_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标为 </b>[COCO2017](https://cocodataset.org/#home)<b> 验证集 Mask AP(0.5:0.95)。</b>

## [文本检测模块](../module_usage/tutorials/ocr_modules/text_detection.md)

<table>
<thead>
<tr>
<th>模型</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M)</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_det</td>
<td>82.56</td>
<td>83.3501</td>
<td>2434.01</td>
<td>109</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv4_server_det.yaml">PP-OCRv4_server_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td>77.35</td>
<td>10.6923</td>
<td>120.177</td>
<td>4.7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml">PP-OCRv4_mobile_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv3_mobile_det</td>
<td>78.68</td>
<td></td>
<td></td>
<td>2.1</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv3_mobile_det.yaml">PP-OCRv3_mobile_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv3_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_det_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv3_server_det</td>
<td>80.11</td>
<td></td>
<td></td>
<td>102.1</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv3_server_det.yaml">PP-OCRv3_server_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv3_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_server_det_pretrained.pdparams">训练模型</a></td>
</tr>
</tbody>
</table>

<b>注：以上精度指标的评估集是 PaddleOCR 自建的中英文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 593 张图片。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b>

## [印章文本检测模块](../module_usage/tutorials/ocr_modules/seal_text_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_seal_det</td>
<td>96.47</td>
<td>10.5878</td>
<td>131.813</td>
<td>4.7M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/seal_text_detection/PP-OCRv4_mobile_seal_det.yaml">PP-OCRv4_mobile_seal_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-OCRv4_server_seal_det</td>
<td>98.21</td>
<td>84.341</td>
<td>2425.06</td>
<td>108.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/seal_text_detection/PP-OCRv4_server_seal_det.yaml">PP-OCRv4_server_seal_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 PaddleX 自建的印章数据集，包含500印章图像。</b>

## [文本识别模块](../module_usage/tutorials/ocr_modules/text_recognition.md)

* <b>中文识别模型</b>

<table>
<tr>
<th>模型</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td>
<td>81.53</td>
<td></td>
<td></td>
<td>74.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv4_server_rec_doc.yaml">PP-OCRv4_server_rec_doc.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv4_server_rec_doc_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td>78.74</td>
<td>7.95018</td>
<td>46.7868</td>
<td>10.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml">PP-OCRv4_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td>
<td>80.61 </td>
<td>7.19439</td>
<td>140.179</td>
<td>71.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv4_server_rec.yaml">PP-OCRv4_server_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td>
<td>72.96</td>
<td></td>
<td></td>
<td>9.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv3_mobile_rec.yaml">PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
</table>

<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 8367 张图片。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>

<table>
<tr>
<th>模型</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时</th>
<th>模型存储大小（M）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td>
<td>68.81</td>
<td>8.36801</td>
<td>165.706</td>
<td>73.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/ch_SVTRv2_rec.yaml">ch_SVTRv2_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_SVTRv2_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">训练模型</a></td>
</tr>
</table>

<p><b>注：以上精度指标的评估集是 <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>A榜。 所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<table>
<tr>
<th>模型</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时</th>
<th>模型存储大小（M）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td>
<td>65.07</td>
<td>10.5047</td>
<td>51.5647</td>
<td>22.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/ch_RepSVTR_rec.yaml">ch_RepSVTR_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_RepSVTR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">训练模型</a></td>
</tr>
</table>

<p><b>注：以上精度指标的评估集是 <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>B榜。 所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>

* <b>英文识别模型</b>

<table>
<tr>
<th>模型</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时</th>
<th>模型存储大小（M）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td>
<td> 70.39</td>
<td></td>
<td></td>
<td>6.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/en_PP-OCRv4_mobile_rec.yaml">en_PP-OCRv4_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
en_PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td>
<td>70.69</td>
<td></td>
<td></td>
<td>7.8 M </td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/en_PP-OCRv3_mobile_rec.yaml">en_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
en_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
</table>

* <b>多语言识别模型</b>

<table>
<tr>
<th>模型</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时</th>
<th>模型存储大小（M）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td>
<td>60.21</td>
<td></td>
<td></td>
<td>8.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/korean_PP-OCRv3_mobile_rec.yaml">korean_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
korean_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td>
<td>45.69</td>
<td></td>
<td></td>
<td>8.8 M </td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/japan_PP-OCRv3_mobile_rec.yaml">japan_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
japan_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td>
<td>82.06</td>
<td></td>
<td></td>
<td>9.7 M </td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/chinese_cht_PP-OCRv3_mobile_rec.yaml">chinese_cht_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td>
<td>95.88</td>
<td></td>
<td></td>
<td>7.8 M </td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/te_PP-OCRv3_mobile_rec.yaml">te_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
te_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td>
<td>96.96</td>
<td></td>
<td></td>
<td>8.0 M </td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/ka_PP-OCRv3_mobile_rec.yaml">ka_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
ka_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td>
<td>76.83</td>
<td></td>
<td></td>
<td>8.0 M </td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/ta_PP-OCRv3_mobile_rec.yaml">ta_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
ta_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td>
<td>76.93</td>
<td></td>
<td></td>
<td>7.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/latin_PP-OCRv3_mobile_rec.yaml">latin_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
latin_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td>
<td>73.55</td>
<td></td>
<td></td>
<td>7.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/arabic_PP-OCRv3_mobile_rec.yaml">arabic_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
arabic_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td>
<td>94.28</td>
<td></td>
<td></td>
<td>7.9 M  </td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/cyrillic_PP-OCRv3_mobile_rec.yaml">cyrillic_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td>
<td>96.44</td>
<td></td>
<td></td>
<td>7.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/devanagari_PP-OCRv3_mobile_rec.yaml">devanagari_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
</table>
<p><b>注：以上精度指标的评估集是 PaddleX 自建的多语种数据集。 所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>

## [公式识别模块](../module_usage/tutorials/ocr_modules/formula_recognition.md)

<table>
<tr>
<th>模型</th>
<th>Avg-BLEU</th>
<th>GPU推理耗时 (ms)</th>
<th>模型存储大小 (M)</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<td>UniMERNet</td>
<td>0.8613</td>
<td>2266.96</td>
<td>1.4 G</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/UniMERNet.yaml">UniMERNet.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UniMERNet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">训练模型</a></td>
<tr>
<td>PP-FormulaNet-S</td>
<td>0.8712</td>
<td>202.25</td>
<td>167.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/PP-FormulaNet-S.yaml">PP-FormulaNet-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-FormulaNet-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">训练模型</a></td>
</tr>
<td>PP-FormulaNet-L</td>
<td>0.9213</td>
<td>1976.52</td>
<td>535.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/PP-FormulaNet-L.yaml">PP-FormulaNet-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-FormulaNet-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">训练模型</a></td>
<tr>
<td>LaTeX_OCR_rec</td>
<td>0.7163</td>
<td>-</td>
<td>89.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/LaTeX_OCR_rec.yaml">LaTeX_OCR_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/LaTeX_OCR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">训练模型</a></td>
</tr>
</table>

<b>注：以上精度指标测量自 PaddleX 内部自建公式识别测试集。LaTeX_OCR_rec在LaTeX-OCR公式识别测试集的BLEU score为 0.8821。所有模型 GPU 推理耗时基于 Tesla V100 GPUs 机器，精度类型为 FP32。</b>

## [表格结构识别模块](../module_usage/tutorials/ocr_modules/table_structure_recognition.md)

<table>
<tr>
<th>模型</th>
<th>精度（%）</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小 (M)</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>SLANet</td>
<td>59.52</td>
<td>522.536</td>
<td>1845.37</td>
<td>6.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_structure_recognition/SLANet.yaml">SLANet.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>SLANet_plus</td>
<td>63.69</td>
<td>522.536</td>
<td>1845.37</td>
<td>6.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_structure_recognition/SLANet_plus.yaml">SLANet_plus.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_plus_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td rowspan="2">69.65</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_structure_recognition/SLANeXt_wired.yaml">SLANeXt_wired.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANeXt_wired_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>SLANeXt_wireless</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_structure_recognition/SLANeXt_wireless.yaml">SLANeXt_wireless.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANeXt_wireless_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wireless_pretrained.pdparams">训练模型</a></td>
</tr>
</table>


<b>注：以上精度指标测量自 PaddleX 内部自建高难度中文表格识别数据集。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b>


## [表格单元格检测模块](../module_usage/tutorials/ocr_modules/table_cells_detection.md)

<table>
<tr>
<th>模型</th>
<th>mAP(%)</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小 (M)</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_cells_detection/RT-DETR-L_wired_table_cell_det.yaml">RT-DETR-L_wired_table_cell_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wired_table_cell_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_cells_detection/RT-DETR-L_wireless_table_cell_det.yaml">RT-DETR-L_wireless_table_cell_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wireless_table_cell_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">训练模型</a></td>
</tr>
</table>

<p><b>注：以上精度指标测量自 PaddleX 内部自建表格单元格检测数据集。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>

## [表格分类模块](../module_usage/tutorials/ocr_modules/table_classification.md)

<table>
<tr>
<th>模型</th>
<th>Top1 Acc(%)</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小 (M)</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td>
<td>--</td>
<td>--</td>
<td>--</td>
<td>--</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_classification/PP-LCNet_x1_0_table_cls.yaml">PP-LCNet_x1_0_table_cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">训练模型</a></td>
</tr>
</table>

<p><b>注：以上精度指标测量自 PaddleX 内部自建表格分类数据集。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>

## [文本图像矫正模块](../module_usage/tutorials/ocr_modules/text_image_unwarping.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>MS-SSIM （%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td>
<td>54.40</td>
<td>-</td>
<td>-</td>
<td>30.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_unwarping/UVDoc.yaml">UVDoc.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UVDoc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标测量自 </b><b>PaddleX自建的图像矫正数据集</b><b>。</b>

## [版面区域检测模块](../module_usage/tutorials/ocr_modules/layout_detection.md)

* <b>表格版面检测模型</b>

<table>
<thead>
<tr>
<th>模型</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x_table</td>
<td>97.5</td>
<td>12.623</td>
<td>90.8934</td>
<td>7.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet_layout_1x_table.yaml">PicoDet_layout_1x_table.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_table_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">训练模型</a></td>
</tr>
</table>

<b>注：以上精度指标的评估集是 PaddleOCR 自建的版面表格区域检测数据集，包含中英文 7835 张带有表格的论文文档类型图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b>

* <b>3类版面检测模型，包含表格、图像、印章</b>

<table>
<thead>
<tr>
<th>模型</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_3cls</td>
<td>88.2</td>
<td>13.5</td>
<td>45.8</td>
<td>4.8</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet-S_layout_3cls.yaml">PicoDet-S_layout_3cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td>
<td>89.0</td>
<td>15.7</td>
<td>159.8</td>
<td>22.6</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet-L_layout_3cls.yaml">PicoDet-L_layout_3cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td>
<td>95.8</td>
<td>114.6</td>
<td>3832.6</td>
<td>470.1</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/RT-DETR-H_layout_3cls.yaml">RT-DETR-H_layout_3cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">训练模型</a></td>
</tr>
</table>

<b>注：以上精度指标的评估集是 PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志和研报等常见的 1154 张文档类型图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b>

* <b>5类英文文档区域检测模型，包含文字、标题、表格、图片以及列表</b>

<table>
<thead>
<tr>
<th>模型</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td>
<td>97.8</td>
<td>13.0</td>
<td>91.3</td>
<td>7.4</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet_layout_1x.yaml">PicoDet_layout_1x.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">训练模型</a></td>
</tr>
</table>

<b>注：以上精度指标的评估集是 [PubLayNet](https://developer.ibm.com/exchanges/data/all/publaynet/) 的评估数据集，包含英文文档的 11245 张文图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b>

* <b>17类区域检测模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</b>

<table>
<thead>
<tr>
<th>模型</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td>
<td>87.4</td>
<td>13.6</td>
<td>46.2</td>
<td>4.8</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet-S_layout_17cls.yaml">PicoDet-S_layout_17cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">训练模型</a></td>
</tr>

<tr>
<td>PicoDet-L_layout_17cls</td>
<td>89.0</td>
<td>17.2</td>
<td>160.2</td>
<td>22.6</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet-L_layout_17cls.yaml">PicoDet-L_layout_17cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">训练模型</a></td>
</tr>

<tr>
<td>RT-DETR-H_layout_17cls</td>
<td>98.3</td>
<td>115.1</td>
<td>3827.2</td>
<td>470.2</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/RT-DETR-H_layout_17cls.yaml">RT-DETR-H_layout_17cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">训练模型</a></td>
</tr>
</tbody>
</table>


<b>注：以上精度指标的评估集是 PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志和研报等常见的 892 张文档类型图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b>

## [文档图像方向分类模块](../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md)

<table>
<thead>
<tr>
<th>模型</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M)</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td>99.06</td>
<td>3.84845</td>
<td>9.23735</td>
<td>7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml">PP-LCNet_x1_0_doc_ori.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td>
</tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b>


## [时序预测模块](../module_usage/tutorials/time_series_modules/time_series_forecasting.md)

<table>
<thead>
<tr>
<th>模型名称</th>
<th>mse</th>
<th>mae</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>DLinear</td>
<td>0.382</td>
<td>0.394</td>
<td>72 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/DLinear.yaml">DLinear.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DLinear_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DLinear_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>NLinear</td>
<td>0.386</td>
<td>0.392</td>
<td>40 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/NLinear.yaml">NLinear.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/NLinear_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/NLinear_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Nonstationary</td>
<td>0.600</td>
<td>0.515</td>
<td>55.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/Nonstationary.yaml">Nonstationary.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Nonstationary_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Nonstationary_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PatchTST</td>
<td>0.379</td>
<td>0.391</td>
<td>2.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/PatchTST.yaml">PatchTST.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PatchTST_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PatchTST_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>RLinear</td>
<td>0.385</td>
<td>0.392</td>
<td>40 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/RLinear.yaml">RLinear.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RLinear_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RLinear_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>TiDE</td>
<td>0.407</td>
<td>0.414</td>
<td>31.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/TiDE.yaml">TiDE.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TiDE_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TiDE_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>TimesNet</td>
<td>0.416</td>
<td>0.429</td>
<td>4.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/TimesNet.yaml">TimesNet.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标测量自 </b>[ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar)<b> 数据集 </b><b>（在测试集test.csv上的评测结果）</b><b>。</b>

## [时序异常检测模块](../module_usage/tutorials/time_series_modules/time_series_anomaly_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>precison</th>
<th>recall</th>
<th>f1_score</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>AutoEncoder_ad</td>
<td>99.36</td>
<td>84.36</td>
<td>91.25</td>
<td>52 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_anomaly_detection/AutoEncoder_ad.yaml">AutoEncoder_ad.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/AutoEncoder_ad_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/AutoEncoder_ad_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>DLinear_ad</td>
<td>98.98</td>
<td>93.96</td>
<td>96.41</td>
<td>112 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_anomaly_detection/DLinear_ad.yaml">DLinear_ad.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DLinear_ad_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DLinear_ad_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>Nonstationary_ad</td>
<td>98.55</td>
<td>88.95</td>
<td>93.51</td>
<td>1.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_anomaly_detection/Nonstationary_ad.yaml">Nonstationary_ad.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Nonstationary_ad_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Nonstationary_ad_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PatchTST_ad</td>
<td>98.78</td>
<td>90.70</td>
<td>94.57</td>
<td>320 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_anomaly_detection/PatchTST_ad.yaml">PatchTST_ad.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PatchTST_ad_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PatchTST_ad_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标测量自 </b>[PSM](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar)<b> 数据集。</b>

## [时序分类模块](../module_usage/tutorials/time_series_modules/time_series_classification.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>acc(%)</th>
<th>模型存储大小</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>TimesNet_cls</td>
<td>87.5</td>
<td>792 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_classification/TimesNet_cls.yaml">TimesNet_cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_cls_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标测量自 [UWaveGestureLibrary](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TEST.csv)数据集。</b>

&gt;<b>注：以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b>

## [多语种语音识别模块](../module_usage/tutorials/speech_modules/multilingual_speech_recognition.md)

<table>
  <tr>
    <th >模型</th>
    <th >训练数据</th>
    <th >模型大小</th>
    <th >词错率</th>
    <th >yaml文件</th>
    <th >模型下载链接</th>
  </tr>
  <tr>
    <td>whisper_large</td>
    <td >680kh</td>
    <td>5.8G</td>
    <td>2.7 (Librispeech)</td>
    <td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/multilingual_speech_recognition/whisper_large.yaml">whisper_large.yaml</a></td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_large.tar">推理模型</a></td>
  </tr>
  <tr>
    <td>whisper_medium</td>
    <td>680kh</td>
    <td>2.9G</td>
    <td>-</td>
    <td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/multilingual_speech_recognition/whisper_medium.yaml">whisper_medium.yaml</a></td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_medium.tar">推理模型</a></td>
  </tr>
  <tr>
    <td>whisper_small</td>
    <td>680kh</td>
    <td>923M</td>
    <td>-</td>
    <td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/multilingual_speech_recognition/whisper_small.yaml">whisper_small.yaml</a></td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_small.tar">推理模型</a></td>
  </tr>
  <tr>
    <td>whisper_base</td>
    <td>680kh</td>
    <td>277M</td>
    <td>-</td>
    <td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/multilingual_speech_recognition/whisper_base.yaml">whisper_base.yaml</a></td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_base.tar">推理模型</a></td>
  </tr>
  <tr>
    <td>whisper_small</td>
    <td>680kh</td>
    <td>145M</td>
    <td>-</td>
    <td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/multilingual_speech_recognition/whisper_small.yaml">whisper_small.yaml</a></td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_tiny.tar">推理模型</a></td>
  </tr>
</table>

## [视频分类模块](../module_usage/tutorials/video_modules/video_classification.md)

<table>
<tr>
<th>模型</th>
<th>Top1 Acc(%)</th>
<th>模型存储大小 (M)</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>PP-TSM-R50_8frames_uniform</td>
<td>74.36</td>
<td>93.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/video_classification/PP-TSM-R50_8frames_uniform.yaml">PP-TSM-R50_8frames_uniform.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSM-R50_8frames_uniform_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSM-R50_8frames_uniform_pretrained.pdparams">训练模型</a></td>
</tr>

<tr>
<td>PP-TSMv2-LCNetV2_8frames_uniform</td>
<td>71.71</td>
<td>22.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/video_classification/PP-TSMv2-LCNetV2_8frames_uniform.yaml">PP-TSMv2-LCNetV2_8frames_uniform.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSMv2-LCNetV2_8frames_uniform_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSMv2-LCNetV2_8frames_uniform_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-TSMv2-LCNetV2_16frames_uniform</td>
<td>73.11</td>
<td>22.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/video_classification/PP-TSMv2-LCNetV2_16frames_uniform.yaml">PP-TSMv2-LCNetV2_16frames_uniform.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSMv2-LCNetV2_16frames_uniform_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSMv2-LCNetV2_16frames_uniform_pretrained.pdparams">训练模型</a></td>
</tr>

</table>



<p><b>注：以上精度指标为 <a href="https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/dataset/k400.md">K400</a> 验证集 Top1 Acc。</b></p>

## [视频检测模块](../module_usage/tutorials/video_modules/video_detection.md)

<table>
<tr>
<th>模型</th>
<th>Frame-mAP(@ IoU 0.5)</th>
<th>模型存储大小 (M)</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>YOWO</td>
<td>80.94</td>
<td>462.891M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/video_detection/YOWO.yaml">YOWO.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOWO_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOWO_pretrained.pdparams">训练模型</a></td>
</tr>

</table>


<p><b>注：以上精度指标为 <a href="http://www.thumos.info/download.html">UCF101-24</a> test数据集上的测试指标Frame-mAP (@ IoU 0.5)。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>