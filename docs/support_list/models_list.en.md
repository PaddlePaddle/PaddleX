---
comments: true
---

# PaddleX Model List (CPU/GPU)

PaddleX includes multiple production lines, each containing several modules, and each module includes several models. You can choose which models to use based on the benchmark data below. If you prioritize model accuracy, choose models with higher accuracy. If you prioritize model inference speed, choose models with faster inference speed. If you prioritize model storage size, choose models with smaller storage size.

## [Image Classification Module](../module_usage/tutorials/cv_modules/image_classification.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Top1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>CLIP_vit_base_patch16_224</td>
<td>85.36</td>
<td>12.84 / 2.82</td>
<td>60.52 / 60.52</td>
<td>306.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/CLIP_vit_base_patch16_224.yaml">CLIP_vit_base_patch16_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_224_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>CLIP_vit_large_patch14_224</td>
<td>88.1</td>
<td>51.72 / 11.13</td>
<td>238.07 / 238.07</td>
<td>1.04 G</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/CLIP_vit_large_patch14_224.yaml">CLIP_vit_large_patch14_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_large_patch14_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_large_patch14_224_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>ConvNeXt_base_224</td>
<td>83.84</td>
<td>13.18 / 12.14</td>
<td>128.39 / 81.78</td>
<td>313.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ConvNeXt_base_224.yaml">ConvNeXt_base_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_base_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_base_224_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>ConvNeXt_base_384</td>
<td>84.90</td>
<td>32.15 / 30.52</td>
<td>279.36 / 220.35</td>
<td>313.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ConvNeXt_base_384.yaml">ConvNeXt_base_384.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_base_384_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_base_384_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>ConvNeXt_large_224</td>
<td>84.26</td>
<td>26.51 / 7.21</td>
<td>213.32 / 157.22</td>
<td>700.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ConvNeXt_large_224.yaml">ConvNeXt_large_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_large_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_large_224_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>ConvNeXt_large_384</td>
<td>85.27</td>
<td>67.07 / 65.26</td>
<td>494.04 / 438.97</td>
<td>700.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ConvNeXt_large_384.yaml">ConvNeXt_large_384.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_large_384_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_large_384_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>ConvNeXt_small</td>
<td>83.13</td>
<td>9.05 / 8.21</td>
<td>97.94 / 55.29</td>
<td>178.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ConvNeXt_small.yaml">ConvNeXt_small.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_small_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>ConvNeXt_tiny</td>
<td>82.03</td>
<td>5.12 / 2.06</td>
<td>63.96 / 29.77</td>
<td>101.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ConvNeXt_tiny.yaml">ConvNeXt_tiny.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_tiny_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_tiny_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>FasterNet-L</td>
<td>83.5</td>
<td>15.67 / 3.10</td>
<td>52.24 / 52.24</td>
<td>357.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/FasterNet-L.yaml">FasterNet-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-L_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>FasterNet-M</td>
<td>83.0</td>
<td>9.72 / 2.30</td>
<td>35.29 / 35.29</td>
<td>204.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/FasterNet-M.yaml">FasterNet-M.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-M_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>FasterNet-S</td>
<td>81.3</td>
<td>5.46 / 1.27</td>
<td>20.46 / 18.03</td>
<td>119.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/FasterNet-S.yaml">FasterNet-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-S_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>FasterNet-T0</td>
<td>71.9</td>
<td>4.18 / 0.60</td>
<td>6.34 / 3.44</td>
<td>15.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/FasterNet-T0.yaml">FasterNet-T0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T0_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>FasterNet-T1</td>
<td>75.9</td>
<td>4.24 / 0.64</td>
<td>9.57 / 5.20</td>
<td>29.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/FasterNet-T1.yaml">FasterNet-T1.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T1_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T1_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>FasterNet-T2</td>
<td>79.1</td>
<td>3.87 / 0.78</td>
<td>11.14 / 9.98</td>
<td>57.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/FasterNet-T2.yaml">FasterNet-T2.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T2_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T2_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>MobileNetV1_x0_5</td>
<td>63.5</td>
<td>1.39 / 0.28</td>
<td>2.74 / 1.02</td>
<td>4.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV1_x0_5.yaml">MobileNetV1_x0_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_5_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV1_x0_25</td>
<td>51.4</td>
<td>1.32 / 0.30</td>
<td>2.04 / 0.58</td>
<td>1.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV1_x0_25.yaml">MobileNetV1_x0_25.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_25_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV1_x0_75</td>
<td>68.8</td>
<td>1.75 / 0.33</td>
<td>3.41 / 1.57</td>
<td>9.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV1_x0_75.yaml">MobileNetV1_x0_75.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_75_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_75_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV1_x1_0</td>
<td>71.0</td>
<td>1.89 / 0.34</td>
<td>4.01 / 2.17</td>
<td>15.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV1_x1_0.yaml">MobileNetV1_x1_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x1_0_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV2_x0_5</td>
<td>65.0</td>
<td>3.17 / 0.48</td>
<td>4.52 / 1.35</td>
<td>7.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV2_x0_5.yaml">MobileNetV2_x0_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x0_5_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV2_x0_25</td>
<td>53.2</td>
<td>2.80 / 0.46</td>
<td>3.92 / 0.98</td>
<td>5.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV2_x0_25.yaml">MobileNetV2_x0_25.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x0_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x0_25_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV2_x1_0</td>
<td>72.2</td>
<td>3.57 / 0.49</td>
<td>5.63 / 2.51</td>
<td>12.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV2_x1_0.yaml">MobileNetV2_x1_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x1_0_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV2_x1_5</td>
<td>74.1</td>
<td>3.58 / 0.62</td>
<td>8.02 / 4.49</td>
<td>25.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV2_x1_5.yaml">MobileNetV2_x1_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x1_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x1_5_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV2_x2_0</td>
<td>75.2</td>
<td>3.56 / 0.74</td>
<td>10.24 / 6.83</td>
<td>41.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV2_x2_0.yaml">MobileNetV2_x2_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x2_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x2_0_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV3_large_x0_5</td>
<td>69.2</td>
<td>3.79 / 0.62</td>
<td>6.76 / 1.61</td>
<td>9.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_large_x0_5.yaml">MobileNetV3_large_x0_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_5_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV3_large_x0_35</td>
<td>64.3</td>
<td>3.70 / 0.60</td>
<td>5.54 / 1.41</td>
<td>7.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_large_x0_35.yaml">MobileNetV3_large_x0_35.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_35_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_35_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>MobileNetV3_large_x0_75</td>
<td>73.1</td>
<td>4.82 / 0.66</td>
<td>7.45 / 2.00</td>
<td>14.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_large_x0_75.yaml">MobileNetV3_large_x0_75.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_75_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_75_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>MobileNetV3_large_x1_0</td>
<td>75.3</td>
<td>4.86 / 0.68</td>
<td>6.88 / 2.61</td>
<td>19.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_large_x1_0.yaml">MobileNetV3_large_x1_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x1_0_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>MobileNetV3_large_x1_25</td>
<td>76.4</td>
<td>5.08 / 0.71</td>
<td>7.37 / 3.58</td>
<td>26.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_large_x1_25.yaml">MobileNetV3_large_x1_25.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x1_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x1_25_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>MobileNetV3_small_x0_5</td>
<td>59.2</td>
<td>3.41 / 0.57</td>
<td>5.60 / 1.14</td>
<td>6.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_small_x0_5.yaml">MobileNetV3_small_x0_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_5_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>MobileNetV3_small_x0_35</td>
<td>53.0</td>
<td>3.49 / 0.60</td>
<td>4.63 / 1.07</td>
<td>6.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_small_x0_35.yaml">MobileNetV3_small_x0_35.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_35_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_35_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV3_small_x0_75</td>
<td>66.0</td>
<td>3.49 / 0.60</td>
<td>5.19 / 1.28</td>
<td>8.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_small_x0_75.yaml">MobileNetV3_small_x0_75.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_75_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_75_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV3_small_x1_0</td>
<td>68.2</td>
<td>3.76 / 0.53</td>
<td>5.11 / 1.43</td>
<td>10.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_small_x1_0.yaml">MobileNetV3_small_x1_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_0_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV3_small_x1_25</td>
<td>70.7</td>
<td>4.23 / 0.58</td>
<td>6.48 / 1.68</td>
<td>13.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV3_small_x1_25.yaml">MobileNetV3_small_x1_25.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x1_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_25_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV4_conv_large</td>
<td>83.4</td>
<td>8.33 / 2.24</td>
<td>33.56 / 23.70</td>
<td>125.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV4_conv_large.yaml">MobileNetV4_conv_large.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_large_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_large_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV4_conv_medium</td>
<td>79.9</td>
<td>6.81 / 0.92</td>
<td>12.47 / 6.27</td>
<td>37.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV4_conv_medium.yaml">MobileNetV4_conv_medium.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_medium_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_medium_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV4_conv_small</td>
<td>74.6</td>
<td>3.25 / 0.46</td>
<td>4.42 / 1.54</td>
<td>14.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV4_conv_small.yaml">MobileNetV4_conv_small.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_small_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV4_hybrid_large</td>
<td>83.8</td>
<td>12.27 / 4.18</td>
<td>58.64 / 58.64</td>
<td>145.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV4_hybrid_large.yaml">MobileNetV4_hybrid_large.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_hybrid_large_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_hybrid_large_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MobileNetV4_hybrid_medium</td>
<td>80.5</td>
<td>12.08 / 1.34</td>
<td>24.69 / 8.10</td>
<td>42.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/MobileNetV4_hybrid_medium.yaml">MobileNetV4_hybrid_medium.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_hybrid_medium_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_hybrid_medium_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-HGNet_base</td>
<td>85.0</td>
<td>14.10 / 4.19</td>
<td>68.92 / 68.92</td>
<td>249.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNet_base.yaml">PP-HGNet_base.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_base_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_base_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-HGNet_small</td>
<td>81.51</td>
<td>5.12 / 1.73</td>
<td>25.01 / 25.01</td>
<td>86.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNet_small.yaml">PP-HGNet_small.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_small_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-HGNet_tiny</td>
<td>79.83</td>
<td>3.28 / 1.29</td>
<td>16.40 / 15.97</td>
<td>52.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNet_tiny.yaml">PP-HGNet_tiny.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_tiny_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_tiny_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-HGNetV2-B0</td>
<td>77.77</td>
<td>3.83 / 0.57</td>
<td>9.95 / 2.37</td>
<td>21.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B0.yaml">PP-HGNetV2-B0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-HGNetV2-B1</td>
<td>79.18</td>
<td>3.87 / 0.62</td>
<td>8.77 / 3.79</td>
<td>22.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B1.yaml">PP-HGNetV2-B1.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B1_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B1_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-HGNetV2-B2</td>
<td>81.74</td>
<td>5.73 / 0.86</td>
<td>15.11 / 7.05</td>
<td>39.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B2.yaml">PP-HGNetV2-B2.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B2_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B2_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-HGNetV2-B3</td>
<td>82.98</td>
<td>6.26 / 1.01</td>
<td>18.47 / 10.34</td>
<td>57.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B3.yaml">PP-HGNetV2-B3.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B3_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B3_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-HGNetV2-B4</td>
<td>83.57</td>
<td>5.47 / 1.10</td>
<td>14.42 / 9.89</td>
<td>70.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B4.yaml">PP-HGNetV2-B4.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-HGNetV2-B5</td>
<td>84.75</td>
<td>10.24 / 1.96</td>
<td>29.71 / 29.71</td>
<td>140.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B5.yaml">PP-HGNetV2-B5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B5_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-HGNetV2-B6</td>
<td>86.30</td>
<td>12.25 / 3.76</td>
<td>62.29 / 62.29</td>
<td>268.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-HGNetV2-B6.yaml">PP-HGNetV2-B6.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-LCNet_x0_5</td>
<td>63.14</td>
<td>2.28 / 0.42</td>
<td>2.86 / 0.83</td>
<td>6.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x0_5.yaml">PP-LCNet_x0_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_5_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-LCNet_x0_25</td>
<td>51.86</td>
<td>1.89 / 0.45</td>
<td>2.49 / 0.68</td>
<td>5.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x0_25.yaml">PP-LCNet_x0_25.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_25_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-LCNet_x0_35</td>
<td>58.09</td>
<td>1.94 / 0.41</td>
<td>2.73 / 0.77</td>
<td>5.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x0_35.yaml">PP-LCNet_x0_35.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_35_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_35_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-LCNet_x0_75</td>
<td>68.18</td>
<td>2.30 / 0.41</td>
<td>2.95 / 1.07</td>
<td>8.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x0_75.yaml">PP-LCNet_x0_75.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_75_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_75_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-LCNet_x1_0</td>
<td>71.32</td>
<td>2.35 / 0.47</td>
<td>4.03 / 1.35</td>
<td>10.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml">PP-LCNet_x1_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-LCNet_x1_5</td>
<td>73.71</td>
<td>2.33 / 0.53</td>
<td>4.17 / 2.29</td>
<td>16.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x1_5.yaml">PP-LCNet_x1_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_5_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-LCNet_x2_0</td>
<td>75.18</td>
<td>2.40 / 0.51</td>
<td>5.37 / 3.46</td>
<td>23.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x2_0.yaml">PP-LCNet_x2_0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x2_0_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x2_0_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-LCNet_x2_5</td>
<td>76.60</td>
<td>2.36 / 0.61</td>
<td>6.29 / 5.05</td>
<td>32.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNet_x2_5.yaml">PP-LCNet_x2_5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x2_5_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x2_5_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-LCNetV2_base</td>
<td>77.05</td>
<td>3.33 / 0.55</td>
<td>6.86 / 3.77</td>
<td>23.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNetV2_base.yaml">PP-LCNetV2_base.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_base_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_base_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-LCNetV2_large</td>
<td>78.51</td>
<td>4.37 / 0.71</td>
<td>9.43 / 8.07</td>
<td>37.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNetV2_large.yaml">PP-LCNetV2_large.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_large_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_large_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-LCNetV2_small</td>
<td>73.97</td>
<td>2.53 / 0.41</td>
<td>5.14 / 1.98</td>
<td>14.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/PP-LCNetV2_small.yaml">PP-LCNetV2_small.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_small_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>ResNet18_vd</td>
<td>72.3</td>
<td>2.47 / 0.61</td>
<td>6.97 / 5.15</td>
<td>41.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet18_vd.yaml">ResNet18_vd.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet18_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet18_vd_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>ResNet18</td>
<td>71.0</td>
<td>2.35 / 0.67</td>
<td>6.35 / 4.61</td>
<td>41.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet18.yaml">ResNet18.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet18_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet18_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>ResNet34_vd</td>
<td>76.0</td>
<td>4.01 / 1.03</td>
<td>11.99 / 9.86</td>
<td>77.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet34_vd.yaml">ResNet34_vd.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet34_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet34_vd_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>ResNet34</td>
<td>74.6</td>
<td>3.99 / 1.02</td>
<td>12.42 / 9.81</td>
<td>77.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet34.yaml">ResNet34.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet34_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet34_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>ResNet50_vd</td>
<td>79.1</td>
<td>6.04 / 1.16</td>
<td>16.08 / 12.07</td>
<td>90.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet50_vd.yaml">ResNet50_vd.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_vd_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>ResNet50</td>
<td>76.5</td>
<td>6.44 / 1.16</td>
<td>15.04 / 11.63</td>
<td>90.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet50.yaml">ResNet50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>ResNet101_vd</td>
<td>80.2</td>
<td>11.16 / 2.07</td>
<td>32.14 / 32.14</td>
<td>158.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet101_vd.yaml">ResNet101_vd.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet101_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet101_vd_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>ResNet101</td>
<td>77.6</td>
<td>10.91 / 2.06</td>
<td>31.14 / 22.93</td>
<td>158.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet101.yaml">ResNet101.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet101_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>ResNet152_vd</td>
<td>80.6</td>
<td>15.96 / 2.99</td>
<td>49.33 / 49.33</td>
<td>214.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet152_vd.yaml">ResNet152_vd.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet152_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet152_vd_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>ResNet152</td>
<td>78.3</td>
<td>15.61 / 2.90</td>
<td>47.33 / 36.60</td>
<td>214.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet152.yaml">ResNet152.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet152_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet152_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>ResNet200_vd</td>
<td>80.9</td>
<td>24.20 / 3.69</td>
<td>62.62 / 62.62</td>
<td>266.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/ResNet200_vd.yaml">ResNet200_vd.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet200_vd_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet200_vd_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>StarNet-S1</td>
<td>73.6</td>
<td>6.33 / 1.98</td>
<td>7.56 / 3.26</td>
<td>11.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/StarNet-S1.yaml">StarNet-S1.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S1_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S1_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>StarNet-S2</td>
<td>74.8</td>
<td>4.49 / 1.55</td>
<td>7.38 / 3.38</td>
<td>14.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/StarNet-S2.yaml">StarNet-S2.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S2_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S2_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>StarNet-S3</td>
<td>77.0</td>
<td>6.70 / 1.62</td>
<td>11.05 / 4.76</td>
<td>22.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/StarNet-S3.yaml">StarNet-S3.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S3_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S3_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>StarNet-S4</td>
<td>79.0</td>
<td>8.50 / 2.86</td>
<td>15.40 / 6.76</td>
<td>28.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/StarNet-S4.yaml">StarNet-S4.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S4_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S4_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>SwinTransformer_base_patch4_window7_224</td>
<td>83.37</td>
<td>14.29 / 5.13</td>
<td>130.89 / 130.89</td>
<td>310.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_base_patch4_window7_224.yaml">SwinTransformer_base_patch4_window7_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_base_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_base_patch4_window7_224_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>SwinTransformer_base_patch4_window12_384</td>
<td>84.17</td>
<td>37.74 / 10.10</td>
<td>362.56 / 362.56</td>
<td>311.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_base_patch4_window12_384.yaml">SwinTransformer_base_patch4_window12_384.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_base_patch4_window12_384_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_base_patch4_window12_384_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>SwinTransformer_large_patch4_window7_224</td>
<td>86.19</td>
<td>26.48 / 7.94</td>
<td>228.23 / 228.23</td>
<td>694.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_large_patch4_window7_224.yaml">SwinTransformer_large_patch4_window7_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_large_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_large_patch4_window7_224_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>SwinTransformer_large_patch4_window12_384</td>
<td>87.06</td>
<td>74.72 / 18.16</td>
<td>652.04 / 652.04</td>
<td>696.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_large_patch4_window12_384.yaml">SwinTransformer_large_patch4_window12_384.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_large_patch4_window12_384_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_large_patch4_window12_384_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>SwinTransformer_small_patch4_window7_224</td>
<td>83.21</td>
<td>10.37 / 3.90</td>
<td>94.20 / 94.20</td>
<td>175.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_small_patch4_window7_224.yaml">SwinTransformer_small_patch4_window7_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_small_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_small_patch4_window7_224_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>SwinTransformer_tiny_patch4_window7_224</td>
<td>81.10</td>
<td>6.66 / 2.15</td>
<td>60.45 / 60.45</td>
<td>100.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_classification/SwinTransformer_tiny_patch4_window7_224.yaml">SwinTransformer_tiny_patch4_window7_224.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_tiny_patch4_window7_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are based on the </b>[ImageNet-1k](https://www.image-net.org/index.php)<b> validation set Top1 Acc.</b>

## [Image Multi-label Classification Module](../module_usage/tutorials/cv_modules/image_multilabel_classification.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mAP (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>CLIP_vit_base_patch16_448_ML</td>
<td>89.15</td>
<td>54.75 / 14.30</td>
<td>280.23 / 280.23</td>
<td>325.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_multilabel_classification/CLIP_vit_base_patch16_448_ML.yaml">CLIP_vit_base_patch16_448_ML.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_448_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_448_ML_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>PP-HGNetV2-B0_ML</td>
<td>80.98</td>
<td>6.47 / 1.38</td>
<td>21.56 / 13.69</td>
<td>39.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_multilabel_classification/PP-HGNetV2-B0_ML.yaml">PP-HGNetV2-B0_ML.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_ML_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>PP-HGNetV2-B4_ML</td>
<td>87.96</td>
<td>9.63 / 2.79</td>
<td>43.98 / 36.63</td>
<td>88.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_multilabel_classification/PP-HGNetV2-B4_ML.yaml">PP-HGNetV2-B4_ML.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_ML_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>PP-HGNetV2-B6_ML</td>
<td>91.06</td>
<td>37.07 / 9.43</td>
<td>188.58 / 188.58</td>
<td>286.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_multilabel_classification/PP-HGNetV2-B6_ML.yaml">PP-HGNetV2-B6_ML.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_ML_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>PP-LCNet_x1_0_ML</td>
<td>77.96</td>
<td>4.04 / 1.15</td>
<td>11.76 / 8.32</td>
<td>29.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_multilabel_classification/PP-LCNet_x1_0_ML.yaml">PP-LCNet_x1_0_ML.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_ML_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>ResNet50_ML</td>
<td>83.42</td>
<td>12.12 / 3.27</td>
<td>51.79 / 44.36</td>
<td>108.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_multilabel_classification/ResNet50_ML.yaml">ResNet50_ML.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_ML_pretrained.pdparams">Training Model</a></td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are for the multi-label classification task mAP on [COCO2017](https://cocodataset.org/#home).</b>

## [Pedestrian Attribute Module](../module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mA (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_pedestrian_attribute</td>
<td>92.2</td>
<td>2.35 / 0.49</td>
<td>3.17 / 1.25</td>
<td>6.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/pedestrian_attribute_recognition/PP-LCNet_x1_0_pedestrian_attribute.yaml">PP-LCNet_x1_0_pedestrian_attribute.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_pedestrian_attribute_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pedestrian_attribute_pretrained.pdparams">Training Model</a></td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are for the internal PaddleX dataset mA.</b>

## [Vehicle Attribute Module](../module_usage/tutorials/cv_modules/vehicle_attribute_recognition.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mA (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_vehicle_attribute</td>
<td>91.7</td>
<td>2.32 / 2.32</td>
<td>3.22 / 1.26</td>
<td>6.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/vehicle_attribute_recognition/PP-LCNet_x1_0_vehicle_attribute.yaml">PP-LCNet_x1_0_vehicle_attribute.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_vehicle_attribute_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_vehicle_attribute_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are based on the VeRi dataset mA.</b>

## [Image Feature Module](../module_usage/tutorials/cv_modules/image_feature.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>recall@1 (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PP-ShiTuV2_rec</td>
<td>84.2</td>
<td>3.48 / 0.55</td>
<td>8.04 / 4.04</td>
<td>16.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml">PP-ShiTuV2_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_base</td>
<td>88.69</td>
<td>12.94 / 2.88</td>
<td>58.36 / 58.36</td>
<td>306.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_feature/PP-ShiTuV2_rec_CLIP_vit_base.yaml">PP-ShiTuV2_rec_CLIP_vit_base.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_CLIP_vit_base_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_base_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_large</td>
<td>91.03</td>
<td>51.65 / 11.18</td>
<td>255.78 / 255.78</td>
<td>1.05 G</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_feature/PP-ShiTuV2_rec_CLIP_vit_large.yaml">PP-ShiTuV2_rec_CLIP_vit_large.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_rec_CLIP_vit_large_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_large_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are based on the AliProducts recall@1.</b>

## [Document Orientation Classification Module](../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml">PP-LCNet_x1_0_doc_ori.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are based on the Top-1 Acc of the internal dataset of PaddleX.</b>

## [Face Feature Module](../module_usage/tutorials/cv_modules/face_feature.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Output Feature Dimension</th>
<th>Acc (%)<br/>AgeDB-30/CFP-FP/LFW</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>MobileFaceNet</td>
<td>128</td>
<td>96.28/96.71/99.58</td>
<td>3.16 / 0.48</td>
<td>6.49 / 6.49</td>
<td>4.1</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/face_recognition/MobileFaceNet.yaml">MobileFaceNet.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileFaceNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileFaceNet_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>ResNet50_face</td>
<td>512</td>
<td>98.12/98.56/99.77</td>
<td>5.68 / 1.09</td>
<td>14.96 / 11.90</td>
<td>87.2</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/face_recognition/ResNet50_face.yaml">ResNet50_face.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_face_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_face_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are measured on the AgeDB-30, CFP-FP, and LFW datasets.</b>

## [Main Body Detection Module](../module_usage/tutorials/cv_modules/mainbody_detection.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mAP (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PP-ShiTuV2_det</td>
<td>41.5</td>
<td>12.79 / 4.51</td>
<td>44.14 / 44.14</td>
<td>27.54</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/mainbody_detection/PP-ShiTuV2_det.yaml">PP-ShiTuV2_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-ShiTuV2_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_det_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are based on the [PaddleClas Main Body Detection Dataset](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/training/PP-ShiTu/mainbody_detection.md) mAP(0.5:0.95).</b>

## [Object Detection Module](../module_usage/tutorials/cv_modules/object_detection.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mAP (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>Cascade-FasterRCNN-ResNet50-FPN</td>
<td>41.1</td>
<td>135.92 / 135.92</td>
<td>-</td>
<td>245.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/Cascade-FasterRCNN-ResNet50-FPN.yaml">Cascade-FasterRCNN-ResNet50-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-FasterRCNN-ResNet50-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-FasterRCNN-ResNet50-FPN_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
<td>45.0</td>
<td>138.23 / 138.23</td>
<td>-</td>
<td>246.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml">Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>CenterNet-DLA-34</td>
<td>37.6</td>
<td>-</td>
<td>-</td>
<td>75.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/CenterNet-DLA-34.yaml">CenterNet-DLA-34.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CenterNet-DLA-34_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CenterNet-DLA-34_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>CenterNet-ResNet50</td>
<td>38.9</td>
<td>-</td>
<td>-</td>
<td>319.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/CenterNet-ResNet50.yaml">CenterNet-ResNet50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CenterNet-ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CenterNet-ResNet50_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>DETR-R50</td>
<td>42.3</td>
<td>62.91 / 17.33</td>
<td>392.63 / 392.63</td>
<td>159.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/DETR-R50.yaml">DETR-R50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DETR-R50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DETR-R50_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>FasterRCNN-ResNet34-FPN</td>
<td>37.8</td>
<td>83.33 / 31.64</td>
<td>-</td>
<td>137.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet34-FPN.yaml">FasterRCNN-ResNet34-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet34-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet34-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-FPN</td>
<td>38.4</td>
<td>107.08 / 35.40</td>
<td>-</td>
<td>148.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet50-FPN.yaml">FasterRCNN-ResNet50-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-vd-FPN</td>
<td>39.5</td>
<td>109.36 / 36.00</td>
<td>-</td>
<td>148.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet50-vd-FPN.yaml">FasterRCNN-ResNet50-vd-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-vd-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
<td>41.4</td>
<td>109.06 / 36.19</td>
<td>-</td>
<td>148.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml">FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>FasterRCNN-ResNet50</td>
<td>36.7</td>
<td>496.33 / 109.12</td>
<td>-</td>
<td>120.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet50.yaml">FasterRCNN-ResNet50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>FasterRCNN-ResNet101-FPN</td>
<td>41.4</td>
<td>148.21 / 42.21</td>
<td>-</td>
<td>216.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet101-FPN.yaml">FasterRCNN-ResNet101-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet101-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet101-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>FasterRCNN-ResNet101</td>
<td>39.0</td>
<td>538.58 / 120.88</td>
<td>-</td>
<td>188.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNet101.yaml">FasterRCNN-ResNet101.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNet101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet101_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>FasterRCNN-ResNeXt101-vd-FPN</td>
<td>43.4</td>
<td>258.01 / 58.25</td>
<td>-</td>
<td>360.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-ResNeXt101-vd-FPN.yaml">FasterRCNN-ResNeXt101-vd-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-ResNeXt101-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNeXt101-vd-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>FasterRCNN-Swin-Tiny-FPN</td>
<td>42.6</td>
<td>-</td>
<td>-</td>
<td>159.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FasterRCNN-Swin-Tiny-FPN.yaml">FasterRCNN-Swin-Tiny-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterRCNN-Swin-Tiny-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-Swin-Tiny-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>FCOS-ResNet50</td>
<td>39.6</td>
<td>106.13 / 28.32</td>
<td>721.79 / 721.79</td>
<td>124.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/FCOS-ResNet50.yaml">FCOS-ResNet50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FCOS-ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FCOS-ResNet50_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PicoDet-L</td>
<td>42.6</td>
<td>14.68 / 5.81</td>
<td>47.32 / 47.32</td>
<td>20.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PicoDet-L.yaml">PicoDet-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PicoDet-M</td>
<td>37.5</td>
<td>9.62 / 3.23</td>
<td>23.75 / 14.88</td>
<td>16.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PicoDet-M.yaml">PicoDet-M.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-M_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PicoDet-S</td>
<td>29.1</td>
<td>7.98 / 2.33</td>
<td>14.82 / 5.60</td>
<td>4.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PicoDet-S.yaml">PicoDet-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PicoDet-XS</td>
<td>26.2</td>
<td>9.66 / 2.75</td>
<td>19.15 / 7.24</td>
<td>5.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PicoDet-XS.yaml">PicoDet-XS.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-XS_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-XS_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-YOLOE_plus-L</td>
<td>52.9</td>
<td>33.55 / 10.46</td>
<td>189.05 / 189.05</td>
<td>185.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PP-YOLOE_plus-L.yaml">PP-YOLOE_plus-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-L_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-YOLOE_plus-M</td>
<td>49.8</td>
<td>19.52 / 7.46</td>
<td>113.36 / 113.36</td>
<td>83.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PP-YOLOE_plus-M.yaml">PP-YOLOE_plus-M.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-M_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-YOLOE_plus-S</td>
<td>43.7</td>
<td>12.16 / 4.58</td>
<td>73.86 / 52.90</td>
<td>28.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PP-YOLOE_plus-S.yaml">PP-YOLOE_plus-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-YOLOE_plus-X</td>
<td>54.7</td>
<td>58.87 / 15.84</td>
<td>292.93 / 292.93</td>
<td>349.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/PP-YOLOE_plus-X.yaml">PP-YOLOE_plus-X.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-X_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-X_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>RT-DETR-H</td>
<td>56.3</td>
<td>115.92 / 28.16</td>
<td>971.32 / 971.32</td>
<td>435.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/RT-DETR-H.yaml">RT-DETR-H.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>RT-DETR-L</td>
<td>53.0</td>
<td>35.00 / 10.45</td>
<td>495.51 / 495.51</td>
<td>113.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/RT-DETR-L.yaml">RT-DETR-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>RT-DETR-R18</td>
<td>46.5</td>
<td>20.21 / 6.23</td>
<td>266.01 / 266.01</td>
<td>70.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/RT-DETR-R18.yaml">RT-DETR-R18.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-R18_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-R18_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>RT-DETR-R50</td>
<td>53.1</td>
<td>42.14 / 11.31</td>
<td>523.97 / 523.97</td>
<td>149.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/RT-DETR-R50.yaml">RT-DETR-R50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-R50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-R50_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>RT-DETR-X</td>
<td>54.8</td>
<td>61.24 / 15.83</td>
<td>647.08 / 647.08</td>
<td>232.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/RT-DETR-X.yaml">RT-DETR-X.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-X_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-X_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>YOLOv3-DarkNet53</td>
<td>39.1</td>
<td>41.58 / 10.10</td>
<td>158.78 / 158.78</td>
<td>219.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOv3-DarkNet53.yaml">YOLOv3-DarkNet53.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-DarkNet53_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-DarkNet53_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>YOLOv3-MobileNetV3</td>
<td>31.4</td>
<td>16.53 / 5.70</td>
<td>60.44 / 60.44</td>
<td>83.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOv3-MobileNetV3.yaml">YOLOv3-MobileNetV3.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-MobileNetV3_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-MobileNetV3_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>YOLOv3-ResNet50_vd_DCN</td>
<td>40.6</td>
<td>32.91 / 10.07</td>
<td>225.72 / 224.32</td>
<td>163.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOv3-ResNet50_vd_DCN.yaml">YOLOv3-ResNet50_vd_DCN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOv3-ResNet50_vd_DCN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-ResNet50_vd_DCN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>YOLOX-L</td>
<td>50.1</td>
<td>121.19 / 13.55</td>
<td>295.38 / 274.15</td>
<td>192.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOX-L.yaml">YOLOX-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-L_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>YOLOX-M</td>
<td>46.9</td>
<td>87.19 / 10.09</td>
<td>183.95 / 172.67</td>
<td>90.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOX-M.yaml">YOLOX-M.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-M_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>YOLOX-N</td>
<td>26.1</td>
<td>53.31 / 45.02</td>
<td>69.69 / 59.18</td>
<td>3.4M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOX-N.yaml">YOLOX-N.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-N_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-N_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>YOLOX-S</td>
<td>40.4</td>
<td>129.52 / 13.19</td>
<td>181.39 / 179.01</td>
<td>32.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOX-S.yaml">YOLOX-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-S_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>YOLOX-T</td>
<td>32.9</td>
<td>66.81 / 61.31</td>
<td>92.30 / 83.90</td>
<td>18.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOX-T.yaml">YOLOX-T.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-T_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-T_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>YOLOX-X</td>
<td>51.8</td>
<td>156.40 / 20.17</td>
<td>480.14 / 454.35</td>
<td>351.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/object_detection/YOLOX-X.yaml">YOLOX-X.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOLOX-X_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-X_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are based on the COCO2017 validation set mAP(0.5:0.95).</b>

## [Small Object Detection Module](../module_usage/tutorials/cv_modules/small_object_detection.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mAP (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size</th>
<th>yaml File</th>
<th>Model Download Links</th></tr>
</thead>
<tbody>
<tr>
<td>PP-YOLOE_plus_SOD-S</td>
<td>25.1</td>
<td>135.68 / 122.94</td>
<td>188.09 / 107.74</td>
<td>77.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/small_object_detection/PP-YOLOE_plus_SOD-S.yaml">PP-YOLOE_plus_SOD-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-S_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<tr>
<td>PP-YOLOE_plus_SOD-L</td>
<td>31.9</td>
<td>57.1448</td>
<td>1006.98</td>
<td>325.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/small_object_detection/PP-YOLOE_plus_SOD-L.yaml">PP-YOLOE_plus_SOD-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-L_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-YOLOE_plus_SOD-largesize-L</td>
<td>42.7</td>
<td>458.521</td>
<td>11172.7</td>
<td>340.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/small_object_detection/PP-YOLOE_plus_SOD-largesize-L.yaml">PP-YOLOE_plus_SOD-largesize-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-largesize-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-largesize-L_pretrained.pdparams">Training Model</a></td>
</tr>


<b>Note: The above accuracy metrics are based on the validation set mAP(0.5:0.95) of </b>[VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset)<b>.</b>

## [Open-Vocabulary Object Detection](../module_usage/tutorials/cv_modules/open_vocabulary_detection.en.md)

<table>
<tr>
<th>Model</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>GroundingDINO-T</td>
<td>49.4</td>
<td>64.4</td>
<td>253.72</td>
<td>1807.4</td>
<td>658.3</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/GroundingDINO-T_infer.tar">Inference Model</a></td>
</tr>
</table>
<b>Note: The above accuracy metrics are based on the COCO val2017 validation set mAP(0.5:0.95). All models' GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## [Open Vocabulary Segmentation](../module_usage/tutorials/cv_modules/open_vocabulary_segmentation.en.md)

<table>
<tr>
<th>Model</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>SAM-H_box</td>
<td>144.9</td>
<td>33920.7</td>
<td>2433.7</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SAM-H_box_infer.tar">Inference Model</a></td>
</tr>
<tr>
<td>SAM-H_point</td>
<td>144.9</td>
<td>33920.7</td>
<td>2433.7</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SAM-H_point_infer.tar">Inference Model</a></td>
</tr>
</table>
<b>Note: All model GPU inference times are based on NVIDIA Tesla T4, with precision type FP32. CPU inference speed is based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads, and precision type FP32.</b>

## [Rotated Object Detection](../module_usage/tutorials/cv_modules/rotated_object_detection.en.md)

<table>
<tr>
<th>Model</th>
<th>mAP(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>PP-YOLOE-R-L</td>
<td>78.14</td>
<td>20.7039</td>
<td>157.942</td>
<td>211.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml">PP-YOLOE-R.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-R-L_infer.tar">Inference Model</a>/<a href="https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams">Training Model</a></td>
</tr>
</table>
<p><b>Note: The above accuracy metrics are based on the <a href="https://captain-whu.github.io/DOTA/">DOTA</a> validation set mAP(0.5:0.95). All model GPU inference times are based on NVIDIA TRX2080 Ti, with precision type F16. CPU inference speed is based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads, and precision type FP32.</b></p>

## [Pedestrian Detection Module](../module_usage/tutorials/cv_modules/human_detection.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mAP (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PP-YOLOE-L_human</td>
<td>48.0</td>
<td>33.27 / 9.19</td>
<td>173.72 / 173.72</td>
<td>196.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/human_detection/PP-YOLOE-L_human.yaml">PP-YOLOE-L_human.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-L_human_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-L_human_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>PP-YOLOE-S_human</td>
<td>42.5</td>
<td>9.94 / 3.42</td>
<td>54.48 / 46.52</td>
<td>28.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/human_detection/PP-YOLOE-S_human.yaml">PP-YOLOE-S_human.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-S_human_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-S_human_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are based on the validation set mAP(0.5:0.95) of </b>[CrowdHuman](https://bj.bcebos.com/v1/paddledet/data/crowdhuman.zip)<b>.</b>

## [Vehicle Detection Module](../module_usage/tutorials/cv_modules/vehicle_detection.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mAP (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PP-YOLOE-L_vehicle</td>
<td>63.9</td>
<td>32.84 / 9.03</td>
<td>176.60 / 176.60</td>
<td>196.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/vehicle_detection/PP-YOLOE-L_vehicle.yaml">PP-YOLOE-L_vehicle.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-L_vehicle_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-L_vehicle_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>PP-YOLOE-S_vehicle</td>
<td>61.3</td>
<td>9.79 / 3.48</td>
<td>54.14 / 46.69</td>
<td>28.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/vehicle_detection/PP-YOLOE-S_vehicle.yaml">PP-YOLOE-S_vehicle.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-S_vehicle_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-S_vehicle_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above precision metrics are based on the validation set mAP(0.5:0.95) of </b>[PPVehicle](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/modules/ppvehicle)<b></b>

## [Face Detection Module](../module_usage/tutorials/cv_modules/face_detection.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th style="text-align: center;">AP (%)<br/>Easy/Medium/Hard</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>BlazeFace</td>
<td style="text-align: center;">77.7/73.4/49.5</td>
<td>60.34 / 54.76</td>
<td>84.18 / 84.18</td>
<td>0.447 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/face_detection/BlazeFace.yaml">BlazeFace.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BlazeFace_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>BlazeFace-FPN-SSH</td>
<td style="text-align: center;">83.2/80.5/60.5</td>
<td>69.29 / 63.42</td>
<td>86.96 / 86.96</td>
<td>0.606 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/face_detection/BlazeFace-FPN-SSH.yaml">BlazeFace-FPN-SSH.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BlazeFace-FPN-SSH_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace-FPN-SSH_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>PicoDet_LCNet_x2_5_face</td>
<td style="text-align: center;">93.7/90.7/68.1</td>
<td>35.37 / 12.88</td>
<td>126.24 / 126.24</td>
<td>28.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml">PicoDet_LCNet_x2_5_face.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_LCNet_x2_5_face_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_LCNet_x2_5_face_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>PP-YOLOE_plus-S_face</td>
<td style="text-align: center;">93.9/91.8/79.8</td>
<td>22.54 / 8.33</td>
<td>138.67 / 138.67</td>
<td>26.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/face_detection/PP-YOLOE_plus-S_face.yaml">PP-YOLOE_plus-S_face</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_face_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_face_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>

**Note: The above precision metrics are evaluated on the WIDER-FACE validation set with an input size of 640x640.**

## [Anomaly Detection Module](../module_usage/tutorials/cv_modules/anomaly_detection.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mIoU</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>STFPM</td>
<td>0.9901</td>
<td>2.97 / 1.57</td>
<td>38.86 / 13.24</td>
<td>22.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/anomaly_detection/STFPM.yaml">STFPM.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/STFPM_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/STFPM_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above precision metrics are the average anomaly scores on the validation set of </b>[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)<b>.</b>

## [Human Keypoint Detection Module](../module_usage/tutorials//cv_modules/human_keypoint_detection.en.md)

## [Human Keypoint Detection Module](../module_usage/tutorials//cv_modules/human_keypoint_detection.md)

<table>
<tr>
<th>Model</th>
<th>Scheme</th>
<th>Input Size</th>
<th>AP(0.5:0.95)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
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
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TinyPose_128x96_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TinyPose_128x96_pretrained.pdparams">Training Model</a></td>
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
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TinyPose_256x192_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TinyPose_256x192_pretrained.pdparams">Training Model</a></td>
</tr>
</table>

**Note: The above accuracy metrics are based on the COCO dataset AP(0.5:0.95), with detection boxes obtained from ground truth annotations. All GPU inference times are based on an NVIDIA Tesla T4 machine, with precision type FP32. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads and precision type FP32.**

## [3D Multi-modal Fusion Detection Module](../module_usage/tutorials//cv_modules/3d_bev_detection.en.md)

<table>
<tr>
<th>Model</th>
<th>mAP(%)</th>
<th>NDS</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>BEVFusion</td>
<td>53.9</td>
<td>60.9</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/3d_bev_detection/BEVFusion.yaml">BEVFusion.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BEVFusion_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BEVFusion_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
<p><b>Note: The above accuracy metrics are based on the <a href="https://www.nuscenes.org/nuscenes">nuscenes</a> validation set with mAP(0.5:0.95) and NDS 60.9, and the precision type is FP32.</b></p>

## [Semantic Segmentation Module](../module_usage/tutorials/cv_modules/semantic_segmentation.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mloU（%）</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>Deeplabv3_Plus-R50</td>
<td>80.36</td>
<td>503.51 / 122.30</td>
<td>3543.91 / 3543.91</td>
<td>94.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/Deeplabv3_Plus-R50.yaml">Deeplabv3_Plus-R50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R50_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>Deeplabv3_Plus-R101</td>
<td>81.10</td>
<td>803.79 / 175.45</td>
<td>5136.21 / 5136.21</td>
<td>162.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/Deeplabv3_Plus-R101.yaml">Deeplabv3_Plus-R101.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R101_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>Deeplabv3-R50</td>
<td>79.90</td>
<td>647.56 / 121.67</td>
<td>3803.09 / 3803.09</td>
<td>138.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/Deeplabv3-R50.yaml">Deeplabv3-R50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3-R50_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>Deeplabv3-R101</td>
<td>80.85</td>
<td>950.43 / 178.50</td>
<td>5517.14 / 5517.14</td>
<td>205.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/Deeplabv3-R101.yaml">Deeplabv3-R101.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3-R101_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>OCRNet_HRNet-W18</td>
<td>80.67</td>
<td>286.12 / 80.76</td>
<td>1794.03 / 1794.03</td>
<td>43.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/OCRNet_HRNet-W18.yaml">OCRNet_HRNet-W18.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W18_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W18_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>OCRNet_HRNet-W48</td>
<td>82.15</td>
<td>627.36 / 170.76</td>
<td>3531.61 / 3531.61</td>
<td>249.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/OCRNet_HRNet-W48.yaml">OCRNet_HRNet-W48.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W48_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W48_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-LiteSeg-T</td>
<td>73.10</td>
<td>30.16 / 14.03</td>
<td>420.07 / 235.01</td>
<td>28.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml">PP-LiteSeg-T.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-T_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-T_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-LiteSeg-B</td>
<td>75.25</td>
<td>40.92 / 20.18</td>
<td>494.32 / 310.34</td>
<td>47.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-B.yaml">PP-LiteSeg-B.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-B_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-B_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>SegFormer-B0 (slice)</td>
<td>76.73</td>
<td>11.1946</td>
<td>268.929</td>
<td>13.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SegFormer-B0.yaml">SegFormer-B0.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B0 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B0 (slice)_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>SegFormer-B1 (slice)</td>
<td>78.35</td>
<td>17.9998</td>
<td>403.393</td>
<td>48.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SegFormer-B1.yaml">SegFormer-B1.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B1 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B1 (slice)_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>SegFormer-B2 (slice)</td>
<td>81.60</td>
<td>48.0371</td>
<td>1248.52</td>
<td>96.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SegFormer-B2.yaml">SegFormer-B2.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B2 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B2 (slice)_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>SegFormer-B3 (slice)</td>
<td>82.47</td>
<td>64.341</td>
<td>1666.35</td>
<td>167.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SegFormer-B3.yaml">SegFormer-B3.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B3 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B3 (slice)_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>SegFormer-B4 (slice)</td>
<td>82.38</td>
<td>82.4336</td>
<td>1995.42</td>
<td>226.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SegFormer-B4.yaml">SegFormer-B4.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B4 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B4 (slice)_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>SegFormer-B5 (slice)</td>
<td>82.58</td>
<td>97.3717</td>
<td>2420.19</td>
<td>229.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SegFormer-B5.yaml">SegFormer-B5.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B5 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B5 (slice)_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are based on the </b>[Cityscapes](https://www.cityscapes-dataset.com/)<b> dataset mIoU.</b>
<table>
<thead>
<tr>
<th>Model Name</th>
<th>mIoU (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>SeaFormer_base(slice)</td>
<td>40.92</td>
<td>24.4073</td>
<td>397.574</td>
<td>30.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SeaFormer_base.yaml">SeaFormer_base.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_base(slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_base(slice)_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>SeaFormer_large (slice)</td>
<td>43.66</td>
<td>27.8123</td>
<td>550.464</td>
<td>49.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SeaFormer_large.yaml">SeaFormer_large.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_large (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_large (slice)_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>SeaFormer_small (slice)</td>
<td>38.73</td>
<td>19.2295</td>
<td>358.343</td>
<td>14.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SeaFormer_small.yaml">SeaFormer_small.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_small (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_small (slice)_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>SeaFormer_tiny (slice)</td>
<td>34.58</td>
<td>13.9496</td>
<td>330.132</td>
<td>6.1M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/semantic_segmentation/SeaFormer_tiny.yaml">SeaFormer_tiny.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_tiny (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_tiny (slice)_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are based on the </b>[ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)<b> dataset. "Slice" indicates that the input images have been cropped.</b>

## [Instance Segmentation Module](../module_usage/tutorials/cv_modules/instance_segmentation.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Mask AP</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>Mask-RT-DETR-H</td>
<td>50.6</td>
<td>172.36 / 172.36</td>
<td>1615.75 / 1615.75</td>
<td>449.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-H.yaml">Mask-RT-DETR-H.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-H_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-H_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>Mask-RT-DETR-L</td>
<td>45.7</td>
<td>88.18 / 88.18</td>
<td>1090.84 / 1090.84</td>
<td>113.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-L.yaml">Mask-RT-DETR-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-L_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>Mask-RT-DETR-M</td>
<td>42.7</td>
<td>78.69 / 78.69</td>
<td>-</td>
<td>66.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-M.yaml">Mask-RT-DETR-M.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-M_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>Mask-RT-DETR-S</td>
<td>41.0</td>
<td>33.5007</td>
<td>-</td>
<td>51.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-S.yaml">Mask-RT-DETR-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-S_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>Mask-RT-DETR-X</td>
<td>47.5</td>
<td>114.16 / 114.16</td>
<td>1240.92 / 1240.92</td>
<td>237.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-X.yaml">Mask-RT-DETR-X.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-X_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-X_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>Cascade-MaskRCNN-ResNet50-FPN</td>
<td>36.3</td>
<td>141.69 / 141.69</td>
<td>-</td>
<td>254.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Cascade-MaskRCNN-ResNet50-FPN.yaml">Cascade-MaskRCNN-ResNet50-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-MaskRCNN-ResNet50-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-MaskRCNN-ResNet50-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN</td>
<td>39.1</td>
<td>147.62 / 147.62</td>
<td>-</td>
<td>254.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN.yaml">Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MaskRCNN-ResNet50-FPN</td>
<td>35.6</td>
<td>118.30 / 118.30</td>
<td>-</td>
<td>157.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/MaskRCNN-ResNet50-FPN.yaml">MaskRCNN-ResNet50-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MaskRCNN-ResNet50-vd-FPN</td>
<td>36.4</td>
<td>118.34 / 118.34</td>
<td>-</td>
<td>157.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/MaskRCNN-ResNet50-vd-FPN.yaml">MaskRCNN-ResNet50-vd-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50-vd-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MaskRCNN-ResNet50</td>
<td>32.8</td>
<td>228.83 / 228.83</td>
<td>-</td>
<td>127.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/MaskRCNN-ResNet50.yaml">MaskRCNN-ResNet50.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet50_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MaskRCNN-ResNet101-FPN</td>
<td>36.6</td>
<td>148.14 / 148.14</td>
<td>-</td>
<td>225.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/MaskRCNN-ResNet101-FPN.yaml">MaskRCNN-ResNet101-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet101-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet101-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MaskRCNN-ResNet101-vd-FPN</td>
<td>38.1</td>
<td>151.12 / 151.12</td>
<td>-</td>
<td>225.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/MaskRCNN-ResNet101-vd-FPN.yaml">MaskRCNN-ResNet101-vd-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNet101-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNet101-vd-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>MaskRCNN-ResNeXt101-vd-FPN</td>
<td>39.5</td>
<td>237.55 / 237.55</td>
<td>-</td>
<td>370.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/MaskRCNN-ResNeXt101-vd-FPN.yaml">MaskRCNN-ResNeXt101-vd-FPN.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskRCNN-ResNeXt101-vd-FPN_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskRCNN-ResNeXt101-vd-FPN_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-YOLOE_seg-S</td>
<td>32.5</td>
<td>-</td>
<td>-</td>
<td>31.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/instance_segmentation/PP-YOLOE_seg-S.yaml">PP-YOLOE_seg-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_seg-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_seg-S_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>SOLOv2</td>
<td>35.5</td>
<td>-</td>
<td>-</td>
<td>179.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/paddlex/configs/modules/instance_segmentation/SOLOv2.yaml">SOLOv2.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SOLOv2_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SOLOv2_pretrained.pdparams">Training Model</a></td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are based on the Mask AP(0.5:0.95) on the [COCO2017](https://cocodataset.org/#home) validation set.</b>

## [Text Detection Module](../module_usage/tutorials/ocr_modules/text_detection.en.md)

<table>
<thead>
<tr>
<th>Model</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_det</td>
<td>82.56</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>109</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv4_server_det.yaml">PP-OCRv4_server_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td>77.35</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml">PP-OCRv4_mobile_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-OCRv3_mobile_det</td>
<td>78.68</td>
<td>8.44 / 2.91</td>
<td>27.87 / 27.87</td>
<td>2.1</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv3_mobile_det.yaml">PP-OCRv3_mobile_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv3_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_det_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-OCRv3_server_det</td>
<td>80.11</td>
<td>65.41 / 13.67</td>
<td>305.07 / 305.07</td>
<td>102.1</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv3_server_det.yaml">PP-OCRv3_server_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv3_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_server_det_pretrained.pdparams">Training Model</a></td>
</tr>
</tbody>
</table>
<b>Note: The evaluation dataset for the above accuracy metrics is the self-built Chinese and English dataset of PaddleOCR, covering multiple scenarios such as street view, web images, documents, and handwriting, with 593 images for text recognition. The GPU inference time for all models is based on an NVIDIA Tesla T4 machine with FP32 precision type, while the CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision type.</b>

## [Seal Text Detection Module](../module_usage/tutorials/ocr_modules/seal_text_detection.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_seal_det</td>
<td>96.47</td>
<td>7.82 / 3.09</td>
<td>48.28 / 23.97</td>
<td>4.7M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/seal_text_detection/PP-OCRv4_mobile_seal_det.yaml">PP-OCRv4_mobile_seal_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>PP-OCRv4_server_seal_det</td>
<td>98.21</td>
<td>74.75 / 67.72</td>
<td>382.55 / 382.55</td>
<td>108.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/seal_text_detection/PP-OCRv4_server_seal_det.yaml">PP-OCRv4_server_seal_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The evaluation set for the above precision metrics is the seal dataset built by PaddleX, which includes 500 seal images.</b>

## [Text Recognition Module](../module_usage/tutorials/ocr_modules/text_recognition.en.md)

* <b>Chinese Text Recognition Models</b>
<table>
<tr>
<th>Model</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td>
<td>81.53</td>
<td>6.65 / 6.65</td>
<td>32.92 / 32.92</td>
<td>74.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv4_server_rec_doc.yaml">PP-OCRv4_server_rec_doc.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td>78.74</td>
<td>4.82 / 4.82</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml">PP-OCRv4_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td>
<td>80.61 </td>
<td>6.58 / 6.58</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv4_server_rec.yaml">PP-OCRv4_server_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td>
<td>72.96</td>
<td>5.87 / 5.87</td>
<td>9.07 / 4.28</td>
<td>9.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv3_mobile_rec.yaml">PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
</table>
<p><b>Note: The evaluation set for the above accuracy metrics is a Chinese dataset built by PaddleOCR, covering multiple scenarios such as street view, web images, documents, and handwriting, with 8367 images for text recognition. All models' GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision, while CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<table>
<tr>
<th>Model</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td>
<td>68.81</td>
<td>8.08 / 8.08</td>
<td>50.17 / 42.50</td>
<td>73.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/ch_SVTRv2_rec.yaml">ch_SVTRv2_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
<p><b>Note: The evaluation dataset for the above accuracy metrics is the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task</a> Leaderboard A. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<table>
<tr>
<th>Model</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td>
<td>65.07</td>
<td>5.93 / 5.93</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/ch_RepSVTR_rec.yaml">ch_RepSVTR_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
<p><b>Note: The evaluation dataset for the above accuracy metrics is the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task</a> Leaderboard B. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<b>English Recognition Model</b>
<table>
<tr>
<th>Model</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td>
<td> 70.39</td>
<td>4.81 / 4.81</td>
<td>16.10 / 5.31</td>
<td>6.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/en_PP-OCRv4_mobile_rec.yaml">en_PP-OCRv4_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td>
<td>70.69</td>
<td>5.44 / 5.44</td>
<td>8.65 / 5.57</td>
<td>7.8 M </td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/en_PP-OCRv3_mobile_rec.yaml">en_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/en_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
</table>
<b>Multilingual Recognition Model</b>
<table>
<tr>
<th>Model</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td>
<td>60.21</td>
<td>5.40 / 5.40</td>
<td>9.11 / 4.05</td>
<td>8.6 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/korean_PP-OCRv3_mobile_rec.yaml">korean_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/korean_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td>
<td>45.69</td>
<td>5.70 / 5.70</td>
<td>8.48 / 4.07</td>
<td>8.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/japan_PP-OCRv3_mobile_rec.yaml">japan_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/japan_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td>
<td>82.06</td>
<td>5.90 / 5.90</td>
<td>9.28 / 4.34</td>
<td>9.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/chinese_cht_PP-OCRv3_mobile_rec.yaml">chinese_cht_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td>
<td>95.88</td>
<td>5.42 / 5.42</td>
<td>8.10 / 6.91</td>
<td>7.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/te_PP-OCRv3_mobile_rec.yaml">te_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/te_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td>
<td>96.96</td>
<td>5.25 / 5.25</td>
<td>9.09 / 3.86</td>
<td>8.0 M </td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/ka_PP-OCRv3_mobile_rec.yaml">ka_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ka_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td>
<td>76.83</td>
<td>5.23 / 5.23</td>
<td>10.13 / 4.30</td>
<td>8.0 M </td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/ta_PP-OCRv3_mobile_rec.yaml">ta_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ta_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td>
<td>76.93</td>
<td>5.20 / 5.20</td>
<td>8.83 / 7.15</td>
<td>7.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/latin_PP-OCRv3_mobile_rec.yaml">latin_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/latin_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td>
<td>73.55</td>
<td>5.35 / 5.35</td>
<td>8.80 / 4.56</td>
<td>7.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/arabic_PP-OCRv3_mobile_rec.yaml">arabic_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/arabic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td>
<td>94.28</td>
<td>5.23 / 5.23</td>
<td>8.89 / 3.88</td>
<td>7.9 M  </td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/cyrillic_PP-OCRv3_mobile_rec.yaml">cyrillic_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td>
<td>96.44</td>
<td>5.22 / 5.22</td>
<td>8.56 / 4.06</td>
<td>7.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/devanagari_PP-OCRv3_mobile_rec.yaml">devanagari_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
</tr>
</table>
<p><b>Note: The evaluation set for the above accuracy metrics is a multi-language dataset built by PaddleX. All model GPU inference times are based on NVIDIA Tesla T4 machines, with precision type FP32. CPU inference speed is based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads, and precision type FP32.</b></p>

## [Formula Recognition Module](../module_usage/tutorials/ocr_modules/formula_recognition.en.md)

<table>
<tr>
<th>Model</th>
<th>Avg-BLEU</th>
<th>GPU Inference Time (ms)</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
<td>UniMERNet</td>
<td>0.8613</td>
<td>2266.96</td>
<td>1.4 G</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/UniMERNet.yaml">UniMERNet.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UniMERNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">Training Model</a></td>
<tr>
<td>PP-FormulaNet-S</td>
<td>0.8712</td>
<td>202.25</td>
<td>167.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/PP-FormulaNet-S.yaml">PP-FormulaNet-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-FormulaNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">Training Model</a></td>
</tr>
<td>PP-FormulaNet-L</td>
<td>0.9213</td>
<td>1976.52</td>
<td>535.2 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/PP-FormulaNet-L.yaml">PP-FormulaNet-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-FormulaNet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">Training Model</a></td>
<tr>
<td>LaTeX_OCR_rec</td>
<td>0.7163</td>
<td>-</td>
<td>89.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/LaTeX_OCR_rec.yaml">LaTeX_OCR_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
<b>Note: The above accuracy metrics are measured from the internal formula recognition test set of PaddleX. The BLEU score of LaTeX_OCR_rec on the LaTeX-OCR formula recognition test set is 0.8821. All model GPU inference times are based on Tesla V100 GPUs, with precision type FP32.</b>

## [Table Structure Recognition Module](../module_usage/tutorials/ocr_modules/table_structure_recognition.en.md)

<table>
<tr>
<th>Model</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>SLANet</td>
<td>59.52</td>
<td>103.08 / 103.08</td>
<td>197.99 / 197.99</td>
<td>6.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_structure_recognition/SLANet.yaml">SLANet.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>SLANet_plus</td>
<td>63.69</td>
<td>140.29 / 140.29</td>
<td>195.39 / 195.39</td>
<td>6.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_structure_recognition/SLANet_plus.yaml">SLANet_plus.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_plus_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td rowspan="2">69.65</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_structure_recognition/SLANeXt_wired.yaml">SLANeXt_wired.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANeXt_wired_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>SLANeXt_wireless</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_structure_recognition/SLANeXt_wireless.yaml">SLANeXt_wireless.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANeXt_wireless_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wireless_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
<b>Note: The above accuracy metrics are measured from the high-difficulty Chinese table recognition dataset built internally by PaddleX. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision type. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision type.</b>

## [Table Cell Detection Module](../module_usage/tutorials/ocr_modules/table_cells_detection.en.md)

<table>
<tr>
<th>Model</th>
<th>mAP(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_cells_detection/RT-DETR-L_wired_table_cell_det.yaml">RT-DETR-L_wired_table_cell_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wired_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_cells_detection/RT-DETR-L_wireless_table_cell_det.yaml">RT-DETR-L_wireless_table_cell_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wireless_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
<p><b>Note: The above accuracy metrics are measured from the internal table cell detection dataset of PaddleX. All model GPU inference times are based on an NVIDIA Tesla T4 machine, with precision type FP32. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads, and precision type FP32.</b></p>

## [Table Classification Module](../module_usage/tutorials/ocr_modules/table_classification.en.md)

<table>
<tr>
<th>Model</th>
<th>Top1 Acc(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td>
<td>--</td>
<td>--</td>
<td>--</td>
<td>--</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_classification/PP-LCNet_x1_0_table_cls.yaml">PP-LCNet_x1_0_table_cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
<p><b>Note: The above accuracy metrics are measured from the internal table classification dataset built by PaddleX. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## [Text Image Unwarping Module](../module_usage/tutorials/ocr_modules/text_image_unwarping.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>MS-SSIM (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td>
<td>54.40</td>
<td>16.27 / 7.76</td>
<td>176.97 / 80.60</td>
<td>30.3 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_unwarping/UVDoc.yaml">UVDoc.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are measured from the image unwarping dataset built by PaddleX.</b>

## [Layout Detection Module](../module_usage/tutorials/ocr_modules/layout_detection.en.md)

* <b>Table Layout Detection Model</b>
<table>
<thead>
<tr>
<th>Model</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x_table</td>
<td>97.5</td>
<td>8.02 / 3.09</td>
<td>23.70 / 20.41</td>
<td>7.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet_layout_1x_table.yaml">PicoDet_layout_1x_table.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Training Model</a></td>
</tr>
</tbody></table>
<b>Note: The evaluation set for the above accuracy metrics is the layout table area detection dataset built by PaddleOCR, which contains 7835 images of document types with tables in both Chinese and English. The GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision, and the CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>
<b>3 types of layout detection models, including tables, images, and seals</b>
<table>
<thead>
<tr>
<th>Model</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_3cls</td>
<td>88.2</td>
<td>8.99 / 2.22</td>
<td>16.11 / 8.73</td>
<td>4.8</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet-S_layout_3cls.yaml">PicoDet-S_layout_3cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td>
<td>89.0</td>
<td>13.05 / 4.50</td>
<td>41.30 / 41.30</td>
<td>22.6</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet-L_layout_3cls.yaml">PicoDet-L_layout_3cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td>
<td>95.8</td>
<td>114.93 / 27.71</td>
<td>947.56 / 947.56</td>
<td>470.1</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/RT-DETR-H_layout_3cls.yaml">RT-DETR-H_layout_3cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Training Model</a></td>
</tr>
</tbody></table>
<b>Note: The evaluation dataset for the above accuracy metrics is the layout area detection dataset built by PaddleOCR, which includes 1,154 common types of document images such as Chinese and English papers, magazines, and research reports. The GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision, and the CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

* <b>5-class English document layout detection model, including text, title, table, image, and list</b>
<table>
<thead>
<tr>
<th>Model</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td>
<td>97.8</td>
<td>9.03 / 3.10</td>
<td>25.82 / 20.70</td>
<td>7.4</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet_layout_1x.yaml">PicoDet_layout_1x.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Training Model</a></td>
</tr>
</tbody></table>
<b>Note: The evaluation dataset for the above accuracy metrics is the [PubLayNet](https://developer.ibm.com/exchanges/data/all/publaynet/) evaluation dataset, which contains 11,245 images of English documents. The GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

* <b>17-class layout detection model, including 17 common layout categories: paragraph title, image, text, number, abstract, content, figure title, formula, table, table title, reference, document title, footnote, header, algorithm, footer, and seal</b>
<table>
<thead>
<tr>
<th>Model</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td>
<td>87.4</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet-S_layout_17cls.yaml">PicoDet-S_layout_17cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Training Model</a></td>
</tr>
</tbody></table>
<tr>
<td>PicoDet-L_layout_17cls</td>
<td>89.0</td>
<td>17.2</td>
<td>160.2</td>
<td>22.6</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet-L_layout_17cls.yaml">PicoDet-L_layout_17cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td>
<td>98.3</td>
<td>115.1</td>
<td>3827.2</td>
<td>470.2</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/RT-DETR-H_layout_17cls.yaml">RT-DETR-H_layout_17cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Training Model</a></td>
</tr>


<b>Note: The evaluation set for the above accuracy metrics is the layout area detection dataset built by PaddleOCR, which includes 892 images of common document types such as Chinese and English papers, magazines, and research reports. The GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## [Document Image Orientation Classification Module](../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.en.md)

<table>
<thead>
<tr>
<th>Model</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml">PP-LCNet_x1_0_doc_ori.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
</tr>
</tbody>
</table>
<b>Note: The evaluation set for the above accuracy metrics is a self-built dataset covering multiple scenarios such as documents and certificates, with 1000 images. The GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## [Time Series Forecasting Module](../module_usage/tutorials/time_series_modules/time_series_forecasting.en.md)

<table>
<thead>
<tr>
<th>Model Name</th>
<th>mse</th>
<th>mae</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>DLinear</td>
<td>0.382</td>
<td>0.394</td>
<td>72 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/DLinear.yaml">DLinear.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DLinear_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DLinear_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>NLinear</td>
<td>0.386</td>
<td>0.392</td>
<td>40 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/NLinear.yaml">NLinear.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/NLinear_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/NLinear_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>Nonstationary</td>
<td>0.600</td>
<td>0.515</td>
<td>55.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/Nonstationary.yaml">Nonstationary.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Nonstationary_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Nonstationary_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>PatchTST</td>
<td>0.379</td>
<td>0.391</td>
<td>2.0 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/PatchTST.yaml">PatchTST.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PatchTST_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PatchTST_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>RLinear</td>
<td>0.385</td>
<td>0.392</td>
<td>40 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/RLinear.yaml">RLinear.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RLinear_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RLinear_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>TiDE</td>
<td>0.407</td>
<td>0.414</td>
<td>31.7 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/TiDE.yaml">TiDE.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TiDE_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TiDE_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>TimesNet</td>
<td>0.416</td>
<td>0.429</td>
<td>4.9 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_forecast/TimesNet.yaml">TimesNet.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are measured from the </b>[ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar)<b> dataset </b><b>(evaluation results on the test.csv test set)</b><b>.</b>

## [Time Series Anomaly Detection Module](../module_usage/tutorials/time_series_modules/time_series_anomaly_detection.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Precision</th>
<th>Recall</th>
<th>F1 Score</th>
<th>Model Storage Size</th>
<th>YAML File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>AutoEncoder_ad</td>
<td>99.36</td>
<td>84.36</td>
<td>91.25</td>
<td>52 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_anomaly_detection/AutoEncoder_ad.yaml">AutoEncoder_ad.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/AutoEncoder_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/AutoEncoder_ad_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>DLinear_ad</td>
<td>98.98</td>
<td>93.96</td>
<td>96.41</td>
<td>112 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_anomaly_detection/DLinear_ad.yaml">DLinear_ad.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DLinear_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DLinear_ad_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>Nonstationary_ad</td>
<td>98.55</td>
<td>88.95</td>
<td>93.51</td>
<td>1.8 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_anomaly_detection/Nonstationary_ad.yaml">Nonstationary_ad.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Nonstationary_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Nonstationary_ad_pretrained.pdparams">Training Model</a></td></tr>
<tr>
<td>PatchTST_ad</td>
<td>98.78</td>
<td>90.70</td>
<td>94.57</td>
<td>320 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_anomaly_detection/PatchTST_ad.yaml">PatchTST_ad.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PatchTST_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PatchTST_ad_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above precision metrics are measured from the </b>[PSM](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar)<b> dataset.</b>

## [Time Series Classification Module](../module_usage/tutorials/time_series_modules/time_series_classification.en.md)
<table>
<thead>
<tr>
<th>Model Name</th>
<th>acc(%)</th>
<th>Model Storage Size</th>
<th>yaml File</th>
<th>Model Download Link</th></tr>
</thead>
<tbody>
<tr>
<td>TimesNet_cls</td>
<td>87.5</td>
<td>792 K</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/ts_classification/TimesNet_cls.yaml">TimesNet_cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_cls_pretrained.pdparams">Training Model</a></td></tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are measured from the [UWaveGestureLibrary](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TEST.csv) dataset.</b>

&gt;<b>Note: The GPU inference time for all models above is based on an NVIDIA Tesla T4 machine with FP32 precision. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## [Multilingual Speech Recognition Module](../module_usage/tutorials/speech_modules/multilingual_speech_recognition.en.md)

<table>
<tr>
<th>Model</th>
<th>Training Data</th>
<th>Model Size</th>
<th>Word Error Rate</th>
<th>YAML File</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>whisper_large</td>
<td>680kh</td>
<td>5.8G</td>
<td>2.7 (Librispeech)</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/multilingual_speech_recognition/whisper_large.yaml">whisper_large.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_large.tar">Inference Model</a></td>
</tr>
<tr>
<td>whisper_medium</td>
<td>680kh</td>
<td>2.9G</td>
<td>-</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/multilingual_speech_recognition/whisper_medium.yaml">whisper_medium.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_medium.tar">Inference Model</a></td>
</tr>
<tr>
<td>whisper_small</td>
<td>680kh</td>
<td>923M</td>
<td>-</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/multilingual_speech_recognition/whisper_small.yaml">whisper_small.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_small.tar">Inference Model</a></td>
</tr>
<tr>
<td>whisper_base</td>
<td>680kh</td>
<td>277M</td>
<td>-</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/multilingual_speech_recognition/whisper_base.yaml">whisper_base.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_base.tar">Inference Model</a></td>
</tr>
<tr>
<td>whisper_tiny</td>
<td>680kh</td>
<td>145M</td>
<td>-</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/multilingual_speech_recognition/whisper_small.yaml">whisper_small.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_tiny.tar">Inference Model</a></td>
</tr>
</table>

## [Video Classification Module](../module_usage/tutorials/video_modules/video_classification.en.md)

<table>
<tr>
<th>Model</th>
<th>Top1 Acc(%)</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>PP-TSM-R50_8frames_uniform</td>
<td>74.36</td>
<td>93.4 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/video_classification/PP-TSM-R50_8frames_uniform.yaml">PP-TSM-R50_8frames_uniform.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSM-R50_8frames_uniform_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSM-R50_8frames_uniform_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-TSMv2-LCNetV2_8frames_uniform</td>
<td>71.71</td>
<td>22.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/video_classification/PP-TSMv2-LCNetV2_8frames_uniform.yaml">PP-TSMv2-LCNetV2_8frames_uniform.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSMv2-LCNetV2_8frames_uniform_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSMv2-LCNetV2_8frames_uniform_pretrained.pdparams">Training Model</a></td>
</tr>
<tr>
<td>PP-TSMv2-LCNetV2_16frames_uniform</td>
<td>73.11</td>
<td>22.5 M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/video_classification/PP-TSMv2-LCNetV2_16frames_uniform.yaml">PP-TSMv2-LCNetV2_16frames_uniform.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSMv2-LCNetV2_16frames_uniform_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSMv2-LCNetV2_16frames_uniform_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
<p><b>Note: The above accuracy metrics are based on the <a href="https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/dataset/k400.md">K400</a> validation set Top1 Acc.</b></p>

## [Video Detection Module](../module_usage/tutorials/video_modules/video_detection.en.md)

<table>
<tr>
<th>Model</th>
<th>Frame-mAP(@ IoU 0.5)</th>
<th>Model Storage Size (M)</th>
<th>yaml File</th>
<th>Model Download Link</th>
</tr>
<tr>
<td>YOWO</td>
<td>80.94</td>
<td>462.891M</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/video_detection/YOWO.yaml">YOWO.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOWO_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOWO_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
<p><b>Note: The above accuracy metrics are based on the test dataset <a href="http://www.thumos.info/download.html">UCF101-24</a>, using the Frame-mAP (@ IoU 0.5) metric. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
