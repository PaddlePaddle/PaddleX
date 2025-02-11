---
comments: true
---

# 图像分类模块使用教程

## 一、概述
图像分类模块是计算机视觉系统中的关键组成部分，负责对输入的图像进行分类。该模块的性能直接影响到整个计算机视觉系统的准确性和效率。图像分类模块通常会接收图像作为输入，然后通过深度学习或其他机器学习算法，根据图像的特性和内容，将其分类到预定义的类别中。例如，对于一个动物识别系统，图像分类模块可能需要将输入的图像分类为“猫”、“狗”、“马”等类别。图像分类模块的分类结果将作为输出，供其他模块或系统使用。

## 二、支持模型列表


<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top1 Acc(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
</tr>
<tr>
<td>CLIP_vit_base_patch16_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_224_pretrained.pdparams">训练模型</a></td>
<td>85.36</td>
<td>12.84 / 2.82</td>
<td>60.52 / 60.52</td>
<td>306.5 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_0_pretrained.pdparams">训练模型</a></td>
<td>68.2</td>
<td>3.76 / 0.53</td>
<td>5.11 / 1.43</td>
<td>10.5 M</td>
</tr>
<tr>
<td>PP-HGNet_small</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_small_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_small_pretrained.pdparams">训练模型</a></td>
<td>81.51</td>
<td>5.12 / 1.73</td>
<td>25.01 / 25.01</td>
<td>86.5 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_pretrained.pdparams">训练模型</a></td>
<td>77.77</td>
<td>3.83 / 0.57</td>
<td>9.95 / 2.37</td>
<td>21.4 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B4</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_pretrained.pdparams">训练模型</a></td>
<td>83.57</td>
<td>5.47 / 1.10</td>
<td>14.42 / 9.89</td>
<td>70.4 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B6</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_pretrained.pdparams">训练模型</a></td>
<td>86.30</td>
<td>12.25 / 3.76</td>
<td>62.29 / 62.29</td>
<td>268.4 M</td>
</tr>
<tr>
<td>PP-LCNet_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pretrained.pdparams">训练模型</a></td>
<td>71.32</td>
<td>2.35 / 0.47</td>
<td>4.03 / 1.35</td>
<td>10.5 M</td>
</tr>
<tr>
<td>ResNet50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_pretrained.pdparams">训练模型</a></td>
<td>76.5</td>
<td>6.44 / 1.16</td>
<td>15.04 / 11.63</td>
<td>90.8 M</td>
</tr>
<tr>
<td>SwinTransformer_tiny_patch4_window7_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_tiny_patch4_window7_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams">训练模型</a></td>
<td>81.10</td>
<td>6.66 / 2.15</td>
<td>60.45 / 60.45</td>
<td>100.1 M</td>
</tr>
</table>

> ❗ 以上列出的是图像分类模块重点支持的<b>9个核心模型</b>，该模块总共支持<b>80个模型</b>，完整的模型列表如下：
<details><summary> 👉模型列表详情</summary>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top1 Acc(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>CLIP_vit_base_patch16_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_224_pretrained.pdparams">训练模型</a></td>
<td>85.36</td>
<td>12.84 / 2.82</td>
<td>60.52 / 60.52</td>
<td>306.5 M</td>
<td rowspan="2">CLIP是一种基于视觉和语言相关联的图像分类模型，采用对比学习和预训练方法，实现无监督或弱监督的图像分类，尤其适用于大规模数据集。模型通过将图像和文本映射到同一表示空间，学习到通用特征，具有良好的泛化能力和解释性。其在较好的训练误差，在很多下游任务都有较好的表现。</td>
</tr>
<tr>
<td>CLIP_vit_large_patch14_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_large_patch14_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_large_patch14_224_pretrained.pdparams">训练模型</a></td>
<td>88.1</td>
<td>51.72 / 11.13</td>
<td>238.07 / 238.07</td>
<td>1.04 G</td>
</tr>
<tr>
<td>ConvNeXt_base_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_base_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_base_224_pretrained.pdparams">训练模型</a></td>
<td>83.84</td>
<td>13.18 / 12.14</td>
<td>128.39 / 81.78</td>
<td>313.9 M</td>
<td rowspan="6">ConvNeXt系列模型是Meta在2022年提出的基于CNN架构的模型。该系列模型是在ResNet的基础上，通过借鉴SwinTransformer的优点设计，包括训练策略和网络结构的优化思路，从而改进的纯CNN架构网络，探索了卷积神经网络的性能上限。ConvNeXt系列模型具备卷积神经网络的诸多优点，包括推理效率高和易于迁移到下游任务等。</td>
</tr>
<tr>
<td>ConvNeXt_base_384</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_base_384_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_base_384_pretrained.pdparams">训练模型</a></td>
<td>84.90</td>
<td>32.15 / 30.52</td>
<td>279.36 / 220.35</td>
<td>313.9 M</td>
</tr>
<tr>
<td>ConvNeXt_large_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_large_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_large_224_pretrained.pdparams">训练模型</a></td>
<td>84.26</td>
<td>26.51 / 7.21</td>
<td>213.32 / 157.22</td>
<td>700.7 M</td>
</tr>
<tr>
<td>ConvNeXt_large_384</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_large_384_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_large_384_pretrained.pdparams">训练模型</a></td>
<td>85.27</td>
<td>67.07 / 65.26</td>
<td>494.04 / 438.97</td>
<td>700.7 M</td>
</tr>
<tr>
<td>ConvNeXt_small</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_small_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_small_pretrained.pdparams">训练模型</a></td>
<td>83.13</td>
<td>9.05 / 8.21</td>
<td>97.94 / 55.29</td>
<td>178.0 M</td>
</tr>
<tr>
<td>ConvNeXt_tiny</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ConvNeXt_tiny_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ConvNeXt_tiny_pretrained.pdparams">训练模型</a></td>
<td>82.03</td>
<td>5.12 / 2.06</td>
<td>63.96 / 29.77</td>
<td>104.1 M</td>
</tr>
<tr>
<td>FasterNet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-L_pretrained.pdparams">训练模型</a></td>
<td>83.5</td>
<td>15.67 / 3.10</td>
<td>52.24 / 52.24</td>
<td>357.1 M</td>
<td rowspan="6">FasterNet是一个旨在提高运行速度的神经网络，改进点主要如下：<br/>
1.重新审视了流行的运算符，发现低FLOPS主要来自于运算频繁的内存访问，特别是深度卷积；<br/>
2.提出了部分卷积(PConv)，通过减少冗余计算和内存访问来更高效地提取图像特征；<br/>
3.基于PConv推出了FasterNet系列模型，这是一种新的设计方案，在不影响模型任务性能的情况下，在各种设备上实现了显著更高的运行速度。</td>
</tr>
<tr>
<td>FasterNet-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-M_pretrained.pdparams">训练模型</a></td>
<td>83.0</td>
<td>9.72 / 2.30</td>
<td>35.29 / 35.29</td>
<td>204.6 M</td>
</tr>
<tr>
<td>FasterNet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-S_pretrained.pdparams">训练模型</a></td>
<td>81.3</td>
<td>5.46 / 1.27</td>
<td>20.46 / 18.03</td>
<td>119.3 M</td>
</tr>
<tr>
<td>FasterNet-T0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T0_pretrained.pdparams">训练模型</a></td>
<td>71.9</td>
<td>4.18 / 0.60</td>
<td>6.34 / 3.44</td>
<td>15.1 M</td>
</tr>
<tr>
<td>FasterNet-T1</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T1_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T1_pretrained.pdparams">训练模型</a></td>
<td>75.9</td>
<td>4.24 / 0.64</td>
<td>9.57 / 5.20</td>
<td>29.2 M</td>
</tr>
<tr>
<td>FasterNet-T2</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FasterNet-T2_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterNet-T2_pretrained.pdparams">训练模型</a></td>
<td>79.1</td>
<td>3.87 / 0.78</td>
<td>11.14 / 9.98</td>
<td>57.4 M</td>
</tr>
<tr>
<td>MobileNetV1_x0_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_5_pretrained.pdparams">训练模型</a></td>
<td>63.5</td>
<td>1.39 / 0.28</td>
<td>2.74 / 1.02</td>
<td>4.8 M</td>
<td rowspan="4">MobileNetV1是Google于2017年发布的用于移动设备或嵌入式设备中的网络。该网络将传统的卷积操作拆解成深度可分离卷积，即Depthwise卷积和Pointwise卷积的组合。相比传统的卷积网络，该组合可以大大节省参数量和计算量。同时该网络可以用于图像分类等其他视觉任务中。</td>
</tr>
<tr>
<td>MobileNetV1_x0_25</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_25_pretrained.pdparams">训练模型</a></td>
<td>51.4</td>
<td>1.32 / 0.30</td>
<td>2.04 / 0.58</td>
<td>1.8 M</td>
</tr>
<tr>
<td>MobileNetV1_x0_75</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x0_75_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x0_75_pretrained.pdparams">训练模型</a></td>
<td>68.8</td>
<td>1.75 / 0.33</td>
<td>3.41 / 1.57</td>
<td>9.3 M</td>
</tr>
<tr>
<td>MobileNetV1_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV1_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV1_x1_0_pretrained.pdparams">训练模型</a></td>
<td>71.0</td>
<td>1.89 / 0.34</td>
<td>4.01 / 2.17</td>
<td>15.2 M</td>
</tr>
<tr>
<td>MobileNetV2_x0_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x0_5_pretrained.pdparams">训练模型</a></td>
<td>65.0</td>
<td>3.17 / 0.48</td>
<td>4.52 / 1.35</td>
<td>7.1 M</td>
<td rowspan="5">MobileNetV2是Google继MobileNetV1提出的一种轻量级网络。相比MobileNetV1，MobileNetV2提出了Linear bottlenecks与Inverted residual block作为网络基本结构，通过大量地堆叠这些基本模块，构成了MobileNetV2的网络结构。最后，在FLOPs只有MobileNetV1的一半的情况下取得了更高的分类精度。</td>
</tr>
<tr>
<td>MobileNetV2_x0_25</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x0_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x0_25_pretrained.pdparams">训练模型</a></td>
<td>53.2</td>
<td>2.80 / 0.46</td>
<td>3.92 / 0.98</td>
<td>5.5 M</td>
</tr>
<tr>
<td>MobileNetV2_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x1_0_pretrained.pdparams">训练模型</a></td>
<td>72.2</td>
<td>3.57 / 0.49</td>
<td>5.63 / 2.51</td>
<td>12.6 M</td>
</tr>
<tr>
<td>MobileNetV2_x1_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x1_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x1_5_pretrained.pdparams">训练模型</a></td>
<td>74.1</td>
<td>3.58 / 0.62</td>
<td>8.02 / 4.49</td>
<td>25.0 M</td>
</tr>
<tr>
<td>MobileNetV2_x2_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV2_x2_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV2_x2_0_pretrained.pdparams">训练模型</a></td>
<td>75.2</td>
<td>3.56 / 0.74</td>
<td>10.24 / 6.83</td>
<td>41.2 M</td>
</tr>
<tr>
<td>MobileNetV3_large_x0_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_5_pretrained.pdparams">训练模型</a></td>
<td>69.2</td>
<td>3.79 / 0.62</td>
<td>6.76 / 1.61</td>
<td>9.6 M</td>
<td rowspan="10">MobileNetV3是Google于2019年提出的一种基于NAS的轻量级网络。为了进一步提升效果，将relu和sigmoid激活函数分别替换为hard_swish与hard_sigmoid激活函数，同时引入了一些专门为减少网络计算量的改进策略。</td>
</tr>
<tr>
<td>MobileNetV3_large_x0_35</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_35_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_35_pretrained.pdparams">训练模型</a></td>
<td>64.3</td>
<td>3.70 / 0.60</td>
<td>5.54 / 1.41</td>
<td>7.5 M</td>
</tr>
<tr>
<td>MobileNetV3_large_x0_75</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x0_75_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x0_75_pretrained.pdparams">训练模型</a></td>
<td>73.1</td>
<td>4.82 / 0.66</td>
<td>7.45 / 2.00</td>
<td>14.0 M</td>
</tr>
<tr>
<td>MobileNetV3_large_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x1_0_pretrained.pdparams">训练模型</a></td>
<td>75.3</td>
<td>4.86 / 0.68</td>
<td>6.88 / 2.61</td>
<td>19.5 M</td>
</tr>
<tr>
<td>MobileNetV3_large_x1_25</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_large_x1_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_large_x1_25_pretrained.pdparams">训练模型</a></td>
<td>76.4</td>
<td>5.08 / 0.71</td>
<td>7.37 / 3.58</td>
<td>26.5 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x0_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_5_pretrained.pdparams">训练模型</a></td>
<td>59.2</td>
<td>3.41 / 0.57</td>
<td>5.60 / 1.14</td>
<td>6.8 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x0_35</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_35_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_35_pretrained.pdparams">训练模型</a></td>
<td>53.0</td>
<td>3.49 / 0.60</td>
<td>4.63 / 1.07</td>
<td>6.0 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x0_75</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x0_75_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x0_75_pretrained.pdparams">训练模型</a></td>
<td>66.0</td>
<td>3.49 / 0.60</td>
<td>5.19 / 1.28</td>
<td>8.5 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_0_pretrained.pdparams">训练模型</a></td>
<td>68.2</td>
<td>3.76 / 0.53</td>
<td>5.11 / 1.43</td>
<td>10.5 M</td>
</tr>
<tr>
<td>MobileNetV3_small_x1_25</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV3_small_x1_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV3_small_x1_25_pretrained.pdparams">训练模型</a></td>
<td>70.7</td>
<td>4.23 / 0.58</td>
<td>6.48 / 1.68</td>
<td>13.0 M</td>
</tr>
<tr>
<td>MobileNetV4_conv_large</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_large_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_large_pretrained.pdparams">训练模型</a></td>
<td>83.4</td>
<td>8.33 / 2.24</td>
<td>33.56 / 23.70</td>
<td>125.2 M</td>
<td rowspan="5">MobileNetV4是专为移动设备设计的高效架构。其核心在于引入了UIB（Universal Inverted Bottleneck）模块，这是一种统一且灵活的结构，融合了IB（Inverted Bottleneck）、ConvNeXt、FFN（Feed Forward Network）以及最新的ExtraDW（Extra Depthwise）模块。与UIB同时推出的还有Mobile MQA，这是种专为移动加速器定制的注意力块，可实现高达39%的显著加速。此外，MobileNetV4引入了一种新的神经架构搜索（Neural Architecture Search, NAS）方案，以提升搜索的有效性。</td>
</tr>
<tr>
<td>MobileNetV4_conv_medium</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_medium_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_medium_pretrained.pdparams">训练模型</a></td>
<td>79.9</td>
<td>6.81 / 0.92</td>
<td>12.47 / 6.27</td>
<td>37.6 M</td>
</tr>
<tr>
<td>MobileNetV4_conv_small</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_conv_small_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_conv_small_pretrained.pdparams">训练模型</a></td>
<td>74.6</td>
<td>3.25 / 0.46</td>
<td>4.42 / 1.54</td>
<td>14.7 M</td>
</tr>
<tr>
<td>MobileNetV4_hybrid_large</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_hybrid_large_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_hybrid_large_pretrained.pdparams">训练模型</a></td>
<td>83.8</td>
<td>12.27 / 4.18</td>
<td>58.64 / 58.64</td>
<td>145.1 M</td>
</tr>
<tr>
<td>MobileNetV4_hybrid_medium</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileNetV4_hybrid_medium_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileNetV4_hybrid_medium_pretrained.pdparams">训练模型</a></td>
<td>80.5</td>
<td>12.08 / 1.34</td>
<td>24.69 / 8.10</td>
<td>42.9 M</td>
</tr>
<tr>
<td>PP-HGNet_base</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_base_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_base_pretrained.pdparams">训练模型</a></td>
<td>85.0</td>
<td>14.10 / 4.19</td>
<td>68.92 / 68.92</td>
<td>249.4 M</td>
<td rowspan="3">PP-HGNet（High Performance GPU Net）是百度飞桨视觉团队研发的适用于GPU平台的高性能骨干网络。该网络结合VOVNet的基础出使用了可学习的下采样层（LDS Layer），融合了ResNet_vd、PPHGNet等模型的优点。该模型在GPU平台上与其他SOTA模型在相同的速度下有着更高的精度。在同等速度下，该模型高于ResNet34-0模型3.8个百分点，高于ResNet50-0模型2.4个百分点，在使用相同的SLSD条款下，最终超越了ResNet50-D模型4.7个百分点。与此同时，在相同精度下，其推理速度也远超主流VisionTransformer的推理速度。</td>
</tr>
<tr>
<td>PP-HGNet_small</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_small_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_small_pretrained.pdparams">训练模型</a></td>
<td>81.51</td>
<td>5.12 / 1.73</td>
<td>25.01 / 25.01</td>
<td>86.5 M</td>
</tr>
<tr>
<td>PP-HGNet_tiny</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNet_tiny_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNet_tiny_pretrained.pdparams">训练模型</a></td>
<td>79.83</td>
<td>3.28 / 1.29</td>
<td>16.40 / 15.97</td>
<td>52.4 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_pretrained.pdparams">训练模型</a></td>
<td>77.77</td>
<td>3.83 / 0.57</td>
<td>9.95 / 2.37</td>
<td>21.4 M</td>
<td rowspan="7">PP-HGNetV2（High Performance GPU Network V2）是百度飞桨视觉团队的PP-HGNet的下一代版本，其在PP-HGNet的基础上，做了进一步优化和改进，其在NVIDIA发布的“Accuracy-Latency Balance”做到了极致，精度大幅超越了其他同样推理速度的模型。在每种标签分类，考标场景中，都有较强的表现。</td>
</tr>
<tr>
<td>PP-HGNetV2-B1</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B1_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B1_pretrained.pdparams">训练模型</a></td>
<td>79.18</td>
<td>3.87 / 0.62</td>
<td>8.77 / 3.79</td>
<td>22.6 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B2</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B2_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B2_pretrained.pdparams">训练模型</a></td>
<td>81.74</td>
<td>5.73 / 0.86</td>
<td>15.11 / 7.05</td>
<td>39.9 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B3</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B3_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B3_pretrained.pdparams">训练模型</a></td>
<td>82.98</td>
<td>6.26 / 1.01</td>
<td>18.47 / 10.34</td>
<td>57.9 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B4</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_pretrained.pdparams">训练模型</a></td>
<td>83.57</td>
<td>5.47 / 1.10</td>
<td>14.42 / 9.89</td>
<td>70.4 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B5_pretrained.pdparams">训练模型</a></td>
<td>84.75</td>
<td>10.24 / 1.96</td>
<td>29.71 / 29.71</td>
<td>140.8 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B6</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_pretrained.pdparams">训练模型</a></td>
<td>86.30</td>
<td>12.25 / 3.76</td>
<td>62.29 / 62.29</td>
<td>268.4 M</td>
</tr>
<tr>
<td>PP-LCNet_x0_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_5_pretrained.pdparams">训练模型</a></td>
<td>63.14</td>
<td>2.28 / 0.42</td>
<td>2.86 / 0.83</td>
<td>6.7 M</td>
<td rowspan="8">PP-LCNet是百度飞桨视觉团队自研的轻量级骨干网络，它能在不增加推理时间的前提下，进一步提升模型的性能，大幅超越其他轻量级SOTA模型。</td>
</tr>
<tr>
<td>PP-LCNet_x0_25</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_25_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_pretrained.pdparams">训练模型</a></td>
<td>51.86</td>
<td>1.89 / 0.45</td>
<td>2.49 / 0.68</td>
<td>5.5 M</td>
</tr>
<tr>
<td>PP-LCNet_x0_35</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_35_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_35_pretrained.pdparams">训练模型</a></td>
<td>58.09</td>
<td>1.94 / 0.41</td>
<td>2.73 / 0.77</td>
<td>5.9 M</td>
</tr>
<tr>
<td>PP-LCNet_x0_75</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_75_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_75_pretrained.pdparams">训练模型</a></td>
<td>68.18</td>
<td>2.30 / 0.41</td>
<td>2.95 / 1.07</td>
<td>8.4 M</td>
</tr>
<tr>
<td>PP-LCNet_x1_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_pretrained.pdparams">训练模型</a></td>
<td>71.32</td>
<td>2.35 / 0.47</td>
<td>4.03 / 1.35</td>
<td>10.5 M</td>
</tr>
<tr>
<td>PP-LCNet_x1_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_5_pretrained.pdparams">训练模型</a></td>
<td>73.71</td>
<td>2.33 / 0.53</td>
<td>4.17 / 2.29</td>
<td>16.0 M</td>
</tr>
<tr>
<td>PP-LCNet_x2_0</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x2_0_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x2_0_pretrained.pdparams">训练模型</a></td>
<td>75.18</td>
<td>2.40 / 0.51</td>
<td>5.37 / 3.46</td>
<td>23.2 M</td>
</tr>
<tr>
<td>PP-LCNet_x2_5</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x2_5_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x2_5_pretrained.pdparams">训练模型</a></td>
<td>76.60</td>
<td>2.36 / 0.61</td>
<td>6.29 / 5.05</td>
<td>32.1 M</td>
</tr>
<tr>
<td>PP-LCNetV2_base</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_base_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_base_pretrained.pdparams">训练模型</a></td>
<td>77.05</td>
<td>3.33 / 0.55</td>
<td>6.86 / 3.77</td>
<td>23.7 M</td>
<td rowspan="3">PP-LCNetV2 图像分类模型是百度飞桨视觉团队自研的 PP-LCNet 的下一代版本，其在 PP-LCNet 的基础上，做了进一步优化和改进，主要使用重参数化策略组合了不同大小卷积核的深度卷积，并优化了点卷积、Shortcut等。在不使用额外数据的前提下，PPLCNetV2_base 模型在图像分类 ImageNet 数据集上能够取得超过 77% 的 Top1 Acc，同时在 Intel CPU 平台的推理时间在 4.4 ms 以下</td>
</tr>
<tr>
<td>PP-LCNetV2_large </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_large_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_large_pretrained.pdparams">训练模型</a></td>
<td>78.51</td>
<td>4.37 / 0.71</td>
<td>9.43 / 8.07</td>
<td>37.3 M</td>
</tr>
<tr>
<td>PP-LCNetV2_small</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNetV2_small_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNetV2_small_pretrained.pdparams">训练模型</a></td>
<td>73.97</td>
<td>2.53 / 0.41</td>
<td>5.14 / 1.98</td>
<td>14.6 M</td>
</tr>
<tr>
<td>ResNet18_vd</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet18_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet18_vd_pretrained.pdparams">训练模型</a></td>
<td>72.3</td>
<td>2.47 / 0.61</td>
<td>6.97 / 5.15</td>
<td>41.5 M</td>
<td rowspan="11">ResNet 系列模型是在 2015 年提出的，一举在 ILSVRC2015 比赛中取得冠军，top5 错误率为 3.57%。该网络创新性的提出了残差结构，通过堆叠多个残差结构从而构建了 ResNet 网络。实验表明使用残差块可以有效地提升收敛速度和精度。</td>
</tr>
<tr>
<td>ResNet18 </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet18_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet18_pretrained.pdparams">训练模型</a></td>
<td>71.0</td>
<td>2.35 / 0.67</td>
<td>6.35 / 4.61</td>
<td>41.5 M</td>
</tr>
<tr>
<td>ResNet34_vd</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet34_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet34_vd_pretrained.pdparams">训练模型</a></td>
<td>76.0</td>
<td>4.01 / 1.03</td>
<td>11.99 / 9.86</td>
<td>77.3 M</td>
</tr>
<tr>
<td>ResNet34</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet34_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet34_pretrained.pdparams">训练模型</a></td>
<td>74.6</td>
<td>3.99 / 1.02</td>
<td>12.42 / 9.81</td>
<td>77.3 M</td>
</tr>
<tr>
<td>ResNet50_vd</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_vd_pretrained.pdparams">训练模型</a></td>
<td>79.1</td>
<td>6.04 / 1.16</td>
<td>16.08 / 12.07</td>
<td>90.8 M</td>
</tr>
<tr>
<td>ResNet50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_pretrained.pdparams">训练模型</a></td>
<td>76.5</td>
<td>6.44 / 1.16</td>
<td>15.04 / 11.63</td>
<td>90.8 M</td>
</tr>
<tr>
<td>ResNet101_vd</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet101_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet101_vd_pretrained.pdparams">训练模型</a></td>
<td>80.2</td>
<td>11.16 / 2.07</td>
<td>32.14 / 32.14</td>
<td>158.4 M</td>
</tr>
<tr>
<td>ResNet101</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet101_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet101_pretrained.pdparams">训练模型</a></td>
<td>77.6</td>
<td>10.91 / 2.06</td>
<td>31.14 / 22.93</td>
<td>158.4 M</td>
</tr>
<tr>
<td>ResNet152_vd</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet152_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet152_vd_pretrained.pdparams">训练模型</a></td>
<td>80.6</td>
<td>15.96 / 2.99</td>
<td>49.33 / 49.33</td>
<td>214.3 M</td>
</tr>
<tr>
<td>ResNet152</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet152_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet152_pretrained.pdparams">训练模型</a></td>
<td>78.3</td>
<td>15.61 / 2.90</td>
<td>47.33 / 36.60</td>
<td>214.2 M</td>
</tr>
<tr>
<td>ResNet200_vd</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet200_vd_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet200_vd_pretrained.pdparams">训练模型</a></td>
<td>80.9</td>
<td>24.20 / 3.69</td>
<td>62.62 / 62.62</td>
<td>266.0 M</td>
</tr>
<tr>
<td>StarNet-S1</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S1_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S1_pretrained.pdparams">训练模型</a></td>
<td>73.6</td>
<td>6.33 / 1.98</td>
<td>7.56 / 3.26</td>
<td>11.2 M</td>
<td rowspan="4">StarNet 聚焦于研究网络设计中“星操作”（即元素级乘法）的未开发潜力。揭示星操作能够将输入映射到高维、非线性特征空间的能力，这一过程类似于核技巧，但无需扩大网络规模。因此进一步提出了 StarNet，一个简单而强大的原型网络，该网络在紧凑的网络结构和有限的计算资源下，展现出了卓越的性能和低延迟。</td>
</tr>
<tr>
<td>StarNet-S2 </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S2_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S2_pretrained.pdparams">训练模型</a></td>
<td>74.8</td>
<td>4.49 / 1.55</td>
<td>7.38 / 3.38</td>
<td>14.3 M</td>
</tr>
<tr>
<td>StarNet-S3</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S3_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S3_pretrained.pdparams">训练模型</a></td>
<td>77.0</td>
<td>6.70 / 1.62</td>
<td>11.05 / 4.76</td>
<td>22.2 M</td>
</tr>
<tr>
<td>StarNet-S4</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/StarNet-S4_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/StarNet-S4_pretrained.pdparams">训练模型</a></td>
<td>79.0</td>
<td>8.50 / 2.86</td>
<td>15.40 / 6.76</td>
<td>28.9 M</td>
</tr>
<tr>
<td>SwinTransformer_base_patch4_window7_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_base_patch4_window7_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_base_patch4_window7_224_pretrained.pdparams">训练模型</a></td>
<td>83.37</td>
<td>14.29 / 5.13</td>
<td>130.89 / 130.89</td>
<td>310.5 M</td>
<td rowspan="6">SwinTransformer 是一种新的视觉 Transformer 网络，可以用作计算机视觉领域的通用骨干网路。SwinTransformer 由移动窗口（shifted windows）表示的层次 Transformer 结构组成。移动窗口将自注意计算限制在非重叠的局部窗口上，同时允许跨窗口连接，从而提高了网络性能。</td>
</tr>
<tr>
<td>SwinTransformer_base_patch4_window12_384</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_base_patch4_window12_384_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_base_patch4_window12_384_pretrained.pdparams">训练模型</a></td>
<td>84.17</td>
<td>37.74 / 10.10</td>
<td>362.56 / 362.56</td>
<td>311.4 M</td>
</tr>
<tr>
<td>SwinTransformer_large_patch4_window7_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_large_patch4_window7_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_large_patch4_window7_224_pretrained.pdparams">训练模型</a></td>
<td>86.19</td>
<td>26.48 / 7.94</td>
<td>228.23 / 228.23</td>
<td>694.8 M</td>
</tr>
<tr>
<td>SwinTransformer_large_patch4_window12_384</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_large_patch4_window12_384_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_large_patch4_window12_384_pretrained.pdparams">训练模型</a></td>
<td>87.06</td>
<td>74.72 / 18.16</td>
<td>652.04 / 652.04</td>
<td>696.1 M</td>
</tr>
<tr>
<td>SwinTransformer_small_patch4_window7_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_small_patch4_window7_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_small_patch4_window7_224_pretrained.pdparams">训练模型</a></td>
<td>83.21</td>
<td>10.37 / 3.90</td>
<td>94.20 / 94.20</td>
<td>175.6 M</td>
</tr>
<tr>
<td>SwinTransformer_tiny_patch4_window7_224</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SwinTransformer_tiny_patch4_window7_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams">训练模型</a></td>
<td>81.10</td>
<td>6.66 / 2.15</td>
<td>60.45 / 60.45</td>
<td>100.1 M</td>
</tr>
</table>
<p><b>注：以上精度指标为 <a href="https://www.image-net.org/index.php">ImageNet-1k</a> 验证集 Top1 Acc。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p></details>

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)。

完成 wheel 包的安装后，几行代码即可完成图像分类模块的推理，可以任意切换该模块下的模型，您也可以将图像分类的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg)到本地。

```python
from paddlex import create_model
model = create_model(model_name="PP-LCNet_x1_0")
output = model.predict("general_image_classification_001.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

运行后，得到的结果为：
```bash
{'res': {'input_path': 'general_image_classification_001.jpg', 'page_index': None, 'class_ids': array([296, 279, 270, 537, 356], dtype=int32), 'scores': array([0.79155, 0.01738, 0.01428, 0.01301, 0.01222], dtype=float32), 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Arctic fox, white fox, Alopex lagopus', 'white wolf, Arctic wolf, Canis lupus tundrarum', 'dogsled, dog sled, dog sleigh', 'weasel']}}
```

运行结果参数含义如下：
- `input_path`：表示输入图片的路径。
- `class_ids`：表示预测结果的类别id。
- `scores`：表示预测结果的置信度。
- `label_names`：表示预测结果的类别名。

可视化图片如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/image_classification/general_image_classification_001_res.jpg"/>

相关方法、参数等说明如下：

* `create_model`实例化图像分类模型（此处以`PP-LCNet_x1_0`为例），具体说明如下：
<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>可选项</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>模型名称</td>
<td><code>str</code></td>
<td>无</td>
<td><code>无</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>模型存储路径</td>
<td><code>str</code></td>
<td>无</td>
<td>无</td>
</tr>
<tr>
<td><code>topk</code></td>
<td>预测结果的前<code>topk</code>值，如果不指定，将默认使用PaddleX官方模型配置</td>
<td><code>int</code></td>
<td>
<ul>
<li><b>int</b>，如 5 ，表示打印（返回）预测结果的前<code>5</code>个类别和对应的分类概率</li>
</ul>
</td>
<td>5</td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 调用图像分类模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input` 和 `batch_size`，具体说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>可选项</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型</td>
<td><code>Python Var</code>/<code>str</code>/<code>dict</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python变量</b>，如<code>numpy.ndarray</code>表示的图像数据</li>
<li><b>文件路径</b>，如图像文件的本地路径：<code>/root/data/img.jpg</code></li>
<li><b>URL链接</b>，如图像文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg">示例</a></li>
<li><b>本地目录</b>，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code></li>
<li><b>字典</b>，字典的<code>key</code>需与具体任务对应，如图像分类任务对应<code>\"img\"</code>，字典的<code>val</code>支持上述类型数据，例如：<code>{\"img\": \"/root/data1\"}</code></li>
<li><b>列表</b>，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code>，<code>[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]</code></li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>批大小</td>
<td><code>int</code></td>
<td>任意整数</td>
<td>1</td>
</tr>
<tr>
<td><code>topk</code></td>
<td>预测结果的前<code>topk</code>值；如果不指定，将默认使用 <code>creat_model</code> 指定的 <code>topk</code> 参数，如果 <code>creat_model</code> 也没有指定，则默认使用PaddleX官方模型配置</td>
<td><code>int</code></td>
<td>
<ul>
<li><b>int</b>，如 5 ，表示打印（返回）预测结果的前<code>5</code>个类别和对应的分类概率</li>
</ul>
</td>
<td>5</td>
</tr>
</table>

* 对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为`json`文件的操作:

<table>
<thead>
<tr>
<th>方法</th>
<th>方法说明</th>
<th>参数</th>
<th>参数类型</th>
<th>参数说明</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">打印结果到终端</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>是否对输出内容进行使用 <code>JSON</code> 缩进格式化</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">将结果保存为json格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>将结果保存为图像格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
</table>

* 此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">获取预测的<code>json</code>格式的结果</td>
</tr>
<tr>
<td rowspan="1"><code>img</code></td>
<td rowspan="1">获取格式为<code>dict</code>的可视化图像</td>
</tr>
</table>


关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的图像分类模型。在使用 PaddleX 开发图像分类模型之前，请务必安装 PaddleX 的 图像分类  [PaddleX本地安装教程](../../../installation/installation.md)中的二次开发部分。

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX图像分类任务模块数据标注教程](../../../data_annotations/cv_modules/image_classification.md)

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/cls_flowers_examples.tar -P ./dataset
tar -xf ./dataset/cls_flowers_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>
<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "dataset/label.txt",
    "num_classes": 102,
    "train_samples": 1020,
    "train_sample_paths": [
      "check_dataset/demo_img/image_01904.jpg",
      "check_dataset/demo_img/image_06940.jpg"
    ],
    "val_samples": 1020,
    "val_sample_paths": [
      "check_dataset/demo_img/image_01937.jpg",
      "check_dataset/demo_img/image_06958.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/cls_flowers_examples",
  "show_type": "image",
  "dataset_type": "ClsDataset"
}
</code></pre>
<p>上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.num_classes</code>：该数据集类别数为 102；</li>
<li><code>attributes.train_samples</code>：该数据集训练集样本数量为 1020；</li>
<li><code>attributes.val_samples</code>：该数据集验证集样本数量为 1020；</li>
<li><code>attributes.train_sample_paths</code>：该数据集训练集样本可视化图片相对路径列表；</li>
<li><code>attributes.val_sample_paths</code>：该数据集验证集样本可视化图片相对路径列表；</li>
</ul>
<p>另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/image_classification/01.png"/></p></details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过<b>修改配置文件</b>或是<b>追加超参数</b>的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>
<p><b>（1）数据集格式转换</b></p>
<p>图像分类暂不支持数据转换。</p>
<p><b>（2）数据集划分</b></p>
<p>数据集划分的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: 是否进行重新划分数据集，为 <code>True</code> 时进行数据集格式转换，默认为 <code>False</code>；</li>
<li><code>train_percent</code>: 如果重新划分数据集，则需要设置训练集的百分比，类型为 0-100 之间的任意整数，需要保证和 <code>val_percent</code> 值加和为100；</li>
</ul>
<p>例如，您想重新划分数据集为 训练集占比90%、验证集占比10%，则需将配置文件修改为：</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
</code></pre>
<p>随后执行命令：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
</code></pre>
<p>数据划分执行之后，原有标注文件会被在原路径下重命名为 <code>xxx.bak</code>。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处图像分类模型 PP-LCNet_x1_0 的训练为例：

```
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-LCNet_x1_0.yaml`,训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅模型对应任务模块的配置文件说明[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>
<ul>
<li>模型训练过程中，PaddleX 会自动保存模型权重文件，默认为<code>output</code>，如需指定保存路径，可通过配置文件中 <code>-o Global.output</code> 字段进行设置。</li>
<li>PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。</li>
<li>
<p>在完成模型训练后，所有产出保存在指定的输出目录（默认为<code>./output/</code>）下，通常有以下产出：</p>
</li>
<li>
<p><code>train_result.json</code>：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；</p>
</li>
<li><code>train.log</code>：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；</li>
<li><code>config.yaml</code>：训练配置文件，记录了本次训练的超参数的配置；</li>
<li><code>.pdparams</code>、<code>.pdema</code>、<code>.pdopt.pdstate</code>、<code>.pdiparams</code>、<code>.pdmodel</code>：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；</li>
</ul></details>

## <b>4.3 模型评估</b>
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：

```bash
python main.py -c  paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml  \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-LCNet_x1_0.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>
<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>。</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json，其记录了</code>评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 val.top1、val.top5；</p></details>

### <b>4.4 模型推理和模型集成</b>

在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理
通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg)到本地。

```bash
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="general_image_classification_001.jpg"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-LCNet_x1_0.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

图像分类模块可以集成的 PaddleX 产线有[通用图像分类产线](../../../pipeline_usage/tutorials/cv_pipelines/image_classification.md)，只需要替换模型路径即可完成相关产线的图像分类模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.<b>模块集成</b>

您产出的权重可以直接集成到图像分类模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。

您也可以利用 PaddleX 高性能推理插件来优化您模型的推理过程，进一步提升效率，详细的流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。
