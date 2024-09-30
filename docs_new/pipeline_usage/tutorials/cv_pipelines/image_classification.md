# 通用图像分类产线使用教程

## 1. 通用图像分类产线介绍
图像分类是一种将图像分配到预定义类别的技术。它广泛应用于物体识别、场景理解和自动标注等领域。图像分类可以识别各种物体，如动物、植物、交通标志等，并根据其特征将其归类。通过使用深度学习模型，图像分类能够自动提取图像特征并进行准确分类。

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_classification/01.png)

**通用图像分类产线中包含了图像分类模块，如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型**。

<details>
   <summary> 👉模型列表详情</summary>

<table>
  <tr>
    <th>模型</th>
    <th>Top1 Acc(%)</th>
    <th>GPU推理耗时 (ms)</th>
    <th>CPU推理耗时</th>
    <th>模型存储大小 (M)</th>
    <th>介绍</th>
  </tr>
  <tr>
    <td>CLIP_vit_base_patch16_224</td>
    <td>85.36</td>
    <td>13.1957</td>
    <td>285.493</td>
    <td >306.5 M</td>
    <td rowspan="2">CLIP是一种基于视觉和语言相关联的图像分类模型，采用对比学习和预训练方法，实现无监督或弱监督的图像分类，尤其适用于大规模数据集。模型通过将图像和文本映射到同一表示空间，学习到通用特征，具有良好的泛化能力和解释性。其在较好的训练误差，在很多下游任务都有较好的表现。</td>
  </tr>
  <tr>
    <td>CLIP_vit_large_patch14_224</td>
    <td>88.1</td>
    <td>51.1284</td>
    <td>1131.28</td>
    <td>1.04 G</td>
  </tr>
  <tr>
    <td>ConvNeXt_base_224</td>
    <td>83.84</td>
    <td>12.8473</td>
    <td>1513.87</td>
    <td>313.9 M</td>
    <td rowspan="6">ConvNeXt系列模型是Meta在2022年提出的基于CNN架构的模型。该系列模型是在ResNet的基础上，通过借鉴SwinTransformer的优点设计，包括训练策略和网络结构的优化思路，从而改进的纯CNN架构网络，探索了卷积神经网络的性能上限。ConvNeXt系列模型具备卷积神经网络的诸多优点，包括推理效率高和易于迁移到下游任务等。</td>
  </tr>
  <tr>
    <td>ConvNeXt_base_384</td>
    <td>84.90</td>
    <td>31.7607</td>
    <td>3967.05</td>
    <td>313.9 M</td>
  </tr>
  <tr>
    <td>ConvNeXt_large_224</td>
    <td>84.26</td>
    <td>26.8103</td>
    <td>2463.56</td>
    <td>700.7 M</td>
  </tr>
  <tr>
    <td>ConvNeXt_large_384</td>
    <td>85.27</td>
    <td>66.4058</td>
    <td>6598.92</td>
    <td>700.7 M</td>
  </tr>
  <tr>
    <td>ConvNeXt_small</td>
    <td>83.13</td>
    <td>9.74075</td>
    <td>1127.6</td>
    <td>178.0 M</td>
  </tr>
  <tr>
    <td>ConvNeXt_tiny</td>
    <td>82.03</td>
    <td>5.48923</td>
    <td>672.559</td>
    <td>104.1 M</td>
  </tr>
  <tr>
    <td>FasterNet-L</td>
    <td>83.5</td>
    <td>23.4415</td>
    <td>-</td>
    <td>357.1 M</td>
    <td rowspan="6">FasterNet是一个旨在提高运行速度的神经网络，改进点主要如下：<br>
      1.重新审视了流行的运算符，发现低FLOPS主要来自于运算频繁的内存访问，特别是深度卷积；<br>
      2.提出了部分卷积(PConv)，通过减少冗余计算和内存访问来更高效地提取图像特征；<br>
      3.基于PConv推出了FasterNet系列模型，这是一种新的设计方案，在不影响模型任务性能的情况下，在各种设备上实现了显著更高的运行速度。</td>
  </tr>
  <tr>
    <td>FasterNet-M</td>
    <td>83.0</td>
    <td>21.8936</td>
    <td>-</td>
    <td>204.6 M</td>
  </tr>
  <tr>
    <td>FasterNet-S</td>
    <td>81.3</td>
    <td>13.0409</td>
    <td>-</td>
    <td>119.3 M</td>
  </tr>
  <tr>
    <td>FasterNet-T0</td>
    <td>71.9</td>
    <td>12.2432</td>
    <td>-</td>
    <td>15.1 M</td>
  </tr>
  <tr>
    <td>FasterNet-T1</td>
    <td>75.9</td>
    <td>11.3562</td>
    <td>-</td>
    <td>29.2 M</td>
  </tr>
  <tr>
    <td>FasterNet-T2</td>
    <td>79.1</td>
    <td>10.703</td>
    <td>-</td>
    <td>57.4 M</td>
  </tr>
  <tr>
    <td>MobileNetV1_x0_5</td>
    <td>63.5</td>
    <td>1.86754</td>
    <td>7.48297</td>
    <td>4.8 M</td>
    <td rowspan="4">MobileNetV1是Google于2017年发布的用于移动设备或嵌入式设备中的网络。该网络将传统的卷积操作拆解成深度可分离卷积，即Depthwise卷积和Pointwise卷积的组合。相比传统的卷积网络，该组合可以大大节省参数量和计算量。同时该网络可以用于图像分类等其他视觉任务中。</td>
  </tr>
  <tr>
    <td>MobileNetV1_x0_25</td>
    <td>51.4</td>
    <td>1.83478</td>
    <td>4.83674</td>
    <td>1.8 M</td>
  </tr>
  <tr>
    <td>MobileNetV1_x0_75</td>
    <td>68.8</td>
    <td>2.57903</td>
    <td>10.6343</td>
    <td>9.3 M</td>
  </tr>
  <tr>
    <td>MobileNetV1_x1_0</td>
    <td>71.0</td>
    <td>2.78781</td>
    <td>13.98</td>
    <td>15.2 M</td>
  </tr>
  <tr>
    <td>MobileNetV2_x0_5</td>
    <td>65.0</td>
    <td>4.94234</td>
    <td>11.1629</td>
    <td>7.1 M</td>
    <td rowspan="5">MobileNetV2是Google继MobileNetV1提出的一种轻量级网络。相比MobileNetV1，MobileNetV2提出了Linear bottlenecks与Inverted residual block作为网络基本结构，通过大量地堆叠这些基本模块，构成了MobileNetV2的网络结构。最后，在FLOPs只有MobileNetV1的一半的情况下取得了更高的分类精度。</td>
  </tr>
  <tr>
    <td>MobileNetV2_x0_25</td>
    <td>53.2</td>
    <td>4.50856</td>
    <td>9.40991</td>
    <td>5.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV2_x1_0</td>
    <td>72.2</td>
    <td>6.12159</td>
    <td>16.0442</td>
    <td>12.6 M</td>
  </tr>
  <tr>
    <td>MobileNetV2_x1_5</td>
    <td>74.1</td>
    <td>6.28385</td>
    <td>22.5129</td>
    <td>25.0 M</td>
  </tr>
  <tr>
    <td>MobileNetV2_x2_0</td>
    <td>75.2</td>
    <td>6.12888</td>
    <td>30.8612</td>
    <td>41.2 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x0_5</td>
    <td>69.2</td>
    <td>6.31302</td>
    <td>14.5588</td>
    <td>9.6 M</td>
    <td rowspan="10">MobileNetV3是Google于2019年提出的一种基于NAS的轻量级网络。为了进一步提升效果，将relu和sigmoid激活函数分别替换为hard_swish与hard_sigmoid激活函数，同时引入了一些专门为减少网络计算量的改进策略。</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x0_35</td>
    <td>64.3</td>
    <td>5.76207</td>
    <td>13.9041</td>
    <td>7.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x0_75</td>
    <td>73.1</td>
    <td>8.41737</td>
    <td>16.9506</td>
    <td>14.0 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x1_0</td>
    <td>75.3</td>
    <td>8.64112</td>
    <td>19.1614</td>
    <td>19.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_large_x1_25</td>
    <td>76.4</td>
    <td>8.73358</td>
    <td>22.1296</td>
    <td>26.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x0_5</td>
    <td>59.2</td>
    <td>5.16721</td>
    <td>11.2688</td>
    <td>6.8 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x0_35</td>
    <td>53.0</td>
    <td>5.22053</td>
    <td>11.0055</td>
    <td>6.0 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x0_75</td>
    <td>66.0</td>
    <td>5.39831</td>
    <td>12.8313</td>
    <td>8.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x1_0</td>
    <td>68.2</td>
    <td>6.00993</td>
    <td>12.9598</td>
    <td>10.5 M</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x1_25</td>
    <td>70.7</td>
    <td>6.9589</td>
    <td>14.3995</td>
    <td>13.0 M</td>
  </tr>
  <tr>
    <td>MobileNetV4_conv_large</td>
    <td>83.4</td>
    <td>12.5485</td>
    <td>51.6453</td>
    <td>125.2 M</td>
    <td rowspan="5">MobileNetV4是专为移动设备设计的高效架构。其核心在于引入了UIB（Universal Inverted Bottleneck）模块，这是一种统一且灵活的结构，融合了IB（Inverted Bottleneck）、ConvNeXt、FFN（Feed Forward Network）以及最新的ExtraDW（Extra Depthwise）模块。与UIB同时推出的还有Mobile MQA，这是种专为移动加速器定制的注意力块，可实现高达39%的显著加速。此外，MobileNetV4引入了一种新的神经架构搜索（Neural Architecture Search, NAS）方案，以提升搜索的有效性。</td>
  </tr>
  <tr>
    <td>MobileNetV4_conv_medium</td>
    <td>79.9</td>
    <td>9.65509</td>
    <td>26.6157</td>
    <td>37.6 M</td>
  </tr>
  <tr>
    <td>MobileNetV4_conv_small</td>
    <td>74.6</td>
    <td>5.24172</td>
    <td>11.0893</td>
    <td>14.7 M</td>
  </tr>
  <tr>
    <td>MobileNetV4_hybrid_large</td>
    <td>83.8</td>
    <td>20.0726</td>
    <td>213.769</td>
    <td>145.1 M</td>
  </tr>
  <tr>
    <td>MobileNetV4_hybrid_medium</td>
    <td>80.5</td>
    <td>19.7543</td>
    <td>62.2624</td>
    <td>42.9 M</td>
  </tr>
  <tr>
    <td>PP-HGNet_base</td>
    <td>85.0</td>
    <td>14.2969</td>
    <td>327.114</td>
    <td>249.4 M</td>
    <td rowspan="3">PP-HGNet（High Performance GPU Net）是百度飞桨视觉团队研发的适用于GPU平台的高性能骨干网络。该网络结合VOVNet的基础出使用了可学习的下采样层（LDS Layer），融合了ResNet_vd、PPHGNet等模型的优点。该模型在GPU平台上与其他SOTA模型在相同的速度下有着更高的精度。在同等速度下，该模型高于ResNet34-0模型3.8个百分点，高于ResNet50-0模型2.4个百分点，在使用相同的SLSD条款下，最终超越了ResNet50-D模型4.7个百分点。与此同时，在相同精度下，其推理速度也远超主流VisionTransformer的推理速度。</td>
  </tr>
  <tr>
    <td>PP-HGNet_small</td>
    <td>81.51</td>
    <td>5.50661</td>
    <td>119.041</td>
    <td>86.5 M</td>
  </tr>
  <tr>
    <td>PP-HGNet_tiny</td>
    <td>79.83</td>
    <td>5.22006</td>
    <td>69.396</td>
    <td>52.4 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B0</td>
    <td>77.77</td>
    <td>6.53694</td>
    <td>23.352</td>
    <td>21.4 M</td>
    <td rowspan="7">PP-HGNetV2（High Performance GPU Network V2）是百度飞桨视觉团队的PP-HGNet的下一代版本，其在PP-HGNet的基础上，做了进一步优化和改进，其在NVIDIA发布的“Accuracy-Latency Balance”做到了极致，精度大幅超越了其他同样推理速度的模型。在每种标签分类，考标场景中，都有较强的表现。</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B1</td>
    <td>79.18</td>
    <td>6.56034</td>
    <td>27.3099</td>
    <td>22.6 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B2</td>
    <td>81.74</td>
    <td>9.60494</td>
    <td>43.1219</td>
    <td>39.9 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B3</td>
    <td>82.98</td>
    <td>11.0042</td>
    <td>55.1367</td>
    <td>57.9 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B4</td>
    <td>83.57</td>
    <td>9.66407</td>
    <td>54.2462</td>
    <td>70.4 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B5</td>
    <td>84.75</td>
    <td>15.7091</td>
    <td>115.926</td>
    <td>140.8 M</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B6</td>
    <td>86.30</td>
    <td>21.226</td>
    <td>255.279</td>
    <td>268.4 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x0_5</td>
    <td>63.14</td>
    <td>3.67722</td>
    <td>6.66857</td>
    <td>6.7 M</td>
    <td rowspan="8">PP-LCNet是百度飞桨视觉团队自研的轻量级骨干网络，它能在不增加推理时间的前提下，进一步提升模型的性能，大幅超越其他轻量级SOTA模型。</td>
  </tr>
  <tr>
    <td>PP-LCNet_x0_25</td>
    <td>51.86</td>
    <td>2.65341</td>
    <td>5.81357</td>
    <td>5.5 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x0_35</td>
    <td>58.09</td>
    <td>2.7212</td>
    <td>6.28944</td>
    <td>5.9 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x0_75</td>
    <td>68.18</td>
    <td>3.91032</td>
    <td>8.06953</td>
    <td>8.4 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x1_0</td>
    <td>71.32</td>
    <td>3.84845</td>
    <td>9.23735</td>
    <td>10.5 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x1_5</td>
    <td>73.71</td>
    <td>3.97666</td>
    <td>12.3457</td>
    <td>16.0 M</td>
  </tr>
  <tr>
    <td>PP-LCNet_x2_0</td>
    <td>75.18</td>
    <td>4.07556</td>
    <td>16.2752</td>
    <td>23.2 M</td>
  </tr>
     <tr>
    <td>PP-LCNet_x2_5</td>
    <td>76.60</td>
    <td>4.06028</td>
    <td>21.5063</td>
    <td>32.1 M</td>
  </tr>
  <tr>

  <tr>
    <td>PP-LCNetV2_base</td>
    <td>77.05</td>
    <td>5.23428</td>
    <td>19.6005</td>
    <td>23.7 M</td>
    <td rowspan="3">PP-LCNetV2 图像分类模型是百度飞桨视觉团队自研的 PP-LCNet 的下一代版本，其在 PP-LCNet 的基础上，做了进一步优化和改进，主要使用重参数化策略组合了不同大小卷积核的深度卷积，并优化了点卷积、Shortcut等。在不使用额外数据的前提下，PPLCNetV2_base 模型在图像分类 ImageNet 数据集上能够取得超过 77% 的 Top1 Acc，同时在 Intel CPU 平台的推理时间在 4.4 ms 以下</td>
  </tr>
  <tr>
    <td>PP-LCNetV2_large </td>
    <td>78.51</td>
    <td>6.78335</td>
    <td>30.4378</td>
    <td>37.3 M</td>
  </tr>
  <tr>
    <td>PP-LCNetV2_small</td>
    <td>73.97</td>
    <td>3.89762</td>
    <td>13.0273</td>
    <td>14.6 M</td>
  </tr>
<tr>
<tr>
    <td>ResNet18_vd</td>
    <td>72.3</td>
    <td>3.53048</td>
    <td>31.3014</td>
    <td>41.5 M</td>
    <td rowspan="11">ResNet 系列模型是在 2015 年提出的，一举在 ILSVRC2015 比赛中取得冠军，top5 错误率为 3.57%。该网络创新性的提出了残差结构，通过堆叠多个残差结构从而构建了 ResNet 网络。实验表明使用残差块可以有效地提升收敛速度和精度。</td>
  </tr>
  <tr>
    <td>ResNet18 </td>
    <td>71.0</td>
    <td>2.4868</td>
    <td>27.4601</td>
    <td>41.5 M</td>
  </tr>
  <tr>
    <td>ResNet34_vd</td>
    <td>76.0</td>
    <td>5.60675</td>
    <td>56.0653</td>
    <td>77.3 M</td>
  </tr>
    <tr>
    <td>ResNet34</td>
    <td>74.6</td>
    <td>4.16902</td>
    <td>51.925</td>
    <td>77.3 M</td>
  </tr>
  <tr>
    <td>ResNet50_vd</td>
    <td>79.1</td>
    <td>10.1885</td>
    <td>68.446</td>
    <td>90.8 M</td>
  </tr>
    <tr>
    <td>ResNet50</td>
    <td>76.5</td>
    <td>9.62383</td>
    <td>64.8135</td>
    <td>90.8 M</td>
  </tr>
     <tr>
    <td>ResNet101_vd</td>
    <td>80.2</td>
    <td>20.0563</td>
    <td>124.85</td>
    <td>158.4 M</td>
  </tr>
     <tr>
    <td>ResNet101</td>
    <td>77.6</td>
    <td>19.2297</td>
    <td>121.006</td>
    <td>158.4 M</td>
  </tr>
  <tr>
    <td>ResNet152_vd</td>
    <td>80.6</td>
    <td>29.6439</td>
    <td>181.678</td>
    <td>214.3 M</td>
  </tr>
    <tr>
    <td>ResNet152</td>
    <td>78.3</td>
    <td>30.0461</td>
    <td>177.707</td>
    <td>214.2 M</td>
  </tr>
     <tr>
    <td>ResNet200_vd</td>
    <td>80.9</td>
    <td>39.1628</td>
    <td>235.185</td>
    <td>266.0 M</td>
  </tr>
<tr>
  <tr>
    <td>StarNet-S1</td>
    <td>73.6</td>
    <td>9.895</td>
    <td>23.0465</td>
    <td>11.2 M</td>
    <td rowspan="4">StarNet 聚焦于研究网络设计中“星操作”（即元素级乘法）的未开发潜力。揭示星操作能够将输入映射到高维、非线性特征空间的能力，这一过程类似于核技巧，但无需扩大网络规模。因此进一步提出了 StarNet，一个简单而强大的原型网络，该网络在紧凑的网络结构和有限的计算资源下，展现出了卓越的性能和低延迟。</td>
  </tr>
  <tr>
    <td>StarNet-S2 </td>
    <td>74.8</td>
    <td>7.91279</td>
    <td>21.9571</td>
    <td>14.3 M</td>
  </tr>
  <tr>
    <td>StarNet-S3</td>
    <td>77.0</td>
    <td>10.7531</td>
    <td>30.7656</td>
    <td>22.2 M</td>
  </tr>
    <tr>
    <td>StarNet-S4</td>
    <td>79.0</td>
    <td>15.2868</td>
    <td>43.2497</td>
    <td>28.9 M</td>
  </tr>
<tr>
  <tr>
    <td>SwinTransformer_base_patch4_window7_224</td>
    <td>83.37</td>
    <td>16.9848</td>
    <td>383.83</td>
    <td>310.5 M</td>
    <td rowspan="6">SwinTransformer 是一种新的视觉 Transformer 网络，可以用作计算机视觉领域的通用骨干网路。SwinTransformer 由移动窗口（shifted windows）表示的层次 Transformer 结构组成。移动窗口将自注意计算限制在非重叠的局部窗口上，同时允许跨窗口连接，从而提高了网络性能。</td>
  </tr>
  <tr>
    <td>SwinTransformer_base_patch4_window12_384</td>
    <td>84.17</td>
    <td>37.2855</td>
    <td>1178.63</td>
    <td>311.4 M</td>
  </tr>
  <tr>
    <td>SwinTransformer_large_patch4_window7_224</td>
    <td>86.19</td>
    <td>27.5498</td>
    <td>689.729</td>
    <td>694.8 M</td>
  </tr>
    <tr>
    <td>SwinTransformer_large_patch4_window12_384</td>
    <td>87.06</td>
    <td>74.1768</td>
    <td>2105.22</td>
    <td>696.1 M</td>
  </tr>
     <tr>
    <td>SwinTransformer_small_patch4_window7_224</td>
    <td>83.21</td>
    <td>16.3982</td>
    <td>285.56</td>
    <td>175.6 M</td>
  </tr>
       <tr>
    <td>SwinTransformer_tiny_patch4_window7_224</td>
    <td>81.10</td>
    <td>8.54846</td>
    <td>156.306</td>
    <td>100.1 M</td>
  </tr>
<tr>
</table>


**注：以上精度指标为 [ImageNet-1k](https://www.image-net.org/index.php) 验证集 Top1 Acc。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

</details>

## 2. 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验通用图像分类产线的效果，也可以在本地使用命令行或 Python 体验通用图像分类产线的效果。

### 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/100061/webUI)通用图像分类产线的效果，用官方提供的 demo 图片进行识别，例如：

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_classification/02.png)

如果您对产线运行的效果满意，可以直接对产线进行集成部署，如果不满意，您也可以利用私有数据**对产线中的模型进行在线微调**。

### 2.2 本地体验
在本地使用通用图像分类产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.2.1 命令行方式体验
一行命令即可快速体验图像分类产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg)，并将 `--input` 替换为本地路径，进行预测

```
paddlex --pipeline image_classification --input general_image_classification_001.jpg --device gpu:0
```
参数说明：

```
--pipeline：产线名称，此处为图像分类产线
--input：待处理的输入图片的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```

在执行上述 Python 脚本时，加载的是默认的图像分类产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

<details>
   <summary> 👉点击展开</summary>

```
paddlex --get_pipeline_config image_classification
```
执行后，图像分类产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为 `./my_path` ）：

```
paddlex --get_pipeline_config image_classification --config_save_path ./my_path
```

获取产线配置文件后，可将 `--pipeline` 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 `./image_classification.yaml`，只需执行：

```
paddlex --pipeline ./image_classification.yaml --input general_image_classification_001.jpg
```
其中，`--model`、`--device` 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。

</details>

运行后，得到的结果为：

```
{'img_path': './my_path/general_image_classification_001.jpg', 'class_ids': [296, 170, 356, 258, 248], 'scores': [0.62736, 0.03752, 0.03256, 0.0323, 0.03194], 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Irish wolfhound', 'weasel', 'Samoyed, Samoyede', 'Eskimo dog, husky']}
```
![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_classification/03.png)

可视化图片默认保存在 `output` 目录下，您也可以通过 `--save_path` 进行自定义。

#### 2.2.2 Python脚本方式集成
几行代码即可完成产线的快速推理，以通用图像分类产线为例：

```
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="image_classification")

output = pipeline.predict("general_image_classification_001.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

（1）实例化 `create_pipeline` 实例化产线对象：具体参数说明如下：

|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|`pipeline`|产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。|`str`|无|
|`device`|产线模型推理设备。支持：“gpu”，“cpu”。|`str`|`gpu`|
|`enable_hpi`|是否启用高性能推理，仅当该产线支持高性能推理时可用。|`bool`|`False`|

（2）调用图像分类产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

| 参数类型      | 参数说明                                                                                                  |
|---------------|-----------------------------------------------------------------------------------------------------------|
| Python Var    | 支持直接传入Python变量，如numpy.ndarray表示的图像数据。                                               |
| str         | 支持传入待预测数据文件路径，如图像文件的本地路径：`/root/data/img.jpg`。                                   |
| str           | 支持传入待预测数据文件URL，如图像文件的网络URL：[示例](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001)。|
| str           | 支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：`/root/data/`。                               |
| dict          | 支持传入字典类型，字典的key需与具体任务对应，如图像分类任务对应\"img\"，字典的val支持上述类型数据，例如：`{\"img\": \"/root/data1\"}`。|
| list          | 支持传入列表，列表元素需为上述类型数据，如`[numpy.ndarray, numpy.ndarray]，[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]`，`[\"/root/data1\", \"/root/data2\"]`，`[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]`。|

（3）调用`predict`方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以batch为单位对数据进行预测，因此预测结果为list形式表示的一组预测结果。

（4）对预测结果进行处理：每个样本的预测结果均为`dict`类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

| 方法         | 说明                        | 方法参数                                                                                               |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | 打印结果到终端              | `- format_json`：bool类型，是否对输出内容进行使用json缩进格式化，默认为True；<br>`- indent`：int类型，json格式化设置，仅当format_json为True时有效，默认为4；<br>`- ensure_ascii`：bool类型，json格式化设置，仅当format_json为True时有效，默认为False； |
| save_to_json | 将结果保存为json格式的文件   | `- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；<br>`- indent`：int类型，json格式化设置，默认为4；<br>`- ensure_ascii`：bool类型，json格式化设置，默认为False； |
| save_to_img  | 将结果保存为图像格式的文件  | `- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致； |

若您获取了配置文件，即可对图像分类产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 `./my_path/image_classification*.yaml` ，则只需执行：

```
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/image_classification.yaml")
output = pipeline.predict("general_image_classification_001.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 **高性能部署**：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能部署流程请参考[PaddleX高性能部署指南](../../../pipeline_deploy/high_performance_deploy.md)。

☁️ **服务化部署**：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/service_deploy.md)。

📱 **端侧部署**：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/lite_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果通用图像分类产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用**您自己拥有的特定领域或应用场景的数据**对现有模型进行进一步的**微调**，以提升通用图像分类产线的在您的场景中的识别效果。

### 4.1 模型微调
由于通用图像分类产线包含图像分类模块，如果模型产线的效果不及预期，那么您需要参考[图像分类模块开发教程](../../../module_usage/tutorials/cv_modules/image_classification.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/image_classification.md#四二次开发)章节，使用您的私有数据集对图像分类模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```
......
Pipeline:
  model: PP-LCNet_x1_0  #可修改为微调后模型的本地路径
  device: "gpu"
  batch_size: 1
......
```
随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，**仅需修改 `--device` 参数**即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行图像分类产线的推理，使用的命令为：

```
paddlex --pipeline image_classification --input general_image_classification_001.jpg --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需将 `--device` 修改为 npu 即可：

```
paddlex --pipeline image_classification --input general_image_classification_001.jpg --device npu:0
```
若您想在更多种类的硬件上使用通用图像分类产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/installation_other_devices.md)。
