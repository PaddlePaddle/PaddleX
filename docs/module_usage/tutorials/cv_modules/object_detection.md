---
comments: true
---

# 目标检测模块使用教程

## 一、概述
目标检测模块是计算机视觉系统中的关键组成部分，负责在图像或视频中定位和标记出包含特定目标的区域。该模块的性能直接影响到整个计算机视觉系统的准确性和效率。目标检测模块通常会输出目标区域的边界框（Bounding Boxes），这些边界框将作为输入传递给目标识别模块进行后续处理。

## 二、支持模型列表

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>PicoDet-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_pretrained.pdparams">训练模型</a></td>
<td>42.6</td>
<td>14.31 / 11.06</td>
<td>45.95 / 25.06</td>
<td>20.9</td>
<td rowspan="2">PP-PicoDet是一种全尺寸、棱视宽目标的轻量级目标检测算法，它考虑移动端设备运算量。与传统目标检测算法相比，PP-PicoDet具有更小的模型尺寸和更低的计算复杂度，并在保证检测精度的同时更高的速度和更低的延迟。</td>
</tr>
<tr>
<td>PicoDet-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_pretrained.pdparams">训练模型</a></td>
<td>29.1</td>
<td>9.15 / 3.26</td>
<td>16.06 / 4.04</td>
<td>4.4</td>
</tr>
<tr>
<td>PP-YOLOE_plus-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-YOLOE_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-L_pretrained.pdparams">训练模型</a></td>
<td>52.9</td>
<td>32.06 / 28.00</td>
<td>185.32 / 116.21</td>
<td>185.3</td>
<td rowspan="2">PP-YOLOE_plus 是一种是百度飞桨视觉团队自研的云边一体高精度模型PP-YOLOE迭代优化升级的版本，通过使用Objects365大规模数据集、优化预处理，大幅提升了模型端到端推理速度。</td>
</tr>
<tr>
<td>PP-YOLOE_plus-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-YOLOE_plus-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_pretrained.pdparams">训练模型</a></td>
<td>43.7</td>
<td>11.43 / 7.52</td>
<td>60.16 / 26.94</td>
<td>28.3</td>
</tr>
<tr>
<td>RT-DETR-H</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_pretrained.pdparams">训练模型</a></td>
<td>56.3</td>
<td>114.57 / 101.56</td>
<td>938.20 / 938.20</td>
<td>435.8</td>
<td rowspan="2">RT-DETR是第一个实时端到端目标检测器。该模型设计了一个高效的混合编码器，满足模型效果与吞吐率的双需求，高效处理多尺度特征，并提出了加速和优化的查询选择机制，以优化解码器查询的动态化。RT-DETR支持通过使用不同的解码器来实现灵活端到端推理速度。</td>
</tr>
<tr>
<td>RT-DETR-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_pretrained.pdparams">训练模型</a></td>
<td>53.0</td>
<td>34.76 / 27.60</td>
<td>495.39 / 247.68</td>
<td>113.7</td>
</tr>
</table>

> ❗ 以上列出的是目标检测模块重点支持的<b>6个核心模型</b>，该模块总共支持<b>37个模型</b>，完整的模型列表如下：

<details><summary> 👉模型列表详情</summary>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>Cascade-FasterRCNN-ResNet50-FPN</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/Cascade-FasterRCNN-ResNet50-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-FasterRCNN-ResNet50-FPN_pretrained.pdparams">训练模型</a></td>
<td>41.1</td>
<td>120.28 / 120.28</td>
<td>- / 6514.61</td>
<td>245.4</td>
<td rowspan="2">Cascade-FasterRCNN 是一种改进的Faster R-CNN目标检测模型，通过耦联多个检测器，利用不同IoU阈值优化检测结果，解决训练和预测阶段的mismatch问题，提高目标检测的准确性。</td>
</tr>
<tr>
<td>Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">训练模型</a></td>
<td>45.0</td>
<td>124.10 / 124.10</td>
<td>- / 6709.52</td>
<td>246.2</td>
</tr>
<tr>
<td>CenterNet-DLA-34</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CenterNet-DLA-34_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CenterNet-DLA-34_pretrained.pdparams">训练模型</a></td>
<td>37.6</td>
<td>67.19 / 67.19</td>
<td>6622.61 / 6622.61</td>
<td>75.4</td>
<td rowspan="2">CenterNet是一种anchor-free目标检测模型，把待检测物体的关键点视为单一点-即其边界框的中心点，并通过关键点进行回归。</td>
</tr>
<tr>
<td>CenterNet-ResNet50</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CenterNet-ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CenterNet-ResNet50_pretrained.pdparams">训练模型</a></td>
<td>38.9</td>
<td>216.06 / 216.06</td>
<td>2545.79 / 2545.79</td>
<td>319.7</td>
</tr>
<tr>
<td>DETR-R50</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/DETR-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DETR-R50_pretrained.pdparams">训练模型</a></td>
<td>42.3</td>
<td>58.80 / 26.90</td>
<td>370.96 / 208.77</td>
<td>159.3</td>
<td>DETR 是Facebook提出的一种transformer目标检测模型，该模型在不需要预定义的先验框anchor和NMS的后处理策略的情况下，就可以实现端到端的目标检测。</td>
</tr>
<tr>
<td>FasterRCNN-ResNet34-FPN</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterRCNN-ResNet34-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet34-FPN_pretrained.pdparams">训练模型</a></td>
<td>37.8</td>
<td>76.90 / 76.90</td>
<td>- / 4136.79</td>
<td>137.5</td>
<td rowspan="9">Faster R-CNN是典型的two-stage目标检测模型，即先生成区域建议（Region Proposal），然后在生成的Region Proposal上做分类和回归。相较于前代R-CNN和Fast R-CNN，Faster R-CNN的改进主要在于区域建议方面，使用区域建议网络（Region Proposal Network, RPN）提供区域建议，以取代传统选择性搜索。RPN是卷积神经网络，并与检测网络共享图像的卷积特征，减少了区域建议的计算开销。</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-FPN</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterRCNN-ResNet50-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-FPN_pretrained.pdparams">训练模型</a></td>
<td>38.4</td>
<td>95.48 / 95.48</td>
<td>- / 3693.90</td>
<td>148.1</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-vd-FPN</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterRCNN-ResNet50-vd-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-vd-FPN_pretrained.pdparams">训练模型</a></td>
<td>39.5</td>
<td>98.03 / 98.03</td>
<td>- / 4278.36</td>
<td>148.1</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterRCNN-ResNet50-vd-SSLDv2-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50-vd-SSLDv2-FPN_pretrained.pdparams">训练模型</a></td>
<td>41.4</td>
<td>99.23 / 99.23</td>
<td>- / 4415.68</td>
<td>148.1</td>
</tr>
<tr>
<td>FasterRCNN-ResNet50</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterRCNN-ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet50_pretrained.pdparams">训练模型</a></td>
<td>36.7</td>
<td>129.10 / 129.10</td>
<td>- / 3868.44</td>
<td>120.2</td>
</tr>
<tr>
<td>FasterRCNN-ResNet101-FPN</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterRCNN-ResNet101-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet101-FPN_pretrained.pdparams">训练模型</a></td>
<td>41.4</td>
<td>131.48 / 131.48</td>
<td>- / 4380.00</td>
<td>216.3</td>
</tr>
<tr>
<td>FasterRCNN-ResNet101</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterRCNN-ResNet101_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNet101_pretrained.pdparams">训练模型</a></td>
<td>39.0</td>
<td>216.71 / 216.71</td>
<td>- / 5376.45</td>
<td>188.1</td>
</tr>
<tr>
<td>FasterRCNN-ResNeXt101-vd-FPN</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterRCNN-ResNeXt101-vd-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-ResNeXt101-vd-FPN_pretrained.pdparams">训练模型</a></td>
<td>43.4</td>
<td>234.38 / 234.38</td>
<td>- / 6154.61</td>
<td>360.6</td>
</tr>
<tr>
<td>FasterRCNN-Swin-Tiny-FPN</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FasterRCNN-Swin-Tiny-FPN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FasterRCNN-Swin-Tiny-FPN_pretrained.pdparams">训练模型</a></td>
<td>42.6</td>
<td>65.92 / 65.92</td>
<td>- / 2468.98</td>
<td>159.8</td>
</tr>
<tr>
<td>FCOS-ResNet50</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/FCOS-ResNet50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FCOS-ResNet50_pretrained.pdparams">训练模型</a></td>
<td>39.6</td>
<td>101.02 / 34.42</td>
<td>752.15 / 752.15</td>
<td>124.2</td>
<td>FCOS是一种密集预测的anchor-free目标检测模型，使用RetinaNet的骨架，直接在feature map上回归目标物体的长宽，并预测物体的类别以及centerness（feature map上像素点离物体中心的偏移程度），centerness最终会作为权重来调整物体得分。</td>
</tr>
<tr>
<td>PicoDet-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_pretrained.pdparams">训练模型</a></td>
<td>42.6</td>
<td>14.31 / 11.06</td>
<td>45.95 / 25.06</td>
<td>20.9</td>
<td rowspan="4">PP-PicoDet是一种全尺寸、棱视宽目标的轻量级目标检测算法，它考虑移动端设备运算量。与传统目标检测算法相比，PP-PicoDet具有更小的模型尺寸和更低的计算复杂度，并在保证检测精度的同时更高的速度和更低的延迟。</td>
</tr>
<tr>
<td>PicoDet-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-M_pretrained.pdparams">训练模型</a></td>
<td>37.5</td>
<td>10.48 / 5.00</td>
<td>22.88 / 9.03</td>
<td>16.8</td>
</tr>
<tr>
<td>PicoDet-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_pretrained.pdparams">训练模型</a></td>
<td>29.1</td>
<td>9.15 / 3.26</td>
<td>16.06 / 4.04</td>
<td>4.4</td>
</tr>
<tr>
<td>PicoDet-XS</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-XS_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-XS_pretrained.pdparams">训练模型</a></td>
<td>26.2</td>
<td>9.54 / 3.52</td>
<td>17.96 / 5.38</td>
<td>5.7</td>
</tr>
<tr>
<td>PP-YOLOE_plus-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-YOLOE_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-L_pretrained.pdparams">训练模型</a></td>
<td>52.9</td>
<td>32.06 / 28.00</td>
<td>185.32 / 116.21</td>
<td>185.3</td>
<td rowspan="4">PP-YOLOE_plus 是一种是百度飞桨视觉团队自研的云边一体高精度模型PP-YOLOE迭代优化升级的版本，通过使用Objects365大规模数据集、优化预处理，大幅提升了模型端到端推理速度。</td>
</tr>
<tr>
<td>PP-YOLOE_plus-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-YOLOE_plus-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-M_pretrained.pdparams">训练模型</a></td>
<td>49.8</td>
<td>18.37 / 15.04</td>
<td>108.77 / 63.48</td>
<td>82.3</td>
</tr>
<tr>
<td>PP-YOLOE_plus-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-YOLOE_plus-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_pretrained.pdparams">训练模型</a></td>
<td>43.7</td>
<td>11.43 / 7.52</td>
<td>60.16 / 26.94</td>
<td>28.3</td>
</tr>
<tr>
<td>PP-YOLOE_plus-X</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-YOLOE_plus-X_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-X_pretrained.pdparams">训练模型</a></td>
<td>54.7</td>
<td>56.28 / 50.60</td>
<td>292.08 / 212.24</td>
<td>349.4</td>
</tr>
<tr>
<td>RT-DETR-H</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_pretrained.pdparams">训练模型</a></td>
<td>56.3</td>
<td>114.57 / 101.56</td>
<td>938.20 / 938.20</td>
<td>435.8</td>
<td rowspan="5">RT-DETR是第一个实时端到端目标检测器。该模型设计了一个高效的混合编码器，满足模型效果与吞吐率的双需求，高效处理多尺度特征，并提出了加速和优化的查询选择机制，以优化解码器查询的动态化。RT-DETR支持通过使用不同的解码器来实现灵活端到端推理速度。</td>
</tr>
<tr>
<td>RT-DETR-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_pretrained.pdparams">训练模型</a></td>
<td>53.0</td>
<td>34.76 / 27.60</td>
<td>495.39 / 247.68</td>
<td>113.7</td>
</tr>
<tr>
<td>RT-DETR-R18</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-R18_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-R18_pretrained.pdparams">训练模型</a></td>
<td>46.5</td>
<td>19.11 / 14.82</td>
<td>263.13 / 143.05</td>
<td>70.7</td>
</tr>
<tr>
<td>RT-DETR-R50</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-R50_pretrained.pdparams">训练模型</a></td>
<td>53.1</td>
<td>41.11 / 10.12</td>
<td>536.20 / 482.86</td>
<td>149.1</td>
</tr>
<tr>
<td>RT-DETR-X</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-X_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-X_pretrained.pdparams">训练模型</a></td>
<td>54.8</td>
<td>61.91 / 51.41</td>
<td>639.79 / 639.79</td>
<td>232.9</td>
</tr>
<tr>
<td>YOLOv3-DarkNet53</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/YOLOv3-DarkNet53_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-DarkNet53_pretrained.pdparams">训练模型</a></td>
<td>39.1</td>
<td>39.62 / 35.54</td>
<td>166.57 / 136.34</td>
<td>219.7</td>
<td rowspan="3">YOLOv3是一种实时的端到端目标检测器。它使用一个独特的单个卷积神经网络，将目标检测问题分解为一个回归问题，从而实现实时的检测。该模型采用了多个尺度的检测，提高了不同尺度目标物体的检测性能。</td>
</tr>
<tr>
<td>YOLOv3-MobileNetV3</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/YOLOv3-MobileNetV3_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-MobileNetV3_pretrained.pdparams">训练模型</a></td>
<td>31.4</td>
<td>16.54 / 6.21</td>
<td>64.37 / 45.55</td>
<td>83.8</td>
</tr>
<tr>
<td>YOLOv3-ResNet50_vd_DCN</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/YOLOv3-ResNet50_vd_DCN_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOv3-ResNet50_vd_DCN_pretrained.pdparams">训练模型</a></td>
<td>40.6</td>
<td>31.64 / 26.72</td>
<td>226.75 / 226.75</td>
<td>163.0</td>
</tr>
<tr>
<td>YOLOX-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/YOLOX-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-L_pretrained.pdparams">训练模型</a></td>
<td>50.1</td>
<td>49.68 / 45.03</td>
<td>232.52 / 156.24</td>
<td>192.5</td>
<td rowspan="6">YOLOX模型以YOLOv3作为目标检测网络的框架，通过设计Decoupled Head、Data Aug、Anchor Free以及SimOTA组件，显著提升了模型在各种复杂场景下的检测性能。</td>
</tr>
<tr>
<td>YOLOX-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/YOLOX-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-M_pretrained.pdparams">训练模型</a></td>
<td>46.9</td>
<td>43.46 / 29.52</td>
<td>147.64 / 80.06</td>
<td>90.0</td>
</tr>
<tr>
<td>YOLOX-N</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/YOLOX-N_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-N_pretrained.pdparams">训练模型</a></td>
<td>26.1</td>
<td>42.94 / 17.79</td>
<td>64.15 / 7.19</td>
<td>3.4</td>
</tr>
<tr>
<td>YOLOX-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/YOLOX-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-S_pretrained.pdparams">训练模型</a></td>
<td>40.4</td>
<td>46.53 / 29.34</td>
<td>98.37 / 35.02</td>
<td>32.0</td>
</tr>
<tr>
<td>YOLOX-T</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/YOLOX-T_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-T_pretrained.pdparams">训练模型</a></td>
<td>32.9</td>
<td>31.81 / 18.91</td>
<td>55.34 / 11.63</td>
<td>18.1</td>
</tr>
<tr>
<td>YOLOX-X</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/YOLOX-X_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOLOX-X_pretrained.pdparams">训练模型</a></td>
<td>51.8</td>
<td>84.06 / 77.28</td>
<td>390.38 / 272.88</td>
<td>351.5</td>
</tr>
<tr>
<td>Co-Deformable-DETR-R50</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/Co-Deformable-DETR-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Co-Deformable-DETR-R50_pretrained.pdparams">训练模型</a></td>
<td>49.7</td>
<td>259.62 / 259.62</td>
<td>32413.76 / 32413.76</td>
<td>184</td>
<td rowspan="4">Co-DETR是一种先进的端到端目标检测器。它基于DETR架构，通过引入协同混合分配训练策略，将目标检测任务中的传统一对多标签分配与一对一匹配相结合，从而显著提高了检测性能和训练效率</td>
</tr>
<tr>
<td>Co-Deformable-DETR-Swin-T</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/Co-Deformable-DETR-Swin-T_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Co-Deformable-DETR-Swin-T_pretrained.pdparams">训练模型</a></td>
<td>48.0 (640x640 输入尺寸下)</td>
<td>120.17 / 120.17</td>
<td>- / 15620.29</td>
<td>187</td>
</tr>
<tr>
<td>Co-DINO-R50</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/Co-DINO-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Co-DINO-R50_pretrained.pdparams">训练模型</a></td>
<td>52.0</td>
<td>1123.23 / 1123.23</td>
<td>- / -</td>
<td>186</td>
</tr>
<tr>
<td>Co-DINO-Swin-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/Co-DINO-Swin-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Co-DINO-Swin-L_pretrained.pdparams">训练模型</a></td>
<td>55.9 (640x640 输入尺寸下)</td>
<td>- / -</td>
<td>- / -</td>
<td>840</td>
</tr>
</table>

<strong>测试环境说明:</strong>

  <ul>
      <li><b>性能测试环境</b>
          <ul>
            <li><strong>测试数据集：</strong><a href="https://cocodataset.org/#home">COCO2017</a>验证集。 </li>
              <li><strong>硬件配置：</strong>
                  <ul>
                      <li>GPU：NVIDIA Tesla T4</li>
                      <li>CPU：Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>其他环境：Ubuntu 20.04 / CUDA 11.8 / cuDNN 8.9 / TensorRT 8.6.1.6</li>
                  </ul>
              </li>
          </ul>
      </li>
      <li><b>推理模式说明</b></li>
  </ul>

<table border="1">
    <thead>
        <tr>
            <th>模式</th>
            <th>GPU配置</th>
            <th>CPU配置</th>
            <th>加速技术组合</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>常规模式</td>
            <td>FP32精度 / 无TRT加速</td>
            <td>FP32精度 / 8线程</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>高性能模式</td>
            <td>选择先验精度类型和加速策略的最优组合</td>
            <td>FP32精度 / 8线程</td>
            <td>选择先验最优后端（Paddle/OpenVINO/TRT等）</td>
        </tr>
    </tbody>
</table>

</details>

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成 wheel 包的安装后，几行代码即可完成目标检测模块的推理，可以任意切换该模块下的模型，您也可以将目标检测的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png)到本地。

```python
from paddlex import create_model

model = create_model(model_name="PicoDet-S")

output = model.predict(
  "general_object_detection_002.png",
  batch_size=1,
  threshold=0.5
)

for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

运行后，得到的结果为:

```bash
{'res': {'input_path': 'general_object_detection_002.png', 'page_index': None, 'boxes': [{'cls_id': 49, 'label': 'orange', 'score': 0.8188614249229431, 'coordinate': [661.3518, 93.05823, 870.75903, 305.93713]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7745078206062317, 'coordinate': [76.80911, 274.74905, 330.5422, 520.0428]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7271787524223328, 'coordinate': [285.32645, 94.3175, 469.73645, 297.40344]}, {'cls_id': 46, 'label': 'banana', 'score': 0.5576589703559875, 'coordinate': [310.8041, 361.43625, 685.1869, 712.59155]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5490103363990784, 'coordinate': [764.6252, 285.76096, 924.8153, 440.92892]}, {'cls_id': 47, 'label': 'apple', 'score': 0.515821635723114, 'coordinate': [853.9831, 169.41423, 987.803, 303.58615]}, {'cls_id': 60, 'label': 'dining table', 'score': 0.514293372631073, 'coordinate': [0.53089714, 0.32445717, 1072.9534, 720]}, {'cls_id': 47, 'label': 'apple', 'score': 0.510750949382782, 'coordinate': [57.368027, 23.455347, 213.39601, 176.45612]}]}}
```

参数含义如下：
- `input_path`：表示输入图像的路径
- `page_index`：如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`
- `boxes`：预测的目标框信息，一个字典列表。每个字典代表一个检出的目标，包含以下信息：
  - `cls_id`：类别ID，一个整数
  - `label`：类别标签，一个字符串
  - `score`：目标框置信度，一个浮点数
  - `coordinate`：目标框坐标，一个浮点数列表，格式为<code>[xmin, ymin, xmax, ymax]</code>

可视化图像如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/obj_det/general_object_detection_002_res.png"/>

相关方法、参数等说明如下：

* `create_model`实例化目标检测模型（此处以`PicoDet-S`为例），具体说明如下：
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
<td>无</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>模型存储路径</td>
<td><code>str</code></td>
<td>无</td>
<td>无</td>
</tr>
<tr>
<td><code>device</code></td>
<td>模型推理设备</td>
<td><code>str</code></td>
<td>支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>img_size</code></td>
<td>输入图像大小；如果不指定，将默认使用PaddleX官方模型配置</td>
<td><code>int/list/None</code></td>
<td>
<ul>
<li><b>int</b>, 如 640 , 表示将输入图像resize到640x640大小</li>
<li><b>列表</b>, 如 [512, 640] , 表示将输入图像resize到宽为640，高为512大小</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>用于过滤掉低置信度预测结果的阈值；如果不指定，将默认使用PaddleX官方模型配置</td>
<td><code>float/dict/None</code></td>
<td>
<ul>
<li><b>float</b>，如 0.2， 表示过滤掉所有阈值小于0.2的目标框</li>
<li><b>字典</b>，字典的key为<b>int</b>类型，代表<code>cls_id</code>，val为<b>float</b>类型阈值。如 <code>{0: 0.45, 2: 0.48, 7: 0.4}</code>，表示对cls_id为0的类别应用阈值0.45、cls_id为1的类别应用阈值0.48、cls_id为7的类别应用阈值0.4</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理插件</td>
<td><code>bool</code></td>
<td>无</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>高性能推理配置</td>
<td><code>dict</code> | <code>None</code></td>
<td>无</td>
<td><code>None</code></td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 调用目标检测模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input`、`batch_size`和`threshold`，具体说明如下：

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
<td>待预测数据，支持多种输入类型，必填</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
<li><b>List</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
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
<td><code>threshold</code></td>
<td>用于过滤掉低置信度预测结果的阈值；如果不指定，将默认使用 <code>creat_model</code> 指定的 <code>threshold</code> 参数，如果 <code>creat_model</code> 也没有指定，则默认使用PaddleX官方模型配置</td>
<td><code>float/dict/None</code></td>
<td>
<ul>
<li><b>float</b>，如 0.2， 表示过滤掉所有阈值小于0.2的目标框</li>
<li><b>字典</b>，字典的key为<b>int</b>类型，代表<code>cls_id</code>，val为<b>float</b>类型阈值。如 <code>{0: 0.45, 2: 0.48, 7: 0.4}</code>，表示对cls_id为0的类别应用阈值0.45、cls_id为1的类别应用阈值0.48、cls_id为7的类别应用阈值0.4</li>
</ul>
</td>
<td><code>None</code></td>
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
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的目标检测模型。在使用 PaddleX 开发目标检测模型之前，请务必安装 PaddleX的目标检测相关模型训练插件，安装过程可以参考 [PaddleX本地安装教程](../../../installation/installation.md)

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX目标检测任务模块数据标注教程](../../../data_annotations/cv_modules/object_detection.md)。

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_coco_examples.tar -P ./dataset
tar -xf ./dataset/det_coco_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>
<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 4,
    "train_samples": 701,
    "train_sample_paths": [
      "check_dataset/demo_img/road839.png",
      "check_dataset/demo_img/road363.png",
      "check_dataset/demo_img/road148.png",
      "check_dataset/demo_img/road237.png",
      "check_dataset/demo_img/road733.png",
      "check_dataset/demo_img/road861.png",
      "check_dataset/demo_img/road762.png",
      "check_dataset/demo_img/road515.png",
      "check_dataset/demo_img/road754.png",
      "check_dataset/demo_img/road173.png"
    ],
    "val_samples": 176,
    "val_sample_paths": [
      "check_dataset/demo_img/road218.png",
      "check_dataset/demo_img/road681.png",
      "check_dataset/demo_img/road138.png",
      "check_dataset/demo_img/road544.png",
      "check_dataset/demo_img/road596.png",
      "check_dataset/demo_img/road857.png",
      "check_dataset/demo_img/road203.png",
      "check_dataset/demo_img/road589.png",
      "check_dataset/demo_img/road655.png",
      "check_dataset/demo_img/road245.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "det_coco_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}
</code></pre>
<p>上述校验结果中，check_pass 为 true 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.num_classes</code>：该数据集类别数为 4；</li>
<li><code>attributes.train_samples</code>：该数据集训练集样本数量为 704；</li>
<li><code>attributes.val_samples</code>：该数据集验证集样本数量为 176；</li>
<li><code>attributes.train_sample_paths</code>：该数据集训练集样本可视化图片相对路径列表；</li>
<li><code>attributes.val_sample_paths</code>：该数据集验证集样本可视化图片相对路径列表；</li>
</ul>
<p>另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/obj_det/01.png"/></p></details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过<b>修改配置文件</b>或是<b>追加超参数</b>的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>
<p><b>（1）数据集格式转换</b></p>
<p>目标检测支持 <code>VOC</code>、<code>LabelMe</code> 格式的数据集转换为 <code>COCO</code> 格式。</p>
<p>数据集校验相关的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: 是否进行数据集格式转换，目标检测支持 <code>VOC</code>、<code>LabelMe</code> 格式的数据集转换为 <code>COCO</code> 格式，默认为 <code>False</code>;</li>
<li><code>src_dataset_type</code>: 如果进行数据集格式转换，则需设置源数据集格式，默认为 <code>null</code>，可选值为 <code>VOC</code>、<code>LabelMe</code> 和 <code>VOCWithUnlabeled</code>、<code>LabelMeWithUnlabeled</code> ；
例如，您想转换 <code>LabelMe</code> 格式的数据集为 <code>COCO</code> 格式，以下面的<code>LabelMe</code> 格式的数据集为例，则需要修改配置如下：</li>
</ul>
<pre><code class="language-bash">cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_labelme_examples.tar -P ./dataset
tar -xf ./dataset/det_labelme_examples.tar -C ./dataset/
</code></pre>
<pre><code class="language-bash">......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: LabelMe
  ......
</code></pre>
<p>随后执行命令：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_labelme_examples
</code></pre>
<p>当然，以上参数同样支持通过追加命令行参数的方式进行设置，以 <code>LabelMe</code> 格式的数据集为例：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_labelme_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
</code></pre>
<p><b>（2）数据集划分</b></p>
<p>数据集划分的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: 是否进行重新划分数据集，为 <code>True</code> 时进行数据集格式转换，默认为 <code>False</code>；</li>
<li><code>train_percent</code>: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 <code>val_percent</code> 值加和为100；</li>
<li><code>val_percent</code>: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 <code>train_percent</code> 值加和为100；
例如，您想重新划分数据集为 训练集占比90%、验证集占比10%，则需将配置文件修改为：</li>
</ul>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples
</code></pre>
<p>数据划分执行之后，原有标注文件会被在原路径下重命名为 <code>xxx.bak</code>。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处目标检测模型 `PicoDet-S` 的训练为例：

```bash
python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/det_coco_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PicoDet-S.yaml`，训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
* 其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅模型对应任务模块的配置文件说明[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)
* 新特性：Paddle 3.0 版本支持了 CINN 神经网络编译器，在使用 GPU 设备训练时，不同模型有不同程度的训练加速效果。在 PaddleX 中训练模型时，可通过指定参数 `-o Train.dy2st=True` 开启。

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
<li><code>.pdparams</code>、<code>.pdema</code>、<code>.pdopt.pdstate</code>、<code>.pdiparams</code>、<code>.json</code>：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；</li>
<li>【注意】：Paddle 3.0.0 对于静态图网络结构信息的存储格式，由protobuf（原<code>.pdmodel</code>后缀文件）升级为json（现<code>.json</code>后缀文件），以兼容PIR体系，并获得更好的灵活性与扩展性。</li>
</ul></details>

## <b>4.3 模型评估</b>
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：

```bash
python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/det_coco_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PicoDet-S.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>
<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>。</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json，其记录了</code>评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 AP；</p></details>

### <b>4.4 模型推理和模型集成</b>
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理

* 通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png)到本地。
```bash
python main.py -c paddlex/configs/modules/object_detection/PicoDet-S.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="general_object_detection_002.png"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PicoDet-S.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

目标检测模块可以集成的PaddleX产线有[通用目标检测产线](../../../pipeline_usage/tutorials/cv_pipelines/object_detection.md)，只需要替换模型路径即可完成相关产线的目标检测模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.<b>模块集成</b>

您产出的权重可以直接集成到目标检测模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。

您也可以利用 PaddleX 高性能推理插件来优化您模型的推理过程，进一步提升效率，详细的流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。
