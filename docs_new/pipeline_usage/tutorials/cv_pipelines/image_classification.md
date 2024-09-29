# 通用图像分类产线使用教程

## 通用图像分类产线介绍
图像分类是一种将图像分配到预定义类别的技术。它广泛应用于物体识别、场景理解和自动标注等领域。图像分类可以识别各种物体，如动物、植物、交通标志等，并根据其特征将其归类。通过使用深度学习模型，图像分类能够自动提取图像特征并进行准确分类。通用图像分类产线用于解决图像分类任务，对给定的图像。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=d118f459d72a4321a61c94c831f0f667&docGuid=u_VtdOCxJUF8GK "")
**通用图像分类产线中包含了图像分类模块，如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型**。

<details>
   <summary> 👉模型列表详情</summary>

|模型名称|Top1 Acc（%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|介绍|
|-|-|-|-|-|-|
|CLIP_vit_base_patch16_224|85.36|13.1957|285.493|306.5 M|CLIP 一种基于视觉和语言相互关联的图像分类模型，采用对比学习和预训练方法，实现无监督或弱监督的图像分类，尤其适用于大规模数据集。模型通过将图像和文本映射到同一表示空间，学习到通用特征，具有良好的泛化能力和可解释性。其有较好的预训练权重，在很多下游任务都有较好的表现。|
|CLIP_vit_large_patch14_224|88.1|51.1284|1131.28|1.04 G||
|ConvNeXt_base_224|83.84|12.8473|1513.87|313.9 M|ConvNeXt 系列模型是 Meta 在 2022 年提出的基于 CNN 架构的模型，该系列模型是在 ResNet 的基础上，通过借鉴 SwinTransformer 的优秀设计，包括训练策略和网络结构的优化思想，从而设计出的纯 CNN 架构网络，探索了卷积神经网络的性能上限。同时因为基于卷积神经网络实现，因此该系列模型具有卷积神经网络的诸多优点，包括推理效率高和易于迁移到下游任务等。|
|ConvNeXt_base_384|84.90|31.7607|3967.05|313.9 M||
|ConvNeXt_large_224|84.26|26.8103|2463.56|700.7 M||
|ConvNeXt_large_384|85.27|66.4058|6598.92|700.7 M||
|ConvNeXt_small|83.13|9.74075|1127.6|178.0 M||
|ConvNeXt_tiny|82.03|5.48923|672.559|101.4 M||
|FasterNet-L|83.5|23.4415|-|357.1 M|FasterNet 是一个旨在提高运行速度的神经网络，改进点主要如下： 重新审视了流行的运算符，发现低 FLOPS 主要是由于运算符频繁的内存访问，特别是深度卷积； 提出了部分卷积 (PConv)，通过减少冗余计算和内存访问来更高效地提取空间特征； 基于 PConv 推出了 FasterNet 系列模型，这是一种新的神经网络家族，在不影响各种视觉任务准确性的情况下，在各种设备上实现了显著更高的运行速度。|
|FasterNet-M|83.0|21.8936|-|204.6 M||
|FasterNet-S|81.3|13.0409|-|119.3 M||
|FasterNet-T0|71.9|12.2432|-|15.1 M||
|FasterNet-T1|75.9|11.3562|-|29.2 M||
|FasterNet-T2|79.1|10.703|-|57.4 M||
|MobileNetV1_x0_5|63.5|1.86754|7.48297|4.8 M|MobileNetV1 是 Google 于 2017 年发布的用于移动设备或嵌入式设备中的网络。该网络将传统的卷积操作替换深度可分离卷积，即 Depthwise 卷积和 Pointwise 卷积的组合，相比传统的卷积操作，该组合可以大大节省参数量和计算量。与此同时，MobileNetV1 也可以用于目标检测、图像分割等其他视觉任务中。|
|MobileNetV1_x0_25|51.4|1.83478|4.83674|1.8 M||
|MobileNetV1_x0_75|68.8|2.57903|10.6343|9.3 M||
|MobileNetV1_x1_0|71.0|2.78781|13.98|15.2 M||
|MobileNetV2_x0_5|65.0|4.94234|11.1629|7.1 M|MobileNetV2 是 Google 继 MobileNetV1 提出的一种轻量级网络。相比 MobileNetV1，MobileNetV2 提出了 Linear bottlenecks 与 Inverted residual block 作为网络基本结构，通过大量地堆叠这些基本模块，构成了 MobileNetV2 的网络结构。最终，在 FLOPs 只有 MobileNetV1 的一半的情况下取得了更高的分类精度。|
|MobileNetV2_x0_25|53.2|4.50856|9.40991|5.5 M||
|MobileNetV2_x1_0|72.2|6.12159|16.0442|12.6 M||
|MobileNetV2_x1_5|74.1|6.28385|22.5129|25.0 M||
|MobileNetV2_x2_0|75.2|6.12888|30.8612|41.2 M||
|MobileNetV3_large_x0_5|69.2|6.31302|14.5588|9.6 M|MobileNetV3 是 Google 于 2019 年提出的一种基于 NAS 的新的轻量级网络，为了进一步提升效果，将 relu 和 sigmoid 激活函数分别替换为 hard_swish 与 hard_sigmoid 激活函数，同时引入了一些专门减小网络计算量的改进策略。|
|MobileNetV3_large_x0_35|64.3|5.76207|13.9041|7.5 M||
|MobileNetV3_large_x0_75|73.1|8.41737|16.9506|14.0 M||
|MobileNetV3_large_x1_0|75.3|8.64112|19.1614|19.5 M||
|MobileNetV3_large_x1_25|76.4|8.73358|22.1296|26.5 M||
|MobileNetV3_small_x0_5|59.2|5.16721|11.2688|6.8 M||
|MobileNetV3_small_x0_35|53.0|5.22053|11.0055|6.0 M||
|MobileNetV3_small_x0_75|66.0|5.39831|12.8313|8.5 M||
|MobileNetV3_small_x1_0|68.2|6.00993|12.9598|10.5 M||
|MobileNetV3_small_x1_25|70.7|6.9589|14.3995|13.0 M||
|MobileNetV4_conv_large|83.4|12.5485|51.6453|125.2 M|MobileNetV4 是专为移动设备设计的普遍高效的架构。其核心在于引入了 UIB（Universal Inverted Bottleneck）搜索块，这是一种统一且灵活的结构，融合了 IB（Inverted Bottleneck）、ConvNext、FFN（Feed Forward Network）以及新颖的 ExtraDW（Extra Depthwise）模块。与 UIB 同时推出的还有 Mobile MQA，这是一种专为移动加速器定制的注意力块，可实现高达 39% 的显著加速。此外，还引入了一种优化的神经架构搜索（Neural Architecture Search，NAS）方案，以提升搜索的有效性。|
|MobileNetV4_conv_medium|79.9|9.65509|26.6157|37.6 M||
|MobileNetV4_conv_small|74.6|5.24172|11.0893|14.7 M||
|MobileNetV4_hybrid_large|83.8|20.0726|213.769|145.1 M||
|MobileNetV4_hybrid_medium|80.5|19.7543|62.2624|42.9 M||
|PP-HGNet_base|85.0|14.2969|327.114|249.4 M|PP-HGNet(High Performance GPU Net) 是百度飞桨视觉团队自研的更适用于 GPU 平台的高性能骨干网络，该网络在 VOVNet 的基础上使用了可学习的下采样层（LDS Layer），融合了 ResNet_vd、PPHGNet 等模型的优点，该模型在 GPU 平台上与其他 SOTA 模型在相同的速度下有着更高的精度。在同等速度下，该模型高于 ResNet34-D 模型 3.8 个百分点，高于 ResNet50-D 模型 2.4 个百分点，在使用百度自研 SSLD 蒸馏策略后，超越 ResNet50-D 模型 4.7 个百分点。与此同时，在相同精度下，其推理速度也远超主流 VisionTransformer 的推理速度。该模型是PaddleX非常推荐的服务端模型。|
|PP-HGNet_small|81.51|5.50661|119.041|86.5 M||
|PP-HGNet_tiny|79.83|5.22006|69.396|52.4 M||
|PP-HGNetV2-B0|77.77|6.53694|23.352|21.4 M|PP-HGNetV2(High Performance GPU Network V2) 是百度飞桨视觉团队自研的 PP-HGNet 的下一代版本，其在 PP-HGNet 的基础上，做了进一步优化和改进，最终在 NVIDIA GPU 设备上，将 "Accuracy-Latency Balance" 做到了极致，精度大幅超过了其他同样推理速度的模型。其在单标签分类、多标签分类、目标检测、语义分割等任务中，均有较强的表现。该模型是PaddleX非常推荐的服务端模型。|
|PP-HGNetV2-B1|79.18|6.56034|27.3099|22.6 M||
|PP-HGNetV2-B2|81.74|9.60494|43.1219|39.9 M||
|PP-HGNetV2-B3|82.98|11.0042|55.1367|57.9 M||
|PP-HGNetV2-B4|83.57|9.66407|54.2462|70.4 M||
|PP-HGNetV2-B5|84.75|15.7091|115.926|140.8 M||
|PP-HGNetV2-B6|86.30|21.226|255.279|268.4 M||
|PP-LCNet_x0_5|63.14|3.67722|6.66857|6.7 M|PP-LCNet是百度飞桨视觉团队自研的轻量级骨干网络，它能在不增加推理时间的前提下，进一步提升模型的性能，大幅超越其他轻量级SOTA模型。该模型是PaddleX非常推荐的轻量级模型。|
|PP-LCNet_x0_25|51.86|2.65341|5.81357|5.5 M||
|PP-LCNet_x0_35|58.09|2.7212|6.28944|5.9 M||
|PP-LCNet_x0_75|68.18|3.91032|8.06953|8.4 M||
|PP-LCNet_x1_0|71.32|3.84845|9.23735|10.5 M||
|PP-LCNet_x1_5|73.71|3.97666|12.3457|16.0 M||
|PP-LCNet_x2_0|75.18|4.07556|16.2752|23.2 M||
|PP-LCNet_x2_5|76.60|4.06028|21.5063|32.1 M||
|PP-LCNetV2_base|77.05|5.23428|19.6005|23.7 M|PP-LCNetV2 图像分类模型是百度飞桨视觉团队自研的 PP-LCNet 的下一代版本，其在 PP-LCNet 的基础上，做了进一步优化和改进，主要使用重参数化策略组合了不同大小卷积核的深度卷积，并优化了点卷积、Shortcut等。在不使用额外数据的前提下，PPLCNetV2_base 模型在图像分类 ImageNet 数据集上能够取得超过 77% 的 Top1 Acc，同时在 Intel CPU 平台的推理时间在 4.4 ms 以下 |
|PP-LCNetV2_large |78.51|6.78335|30.4378|37.3 M||
|PP-LCNetV2_small|73.97|3.89762|13.0273|14.6 M||
|ResNet18_vd|72.3|3.53048|31.3014|41.5 M|ResNet 系列模型是在 2015 年提出的，一举在 ILSVRC2015 比赛中取得冠军，top5 错误率为 3.57%。该网络创新性的提出了残差结构，通过堆叠多个残差结构从而构建了 ResNet 网络。实验表明使用残差块可以有效地提升收敛速度和精度。|
|ResNet18|71.0|2.4868|27.4601|41.5 M||
|ResNet34_vd|76.0|5.60675|56.0653|77.3 M||
|ResNet34|74.6|4.16902|51.925|77.3 M||
|ResNet50_vd|79.1|10.1885|68.446|90.8 M||
|ResNet50|76.5|9.62383|64.8135|90.8 M||
|ResNet101_vd|80.2|20.0563|124.85|158.4 M||
|ResNet101|77.6|19.2297|121.006|158.7 M||
|ResNet152_vd|80.6|29.6439|181.678|214.3 M||
|ResNet152|78.3|30.0461|177.707|214.2 M||
|ResNet200_vd|80.9|39.1628|235.185|266.0 M||
|StarNet-S1|73.6|9.895|23.0465|11.2 M|StarNet 聚焦于研究网络设计中“星操作”（即元素级乘法）的未开发潜力。揭示星操作能够将输入映射到高维、非线性特征空间的能力，这一过程类似于核技巧，但无需扩大网络规模。因此进一步提出了 StarNet，一个简单而强大的原型网络，该网络在紧凑的网络结构和有限的计算资源下，展现出了卓越的性能和低延迟。 |
|StarNet-S2|74.8|7.91279|21.9571|14.3 M||
|StarNet-S3|77.0|10.7531|30.7656|22.2 M||
|StarNet-S4|79.0|15.2868|43.2497|28.9 M||
|SwinTransformer_base_patch4_window7_224|83.37|16.9848|383.83|310.5 M|SwinTransformer 是一种新的视觉 Transformer 网络，可以用作计算机视觉领域的通用骨干网路。SwinTransformer 由移动窗口（shifted windows）表示的层次 Transformer 结构组成。移动窗口将自注意计算限制在非重叠的局部窗口上，同时允许跨窗口连接，从而提高了网络性能。|
|SwinTransformer_base_patch4_window12_384|84.17|37.2855|1178.63|311.4 M||
|SwinTransformer_large_patch4_window7_224|86.19|27.5498|689.729|694.8 M||
|SwinTransformer_large_patch4_window12_384|87.06|74.1768|2105.22|696.1 M||
|SwinTransformer_small_patch4_window7_224|83.21|16.3982|285.56|175.6 M||
|SwinTransformer_tiny_patch4_window7_224|81.10|8.54846|156.306|100.1 M||
**注**：**以上精度指标为 **[ImageNet-1k](https://www.image-net.org/index.php)** 验证集 Top1 Acc。****所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

</details>

## 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验通用图像分类产线的效果，也可以在本地使用命令行或 Python 体验通用图像分类产线的效果。

### 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/100061/webUI)通用图像分类产线的效果，用官方提供的 demo 图片进行识别，例如：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=78ee34ce1c024d48a0b9895a920c8790&docGuid=u_VtdOCxJUF8GK "")
如果您对产线运行的效果满意，可以直接对产线进行集成部署，如果不满意，您也可以利用私有数据**对产线中的模型进行在线微调**。

### 2.2 本地体验
在本地使用通用图像分类产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.2.1 命令行方式体验
一行命令即可快速体验图像分类产线效果

```
paddlex --pipeline image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0
```
参数说明：

```
--pipeline：产线名称，此处为图像分类产线
--input：待处理的输入图片的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```
执行后，将提示选择图像分类产线配置文件保存路径，默认保存至*当前目录*，也可 *自定义路径*。 

此外，也可在执行命令时加入 -y 参数，则可跳过路径选择，直接将产线配置文件保存至当前目录。

获取产线配置文件后，可将 --pipeline 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 ./image_classification.yaml，只需执行：

```
paddlex --pipeline ./image_classification.yaml --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg
```
其中，--model、--device 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。

运行后，得到的结果为：

```
{'img_path': '/root/.paddlex/predict_input/general_image_classification_001.jpg', 'class_ids': [296, 170, 356, 258, 248], 'scores': [0.62736, 0.03752, 0.03256, 0.0323, 0.03194], 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Irish wolfhound', 'weasel', 'Samoyed, Samoyede', 'Eskimo dog, husky']}
λ szzj-acg-tge0-85
```
![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=73f8d0f2139040cd8217f1c1bd90cc39&docGuid=u_VtdOCxJUF8GK "")
#### 2.2.2 Python脚本方式集成 
几行代码即可完成产线的快速推理，以通用图像分类产线为例：

```
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="image_classification")

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

* 实例化 `create_pipeline` 实例化图像分类产线对象：具体参数说明如下：

|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|pipeline|产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。|str|无|
|device|产线模型推理设备。支持：“gpu”，“cpu”。|str|gpu|
|enable_hpi|是否启用高性能推理，仅当该产线支持高性能推理时可用。|bool|False|
* 调用图像分类产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

|参数类型|参数说明|
|-|-|
|Python Var|支持直接传入Python变量，如numpy.ndarray表示的图像数据；|
|str|支持传入待预测数据文件路径，如图像文件的本地路径：/root/data/img.jpg；|
|str|支持传入待预测数据文件url，如图像文件的网络url：https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg；|
|str|支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：/root/data/；|
|dict|支持传入字典类型，字典的key需要与具体产线对应，如图像分类产线为"img"，字典的val支持上述类型数据，如：{"img": "/root/data1"}；|
|list|支持传入列表，列表元素需为上述类型数据，如[numpy.ndarray, numpy.ndarray, ]，["/root/data/img1.jpg", "/root/data/img2.jpg", ]，["/root/data1", "/root/data2", ]，[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}, ]；|
* 调用 predict 方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以 batch 为单位对数据进行预测，因此预测结果为 list 形式表示的一组预测结果
* 对预测结果进行处理：每个样本的预测结果均为 dict 类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

|方法|说明|方法参数|
|-|-|-|
|print|打印结果到终端|format_json：bool类型，是否对输出内容进行使用json缩进格式化，默认为True；|
|||indent：int类型，json格式化设置，仅当format_json为True时有效，默认为4；|
|||ensure_ascii：bool类型，json格式化设置，仅当format_json为True时有效，默认为False；|
|save_to_json|将结果保存为json格式的文件|save_path：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；|
|||indent：int类型，json格式化设置，默认为4;|
|||ensure_ascii：bool类型，json格式化设置，默认为False；|
|save_to_img|将结果保存为图像格式的文件|save_path：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；|

在执行上述 Python 脚本时，加载的是默认的图像分类产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

```
paddlex --get_pipeline_config image_classification
```
执行后，图像分类产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为* ./my_path*）：

```
paddlex --get_pipeline_config image_classification --config_save_path ./my_path
```
获取配置文件后，您即可对图像分类产线各项配置进行自定义，只需要修改 create_pipeline 方法中的 pipeline 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 *./my_path/*image_classification*.yaml* ，则只需执行：

```
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/image_classification.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
## 开发集成/部署
如果通用图像分类产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将通用图像分类产线直接应用在您的 Python 项目中，可以参考 2.2.2 Python脚本方式中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

* 高性能部署：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能部署流程请参考[PaddleX 高性能部署指南](../../../pipeline_deploy/high_performance_deploy.md)。
* 服务化部署：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX 服务化部署指南](../../../pipeline_deploy/service_deploy.md)。
* 端侧部署：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/lite_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 二次开发
如果通用图像分类产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用**您自己拥有的特定领域或应用场景的数据**对现有模型进行进一步的**微调**，以提升通用图像分类产线的在您的场景中的识别效果。

### 4.1 模型微调
由于通用图像分类产线包含图像分类模块，如果模型产线的效果不及预期，那么您需要参考[图像分类模块开发教程](../../../module_usage/tutorials/cv_modules/image_classification.md)中的**二次开发**章节，使用您的私有数据集对图像分类模型进行微调。

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
随后， 参考 *2.2 本地体验* 中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，**仅需修改 ****--device**** 参数**即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行图像分类产线的推理，使用的 Python 命令为：

```
paddlex --pipeline image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需对 Python 命令中的 --device 修改为 npu 即可：

```
paddlex --pipeline image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device npu:0
```
若您想在更多种类的硬件上使用通用图像分类产线，请参考[PaddleX多硬件使用指南](../../../installation/installation_other_devices.md)。
