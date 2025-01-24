---
comments: true
---

# 通用语义分割产线使用教程

## 1. 通用语义分割产线介绍
语义分割是一种计算机视觉技术，旨在将图像中的每个像素分配到特定的类别，从而实现对图像内容的精细化理解。语义分割不仅要识别出图像中的物体类型，还要对每个像素进行分类，这样使得同一类别的区域能够被完整标记。例如，在一幅街景图像中，语义分割可以将行人、汽车、天空和道路等不同类别的部分逐像素区分开来，形成一个详细的标签图。通用语义分割产线用于解决像素级别的图像理解问题，这项技术广泛应用于自动驾驶、医学影像分析和人机交互等领域，通常依赖于深度学习模型（如SegFormer等），通过卷积神经网络（CNN）或视觉变换器（Transformer）来提取特征并实现高精度的像素级分类，从而为进一步的智能分析提供基础。本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/semantic_segmentation/01.png"/>
<b>通用语义分割产线中包含了语义分割模块，该模块都包含多个模型，您可以根据下方的基准测试数据选择使用的模型</b>。

<b>如果您更注重模型的精度，请选择精度较高的模型；如果您更在意模型的推理速度，请选择推理速度较快的模型；如果您关注模型的存储大小，请选择存储体积较小的模型。</b>
<p><b>通用图像语义分割模块（可选）：</b></p>
<table>
<thead>
<tr>
<th>模型名称</th><th>模型下载链接</th>
<th>mloU（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>OCRNet_HRNet-W48</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W48_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W48_pretrained.pdparams">训练模型</a></td>
<td>82.15</td>
<td>627.36 / 170.76</td>
<td>3531.61 / 3531.61</td>
<td>249.8 M</td>
</tr>
<tr>
<td>PP-LiteSeg-T</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-T_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-T_pretrained.pdparams">训练模型</a></td>
<td>73.10</td>
<td>30.16 / 14.03</td>
<td>420.07 / 235.01</td>
<td>28.5 M</td>
</tr>
</tbody>
</table>
<p><b>注：以上模型精度指标测量自<a href="https://www.cityscapes-dataset.com/">Cityscapes</a>数据集。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p>
>❗ 以上列出的是语义分割模块重点支持的<b>2个核心模型</b>，该模块总共支持<b>18个模型</b>，完整的模型列表如下：

<details><summary> 👉模型列表详情</summary>
<table>
<thead>
<tr>
<th>模型名称</th><th>模型下载链接</th>
<th>mloU（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Deeplabv3_Plus-R50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R50_pretrained.pdparams">训练模型</a></td>
<td>80.36</td>
<td>503.51 / 122.30</td>
<td>3543.91 / 3543.91</td>
<td>94.9 M</td>
</tr>
<tr>
<td>Deeplabv3_Plus-R101</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R101_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R101_pretrained.pdparams">训练模型</a></td>
<td>81.10</td>
<td>803.79 / 175.45</td>
<td>5136.21 / 5136.21</td>
<td>162.5 M</td>
</tr>
<tr>
<td>Deeplabv3-R50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R50_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3-R50_pretrained.pdparams">训练模型</a></td>
<td>79.90</td>
<td>647.56 / 121.67</td>
<td>3803.09 / 3803.09</td>
<td>138.3 M</td>
</tr>
<tr>
<td>Deeplabv3-R101</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R101_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3-R101_pretrained.pdparams">训练模型</a></td>
<td>80.85</td>
<td>950.43 / 178.50</td>
<td>5517.14 / 5517.14</td>
<td>205.9 M</td>
</tr>
<tr>
<td>OCRNet_HRNet-W18</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W18_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W18_pretrained.pdparams">训练模型</a></td>
<td>80.67</td>
<td>286.12 / 80.76</td>
<td>1794.03 / 1794.03</td>
<td>43.1 M</td>
</tr>
<tr>
<td>OCRNet_HRNet-W48</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W48_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W48_pretrained.pdparams">训练模型</a></td>
<td>82.15</td>
<td>627.36 / 170.76</td>
<td>3531.61 / 3531.61</td>
<td>249.8 M</td>
</tr>
<tr>
<td>PP-LiteSeg-T</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-T_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-T_pretrained.pdparams">训练模型</a></td>
<td>73.10</td>
<td>30.16 / 14.03</td>
<td>420.07 / 235.01</td>
<td>28.5 M</td>
</tr>
<tr>
<td>PP-LiteSeg-B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-B_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-B_pretrained.pdparams">训练模型</a></td>
<td>75.25</td>
<td>40.92 / 20.18</td>
<td>494.32 / 310.34</td>
<td>47.0 M</td>
</tr>
<tr>
<td>SegFormer-B0 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B0 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B0 (slice)_pretrained.pdparams">训练模型</a></td>
<td>76.73</td>
<td>11.1946</td>
<td>268.929</td>
<td>13.2 M</td>
</tr>
<tr>
<td>SegFormer-B1 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B1 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B1 (slice)_pretrained.pdparams">训练模型</a></td>
<td>78.35</td>
<td>17.9998</td>
<td>403.393</td>
<td>48.5 M</td>
</tr>
<tr>
<td>SegFormer-B2 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B2 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B2 (slice)_pretrained.pdparams">训练模型</a></td>
<td>81.60</td>
<td>48.0371</td>
<td>1248.52</td>
<td>96.9 M</td>
</tr>
<tr>
<td>SegFormer-B3 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B3 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B3 (slice)_pretrained.pdparams">训练模型</a></td>
<td>82.47</td>
<td>64.341</td>
<td>1666.35</td>
<td>167.3 M</td>
</tr>
<tr>
<td>SegFormer-B4 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B4 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B4 (slice)_pretrained.pdparams">训练模型</a></td>
<td>82.38</td>
<td>82.4336</td>
<td>1995.42</td>
<td>226.7 M</td>
</tr>
<tr>
<td>SegFormer-B5 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B5 (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B5 (slice)_pretrained.pdparams">训练模型</a></td>
<td>82.58</td>
<td>97.3717</td>
<td>2420.19</td>
<td>229.7 M</td>
</tr>
</tbody>
</table>
<p><b>注：以上模型精度指标测量自<a href="https://www.cityscapes-dataset.com/">Cityscapes</a>数据集。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p>
<table>
<thead>
<tr>
<th>模型名称</th><th>模型下载链接</th>
<th>mloU（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>SeaFormer_base(slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_base(slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_base(slice)_pretrained.pdparams">训练模型</a></td>
<td>40.92</td>
<td>24.4073</td>
<td>397.574</td>
<td>30.8 M</td>
</tr>
<tr>
<td>SeaFormer_large (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_large (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_large (slice)_pretrained.pdparams">训练模型</a></td>
<td>43.66</td>
<td>27.8123</td>
<td>550.464</td>
<td>49.8 M</td>
</tr>
<tr>
<td>SeaFormer_small (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_small (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_small (slice)_pretrained.pdparams">训练模型</a></td>
<td>38.73</td>
<td>19.2295</td>
<td>358.343</td>
<td>14.3 M</td>
</tr>
<tr>
<td>SeaFormer_tiny (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_tiny (slice)_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_tiny (slice)_pretrained.pdparams">训练模型</a></td>
<td>34.58</td>
<td>13.9496</td>
<td>330.132</td>
<td>6.1M</td>
</tr>
</tbody>
</table>
<p><b>注：SeaFormer系列模型的精度指标测量自<a href="https://groups.csail.mit.edu/vision/datasets/ADE20K/">ADE20k</a>数据集。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p></details>

## 2. 快速开始
PaddleX 所提供的模型产线均可以快速体验效果，你可以在星河社区线体验通用 语义分割 产线的效果，也可以在本地使用命令行或 Python 体验通用 语义分割 产线的效果。

### 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/100062/webUI?source=appCenter)通用语义分割产线的效果，用官方提供的 Demo 图片进行识别，例如：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/semantic_segmentation/02.png"/>

如果您对产线运行的效果满意，可以直接进行集成部署。您可以选择从云端下载部署包，也可以参考[2.2节本地体验](#22-本地体验)中的方法进行本地部署。如果对效果不满意，您可以利用私有数据<b>对产线中的模型进行微调训练</b>。如果您具备本地训练的硬件资源，可以直接在本地开展训练；如果没有，星河零代码平台提供了一键式训练服务，无需编写代码，只需上传数据后，即可一键启动训练任务。

### 2.2 本地体验
>❗ 在本地使用通用语义分割产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.2.1 命令行方式体验
* 一行命令即可快速体验语义分割产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/application/semantic_segmentation/makassaridn-road_demo.png)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline semantic_segmentation \
        --input makassaridn-road_demo.png \
        --target_size -1 \
        --save_path ./output \
        --device gpu:0 \
```
相关的参数说明可以参考[2.2.2 Python脚本方式集成](#222-python脚本方式集成)中的参数说明。

运行后，会将结果打印到终端上，结果如下：
```bash
{'res': {'input_path': 'makassaridn-road_demo.png', 'page_index': None, 'pred': '...'}}
```
运行结果参数说明可以参考[2.2.2 Python脚本方式集成](#222-python脚本方式集成)中的结果解释。

可视化结果保存在`save_path`下，其中语义分割的可视化结果如下：
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/semantic_segmentation/03.png"/>

#### 2.2.2 Python脚本方式集成
* 上述命令行是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="semantic_segmentation")
output = pipeline.predict(input="makassaridn-road_demo.png", target_size = -1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）通过 `create_pipeline()` 实例化 语义分割 产线对象，具体参数说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>产线具体的配置信息（如果和<code>pipeline</code>同时设置，优先级高于<code>pipeline</code>，且要求产线名和<code>pipeline</code>一致）。</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线推理设备。支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理，仅当该产线支持高性能推理时可用。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

（2）调用 语义分割 产线对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：

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
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_semantic_segmentation_002.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
<li><b>List</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线推理设备</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
<li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
<li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
<li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
<li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
<li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<td><code>target_size</code></td>
<td>模型推理时实际使用的图像分辨率</td>
<td><code>int|-1|None|tuple[int,int]</code></td>
<td>
<ul>
<li><b>int</b>：大于 <code>0</code> 的任意整数
    <li><b>-1</b>：如果设置为<code>-1</code>, 即不进行任何rescale操作, 直接用图像原始分辨率预测
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的参数。即直接原图分辨率预测
    <li><b>tuple[int,int]</b>: 图像实际预测分辨率会被rescale到这个尺寸</li></li></li></li></ul></td>



<td><code>None</code></td>

</table>

（3）对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为`json`文件的操作:

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
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
</table>

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：

    - `input_path`: `(str)` 待预测图像的输入路径

    - `page_index`: `(Union[int, None])` 如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`

    - `pred`: `...` 语义分割模型实际预测的mask，由于数据过大不便于直接print，因此用`...`替换，可以通过res.save_to_img将预测结果保存为图片，通过res.save_to_json将预测结果保存为json文件。

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。

- 调用`save_to_img()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。

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
<td rowspan="1">获取预测的 <code>json</code> 格式的结果</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">获取格式为 <code>dict</code> 的可视化图像</td>
</tr>
</table>

- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。
- `img` 属性返回的预测结果是一个字典类型的数据。其中，键为 `res`, 对应的值是一个 `Image.Image` 对象：一个用于显示 语义分割 的预测结果。

此外，您可以获取 语义分割 产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config semantic_segmentation --save_path ./my_path
```

若您获取了配置文件，即可对语义分割产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。示例如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/semantic_segmentation.yaml")

output = pipeline.predict(
    input="./makassaridn-road_demo.png",
    target_size=-1,
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改通用语义分割产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。同时，CLI 预测也支持传入配置文件，`--pipeline` 指定配置文件的路径即可。

## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 <b>高性能推理</b>：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ <b>服务化部署</b>：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持多种产线服务化部署方案，详细的产线服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/serving.md)。

以下是基础服务化部署的API参考与多语言服务调用示例：

<details><summary>API参考</summary>
<p>对于服务提供的主要操作：</p>
<ul>
<li>HTTP请求方法为POST。</li>
<li>请求体和响应体均为JSON数据（JSON对象）。</li>
<li>当请求处理成功时，响应状态码为<code>200</code>，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。固定为<code>0</code>。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。固定为<code>"Success"</code>。</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>操作结果。</td>
</tr>
</tbody>
</table>
<ul>
<li>当请求处理未成功时，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。与响应状态码相同。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。</td>
</tr>
</tbody>
</table>
<p>服务提供的主要操作如下：</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>对图像进行语义分割。</p>
<p><code>POST /semantic-segmentation</code></p>
<ul>
<li>请求体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>服务器可访问的图像文件的URL或图像文件内容的Base64编码结果。</td>
<td>是</td>
</tr>
<tr>
<td><code>targetSize</code></td>
<td><code>integer</code> | <code>array</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>target_size</code> 参数说明。</td>
<td>否</td>
</tr>
</tbody>
</table>
<ul>
<li>请求处理成功时，响应体的<code>result</code>具有如下属性：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>labelMap</code></td>
<td><code>array</code></td>
<td>记录图像中每个像素的类别标签（按照行优先顺序排列）。</td>
</tr>
<tr>
<td><code>size</code></td>
<td><code>array</code></td>
<td>图像形状。数组中元素依次为图像的高度和宽度。</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code> | <code>null</code></td>
<td>语义分割结果图。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<p><code>result</code>示例如下：</p>
<pre><code class="language-json">{
"labelMap": [
0,
0,
1,
2
],
"size": [
2,
2
],
"image": "xxxxxx"
}
</code></pre></details>
<details><summary>多语言调用服务示例</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/semantic-segmentation" # 服务URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64编码的文件内容或者图像URL

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
# result.labelMap 记录图像中每个像素的类别标签（按照行优先顺序排列）详见API参考文档
</code></pre></details>
<details><summary>C++</summary>
<pre><code class="language-cpp">#include &lt;iostream&gt;
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client("localhost:8080");
    const std::string imagePath = "./demo.jpg";
    const std::string outputImagePath = "./out.jpg";

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    // 对本地图像进行Base64编码
    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; "Error reading file." &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["image"] = encodedImage;
    std::string body = jsonObj.dump();

    // 调用API
    auto response = client.Post("/semantic-segmentation", headers, body, "application/json");
    // 处理接口返回数据
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse["result"];

        encodedImage = result["image"];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; "Output image saved at " &lt;&lt; outPutImagePath &lt;&lt; std::endl;
            // result.labelMap 记录图像中每个像素的类别标签（按照行优先顺序排列）详见API参考文档
        } else {
            std::cerr &lt;&lt; "Unable to open file for writing: " &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; "Failed to send HTTP request." &lt;&lt; std::endl;
        return 1;
    }

    return 0;
}
</code></pre></details>
<details><summary>Java</summary>
<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/semantic-segmentation"; // 服务URL
        String imagePath = "./demo.jpg"; // 本地图像
        String outputImagePath = "./out.jpg"; // 输出图像

        // 对本地图像进行Base64编码
        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("image", imageData); // Base64编码的文件内容或者图像URL

        // 创建 OkHttpClient 实例
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // 调用API并处理接口返回数据
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get("result");
                String base64Image = result.get("image").asText();
                JsonNode labelMap = result.get("labelMap");

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println("Output image saved at " + outputImagePath);
                // result.labelMap 记录图像中每个像素的类别标签（按照行优先顺序排列）详见API参考文档
            } else {
                System.err.println("Request failed with code: " + response.code());
            }
        }
    }
}
</code></pre></details>
<details><summary>Go</summary>
<pre><code class="language-go">package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
)

func main() {
    API_URL := "http://localhost:8080/semantic-segmentation"
    imagePath := "./demo.jpg"
    outputImagePath := "./out.jpg"

    // 对本地图像进行Base64编码
    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println("Error reading image file:", err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{"image": imageData} // Base64编码的文件内容或者图像URL
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println("Error marshaling payload:", err)
        return
    }

    // 调用API
    client := &amp;http.Client{}
    req, err := http.NewRequest("POST", API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println("Error creating request:", err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println("Error sending request:", err)
        return
    }
    defer res.Body.Close()

    // 处理接口返回数据
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }
    type Response struct {
        Result struct {
            Image      string   `json:"image"`
            Labelmap []map[string]interface{} `json:"labelMap"`
        } `json:"result"`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println("Error unmarshaling response body:", err)
        return
    }

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println("Error decoding base64 image data:", err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println("Error writing image to file:", err)
        return
    }
    fmt.Printf("Image saved at %s.jpg\n", outputImagePath)
    // result.labelMap 记录图像中每个像素的类别标签（按照行优先顺序排列）详见API参考文档
}
</code></pre></details>
<details><summary>C#</summary>
<pre><code class="language-csharp">using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = "http://localhost:8080/semantic-segmentation";
    static readonly string imagePath = "./demo.jpg";
    static readonly string outputImagePath = "./out.jpg";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // 对本地图像进行Base64编码
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { "image", image_data } }; // Base64编码的文件内容或者图像URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        // 调用API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // 处理接口返回数据
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse["result"]["image"].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($"Output image saved at {outputImagePath}");
        // result.labelMap 记录图像中每个像素的类别标签（按照行优先顺序排列）详见API参考文档
    }
}
</code></pre></details>
<details><summary>Node.js</summary>
<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/semantic-segmentation'
const imagePath = './demo.jpg'
const outputImagePath = "./out.jpg";

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)  // Base64编码的文件内容或者图像URL
  })
};

// 对本地图像进行Base64编码
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// 调用API
axios.request(config)
.then((response) =&gt; {
    // 处理接口返回数据
    const result = response.data["result"];
    const imageBuffer = Buffer.from(result["image"], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    // result.labelMap 记录图像中每个像素的类别标签（按照行优先顺序排列）详见API参考文档
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>
<details><summary>PHP</summary>
<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/semantic-segmentation"; // 服务URL
$image_path = "./demo.jpg";
$output_image_path = "./out.jpg";

// 对本地图像进行Base64编码
$image_data = base64_encode(file_get_contents($image_path));
$payload = array("image" =&gt; $image_data); // Base64编码的文件内容或者图像URL

// 调用API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// 处理接口返回数据
$result = json_decode($response, true)["result"];
file_put_contents($output_image_path, base64_decode($result["image"]));
echo "Output image saved at " . $output_image_path . "\n";
// result.labelMap 记录图像中每个像素的类别标签（按照行优先顺序排列）详见API参考文档
?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果通用语义分割产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升通用语义分割产线的在您的场景中的识别效果。

### 4.1 模型微调
由于通用语义分割产线包含语义分割模块，如果模型产线的效果不及预期，您可以对分割效果差的图片进行分析，并参考以下表格中对应的微调教程链接进行模型微调。


<table>
<thead>
<tr>
<th>情形</th>
<th>微调模块</th>
<th>微调参考链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>预测结果不达预期</td>
<td>语义分割模块</td>
<td><a href="../../../module_usage/tutorials/cv_modules/semantic_segmentation.md">链接</a></td>
</tr>
</tbody>
</table>

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```yaml
SubModules:
  SemanticSegmentation:
    module_name: semantic_segmentation
    model_name: PP-LiteSeg-T
    model_dir: null # 替换为微调后的语义分割模型权重路径
    batch_size: 1
    target_size: None
```
随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

## 5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device`参数</b>即可完成不同硬件之间的无缝切换。

例如，您使用昇腾 NPU 进行 语义分割 产线的推理，使用的 CLI 命令为：

```bash
paddlex --pipeline semantic_segmentation \
        --input makassaridn-road_demo.png \
        --target_size -1 \
        --save_path ./output \
        --device npu:0
```

当然，您也可以在 Python 脚本中 `create_pipeline()` 时或者 `predict()` 时指定硬件设备。

若您想在更多种类的硬件上使用通用语义分割产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
