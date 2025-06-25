---
comments: true
---

# PaddleX 3.0 版面区域检测（layout_detection）模型产线教程 —— 大模型训练数据构建教程

PaddleX 提供了丰富的模型产线，模型产线由一个或多个模型组合实现，每个模型产线都能够解决特定的场景任务问题。PaddleX 所提供的模型产线均支持快速体验，如果效果不及预期，也同样支持使用私有数据微调模型，并且 PaddleX 提供了 Python API，方便将产线集成到个人项目中。在使用之前，您首先需要安装 PaddleX， 安装方式请参考 [PaddleX本地安装教程](../installation/installation.md)。此处以版面区域检测任务为例子，介绍该模型产线在为大模型提供结构化文本语料的实际场景中的使用流程。


## 1. 选择模型产线

文档版面区域检测技术通过精准识别并定位文档中的标题、文本块、表格等元素及其空间布局关系，为后续文本分析构建结构化上下文，是文档智能处理流程的核心前置环节。在语言大模型、多模态文档理解及RAG（检索增强生成）技术快速发展的背景下，高质量结构化数据已成为模型训练与知识库构建的关键需求。通过版面区域检测技术，我们可以从文档图像中自动提取出标题、作者、摘要、关键词、发表年份、期刊名称、引用信息等关键信息结合OCR识别可编辑文本，以结构化的形式存储，为大模型训练数据提供丰富的语料。为学术研究的深入发展提供了强有力的支持。

首先，需要根据任务场景，选择对应的 PaddleX 产线，本节以版面区域检测的结果后处理优化和结合OCR进行区域文本识别为例，希望获取文档图像中的丰富的语料信息，对应 PaddleX 的版面区域检测模块，可以在目标检测产线中使用。如果无法确定任务和产线的对应关系，您可以在 PaddleX 支持的[模型产线列表](../support_list/pipelines_list.md)中了解相关产线的能力介绍。


## 2. 模型列表

该模块总共支持<b>11个全量模型</b>，包含多个预定义了不同类别的模型，完整的模型列表如下：

* <b>版面区域检测模型，包含23个常见的类别：文档标题、段落标题、文本、页码、摘要、目录、参考文献、脚注、页眉、页脚、算法、公式、公式编号、图像、图表标题、表格、表格标题、印章、图表标题、图表、页眉图像、页脚图像、侧栏文本</b>

<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">训练模型</a></td>
<td>90.4</td>
<td>33.59 / 33.59</td>
<td>503.01 / 251.08</td>
<td>123.76</td>
<td>基于RT-DETR-L在包含中英文论文、杂志、合同、书本、试卷和研报等场景的自建数据集训练的高精度版面区域定位模型</td>
</tr>
<tr>
<td>PP-DocLayout-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">训练模型</a></td>
<td>75.2</td>
<td>13.03 / 4.72</td>
<td>43.39 / 24.44</td>
<td>22.578</td>
<td>基于PicoDet-L在包含中英文论文、杂志、合同、书本、试卷和研报等场景的自建数据集训练的精度效率平衡的版面区域定位模型</td>
</tr>
<tr>
<td>PP-DocLayout-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">训练模型</a></td>
<td>70.9</td>
<td>11.54 / 3.86</td>
<td>18.53 / 6.29</td>
<td>4.834</td>
<td>基于PicoDet-S在中英文论文、杂志、合同、书本、试卷和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
</tbody>
</table>

<b>注：以上精度指标的评估集是 PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志、合同、书本、试卷和研报等常见的 500 张文档类型图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b>


> ❗ 以上列出的是版面区域检测模块重点支持的<b>3个核心模型</b>，完整的模型列表如下：

<details><summary> 👉模型列表详情</summary>

* <b>表格版面区域检测模型</b>

<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x_table</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_table_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">训练模型</a></td>
<td>97.5</td>
<td>9.57 / 6.63</td>
<td>27.66 / 16.75</td>
<td>7.4</td>
<td>基于PicoDet-1x在自建数据集训练的高效率版面区域定位模型，可定位表格这1类区域</td>
</tr>
</table>

<b>注：以上精度指标的评估集是 PaddleOCR 自建的版面表格区域检测数据集，包含中英文 7835 张带有表格的论文文档类型图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b>

* <b>3类版面区域检测模型，包含表格、图像、印章</b>

<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>88.2</td>
<td>8.43 / 3.44</td>
<td>17.60 / 6.51</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>89.0</td>
<td>12.80 / 9.57</td>
<td>45.04 / 23.86</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的效率精度均衡版面区域定位模型</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>95.8</td>
<td>114.80 / 25.65</td>
<td>924.38 / 924.38</td>
<td>470.1</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型</td>
</tr>
</table>

<b>注：以上精度指标的评估集是 PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志和研报等常见的 1154 张文档类型图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b>

* <b>5类英文文档区域检测模型，包含文字、标题、表格、图片以及列表</b>

<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">训练模型</a></td>
<td>97.8</td>
<td>9.62 / 6.75</td>
<td>26.96 / 12.77</td>
<td>7.4</td>
<td>基于PicoDet-1x在PubLayNet数据集训练的高效率英文文档版面区域定位模型</td>
</tr>
</table>

<b>注：以上精度指标的评估集是 [PubLayNet](https://developer.ibm.com/exchanges/data/all/publaynet/) 的评估数据集，包含英文文档的 11245 张图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b>

* <b>17类区域检测模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</b>

<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>87.4</td>
<td>8.80 / 3.62</td>
<td>17.51 / 6.35</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>

<tr>
<td>PicoDet-L_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>89.0</td>
<td>12.60 / 10.27</td>
<td>43.70 / 24.42</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的效率精度均衡版面区域定位模型</td>
</tr>

<tr>
<td>RT-DETR-H_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>98.3</td>
<td>115.29 / 101.18</td>
<td>964.75 / 964.75</td>
<td>470.2</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型</td>
</tr>
</tbody>
</table>

<b>注：以上精度指标的评估集是 PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志和研报等常见的 892 张文档类型图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b>

</details>


## 3. 快速体验

PaddleX 提供了两种本地体验的方式，你可以在本地使用命令行或 Python 体验版面区域检测的效果。在本地使用版面区域检测产线前，请确保您已经按照[PaddleX本地安装教程](../installation/installation.md)完成了PaddleX的wheel包安装。

首先获取产线默认配置文件，由于版面检测任务属于目标检测产线，因此执行以下命令即可获取默认配置文件：

```bash
paddlex --get_pipeline_config object_detection --save_path ./my_path
```

获取的保存在`./my_path/object_detection.yaml`，修改配置文件，即可对产线各项配置进行自定义。

```yaml
pipeline_name: object_detection

SubModules:
  ObjectDetection:
    module_name: object_detection
    model_name: PP-DocLayout-L  # 修改为上文 2. 模型列表中的版面区域检测模型名称
    model_dir: null
    batch_size: 1
    img_size: null
    threshold: null
```

随后，加载自定义配置文件 `./my_path/object_detection.yaml`，参考以下本地体验中的命令行方式或 Python 脚本方式进行在线体验。


### 3.1 本地体验 ———— 命令行方式

运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_test_0.jpg)到本地。自定义配置文件保存在 `./my_path/object_detection.yaml` ，则只需执行：

```bash
paddlex --pipeline ./my_path/object_detection.yaml \
        --input layout_test_0.jpg \
        --threshold 0.5 \
        --save_path ./output/ \
        --device gpu:0
```

<details><summary>👉 <b>运行后，得到的结果为：（点击展开）</b></summary>

```bash
{'res': {'input_path': 'layout_test_0.jpg', 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9813319444656372, 'coordinate': [1194.3062, 1056.4965, 1435.4105, 1357.0413]}, {'cls_id': 2, 'label': 'text', 'score': 0.981147050857544, 'coordinate': [1194.2455, 725.1456, 1434.8359, 1049.3486]}, {'cls_id': 2, 'label': 'text', 'score': 0.9798598885536194, 'coordinate': [651.33997, 1243.4757, 892.80334, 1568.1959]}, {'cls_id': 1, 'label': 'image', 'score': 0.9786806106567383, 'coordinate': [654.7072, 535.5916, 1162.361, 871.6843]}, {'cls_id': 2, 'label': 'text', 'score': 0.9783889651298523, 'coordinate': [109.04134, 676.89685, 349.30542, 908.07996]}, {'cls_id': 2, 'label': 'text', 'score': 0.9776895046234131, 'coordinate': [922.3298, 1223.0814, 1164.6154, 1571.308]}, {'cls_id': 2, 'label': 'text', 'score': 0.9769193530082703, 'coordinate': [108.95575, 1778.8552, 350.25076, 2081.1873]}, {'cls_id': 2, 'label': 'text', 'score': 0.9768841862678528, 'coordinate': [1194.3513, 1363.4364, 1435.0712, 1568.2646]}, {'cls_id': 2, 'label': 'text', 'score': 0.9759471416473389, 'coordinate': [108.46416, 1517.3303, 349.46082, 1771.7134]}, {'cls_id': 2, 'label': 'text', 'score': 0.9742090106010437, 'coordinate': [651.50336, 1028.1143, 892.2849, 1236.0295]}, {'cls_id': 1, 'label': 'image', 'score': 0.9733730554580688, 'coordinate': [764.33875, 1602.9359, 1425.4358, 2066.0457]}, {'cls_id': 2, 'label': 'text', 'score': 0.9730471968650818, 'coordinate': [379.30127, 533.8668, 620.4098, 718.1861]}, {'cls_id': 2, 'label': 'text', 'score': 0.9729955196380615, 'coordinate': [446.23227, 1737.0651, 713.9191, 1896.8988]}, {'cls_id': 2, 'label': 'text', 'score': 0.9700003862380981, 'coordinate': [379.32547, 724.46204, 620.7275, 910.80316]}, {'cls_id': 2, 'label': 'text', 'score': 0.9692158699035645, 'coordinate': [379.22464, 1408.3339, 615.71564, 1567.7607]}, {'cls_id': 2, 'label': 'text', 'score': 0.968329668045044, 'coordinate': [108.44235, 1304.3308, 349.57904, 1511.2039]}, {'cls_id': 2, 'label': 'text', 'score': 0.9678011536598206, 'coordinate': [109.19986, 1017.0264, 349.31723, 1155.697]}, {'cls_id': 2, 'label': 'text', 'score': 0.9621286392211914, 'coordinate': [379.61563, 1262.9558, 620.6101, 1401.8175]}, {'cls_id': 2, 'label': 'text', 'score': 0.9615256190299988, 'coordinate': [379.74408, 1118.2937, 620.74426, 1256.6793]}, {'cls_id': 2, 'label': 'text', 'score': 0.958721935749054, 'coordinate': [1193.918, 532.8018, 1434.7432, 622.4289]}, {'cls_id': 2, 'label': 'text', 'score': 0.9559714198112488, 'coordinate': [922.562, 1080.9818, 1164.0387, 1215.7594]}, {'cls_id': 2, 'label': 'text', 'score': 0.9552412629127502, 'coordinate': [108.80721, 1161.9774, 349.6744, 1299.2362]}, {'cls_id': 2, 'label': 'text', 'score': 0.9507772922515869, 'coordinate': [109.40825, 580.92706, 349.14438, 670.02985]}, {'cls_id': 2, 'label': 'text', 'score': 0.949194073677063, 'coordinate': [1194.5726, 629.2398, 1434.1365, 718.40094]}, {'cls_id': 2, 'label': 'text', 'score': 0.9471532702445984, 'coordinate': [379.42627, 1021.85474, 619.82635, 1111.3379]}, {'cls_id': 2, 'label': 'text', 'score': 0.9420481324195862, 'coordinate': [652.1573, 930.8065, 887.3861, 1020.8891]}, {'cls_id': 2, 'label': 'text', 'score': 0.9257403016090393, 'coordinate': [922.33795, 932.40106, 1162.7422, 974.4318]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8954867124557495, 'coordinate': [134.43443, 939.7384, 325.42734, 987.1273]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8932625651359558, 'coordinate': [971.1828, 1004.2145, 1116.8691, 1052.5201]}, {'cls_id': 13, 'label': 'header', 'score': 0.8896792531013489, 'coordinate': [1129.3605, 160.20511, 1260.6161, 193.23979]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8894897699356079, 'coordinate': [404.63666, 944.63464, 597.64923, 992.9647]}, {'cls_id': 13, 'label': 'header', 'score': 0.8644265532493591, 'coordinate': [411.9833, 174.19376, 922.93945, 194.1207]}, {'cls_id': 3, 'label': 'number', 'score': 0.7865270376205444, 'coordinate': [1385.2574, 155.55762, 1418.2019, 194.63576]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.7727568745613098, 'coordinate': [485.61816, 1676.4192, 676.21356, 1699.8445]}, {'cls_id': 2, 'label': 'text', 'score': 0.7702363729476929, 'coordinate': [642.1937, 1925.9141, 714.12067, 1945.3125]}, {'cls_id': 11, 'label': 'doc_title', 'score': 0.7163856625556946, 'coordinate': [127.23631, 301.37003, 1316.9393, 473.41168]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.6773825287818909, 'coordinate': [657.4694, 892.38696, 1162.7767, 910.70337]}, {'cls_id': 2, 'label': 'text', 'score': 0.6245782375335693, 'coordinate': [110.19977, 534.36707, 318.00195, 554.423]}, {'cls_id': 13, 'label': 'header', 'score': 0.5930579304695129, 'coordinate': [130.28291, 144.46971, 308.31525, 197.32173]}, {'cls_id': 20, 'label': 'header_image', 'score': 0.5242483019828796, 'coordinate': [130.28291, 144.46971, 308.31525, 197.32173]}]}}
```

参数含义如下：
- `input_path`：输入的待预测图像的路径
- `boxes`：预测的目标框信息，一个字典列表。每个字典代表一个检出的目标，包含以下信息：
  - `cls_id`：类别ID，一个整数
  - `label`：类别标签，一个字符串
  - `score`：目标框置信度，一个浮点数
  - `coordinate`：目标框坐标，一个浮点数列表，格式为<code>[xmin, ymin, xmax, ymax]</code>

</details>

在`output`目录中，保存了版面区域检测可视化和json格式保存的结果。版面区域定位结果可视化如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/layout_detection/layout_test_0_res.jpg">


### 3.2 本地体验 ———— Python 方式

通过上述命令行方式可快速体验查看效果，在项目中往往需要代码集成，您可以通过如下几行代码完成产线的快速推理：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/object_detection.yaml") # 加载自定义的配置文件，创建产线

output = pipeline.predict("layout_test_0.jpg", threshold=0.5)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

输出打印的结果与上述命令行体验方式一致。在`output`目录中，保存了版面区域检测可视化和json格式保存的结果。


## 4. 产线后处理调优

版面检测模型产线提供了多种后处理调优手段，帮助您进一步提升预测效果。`predict`方法中可传入的后处理参数请参考 [版面区域检测使用教程](../module_usage/tutorials/ocr_modules/layout_detection.md)。下面我们基于版面检测模型模型产线，介绍如何使用这些调优手段。


### 4.1 动态阈值调优 —— 可优化漏检误检

版面检测模型支持动态阈值调整，可以传入`threshold`参数，支持传入浮点数或自定义各个类别的阈值字典，为每个类别设定专属的检测得分阈值。这意味着您可以根据自己的数据，灵活调节漏检或误检的情况，确保每一次检测更加精准，`PP-DocLayout`系列模型的类别和id对应关系如下：

```yaml
{'paragraph_title': 0, 'image': 1, 'text': 2, 'number': 3, 'abstract': 4, 'content': 5,
'figure_title': 6, 'formula': 7, 'table': 8, 'table_title': 9, 'reference': 10, 'doc_title': 11, 'footnote': 12, 'header': 13, 'algorithm': 14, 'footer': 15, 'seal': 16, 'chart_title': 17, 'chart': 18, 'formula_number': 19, 'header_image': 20, 'footer_image': 21, 'aside_text': 22}
```

运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_test_2.jpg)到本地


```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/object_detection.yaml") # 阈值参数不设置时，默认为0.5
output = pipeline.predict("layout_test_2.jpg")
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

可以发现下左图在右上角有错误的`text`类别框识别出来

这时可以开启`threshold={2: 0.6}`，针对类别`text`，类别id是2，设置检测得分阈值为0.6，可以把错误的text框过滤掉，其余类别沿用默认阈值0.5。执行下面的代码:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/object_detection.yaml")
output = pipeline.predict("layout_test_2.jpg", threshold={2: 0.6}) # 针对类别2text，设置检测得分阈值为0.6，其余类别沿用默认阈值0.5
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

在保存目录查看可视化结果如下，可以发现下有图的右上角多余框已经被阈值过滤，只保留了最优的检测结果:

<div style="display: flex; justify-content: space-around;">
  <div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/layout_detection/layout_test_2_res.jpg" alt="Image 1" style="width:800;">
    <p>不设置</p>
  </div>
  <div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/layout_detection/layout_test_2_res_thred.jpg" alt="Image 2" style="width:800;">
    <p>layout_nms=threshold={2: 0.6}</p>
  </div>
</div>


### 4.2 重叠框过滤 —— 消除多余框干扰

`layout_nms`参数用于重叠框过滤，布尔类型，用于指定是否使用NMS（非极大值抑制）过滤重叠框，启用该功能，可以自动筛选最优的检测结果，消除多余干扰框；重叠框过滤功能，在默认情况下是关闭的，如果要开启该功能，需要在`predict`方法中传入参数`layout_nms=True`。执行下面的代码，`layout_nms=False`不开启重叠框过滤功能，执行下面的代码，查看结果。

在不开启重叠框过滤功能时，可以发现下左图在右上角有重叠框干扰，那么这时可以开启`layout_nms=True`过滤多余框，可以发现下右图的右上角多余框已经被过滤，只保留了最优的检测结果。分别执行不开启和开启过滤功能的代码，查看对比结果:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/object_detection.yaml")
# output = pipeline.predict("layout_test_2.jpg", threshold=0.5)  # 不开启重叠框过滤功能
output = pipeline.predict("layout_test_2.jpg", threshold=0.5, layout_nms=True)  # 开启重叠框过滤功能
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

查看对比的可视化结果如下：

<div style="display: flex; justify-content: space-around;">
  <div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/layout_detection/layout_test_2_res.jpg" alt="Image 1" style="width:800;">
    <p>不设置</p>
  </div>
  <div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/layout_detection/layout_test_2_res_post.jpg" alt="Image 2" style="width:800;">
    <p>layout_nms=True</p>
  </div>
</div>

### 4.3 可调框边长 —— 获取完整区块

`layout_unclip_ratio`参数，可调框边长，不再局限于固定的框大小，通过调整检测框的缩放倍数，在保持中心点不变的情况下，自由扩展或收缩框边长，便于输出正确完整的版面区域内容。

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/object_detection.yaml")
# output = pipeline.predict("layout_test_2.jpg",  threshold={2: 0.6})  # 不调整检测框边的缩放倍数
output = pipeline.predict("layout_test_2.jpg",  threshold={2: 0.6}, layout_unclip_ratio=(1.0, 1.05))  # 调整检测框的高的缩放倍数为1.05
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

在保存目录查看可视化结果如下，可以观察到，通过调整检测框的倍数为`layout_unclip_ratio=(1.0, 1.05)`时，可以获取高度更大的区域。

<div style="display: flex; justify-content: space-around;">
  <div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/layout_detection/layout_test_2_res.jpg" alt="Image 1" style="width:800;">
    <p>不设置</p>
  </div>
  <div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/layout_detection/layout_test_2_unclip_res.jpg" alt="Image 2" style="width:800;">
    <p>(1.0, 1.05)</p>
  </div>
</div>


### 4.4  框合并模式 —— 关注整体或细节

`layout_merge_bboxes_mode`: 框合并模式，模型输出的检测框的合并处理模式，可选“large”保留外框，“small”保留内框，不设置默认保留所有框。例如一个图表区域中含有多个子图，如果选择“large”模式，则保留一个图表最大框，便于整体的图表区域理解和对版面图表位置区域的恢复，如果选择“small”则保留子图多个框，便于对子图进行一一理解或处理。执行下面的代码，分别体验不设置`layout_merge_bboxes_mode`参数、`layout_merge_bboxes_mode='large'`和`layout_merge_bboxes_mode='small'`的结果差异。执行下面的代码，查看结果。下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/PMC4836298_00004.jpg)到本地。

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/object_detection.yaml")
output = pipeline.predict("PMC4836298_00004.jpg") # 默认不设置
# output = pipeline.predict("PMC4836298_00004.jpg", layout_merge_bboxes_mode="small") # 设置'small'模式
# output = pipeline.predict("PMC4836298_00004.jpg", layout_merge_bboxes_mode="large") # 设置'large'模式
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

在保存目录查看可视化结果对比如下，观察`chart`类别的预测结果差异，可以发现不设置`layout_merge_bboxes_mode`参数时，会保留所有框；设置为“large”模式时，则会合并成一个大的图表框，便于整体的图表区域理解和对版面图表位置区域的恢复；设置为“small”模式时，则会保留子图多个框，便于对子图进行一一理解或处理。

<div style="display: flex; justify-content: space-around;">
  <div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/layout_detection/PMC4836298_00004_res.jpg" alt="Image 1" style="width:500;">
    <p>不设置</p>
  </div>
  <div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/layout_detection/PMC4836298_00004_large_res.jpg" alt="Image 2" style="width:500;">
    <p>large模式</p>
  </div>
  <div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/layout_detection/PMC4836298_00004_small_res.jpg" alt="Image 3" style="width:500;">
    <p>small模式</p>
  </div>
</div>


## 5. 版面检测与OCR组合

PaddleX 提供了十分丰富的模型，以及针对不同任务的模型产线，同时PaddleX也支持用户多模型组合使用，以解决复杂、特定的任务。在版面区域检测产线的基础上，可以进一步集成OCR识别组件，实现版面区域的文字内容提取，为后续的大模型文字内容理解、摘要生成等任务的训练数据提供语料。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_test_0.jpg)到本地。执行下面的代码，查看结果。

```python

import cv2
from paddlex import create_pipeline

class LayoutOCRPipeline():

    def __init__(self):
        self.layout_pipeline = create_pipeline(pipeline="./my_path/object_detection.yaml")  # 加载上述自定义的配置文件，创建版面检测产线
        self.ocr_pipeline = create_pipeline(pipeline="OCR") # 加载OCR产线

    def crop_table(self, layout_res, layout_name):
        img_path = layout_res["input_path"]
        img = cv2.imread(img_path)
        table_img_list = []
        for box in layout_res["boxes"]:
            if box["label"] != layout_name: # 只关注layout_name指定的区域，忽略其他版面区域
                continue
            xmin, ymin, xmax, ymax = [int(i) for i in box["coordinate"]]
            table_img = img[ymin:ymax, xmin:xmax]
            table_img_list.append(table_img)
        return table_img_list

    def predict(self, data, layout_name):
        for layout_res in self.layout_pipeline.predict(data):  # 进行版面检测
            final_res = {}
            crop_img_list = self.crop_table(layout_res, layout_name) # 获取指定版面区域的裁剪图片列表
            if len(crop_img_list) == 0:
                continue
            ocr_res = list(self.ocr_pipeline.predict(    # 进行OCR文字识别
                                                input=crop_img_list,
                                                use_doc_orientation_classify=False, # 不使用文档方向分类
                                                use_doc_unwarping=False, # 不使用文档矫正
                                                use_textline_orientation=False # 不使用文字方向分类
                                                ))
            final_res[layout_name] = [res["rec_texts"] for res in ocr_res] # 将OCR识别结果整合到最终的输出中
            yield final_res


if __name__ == "__main__":
    solution = LayoutOCRPipeline()
    output = solution.predict("layout_test_0.jpg", layout_name="paragraph_title") # 只关注段落标题的区域
    for res in output:
        print(res)
```

输出结果如下：

```
{'paragraph_title': [['柔性执法', '好事办在群众心坎上'], ['缓解“停车难”', '加强精细化治理'], ['“潮汐”摊位', '聚拢文旅城市烟火气'], ['普法宣传“零距离”']]}
```

可见，已经正确抽取出了段落标题的文字内容, 形成结构化数据，可以作为训练数据提供给大模型训练文字内容理解、摘要生成等任务使用。


<b>注：这部分主要是演示如何将版面检测和OCR识别组合到一起，实际PaddleX官方已经提供了多种功能丰富的产线, 可以查看[PaddleX产线列表](../support_list/pipelines_list.md)。 </b>


## 6. 开发集成/部署

如果版面检测效果可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

### 6.1 直接后处理调整好的产线应用在您的 Python 项目中，可以参考如下示例代码：
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/object_detection.yaml")
output = pipeline.predict("layout_test_2.jpg", threshold=0.5, layout_nms=True, layout_merge_bboxes_mode="large")
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```
更多参数请参考 [目标检测产线使用教程](../pipeline_usage/tutorials/cv_pipelines/object_detection.md)。


### 6.2 以高稳定性服务化部署作为本教程的实践内容，具体可以参考 [PaddleX 服务化部署指南](../pipeline_deploy/serving.md) 进行实践。

**请注意，当前高稳定性服务化部署方案仅支持 Linux 系统。**

#### 6.2.1 获取SDK

下载目标检测高稳定性服务化部署 SDK <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_object_detection_sdk.tar.gz">paddlex_hps_object_detection_sdk.tar.gz</a>，解压 SDK 并运行部署脚本，如下：

```bash
tar -xvf paddlex_hps_object_detection_sdk.tar.gz
```

#### 6.2.2 获取序列号

- 在 [飞桨 AI Studio 星河社区-人工智能学习与实训社区](https://aistudio.baidu.com/paddlex/commercialization) 的“开源模型产线部署序列号咨询与获取”部分选择“立即获取”，如下图所示：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipeline_deploy/image-1.png">

选择目标检测产线，并点击“获取”。之后，可以在页面下方的“开源产线部署SDK序列号管理”部分找到获取到的序列号：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipeline_deploy/image-2.png">

**请注意**：每个序列号只能绑定到唯一的设备指纹，且只能绑定一次。这意味着用户如果使用不同的机器部署产线，则必须为每台机器准备单独的序列号。

#### 6.2.3 运行服务

运行服务：

- 支持使用 NVIDIA GPU 部署的镜像（机器上需要安装有支持 CUDA 11.8 的 NVIDIA 驱动）：

    ```bash
    docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.0.0rc0-gpu
    ```

- CPU-only 镜像：

    ```bash
    docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.0.0rc0-cpu
    ```

准备好镜像后，切换到 `server` 目录，执行如下命令运行服务器：

```bash
docker run \
    -it \
    -e PADDLEX_HPS_DEVICE_TYPE={部署设备类型} \
    -e PADDLEX_HPS_SERIAL_NUMBER={序列号} \
    -e PADDLEX_HPS_UPDATE_LICENSE=1 \
    -v "$(pwd)":/workspace \
    -v "${HOME}/.baidu/paddlex/licenses":/root/.baidu/paddlex/licenses \
    -v /dev/disk/by-uuid:/dev/disk/by-uuid \
    -w /workspace \
    --rm \
    --gpus all \
    --init \
    --network host \
    --shm-size 8g \
    {镜像名称} \
    /bin/bash server.sh
```

- 部署设备类型可以为 `cpu` 或 `gpu`，CPU-only 镜像仅支持 `cpu`。
- 如果希望使用 CPU 部署，则不需要指定 `--gpus`。
- 以上命令必须在激活成功后才可以正常执行。PaddleX 提供两种激活方式：离线激活和在线激活。具体说明如下：

    - 联网激活：在第一次执行时设置 `PADDLEX_HPS_UPDATE_LICENSE` 为 `1`，使程序自动更新证书并完成激活。再次执行命令时可以将 `PADDLEX_HPS_UPDATE_LICENSE` 设置为 `0` 以避免联网更新证书。
    - 离线激活：按照序列号管理部分中的指引，获取机器的设备指纹，并将序列号与设备指纹绑定以获取证书，完成激活。使用这种激活方式，需要手动将证书存放在机器的 `${HOME}/.baidu/paddlex/licenses` 目录中（如果目录不存在，需要创建目录）。使用这种方式时，将 `PADDLEX_HPS_UPDATE_LICENSE` 设置为 `0` 以避免联网更新证书。

- 必须确保宿主机的 `/dev/disk/by-uuid` 存在且非空，并正确挂载该目录，才能正常执行激活。
- 如果需要进入容器内部调试，可以将命令中的 `/bin/bash server.sh` 替换为 `/bin/bash`，然后在容器中执行 `/bin/bash server.sh`。
- 如果希望服务器在后台运行，可以将命令中的 `-it` 替换为 `-d`。容器启动后，可通过 `docker logs -f {容器 ID}` 查看容器日志。
- 在命令中添加 `-e PADDLEX_USE_HPIP=1` 可以使用 PaddleX 高性能推理插件加速产线推理过程。但请注意，并非所有产线都支持使用高性能推理插件。请参考 [PaddleX 高性能推理指南](../pipeline_deploy/high_performance_inference.md) 获取更多信息。

可观察到类似下面的输出信息：

```text
I1216 11:37:21.601943 35 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I1216 11:37:21.602333 35 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I1216 11:37:21.643494 35 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

#### 6.2.4 调用服务

目前，仅支持使用 Python 客户端调用服务。支持的 Python 版本为 3.8 至 3.12。

切换到高稳定性服务化部署 SDK 的 `client` 目录，执行如下命令安装依赖：

```bash
# 建议在虚拟环境中安装
python -m pip install -r requirements.txt
python -m pip install paddlex_hps_client-*.whl
```

`client` 目录的 `client.py` 脚本包含服务的调用示例，并提供命令行接口。


### 6.3 此外，PaddleX 也提供了其他三种部署方式，说明如下：

* 高性能部署：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能部署流程请参考 [PaddleX 高性能推理指南](../pipeline_deploy/high_performance_inference.md)。
* 基础服务化部署：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考 [PaddleX 服务化部署指南](../pipeline_deploy/serving.md)。
* 端侧部署：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考 [PaddleX端侧部署指南](../pipeline_deploy/on_device_deployment.md)。

您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。
