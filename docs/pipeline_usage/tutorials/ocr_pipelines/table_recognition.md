---
comments: true
---

# 通用表格识别产线使用教程

## 1. 通用表格识别产线介绍
表格识别是一种自动从文档或图像中识别和提取表格内容及其结构的技术，广泛应用于数据录入、信息检索和文档分析等领域。通过使用计算机视觉和机器学习算法，表格识别能够将复杂的表格信息转换为可编辑的格式，方便用户进一步处理和分析数据。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/01.png">

<b>通用</b><b>表格识别</b><b>产线中包含表格结构识别模块、版面区域分析模块、文本检测模块和文本识别模块</b>。

<b>如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。

<details><summary> 👉模型列表详情</summary>

<p><b>表格识别模块模型：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>精度（%）</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>SLANet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/SLANet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">训练模型</a></td>
<td>59.52</td>
<td>522.536</td>
<td>1845.37</td>
<td>6.9 M</td>
<td rowspan="2">SLANet 是百度飞桨视觉团队自研的表格结构识别模型。该模型通过采用CPU 友好型轻量级骨干网络PP-LCNet、高低层特征融合模块CSP-PAN、结构与位置信息对齐的特征解码模块SLA Head，大幅提升了表格结构识别的精度和推理速度。</td>
</tr>
<tr>
<td>SLANet_plus</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/SLANet_plus_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">训练模型</a></td>
<td>63.69</td>
<td>522.536</td>
<td>1845.37</td>
<td>6.9 M</td>
</tr>
</table>

<p><b>注：以上精度指标测量PaddleX 内部自建英文表格识别数据集。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<p><b>版面区域分析模块模型：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet_layout_1x_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">训练模型</a></td>
<td>86.8</td>
<td>13.0</td>
<td>91.3</td>
<td>7.4</td>
<td>基于PicoDet-1x在PubLayNet数据集训练的高效率版面区域定位模型，可定位包含文字、标题、表格、图片以及列表这5类区域</td>
</tr>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet_layout_1x_table_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">训练模型</a></td>
<td>95.7</td>
<td>12.623</td>
<td>90.8934</td>
<td>7.4 M</td>
<td>基于PicoDet-1x在自建数据集训练的高效率版面区域定位模型，可定位包含表格这1类区域</td>
</tr>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-S_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>87.1</td>
<td>13.5</td>
<td>45.8</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含3个类别：表格，图像和印章</td>
</tr>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-S_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>70.3</td>
<td>13.6</td>
<td>46.2</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-L_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>89.3</td>
<td>15.7</td>
<td>159.8</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含3个类别：表格，图像和印章</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-L_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>79.9</td>
<td>17.2</td>
<td>160.2</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/RT-DETR-H_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>95.9</td>
<td>114.6</td>
<td>3832.6</td>
<td>470.1</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型，包含3个类别：表格，图像和印章</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/RT-DETR-H_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>92.6</td>
<td>115.1</td>
<td>3827.2</td>
<td>470.2</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是 PaddleX 自建的版面区域分析数据集，包含 1w 张图片。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<p><b>文本检测模块模型：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">训练模型</a></td>
<td>77.79</td>
<td>10.6923</td>
<td>120.177</td>
<td>4.2 M</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">训练模型</a></td>
<td>82.69</td>
<td>83.3501</td>
<td>2434.01</td>
<td>100.1M</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<p><b>文本识别模块模型：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>78.20</td>
<td>7.95018</td>
<td>46.7868</td>
<td>10.6 M</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td>
<td>79.20</td>
<td>7.19439</td>
<td>140.179</td>
<td>71.2 M</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集 ，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p></details>

## 2. 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验通用图像分类产线的效果，也可以在本地使用命令行或 Python 体验通用图像分类产线的效果。

### 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/91661/webUI)通用表格识别产线的效果，用官方提供的 demo 图片进行识别，例如：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/02.png">

如果您对产线运行的效果满意，可以直接对产线进行集成部署，如果不满意，您也可以利用私有数据<b>对产线中的模型进行在线微调</b>。

### 2.2 本地体验
在本地使用通用表格识别产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

### 2.1 命令行方式体验
一行命令即可快速体验表格识别产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline table_recognition --input table_recognition.jpg --device gpu:0
```
参数说明：

```
--pipeline：产线名称，此处为表格识别产线
--input：待处理的输入图片的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```

在执行上述 Python 脚本时，加载的是默认的表格识别产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

<details><summary> 👉点击展开</summary>

<pre><code>paddlex --get_pipeline_config table_recognition
</code></pre>
<p>执行后，表格识别产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为 <code>./my_path</code> ）：</p>
<pre><code>paddlex --get_pipeline_config table_recognition --save_path ./my_path
</code></pre>
<p>获取产线配置文件后，可将 <code>--pipeline</code> 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 <code>./table_recognition.yaml</code>，只需执行：</p>
<pre><code class="language-bash">paddlex --pipeline ./table_recognition.yaml --input table_recognition.jpg --device gpu:0
</code></pre>
<p>其中，<code>--model</code>、<code>--device</code> 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。</p></details>

运行后，得到的结果为：

<details><summary> 👉点击展开</summary>

<pre><code>{'input_path': 'table_recognition.jpg', 'layout_result': {'input_path': 'table_recognition.jpg', 'boxes': [{'cls_id': 3, 'label': 'Table', 'score': 0.6014542579650879, 'coordinate': [0, 21, 551, 118]}]}, 'ocr_result': {'dt_polys': [array([[37., 40.],
       [75., 40.],
       [75., 60.],
       [37., 60.]], dtype=float32), array([[123.,  37.],
       [200.,  37.],
       [200.,  59.],
       [123.,  59.]], dtype=float32), array([[227.,  37.],
       [391.,  37.],
       [391.,  58.],
       [227.,  58.]], dtype=float32), array([[416.,  36.],
       [535.,  38.],
       [535.,  61.],
       [415.,  58.]], dtype=float32), array([[35., 73.],
       [78., 73.],
       [78., 92.],
       [35., 92.]], dtype=float32), array([[287.,  73.],
       [328.,  73.],
       [328.,  92.],
       [287.,  92.]], dtype=float32), array([[453.,  72.],
       [495.,  72.],
       [495.,  94.],
       [453.,  94.]], dtype=float32), array([[ 17., 103.],
       [ 94., 103.],
       [ 94., 118.],
       [ 17., 118.]], dtype=float32), array([[145., 104.],
       [178., 104.],
       [178., 118.],
       [145., 118.]], dtype=float32), array([[277., 104.],
       [337., 102.],
       [338., 118.],
       [278., 118.]], dtype=float32), array([[446., 102.],
       [504., 104.],
       [503., 118.],
       [445., 118.]], dtype=float32)], 'rec_text': ['Dres', '连续工作3', '取出来放在网上，没想', '江、整江等八大', 'Abstr', 'rSrivi', '$709.', 'cludingGiv', '2.72', 'Ingcubic', '$744.78'], 'rec_score': [0.9934158325195312, 0.9990204572677612, 0.9967061877250671, 0.9375461935997009, 0.9947397112846375, 0.9972746968269348, 0.9904290437698364, 0.973427414894104, 0.9983080625534058, 0.993423342704773, 0.9964120984077454], 'input_path': 'table_recognition.jpg'}, 'table_result': [{'input_path': 'table_recognition.jpg', 'layout_bbox': [0, 21, 551, 118], 'bbox': array([[  4.395736 ,  25.238262 , 113.31014  ,  25.316246 , 115.454315 ,
         71.8867   ,   3.7177477,  71.7937   ],
       [110.727455 ,  25.94007  , 210.07187  ,  26.028755 , 209.66394  ,
         65.96484  , 109.59861  ,  66.09809  ],
       [214.45381  ,  26.027939 , 407.95276  ,  26.112846 , 409.6684   ,
         66.91336  , 215.27292  ,  67.002014 ],
       [402.81863  ,  26.123789 , 549.03656  ,  26.231564 , 549.19995  ,
         66.88339  , 404.48068  ,  66.74034  ],
       [  2.4458022,  64.68588  , 102.7665   ,  65.10228  , 105.79447  ,
         96.051254 ,   2.5367072,  95.35514  ],
       [108.85877  ,  65.80549  , 211.70216  ,  66.02091  , 210.79245  ,
         94.75581  , 107.59308  ,  94.42664  ],
       [217.05621  ,  64.98496  , 407.76328  ,  65.133484 , 406.8436   ,
         96.00133  , 214.67896  ,  95.87226  ],
       [401.73572  ,  64.60494  , 547.9967   ,  64.73921  , 548.19135  ,
         96.09901  , 402.26733  ,  95.95529  ],
       [  2.4882016,  93.589554 , 107.01325  ,  93.67592  , 107.8446   ,
        120.13259  ,   2.508764 , 119.85027  ],
       [110.773125 ,  93.98633  , 213.354    ,  94.08046  , 212.46033  ,
        120.80207  , 109.29008  , 120.613045 ],
       [216.08781  ,  94.19984  , 405.843    ,  94.28341  , 405.9974   ,
        121.33152  , 215.10301  , 121.299034 ],
       [403.92212  ,  94.44883  , 548.30963  ,  94.54982  , 548.4949   ,
        122.610176 , 404.53433  , 122.49881  ]], dtype=float32), 'img_idx': 0, 'html': '&lt;html&gt;&lt;body&gt;&lt;table&gt;&lt;tr&gt;&lt;td&gt;Dres&lt;/td&gt;&lt;td&gt;连续工作3&lt;/td&gt;&lt;td&gt;取出来放在网上，没想&lt;/td&gt;&lt;td&gt;江、整江等八大&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;Abstr&lt;/td&gt;&lt;td&gt;&lt;/td&gt;&lt;td&gt;rSrivi&lt;/td&gt;&lt;td&gt;$709.&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;cludingGiv&lt;/td&gt;&lt;td&gt;2.72&lt;/td&gt;&lt;td&gt;Ingcubic&lt;/td&gt;&lt;td&gt;$744.78&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;/body&gt;&lt;/html&gt;'}]}
</code></pre></details>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/03.png">

可视化图片默认不进行保存，您可以通过 `--save_path` 自定义保存路径，随后所有结果将被保存在指定路径下。

### 2.2 Python脚本方式集成
几行代码即可完成产线的快速推理，以通用表格识别产线为例：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="table_recognition")

output = pipeline.predict("table_recognition.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存img格式结果
    res.save_to_xlsx("./output/") ## 保存表格格式结果
    res.save_to_html("./output/") ## 保存html结果
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

（1）实例化 `create_pipeline` 实例化产线对象：具体参数说明如下：

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
<td>无</td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线模型推理设备。支持：“gpu”，“cpu”。</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理，仅当该产线支持高性能推理时可用。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
（2）调用产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

<table>
<thead>
<tr>
<th>参数类型</th>
<th>参数说明</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>支持直接传入Python变量，如numpy.ndarray表示的图像数据。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入待预测数据文件路径，如图像文件的本地路径：<code>/root/data/img.jpg</code>。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入待预测数据文件URL，如图像文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg">示例</a>。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code>。</td>
</tr>
<tr>
<td>dict</td>
<td>支持传入字典类型，字典的key需与具体任务对应，如图像分类任务对应\"img\"，字典的val支持上述类型数据，例如：<code>{\"img\": \"/root/data1\"}</code>。</td>
</tr>
<tr>
<td>list</td>
<td>支持传入列表，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]，[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code>，<code>[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]</code>。</td>
</tr>
</tbody>
</table>
（3）调用`predict`方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以batch为单位对数据进行预测，因此预测结果为list形式表示的一组预测结果。

（4）对预测结果进行处理：每个样本的预测结果均为`dict`类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：


<table>
<thead>
<tr>
<th>方法</th>
<th>说明</th>
<th>方法参数</th>
</tr>
</thead>
<tbody>
<tr>
<td>save_to_img</td>
<td>将结果保存为img格式的文件</td>
<td><code>- save_path</code>：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；</td>
</tr>
<tr>
<td>save_to_html</td>
<td>将结果保存为html格式的文件</td>
<td><code>- save_path</code>：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；</td>
</tr>
<tr>
<td>save_to_xlsx</td>
<td>将结果保存为表格格式的文件</td>
<td><code>- save_path</code>：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；</td>
</tr>
</tbody>
</table>
其中，`save_to_img` 能够保存可视化结果（包括OCR结果图片、版面分析结果图片、表格结构识别结果图片）， `save_to_html` 能够将表格直接保存为html文件（包括文本和表格格式），`save_to_xlsx` 能够将表格保存为Excel格式文件（包括文本和格式）。

若您获取了配置文件，即可对表格识别产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 `./my_path/table_recognition.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/table_recognition.yaml")
output = pipeline.predict("table_recognition.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存img格式结果
    res.save_to_xlsx("./output/") ## 保存表格格式结果
    res.save_to_html("./output/") ## 保存html结果
```
## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

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
<p>定位并识别图中的表格。</p>
<p><code>POST /table-recognition</code></p>
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
<td><code>file</code></td>
<td><code>string</code></td>
<td>服务器可访问的图像文件或PDF文件的URL，或上述类型文件内容的Base64编码结果。对于超过10页的PDF文件，只有前10页的内容会被使用。</td>
<td>是</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code></td>
<td>文件类型。<code>0</code>表示PDF文件，<code>1</code>表示图像文件。若请求体无此属性，则将根据URL推断文件类型。</td>
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
<td><code>tableRecResults</code></td>
<td><code>object</code></td>
<td>表格识别结果。数组长度为1（对于图像输入）或文档页数与10中的较小者（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中每一页的处理结果。</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>输入数据信息。</td>
</tr>
</tbody>
</table>
<p><code>tableRecResults</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>tables</code></td>
<td><code>array</code></td>
<td>表格位置和内容。</td>
</tr>
<tr>
<td><code>layoutImage</code></td>
<td><code>string</code></td>
<td>版面区域检测结果图。图像为JPEG格式，使用Base64编码。</td>
</tr>
<tr>
<td><code>ocrImage</code></td>
<td><code>string</code></td>
<td>OCR结果图。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<p><code>tables</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>表格位置。数组中元素依次为边界框左上角x坐标、左上角y坐标、右下角x坐标以及右下角y坐标。</td>
</tr>
<tr>
<td><code>html</code></td>
<td><code>string</code></td>
<td>HTML格式的表格识别结果。</td>
</tr>
</tbody>
</table></details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/table-recognition&quot;
file_path = &quot;./demo.jpg&quot;

with open(file_path, &quot;rb&quot;) as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode(&quot;ascii&quot;)

payload = {&quot;file&quot;: file_data, &quot;fileType&quot;: 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
for i, res in enumerate(result[&quot;tableRecResults&quot;]):
    print(&quot;Detected tables:&quot;)
    print(res[&quot;tables&quot;])
    layout_img_path = f&quot;layout_{i}.jpg&quot;
    with open(layout_img_path, &quot;wb&quot;) as f:
        f.write(base64.b64decode(res[&quot;layoutImage&quot;]))
    ocr_img_path = f&quot;ocr_{i}.jpg&quot;
    with open(ocr_img_path, &quot;wb&quot;) as f:
        f.write(base64.b64decode(res[&quot;ocrImage&quot;]))
    print(f&quot;Output images saved at {layout_img_path} and {ocr_img_path}&quot;)
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果通用表格识别产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升通用表格识别产线的在您的场景中的识别效果。

### 4.1 模型微调
由于通用表格识别产线包含四个模块，模型产线的效果不及预期可能来自于其中任何一个模块。

您可以对识别效果差的图片进行分析，参考如下规则进行分析和模型微调：

* 检测到的表格结构错误（如行列识别错误、单元格位置错误），那么可能是表格结构识别模块存在不足，您需要参考[表格结构识别模块开发教程](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md)中的[二次开发](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md#四二次开发)章节，使用您的私有数据集对表格结构识别模型进行微调。
* 表格区域在整体版面中定位错误，那么可能是版面区域定位模块存在不足，您需要参考[版面区域检测模块开发教程](../../../module_usage/tutorials/ocr_modules/layout_detection.md)中的[二次开发](../../../module_usage/tutorials/ocr_modules/layout_detection.md#四二次开发)章节，使用您的私有数据集对版面区域定位模型进行微调。
* 有较多的文本未被检测出来（即文本漏检现象），那么可能是文本检测模型存在不足，您需要参考[文本检测模块开发教程](../../../module_usage/tutorials/ocr_modules/text_recognition.md)中的[二次开发](../../../module_usage/tutorials/ocr_modules/text_recognition.md#四二次开发)章节，使用您的私有数据集对文本检测模型进行微调。
* 已检测到的文本中出现较多的识别错误（即识别出的文本内容与实际文本内容不符），这表明文本识别模型需要进一步改进，您需要参考[文本识别模块开发教程](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md)中的[二次开发](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md#四二次开发)章节对文本识别模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```python
......
 Pipeline:
  layout_model: PicoDet_layout_1x  #可修改为微调后模型的本地路径
  table_model: SLANet  #可修改为微调后模型的本地路径
  text_det_model: PP-OCRv4_mobile_det  #可修改为微调后模型的本地路径
  text_rec_model: PP-OCRv4_mobile_rec  #可修改为微调后模型的本地路径
  layout_batch_size: 1
  text_rec_batch_size: 1
  table_batch_size: 1
  device: "gpu:0"
......
```
随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device` 参数</b>即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行表格识别产线的推理，使用的 Python 命令为：

```bash
paddlex --pipeline table_recognition --input table_recognition.jpg --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需对 Python 命令中的 `--device` 修改为npu 即可：

```bash
paddlex --pipeline table_recognition --input table_recognition.jpg --device npu:0
```
若您想在更多种类的硬件上使用通用表格识别产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
