---
comments: true
---

# 文档场景信息抽取v3产线使用教程

## 1. 文档场景信息抽取v3产线介绍
文档场景信息抽取v3（PP-ChatOCRv3）是飞桨特色的文档和图像智能分析解决方案，结合了 LLM 和 OCR 技术，一站式解决版面分析、生僻字、多页 pdf、表格、印章识别等常见的复杂文档信息抽取难点问题，结合文心大模型将海量数据和知识相融合，准确率高且应用广泛。

<img src="https://github.com/user-attachments/assets/90cb740b-7741-4383-bc4c-663f9d042d02">

文档场景信息抽取v3产线中包含<b>表格结构识别模块</b>、<b>版面区域检测模块</b>、<b>文本检测模块</b>、<b>文本识别模块</b>、<b>印章文本检测模块</b>、<b>文本图像矫正模块</b>、<b>文档图像方向分类模块</b>。

<b>如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。其中部分模型的 benchmark 如下：

<details><summary> 👉模型列表详情</summary>

<p><b>表格结构识别模块模型：</b></p>
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
<td>SLANet 是百度飞桨视觉团队自研的表格结构识别模型。该模型通过采用CPU 友好型轻量级骨干网络PP-LCNet、高低层特征融合模块CSP-PAN、结构与位置信息对齐的特征解码模块SLA Head，大幅提升了表格结构识别的精度和推理速度。</td>
</tr>
<tr>
<td>SLANet_plus</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/SLANet_plus_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">训练模型</a></td>
<td>63.69</td>
<td>522.536</td>
<td>1845.37</td>
<td>6.9 M</td>
<td>SLANet_plus 是百度飞桨视觉团队自研的表格结构识别模型SLANet的增强版。相较于SLANet，SLANet_plus 对无线表、复杂表格的识别能力得到了大幅提升，并降低了模型对表格定位准确性的敏感度，即使表格定位出现偏移，也能够较准确地进行识别。</td>
</tr>
</table>

<p><b>注：以上精度指标测量PaddleX 内部自建英文表格识别数据集。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<p><b>版面区域检测模块模型：</b></p>
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
<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的版面区域分析数据集，包含中英文论文、杂志和研报等常见的 1w 张文档类型图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p>
<p><b>文本检测模块模型：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">训练模型</a></td>
<td>82.69</td>
<td>83.3501</td>
<td>2434.01</td>
<td>109</td>
<td>PP-OCRv4 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">训练模型</a></td>
<td>77.79</td>
<td>10.6923</td>
<td>120.177</td>
<td>4.7</td>
<td>PP-OCRv4 的移动端文本检测模型，效率更高，适合在端侧设备部署</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<p><b>文本识别模块模型：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>78.20</td>
<td>7.95018</td>
<td>46.7868</td>
<td>10.6 M</td>
<td rowspan="2">PP-OCRv4是百度飞桨视觉团队自研的文本识别模型PP-OCRv3的下一个版本，通过引入数据增强方案、GTC-NRTR指导分支等策略，在模型推理速度不变的情况下，进一步提升了文本识别精度。该模型提供了服务端（server）和移动端（mobile）两个不同版本，来满足不同场景下的工业需求。</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td>
<td>79.20</td>
<td>7.19439</td>
<td>140.179</td>
<td>71.2 M</td>
</tr>
</table>

<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/ch_SVTRv2_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">训练模型</a></td>
<td>68.81</td>
<td>8.36801</td>
<td>165.706</td>
<td>73.9 M</td>
<td rowspan="1">
SVTRv2 是一种由复旦大学视觉与学习实验室（FVL）的OpenOCR团队研发的服务端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，A榜端到端识别精度相比PP-OCRv4提升6%。
</td>
</tr>
</table>

<p><b>注：以上精度指标的评估集是 <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>A榜。 所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/ch_RepSVTR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">训练模型</a></td>
<td>65.07</td>
<td>10.5047</td>
<td>51.5647</td>
<td>22.1 M</td>
<td rowspan="1">    RepSVTR 文本识别模型是一种基于SVTRv2 的移动端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，B榜端到端识别精度相比PP-OCRv4提升2.5%，推理速度持平。</td>
</tr>
</table>

<p><b>注：以上精度指标的评估集是 <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>B榜。 所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<p><b>印章文本检测模块模型：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">训练模型</a></td>
<td>98.21</td>
<td>84.341</td>
<td>2425.06</td>
<td>109</td>
<td>PP-OCRv4的服务端印章文本检测模型，精度更高，适合在较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">训练模型</a></td>
<td>96.47</td>
<td>10.5878</td>
<td>131.813</td>
<td>4.6</td>
<td>PP-OCRv4的移动端印章文本检测模型，效率更高，适合在端侧部署</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是自建的数据集，包含500张圆形印章图像。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p>
<p><b>文本图像矫正模块模型：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>MS-SSIM （%）</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/UVDoc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">训练模型</a></td>
<td>54.40</td>
<td>30.3 M</td>
<td>高精度文本图像矫正模型</td>
</tr>
</tbody>
</table>
<p><b>模型的精度指标测量自 <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet benchmark</a>。</b></p>
<p><b>文档图像方向分类模块模型：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td>
<td>99.06</td>
<td>3.84845</td>
<td>9.23735</td>
<td>7</td>
<td>基于PP-LCNet_x1_0的文档图像分类模型，含有四个类别，即0度，90度，180度，270度</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p></details>

<b></b>

## 2. 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验文档场景信息抽取v3产线的效果，也可以在本地使用  Python 体验文档场景信息抽取v3产线的效果。

### 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/182491/webUI)文档场景信息抽取v3产线的效果，用官方提供的 Demo 图片进行识别，例如：

<img src="https://github.com/user-attachments/assets/aa261b2b-b79c-4487-9323-dfcc43c3d581">

如果您对产线运行的效果满意，可以直接对产线进行集成部署，如果不满意，您也可以利用私有数据<b>对产线中的模型进行在线微调</b>。

### 2.2 本地体验
在本地使用文档场景信息抽取v3产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

几行代码即可完成产线的快速推理，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf)，以通用文档场景信息抽取v3产线为例：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="PP-ChatOCRv3-doc",
    llm_name="ernie-3.5",
    llm_params={"api_type": "qianfan", "ak": "", "sk": ""} # 请填入您的ak与sk，否则无法调用大模型
    # llm_params={"api_type": "aistudio", "access_token": ""} # 请填入您的access_token，否则无法调用大模型
    )

visual_result, visual_info = pipeline.visual_predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf")

for res in visual_result:
    res.save_to_img("./output")
    res.save_to_html('./output')
    res.save_to_xlsx('./output')

vector = pipeline.build_vector(visual_info=visual_info)

chat_result = pipeline.chat(
    key_list=["乙方", "手机号"],
    visual_info=visual_info,
    vector=vector,
    )
chat_result.print()
```
<b>注</b>：目前仅支持文心大模型，支持在[百度云千帆平台](https://console.bce.baidu.com/qianfan/ais/console/onlineService)或者[星河社区 AIStudio](https://aistudio.baidu.com/)上获取相关的 ak/sk(access_token)。如果使用百度云千帆平台，可以参考[AK和SK鉴权调用API流程](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Hlwerugt8) 获取ak/sk，如果使用星河社区 AIStudio，可以在[星河社区 AIStudio 访问令牌](https://aistudio.baidu.com/account/accessToken)中获取 access_token。

运行后，输出结果如下：

```
{'chat_res': {'乙方': '股份测试有限公司', '手机号': '19331729920'}, 'prompt': ''}
```

在上述 Python 脚本中，执行了如下四个步骤：

（1）调用 `create_pipeline` 方法实例化文档场景信息抽取v3产线对象，相关参数说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数类型</th>
<th>默认值</th>
<th>参数说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>str</td>
<td>无</td>
<td>产线名称或是产线配置文件路径，如为产线名称，则必须为 PaddleX 所支持的产线；</td>
</tr>
<tr>
<td><code>llm_name</code></td>
<td>str</td>
<td>"ernie-3.5"</td>
<td>大语言模型名称，目前支持<code>ernie-4.0</code>，<code>ernie-3.5</code>，更多模型支持中;</td>
</tr>
<tr>
<td><code>llm_params</code></td>
<td>dict</td>
<td><code>{}</code></td>
<td>LLM相关API配置；</td>
</tr>
<tr>
<td><code>device</code></td>
<td>str、None</td>
<td><code>None</code></td>
<td>运行设备（<code>None</code>为自动适配）,支持传入'cpu'，'gpu'或'gpu:0'等；</td>
</tr>
</tbody>
</table>
（2）调用文档场景信息抽取v3产线对象的 `visual_predict` 方法进行视觉推理预测，相关参数说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数类型</th>
<th>默认值</th>
<th>参数说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>Python Var</td>
<td>无</td>
<td>用于输入待预测数据，支持直接传入Python变量，如<code>numpy.ndarray</code>表示的图像数据；</td>
</tr>
<tr>
<td><code>input</code></td>
<td>str</td>
<td>无</td>
<td>用于输入待预测数据，支持传入待预测数据文件路径，如图像文件的本地路径：<code>/root/data/img.jpg</code>；</td>
</tr>
<tr>
<td><code>input</code></td>
<td>str</td>
<td>无</td>
<td>用于输入待预测数据，支持传入待预测数据文件url，如<code>https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf</code>；</td>
</tr>
<tr>
<td><code>input</code></td>
<td>str</td>
<td>无</td>
<td>用于输入待预测数据，支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code>；</td>
</tr>
<tr>
<td><code>input</code></td>
<td>dict</td>
<td>无</td>
<td>用于输入待预测数据，支持传入字典类型，字典的key需要与具体产线对应，如文档场景信息抽取v3产线为"img"，字典的val支持上述类型数据，如：<code>{"img": "/root/data1"}</code>；</td>
</tr>
<tr>
<td><code>input</code></td>
<td>list</td>
<td>无</td>
<td>用于输入待预测数据，支持传入列表，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>，<code>["/root/data1", "/root/data2"]</code>，<code>[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>；</td>
</tr>
<tr>
<td><code>use_doc_image_ori_cls_model</code></td>
<td>bool</td>
<td><code>True</code></td>
<td>是否使用方向分类模型；</td>
</tr>
<tr>
<td><code>use_doc_image_unwarp_model</code></td>
<td>bool</td>
<td><code>True</code></td>
<td>是否使用版面矫正产线；</td>
</tr>
<tr>
<td><code>use_seal_text_det_model</code></td>
<td>bool</td>
<td><code>True</code></td>
<td>是否使用弯曲文本检测产线；</td>
</tr>
</tbody>
</table>
（3）调用视觉推理预测结果对象的相关方法对视觉推理预测结果进行保存，具体方法如下：

<table>
<thead>
<tr>
<th>方法</th>
<th>参数</th>
<th>方法说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>save_to_img</code></td>
<td><code>save_path</code></td>
<td>将OCR预测结果、版面分析结果、表格识别结果保存为图片文件，参数<code>save_path</code>用于指定保存的路径；</td>
</tr>
<tr>
<td><code>save_to_html</code></td>
<td><code>save_path</code></td>
<td>将表格识别结果保存为html文件，参数<code>save_path</code>用于指定保存的路径；</td>
</tr>
<tr>
<td><code>save_to_xlsx</code></td>
<td><code>save_path</code></td>
<td>将表格识别结果保存为xlsx文件，参数<code>save_path</code>用于指定保存的路径；</td>
</tr>
</tbody>
</table>
（4）调用文档场景信息抽取v3产线对象的 `chat` 方法与大模型进行交互，相关参数说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数类型</th>
<th>默认值</th>
<th>参数说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>key_list</code></td>
<td>str</td>
<td>无</td>
<td>用于查询的关键字（query）；支持“，”或“,”作为分隔符的多个关键字组成的字符串，如“乙方，手机号”；</td>
</tr>
<tr>
<td><code>key_list</code></td>
<td>list</td>
<td>无</td>
<td>用于查询的关键字（query），支持<code>list</code>形式表示的一组关键字，其元素为<code>str</code>类型；</td>
</tr>
</tbody>
</table>
在执行上述 Python 脚本时，加载的是默认的文档场景信息抽取v3产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

```
paddlex --get_pipeline_config PP-ChatOCRv3-doc
```

执行后，文档场景信息抽取v3产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为 `./my_path` ）：

```
paddlex --get_pipeline_config PP-ChatOCRv3-doc --save_path ./my_path
```

获取配置文件后，您即可对文档场景信息抽取v3产线各项配置进行自定义：

```yaml
Pipeline:
  layout_model: RT-DETR-H_layout_3cls
  table_model: SLANet_plus
  text_det_model: PP-OCRv4_server_det
  text_rec_model: PP-OCRv4_server_rec
  seal_text_det_model: PP-OCRv4_server_seal_det
  doc_image_ori_cls_model: null
  doc_image_unwarp_model: null
  llm_name: "ernie-3.5"
  llm_params:
    api_type: qianfan
    ak:
    sk:
```

在上述配置中，您可以修改产线各模块加载的模型，也可以修改使用的大模型。各模块支持模型列表请参考模块文档，大模型支持列表为：ernie-4.0、ernie-3.5、ernie-3.5-8k、ernie-lite、ernie-tiny-8k、ernie-speed、ernie-speed-128k、ernie-char-8k。

修改后，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可应用配置。

例如，若您的配置文件保存在 `./my_path/PP-ChatOCRv3-doc.yaml` ，则只需执行：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="./my_path/PP-ChatOCRv3-doc.yaml",
    llm_name="ernie-3.5",
    llm_params={"api_type": "qianfan", "ak": "", "sk": ""} # 请填入您的ak与sk，否则无法调用大模型
    # llm_params={"api_type": "aistudio", "access_token": ""} # 请填入您的access_token，否则无法调用大模型
    )

visual_result, visual_info = pipeline.visual_predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf")

for res in visual_result:
    res.save_to_img("./output")
    res.save_to_html('./output')
    res.save_to_xlsx('./output')

vector = pipeline.build_vector(visual_info=visual_info)

chat_result = pipeline.chat(
    key_list=["乙方", "手机号"],
    visual_info=visual_info,
    vector=vector,
    )
chat_result.print()
```

## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2 本地体验](#22-本地体验)中的示例代码。

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
<li><b><code>analyzeImages</code></b></li>
</ul>
<p>使用计算机视觉模型对图像进行分析，获得OCR、表格识别结果等，并提取图像中的关键信息。</p>
<p><code>POST /chatocr-visual</code></p>
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
<tr>
<td><code>useImgOrientationCls</code></td>
<td><code>boolean</code></td>
<td>是否启用文档图像方向分类功能。默认启用该功能。</td>
<td>否</td>
</tr>
<tr>
<td><code>useImgUnwarping</code></td>
<td><code>boolean</code></td>
<td>是否启用文本图像矫正功能。默认启用该功能。</td>
<td>否</td>
</tr>
<tr>
<td><code>useSealTextDet</code></td>
<td><code>boolean</code></td>
<td>是否启用印章文本检测功能。默认启用该功能。</td>
<td>否</td>
</tr>
<tr>
<td><code>inferenceParams</code></td>
<td><code>object</code></td>
<td>推理参数。</td>
<td>否</td>
</tr>
</tbody>
</table>
<p><code>inferenceParams</code>的属性如下：</p>
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
<td><code>maxLongSide</code></td>
<td><code>integer</code></td>
<td>推理时，若文本检测模型的输入图像较长边的长度大于<code>maxLongSide</code>，则将对图像进行缩放，使其较长边的长度等于<code>maxLongSide</code>。</td>
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
<td><code>visualResults</code></td>
<td><code>array</code></td>
<td>使用计算机视觉模型得到的分析结果。数组长度为1（对于图像输入）或文档页数与10中的较小者（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中每一页的处理结果。</td>
</tr>
<tr>
<td><code>visualInfo</code></td>
<td><code>object</code></td>
<td>图像中的关键信息，可用作其他操作的输入。</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>输入数据信息。</td>
</tr>
</tbody>
</table>
<p><code>visualResults</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>texts</code></td>
<td><code>array</code></td>
<td>文本位置、内容和得分。</td>
</tr>
<tr>
<td><code>tables</code></td>
<td><code>array</code></td>
<td>表格位置和内容。</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code></td>
<td>输入图像。图像为JPEG格式，使用Base64编码。</td>
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
<p><code>texts</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>poly</code></td>
<td><code>array</code></td>
<td>文本位置。数组中元素依次为包围文本的多边形的顶点坐标。</td>
</tr>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>文本内容。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>文本识别得分。</td>
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
</table>
<ul>
<li><b><code>buildVectorStore</code></b></li>
</ul>
<p>构建向量数据库。</p>
<p><code>POST /chatocr-vector</code></p>
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
<td><code>visualInfo</code></td>
<td><code>object</code></td>
<td>图像中的关键信息。由<code>analyzeImages</code>操作提供。</td>
<td>是</td>
</tr>
<tr>
<td><code>minChars</code></td>
<td><code>integer</code></td>
<td>启用向量数据库的最小数据长度。</td>
<td>否</td>
</tr>
<tr>
<td><code>llmRequestInterval</code></td>
<td><code>number</code></td>
<td>调用大语言模型API的间隔时间。</td>
<td>否</td>
</tr>
<tr>
<td><code>llmName</code></td>
<td><code>string</code></td>
<td>大语言模型名称。</td>
<td>否</td>
</tr>
<tr>
<td><code>llmParams</code></td>
<td><code>object</code></td>
<td>大语言模型API参数。</td>
<td>否</td>
</tr>
</tbody>
</table>
<p>当前，<code>llmParams</code> 可以采用如下形式之一：</p>
<pre><code class="language-json">{
&quot;apiType&quot;: &quot;qianfan&quot;,
&quot;apiKey&quot;: &quot;{千帆平台API key}&quot;,
&quot;secretKey&quot;: &quot;{千帆平台secret key}&quot;
}
</code></pre>
<pre><code class="language-json">{
&quot;apiType&quot;: &quot;aistudio&quot;,
&quot;accessToken&quot;: &quot;{星河社区access token}&quot;
}
</code></pre>
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
<td><code>vectorStore</code></td>
<td><code>string</code></td>
<td>向量数据库序列化结果，可用作其他操作的输入。</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>retrieveKnowledge</code></b></li>
</ul>
<p>进行知识检索。</p>
<p><code>POST /chatocr-retrieval</code></p>
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
<td><code>keys</code></td>
<td><code>array</code></td>
<td>关键词列表。</td>
<td>是</td>
</tr>
<tr>
<td><code>vectorStore</code></td>
<td><code>string</code></td>
<td>向量数据库序列化结果。由<code>buildVectorStore</code>操作提供。</td>
<td>是</td>
</tr>
<tr>
<td><code>llmName</code></td>
<td><code>string</code></td>
<td>大语言模型名称。</td>
<td>否</td>
</tr>
<tr>
<td><code>llmParams</code></td>
<td><code>object</code></td>
<td>大语言模型API参数。</td>
<td>否</td>
</tr>
</tbody>
</table>
<p>当前，<code>llmParams</code> 可以采用如下形式之一：</p>
<pre><code class="language-json">{
&quot;apiType&quot;: &quot;qianfan&quot;,
&quot;apiKey&quot;: &quot;{千帆平台API key}&quot;,
&quot;secretKey&quot;: &quot;{千帆平台secret key}&quot;
}
</code></pre>
<pre><code class="language-json">{
&quot;apiType&quot;: &quot;aistudio&quot;,
&quot;accessToken&quot;: &quot;{星河社区access token}&quot;
}
</code></pre>
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
<td><code>retrievalResult</code></td>
<td><code>string</code></td>
<td>知识检索结果，可用作其他操作的输入。</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>chat</code></b></li>
</ul>
<p>与大语言模型交互，利用大语言模型提炼关键信息。</p>
<p><code>POST /chatocr-chat</code></p>
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
<td><code>keys</code></td>
<td><code>array</code></td>
<td>关键词列表。</td>
<td>是</td>
</tr>
<tr>
<td><code>visualInfo</code></td>
<td><code>object</code></td>
<td>图像中的关键信息。由<code>analyzeImages</code>操作提供。</td>
<td>是</td>
</tr>
<tr>
<td><code>vectorStore</code></td>
<td><code>string</code></td>
<td>向量数据库序列化结果。由<code>buildVectorStore</code>操作提供。</td>
<td>否</td>
</tr>
<tr>
<td><code>retrievalResult</code></td>
<td><code>string</code></td>
<td>知识检索结果。由<code>retrieveKnowledge</code>操作提供。</td>
<td>否</td>
</tr>
<tr>
<td><code>taskDescription</code></td>
<td><code>string</code></td>
<td>提示词任务。</td>
<td>否</td>
</tr>
<tr>
<td><code>rules</code></td>
<td><code>string</code></td>
<td>提示词规则。用于自定义信息抽取规则，例如规范输出格式。</td>
<td>否</td>
</tr>
<tr>
<td><code>fewShot</code></td>
<td><code>string</code></td>
<td>提示词示例。</td>
<td>否</td>
</tr>
<tr>
<td><code>llmName</code></td>
<td><code>string</code></td>
<td>大语言模型名称。</td>
<td>否</td>
</tr>
<tr>
<td><code>llmParams</code></td>
<td><code>object</code></td>
<td>大语言模型API参数。</td>
<td>否</td>
</tr>
<tr>
<td><code>returnPrompts</code></td>
<td><code>boolean</code></td>
<td>是否返回使用的提示词。默认禁用。</td>
<td>否</td>
</tr>
</tbody>
</table>
<p>当前，<code>llmParams</code> 可以采用如下形式之一：</p>
<pre><code class="language-json">{
&quot;apiType&quot;: &quot;qianfan&quot;,
&quot;apiKey&quot;: &quot;{千帆平台API key}&quot;,
&quot;secretKey&quot;: &quot;{千帆平台secret key}&quot;
}
</code></pre>
<pre><code class="language-json">{
&quot;apiType&quot;: &quot;aistudio&quot;,
&quot;accessToken&quot;: &quot;{星河社区access token}&quot;
}
</code></pre>
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
<td><code>chatResult</code></td>
<td><code>object</code></td>
<td>关键信息抽取结果。</td>
</tr>
<tr>
<td><code>prompts</code></td>
<td><code>object</code></td>
<td>使用的提示词。</td>
</tr>
</tbody>
</table>
<p><code>prompts</code>的属性如下：</p>
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
<td><code>ocr</code></td>
<td><code>array</code></td>
<td>OCR提示词。</td>
</tr>
<tr>
<td><code>table</code></td>
<td><code>array</code></td>
<td>表格提示词。</td>
</tr>
<tr>
<td><code>html</code></td>
<td><code>array</code></td>
<td>HTML提示词。</td>
</tr>
</tbody>
</table></details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import pprint
import sys

import requests


API_BASE_URL = &quot;http://0.0.0.0:8080&quot;
API_KEY = &quot;{千帆平台API key}&quot;
SECRET_KEY = &quot;{千帆平台secret key}&quot;
LLM_NAME = &quot;ernie-3.5&quot;
LLM_PARAMS = {
    &quot;apiType&quot;: &quot;qianfan&quot;,
    &quot;apiKey&quot;: API_KEY,
    &quot;secretKey&quot;: SECRET_KEY,
}

file_path = &quot;./demo.jpg&quot;
keys = [&quot;电话&quot;]

with open(file_path, &quot;rb&quot;) as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode(&quot;ascii&quot;)

payload = {
    &quot;file&quot;: file_data,
    &quot;fileType&quot;: 1,
    &quot;useImgOrientationCls&quot;: True,
    &quot;useImgUnwarping&quot;: True,
    &quot;useSealTextDet&quot;: True,
}
resp_visual = requests.post(url=f&quot;{API_BASE_URL}/chatocr-visual&quot;, json=payload)
if resp_visual.status_code != 200:
    print(
        f&quot;Request to chatocr-visual failed with status code {resp_visual.status_code}.&quot;,
        file=sys.stderr,
    )
    pprint.pp(resp_visual.json())
    sys.exit(1)
result_visual = resp_visual.json()[&quot;result&quot;]

for i, res in enumerate(result_visual[&quot;visualResults&quot;]):
    print(&quot;Texts:&quot;)
    pprint.pp(res[&quot;texts&quot;])
    print(&quot;Tables:&quot;)
    pprint.pp(res[&quot;tables&quot;])
    layout_img_path = f&quot;layout_{i}.jpg&quot;
    with open(layout_img_path, &quot;wb&quot;) as f:
        f.write(base64.b64decode(res[&quot;layoutImage&quot;]))
    ocr_img_path = f&quot;ocr_{i}.jpg&quot;
    with open(ocr_img_path, &quot;wb&quot;) as f:
        f.write(base64.b64decode(res[&quot;ocrImage&quot;]))
    print(f&quot;Output images saved at {layout_img_path} and {ocr_img_path}&quot;)

payload = {
    &quot;visualInfo&quot;: result_visual[&quot;visualInfo&quot;],
    &quot;minChars&quot;: 200,
    &quot;llmRequestInterval&quot;: 1000,
    &quot;llmName&quot;: LLM_NAME,
    &quot;llmParams&quot;: LLM_PARAMS,
}
resp_vector = requests.post(url=f&quot;{API_BASE_URL}/chatocr-vector&quot;, json=payload)
if resp_vector.status_code != 200:
    print(
        f&quot;Request to chatocr-vector failed with status code {resp_vector.status_code}.&quot;,
        file=sys.stderr,
    )
    pprint.pp(resp_vector.json())
    sys.exit(1)
result_vector = resp_vector.json()[&quot;result&quot;]

payload = {
    &quot;keys&quot;: keys,
    &quot;vectorStore&quot;: result_vector[&quot;vectorStore&quot;],
    &quot;llmName&quot;: LLM_NAME,
    &quot;llmParams&quot;: LLM_PARAMS,
}
resp_retrieval = requests.post(url=f&quot;{API_BASE_URL}/chatocr-retrieval&quot;, json=payload)
if resp_retrieval.status_code != 200:
    print(
        f&quot;Request to chatocr-retrieval failed with status code {resp_retrieval.status_code}.&quot;,
        file=sys.stderr,
    )
    pprint.pp(resp_retrieval.json())
    sys.exit(1)
result_retrieval = resp_retrieval.json()[&quot;result&quot;]

payload = {
    &quot;keys&quot;: keys,
    &quot;visualInfo&quot;: result_visual[&quot;visualInfo&quot;],
    &quot;vectorStore&quot;: result_vector[&quot;vectorStore&quot;],
    &quot;retrievalResult&quot;: result_retrieval[&quot;retrievalResult&quot;],
    &quot;taskDescription&quot;: &quot;&quot;,
    &quot;rules&quot;: &quot;&quot;,
    &quot;fewShot&quot;: &quot;&quot;,
    &quot;llmName&quot;: LLM_NAME,
    &quot;llmParams&quot;: LLM_PARAMS,
    &quot;returnPrompts&quot;: True,
}
resp_chat = requests.post(url=f&quot;{API_BASE_URL}/chatocr-chat&quot;, json=payload)
if resp_chat.status_code != 200:
    print(
        f&quot;Request to chatocr-chat failed with status code {resp_chat.status_code}.&quot;,
        file=sys.stderr,
    )
    pprint.pp(resp_chat.json())
    sys.exit(1)
result_chat = resp_chat.json()[&quot;result&quot;]
print(&quot;\nPrompts:&quot;)
pprint.pp(result_chat[&quot;prompts&quot;])
print(&quot;Final result:&quot;)
print(result_chat[&quot;chatResult&quot;])
</code></pre>


<b>注</b>：请在 `API_KEY`、`SECRET_KEY` 处填入您的 API key 和 secret key。</details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果通用文档场景信息抽取v3产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升通用表格识别产线的在您的场景中的识别效果。

### 4.1 模型微调
由于通用文档场景信息抽取v3产线包含六个模块，模型产线的效果不及预期可能来自于其中任何一个模块（文本图像矫正模块暂不支持二次开发）。

您可以对识别效果差的图片进行分析，参考如下规则进行分析和模型微调：

* 检测到的表格结构错误（如行列识别错误、单元格位置错误），那么可能是表格结构识别模块存在不足，您需要参考[表格结构识别模块开发教程](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md)中的<b>二次开发</b>章节，使用您的私有数据集对表格结构识别模型进行微调。
* 版面中存在定位错误（例如对表格、印章的位置识别错误），那么可能是版面区域定位模块存在不足，您需要参考[版面区域检测模块开发教程](../../../module_usage/tutorials/ocr_modules/layout_detection.md)中的<b>二次开发</b>章节，使用您的私有数据集对版面区域定位模型进行微调。
* 有较多的文本未被检测出来（即文本漏检现象），那么可能是文本检测模型存在不足，您需要参考[文本检测模块开发教程](../../../module_usage/tutorials/ocr_modules/text_detection.md)中的<b>二次开发</b>章节，使用您的私有数据集对文本检测模型进行微调。
* 已检测到的文本中出现较多的识别错误（即识别出的文本内容与实际文本内容不符），这表明文本识别模型需要进一步改进，您需要参考[文本识别模块开发教程](../../../module_usage/tutorials/ocr_modules/text_recognition.md)中的<b>二次开发</b>章节对文本识别模型进行微调。
* 已检测到的印章文本出现较多的识别错误，这表明印章文本检测模块模型需要进一步改进，您需要参考[印章文本检测模块开发教程](../../../module_usage/tutorials/ocr_modules/)中的<b>二次开发</b>章节对印章文本检测模型进行微调。
* 含文字区域的文档或证件的方向存在较多的识别错误，这表明文档图像方向分类模型需要进一步改进，您需要参考[文档图像方向分类模块开发教程](../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md)中的<b>二次开发</b>章节对文档图像方向分类模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```
......
Pipeline:
  layout_model: RT-DETR-H_layout_3cls  #可修改为微调后模型的本地路径
  table_model: SLANet_plus  #可修改为微调后模型的本地路径
  text_det_model: PP-OCRv4_server_det  #可修改为微调后模型的本地路径
  text_rec_model: PP-OCRv4_server_rec  #可修改为微调后模型的本地路径
  seal_text_det_model: PP-OCRv4_server_seal_det  #可修改为微调后模型的本地路径
  doc_image_ori_cls_model: null   #可修改为微调后模型的本地路径
  doc_image_unwarp_model: null   #可修改为微调后模型的本地路径
......
```

随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU 和寒武纪 MLU 等多种主流硬件设备，<b>仅需设置 `device` 参数</b>即可完成不同硬件之间的无缝切换。

例如，使用文档场景信息抽取v3产线时，将运行设备从英伟达 GPU 更改为昇腾 NPU，仅需将脚本中的 `device` 修改为 npu 即可：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(
    pipeline="PP-ChatOCRv3-doc",
    llm_name="ernie-3.5",
    llm_params={"api_type": "qianfan", "ak": "", "sk": ""},
    device="npu:0" # gpu:0 --> npu:0
    )
```
若您想在更多种类的硬件上使用通用文档场景信息抽取产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
