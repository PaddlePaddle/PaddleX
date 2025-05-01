---
comments: true
---

# PaddleX 3.0 通用表格识别v2产线（PP-TableMagic）实践教程

PaddleX 提供了丰富的模型产线，模型产线由一个或多个模型组合实现，每个模型产线都能够解决特定的场景任务问题。PaddleX 所提供的模型产线均支持快速体验，如果效果不及预期，也同样支持使用私有数据微调模型，并且 PaddleX 提供了 Python API，方便将产线集成到个人项目中。在使用之前，您首先需要安装 PaddleX， 安装方式请参考 [PaddleX本地安装教程](../installation/installation.md)。请注意，本文档是通用表格识别v2产线（PP-TableMagic）的实践教程，提供一些实践经验，并非该产线的完整使用教程，完整的使用教程请参考 [PaddleX 通用表格识别v2产线](../pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.md)。


## 1. 选择模型产线

表格作为结构化数据的最关键载体，在金融、科研、信息统计、文档分析等领域或场景承担着复杂信息整合的核心作用。基于此，表格识别技术旨在从表格图像或文档中识别表格结构与内容，并将其转为结构化的表格形式（例如 HTML、Excel 等）。在“信息爆炸”的现代社会，怎样从海量的非结构化表格中获取到想要的数据，成为了大量用户最为关心的问题之一。除此之外，在大语言模型、多模态文档理解技术高速发展的背景下，高质量的结构化表格已成为模型训练与知识库构建的关键需求。因此，表格识别技术获得了越来越多的关注。通用表格识别v2产线（PP-TableMagic）是表格识别任务的新一代解决方案，通过采用多模型串联组网实现了优秀的端到端表格识别性能，并且为不同使用场景提供了大量可以定制化的选项或模型微调方案.除此之外，通用表格识别v2产线同样支持使用端到端表格结构识别模型（例如 SLANet、SLANet_plus 等），并且支持有线表、无线表独立配置表格识别方式，开发者可以自由选取和组合最佳的表格识别方案。

首先，需要根据任务场景，选择对应的 PaddleX 产线，本节以通用表格识别v2产线的结果后处理优化和模型微调为例，希望对图像中表格的结构和内容进行准确识别，对应 PaddleX 的表格结构识别模块、表格分类模块、表格单元格检测模块、版面区域检测模块和，可以在通用表格识别v2产线中使用。如果无法确定任务和产线的对应关系，您可以在 PaddleX 支持的[模型产线列表](../support_list/pipelines_list.md)中了解相关产线的能力介绍。


## 2. 模型列表

<b>通用</b><b>表格识别v2</b><b>产线中包含必选的表格结构识别模块、表格分类模块、表格单元格定位模块、文本检测模块和文本识别模块，以及可选的版面区域检测模块、文档图像方向分类模块和文本图像矫正模块</b>。

<b>如果您更注重模型的精度，请选择精度较高的模型；如果您更在意模型的推理速度，请选择推理速度较快的模型；如果您关注模型的存储大小，请选择存储体积较小的模型。</b>
<details><summary> 👉模型列表详情</summary>
<p><b>表格结构识别模块模型：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>精度（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wired_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">训练模型</a></td>
<td rowspan="2">69.65</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">351M</td>
<td rowspan="2">SLANeXt 系列是百度飞桨视觉团队自研的新一代表格结构识别模型。相较于 SLANet 和 SLANet_plus，SLANeXt 专注于对表格结构进行识别，并且对有线表格(wired)和无线表格(wireless)的识别分别训练了专用的权重，对各类型表格的识别能力都得到了明显提高，特别是对有线表格的识别能力得到了大幅提升。</td>
</tr>
<tr>
<td>SLANeXt_wireless</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wireless_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wireless_pretrained.pdparams">训练模型</a></td>
</tr>
</table>

<p><b>表格分类模块模型：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top1 Acc(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CLIP_vit_base_patch16_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">训练模型</a></td>
<td>94.2</td>
<td>2.35 / 0.47</td>
<td>4.03 / 1.35</td>
<td>6.6M</td>
</tr>
</table>

<p><b>表格单元格检测模块模型：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wired_table_cell_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">训练模型</a></td>
<td rowspan="2">82.7</td>
<td rowspan="2">35.00 / 10.45</td>
<td rowspan="2">495.51 / 495.51</td>
<td rowspan="2">124M</td>
<td rowspan="2">RT-DETR 是第一个实时的端到端目标检测模型。百度飞桨视觉团队基于 RT-DETR-L 作为基础模型，在自建表格单元格检测数据集上完成预训练，实现了对有线表格、无线表格均有较好性能的表格单元格检测。
</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wireless_table_cell_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams">训练模型</a></td>
</tr>
</table>

<p><b>文本检测模块模型：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">训练模型</a></td>
<td>82.69</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>109</td>
<td>PP-OCRv4 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">训练模型</a></td>
<td>77.79</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>PP-OCRv4 的移动端文本检测模型，效率更高，适合在端侧设备部署</td>
</tr>
</tbody>
</table>

<p><b>文本识别模块模型：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">训练模型</a></td>
<td>81.53</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>74.7 M</td>
<td>PP-OCRv4_server_rec_doc是在PP-OCRv4_server_rec的基础上，在更多中文文档数据和PP-OCR训练数据的混合数据训练而成，增加了部分繁体字、日文、特殊字符的识别能力，可支持识别的字符为1.5万+，除文档相关的文字识别能力提升外，也同时提升了通用文字的识别能力</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>78.74</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td>PP-OCRv4的轻量级识别模型，推理效率高，可以部署在包含端侧设备的多种硬件设备中</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td>
<td>80.61 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
<td>PP-OCRv4的服务器端模型，推理精度高，可以部署在多种不同的服务器上</td>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>6.8 M</td>
<td>基于PP-OCRv4识别模型训练得到的超轻量英文识别模型，支持英文、数字识别</td>
</tr>
</table>

> ❗ 以上列出的是文本识别模块重点支持的<b>4个核心模型</b>，该模块总共支持<b>18个全量模型</b>，包含多个多语言文本识别模型，完整的模型列表如下：

<details><summary> 👉模型列表详情</summary>

* <b>中文识别模型</b>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">训练模型</a></td>
<td>81.53</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>74.7 M</td>
<td>PP-OCRv4_server_rec_doc是在PP-OCRv4_server_rec的基础上，在更多中文文档数据和PP-OCR训练数据的混合数据训练而成，增加了部分繁体字、日文、特殊字符的识别能力，可支持识别的字符为1.5万+，除文档相关的文字识别能力提升外，也同时提升了通用文字的识别能力</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>78.74</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td>PP-OCRv4的轻量级识别模型，推理效率高，可以部署在包含端侧设备的多种硬件设备中</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td>
<td>80.61 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
<td>PP-OCRv4的服务器端模型，推理精度高，可以部署在多种不同的服务器上</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>72.96</td>
<td>5.87 / 1.19</td>
<td>9.07 / 4.28</td>
<td>9.2 M</td>
<td>PP-OCRv3的轻量级识别模型，推理效率高，可以部署在包含端侧设备的多种硬件设备中</td>
</tr>
</table>

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">训练模型</a></td>
<td>68.81</td>
<td>8.08 / 2.74</td>
<td>50.17 / 42.50</td>
<td>73.9 M</td>
<td rowspan="1">
SVTRv2 是一种由复旦大学视觉与学习实验室（FVL）的OpenOCR团队研发的服务端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，A榜端到端识别精度相比PP-OCRv4提升6%。
</td>
</tr>
</table>

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">训练模型</a></td>
<td>65.07</td>
<td>5.93 / 1.62</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
<td rowspan="1">    RepSVTR 文本识别模型是一种基于SVTRv2 的移动端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，B榜端到端识别精度相比PP-OCRv4提升2.5%，推理速度持平。</td>
</tr>
</table>

* <b>英文识别模型</b>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td> 70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>6.8 M</td>
<td>基于PP-OCRv4识别模型训练得到的超轻量英文识别模型，支持英文、数字识别</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>70.69</td>
<td>5.44 / 0.75</td>
<td>8.65 / 5.57</td>
<td>7.8 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量英文识别模型，支持英文、数字识别</td>
</tr>
</table>


* <b>多语言识别模型</b>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
korean_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>60.21</td>
<td>5.40 / 0.97</td>
<td>9.11 / 4.05</td>
<td>8.6 M</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量韩文识别模型，支持韩文、数字识别</td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
japan_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/japan_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>45.69</td>
<td>5.70 / 1.02</td>
<td>8.48 / 4.07</td>
<td>8.8 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量日文识别模型，支持日文、数字识别</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/chinese_cht_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>82.06</td>
<td>5.90 / 1.28</td>
<td>9.28 / 4.34</td>
<td>9.7 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量繁体中文识别模型，支持繁体中文、数字识别</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
te_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/te_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>95.88</td>
<td>5.42 / 0.82</td>
<td>8.10 / 6.91</td>
<td>7.8 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量泰卢固文识别模型，支持泰卢固文、数字识别</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ka_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ka_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>96.96</td>
<td>5.25 / 0.79</td>
<td>9.09 / 3.86</td>
<td>8.0 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量卡纳达文识别模型，支持卡纳达文、数字识别</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ta_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>76.83</td>
<td>5.23 / 0.75</td>
<td>10.13 / 4.30</td>
<td>8.0 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量泰米尔文识别模型，支持泰米尔文、数字识别</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
latin_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>76.93</td>
<td>5.20 / 0.79</td>
<td>8.83 / 7.15</td>
<td>7.8 M</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量拉丁文识别模型，支持拉丁文、数字识别</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
arabic_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>73.55</td>
<td>5.35 / 0.79</td>
<td>8.80 / 4.56</td>
<td>7.8 M</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量阿拉伯字母识别模型，支持阿拉伯字母、数字识别</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/cyrillic_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>94.28</td>
<td>5.23 / 0.76</td>
<td>8.89 / 3.88</td>
<td>7.9 M  </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量斯拉夫字母识别模型，支持斯拉夫字母、数字识别</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/devanagari_PP-OCRv3_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>96.44</td>
<td>5.22 / 0.79</td>
<td>8.56 / 4.06</td>
<td>7.9 M</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量梵文字母识别模型，支持梵文字母、数字识别</td>
</tr>
</table>
</details>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">训练模型</a></td>
<td>90.4</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / -</td>
<td>123.76 M</td>
<td>基于RT-DETR-L在包含中英文论文、杂志、合同、书本、试卷和研报等场景的自建数据集训练的高精度版面区域定位模型</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">训练模型</a></td>
<td>75.2</td>
<td>13.3259 / 4.8685</td>
<td>44.0680 / 44.0680</td>
<td>22.578</td>
<td>基于PicoDet-L在包含中英文论文、杂志、合同、书本、试卷和研报等场景的自建数据集训练的精度效率平衡的版面区域定位模型</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">训练模型</a></td>
<td>70.9</td>
<td>8.3008 / 2.3794</td>
<td>10.0623 / 9.9296</td>
<td>4.834</td>
<td>基于PicoDet-S在中英文论文、杂志、合同、书本、试卷和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
</tbody>
</table>

> ❗ 以上列出的是版面检测模块重点支持的<b>3个核心模型</b>，该模块总共支持<b>11个全量模型</b>，包含多个预定义了不同类别的模型，完整的模型列表如下：
<details><summary> 👉模型列表详情</summary>
* <b>表格版面检测模型</b>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_table_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">训练模型</a></td>
<td>97.5</td>
<td>8.02 / 3.09</td>
<td>23.70 / 20.41</td>
<td>7.4 M</td>
<td>基于PicoDet-1x在自建数据集训练的高效率版面区域定位模型，可定位表格这1类区域</td>
</tr>
</tbody></table>

* <b>3类版面检测模型，包含表格、图像、印章</b>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>88.2</td>
<td>8.99 / 2.22</td>
<td>16.11 / 8.73</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>89.0</td>
<td>13.05 / 4.50</td>
<td>41.30 / 41.30</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的效率精度均衡版面区域定位模型</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>95.8</td>
<td>114.93 / 27.71</td>
<td>947.56 / 947.56</td>
<td>470.1</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型</td>
</tr>
</tbody></table>

* <b>5类英文文档区域检测模型，包含文字、标题、表格、图片以及列表</b>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">训练模型</a></td>
<td>97.8</td>
<td>9.03 / 3.10</td>
<td>25.82 / 20.70</td>
<td>7.4</td>
<td>基于PicoDet-1x在PubLayNet数据集训练的高效率英文文档版面区域定位模型</td>
</tr>
</tbody></table>
</b>

* <b>17类区域检测模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</b>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>87.4</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>89.0</td>
<td>13.50 / 4.69</td>
<td>43.32 / 43.32</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的效率精度均衡版面区域定位模型</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>98.3</td>
<td>115.29 / 104.09</td>
<td>995.27 / 995.27</td>
<td>470.2</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型</td>
</tr>
</tbody>
</table>
<p><b>文本图像矫正模块模型（可选）：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>MS-SSIM （%）</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">训练模型</a></td>
<td>54.40</td>
<td>30.3 M</td>
<td>高精度文本图像矫正模型</td>
</tr>
</tbody>
</table>
</details>
<p><b>文档图像方向分类模块模型（可选）：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>基于PP-LCNet_x1_0的文档图像分类模型，含有四个类别，即0度，90度，180度，270度</td>
</tr>
</tbody>
</table>

**测试环境说明：**

- **性能测试环境**
  - **测试数据集**
    - 文档图像方向分类模型：PaddleX 自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。
    - 版面区域检测模型：PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志、合同、书本、试卷和研报等常见的 500 张文档类型图片。
    - 表格版面检测模型：PaddleOCR 自建的版面表格区域检测数据集，包含中英文 7835 张带有表格的论文文档类型图片。
    - 3类版面检测模型：PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志和研报等常见的 1154 张文档类型图片。
    - 5类英文文档区域检测模型：[PubLayNet](https://developer.ibm.com/exchanges/data/all/publaynet) 的评估数据集，包含英文>文档的 11245 张文图片。
    - 17类区域检测模型：PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志和研报等常见的 892 张文档类型图片。
    - 表格结构识别模型：PaddleX 内部自建高难度中文表格识别数据集。
    - 表格单元格检测模型：PaddleX 内部自建评测集。
    - 表格分类模型：PaddleX 内部自建评测集。
    - 文本检测模型：PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。
    - 中文识别模型： PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。
    - ch_SVTRv2_rec：<a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>A榜评估集。
    - ch_RepSVTR_rec：<a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>B榜评估集。
    - 英文识别模型：PaddleX 自建的英文数据集。
    - 多语言识别模型：PaddleX 自建的多语种数据集。
  - **硬件配置**：
    - GPU：NVIDIA Tesla T4
    - CPU：Intel Xeon Gold 6271C @ 2.60GHz
    - 其他环境：Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2

- **推理模式说明**

| 模式        | GPU配置                          | CPU配置          | 加速技术组合                                |
|-------------|----------------------------------|------------------|---------------------------------------------|
| 常规模式    | FP32精度 / 无TRT加速             | FP32精度 / 8线程       | PaddleInference                             |
| 高性能模式  | 选择先验精度类型和加速策略的最优组合         | FP32精度 / 8线程       | 选择先验最优后端（Paddle/OpenVINO/TRT等） |

</details>


## 3. 快速体验
### 3.1 本地体验 ———— 命令行方式

一行命令即可快速体验表格识别产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition_v2.jpg)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline table_recognition_v2 \
        --use_doc_orientation_classify=False \
        --use_doc_unwarping=False \
        --input table_recognition_v2.jpg \
        --save_path ./output \
        --device gpu:0
```

相关的参数说明可以参考[2.2 Python脚本方式集成](#22-python脚本方式集成)中的参数说明。

<details><summary>👉 <b>运行后，得到的结果为：（点击展开）</b></summary>

```
{'res': {'input_path': 'table_recognition_v2.jpg', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True, 'use_ocr_model': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 8, 'label': 'table', 'score': 0.86655592918396, 'coordinate': [0.0125130415, 0.41920784, 1281.3737, 585.3884]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[   9,   21],
        ...,
        [   9,   59]],

       ...,

       [[1046,  536],
        ...,
        [1046,  573]]], dtype=int16), 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0, 'rec_texts': ['部门', '报销人', '报销事由', '批准人：', '单据', '张', '合计金额', '元', '车费票', '其', '火车费票', '飞机票', '中', '旅住宿费', '其他', '补贴'], 'rec_scores': array([0.99958128, ..., 0.99317062]), 'rec_polys': array([[[   9,   21],
        ...,
        [   9,   59]],

       ...,

       [[1046,  536],
        ...,
        [1046,  573]]], dtype=int16), 'rec_boxes': array([[   9, ...,   59],
       ...,
       [1046, ...,  573]], dtype=int16)}, 'table_res_list': [{'cell_box_list': [array([ 0.13052222, ..., 73.08310249]), array([104.43082511, ...,  73.27777413]), array([319.39041221, ...,  73.30439308]), array([424.2436837 , ...,  73.44736794]), array([580.75836265, ...,  73.24003914]), array([723.04370201, ...,  73.22717598]), array([984.67315757, ...,  73.20420387]), array([1.25130415e-02, ..., 5.85419208e+02]), array([984.37072837, ..., 137.02281502]), array([984.26586998, ..., 201.22290352]), array([984.24017417, ..., 585.30775765]), array([1039.90606773, ...,  265.44664314]), array([1039.69549644, ...,  329.30540779]), array([1039.66546714, ...,  393.57319954]), array([1039.5122689 , ...,  457.74644783]), array([1039.55535972, ...,  521.73030403]), array([1039.58612144, ...,  585.09468392])], 'pred_html': '<html><body><table><tbody><tr><td>部门</td><td></td><td>报销人</td><td></td><td>报销事由</td><td></td><td colspan="2">批准人：</td></tr><tr><td colspan="6" rowspan="8"></td><td colspan="2">单据 张</td></tr><tr><td colspan="2">合计金额 元</td></tr><tr><td rowspan="6">其 中</td><td>车费票</td></tr><tr><td>火车费票</td></tr><tr><td>飞机票</td></tr><tr><td>旅住宿费</td></tr><tr><td>其他</td></tr><tr><td>补贴</td></tr></tbody></table></body></html>', 'table_ocr_pred': {'rec_polys': array([[[   9,   21],
        ...,
        [   9,   59]],

       ...,

       [[1046,  536],
        ...,
        [1046,  573]]], dtype=int16), 'rec_texts': ['部门', '报销人', '报销事由', '批准人：', '单据', '张', '合计金额', '元', '车费票', '其', '火车费票', '飞机票', '中', '旅住宿费', '其他', '补贴'], 'rec_scores': array([0.99958128, ..., 0.99317062]), 'rec_boxes': array([[   9, ...,   59],
       ...,
       [1046, ...,  573]], dtype=int16)}}]}}
```

</details>

运行结果参数说明可以参考[2.2 Python脚本方式集成](#22-python脚本方式集成)中的结果解释。

可视化结果保存在`save_path`下，其中表格识别的可视化结果如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition_v2/02.jpg">

### 3.2 本地体验 ———— Python 方式

通过上述命令行能够快速体验查看效果，在项目中往往需要代码集成，您可以通过如下几行代码完成产线的快速推理：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="table_recognition_v2")

output = pipeline.predict(
    input="table_recognition_v2.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)

for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_xlsx("./output/")
    res.save_to_html("./output/")
    res.save_to_json("./output/")
```

输出打印的结果与上述命令行体验方式一致。在`output`目录中，保存了版面区域检测可视化结果、OCR 可视化结果、表格单元格检测可视化结果、json 格式保存的结果、HTML 格式保存的预测表格和Excel 格式保存的预测表格。


## 4. 产线模块选择与结果保存

通用表格识别v2产线提供了多种模块选择与结果保存方式，能够满足各种使用场景下的不同使用需求。所有后处理参数、模块选择和结果保存方式请参考 [通用表格识别v2产线使用教程](../pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.md)。下面我们基于公式识别模型产线，介绍如何使用这些调优手段。

### 4.1 可选版面区域检测模块 —— 适配不同类型输入图像

通用表格识别v2产线提供了多种可选的功能模块，帮助您在不同使用场景下根据实际情况灵活选取。下面我们以通用表格识别v2产线中可选的版面区域检测模块为例，介绍如何选取功能模块。

运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_rec_v2_pipe.jpg)到本地.

首先，我们不使用版面区域检测模块对示例图片进行处理：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="table_recognition_v2")

output = pipeline.predict(
    input="table_rec_v2_pipe.jpg",
    use_layout_detection=False, # 不使用版面区域检测模块
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)

for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_html("./output/")
```

查看保存下来的可视化结果和HTML表格，可以看到通用表格识别v2产线将整张图都作为了表格进行处理，导致了识别错误现象。

接下来，我们启用版面区域检测模块对这张图进行处理：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="table_recognition_v2")

output = pipeline.predict(
    input="table_rec_v2_pipe.jpg",
    use_layout_detection=True, # 使用版面区域检测模块，或不设置此项，其默认值为True
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)

for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_html("./output/")
```

查看保存下来的可视化结果和HTML表格，可以看到通用表格识别v2产线正确地将表格区域识别出来，并基于此得到了正确的表格识别结果。

因此，在您的实际使用中，如果输入图片带有非表格区域（例如一页完整的文档图像），则一定要启用版面区域检测模块进行处理，才能得到正确的表格识别结果。反之，若您的输出图片已经确定为仅有表格区域，则可以不启用版面区域检测模块以提升产线整体的推理速度。


### 4.2 多类型结果保存 —— 适配不同场景使用需求

通用表格识别v2产线支持了多种结果保存形式，帮助您在不同使用场景下根据实际情况灵活选取。

接下来，我们基于示例图片，查看所有支持的结果保存类型：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="table_recognition_v2")

output = pipeline.predict(
    input="table_rec_v2_pipe.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)

for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_xlsx("./output/")
    res.save_to_html("./output/")
    res.save_to_json("./output/")
```

观察命令行输出和 ./output/ 路径下的保存结果，可以看到有如下几种结果：

1. 直接将预测结果相关信息打印在命令行界面。

2. 在保存路径下，存储了产线的所有可视化结果图像，包括版面区域检测结果（若开启）、OCR检测识别结果、表格单元格识别结果。

3. 在保存路径下，存储了产线预测的 HTML 格式表格（以 html 文件形式存储），通过浏览器渲染后可以看到完整的表格结构与内容。

4. 在保存路径下，存储了产线预测的 Excel 格式表格（以 xlsx 文件形式存储），通过 Excel 打开后可以看到完整的表格结构与内容。

在您的实际使用中，您可以根据实际使用需求选择其中一种或多种形式来保存结果。若您不需要其中某一种或几种保存结果，建议无需将其保存，以提升产线的整体推理速度，同时节省您的存储空间。



## 5. 产线模型微调 —— 以表格单元格检测模块为例

表格识别v2产线采用了串联组网架构进行表格识别，因此其原生支持在任何使用场景下进行针对性微调。在实际使用时，表格识别v2产线如果在您的使用场景上表现良好，则不需要进行微调；若在您的场景上存在需解决的 bad case，则需要根据 bad case 的特征定位到具体的待微调模块，并对此模块进行针对性微调，使得整体的表格识别性能得到提高。特别地，表格识别v2产线支持在有线表、无线表其中的一个或全部分支上采用端到端表格识别模型（如 SLANet、SLANet_plus ）进行表格识别，您也可以在您的使用场景下尝试此方案解决 bad case。具体的待微调模块定位方式、端到端表格识别模型使用方式请参考 [PaddleX通用表格识别v2产线官方教程](../pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.md)。

接下来，我们以无线表格单元格检测模型 RT-DETR-L_wireless_table_cell_det 为例，来尝试进行一次完整的子模型调优过程。

### 5.1 数据准备

本教程已为您准备好示例数据集，可通过以下命令获取。关于数据格式介绍，您可以参考 [PaddleX 目标检测模块数据标注教程](../data_annotations/cv_modules/object_detection.md)。

数据集获取命令：
```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/table-rec-v2-pipe_practical_datasets.tar -P ./dataset
tar -xf ./dataset/table-rec-v2-pipe_practical_datasets.tar -C ./dataset/
```

示例数据集已提前完成数据校验和数据划分，若您需要了解表格单元格检测模块支持的数据集操作及其命令，请参考 [表格单元格检测模块使用教程](../module_usage/tutorials/ocr_modules/table_cells_detection.md)。


### 5.2 官方权重性能评估

在开始微调训练之前，我们首先需要评估当前模型的性能是否达到了预期。通过以下命令，将无有线表格单元格检测模型的官方预训练权重拉取到本地：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams
```

然后，我们将配置文件 PaddleX/paddlex/configs/modules/table_cells_detection/RT-DETR-L_wireless_table_cell_det.yaml 中 Evaluate 部分的权重路径修改为拉取到本地的官方权重，如下所示：

```yaml
Evaluate:
  weight_path: "./RT-DETR-L_wireless_table_cell_det_pretrained.pdparams"
  log_interval: 10
```

最后，仅需一行指令即可在示例数据集上评估模型性能：

```bash
python main.py -c paddlex/configs/modules/table_cells_detection/RT-DETR-L_wireless_table_cell_det.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./datasets/table-rec-v2-pipe_practical_datasets
```

查看命令行界面打印出来的评估结果，选取其中的 IoU=0.5 时的 mAP 作为评估指标，可以看到官方权重在测试集上的指标是 0.965。

### 5.2 模型微调训练

虽然官方权重在测试集上已经表现极好，但我们仍然可以尝试提升其针对当前数据集的单元格检测性能。

接下来，我们仅需一行指令即可启动模型训练：

```bash
python main.py -c paddlex/configs/modules/table_cells_detection/RT-DETR-L_wireless_table_cell_det.yaml \
    -o Global.mode=train \
    -o Global.device=gpu:0,1,2,3,4,5,6,7 \
    -o Train.epochs_iters=10 \
    -o Train.batch_size=8 \
    -o Train.learning_rate=0.0001 \
    -o Global.output=output \
    -o Global.dataset_dir=./datasets/table-rec-v2-pipe_practical_datasets
```

其中，Global.output 代表训练后模型的保存路径，请根据自己的实际情况进行设置。除此之外，Global.device、Train.epochs_iters、Train.batch_size、Train.learning_rate 分别代表训练使用的硬件（及其编号）、模型训练的轮次、批大小、学习率，如果需要调整其中某一训练参数，其他训练参数需要相应做调整，以保证最佳的训练效果。

### 5.3 微调后模型评估

训练结束后，将配置文件中 Evaluate 部分的权重路径修改为模型保存路径下的 best model，如下所示：

```yaml
Evaluate:
  weight_path: "output/best_model/best_model.pdparams"
  log_interval: 10
```

查看命令行界面打印出来的评估结果，可以看到微调训练后的模型权重性能得到了进一步提高，在测试集上的指标可以达到 0.990。
至此，我们就完成了一次针对特定场景数据的产线模型微调。


## 6. 开发集成/部署
如果表格识别效果可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。
### 6.1 直接后处理调整好的产线应用在您的 Python 项目中，可以参考如下示例代码：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/table_recognition_v2.yaml")

output = pipeline.predict(
    input="table_recognition_v2.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)

for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_xlsx("./output/")
    res.save_to_html("./output/")
    res.save_to_json("./output/")
```

更多参数请参考 [通用表格识别v2产线使用教程](../pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.md)。

### 6.2 以高稳定性服务化部署作为本教程的实践内容，具体可以参考 [PaddleX 服务化部署指南](../pipeline_deploy/serving.md) 进行实践。

下载表格识别高稳定性服务化部署 SDK <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_table_recognition_v2_sdk.tar.gz">paddlex_hps_table_recognition_v2_sdk.tar.gz</a>，解压 SDK 并运行部署脚本，如下：

```bash
tar -xvf paddlex_hps_table_recognition_v2_sdk.tar.gz
```

#### 5.2.2 获取序列号

- 在 [飞桨 AI Studio 星河社区-人工智能学习与实训社区](https://aistudio.baidu.com/paddlex/commercialization) 的“开源模型产线部署序列号咨询与获取”部分选择“立即获取”，如下图所示：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipeline_deploy/image-1.png">

选择表格识别产线，并点击“获取”。之后，可以在页面下方的“开源产线部署SDK序列号管理”部分找到获取到的序列号：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipeline_deploy/image-2.png">

**请注意**：每个序列号只能绑定到唯一的设备指纹，且只能绑定一次。这意味着用户如果使用不同的机器部署产线，则必须为每台机器准备单独的序列号。

#### 5.2.3 运行服务

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

#### 5.2.4 调用服务

目前，仅支持使用 Python 客户端调用服务。支持的 Python 版本为 3.8 至 3.12。

切换到高稳定性服务化部署 SDK 的 `client` 目录，执行如下命令安装依赖：

```bash
# 建议在虚拟环境中安装
python -m pip install -r requirements.txt
python -m pip install paddlex_hps_client-*.whl
```

`client` 目录的 `client.py` 脚本包含服务的调用示例，并提供命令行接口。


### 5.3 此外，PaddleX 也提供了其他三种部署方式，说明如下：

* 高性能部署：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能部署流程请参考 [PaddleX 高性能推理指南](../pipeline_deploy/high_performance_inference.md)。
* 基础服务化部署：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考 [PaddleX 服务化部署指南](../pipeline_deploy/serving.md)。
* 端侧部署：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考 [PaddleX端侧部署指南](../pipeline_deploy/edge_deploy.md)。

您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。
