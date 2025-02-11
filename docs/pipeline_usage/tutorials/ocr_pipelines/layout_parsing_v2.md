---
comments: true
---

# 通用版面解析v2产线使用教程

## 1. 通用版面解析v2产线介绍
版面解析是一种从文档图像中提取结构化信息的技术，主要用于将复杂的文档版面转换为机器可读的数据格式。这项技术在文档管理、信息提取和数据数字化等领域具有广泛的应用。版面解析通过结合光学字符识别（OCR）、图像处理和机器学习算法，能够识别和提取文档中的文本块、标题、段落、图片、表格以及其他版面元素。此过程通常包括版面分析、元素分析和数据格式化三个主要步骤，最终生成结构化的文档数据，提升数据处理的效率和准确性。<b>通用版面解析v2产线在通用版面解析v1产线的基础上，强化了版面区域检测、表格识别、公式识别的能力，增加了多栏阅读顺序的恢复能力、结果转换 Markdown 文件的能力，在多种文档数据中，表现优异，可以处理较复杂的文档数据。</b>本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<b>通用版面解析v2产线中包含必选的版面区域分析模块、通用OCR子产线，</b>以及可选的文档图像预处理子产线、表格识别子产线、印章识别子产线和公式识别子产线。

<b>如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。

<details><summary> 👉模型列表详情</summary>
<p><b>文档图像方向分类模块（可选）：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>基于PP-LCNet_x1_0的文档图像分类模型，含有四个类别，即0度，90度，180度，270度</td>
</tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b>
<p><b>文本图像矫正模块（可选）：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>CER </th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UVDoc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">训练模型</a></td>
<td>0.179</td>
<td>30.3 M</td>
<td>高精度文本图像矫正模型</td>
</tr>
</tbody>
</table>
<b>注：模型的精度指标测量自 <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet benchmark</a>。</b>
<p><b>版面区域检测模块模型（必选）：</b></p>
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
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">训练模型</a></td>
<td>90.4</td>
<td>34.5252</td>
<td>1454.27</td>
<td>123.76 M</td>
<td>基于RT-DETR-L在包含中英文论文、杂志、合同、书本、试卷和研报等场景的自建数据集训练的高精度版面区域定位模型</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">训练模型</a></td>
<td>75.2</td>
<td>15.9</td>
<td>160.1</td>
<td>22.578</td>
<td>基于PicoDet-L在包含中英文论文、杂志、合同、书本、试卷和研报等场景的自建数据集训练的精度效率平衡的版面区域定位模型</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">训练模型</a></td>
<td>70.9</td>
<td>13.8</td>
<td>46.7</td>
<td>4.834</td>
<td>基于PicoDet-S在中英文论文、杂志、合同、书本、试卷和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的版面区域分析数据集，包含中英文论文、杂志和研报等常见的 1w 张文档类型图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p>
<p><b>表格结构识别模块（可选）：</b></p>
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
<td>SLANet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">训练模型</a></td>
<td>59.52</td>
<td>103.08 / 103.08</td>
<td>197.99 / 197.99</td>
<td>6.9 M</td>
<td>SLANet 是百度飞桨视觉团队自研的表格结构识别模型。该模型通过采用CPU 友好型轻量级骨干网络PP-LCNet、高低层特征融合模块CSP-PAN、结构与位置信息对齐的特征解码模块SLA Head，大幅提升了表格结构识别的精度和推理速度。</td>
</tr>
<tr>
<td>SLANet_plus</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_plus_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">训练模型</a></td>
<td>63.69</td>
<td>140.29 / 140.29</td>
<td>195.39 / 195.39</td>
<td>6.9 M</td>
<td>SLANet_plus 是百度飞桨视觉团队自研的表格结构识别模型SLANet的增强版。相较于SLANet，SLANet_plus 对无线表、复杂表格的识别能力得到了大幅提升，并降低了模型对表格定位准确性的敏感度，即使表格定位出现偏移，也能够较准确地进行识别。</td>
</tr>
</table>
<p><b>注：以上精度指标测量PaddleX 内部自建英文表格识别数据集。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<p><b>文本检测模块（必选）：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">训练模型</a></td>
<td>82.56</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>109</td>
<td>PP-OCRv4 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">训练模型</a></td>
<td>77.35</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>PP-OCRv4 的移动端文本检测模型，效率更高，适合在端侧设备部署</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv3_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_det_pretrained.pdparams">训练模型</a></td>
<td>78.68</td>
<td>8.44 / 2.91</td>
<td>27.87 / 27.87</td>
<td>2.1</td>
<td>PP-OCRv3 的移动端文本检测模型，效率更高，适合在端侧设备部署</td>
</tr>
<tr>
<td>PP-OCRv3_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv3_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_server_det_pretrained.pdparams">训练模型</a></td>
<td>80.11</td>
<td>65.41 / 13.67</td>
<td>305.07 / 305.07</td>
<td>102.1</td>
<td>PP-OCRv3 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<p><b>文本识别模块模型（必选）：</b></p>

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
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv4_server_rec_doc_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td>81.53</td>
<td>6.65 / 6.65</td>
<td>32.92 / 32.92</td>
<td>74.7 M</td>
<td>PP-OCRv4_server_rec_doc是在PP-OCRv4_server_rec的基础上，在更多中文文档数据和PP-OCR训练数据的混合数据训练而成，增加了部分繁体字、日文、特殊字符的识别能力，可支持识别的字符为1.5万+，除文档相关的文字识别能力提升外，也同时提升了通用文字的识别能力</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>78.74</td>
<td>4.82 / 4.82</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td>PP-OCRv4的轻量级识别模型，推理效率高，可以部署在包含端侧设备的多种硬件设备中</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td>
<td>80.61 </td>
<td>6.58 / 6.58</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
<td>PP-OCRv4的服务器端模型，推理精度高，可以部署在多种不同的服务器上</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td>72.96</td>
<td>5.87 / 5.87</td>
<td>9.07 / 4.28</td>
<td>9.2 M</td>
<td>PP-OCRv3的轻量级识别模型，推理效率高，可以部署在包含端侧设备的多种硬件设备中</td>
</tr>
</table>
<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 8367 张图片。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
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
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_SVTRv2_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">训练模型</a></td>
<td>68.81</td>
<td>8.08 / 8.08</td>
<td>50.17 / 42.50</td>
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
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_RepSVTR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">训练模型</a></td>
<td>65.07</td>
<td>5.93 / 5.93</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
<td rowspan="1">    RepSVTR 文本识别模型是一种基于SVTRv2 的移动端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，B榜端到端识别精度相比PP-OCRv4提升2.5%，推理速度持平。</td>
</tr>
</table>
<p><b>注：以上精度指标的评估集是 <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>B榜。 所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>

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
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
en_PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td> 70.39</td>
<td>4.81 / 4.81</td>
<td>16.10 / 5.31</td>
<td>6.8 M</td>
<td>基于PP-OCRv4识别模型训练得到的超轻量英文识别模型，支持英文、数字识别</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
en_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td>70.69</td>
<td>5.44 / 5.44</td>
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
<td>korean_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
korean_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td>60.21</td>
<td>5.40 / 5.40</td>
<td>9.11 / 4.05</td>
<td>8.6 M</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量韩文识别模型，支持韩文、数字识别</td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
japan_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td>45.69</td>
<td>5.70 / 5.70</td>
<td>8.48 / 4.07</td>
<td>8.8 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量日文识别模型，支持日文、数字识别</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td>82.06</td>
<td>5.90 / 5.90</td>
<td>9.28 / 4.34</td>
<td>9.7 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量繁体中文识别模型，支持繁体中文、数字识别</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
te_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td>95.88</td>
<td>5.42 / 5.42</td>
<td>8.10 / 6.91</td>
<td>7.8 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量泰卢固文识别模型，支持泰卢固文、数字识别</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
ka_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td>96.96</td>
<td>5.25 / 5.25</td>
<td>9.09 / 3.86</td>
<td>8.0 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量卡纳达文识别模型，支持卡纳达文、数字识别</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
ta_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td>76.83</td>
<td>5.23 / 5.23</td>
<td>10.13 / 4.30</td>
<td>8.0 M </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量泰米尔文识别模型，支持泰米尔文、数字识别</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
latin_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td>76.93</td>
<td>5.20 / 5.20</td>
<td>8.83 / 7.15</td>
<td>7.8 M</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量拉丁文识别模型，支持拉丁文、数字识别</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
arabic_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td>73.55</td>
<td>5.35 / 5.35</td>
<td>8.80 / 4.56</td>
<td>7.8 M</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量阿拉伯字母识别模型，支持阿拉伯字母、数字识别</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td>94.28</td>
<td>5.23 / 5.23</td>
<td>8.89 / 3.88</td>
<td>7.9 M  </td>
<td>基于PP-OCRv3识别模型训练得到的超轻量斯拉夫字母识别模型，支持斯拉夫字母、数字识别</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
<td>96.44</td>
<td>5.22 / 5.22</td>
<td>8.56 / 4.06</td>
<td>7.9 M</td>
<td>基于PP-OCRv3识别模型训练得到的超轻量梵文字母识别模型，支持梵文字母、数字识别</td>
</tr>
</table>
<p><b>注：以上精度指标的评估集是 PaddleX 自建的多语种数据集。 所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<p><b>文本行方向分类模块（可选）：</b></p>
<table>
<thead>
<tr>
<th>模型</th>
<th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x0_25_textline_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">训练模型</a></td>
<td>95.54</td>
<td>-</td>
<td>-</td>
<td>0.32</td>
<td>基于PP-LCNet_x0_25的文本行分类模型，含有两个类别，即0度，180度</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p>
<p><b>公式识别模块（可选）：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>BLEU score</th>
<th>normed edit distance</th>
<th>ExpRate （%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小</th>
</tr>
</thead>
<tbody>
<tr>
<td>LaTeX_OCR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/LaTeX_OCR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">训练模型</a></td>
<td>0.8821</td>
<td>0.0823</td>
<td>40.01</td>
<td>2047.13 / 2047.13</td>
<td>10582.73 / 10582.73</td>
<td>89.7 M</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标测量自 <a href="https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO">LaTeX-OCR公式识别测试集</a>。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<p><b>印章文本检测模块（可选）：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">训练模型</a></td>
<td>98.21</td>
<td>74.75 / 67.72</td>
<td>382.55 / 382.55</td>
<td>109</td>
<td>PP-OCRv4的服务端印章文本检测模型，精度更高，适合在较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">训练模型</a></td>
<td>96.47</td>
<td>7.82 / 3.09</td>
<td>48.28 / 23.97</td>
<td>4.6</td>
<td>PP-OCRv4的移动端印章文本检测模型，效率更高，适合在端侧部署</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是自建的数据集，包含500张圆形印章图像。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p>
</b></p></details>
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
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UVDoc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">训练模型</a></td>
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
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>基于PP-LCNet_x1_0的文档图像分类模型，含有四个类别，即0度，90度，180度，270度</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p></details>
<b></b>

## 2. 快速开始
PaddleX 所提供的模型产线均可以快速体验效果，你可以在本地使用命令行或 Python 体验通用通用版面解析v2产线的效果。

在本地使用通用版面解析v2产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

### 2.1 命令行方式体验
一行命令即可快速体验版面解析产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_parsing_v2_demo.png)，并将 `--input` 替换为本地路径，进行预测

```
paddlex --pipeline layout_parsing_v2 \
        --input layout_parsing_v2_demo.png \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --save_path ./output \
        --device gpu:0
```
相关的参数说明可以参考[2.2.2 Python脚本方式集成](#222-python脚本方式集成)中的参数说明。

运行后，会将结果打印到终端上，结果如下：

<details><summary> 👉点击展开</summary>
<pre><code>
{'res': {'input_path': 'layout_parsing_v2_demo.png', 'model_settings': {'use_doc_preprocessor': False, 'use_general_ocr': True, 'use_seal_recognition': True, 'use_table_recognition': True, 'use_formula_recognition': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9853514432907104, 'coordinate': [770.9531, 776.6814, 1122.6057, 1058.7322]}, {'cls_id': 1, 'label': 'image', 'score': 0.9848673939704895, 'coordinate': [775.7434, 202.27979, 1502.8113, 686.02136]}, {'cls_id': 2, 'label': 'text', 'score': 0.983731746673584, 'coordinate': [1152.3197, 1113.3275, 1503.3029, 1346.586]}, {'cls_id': 2, 'label': 'text', 'score': 0.9832221865653992, 'coordinate': [1152.5602, 801.431, 1503.8436, 986.3563]}, {'cls_id': 2, 'label': 'text', 'score': 0.9829439520835876, 'coordinate': [9.549545, 849.5713, 359.1173, 1058.7488]}, {'cls_id': 2, 'label': 'text', 'score': 0.9811657667160034, 'coordinate': [389.58298, 1137.2659, 740.66235, 1346.7488]}, {'cls_id': 2, 'label': 'text', 'score': 0.9775941371917725, 'coordinate': [9.1302185, 201.85, 359.0409, 339.05692]}, {'cls_id': 2, 'label': 'text', 'score': 0.9750366806983948, 'coordinate': [389.71454, 752.96924, 740.544, 889.92456]}, {'cls_id': 2, 'label': 'text', 'score': 0.9738152027130127, 'coordinate': [389.94565, 298.55988, 740.5585, 435.5124]}, {'cls_id': 2, 'label': 'text', 'score': 0.9737328290939331, 'coordinate': [771.50256, 1065.4697, 1122.2582, 1178.7324]}, {'cls_id': 2, 'label': 'text', 'score': 0.9728517532348633, 'coordinate': [1152.5154, 993.3312, 1503.2349, 1106.327]}, {'cls_id': 2, 'label': 'text', 'score': 0.9725610017776489, 'coordinate': [9.372787, 1185.823, 359.31738, 1298.7227]}, {'cls_id': 2, 'label': 'text', 'score': 0.9724331498146057, 'coordinate': [389.62848, 610.7389, 740.83234, 746.2377]}, {'cls_id': 2, 'label': 'text', 'score': 0.9720287322998047, 'coordinate': [389.29898, 897.0936, 741.41516, 1034.6616]}, {'cls_id': 2, 'label': 'text', 'score': 0.9713053703308105, 'coordinate': [10.323685, 1065.4663, 359.6786, 1178.8872]}, {'cls_id': 2, 'label': 'text', 'score': 0.9689728021621704, 'coordinate': [9.336395, 537.6609, 359.2901, 652.1881]}, {'cls_id': 2, 'label': 'text', 'score': 0.9684857130050659, 'coordinate': [10.7608185, 345.95068, 358.93616, 434.64087]}, {'cls_id': 2, 'label': 'text', 'score': 0.9681928753852844, 'coordinate': [9.674866, 658.89075, 359.56528, 770.4319]}, {'cls_id': 2, 'label': 'text', 'score': 0.9634978175163269, 'coordinate': [770.9464, 1281.1785, 1122.6522, 1346.7156]}, {'cls_id': 2, 'label': 'text', 'score': 0.96304851770401, 'coordinate': [390.0113, 201.28055, 740.1684, 291.53073]}, {'cls_id': 2, 'label': 'text', 'score': 0.962053120136261, 'coordinate': [391.21393, 1040.952, 740.5046, 1130.32]}, {'cls_id': 2, 'label': 'text', 'score': 0.9565253853797913, 'coordinate': [10.113251, 777.1482, 359.439, 842.437]}, {'cls_id': 2, 'label': 'text', 'score': 0.9497362375259399, 'coordinate': [390.31357, 537.86285, 740.47595, 603.9285]}, {'cls_id': 2, 'label': 'text', 'score': 0.9371236562728882, 'coordinate': [10.2034, 1305.9753, 359.5958, 1346.7295]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9338151216506958, 'coordinate': [791.6062, 1200.8479, 1103.3257, 1259.9324]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9326773285865784, 'coordinate': [408.0737, 457.37024, 718.9509, 516.63464]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9274250864982605, 'coordinate': [29.448685, 456.6762, 340.99194, 515.6999]}, {'cls_id': 2, 'label': 'text', 'score': 0.8742568492889404, 'coordinate': [1154.7095, 777.3624, 1330.3086, 794.5853]}, {'cls_id': 2, 'label': 'text', 'score': 0.8442489504814148, 'coordinate': [586.49316, 160.15454, 927.468, 179.64203]}, {'cls_id': 11, 'label': 'doc_title', 'score': 0.8332607746124268, 'coordinate': [133.80017, 37.41908, 1380.8601, 124.1429]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.6770150661468506, 'coordinate': [812.1718, 705.1199, 1484.6973, 747.1692]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[ 133,   35],
        ...,
        [ 133,  131]],

       ...,

       [[1154, 1323],
        ...,
        [1152, 1355]]], dtype=int16), 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['助力双方交往', '搭建友谊桥梁', '本报记者', '沈小晓', '任', '彦', '黄培昭', '身着中国传统民族服装的厄立特里亚青', '厄立特里亚高等教育与研究院合作建立，开', '年依次登台表演中国民族舞、现代舞、扇子舞', '设了中国语言课程和中国文化课程，注册学', '等,曼妙的舞姿赢得现场观众阵阵掌声。这', '生2万余人次。10余年来，厄特孔院已成为', '是日前厄立特里亚高等教育与研究院孔子学', '当地民众了解中国的一扇窗口。', '院(以下简称"厄特孔院")举办"喜迎新年"中国', '黄鸣飞表示，随着来学习中文的人日益', '歌舞比赛的场景。', '增多，阿斯马拉大学教学点已难以满足教学', '中国和厄立特里亚传统友谊深厚。近年', '需要。2024年4月，由中企蜀道集团所属四', '来,在高质量共建"一带一路"框架下，中厄两', '川路桥承建的孔院教学楼项目在阿斯马拉开', '国人文交流不断深化，互利合作的民意基础', '工建设，预计今年上半年峻工，建成后将为厄', '日益深厚。', '特孔院提供全新的办学场地。', '“学好中文，我们的', '“在中国学习的经历', '未来不是梦”', '让我看到更广阔的世界”', '“鲜花曾告诉我你怎样走过，大地知道你', '多年来，厄立特里亚广大赴华留学生和', '心中的每一个角落…"厄立特里亚阿斯马拉', '培训人员积极投身国家建设，成为助力该国', '大学综合楼二层，一阵优美的歌声在走廊里回', '发展的人才和厄中友好的见证者和推动者。', '响。循着熟悉的旋律轻轻推开一间教室的门，', '在厄立特里亚全国妇女联盟工作的约翰', '学生们正跟着老师学唱中文歌曲《同一首歌》。', '娜·特韦尔德·凯莱塔就是其中一位。她曾在', '这是厄特孔院阿斯马拉大学教学点的一', '中华女子学院攻读硕士学位，研究方向是女', '节中文歌曲课。为了让学生们更好地理解歌', '性领导力与社会发展。其间，她实地走访中国', '词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐', '多个地区，获得了观察中国社会发展的第一', '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。', '字翻译和解释歌词。随着伴奏声响起，学生们', '手资料。', '中国驻厄立特里亚大使馆供图', '边唱边随着节拍摇动身体，现场气氛热烈。', '谈起在中国求学的经历，约翰娜记忆犹', '“这是中文歌曲初级班，共有32人。学', '新："中国的发展在当今世界是独一无二的。', '“不管远近都是客人，请不用客气；相约', '瓦的北红海省博物馆。', '生大部分来自首都阿斯马拉的中小学，年龄', '沿着中国特色社会主义道路坚定前行，中国', '好了在一起我们欢迎你"在一场中厄青', '博物馆二层陈列着一个发掘自阿杜利', '最小的仅有6岁。”尤斯拉告诉记者。', '创造了发展奇迹，这一切都离不开中国共产党', '年联谊活动上，四川路桥中方员工同当地大', '斯古城的中国古代陶制酒器，罐身上写着', '尤斯拉今年23岁，是厄立特里亚一所公立', '的领导。中国的发展经验值得许多国家学习', '学生合唱《北京欢迎你》。厄立特里亚技术学', '“万""和""禅"“山"等汉字。“这件文物证', '学校的艺术老师。她12岁开始在厄特孔院学', '借鉴。”', '院计算机科学与工程专业学生鲁夫塔·谢拉', '明,很早以前我们就通过海上丝绸之路进行', '习中文,在2017年第十届"汉语桥"世界中学生', '正在西南大学学习的厄立特里亚博士生', '是其中一名演唱者，她很早便在孔院学习中', '贸易往来与文化交流。这也是厄立特里亚', '中文比赛中获得厄立特里亚赛区第一名，并和', '穆卢盖塔·泽穆伊对中国怀有深厚感情。8', '文，一直在为去中国留学作准备。"这句歌词', '与中国友好交往历史的有力证明。"北红海', '同伴代表厄立特里亚前往中国参加决赛，获得', '年前，在北京师范大学获得硕士学位后，穆卢', '是我们两国人民友谊的生动写照。无论是投', '省博物馆研究与文献部负责人伊萨亚斯·特', '团体优胜奖。2022年起，尤斯拉开始在厄特孔', '盖塔在社交媒体上写下这样一段话："这是我', '身于厄立特里亚基础设施建设的中企员工，', '斯法兹吉说。', '院兼职教授中文歌曲，每周末两个课时。“中国', '人生的重要一步，自此我拥有了一双坚固的', '还是在中国留学的厄立特里亚学子，两国人', '厄立特里亚国家博物馆考古学和人类学', '文化博大精深，我希望我的学生们能够通过中', '鞋子，赋予我穿越荆棘的力量。”', '民携手努力，必将推动两国关系不断向前发', '研究员菲尔蒙·特韦尔德十分喜爱中国文', '文歌曲更好地理解中国文化。"她说。', '穆卢盖塔密切关注中国在经济、科技、教', '展。"鲁夫塔说。', '化。他表示：“学习彼此的语言和文化，将帮', '“姐姐，你想去中国吗？""非常想！我想', '育等领域的发展，中国在科研等方面的实力', '厄立特里亚高等教育委员会主任助理萨', '助厄中两国人民更好地理解彼此，助力双方', '去看故宫、爬长城。"尤斯拉的学生中有一对', '与日俱增。在中国学习的经历让我看到更广', '马瑞表示："每年我们都会组织学生到中国访', '交往，搭建友谊桥梁。"', '能歌善舞的姐妹，姐姐露娅今年15岁，妹妹', '阔的世界，从中受益匪浅。', '问学习，目前有超过5000名厄立特里亚学生', '厄立特里亚国家博物馆馆长塔吉丁·努', '莉娅14岁，两人都已在厄特孔院学习多年，', '23岁的莉迪亚·埃斯蒂法诺斯已在厄特', '在中国留学。学习中国的教育经验，有助于', '里达姆·优素福曾多次访问中国，对中华文明', '中文说得格外流利。', '孔院学习3年，在中国书法、中国画等方面表', '提升厄立特里亚的教育水平。”', '的传承与创新、现代化博物馆的建设与发展', '露娅对记者说："这些年来，怀着对中文', '现十分优秀，在2024年厄立特里亚赛区的', '“共同向世界展示非', '印象深刻。“中国博物馆不仅有许多保存完好', '和中国文化的热爱，我们姐妹俩始终相互鼓', '“汉语桥"比赛中获得一等奖。莉迪亚说："学', '的文物，还充分运用先进科技手段进行展示，', '励，一起学习。我们的中文一天比一天好，还', '习中国书法让我的内心变得安宁和纯粹。我', '洲和亚洲的灿烂文明”', '帮助人们更好理解中华文明。"塔吉丁说，"厄', '学会了中文歌和中国舞。我们一定要到中国', '也喜欢中国的服饰,希望未来能去中国学习，', '立特里亚与中国都拥有悠久的文明，始终相', '去。学好中文，我们的未来不是梦！"', '把中国不同民族元素融入服装设计中，创作', '从阿斯马拉出发，沿着蜿蜓曲折的盘山', '互理解、相互尊重。我希望未来与中国同行', '据厄特孔院中方院长黄鸣飞介绍，这所', '出更多精美作品，也把厄特文化分享给更多', '公路一路向东寻找丝路印迹。驱车两个小', '加强合作，共同向世界展示非洲和亚洲的灿', '孔院成立于2013年3月，由贵州财经大学和', '的中国朋友。”', '时，记者来到位于厄立特里亚港口城市马萨', '烂文明。”'], 'rec_scores': array([0.99943757, ..., 0.98181838]), 'rec_polys': array([[[ 133,   35],
        ...,
        [ 133,  131]],

       ...,

       [[1154, 1323],
        ...,
        [1152, 1355]]], dtype=int16), 'rec_boxes': array([[ 133, ...,  131],
       ...,
       [1152, ..., 1359]], dtype=int16)}, 'text_paragraphs_ocr_res': {'rec_polys': array([[[ 133,   35],
        ...,
        [ 133,  131]],

       ...,

       [[1154, 1323],
        ...,
        [1152, 1355]]], dtype=int16), 'rec_texts': ['助力双方交往', '搭建友谊桥梁', '本报记者', '沈小晓', '任', '彦', '黄培昭', '身着中国传统民族服装的厄立特里亚青', '厄立特里亚高等教育与研究院合作建立，开', '年依次登台表演中国民族舞、现代舞、扇子舞', '设了中国语言课程和中国文化课程，注册学', '等,曼妙的舞姿赢得现场观众阵阵掌声。这', '生2万余人次。10余年来，厄特孔院已成为', '是日前厄立特里亚高等教育与研究院孔子学', '当地民众了解中国的一扇窗口。', '院(以下简称"厄特孔院")举办"喜迎新年"中国', '黄鸣飞表示，随着来学习中文的人日益', '歌舞比赛的场景。', '增多，阿斯马拉大学教学点已难以满足教学', '中国和厄立特里亚传统友谊深厚。近年', '需要。2024年4月，由中企蜀道集团所属四', '来,在高质量共建"一带一路"框架下，中厄两', '川路桥承建的孔院教学楼项目在阿斯马拉开', '国人文交流不断深化，互利合作的民意基础', '工建设，预计今年上半年峻工，建成后将为厄', '日益深厚。', '特孔院提供全新的办学场地。', '“学好中文，我们的', '“在中国学习的经历', '未来不是梦”', '让我看到更广阔的世界”', '“鲜花曾告诉我你怎样走过，大地知道你', '多年来，厄立特里亚广大赴华留学生和', '心中的每一个角落…"厄立特里亚阿斯马拉', '培训人员积极投身国家建设，成为助力该国', '大学综合楼二层，一阵优美的歌声在走廊里回', '发展的人才和厄中友好的见证者和推动者。', '响。循着熟悉的旋律轻轻推开一间教室的门，', '在厄立特里亚全国妇女联盟工作的约翰', '学生们正跟着老师学唱中文歌曲《同一首歌》。', '娜·特韦尔德·凯莱塔就是其中一位。她曾在', '这是厄特孔院阿斯马拉大学教学点的一', '中华女子学院攻读硕士学位，研究方向是女', '节中文歌曲课。为了让学生们更好地理解歌', '性领导力与社会发展。其间，她实地走访中国', '词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐', '多个地区，获得了观察中国社会发展的第一', '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。', '字翻译和解释歌词。随着伴奏声响起，学生们', '手资料。', '中国驻厄立特里亚大使馆供图', '边唱边随着节拍摇动身体，现场气氛热烈。', '谈起在中国求学的经历，约翰娜记忆犹', '“这是中文歌曲初级班，共有32人。学', '新："中国的发展在当今世界是独一无二的。', '“不管远近都是客人，请不用客气；相约', '瓦的北红海省博物馆。', '生大部分来自首都阿斯马拉的中小学，年龄', '沿着中国特色社会主义道路坚定前行，中国', '好了在一起我们欢迎你"在一场中厄青', '博物馆二层陈列着一个发掘自阿杜利', '最小的仅有6岁。”尤斯拉告诉记者。', '创造了发展奇迹，这一切都离不开中国共产党', '年联谊活动上，四川路桥中方员工同当地大', '斯古城的中国古代陶制酒器，罐身上写着', '尤斯拉今年23岁，是厄立特里亚一所公立', '的领导。中国的发展经验值得许多国家学习', '学生合唱《北京欢迎你》。厄立特里亚技术学', '“万""和""禅"“山"等汉字。“这件文物证', '学校的艺术老师。她12岁开始在厄特孔院学', '借鉴。”', '院计算机科学与工程专业学生鲁夫塔·谢拉', '明,很早以前我们就通过海上丝绸之路进行', '习中文,在2017年第十届"汉语桥"世界中学生', '正在西南大学学习的厄立特里亚博士生', '是其中一名演唱者，她很早便在孔院学习中', '贸易往来与文化交流。这也是厄立特里亚', '中文比赛中获得厄立特里亚赛区第一名，并和', '穆卢盖塔·泽穆伊对中国怀有深厚感情。8', '文，一直在为去中国留学作准备。"这句歌词', '与中国友好交往历史的有力证明。"北红海', '同伴代表厄立特里亚前往中国参加决赛，获得', '年前，在北京师范大学获得硕士学位后，穆卢', '是我们两国人民友谊的生动写照。无论是投', '省博物馆研究与文献部负责人伊萨亚斯·特', '团体优胜奖。2022年起，尤斯拉开始在厄特孔', '盖塔在社交媒体上写下这样一段话："这是我', '身于厄立特里亚基础设施建设的中企员工，', '斯法兹吉说。', '院兼职教授中文歌曲，每周末两个课时。“中国', '人生的重要一步，自此我拥有了一双坚固的', '还是在中国留学的厄立特里亚学子，两国人', '厄立特里亚国家博物馆考古学和人类学', '文化博大精深，我希望我的学生们能够通过中', '鞋子，赋予我穿越荆棘的力量。”', '民携手努力，必将推动两国关系不断向前发', '研究员菲尔蒙·特韦尔德十分喜爱中国文', '文歌曲更好地理解中国文化。"她说。', '穆卢盖塔密切关注中国在经济、科技、教', '展。"鲁夫塔说。', '化。他表示：“学习彼此的语言和文化，将帮', '“姐姐，你想去中国吗？""非常想！我想', '育等领域的发展，中国在科研等方面的实力', '厄立特里亚高等教育委员会主任助理萨', '助厄中两国人民更好地理解彼此，助力双方', '去看故宫、爬长城。"尤斯拉的学生中有一对', '与日俱增。在中国学习的经历让我看到更广', '马瑞表示："每年我们都会组织学生到中国访', '交往，搭建友谊桥梁。"', '能歌善舞的姐妹，姐姐露娅今年15岁，妹妹', '阔的世界，从中受益匪浅。', '问学习，目前有超过5000名厄立特里亚学生', '厄立特里亚国家博物馆馆长塔吉丁·努', '莉娅14岁，两人都已在厄特孔院学习多年，', '23岁的莉迪亚·埃斯蒂法诺斯已在厄特', '在中国留学。学习中国的教育经验，有助于', '里达姆·优素福曾多次访问中国，对中华文明', '中文说得格外流利。', '孔院学习3年，在中国书法、中国画等方面表', '提升厄立特里亚的教育水平。”', '的传承与创新、现代化博物馆的建设与发展', '露娅对记者说："这些年来，怀着对中文', '现十分优秀，在2024年厄立特里亚赛区的', '“共同向世界展示非', '印象深刻。“中国博物馆不仅有许多保存完好', '和中国文化的热爱，我们姐妹俩始终相互鼓', '“汉语桥"比赛中获得一等奖。莉迪亚说："学', '的文物，还充分运用先进科技手段进行展示，', '励，一起学习。我们的中文一天比一天好，还', '习中国书法让我的内心变得安宁和纯粹。我', '洲和亚洲的灿烂文明”', '帮助人们更好理解中华文明。"塔吉丁说，"厄', '学会了中文歌和中国舞。我们一定要到中国', '也喜欢中国的服饰,希望未来能去中国学习，', '立特里亚与中国都拥有悠久的文明，始终相', '去。学好中文，我们的未来不是梦！"', '把中国不同民族元素融入服装设计中，创作', '从阿斯马拉出发，沿着蜿蜓曲折的盘山', '互理解、相互尊重。我希望未来与中国同行', '据厄特孔院中方院长黄鸣飞介绍，这所', '出更多精美作品，也把厄特文化分享给更多', '公路一路向东寻找丝路印迹。驱车两个小', '加强合作，共同向世界展示非洲和亚洲的灿', '孔院成立于2013年3月，由贵州财经大学和', '的中国朋友。”', '时，记者来到位于厄立特里亚港口城市马萨', '烂文明。”'], 'rec_scores': array([0.99943757, ..., 0.98181838]), 'rec_boxes': array([[ 133, ...,  131],
       ...,
       [1152, ..., 1359]], dtype=int16)}}}
</code></pre></details>

运行结果参数说明可以参考[2.2.2 Python脚本方式集成](#222-python脚本方式集成)中的结果解释。

<b>注：</b>由于产线的默认模型较大，推理速度可能较慢，您可以参考第一节的模型列表，替换推理速度更快的模型。

### 2.2 Python脚本方式集成
几行代码即可完成产线的快速推理，以通用版面解析v2产线为例：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="layout_parsing_v2")

output = pipeline.predict(
    input="./layout_parsing_v2_demo.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path="output") ## 保存当前图像的结构化json结果
    res.save_to_markdown(save_path="output") ## 保存当前图像的markdown格式的结果
```

如果是 PDF 文件，会将 PDF 的每一页单独处理，每一页的 Markdown 文件也会对应单独的结果。如果希望整个 PDF 文件转换为 Markdown 文件，建议使用以下的方式运行：

```python
from pathlib import Path
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="layout_parsing_v2")

input_file = "./your_pdf_file.pdf"
output_path = Path("./output")

output = pipeline.predict(
    input=input_file,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

markdown_texts = ""
markdown_images = []

for res in output:
    markdown_texts += res.markdown["markdown_texts"]
    markdown_images.append(res.markdown["markdown_images"])

mkd_file_path = output_path / f"{Path(input_file).stem}.md"
mkd_file_path.parent.mkdir(parents=True, exist_ok=True)
with open(mkd_file_path, "w", encoding="utf-8") as f:
    f.write(markdown_texts)

for item in markdown_images:
    if item:
        for path, image in item.items():
            file_path = output_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path)
```


在上述 Python 脚本中，执行了如下几个步骤：

（1）实例化 `create_pipeline` 实例化产线对象，具体参数说明如下：

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
<td>产线配置文件路径。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线推理设备。支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理，仅当该产线支持高性能推理时可用。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

（2）调用版面解析产线对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：

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
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
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
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否使用文档方向分类模块</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>：<code>True</code> 或者 <code>False</code>；</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否使用文档扭曲矫正模块</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>：<code>True</code> 或者 <code>False</code>；</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>是否使用文本行方向分类模块</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>：<code>True</code> 或者 <code>False</code>；</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_general_ocr</code></td>
<td>是否使用 OCR 子产线</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>：<code>True</code> 或者 <code>False</code>；</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>是否使用印章识别子产线</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>：<code>True</code> 或者 <code>False</code>；</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>是否使用表格识别子产线</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>：<code>True</code> 或者 <code>False</code>；</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td>是否使用公式识别子产线</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>：<code>True</code> 或者 <code>False</code>；</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>版面模型得分阈值</td>
<td><code>float|dict|None</code></td>
<td>
<ul>
<li><b>float</b>：<code>0-1</code> 之间的任意浮点数；</li>
<li><b>dict</b>： <code>{0:0.1}</code> key为类别ID，value为该类别的阈值；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>0.5</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>版面区域检测模型是否使用NMS后处理</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>：<code>True</code> 或者 <code>False</code>；</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>版面区域检测模型检测框的扩张系数</td>
<td><code>float|Tuple[float,float]|None</code></td>
<td>
<ul>
<li><b>float</b>：任意大于 <code>0</code>  浮点数；</li>
<li><b>Tuple[float,float]</b>：在横纵两个方向各自的扩张系数；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>1.0</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>版面区域检测的重叠框过滤方式</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>：<code>large</code>，<code>small</code>, <code>union</code>，分别表示重叠框过滤时选择保留大框，小框还是同时保留</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>large</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>文本检测的图像边长限制</td>
<td><code>int|None</code></td>
<td>
<ul>
<li><b>int</b>：大于 <code>0</code> 的任意整数；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>960</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>文本检测的图像边长限制类型</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>：支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code></li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>max</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>检测像素阈值，输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.3</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.6</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>2.0</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>文本识别阈值，得分大于该阈值的文本结果会被保留</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.0</code>。即不设阈值</li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>印章检测的图像边长限制</td>
<td><code>int|None</code></td>
<td>
<ul>
<li><b>int</b>：大于 <code>0</code> 的任意整数；</li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>960</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>印章检测的图像边长限制类型</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>：支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code></li>
<li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>max</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>检测像素阈值，输出的概率图中，得分大于该阈值的像素点才会被认为是印章像素点</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.3</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是印章区域</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.6</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>印章检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>2.0</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>印章识别阈值，得分大于该阈值的文本结果会被保留</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.0</code>。即不设阈值</li></li></ul></td>
<td><code>None</code></td>
</tr>
</table>

（3）对预测结果进行处理：每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为`json`文件的操作:


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
<td>将中间各个模块的可视化图像保存在png格式的图像</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_markdown()</code></td>
<td>将图像或者PDF文件中的每一页分别保存为markdown格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>将文件中的表格保存为html格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>将文件中的表格保存为xlsx格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
</table>

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：
    - `input_path`: `(str)` 待预测图像或者PDF的输入路径

    - `page_index`: `(Union[int, None])` 如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`

    - `model_settings`: `(Dict[str, bool])` 配置产线所需的模型参数

        - `use_doc_preprocessor`: `(bool)` 控制是否启用文档预处理子产线
        - `use_general_ocr`: `(bool)` 控制是否启用 OCR 子产线
        - `use_seal_recognition`: `(bool)` 控制是否启用印章识别子产线
        - `use_table_recognition`: `(bool)` 控制是否启用表格识别子产线
        - `use_formula_recognition`: `(bool)` 控制是否启用公式识别子产线

    - `doc_preprocessor_res`: `(Dict[str, Union[List[float], str]])` 文档预处理结果字典，仅当`use_doc_preprocessor=True`时存在
        - `input_path`: `(str)` 文档预处理子产线接受的图像路径，当输入为`numpy.ndarray`时，保存为`None`，此处为`None`
        - `page_index`: `None`，此处的输入为`numpy.ndarray`，所以值为`None`
        - `model_settings`: `(Dict[str, bool])` 文档预处理子产线的模型配置参数
          - `use_doc_orientation_classify`: `(bool)` 控制是否启用文档图像方向分类子模块
          - `use_doc_unwarping`: `(bool)` 控制是否启用文本图像扭曲矫正子模块
        - `angle`: `(int)` 文档图像方向分类子模块的预测结果，启用时返回实际角度值

    - `parsing_res_list`: `(List[Dict])` 解析结果的列表，每个元素为一个字典，列表顺序为解析后的阅读顺序。
        - `layout_bbox`: `(np.ndarray)` 版面区域的边界框。
        - `label`: `(str)` key 为版面区域的标签，例如`text`, `table`等，内容为版面区域内的内容。
        - `layout`: `(str)` 版面排版类型，例如 `double`, `single` 等。

    - `overall_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` 全局 OCR 结果的字典
      - `input_path`: `(Union[str, None])` 图像OCR子产线接受的图像路径，当输入为`numpy.ndarray`时，保存为`None`
      - `page_index`: `None`，此处的输入为`numpy.ndarray`，所以值为`None`
      - `model_settings`: `(Dict)` OCR子产线的模型配置参数
      - `dt_polys`: `(List[numpy.ndarray])` 文本检测的多边形框列表。每个检测框由4个顶点坐标构成的numpy数组表示，数组shape为(4, 2)，数据类型为int16
      - `dt_scores`: `(List[float])` 文本检测框的置信度列表
      - `text_det_params`: `(Dict[str, Dict[str, int, float]])` 文本检测模块的配置参数
        - `limit_side_len`: `(int)` 图像预处理时的边长限制值
        - `limit_type`: `(str)` 边长限制的处理方式
        - `thresh`: `(float)` 文本像素分类的置信度阈值
        - `box_thresh`: `(float)` 文本检测框的置信度阈值
        - `unclip_ratio`: `(float)` 文本检测框的膨胀系数
        - `text_type`: `(str)` 文本检测的类型，当前固定为"general"

      - `text_type`: `(str)` 文本检测的类型，当前固定为"general"
      - `textline_orientation_angles`: `(List[int])` 文本行方向分类的预测结果。启用时返回实际角度值（如[0,0,1]
      - `text_rec_score_thresh`: `(float)` 文本识别结果的过滤阈值
      - `rec_texts`: `(List[str])` 文本识别结果列表，仅包含置信度超过`text_rec_score_thresh`的文本
      - `rec_scores`: `(List[float])` 文本识别的置信度列表，已按`text_rec_score_thresh`过滤
      - `rec_polys`: `(List[numpy.ndarray])` 经过置信度过滤的文本检测框列表，格式同`dt_polys`

    - `text_paragraphs_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` 段落OCR结果，版面类型非表格、印章和公式类型的段落OCR结果
        - `rec_polys`: `(List[numpy.ndarray])` 文本检测框列表，格式同`dt_polys`
        - `rec_texts`: `(List[str])` 文本识别结果列表
        - `rec_scores`: `(List[float])` 文本识别结果的置信度列表
        - `rec_boxes`: `(numpy.ndarray)` 检测框的矩形边界框数组，shape为(n, 4)，dtype为int16。每一行表示一个

    - `formula_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` 公式识别结果列表，每个元素为一个字典
        - `rec_formula`: `(str)` 公式识别结果
        - `rec_polys`: `(numpy.ndarray)` 公式检测框，shape为(4, 2)，dtype为int16
        - `formula_region_id`: `(int)` 公式所在的区域编号

    - `seal_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` 印章识别结果列表，每个元素为一个字典
        - `input_path`: `(str)` 印章图像的输入路径
        - `page_index`: `None`，此处的输入为`numpy.ndarray`，所以值为`None`
        - `model_settings`: `(Dict)` 印章识别子产线的模型配置参数
        - `dt_polys`: `(List[numpy.ndarray])` 印章检测框列表，格式同`dt_polys`
        - `text_det_params`: `(Dict[str, Dict[str, int, float]])` 印章检测模块的配置参数, 具体参数含义同上
        - `text_type`: `(str)` 印章检测的类型，当前固定为"seal"
        - `text_rec_score_thresh`: `(float)` 印章识别结果的过滤阈值
        - `rec_texts`: `(List[str])` 印章识别结果列表，仅包含置信度超过`text_rec_score_thresh`的文本
        - `rec_scores`: `(List[float])` 印章识别的置信度列表，已按`text_rec_score_thresh`过滤
        - `rec_polys`: `(List[numpy.ndarray])` 经过置信度过滤的印章检测框列表，格式同`dt_polys`
        - `rec_boxes`: `(numpy.ndarray)` 检测框的矩形边界框数组，shape为(n, 4)，dtype为int16。每一行表示一个矩形

    - `table_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` 表格识别结果列表，每个元素为一个字典
        - `cell_box_list`: `(List[numpy.ndarray])` 表格单元格的边界框列表
        - `pred_html`: `(str)` 表格的HTML格式字符串
        - `table_ocr_pred`: `(dict)` 表格的OCR识别结果
            - `rec_polys`: `(List[numpy.ndarray])` 单元格的检测框列表
            - `rec_texts`: `(List[str])` 单元格的识别结果
            - `rec_scores`: `(List[float])` 单元格的识别置信度
            - `rec_boxes`: `(numpy.ndarray)` 检测框的矩形边界框数组，shape为(n, 4)，dtype为int16。每一行表示一个矩形

- 调用`save_to_json()` 方法会将上述内容保存到指定的 `save_path` 中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于 json 文件不支持保存numpy数组，因此会将其中的 `numpy.array` 类型转换为列表形式。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的 `save_path` 中，如果指定为目录，则会将版面区域检测可视化图像、全局OCR可视化图像、版面阅读顺序可视化图像等内容保存，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)
- 调用`save_to_markdown()` 方法会将转化后的 Markdown 文件保存到指定的 `save_path` 中，保存的文件路径为`save_path/{your_img_basename}.md`，如果输入是 PDF 文件，建议直接指定目录，否责多个 markdown 文件会被覆盖。

此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：
<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>json</code></td>
<td>获取预测的 <code>json</code> 格式的结果</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">获取格式为 <code>dict</code> 的可视化图像</td>
</tr>
<tr>
</tr>
<tr>
<td rowspan="3"><code>markdown</code></td>
<td rowspan="3">获取格式为 <code>dict</code> 的 markdown 结果</td>
</tr>
<tr>
</tr>
<tr>
</tr>
</tbody>
</table>

- `json` 属性获取的预测结果为字典类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。
- `img` 属性返回的预测结果是一个字典类型的数据。其中，键分别为 `layout_det_res`、`overall_ocr_res`、`text_paragraphs_ocr_res`、`formula_res_region1`、`table_cell_img` 和 `seal_res_region1`，对应的值是 `Image.Image` 对象：分别用于显示版面区域检测、OCR、OCR文本段落、公式、表格和印章结果的可视化图像。如果没有使用可选模块，则字典中只包含 `layout_det_res`。
- `markdown` 属性返回的预测结果是一个字典类型的数据。其中，键分别为 `markdown_texts` 和 `markdown_images`，对应的值分别是 markdown 文本和用于在 Markdown 中显示的图像（`Image.Image` 对象）。

此外，您可以获取版面解析产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：
```
paddlex --get_pipeline_config layout_parsing_v2 --save_path ./my_path
```
若您获取了配置文件，即可对版面解析产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。示例如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/layout_parsing_v2.yaml")

output = pipeline.predict(
    input="./layout_parsing_v2_demo.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path="output") ## 保存当前图像的结构化json结果
    res.save_to_markdown(save_path="output") ## 保存当前图像的markdown格式的结果
```
<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改通用版面解析v2产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。同时，CLI 预测也支持传入配置文件，`--pipeline` 指定配置文件的路径即可。

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
<p>进行版面解析。</p>
<p><code>POST /layout-parsing</code></p>
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
<td><code>integer</code>｜<code>null</code></td>
<td>文件类型。<code>0</code>表示PDF文件，<code>1</code>表示图像文件。若请求体无此属性，则将根据URL推断文件类型。</td>
<td>否</td>
</tr>

<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>use_doc_orientation_classify</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>use_doc_unwarping</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useTextlineOrientation</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>use_textline_orientation</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useGeneralOcr</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>use_general_ocr</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useSealRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>use_seal_recognition</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useTableRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>use_table_recognition</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useFormulaRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>use_formula_recognition</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>text_det_limit_side_len</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>text_det_limit_type</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>text_det_thresh</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>text_det_box_thresh</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>text_det_unclip_ratio</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>text_rec_score_thresh</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>seal_det_limit_side_len</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>seal_det_limit_type</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>seal_det_thresh</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>seal_det_box_thresh</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>seal_det_unclip_ratio</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>sealRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>seal_rec_score_thresh</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>layout_threshold</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>layout_nms</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>layout_unclip_ratio</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>layout_merge_bboxes_mode</code> 参数说明。</td>
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
<td><code>layoutParsingResults</code></td>
<td><code>array</code></td>
<td>版面解析结果。数组长度为1（对于图像输入）或文档页数与10中的较小者（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中每一页的处理结果。</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>输入数据信息。</td>
</tr>
</tbody>
</table>
<p><code>layoutParsingResults</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>产线对象的 <code>predict</code> 方法生成结果的 JSON 表示中 <code>res</code> 字段的简化版本，其中去除了 <code>input_path</code> 字段</td>
</tr>
<tr>
<td><code>markdown</code></td>
<td><code>object</code></td>
<td>markdown结果。</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>输入图像和预测结果图像的键值对。图像为JPEG格式，使用Base64编码。</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>输入图像。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<p><code>markdown</code>为一个<code>object</code>，具有如下属性：</p>
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
<td><code>text</code></td>
<td><code>string</code></td>
<td>Markdown文本。</td>
</tr>
<tr>
<td><code>images</code></td>
<td><code>object</code></td>
<td>Markdown图片相对路径和base64编码图像的键值对。</td>
</tr>
</tbody>
</table></details>
<details><summary>多语言调用服务示例</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/layout-parsing" # 服务URL

image_path = "./demo.jpg"

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "file": image_data, # Base64编码的文件内容或者文件URL
}

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()["result"]
print("\nDetected layout elements:")
for res in result["layoutParsingResults"]:
    print(res["prunedResult"])
    md_dir = pathlib.Path(f"markdown_{i}")
    md_dir.mkdir(exist_ok=True)
    (md_dir / "doc.md").write_text(res["markdown"]["text"])
    for img_path, img in res["markdown"]["images"].items():
        img_path = md_dir / img_path
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(base64.b64decode(img))
    print(f"Markdown document saved at {md_dir / 'doc.md'}")
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果通用版面解析v2产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升通用版面解析v2产线的在您的场景中的识别效果。

### 4.1 模型微调

由于通用版面解析v2产线包含若干模块，模型产线的效果不及预期可能来自于其中任何一个模块。您可以对提取效果差的 case 进行分析，通过可视化图像，确定是哪个模块存在问题，并参考以下表格中对应的微调教程链接进行模型微调。


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
<td>版面区域检测不准，如印章、表格未检出等</td>
<td>版面区域检测模块</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/layout_detection.md">链接</a></td>
</tr>
<tr>
<td>表格结构识别不准</td>
<td>表格结构识别模块</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md">链接</a></td>
</tr>
<tr>
<td>公式识别不准</td>
<td>公式识别模块</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/formula_recognition.md">链接</a></td>
</tr>
<tr>
<td>印章文本存在漏检</td>
<td>印章文本检测模块</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/seal_text_detection.md">链接</a></td>
</tr>
<tr>
<td>文本存在漏检</td>
<td>文本检测模块</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/text_detection.md">链接</a></td>
</tr>
<tr>
<td>文本内容都不准</td>
<td>文本识别模块</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/text_recognition.md">链接</a></td>
</tr>
<tr>
<td>垂直或者旋转文本行矫正不准</td>
<td>文本行方向分类模块</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/textline_orientation_classification.md">链接</a></td>
</tr>
<tr>
<td>整图旋转矫正不准</td>
<td>文档图像方向分类模块</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md">链接</a></td>
</tr>
<tr>
<td>图像扭曲矫正不准</td>
<td>文本图像矫正模块</td>
<td>暂不支持微调</td>
</tr>
</tbody>
</table>

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```yaml
......
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayout-L
    model_dir: null # 替换为微调后的版面区域检测模型权重路径
......
SubPipelines:
  GeneralOCR:
    pipeline_name: OCR
    text_type: general
    use_doc_preprocessor: False
    use_textline_orientation: False
    SubModules:
      TextDetection:
        module_name: text_detection
        model_name: PP-OCRv4_server_det
        model_dir: null # 替换为微调后的文本测模型权重路径
        limit_side_len: 960
        limit_type: max
        thresh: 0.3
        box_thresh: 0.6
        unclip_ratio: 2.0

      TextRecognition:
        module_name: text_recognition
        model_name: PP-OCRv4_server_rec_doc
        model_dir: null # 替换为微调后的文本识别模型权重路径
        batch_size: 1
        score_thresh: 0
......
```
随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device`参数</b>即可完成不同硬件之间的无缝切换。

例如，您使用昇腾 NPU 进行版面解析产线的推理，使用的 CLI 命令为：

```bash
paddlex --pipeline layout_parsing_v2 \
        --input layout_parsing_v2_demo.png  \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --save_path ./output \
        --device npu:0
```

当然，您也可以在 Python 脚本中 `create_pipeline()` 时或者 `predict()` 时指定硬件设备。

若您想在更多种类的硬件上使用通用版面解析v2产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
