---
comments: true
---

# 公式识别产线使用教程

## 1. 公式识别产线介绍

公式识别是一种自动从文档或图像中识别和提取LaTeX公式内容及其结构的技术，广泛应用于数学、物理、计算机科学等领域的文档编辑和数据分析。通过使用计算机视觉和机器学习算法，公式识别能够将复杂的数学公式信息转换为可编辑的LaTeX格式，方便用户进一步处理和分析数据。

公式识别产线用于解决公式识别任务，提取图片中的公式信息以LaTeX源码形式输出，本产线是一个集成了百度飞桨视觉团队自研的先进公式识别模型PP-FormulaNet 和业界知名公式识别模型 UniMERNet的端到端公式识别系统，支持简单印刷公式、复杂印刷公式、手写公式的识别，并在此基础上，增加了对图像的方向矫正和扭曲矫正功能。基于本产线，可实现公式内容精准预测，使用场景覆盖教育、科研、金融、制造等各个领域。本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。


<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/03.png" style="width: 70%"/>

<b>公式识别产线中包含必选的公式识别模块，</b>以及可选的版面区域检测模块、文档图像方向分类模块和文本图像矫正模块。其中，文档图像方向分类模块和文本图像矫正模块作为文档预处理子产线被集成到公式识别产线中。每个模块都包含多个模型，您可以根据下方的基准测试数据选择使用的模型。

<b>如果您更注重模型的精度，请选择精度较高的模型；如果您更在意模型的推理速度，请选择推理速度较快的模型；如果您关注模型的存储大小，请选择存储体积较小的模型。</b>
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
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>基于PP-LCNet_x1_0的文档图像分类模型，含有四个类别，即0度，90度，180度，270度</td>
</tr>
</tbody>
</table>

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
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">训练模型</a></td>
<td>0.179</td>
<td>30.3 M</td>
<td>高精度文本图像矫正模型</td>
</tr>
</tbody>
</table>
<p><b>版面区域检测模块（可选）：</b></p>


* <b>版面检测模型，包含20个常见的类别：文档标题、段落标题、文本、页码、摘要、目录、参考文献、脚注、页眉、页脚、算法、公式、公式编号、图像、表格、图和表标题（图标题、表格标题和图表标题）、印章、图表、侧栏文本和参考文献内容</b>
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
<td>PP-DocLayout_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">训练模型</a></td>
<td>83.2</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / - </td>
<td>126.01 M</td>
<td>基于RT-DETR-L在包含中英文论文、多栏杂志、报纸、PPT、合同、书本、试卷、研报、古籍、日文文档、竖版文字文档等场景的自建数据集训练的更高精度版面区域定位模型</td>
</tr>
<tr>
</tbody>
</table>

* <b>版面检测模型，包含23个常见的类别：文档标题、段落标题、文本、页码、摘要、目录、参考文献、脚注、页眉、页脚、算法、公式、公式编号、图像、图表标题、表格、表格标题、印章、图表标题、图表、页眉图像、页脚图像、侧栏文本</b>
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

>❗ 以上列出的是版面区域检测模块重点支持的<b>4个核心模型</b>，该模块总共支持<b>7个全量模型</b>，包含多个预定义了不同类别的模型，完整的模型列表如下：

<details><summary> 👉模型列表详情</summary>

* <b>17类版面区域检测模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</b>
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
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>87.4</td>
<td>13.6</td>
<td>46.2</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>89.0</td>
<td>17.2</td>
<td>160.2</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的效率精度均衡版面区域定位模型</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>98.3</td>
<td>115.1</td>
<td>3827.2</td>
<td>470.2</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型</td>
</tr>
</tbody>
</table>


* <b>23类版面区域检测模型，包含23个常见的类别：文档标题、段落标题、文本、页码、摘要、目录、参考文献、脚注、页眉、页脚、算法、公式、公式编号、图像、图表标题、表格、表格标题、印章、图表标题、图表、页眉图像、页脚图像、侧栏文本</b>
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

* <b>版面检测模型，包含20个常见的类别：文档标题、段落标题、文本、页码、摘要、目录、参考文献、脚注、页眉、页脚、算法、公式、公式编号、图像、表格、图和表标题（图标题、表格标题和图表标题）、印章、图表、侧栏文本和参考文献内容</b>
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
<td>PP-DocLayout_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">训练模型</a></td>
<td>83.2</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / - </td>
<td>126.01 M</td>
<td>基于RT-DETR-L在包含中英文论文、多栏杂志、报纸、PPT、合同、书本、试卷、研报、古籍、日文文档、竖版文字文档等场景的自建数据集训练的更高精度版面区域定位模型</td>
</tr>
<tr>
</tbody>
</table>

</details>

<p><b>公式识别模块：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>En-BLEU(%)</th>
<th>Zh-BLEU(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<td>UniMERNet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UniMERNet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">训练模型</a></td>
<td>85.91</td>
<td>43.50</td>
<td>2266.96/-</td>
<td>-/-</td>
<td>1.53 G</td>
<td>UniMERNet是由上海AI Lab研发的一款公式识别模型。该模型采用Donut Swin作为编码器，MBartDecoder作为解码器，并通过在包含简单公式、复杂公式、扫描捕捉公式和手写公式在内的一百万数据集上进行训练，大幅提升了模型对真实场景公式的识别准确率</td>
<tr>
<td>PP-FormulaNet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">训练模型</a></td>
<td>87.00</td>
<td>45.71</td>
<td>202.25/-</td>
<td>-/-</td>
<td>224 M</td>
<td rowspan="2">PP-FormulaNet 是由百度飞桨视觉团队开发的一款先进的公式识别模型，支持5万个常见LateX源码词汇的识别。PP-FormulaNet-S 版本采用了 PP-HGNetV2-B4 作为其骨干网络，通过并行掩码和模型蒸馏等技术，大幅提升了模型的推理速度，同时保持了较高的识别精度，适用于简单印刷公式、跨行简单印刷公式等场景。而 PP-FormulaNet-L 版本则基于 Vary_VIT_B 作为骨干网络，并在大规模公式数据集上进行了深入训练，在复杂公式的识别方面，相较于PP-FormulaNet-S表现出显著的提升，适用于简单印刷公式、复杂印刷公式、手写公式等场景。 </td>

</tr>
<td>PP-FormulaNet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">训练模型</a></td>
<td>90.36</td>
<td>45.78</td>
<td>1976.52/-</td>
<td>-/-</td>
<td>695 M</td>
<tr>
<td>PP-FormulaNet_plus-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-S_pretrained.pdparams">训练模型</a></td>
<td>88.71</td>
<td>53.32</td>
<td>191.69/-</td>
<td>-/-</td>
<td>248 M</td>
<td rowspan="3">PP-FormulaNet_plus 是百度飞桨视觉团队在 PP-FormulaNet 的基础上开发的增强版公式识别模型。与原版相比，PP-FormulaNet_plus 在训练中使用了更为丰富的公式数据集，包括中文学位论文、专业书籍、教材试卷以及数学期刊等多种来源。这一扩展显著提升了模型的识别能力。

其中，PP-FormulaNet_plus-M 和 PP-FormulaNet_plus-L 模型新增了对中文公式的支持，并将公式的最大预测 token 数从 1024 扩大至 2560，大幅提升了对复杂公式的识别性能。同时，PP-FormulaNet_plus-S 模型则专注于增强英文公式的识别能力。通过这些改进，PP-FormulaNet_plus 系列模型在处理复杂多样的公式识别任务时表现更加出色。 </td>
</tr>
<tr>
<td>PP-FormulaNet_plus-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-M_pretrained.pdparams">训练模型</a></td>
<td>91.45</td>
<td>89.76</td>
<td>1301.56/-</td>
<td>-/-</td>
<td>592 M</td>
</tr>
<tr>
<td>PP-FormulaNet_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-L_pretrained.pdparams">训练模型</a></td>
<td>92.22</td>
<td>90.64</td>
<td>1745.25/-</td>
<td>-/-</td>
<td>698 M</td>
</tr>

<tr>
<td>LaTeX_OCR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">训练模型</a></td>
<td>74.55</td>
<td>39.96</td>
<td>1244.61/-</td>
<td>-/-</td>
<td>99 M</td>
<td>LaTeX-OCR是一种基于自回归大模型的公式识别算法，通过采用 Hybrid ViT 作为骨干网络，transformer作为解码器，显著提升了公式识别的准确性。</td>
</tr>
</table>

<strong>测试环境说明:</strong>

  <ul>
      <li><b>性能测试环境</b>
          <ul>
            <li><strong>测试数据集：
             </strong>
                <ul>
                  <li>文档图像方向分类模型：PaddleX自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。</li>
                  <li> 文本图像矫正模型：<a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a>。</li>
                  <li>版面区域检测模型：PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志、合同、书本、试卷和研报等常见的 500 张文档类型图片。</li>
                  <li>17类区域检测模型：PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志和研报等常见的 892 张文档类型图片。</li>
                  <li>公式识别模型：PaddleX 内部自建公式识别测试集。</li>
                </ul>
             </li>
              <li><strong>硬件配置：</strong>
                  <ul>
                      <li>GPU：NVIDIA Tesla T4</li>
                      <li>CPU：Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>其他环境：Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
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



## 2. 快速开始
PaddleX 所提供的模型产线均可以快速体验效果，你可以在星河社区线体验公式识别产线的效果，也可以在本地使用命令行或 Python 体验公式识别产线的效果。

### 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/387976/webUI?source=appCenter)公式识别产线的效果，用官方提供的 Demo 图片进行识别，例如：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/formula_aistudio.png"/>

如果您对产线运行的效果满意，可以直接进行集成部署。您可以选择从云端下载部署包，也可以参考[2.2节本地体验](#22-本地体验)中的方法进行本地部署。如果对效果不满意，您可以利用私有数据<b>对产线中的模型进行微调训练</b>。如果您具备本地训练的硬件资源，可以直接在本地开展训练；如果没有，星河零代码平台提供了一键式训练服务，无需编写代码，只需上传数据后，即可一键启动训练任务。

### 2.2 本地体验
❗ 在本地使用公式识别产线前，请确保您已经按照[PaddleX安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。如果您希望选择性安装依赖，请参考安装教程中的相关说明。该产线对应的依赖分组为 `ocr`。

#### 2.2.1 命令行方式体验
一行命令即可快速体验公式识别产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/pipelines/general_formula_recognition_001.png)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline formula_recognition \
        --input general_formula_recognition_001.png \
        --use_layout_detection True \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --layout_threshold 0.5 \
        --layout_nms True \
        --layout_unclip_ratio  1.0 \
        --layout_merge_bboxes_mode "'large'"\
        --save_path ./output \
        --device gpu:0
```

<b>注：</b>PaddleX 官方模型默认从 HuggingFace 获取，如运行环境访问 HuggingFace 不便，可通过环境变量修改模型源为 BOS：`PADDLE_PDX_MODEL_SOURCE="BOS"`，未来将支持更多主流模型源；

相关的参数说明可以参考[2.2 Python脚本方式集成](#22-python脚本方式集成)中的参数说明。支持同时指定多个设备以进行并行推理，详情请参考 [产线并行推理](../../instructions/parallel_inference.md#指定多个推理设备)。

运行后，会将结果打印到终端上，结果如下：

```bash
{'res': {'input_path': 'general_formula_recognition_001.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9849682450294495, 'coordinate': [112.44934, 1074.9535, 663.01184, 1527.8729]}, {'cls_id': 2, 'label': 'text', 'score': 0.9809818267822266, 'coordinate': [112.761505, 160.65752, 662.4872, 414.6153]}, {'cls_id': 2, 'label': 'text', 'score': 0.9776389002799988, 'coordinate': [697.2594, 597.79034, 1246.6486, 746.98676]}, {'cls_id': 2, 'label': 'text', 'score': 0.9713516235351562, 'coordinate': [696.8539, 751.34827, 1246.7255, 875.0985]}, {'cls_id': 2, 'label': 'text', 'score': 0.9704778790473938, 'coordinate': [696.94653, 118.34517, 1247.156, 215.88304]}, {'cls_id': 2, 'label': 'text', 'score': 0.9666076898574829, 'coordinate': [113.78964, 795.92883, 662.4155, 893.1124]}, {'cls_id': 2, 'label': 'text', 'score': 0.9665274620056152, 'coordinate': [697.57666, 310.11346, 1247.8262, 408.27625]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9658663272857666, 'coordinate': [725.58777, 454.7463, 1183.6041, 577.70825]}, {'cls_id': 2, 'label': 'text', 'score': 0.9638860821723938, 'coordinate': [697.0996, 1481.4896, 1246.7573, 1606.951]}, {'cls_id': 2, 'label': 'text', 'score': 0.9625014066696167, 'coordinate': [698.0481, 945.31726, 1246.6857, 1017.86316]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9616876244544983, 'coordinate': [723.08606, 1304.9329, 1221.5212, 1433.7351]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9603673219680786, 'coordinate': [171.7144, 915.73035, 606.7387, 1026.4806]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9600085020065308, 'coordinate': [803.9297, 1040.9861, 1142.0847, 1099.2834]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9595950245857239, 'coordinate': [767.97546, 236.60234, 1180.0568, 291.77338]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9594774842262268, 'coordinate': [753.474, 1189.0759, 1154.2972, 1242.22]}, {'cls_id': 2, 'label': 'text', 'score': 0.9578917622566223, 'coordinate': [112.63562, 1532.9048, 663.50165, 1606.6626]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9559630155563354, 'coordinate': [179.8501, 566.4032, 599.8784, 620.08655]}, {'cls_id': 2, 'label': 'text', 'score': 0.953528642654419, 'coordinate': [113.30719, 490.1695, 662.4507, 546.1473]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9524251222610474, 'coordinate': [133.87987, 713.4751, 617.22253, 772.76074]}, {'cls_id': 2, 'label': 'text', 'score': 0.9439663290977478, 'coordinate': [114.23682, 642.90356, 662.42474, 692.64404]}, {'cls_id': 2, 'label': 'text', 'score': 0.9418545961380005, 'coordinate': [696.9155, 1120.0913, 1246.4368, 1168.2529]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9261635541915894, 'coordinate': [208.07648, 440.37366, 569.06274, 464.80005]}, {'cls_id': 7, 'label': 'formula', 'score': 0.920579731464386, 'coordinate': [843.95044, 899.7913, 1102.5901, 922.502]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.9113483428955078, 'coordinate': [1208.2693, 1061.5608, 1245.5154, 1085.906]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.910507321357727, 'coordinate': [1208.536, 1204.4053, 1245.2667, 1228.9385]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.909659206867218, 'coordinate': [1207.9521, 899.5998, 1245.6323, 923.64777]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.9072737097740173, 'coordinate': [1208.1934, 505.28503, 1245.4221, 529.67957]}, {'cls_id': 2, 'label': 'text', 'score': 0.9037376046180725, 'coordinate': [698.28625, 1262.6298, 1059.2609, 1285.3319]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9015905857086182, 'coordinate': [183.00662, 161.18907, 512.1726, 185.87654]}, {'cls_id': 2, 'label': 'text', 'score': 0.9007592797279358, 'coordinate': [721.2688, 412.6385, 1216.6045, 436.58807]}, {'cls_id': 2, 'label': 'text', 'score': 0.896611750125885, 'coordinate': [113.65086, 1048.154, 451.65906, 1070.5525]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8945134878158569, 'coordinate': [1208.4277, 1436.2947, 1245.581, 1459.6746]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.892361044883728, 'coordinate': [1220.0835, 251.3541, 1245.6968, 275.61313]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8889164924621582, 'coordinate': [116.8533, 265.05603, 307.09003, 289.93555]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8836347460746765, 'coordinate': [635.03656, 441.60486, 660.72943, 465.45752]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8798341751098633, 'coordinate': [635.3544, 928.30615, 660.801, 952.7964]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8789814710617065, 'coordinate': [635.7049, 582.2196, 660.6303, 606.67004]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.878730833530426, 'coordinate': [635.7882, 987.47, 660.4442, 1011.64307]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8710389137268066, 'coordinate': [635.75195, 731.1045, 660.3816, 754.42114]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8667231798171997, 'coordinate': [540.2363, 489.12628, 661.61475, 517.5715]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8634350299835205, 'coordinate': [117.003365, 1302.1862, 277.30377, 1326.8046]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8547449707984924, 'coordinate': [1072.7319, 1558.7657, 1242.1775, 1581.1473]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8485350608825684, 'coordinate': [373.27258, 290.88776, 518.7126, 316.89764]}, {'cls_id': 3, 'label': 'number', 'score': 0.8004439473152161, 'coordinate': [1236.0953, 62.784798, 1247.5856, 79.25625]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7683551907539368, 'coordinate': [258.48035, 189.60883, 325.27173, 213.70544]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7660759091377258, 'coordinate': [909.1803, 1584.3768, 999.6908, 1606.9323]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7568713426589966, 'coordinate': [701.779, 1532.1399, 891.00665, 1556.2966]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7310451865196228, 'coordinate': [701.32715, 1583.2532, 857.19336, 1605.7214]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7173376083374023, 'coordinate': [116.34981, 518.7822, 268.02765, 544.4822]}, {'cls_id': 7, 'label': 'formula', 'score': 0.6384682059288025, 'coordinate': [768.198, 601.2766, 793.38574, 622.52]}, {'cls_id': 7, 'label': 'formula', 'score': 0.6290246248245239, 'coordinate': [583.8933, 1253.2761, 609.42676, 1276.0557]}, {'cls_id': 7, 'label': 'formula', 'score': 0.6267108917236328, 'coordinate': [1095.0366, 1533.7931, 1163.5054, 1555.3956]}, {'cls_id': 7, 'label': 'formula', 'score': 0.6116171479225159, 'coordinate': [1214.5708, 337.85333, 1244.5303, 359.7912]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5208110809326172, 'coordinate': [701.8009, 336.1151, 729.7782, 357.85132]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5141622424125671, 'coordinate': [189.37997, 368.27167, 253.99014, 393.74164]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5111705660820007, 'coordinate': [778.9128, 365.22208, 801.6574, 385.84518]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5065054893493652, 'coordinate': [267.81232, 341.93176, 336.70538, 366.84973]}]}, 'formula_res_list': [{'rec_formula': '\\begin{aligned}{\\psi_{0}(M)-\\psi(M,z)=}&{{}\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}}\\frac{\\lambda^{2}c^{2}}{t_{\\operatorname{E}}^{2}\\operatorname{l n}(10)}\\times}\\\\ {}&{{}\\left.\\int_{0}^{z}d z^{\\prime}\\frac{d t}{d z^{\\prime}}\\;\\frac{\\partial\\phi}{\\partial L}\\right|_{L=\\lambda M c^{2}/t_{\\operatorname{E}}},}\\\\ \\end{aligned}', 'formula_region_id': 1, 'dt_polys': ([725.58777, 454.7463, 1183.6041, 577.70825],)}, {'rec_formula': '\\begin{aligned}{p(\\operatorname{l o g}_{10}}&{{}M|\\operatorname{l o g}_{10}\\sigma)=\\frac{1}{\\sqrt{2\\pi}\\epsilon_{0}}}\\\\ {}&{{}\\times\\operatorname{e x p}\\left[-\\frac{1}{2}\\left(\\frac{\\operatorname{l o g}_{10}M-a_{\\bullet}-b_{\\bullet}\\operatorname{l o g}_{10}\\sigma}{\\epsilon_{0}}\\right)^{2}\\right].}\\\\ \\end{aligned}', 'formula_region_id': 2, 'dt_polys': ([723.08606, 1304.9329, 1221.5212, 1433.7351],)}, {'rec_formula': '\\begin{aligned}{\\rho_{\\operatorname{B H}}}&{{}=\\int d M\\psi(M)M}\\\\ {}&{{}=\\frac{1-\\epsilon_{r}}{\\epsilon_{r}c^{2}}\\int_{0}^{\\infty}d z\\frac{d t}{d z}\\int d\\operatorname{l o g}_{10}L\\phi(L,z)L,}\\\\ \\end{aligned}', 'formula_region_id': 3, 'dt_polys': ([171.7144, 915.73035, 606.7387, 1026.4806],)}, {'rec_formula': '\\frac{d n}{d\\sigma}d\\sigma=\\psi_{*}\\left(\\frac{\\sigma}{\\sigma_{*}}\\right)^{\\alpha}\\frac{e^{-(\\sigma/\\sigma_{*})^{\\beta}}}{\\Gamma(\\alpha/\\beta)}\\beta\\frac{d\\sigma}{\\sigma}.', 'formula_region_id': 4, 'dt_polys': ([803.9297, 1040.9861, 1142.0847, 1099.2834],)}, {'rec_formula': '\\phi(L)\\equiv\\frac{d n}{d\\log_{10}L}=\\frac{\\phi_{*}}{(L/L_{*})^{\\gamma_{1}}+(L/L_{*})^{\\gamma_{2}}}.', 'formula_region_id': 5, 'dt_polys': ([767.97546, 236.60234, 1180.0568, 291.77338],)}, {'rec_formula': '\\psi_{0}(M)=\\int d\\sigma\\frac{p(\\log_{10}M|\\log_{10}\\sigma)}{M\\log(10)}\\frac{d n}{d\\sigma}(\\sigma),', 'formula_region_id': 6, 'dt_polys': ([753.474, 1189.0759, 1154.2972, 1242.22],)}, {'rec_formula': '\\langle\\dot{M}(M,t)\\rangle\\psi(M,t)=\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}c^{2}\\operatorname{l n}(10)}\\phi(L,t)\\frac{d L}{d M}.', 'formula_region_id': 7, 'dt_polys': ([179.8501, 566.4032, 599.8784, 620.08655],)}, {'rec_formula': '\\frac{\\partial\\psi}{\\partial t}(M,t)+\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}}\\frac{\\lambda^{2}c^{2}}{t_{\\operatorname{E}}^{2}\\operatorname{l n}(10)}\\left.\\frac{\\partial\\phi}{\\partial L}\\right|_{L=\\lambda M c^{2}/t_{\\operatorname{E}}}=0,', 'formula_region_id': 8, 'dt_polys': ([133.87987, 713.4751, 617.22253, 772.76074],)}, {'rec_formula': '\\phi(L,t)d\\log_{10}L=\\delta(M,t)\\psi(M,t)d M.', 'formula_region_id': 9, 'dt_polys': ([208.07648, 440.37366, 569.06274, 464.80005],)}, {'rec_formula': '\\log_{10}M=a_{\\bullet}+b_{\\bullet}\\log_{10}X.', 'formula_region_id': 10, 'dt_polys': ([843.95044, 899.7913, 1102.5901, 922.502],)}, {'rec_formula': 't_{E}=\\sigma_{T}c/4\\pi G m_{p}=4.5\\times10^{8}\\mathrm{y r}', 'formula_region_id': 11, 'dt_polys': ([183.00662, 161.18907, 512.1726, 185.87654],)}, {'rec_formula': '\\dot{M}\\:=\\:(1\\:-\\:\\epsilon_{r})\\dot{M}_{\\mathrm{a c c}}', 'formula_region_id': 12, 'dt_polys': ([116.8533, 265.05603, 307.09003, 289.93555],)}, {'rec_formula': '\\langle\\dot{M}(M,t)\\rangle=', 'formula_region_id': 13, 'dt_polys': ([540.2363, 489.12628, 661.61475, 517.5715],)}, {'rec_formula': 'M_{*}=L_{*}t_{E}/\\lambda c^{2}', 'formula_region_id': 14, 'dt_polys': ([117.003365, 1302.1862, 277.30377, 1326.8046],)}, {'rec_formula': 'a_{\\bullet}=8.32\\pm0.05', 'formula_region_id': 15, 'dt_polys': ([1072.7319, 1558.7657, 1242.1775, 1581.1473],)}, {'rec_formula': '\\phi(L,t)d\\operatorname{l o g}_{10}L', 'formula_region_id': 16, 'dt_polys': ([373.27258, 290.88776, 518.7126, 316.89764],)}, {'rec_formula': '\\epsilon_{r}\\dot{M}_{\\mathrm{a c c}}', 'formula_region_id': 17, 'dt_polys': ([258.48035, 189.60883, 325.27173, 213.70544],)}, {'rec_formula': '\\epsilon_{0}=0.38', 'formula_region_id': 18, 'dt_polys': ([909.1803, 1584.3768, 999.6908, 1606.9323],)}, {'rec_formula': 'X=\\sigma/200\\mathrm{k m}\\mathrm{s}^{-1}', 'formula_region_id': 19, 'dt_polys': ([701.779, 1532.1399, 891.00665, 1556.2966],)}, {'rec_formula': 'b_{\\bullet}=5.64\\pm\\widetilde{0.32}', 'formula_region_id': 20, 'dt_polys': ([701.32715, 1583.2532, 857.19336, 1605.7214],)}, {'rec_formula': '\\delta(M,t)\\dot{M}(M,t)', 'formula_region_id': 21, 'dt_polys': ([116.34981, 518.7822, 268.02765, 544.4822],)}, {'rec_formula': '\\psi_{0}', 'formula_region_id': 22, 'dt_polys': ([768.198, 601.2766, 793.38574, 622.52],)}, {'rec_formula': 'L_{*}', 'formula_region_id': 23, 'dt_polys': ([583.8933, 1253.2761, 609.42676, 1276.0557],)}, {'rec_formula': 'M-\\sigma', 'formula_region_id': 24, 'dt_polys': ([1095.0366, 1533.7931, 1163.5054, 1555.3956],)}, {'rec_formula': 'L_{*}', 'formula_region_id': 25, 'dt_polys': ([1214.5708, 337.85333, 1244.5303, 359.7912],)}, {'rec_formula': '\\mathrm{A^{\\prime\\prime}}', 'formula_region_id': 26, 'dt_polys': ([701.8009, 336.1151, 729.7782, 357.85132],)}, {'rec_formula': '\\phi(L,t)', 'formula_region_id': 27, 'dt_polys': ([189.37997, 368.27167, 253.99014, 393.74164],)}, {'rec_formula': '\\gamma_{2}', 'formula_region_id': 28, 'dt_polys': ([778.9128, 365.22208, 801.6574, 385.84518],)}, {'rec_formula': '\\delta(\\bar{M,t})', 'formula_region_id': 29, 'dt_polys': ([267.81232, 341.93176, 336.70538, 366.84973],)}]}}
```
运行结果参数说明可以参考[2.2 Python脚本方式集成](#22-python脚本方式集成)中的结果解释。


可视化结果保存在`save_path`下，其中公式识别的可视化结果如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/04_paddleocr3.png" style="width: 70%"/>

<b> 如果您需要对公式识别产线进行可视化，需要运行如下命令来对LaTeX渲染环境进行安装。目前公式识别产线可视化只支持Ubuntu环境，其他环境暂不支持。对于复杂公式，LaTeX 结果可能包含部分高级的表示，Markdown等环境中未必可以成功显示：</b>

```bash
sudo apt-get update
sudo apt-get install texlive texlive-latex-base texlive-xetex latex-cjk-all texlive-latex-extra -y
```
<b>备注</b>： 由于公式识别可视化过程中需要对每张公式图片进行渲染，因此耗时较长，请您耐心等待。

#### 2.2.2 Python脚本方式集成
几行代码即可完成产线的快速推理，以公式识别产线为例：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="formula_recognition")

output = pipeline.predict(
    input="./general_formula_recognition_001.png",
    use_layout_detection=True ,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    layout_threshold=0.5,
    layout_nms=True,
    layout_unclip_ratio=1.0,
    layout_merge_bboxes_mode="large"
)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")
```


在上述 Python 脚本中，执行了如下几个步骤：

（1）通过 `create_pipeline()` 实例化公式识别产线对象，具体参数说明如下：

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
<td>产线推理设备。支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。支持同时指定多个设备以进行并行推理，详情请参考产线并行推理文档。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理插件。如果为 <code>None</code>，则使用配置文件或 <code>config</code> 中的配置。</td>
<td><code>bool</code> | <code>None</code></td>
<td>无</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>高性能推理配置</td>
<td><code>dict</code> | <code>None</code></td>
<td>无</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

（2）调用 公式识别产线对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：
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
<tbody>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型，必填</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/pipelines/general_formula_recognition_001.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
<li><b>List</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>None</code></td>
<tr>
<td><code>use_layout_detection</code></td>
<td>是否使用文档区域检测模块</td>
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
<td><code> layout_threshold</code></td>
<td>用于过滤掉低置信度预测结果的阈值；如果不指定，将默认使用PaddleX官方模型配置</td>
<td><code>float/dict/None</code></td>
<td>
<ul>
<li><b>float</b>，如 0.2， 表示过滤掉所有阈值小于0.2的目标框</li>
<li><b>字典</b>，字典的key为<b>int</b>类型，代表<code>cls_id</code>，val为<b>float</b>类型阈值。如 <code>{0: 0.45, 2: 0.48, 7: 0.4}</code>，表示对cls_id为0的类别应用阈值0.45、cls_id为1的类别应用阈值0.48、cls_id为7的类别应用阈值0.4</li>
<li><b>None</b>, 不指定，将默认使用PaddleX官方模型配置</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>是否使用NMS后处理，过滤重叠框；如果不指定，将默认使用PaddleX官方模型配置</td>
<td><code>bool/None</code></td>
<td>
<ul>
<li><b>bool</b>, <code>True</code> 或者 <code>False</code>； 表示使用/不使用NMS进行检测框的后处理过滤重叠框</li>
<li><b>None</b>, 不指定，将默认使用PaddleX官方模型配置</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>检测框的边长缩放倍数；如果不指定，将默认使用PaddleX官方模型配置</td>
<td><code>float/list/None</code></td>
<td>
<ul>
<li><b>float</b>, 大于0的浮点数，如 1.1 , 表示将模型输出的检测框中心不变，宽和高都扩张1.1倍</li>
<li><b>列表</b>, 如 [1.2, 1.5] , 表示将模型输出的检测框中心不变，宽度扩张1.2倍，高度扩张1.5倍</li>
<li><b>None</b>, 不指定，将默认使用PaddleX官方模型配置</li>
</ul>
</td>
<td><code>None</code></td>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>模型输出的检测框的合并处理模式；如果不指定，将默认使用PaddleX官方模型配置</td>
<td><code>string/None</code></td>
<td>
<ul>
<li><b>large</b>, 设置为large时，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留外部最大的框，删除重叠的内部框。</li>
<li><b>small</b>, 设置为small，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留内部被包含的小框，删除重叠的外部框。</li>
<li><b>union</b>, 不进行框的过滤处理，内外框都保留</li>
<li><b>None</b>, 不指定，将默认使用PaddleX官方模型配置</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tr></tr></tbody>
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

    - `model_settings`: `(Dict[str, bool])` 配置产线所需的模型参数

        - `use_doc_preprocessor`: `(bool)` 控制是否启用文档预处理子产线
        - `use_layout_detection`: `(bool)` 控制是否启用版面区域检测模块

    - `doc_preprocessor_res`: `(Dict[str, Union[str, Dict[str, bool], int]])` 文档预处理子产线的输出结果。仅当`use_doc_preprocessor=True`时存在
        - `input_path`: `(Union[str, None])` 图像预处理子产线接受的图像路径，当输入为`numpy.ndarray`时，保存为`None`
        - `model_settings`: `(Dict)` 预处理子产线的模型配置参数
            - `use_doc_orientation_classify`: `(bool)` 控制是否启用文档方向分类
            - `use_doc_unwarping`: `(bool)` 控制是否启用文档扭曲矫正
        - `angle`: `(int)` 文档方向分类的预测结果。启用时取值为[0,1,2,3]，分别对应[0°,90°,180°,270°]；未启用时为-1
    - `layout_det_res`: `(Dict[str, List[Dict]])` 版面区域检测模块的输出结果。仅当`use_layout_detection=True`时存在
        - `input_path`: `(Union[str, None])` 版面区域检测模块接收的图像路径，当输入为`numpy.ndarray`时，保存为`None`
        - `boxes`: `(List[Dict[int, str, float, List[float]]])` 版面区域检测预测结果列表
            - `cls_id`: `(int)` 版面区域检测预测的类别id
            - `label`: `(str)` 版面区域检测预测的类别
            - `score`: `(float)` 版面区域检测预测的类别置信度分数
            - `coordinate`: `(List[float])` 版面区域检测预测的边界框坐标，格式为[x_min, y_min, x_max, y_max]，其中(x_min, y_min)为左上角坐标，(x_max, y_max) 为右上角坐标
    - `formula_res_list`:  `(List[Dict[str, int, List[float]]])` 公式识别的预测结果列表
        - `rec_formula`: `(str)` 公式识别预测的LaTeX源码
        - `formula_region_id`: `(int)` 公式识别预测的id编号
        - `dt_polys`:  `(List[float])` 公式识别预测的边界框坐标，格式为[x_min, y_min, x_max, y_max]，其中(x_min, y_min)为左上角坐标，(x_max, y_max) 为右上角坐标

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_formula_res_img.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)

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
- `img` 属性返回的预测结果是一个字典类型的数据。其中，键分别为 `preprocessed_img`、 `layout_det_res`和 `formula_res_img`，对应的值是三个 `Image.Image` 对象：第一个用于展示图像预处理的可视化图像，第二个用于展示版面区域检测的可视化图像，第三个用于展示公式识别的可视化图像。如果没有使用图像预处理子模块，则字典中不包含 `preprocessed_img`；如果没有使用版面区域检测子模块，则字典中不包含`layout_det_res`。

此外，您可以获取公式识别产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config formula_recognition --save_path ./my_path
```
若您获取了配置文件，即可对公式识别产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。示例如下：
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/formula_recognition.yaml")

output = pipeline.predict(
    input="./general_formula_recognition_001.png",
    use_layout_detection=True ,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    layout_threshold=0.5,
    layout_nms=True,
    layout_unclip_ratio=1.0,
    layout_merge_bboxes_mode="large"
)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")
```

<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改公式识别产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。同时，CLI 预测也支持传入配置文件，`--pipeline` 指定配置文件的路径即可。



## 3. 开发集成/部署
如果公式识别产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将公式识别产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

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
<p>获取图像公式识别结果。</p>
<p><code>POST /formula-recognition</code></p>
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
<td>服务器可访问的图像文件或PDF文件的URL，或上述类型文件内容的Base64编码结果。默认对于超过10页的PDF文件，只有前10页的内容会被处理。<br /> 要解除页数限制，请在产线配置文件中添加以下配置：
<pre><code>Serving:
  extra:
    max_num_input_imgs: null
</code></pre>
</td>
<td>是</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>文件类型。<code>0</code>表示PDF文件，<code>1</code>表示图像文件。若请求体无此属性，则将根据URL推断文件类型。</td>
<td>否</td>
</tr>
<tr>
<td><code>visualize</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>是否返回可视化结果图以及处理过程中的中间图像等。
<ul style="margin: 0 0 0 1em; padding-left: 0em;">
<li>传入 <code>true</code>：返回图像。</li>
<li>传入 <code>false</code>：不返回图像。</li>
<li>若请求体中未提供该参数或传入 <code>null</code>：遵循产线配置文件<code>Serving.visualize</code> 的设置。</li>
</ul>
<br/>例如，在产线配置文件中添加如下字段：<br/>
<pre><code>Serving:
  visualize: False
</code></pre>
将默认不返回图像，通过请求体中的<code>visualize</code>参数可以覆盖默认行为。如果请求体和配置文件中均未设置（或请求体传入<code>null</code>、配置文件中未设置），则默认返回图像。
</td>
<td>否</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_doc_orientation_classify</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_doc_unwarping</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useLayoutDetection</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_layout_detection</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>object</code> | </code><code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>layout_threshold</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>layout_nms</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>layout_unclip_ratio</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>layout_merge_bboxes_mode</code> 参数相关说明。</td>
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
<td><code>formulaRecResults</code></td>
<td><code>object</code></td>
<td>公式识别结果。数组长度为1（对于图像输入）或实际处理的文档页数（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中实际处理的每一页的结果。</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>输入数据信息。</td>
</tr>
</tbody>
</table>
<p><code>formulaRecResults</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td>产线对象的 <code>predict</code> 方法生成结果的 JSON 表示中 <code>res</code> 字段的简化版本，其中去除了 <code>input_path</code> 和 <code>page_index</code> 字段。</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>参见产线预测结果的 <code>img</code> 属性说明。图像为JPEG格式，使用Base64编码。</td>
</tr>
<tr>
<td><code>inputImage</code> | <code>null</code></td>
<td><code>string</code></td>
<td>输入图像。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
</details>
<details><summary>多语言调用服务示例</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/formula-recognition"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["formulaRecResults"]):
    print(res["prunedResult"])
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &lt;fstream&gt;
#include &lt;vector&gt;
#include &lt;string&gt;
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client("localhost", 8080);

    const std::string filePath = "./demo.jpg";

    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return 1;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }

    std::string bufferStr(buffer.data(), static_cast<size_t>(size));
    std::string encodedFile = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["file"] = encodedFile;
    jsonObj["fileType"] = 1;

    auto response = client.Post("/formula-recognition", jsonObj.dump(), "application/json");

    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        if (!result.is_object() || !result["formulaRecResults"].is_array()) {
            std::cerr << "Unexpected response format." << std::endl;
            return 1;
        }

        for (size_t i = 0; i < result["formulaRecResults"].size(); ++i) {
            auto res = result["formulaRecResults"][i];

            if (res.contains("prunedResult")) {
                std::cout << "Recognized formula: " << res["prunedResult"].dump() << std::endl;
            }

            if (res.contains("outputImages") && res["outputImages"].is_object()) {
                for (auto& [imgName, imgData] : res["outputImages"].items()) {
                    std::string outputPath = imgName + "_" + std::to_string(i) + ".jpg";
                    std::string decodedImage = base64::from_base64(imgData.get<std::string>());

                    std::ofstream outFile(outputPath, std::ios::binary);
                    if (outFile.is_open()) {
                        outFile.write(decodedImage.c_str(), decodedImage.size());
                        outFile.close();
                        std::cout << "Saved image: " << outputPath << std::endl;
                    } else {
                        std::cerr << "Failed to write image: " << outputPath << std::endl;
                    }
                }
            }
        }
    } else {
        std::cerr << "Request failed." << std::endl;
        if (response) {
            std::cerr << "HTTP status: " << response->status << std::endl;
            std::cerr << "Response body: " << response->body << std::endl;
        }
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
        String API_URL = "http://localhost:8080/formula-recognition";
        String imagePath = "./demo.jpg";

        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String base64Image = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("file", base64Image);
        payload.put("fileType", 1);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.create(JSON, payload.toString());

        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode root = objectMapper.readTree(responseBody);
                JsonNode result = root.get("result");

                JsonNode formulaRecResults = result.get("formulaRecResults");
                for (int i = 0; i < formulaRecResults.size(); i++) {
                    JsonNode item = formulaRecResults.get(i);
                    int finalI = i;

                    JsonNode prunedResult = item.get("prunedResult");
                    System.out.println("Pruned Result [" + i + "]: " + prunedResult.toString());

                    JsonNode outputImages = item.get("outputImages");
                    if (outputImages != null && outputImages.isObject()) {
                        outputImages.fieldNames().forEachRemaining(imgName -> {
                            try {
                                String imgBase64 = outputImages.get(imgName).asText();
                                byte[] imgBytes = Base64.getDecoder().decode(imgBase64);
                                String imgPath = imgName + "_" + finalI + ".jpg";
                                try (FileOutputStream fos = new FileOutputStream(imgPath)) {
                                    fos.write(imgBytes);
                                    System.out.println("Saved image: " + imgPath);
                                }
                            } catch (IOException e) {
                                System.err.println("Failed to save image: " + e.getMessage());
                            }
                        });
                    }
                }
            } else {
                System.err.println("Request failed with HTTP code: " + response.code());
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
    "os"
)

func main() {
    API_URL := "http://localhost:8080/formula-recognition"
    filePath := "./demo.jpg"

    fileBytes, err := ioutil.ReadFile(filePath)
    if err != nil {
        fmt.Printf("Error reading file: %v\n", err)
        return
    }
    fileData := base64.StdEncoding.EncodeToString(fileBytes)

    payload := map[string]interface{}{
        "file":     fileData,
        "fileType": 1,
    }
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Printf("Error marshaling payload: %v\n", err)
        return
    }

    client := &http.Client{}
    req, err := http.NewRequest("POST", API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Printf("Error creating request: %v\n", err)
        return
    }
    req.Header.Set("Content-Type", "application/json")

    res, err := client.Do(req)
    if err != nil {
        fmt.Printf("Error sending request: %v\n", err)
        return
    }
    defer res.Body.Close()

    if res.StatusCode != http.StatusOK {
        fmt.Printf("Unexpected status code: %d\n", res.StatusCode)
        return
    }

    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Printf("Error reading response body: %v\n", err)
        return
    }

    type FormulaRecResult struct {
        PrunedResult map[string]interface{} `json:"prunedResult"`
        OutputImages map[string]string      `json:"outputImages"`
        InputImage   *string                `json:"inputImage"`
    }

    type Response struct {
        Result struct {
            FormulaRecResults []FormulaRecResult `json:"formulaRecResults"`
            DataInfo          interface{}        `json:"dataInfo"`
        } `json:"result"`
    }

    var respData Response
    if err := json.Unmarshal(body, &respData); err != nil {
        fmt.Printf("Error unmarshaling response: %v\n", err)
        return
    }

    for i, res := range respData.Result.FormulaRecResults {
        fmt.Printf("Result %d - prunedResult: %+v\n", i, res.PrunedResult)

        for imgName, imgData := range res.OutputImages {
            imgBytes, err := base64.StdEncoding.DecodeString(imgData)
            if err != nil {
                fmt.Printf("Error decoding image %s_%d: %v\n", imgName, i, err)
                continue
            }

            filename := fmt.Sprintf("%s_%d.jpg", imgName, i)
            if err := os.WriteFile(filename, imgBytes, 0644); err != nil {
                fmt.Printf("Error saving image %s: %v\n", filename, err)
                continue
            }
            fmt.Printf("Saved image to %s\n", filename)
        }
    }
}
</code></pre></details>

<details><summary>C#</summary>

<pre><code class="language-csharp">using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = "http://localhost:8080/formula-recognition";
    static readonly string inputFilePath = "./demo.jpg";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] fileBytes = File.ReadAllBytes(inputFilePath);
        string fileData = Convert.ToBase64String(fileBytes);

        var payload = new JObject
        {
            { "file", fileData },
            { "fileType", 1 }
        };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        JArray formulaRecResults = (JArray)jsonResponse["result"]["formulaRecResults"];
        for (int i = 0; i < formulaRecResults.Count; i++)
        {
            var res = formulaRecResults[i];
            Console.WriteLine($"[{i}] prunedResult:\n{res["prunedResult"]}");

            JObject outputImages = res["outputImages"] as JObject;
            if (outputImages != null)
            {
                foreach (var img in outputImages)
                {
                    string imgName = img.Key;
                    string base64Img = img.Value?.ToString();
                    if (!string.IsNullOrEmpty(base64Img))
                    {
                        string imgPath = $"{imgName}_{i}.jpg";
                        byte[] imageBytes = Convert.FromBase64String(base64Img);
                        File.WriteAllBytes(imgPath, imageBytes);
                        Console.WriteLine($"Output image saved at {imgPath}");
                    }
                }
            }
        }
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');
const path = require('path');

const API_URL = 'http://localhost:8080/formula-recognition';
const inputFilePath = './demo.jpg';
const fileType = 1;

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

const payload = {
  file: encodeImageToBase64(inputFilePath),
  fileType: fileType
};

axios.post(API_URL, payload)
  .then((response) => {
    const resultArray = response.data.result.formulaRecResults;

    resultArray.forEach((res, index) => {
      console.log(`\n[${index}] prunedResult:`);
      console.log(res.prunedResult);

      const outputImages = res.outputImages;
      if (outputImages) {
        Object.entries(outputImages).forEach(([imgName, base64Img]) => {
          const outputPath = `${imgName}_${index}.jpg`;
          fs.writeFileSync(outputPath, Buffer.from(base64Img, 'base64'));
          console.log(`Saved output image: ${outputPath}`);
        });
      } else {
        console.log(`[${index}] outputImages is null`);
      }
    });
  })
  .catch((error) => {
    console.error('API error：', error.message);
  });
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/formula-recognition";
$image_path = "./demo.jpg";

$image_data = base64_encode(file_get_contents($image_path));
$payload = array("file" => $image_data, "fileType" => 1);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)["result"]["formulaRecResults"];

foreach ($result as $i => $item) {
    echo "[$i] prunedResult:\n";
    print_r($item["prunedResult"]);

    if (!empty($item["outputImages"])) {
        foreach ($item["outputImages"] as $img_name => $base64_img) {
            $img_path = "{$img_name}_{$i}.jpg";
            file_put_contents($img_path, base64_decode($base64_img));
            echo "Output image saved at $img_path\n";
        }
    } else {
        echo "No outputImages found for item $i\n";
    }
}
?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/on_device_deployment.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。


## 4. 二次开发
如果公式识别产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升公式识别产线的在您的场景中的识别效果。

### 4.1 模型微调
由于公式识别产线包含若干模块，模型产线的效果如果不及预期，可能来自于其中任何一个模块。您可以对识别效果差的图片进行分析，进而确定是哪个模块存在问题，并参考以下表格中对应的微调教程链接进行模型微调。


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
<td>公式存在漏检</td>
<td>版面区域检测模块</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/layout_detection.html">链接</a></td>
</tr>
<tr>
<td>公式内容不准</td>
<td>公式识别模块</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/formula_recognition.html">链接</a></td>
</tr>
<tr>
<td>整图旋转矫正不准</td>
<td>文档图像方向分类模块</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html">链接</a></td>
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
...
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayout-L
    model_dir: null # 替换为微调后的版面区域检测模型权重路径
...

  FormulaRecognition:
    module_name: formula_recognition
    model_name: PP-FormulaNet-L
    model_dir: null # 替换为微调后的公式识别模型权重路径
    batch_size: 5

SubPipelines:
  DocPreprocessor:
    pipeline_name: doc_preprocessor
    use_doc_orientation_classify: True
    use_doc_unwarping: True
    SubModules:
      DocOrientationClassify:
        module_name: doc_text_orientation
        model_name: PP-LCNet_x1_0_doc_ori
        model_dir: null # 替换为微调后的文档图像方向分类模型权重路径
        batch_size: 1
...
```
随后， 参考[2. 快速开始](#2-快速开始)中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device`参数</b>即可完成不同硬件之间的无缝切换。

例如，您使用昇腾 NPU 进行 公式识别产线的推理，使用的 CLI 命令为：

```bash
paddlex --pipeline formula_recognition \
        --input general_formula_recognition_001.png \
        --use_layout_detection True \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --layout_threshold 0.5 \
        --layout_nms True \
        --layout_unclip_ratio  1.0 \
        --layout_merge_bboxes_mode "'large'"\
        --save_path ./output \
        --device npu:0

```
当然，您也可以在 Python 脚本中 `create_pipeline()` 时或者 `predict()` 时指定硬件设备。

若您想在更多种类的硬件上使用公式识别产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
