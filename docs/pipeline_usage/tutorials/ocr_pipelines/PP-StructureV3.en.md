---
comments: true
---

# PP-StructureV3 Pipeline Tutorial

## 1. Introduction to PP-StructureV3 pipeline
Layout parsing is a technology that extracts structured information from document images, primarily used to convert complex document layouts into machine-readable data formats. This technology is widely applied in document management, information extraction, and data digitization. By combining Optical Character Recognition (OCR), image processing, and machine learning algorithms, layout parsing can identify and extract text blocks, headings, paragraphs, images, tables, and other layout elements from documents. The process typically involves three main steps: layout detection, element analysis, and data formatting, ultimately generating structured document data to improve the efficiency and accuracy of data processing. <b>The PP-StructureV3 pipeline, based on the v1 pipeline, enhances the capabilities of layout region detection, table recognition, and formula recognition, and adds the ability to restore multi-column reading order and convert results into Markdown files. It performs excellently on various document data and can handle more complex document data.</b> This pipeline also provides flexible serving deployment options, supporting the use of multiple programming languages on various hardware. Moreover, this pipeline offers the capability for custom development; you can train and optimize models on your own dataset based on this pipeline, and the trained models can be seamlessly integrated.

<b>The PP-StructureV3 pipeline includes a mandatory layout region analysis module and a general OCR sub-pipeline,</b> as well as optional sub-pipelines for document image preprocessing, table recognition, seal recognition, and formula recognition. Each module contains multiple models, and you can choose the model based on the benchmark test data below.

### 1.1 Model benchmark data

<b>If you prioritize model accuracy, choose a high-accuracy model; if you prioritize model inference speed, choose a faster inference model; if you prioritize model storage size, choose a smaller storage model.</b>

> The inference time only includes the model inference time and does not include the time for pre- or post-processing.

<details><summary>👉 Model List Details</summary>
<p><b>Document Image Orientation Classification Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
<td>99.06</td>
<td>2.62 / 0.59</td>
<td>3.24 / 1.19</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, containing four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees.</td>
</tr>
</tbody>
</table>

<p><b>Text Image Rectification Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>CER</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
<td>0.179</td>
<td>19.05 / 19.05</td>
<td>- / 869.82</td>
<td>30.3</td>
<td>High-accuracy text image rectification model</td>
</tr>
</tbody>
</table>
</table>

<p><b>Layout Detection Module Model (Required):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout_plus-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">训练模型</a></td>
<td>83.2</td>
<td>53.03 / 17.23</td>
<td>634.62 / 378.32</td>
<td>126.01</td>
<td>基于RT-DETR-L在包含中英文论文、多栏杂志、报纸、PPT、合同、书本、试卷、研报、古籍、日文文档、竖版文字文档等场景的自建数据集训练的更高精度版面区域定位模型</td>
</tr>
<tr>
<td>PP-DocBlockLayout</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBlockLayout_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocBlockLayout_pretrained.pdparams">Training Model</a></td>
<td>95.9</td>
<td>34.60 / 28.54</td>
<td>506.43 / 256.83</td>
<td>123.92</td>
<td>基于RT-DETR-L在包含中英文论文、多栏杂志、报纸、PPT、合同、书本、试卷、研报、古籍、日文文档、竖版文字文档等场景的自建数据集训练的文档图像版面子模块检测模型</td>
</tr>
<tr>
<td>PP-DocLayout-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Training Model</a></td>
<td>90.4</td>
<td>33.59 / 33.59</td>
<td>503.01 / 251.08</td>
<td>123.76</td>
<td>A high-precision layout region detection model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exam papers, and research reports, based on RT-DETR-L.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>13.03 / 4.72</td>
<td>43.39 / 24.44</td>
<td>22.578</td>
<td>A layout region detection model with balanced precision and efficiency, trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exam papers, and research reports, based on PicoDet-L.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training Model</a></td>
<td>70.9</td>
<td>11.54 / 3.86</td>
<td>18.53 / 6.29</td>
<td>4.834</td>
<td>A high-efficiency layout region detection model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exam papers, and research reports, based on PicoDet-S.</td>
</tr>
</tbody>
</table>

<p><b>Table Structure Recognition Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wired_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">Training Model</a></td>
<td rowspan="2">69.65</td>
<td rowspan="2">85.92 / 85.92</td>
<td rowspan="2">- / 501.66</td>
<td rowspan="2">351M</td>
<td rowspan="2">The SLANeXt series are the latest table structure recognition models developed by the PaddlePaddle Vision Team. Compared to SLANet and SLANet_plus, SLANeXt focuses on table structure recognition and has dedicated weights trained for wired and wireless tables, significantly improving the recognition ability for both types, especially for wired tables.</td>
</tr>
<tr>
<td>SLANeXt_wireless</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wireless_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wireless_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
<p><b>Table Classification Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top1 Acc(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">Training Model</a></td>
<td>94.2</td>
<td>2.62 / 0.60</td>
<td>3.17 / 1.14</td>
<td>6.6</td>
</tr>
</table>

<p><b>Table Cell Detection Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wired_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">Training Model</a></td>
<td rowspan="2">82.7</td>
<td rowspan="2">33.47 / 27.02</td>
<td rowspan="2">402.55 / 256.56</td>
<td rowspan="2">124M</td>
<td rowspan="2">RT-DETR is the first real-time end-to-end object detection model. The PaddlePaddle Vision Team used RT-DETR-L as the base model and pre-trained it on a self-built table cell detection dataset, achieving good performance for both wired and wireless table cell detection.</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wireless_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
<p><b>Text Detection Module (Required):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv5_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams">Training Model</a></td>
<td>83.8</td>
<td>89.55 / 70.19</td>
<td>383.15 / 383.15</td>
<td>84.3</td>
<td>PP-OCRv5 server-side text detection model with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>79.0</td>
<td>10.67 / 6.36</td>
<td>57.77 / 28.15</td>
<td>4.7</td>
<td>PP-OCRv5 mobile-side text detection model with higher efficiency, suitable for deployment on edge devices</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Training Model</a></td>
<td>69.2</td>
<td>127.82 / 98.87</td>
<td>585.95 / 489.77</td>
<td>109</td>
<td>PP-OCRv4 server-side text detection model with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>63.8</td>
<td>9.87 / 4.17</td>
<td>56.60 / 20.79</td>
<td>4.7</td>
<td>PP-OCRv4 mobile-side text detection model with higher efficiency, suitable for deployment on edge devices</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>Accuracy comparable to PP-OCRv4_mobile_det</td>
<td>9.90 / 3.60</td>
<td>41.93 / 20.76</td>
<td>2.1</td>
<td>PP-OCRv3 mobile text detection model with higher efficiency, suitable for edge device deployment</td>
</tr>
<tr>
<td>PP-OCRv3_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_server_det_pretrained.pdparams">Training Model</a></td>
<td>Accuracy comparable to PP-OCRv4_server_det</td>
<td>119.50 / 75.00</td>
<td>379.35 / 318.35</td>
<td>102.1</td>
<td>PP-OCRv3 server text detection model with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
</tbody>
</table>

<p><b>Text Recognition Module Model (Required):</b></p>

* <b>Chinese Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">训练模型</a></td>
<td>86.38</td>
<td>8.46 / 2.36</td>
<td>31.21 / 31.21</td>
<td>81</td>
<td rowspan="2">PP-OCRv5_rec 是新一代文本识别模型。该模型致力于以单一模型高效、精准地支持简体中文、繁体中文、英文、日文四种主要语言，以及手写、竖版、拼音、生僻字等复杂文本场景的识别。在保持识别效果的同时，兼顾推理速度和模型鲁棒性，为各种场景下的文档理解提供高效、精准的技术支撑。</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>81.29</td>
<td>5.43 / 1.46</td>
<td>21.20 / 5.32</td>
<td>16</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>86.58</td>
<td>8.69 / 2.78</td>
<td>37.93 / 37.93</td>
<td>182</td>
<td>PP-OCRv4_server_rec_doc is trained on a mixed dataset of more Chinese document data and PP-OCR training data based on PP-OCRv4_server_rec. It has added the recognition capabilities for some traditional Chinese characters, Japanese, and special characters. The number of recognizable characters is over 15,000. In addition to the improvement in document-related text recognition, it also enhances the general text recognition capability.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>78.74</td>
<td>5.26 / 1.12</td>
<td>17.48 / 3.61</td>
<td>10.5</td>
<td>The lightweight recognition model of PP-OCRv4 has high inference efficiency and can be deployed on various hardware devices, including edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Trained Model</a></td>
<td>85.19</td>
<td>8.75 / 2.49</td>
<td>36.93 / 36.93</td>
<td>173</td>
<td>The server-side model of PP-OCRv4 offers high inference accuracy and can be deployed on various types of servers.</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>72.96</td>
<td>3.89 / 1.16</td>
<td>8.72 / 3.56</td>
<td>10.3</td>
<td>PP-OCRv3’s lightweight recognition model is designed for high inference efficiency and can be deployed on a variety of hardware devices, including edge devices.</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
<td>68.81</td>
<td>10.38 / 8.31</td>
<td>66.52 / 30.83</td>
<td>80.5</td>
<td rowspan="1">
SVTRv2 is a server text recognition model developed by the OpenOCR team of Fudan University's Visual and Learning Laboratory (FVL). It won the first prize in the PaddleOCR Algorithm Model Challenge - Task One: OCR End-to-End Recognition Task. The end-to-end recognition accuracy on the A list is 6% higher than that of PP-OCRv4.
</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
<td>65.07</td>
<td>6.29 / 1.57</td>
<td>20.64 / 5.40</td>
<td>48.8</td>
<td rowspan="1">    The RepSVTR text recognition model is a mobile text recognition model based on SVTRv2. It won the first prize in the PaddleOCR Algorithm Model Challenge - Task One: OCR End-to-End Recognition Task. The end-to-end recognition accuracy on the B list is 2.5% higher than that of PP-OCRv4, with the same inference speed.</td>
</tr>
</table>

* <b>English Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td> 70.39</td>
<td>4.81 / 1.23</td>
<td>17.20 / 4.18</td>
<td>7.5</td>
<td>The ultra-lightweight English recognition model trained based on the PP-OCRv4 recognition model supports the recognition of English and numbers.</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>70.69</td>
<td>3.56 / 0.78</td>
<td>8.44 / 5.78</td>
<td>17.3</td>
<td>The ultra-lightweight English recognition model trained based on the PP-OCRv3 recognition model supports the recognition of English and numbers.</td>
</tr>
</table>

* <b>Multilingual Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/korean_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>60.21</td>
<td>3.73 / 0.98</td>
<td>8.76 / 2.91</td>
<td>9.6</td>
<td>The ultra-lightweight Korean recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Korean and numbers. </td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/japan_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>45.69</td>
<td>3.86 / 1.01</td>
<td>8.62 / 2.92</td>
<td>9.8</td>
<td>The ultra-lightweight Japanese recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Japanese and numbers.</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>82.06</td>
<td>3.90 / 1.16</td>
<td>9.24 / 3.18</td>
<td>10.8</td>
<td>The ultra-lightweight Traditional Chinese recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Traditional Chinese and numbers.</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/te_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>95.88</td>
<td>3.59 / 0.81</td>
<td>8.28 / 6.21</td>
<td>8.7</td>
<td>The ultra-lightweight Telugu recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Telugu and numbers.</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ka_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>96.96</td>
<td>3.49 / 0.89</td>
<td>8.63 / 2.77</td>
<td>17.4</td>
<td>The ultra-lightweight Kannada recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Kannada and numbers.</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ta_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>76.83</td>
<td>3.49 / 0.86</td>
<td>8.35 / 3.41</td>
<td>8.7</td>
<td>The ultra-lightweight Tamil recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Tamil and numbers.</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/latin_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>76.93</td>
<td>3.53 / 0.78</td>
<td>8.50 / 6.83</td>
<td>8.7</td>
<td>The ultra-lightweight Latin recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Latin script and numbers.</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/arabic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>73.55</td>
<td>3.60 / 0.83</td>
<td>8.44 / 4.69</td>
<td>17.3</td>
<td>The ultra-lightweight Arabic script recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Arabic script and numbers.</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>94.28</td>
<td>3.56 / 0.79</td>
<td>8.22 / 2.76</td>
<td>8.7</td>
<td>
The ultra-lightweight cyrillic alphabet recognition model trained based on the PP-OCRv3 recognition model supports the recognition of cyrillic letters and numbers.</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>96.44</td>
<td>3.60 / 0.78</td>
<td>6.95 / 2.87</td>
<td>7.9</td>
<td>The ultra-lightweight Devanagari script recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Devanagari script and numbers.</td>
</tr>
</table>

<p><b>Text Line Orientation Classification Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th>
<th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">Training Model</a></td>
<td>98.85</td>
<td>2.16 / 0.41</td>
<td>2.37 / 0.73</td>
<td>0.96</td>
<td>Text line classification model based on PP-LCNet_x0_25, with two classes: 0 degrees and 180 degrees</td>
</tr>
<tr>
<td>PP-LCNet_x1_0_textline_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_textline_ori_pretrained.pdparams">Training Model</a></td>
<td>99.42</td>
<td>- / -</td>
<td>2.98 / 2.98</td>
<td>6.5</td>
<td>Text line classification model based on PP-LCNet_x1_0, with two classes: 0 degrees and 180 degrees</td>
</tr>
</tbody>
</table>

<p><b>Formula Recognition Module (Optional):</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>En-BLEU(%)</th>
<th>Zh-BLEU(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
<tr>
<td>UniMERNet</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UniMERNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">Training Model</a></td>
<td>85.91</td>
<td>43.50</td>
<td>1311.84 / 1311.84</td>
<td>- / 8288.07</td>
<td>1530</td>
<td>UniMERNet is a formula recognition model developed by Shanghai AI Lab. It uses Donut Swin as the encoder and MBartDecoder as the decoder. The model is trained on a dataset of one million samples, including simple formulas, complex formulas, scanned formulas, and handwritten formulas, significantly improving the recognition accuracy of real-world formulas.</td>
</tr>
<tr>
<td>PP-FormulaNet-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">Training Model</a></td>
<td>87.00</td>
<td>45.71</td>
<td>182.25 / 182.25</td>
<td>- / 254.39</td>
<td>224</td>
<td rowspan="2">PP-FormulaNet is an advanced formula recognition model developed by the Baidu PaddlePaddle Vision Team. The PP-FormulaNet-S version uses PP-HGNetV2-B4 as its backbone network. Through parallel masking and model distillation techniques, it significantly improves inference speed while maintaining high recognition accuracy, making it suitable for applications requiring fast inference. The PP-FormulaNet-L version, on the other hand, uses Vary_VIT_B as its backbone network and is trained on a large-scale formula dataset, showing significant improvements in recognizing complex formulas compared to PP-FormulaNet-S.</td>
</tr>
<tr>
<td>PP-FormulaNet-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">Training Model</a></td>
<td>90.36</td>
<td>45.78</td>
<td>1482.03 / 1482.03</td>
<td>- / 3131.54</td>
<td>695</td>
</tr>
<tr>
<td>PP-FormulaNet_plus-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-S_pretrained.pdparams">Training Model</a></td>
<td>88.71</td>
<td>53.32</td>
<td>179.20 / 179.20</td>
<td>- / 260.99</td>
<td>248</td>
<td rowspan="3">PP-FormulaNet_plus is an enhanced version of the formula recognition model developed by the Baidu PaddlePaddle Vision Team, building upon the original PP-FormulaNet. Compared to the original version, PP-FormulaNet_plus utilizes a more diverse formula dataset during training, including sources such as Chinese dissertations, professional books, textbooks, exam papers, and mathematics journals. This expansion significantly improves the model’s recognition capabilities. Among the models, PP-FormulaNet_plus-M and PP-FormulaNet_plus-L have added support for Chinese formulas and increased the maximum number of predicted tokens for formulas from 1,024 to 2,560, greatly enhancing the recognition performance for complex formulas. Meanwhile, the PP-FormulaNet_plus-S model focuses on improving the recognition of English formulas. With these improvements, the PP-FormulaNet_plus series models perform exceptionally well in handling complex and diverse formula recognition tasks. </td>
</tr>
<tr>
<td>PP-FormulaNet_plus-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-M_pretrained.pdparams">Training Model</a></td>
<td>91.45</td>
<td>89.76</td>
<td>1040.27 / 1040.27</td>
<td>- / 1615.80</td>
<td>592</td>
</tr>
<tr>
<td>PP-FormulaNet_plus-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-L_pretrained.pdparams">Training Model</a></td>
<td>92.22</td>
<td>90.64</td>
<td>1476.07 / 1476.07</td>
<td>- / 3125.58</td>
<td>698</td>
</tr>
<tr>
<td>LaTeX_OCR_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Training Model</a></td>
<td>74.55</td>
<td>39.96</td>
<td>1088.89 / 1088.89</td>
<td>- / -</td>
<td>99</td>
<td>LaTeX-OCR is a formula recognition algorithm based on an autoregressive large model. It uses Hybrid ViT as the backbone network and a transformer as the decoder, significantly improving the accuracy of formula recognition.</td>
</tr>
</table>

<p><b>Seal Text Detection Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Training Model</a></td>
<td>98.40</td>
<td>124.64 / 91.57</td>
<td>545.68 / 439.86</td>
<td>109</td>
<td>The PP-OCRv4 server seal text detection model offers higher precision and is suitable for deployment on high-performance servers.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Training Model</a></td>
<td>96.36</td>
<td>9.70 / 3.56</td>
<td>50.38 / 19.64</td>
<td>4.7</td>
<td>The PP-OCRv4 mobile seal text detection model provides higher efficiency and is suitable for deployment on edge devices.</td>
</tr>
</tbody>
</table>

<p><b>Text Image Rectification Module Model:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>CER</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
<td>0.179</td>
<td>19.05 / 19.05</td>
<td>- / 869.82</td>
<td>30.3</td>
<td>High-accuracy text image rectification model</td>
</tr>
</tbody>
</table>
<p><b>The precision metrics of the model are measured from the <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet benchmark</a>.</b></p>
<p><b>Document Image Orientation Classification Module Model:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
<td>99.06</td>
<td>2.62 / 0.59</td>
<td>3.24 / 1.19</td>
<td>7</td>
<td>The document image classification model based on PP-LCNet_x1_0 includes four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees.</td>
</tr>
</tbody>
</table>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
          <li><strong>Test Dataset：</strong>
                        <ul>
                         <li>Document Image Orientation Classification Module: A self-built dataset using PaddleX, covering multiple scenarios such as ID cards and documents, containing 1000 images.</li>
                          <li>Text Image Rectification Model: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a></li>
                          <li>Layout Region Detection Model: A self-built layout detection dataset using PaddleOCR, containing 10,000 images of common document types such as Chinese and English papers, magazines, and research reports.</li>
                          <li>Table Structure Recognition Model: A self-built English table recognition dataset using PaddleX.</li>
                          <li>Text Detection Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 500 images for detection.</li>
                          <li>Chinese Recognition Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 11,000 images for text recognition.</li>
                          <li>ch_SVTRv2_rec: Evaluation set A for "OCR End-to-End Recognition Task" in the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a></li>
                          <li>ch_RepSVTR_rec: Evaluation set B for "OCR End-to-End Recognition Task" in the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a>.</li>
                          <li>English Recognition Model: A self-built English dataset using PaddleX.</li>
                          <li>Multilingual Recognition Model: A self-built multilingual dataset using PaddleX.</li>
                          <li>Text Line Orientation Classification Model: A self-built dataset using PaddleX, covering various scenarios such as ID cards and documents, containing 1000 images.</li>
                          <li>Seal Text Detection Model: A self-built dataset using PaddleX, containing 500 images of circular seal textures.</li>
                        </ul>
                </li>
              <li><strong>Hardware Configuration:</strong>
                  <ul>
                      <li>GPU: NVIDIA Tesla T4</li>
                      <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
                  </ul>
              </li>
              <li><strong>Software Environment:</strong>
                  <ul>
                      <li>Ubuntu 20.04 / CUDA 11.8 / cuDNN 8.9 / TensorRT 8.6.1.6</li>
                      <li>paddlepaddle 3.0.0 / paddlex 3.0.3</li>
                  </ul>
              </li>
              </li>
          </ul>
      </li>
      <li><b>Inference Mode Description</b></li>
  </ul>

<table border="1">
    <thead>
        <tr>
            <th>Mode</th>
            <th>GPU Configuration </th>
            <th>CPU Configuration </th>
            <th>Acceleration Technology Combination</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Normal Mode</td>
            <td>FP32 Precision / No TRT Acceleration</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>High-Performance Mode</td>
            <td>Optimal combination of pre-selected precision types and acceleration strategies</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>Pre-selected optimal backend (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>

</details>

### 1.2 Pipeline benchmark data

<details>
<summary>Click to expand/collapse the table</summary>

<table border="1">
<tr><th>Pipeline configuration</th><th>Hardware</th><th>Avg. inference time (s)</th><th>Peak CPU utilization (%)</th><th>Avg. CPU utilization (%)</th><th>Peak host memory (MB)</th><th>Avg. host memory (MB)</th><th>Peak GPU utilization (%)</th><th>Avg. GPU utilization (%)</th><th>Peak device memory (MB)</th><th>Avg. device memory (MB)</th></tr>
<tr>
<td rowspan="5">PP_StructureV3-default</td>
<td>Intel 8350C + A100</td>
<td>1.38</td>
<td>1384.60</td>
<td>113.26</td>
<td>5781.59</td>
<td>3431.21</td>
<td>100</td>
<td>32.79</td>
<td>37370.00</td>
<td>34165.68</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>2.38</td>
<td>608.70</td>
<td>109.96</td>
<td>6388.91</td>
<td>3737.19</td>
<td>100</td>
<td>39.08</td>
<td>26824.00</td>
<td>24581.61</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>1.36</td>
<td>744.30</td>
<td>112.82</td>
<td>6199.01</td>
<td>3865.78</td>
<td>100</td>
<td>43.81</td>
<td>35132.00</td>
<td>32077.12</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>1.74</td>
<td>418.50</td>
<td>105.96</td>
<td>6138.25</td>
<td>3503.41</td>
<td>100</td>
<td>48.54</td>
<td>18536.00</td>
<td>18353.93</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>3.70</td>
<td>434.40</td>
<td>105.45</td>
<td>6865.87</td>
<td>3595.68</td>
<td>100</td>
<td>71.92</td>
<td>13970.00</td>
<td>12668.58</td>
</tr>
<tr>
<td rowspan="3">PP_StructureV3-pp</td>
<td>Intel 8350C + A100</td>
<td>3.50</td>
<td>679.30</td>
<td>105.96</td>
<td>13850.20</td>
<td>5146.50</td>
<td>100</td>
<td>14.01</td>
<td>37656.00</td>
<td>34716.95</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>5.03</td>
<td>494.20</td>
<td>105.63</td>
<td>13542.94</td>
<td>4833.55</td>
<td>100</td>
<td>20.36</td>
<td>29402.00</td>
<td>26607.92</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>3.17</td>
<td>481.50</td>
<td>105.13</td>
<td>14179.97</td>
<td>5608.80</td>
<td>100</td>
<td>19.35</td>
<td>35454.00</td>
<td>32512.19</td>
</tr>
<tr>
<td rowspan="2">PP_StructureV3-full</td>
<td>Intel 8350C + A100</td>
<td>8.92</td>
<td>697.30</td>
<td>102.88</td>
<td>13777.07</td>
<td>4573.65</td>
<td>100</td>
<td>18.39</td>
<td>38776.00</td>
<td>37554.09</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>13.12</td>
<td>437.40</td>
<td>102.36</td>
<td>13974.00</td>
<td>4484.00</td>
<td>100</td>
<td>17.50</td>
<td>29878.00</td>
<td>28733.59</td>
</tr>
<tr>
<td rowspan="5">PP_StructureV3-seal</td>
<td>Intel 8350C + A100</td>
<td>1.39</td>
<td>747.50</td>
<td>112.55</td>
<td>5788.79</td>
<td>3742.03</td>
<td>100</td>
<td>33.81</td>
<td>38966.00</td>
<td>35832.44</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>2.44</td>
<td>630.10</td>
<td>110.18</td>
<td>6343.39</td>
<td>3725.98</td>
<td>100</td>
<td>42.23</td>
<td>28078.00</td>
<td>25834.70</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>1.40</td>
<td>792.20</td>
<td>113.63</td>
<td>6673.60</td>
<td>4417.34</td>
<td>100</td>
<td>46.33</td>
<td>35530.00</td>
<td>32516.87</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>1.75</td>
<td>422.40</td>
<td>106.08</td>
<td>6068.87</td>
<td>3973.49</td>
<td>100</td>
<td>50.12</td>
<td>19630.00</td>
<td>18374.37</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>3.76</td>
<td>400.30</td>
<td>105.10</td>
<td>6296.28</td>
<td>3651.42</td>
<td>100</td>
<td>72.57</td>
<td>14304.00</td>
<td>13268.36</td>
</tr>
<tr>
<td rowspan="4">PP_StructureV3-chart</td>
<td>Intel 8350C + A100</td>
<td>7.70</td>
<td>746.80</td>
<td>102.69</td>
<td>6355.58</td>
<td>4006.48</td>
<td>100</td>
<td>22.38</td>
<td>37380.00</td>
<td>36730.73</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>10.58</td>
<td>599.20</td>
<td>102.51</td>
<td>5754.14</td>
<td>3333.78</td>
<td>100</td>
<td>21.99</td>
<td>26820.00</td>
<td>26253.70</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>8.03</td>
<td>413.30</td>
<td>101.31</td>
<td>6473.29</td>
<td>3689.84</td>
<td>100</td>
<td>26.19</td>
<td>18540.00</td>
<td>18494.69</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>11.69</td>
<td>460.90</td>
<td>101.85</td>
<td>6503.12</td>
<td>3524.06</td>
<td>100</td>
<td>46.81</td>
<td>13966.00</td>
<td>12481.94</td>
</tr>
<tr>
<td rowspan="5">PP_StructureV3-notable</td>
<td>Intel 8350C + A100</td>
<td>1.24</td>
<td>738.30</td>
<td>110.45</td>
<td>5638.16</td>
<td>3278.30</td>
<td>100</td>
<td>35.32</td>
<td>30320.00</td>
<td>27026.17</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>2.24</td>
<td>452.40</td>
<td>107.79</td>
<td>5579.15</td>
<td>3635.95</td>
<td>100</td>
<td>43.00</td>
<td>23098.00</td>
<td>20684.43</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>1.18</td>
<td>989.00</td>
<td>107.71</td>
<td>6041.76</td>
<td>4024.76</td>
<td>100</td>
<td>50.67</td>
<td>33780.00</td>
<td>29733.15</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>1.58</td>
<td>225.00</td>
<td>102.56</td>
<td>5518.10</td>
<td>3333.08</td>
<td>100</td>
<td>49.90</td>
<td>21532.00</td>
<td>18567.99</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>3.40</td>
<td>413.30</td>
<td>103.58</td>
<td>5874.88</td>
<td>3662.49</td>
<td>100</td>
<td>76.82</td>
<td>13764.00</td>
<td>11890.62</td>
</tr>
<tr>
<td rowspan="7">PP_StructureV3-noformula</td>
<td>Intel 6271C</td>
<td>7.85</td>
<td>1172.50</td>
<td>964.70</td>
<td>17739.00</td>
<td>11101.02</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>8.83</td>
<td>1053.50</td>
<td>970.64</td>
<td>15463.48</td>
<td>9408.19</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.84</td>
<td>788.60</td>
<td>124.25</td>
<td>6246.39</td>
<td>3674.32</td>
<td>100</td>
<td>30.57</td>
<td>40084.00</td>
<td>37358.45</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>1.42</td>
<td>606.20</td>
<td>115.53</td>
<td>7015.57</td>
<td>3707.03</td>
<td>100</td>
<td>35.63</td>
<td>29540.00</td>
<td>27620.28</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.87</td>
<td>644.10</td>
<td>119.23</td>
<td>6895.76</td>
<td>4222.85</td>
<td>100</td>
<td>50.00</td>
<td>36878.00</td>
<td>34104.59</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>1.03</td>
<td>377.50</td>
<td>106.87</td>
<td>5819.88</td>
<td>3830.19</td>
<td>100</td>
<td>42.87</td>
<td>19340.00</td>
<td>17550.94</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>2.02</td>
<td>430.20</td>
<td>109.21</td>
<td>6600.62</td>
<td>3824.18</td>
<td>100</td>
<td>65.75</td>
<td>14332.00</td>
<td>12712.18</td>
</tr>
<tr>
<td rowspan="9">PP_StructureV3-lightweight</td>
<td>Intel 6271C</td>
<td>4.36</td>
<td>1189.70</td>
<td>995.78</td>
<td>14000.50</td>
<td>9374.97</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>3.74</td>
<td>1049.60</td>
<td>967.77</td>
<td>12960.96</td>
<td>7644.25</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.86</td>
<td>572.20</td>
<td>120.84</td>
<td>8290.49</td>
<td>3569.44</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.61</td>
<td>823.40</td>
<td>126.25</td>
<td>9258.22</td>
<td>3776.63</td>
<td>52</td>
<td>18.95</td>
<td>7456.00</td>
<td>7131.95</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>1.07</td>
<td>686.80</td>
<td>116.70</td>
<td>9381.75</td>
<td>4126.28</td>
<td>58</td>
<td>22.92</td>
<td>8450.00</td>
<td>8083.30</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.46</td>
<td>999.00</td>
<td>122.21</td>
<td>9734.78</td>
<td>4516.40</td>
<td>61</td>
<td>24.41</td>
<td>7524.00</td>
<td>7167.52</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.70</td>
<td>355.40</td>
<td>111.51</td>
<td>9415.45</td>
<td>4094.06</td>
<td>89</td>
<td>30.85</td>
<td>7248.00</td>
<td>6927.58</td>
</tr>
<tr>
<td>M4</td>
<td>12.22</td>
<td>223.60</td>
<td>107.35</td>
<td>9531.22</td>
<td>7884.61</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>1.13</td>
<td>461.40</td>
<td>112.16</td>
<td>7923.09</td>
<td>3837.31</td>
<td>85</td>
<td>41.67</td>
<td>8218.00</td>
<td>7902.04</td>
</tr>
</table>


<table border="1">
<tr><th>Pipeline configuration</th><th>description</th></tr>
<tr>
<td>PP_StructureV3-default</td>
<td>Default configuration</td>
</tr>
<tr>
<td>PP_StructureV3-pp</td>
<td>Based on the default configuration, document image preprocessing is enabled</td>
</tr>
<tr>
<td>PP_StructureV3-full</td>
<td>Based on the default configuration, document image preprocessing and chart parsing are enabled</td>
</tr>
<tr>
<td>PP_StructureV3-seal</td>
<td>Based on the default configuration, seal text recognition is enabled</td>
</tr>
<tr>
<td>PP_StructureV3-chart</td>
<td>Based on the default configuration, chart parsing is enabled</td>
</tr>
<tr>
<td>PP_StructureV3-notable</td>
<td>Based on the default configuration, table recognition is disabled</td>
</tr>
<tr>
<td>PP_StructureV3-noformula</td>
<td>Based on the default configuration, formula recognition is disabled</td>
</tr>
<tr>
<td>PP_StructureV3-lightweight</td>
<td>Based on the default configuration, all task models are replaced with lightweight versions</td>
</tr>
</table>
</details>


* Test environment:
    * PaddlePaddle 3.1.0、CUDA 11.8、cuDNN 8.9
    * PaddleX @ develop (f1eb28e23cfa54ce3e9234d2e61fcb87c93cf407)
    * Docker image: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda11.8-cudnn8.9
* Test data:
    * Test data containing 280 images including tables, seals, formulas, and charts.
* Test strategy:
    * Warm up with 20 samples, then repeat the full dataset once for performance testing.
* Note:
    * Since we did not collect device memory data for NPU and XPU, the corresponding entries in the table are marked as N/A.

## 2. Quick Start
All the model pipelines provided by PaddleX can be quickly experienced. You can use the command line or Python on your local machine to experience the effect of the PP-StructureV3 pipeline.

Before using the PP-StructureV3 pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md). If you wish to selectively install dependencies, please refer to the relevant instructions in the installation guide. The dependency group corresponding to this pipeline is `ocr`.

> When performing GPU inference, the default configuration may use more than 16 GB of VRAM. Please ensure that your GPU has sufficient memory. To reduce VRAM usage, you can modify the configuration file as described below to disable unnecessary features.

### 2.1 Experiencing via Command Line

You can quickly experience the PP-StructureV3 pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png) and replace `--input` with the local path to perform prediction.

```
paddlex --pipeline PP-StructureV3 \
        --input pp_structure_v3_demo.png \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --use_e2e_wireless_table_rec_model True \
        --save_path ./output \
        --device gpu:0
```

<b>Note: </b>The official models would be download from HuggingFace by first. PaddleX also support to specify the preferred source by setting the environment variable `PADDLE_PDX_MODEL_SOURCE`. The supported values are `huggingface`, `aistudio`, `bos`, and `modelscope`. For example, to prioritize using `bos`, set: `PADDLE_PDX_MODEL_SOURCE="bos"`.

The parameter description can be found in [2.2.2 Python Script Integration](#222-python-script-integration). Supports specifying multiple devices simultaneously for parallel inference. For details, please refer to the documentation on pipeline parallel inference.

After running, the result will be printed to the terminal, as follows:

<details><summary>👉Click to Expand</summary>
<pre><code>

{'res': {'input_path': 'pp_structure_v3_demo.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_seal_recognition': False, 'use_table_recognition': True, 'use_formula_recognition': True, 'use_chart_recognition': False, 'use_region_detection': True}, 'parsing_res_list': [{'block_label': 'doc_title', 'block_content': '助力双方交往搭建友谊桥梁', 'block_bbox': [133, 36, 1379, 123], 'block_id': 0, 'block_order': 1}, {'block_label': 'text', 'block_content': '本报记者沈小晓任彦黄培昭', 'block_bbox': [584, 159, 927, 179], 'block_id': 1, 'block_order': 2}, {'block_label': 'image', 'block_content': '', 'block_bbox': [774, 201, 1502, 685], 'block_id': 2, 'block_order': None}, {'block_label': 'figure_title', 'block_content': '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。中国驻厄立特里亚大使馆供图', 'block_bbox': [808, 704, 1484, 747], 'block_id': 3, 'block_order': None}, {'block_label': 'text', 'block_content': '身着中国传统民族服装的厄立特里亚青年依次登台表演中国民族舞、现代舞、扇子舞等，曼妙的舞姿赢得现场观众阵阵掌声。这是日前危立特里亚高等教育与研究院孔子学院(以下简称“厄特孔院")举办“喜迎新年"中国歌舞比赛的场景。\n', 'block_bbox': [9, 201, 358, 338], 'block_id': 4, 'block_order': 3}, {'block_label': 'text', 'block_content': '中国和厄立特里亚传统友谊深厚。近年来，在高质量共建“一带一路”框架下，中厄两国人文交流不断深化，互利合作的民意基础日益深厚。\n', 'block_bbox': [9, 345, 358, 435], 'block_id': 5, 'block_order': 4}, {'block_label': 'paragraph_title', 'block_content': '“学好中文，我们的未来不是梦”\n', 'block_bbox': [28, 456, 339, 514], 'block_id': 6, 'block_order': 5}, {'block_label': 'text', 'block_content': '“鲜花曾告诉我你怎样走过，大地知道你心中的每一个角落……"厄立特里亚阿斯马拉大学综合楼二层，一阵优美的歌声在走廊里回响。循着熟悉的旋律轻轻推开一间教室的门，学生们正跟着老师学唱中文歌曲《同一首歌》。', 'block_bbox': [9, 536, 358, 651], 'block_id': 7, 'block_order': 6}, {'block_label': 'text', 'block_content': '这是厄特孔院阿斯马拉大学教学点的一节中文歌曲课。为了让学生们更好地理解歌词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐字翻译和解释歌词。随着伴奏声响起，学生们边唱边随着节拍摇动身体，现场气氛热烈。', 'block_bbox': [9, 658, 359, 770], 'block_id': 8, 'block_order': 7}, {'block_label': 'text', 'block_content': '“这是中文歌曲初级班，共有32人。学生大部分来自首都阿斯马拉的中小学，年龄最小的仅有6岁。”尤斯拉告诉记者。', 'block_bbox': [10, 776, 359, 842], 'block_id': 9, 'block_order': 8}, {'block_label': 'text', 'block_content': '尤斯拉今年23岁，是厄立特里亚一所公立学校的艺术老师。她12岁开始在厄特孔院学习中文，在2017年第十届“汉语桥"世界中学生中文比赛中获得厄立特里亚赛区第一名，并和同伴代表厄立特里亚前往中国参加决赛，获得团体优胜奖。2022年起，尤斯拉开始在厄特孔院兼职教授中文歌曲，每周末两个课时。“中国文化博大精深，我希望我的学生们能够通过中文歌曲更好地理解中国文化。”她说。', 'block_bbox': [9, 848, 358, 1057], 'block_id': 10, 'block_order': 9}, {'block_label': 'text', 'block_content': '“姐姐，你想去中国吗?”“非常想！我想去看故宫、爬长城。”尤斯拉的学生中有一对能歌善舞的姐妹，姐姐露娅今年15岁，妹妹莉娅14岁，两人都已在厄特孔院学习多年，中文说得格外流利。\n', 'block_bbox': [9, 1064, 358, 1177], 'block_id': 11, 'block_order': 10}, {'block_label': 'text', 'block_content': '露娅对记者说：“这些年来，怀着对中文和中国文化的热爱，我们姐妹俩始终相互鼓励，一起学习。我们的中文一天比一天好，还学会了中文歌和中国舞。我们一定要到中国去。学好中文，我们的未来不是梦！”', 'block_bbox': [8, 1184, 358, 1297], 'block_id': 12, 'block_order': 11}, {'block_label': 'text', 'block_content': '据厄特孔院中方院长黄鸣飞介绍，这所孔院成立于2013年3月，由贵州财经大学和', 'block_bbox': [10, 1304, 358, 1346], 'block_id': 13, 'block_order': 12}, {'block_label': 'text', 'block_content': '厄立特里亚高等教育与研究院合作建立，开设了中国语言课程和中国文化课程，注册学生2万余人次。10余年来，厄特孔院已成为当地民众了解中国的一扇窗口。', 'block_bbox': [388, 200, 740, 290], 'block_id': 14, 'block_order': 13}, {'block_label': 'text', 'block_content': '黄鸣飞表示，随着来学习中文的人日益增多，阿斯马拉大学教学点已难以满足教学需要。2024年4月，由中企蜀道集团所属四川路桥承建的孔院教学楼项目在阿斯马拉开工建设，预计今年上半年竣工，建成后将为危特孔院提供全新的办学场地。\n', 'block_bbox': [389, 297, 740, 435], 'block_id': 15, 'block_order': 14}, {'block_label': 'paragraph_title', 'block_content': '“在中国学习的经历让我看到更广阔的世界”', 'block_bbox': [409, 456, 718, 515], 'block_id': 16, 'block_order': 15}, {'block_label': 'text', 'block_content': '多年来，厄立特里亚广大赴华留学生和培训人员积极投身国家建设，成为助力该国发展的人才和厄中友好的见证者和推动者。', 'block_bbox': [389, 537, 740, 603], 'block_id': 17, 'block_order': 16}, {'block_label': 'text', 'block_content': '在厄立特里亚全国妇女联盟工作的约翰娜·特韦尔德·凯莱塔就是其中一位。她曾在中华女子学院攻读硕士学位，研究方向是女性领导力与社会发展。其间，她实地走访中国多个地区，获得了观察中国社会发展的第一手资料。\n', 'block_bbox': [389, 609, 740, 745], 'block_id': 18, 'block_order': 17}, {'block_label': 'text', 'block_content': '谈起在中国求学的经历，约翰娜记忆犹新：“中国的发展在当今世界是独一无二的。沿着中国特色社会主义道路坚定前行，中国创造了发展奇迹，这一切都离不开中国共产党的领导。中国的发展经验值得许多国家学习借鉴。”\n', 'block_bbox': [389, 752, 740, 889], 'block_id': 19, 'block_order': 18}, {'block_label': 'text', 'block_content': '正在西南大学学习的厄立特里亚博士生穆卢盖塔·泽穆伊对中国怀有深厚感情。8年前，在北京师范大学获得硕士学位后，穆卢盖塔在社交媒体上写下这样一段话：“这是我人生的重要一步，自此我拥有了一双坚固的鞋子，赋予我穿越荆棘的力量。”', 'block_bbox': [389, 896, 740, 1033], 'block_id': 20, 'block_order': 19}, {'block_label': 'text', 'block_content': '穆卢盖塔密切关注中国在经济、科技、教育等领域的发展，“中国在科研等方面的实力与日俱增。在中国学习的经历让我看到更广阔的世界，从中受益匪浅。”\n', 'block_bbox': [389, 1040, 740, 1129], 'block_id': 21, 'block_order': 20}, {'block_label': 'text', 'block_content': '23岁的莉迪亚·埃斯蒂法诺斯已在厄特孔院学习3年，在中国书法、中国画等方面表现干分优秀，在2024年厄立特里亚赛区的“汉语桥”比赛中获得一等奖。莉迪亚说：“学习中国书法让我的内心变得安宁和纯粹。我也喜欢中国的服饰，希望未来能去中国学习，把中国不同民族元素融入服装设计中，创作出更多精美作品，也把厄特文化分享给更多的中国朋友。”\n', 'block_bbox': [389, 1136, 740, 1345], 'block_id': 22, 'block_order': 21}, {'block_label': 'text', 'block_content': '“不管远近都是客人，请不用客气；相约好了在一起，我们欢迎你……”在一场中厄青年联谊活动上，四川路桥中方员工同当地大学生合唱《北京欢迎你》。厄立特里亚技术学院计算机科学与工程专业学生鲁夫塔·谢拉是其中一名演唱者，她很早便在孔院学习中文，一直在为去中国留学作准备。“这句歌词是我们两国人民友谊的生动写照。无论是投身于厄立特里亚基础设施建设的中企员工，还是在中国留学的厄立特里亚学子，两国人民携手努力，必将推动两国关系不断向前发展。”鲁夫塔说。\n', 'block_bbox': [769, 776, 1121, 1058], 'block_id': 23, 'block_order': 22}, {'block_label': 'text', 'block_content': '厄立特里亚高等教育委员会主任助理萨马瑞表示：“每年我们都会组织学生到中国访问学习，自前有超过5000名厄立特里亚学生在中国留学。学习中国的教育经验，有助于提升厄立特里亚的教育水平。”', 'block_bbox': [770, 1064, 1121, 1177], 'block_id': 24, 'block_order': 23}, {'block_label': 'paragraph_title', 'block_content': '“共同向世界展示非洲和亚洲的灿烂文明”', 'block_bbox': [790, 1200, 1102, 1259], 'block_id': 25, 'block_order': 24}, {'block_label': 'text', 'block_content': '从阿斯马拉出发，沿着蜿蜒曲折的盘山公路一路向东寻找丝路印迹。驱车两个小时，记者来到位于厄立特里亚港口城市马萨', 'block_bbox': [770, 1280, 1122, 1346], 'block_id': 26, 'block_order': 25}, {'block_label': 'text', 'block_content': '瓦的北红海省博物馆。', 'block_bbox': [1154, 776, 1331, 794], 'block_id': 27, 'block_order': 26}, {'block_label': 'text', 'block_content': '博物馆二层陈列着一个发掘自阿杜利斯古城的中国古代陶制酒器，罐身上写着“万”“和”“禅”“山”等汉字。“这件文物证明，很早以前我们就通过海上丝绸之路进行贸易往来与文化交流。这也是厄立特里亚与中国友好交往历史的有力证明。”北红海省博物馆研究与文献部负责人伊萨亚斯·特斯法兹吉说。\n', 'block_bbox': [1152, 800, 1502, 986], 'block_id': 28, 'block_order': 27}, {'block_label': 'text', 'block_content': '厄立特里亚国家博物馆考古学和人类学研究员菲尔蒙·特韦尔德十分喜爱中国文化。他表示：“学习彼此的语言和文化，将帮助厄中两国人民更好地理解彼此，助力双方交往，搭建友谊桥梁。”\n', 'block_bbox': [1152, 992, 1502, 1106], 'block_id': 29, 'block_order': 28}, {'block_label': 'text', 'block_content': '厄立特里亚国家博物馆馆长塔吉丁·努重达姆·优素福曾多次访问中国，对中华文明的传承与创新、现代化博物馆的建设与发展印象深刻。“中国博物馆不仅有许多保存完好的文物，还充分运用先进科技手段进行展示，帮助人们更好理解中华文明。”塔吉丁说，“危立特里亚与中国都拥有悠久的文明，始终相互理解、相互尊重。我希望未来与中国同行加强合作，共同向世界展示非洲和亚洲的灿烂文明。”\n', 'block_bbox': [1151, 1112, 1502, 1346], 'block_id': 30, 'block_order': 29}], 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 1, 'label': 'image', 'score': 0.9864752888679504, 'coordinate': [774.821, 201.05176, 1502.1008, 685.7733]}, {'cls_id': 2, 'label': 'text', 'score': 0.9859225749969482, 'coordinate': [769.8655, 776.2444, 1121.5986, 1058.4167]}, {'cls_id': 2, 'label': 'text', 'score': 0.9857110381126404, 'coordinate': [1151.98, 1112.5356, 1502.7852, 1346.3569]}, {'cls_id': 2, 'label': 'text', 'score': 0.9847239255905151, 'coordinate': [389.0322, 1136.3547, 740.2322, 1345.928]}, {'cls_id': 2, 'label': 'text', 'score': 0.9842492938041687, 'coordinate': [1152.1504, 800.1625, 1502.1265, 986.1522]}, {'cls_id': 2, 'label': 'text', 'score': 0.9840831160545349, 'coordinate': [9.158066, 848.8696, 358.5725, 1057.832]}, {'cls_id': 2, 'label': 'text', 'score': 0.9802583456039429, 'coordinate': [9.335953, 201.10046, 358.31543, 338.78876]}, {'cls_id': 2, 'label': 'text', 'score': 0.9801402688026428, 'coordinate': [389.1556, 297.4113, 740.07556, 435.41647]}, {'cls_id': 2, 'label': 'text', 'score': 0.9793564081192017, 'coordinate': [389.18976, 752.0959, 740.0832, 889.88043]}, {'cls_id': 2, 'label': 'text', 'score': 0.9793409109115601, 'coordinate': [389.02496, 896.34143, 740.7431, 1033.9465]}, {'cls_id': 2, 'label': 'text', 'score': 0.9776486754417419, 'coordinate': [8.950775, 1184.7842, 358.75067, 1297.8755]}, {'cls_id': 2, 'label': 'text', 'score': 0.9773538708686829, 'coordinate': [770.7178, 1064.5714, 1121.2249, 1177.9928]}, {'cls_id': 2, 'label': 'text', 'score': 0.9773064255714417, 'coordinate': [389.38086, 609.7071, 740.0553, 745.3206]}, {'cls_id': 2, 'label': 'text', 'score': 0.9765821099281311, 'coordinate': [1152.0115, 992.296, 1502.4929, 1106.1166]}, {'cls_id': 2, 'label': 'text', 'score': 0.9761461019515991, 'coordinate': [9.46727, 536.993, 358.2047, 651.32025]}, {'cls_id': 2, 'label': 'text', 'score': 0.975399911403656, 'coordinate': [9.353531, 1064.3059, 358.45312, 1177.8347]}, {'cls_id': 2, 'label': 'text', 'score': 0.9730532169342041, 'coordinate': [9.932312, 345.36237, 358.03476, 435.1646]}, {'cls_id': 2, 'label': 'text', 'score': 0.9722575545310974, 'coordinate': [388.91736, 200.93637, 740.00793, 290.80692]}, {'cls_id': 2, 'label': 'text', 'score': 0.9710634350776672, 'coordinate': [389.39496, 1040.3186, 740.0091, 1129.7168]}, {'cls_id': 2, 'label': 'text', 'score': 0.9696939587593079, 'coordinate': [9.6145935, 658.1123, 359.06088, 770.0288]}, {'cls_id': 2, 'label': 'text', 'score': 0.9664148092269897, 'coordinate': [770.235, 1280.4562, 1122.0927, 1346.4742]}, {'cls_id': 2, 'label': 'text', 'score': 0.9597565531730652, 'coordinate': [389.66678, 537.5609, 740.06274, 603.17725]}, {'cls_id': 2, 'label': 'text', 'score': 0.9594324827194214, 'coordinate': [10.162949, 776.86414, 359.08307, 842.1771]}, {'cls_id': 2, 'label': 'text', 'score': 0.9484634399414062, 'coordinate': [10.402863, 1304.7743, 358.9441, 1346.3749]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9476125240325928, 'coordinate': [28.159409, 456.7627, 339.5631, 514.9665]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9427680969238281, 'coordinate': [790.6992, 1200.3663, 1102.3799, 1259.1647]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.9424256682395935, 'coordinate': [409.02832, 456.6831, 718.8154, 515.5757]}, {'cls_id': 10, 'label': 'doc_title', 'score': 0.9376171827316284, 'coordinate': [133.77905, 36.884415, 1379.6667, 123.46867]}, {'cls_id': 2, 'label': 'text', 'score': 0.9020254015922546, 'coordinate': [584.9165, 159.1416, 927.22876, 179.01605]}, {'cls_id': 2, 'label': 'text', 'score': 0.895164430141449, 'coordinate': [1154.3364, 776.74646, 1331.8564, 794.2301]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.7892374992370605, 'coordinate': [808.9641, 704.2555, 1484.0623, 747.2296]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[ 129,   42],
        ...,
        [ 129,  140]],

       ...,

       [[1156, 1330],
        ...,
        [1156, 1351]]], dtype=int16), 'text_det_params': {'limit_side_len': 736, 'limit_type': 'min', 'thresh': 0.3, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 1.5}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0.0, 'return_word_box': False, 'rec_texts': ['助力双方交往', '搭建友谊桥梁', '本报记者沈小晓', '任', '彦', '黄培昭', '身着中国传统民族服装的厄立特里亚青', '厄立特里亚高等教育与研究院合作建立，开', '年依次登台表演中国民族舞、现代舞、扇子舞', '设了中国语言课程和中国文化课程，注册学', '等，曼妙的舞姿赢得现场观众阵阵掌声。这', '生2万余人次。10余年来，厄特孔院已成为', '是日前危立特里亚高等教育与研究院孔子学', '当地民众了解中国的一扇窗口。', '院(以下简称“厄特孔院")举办“喜迎新年"中国', '黄鸣飞表示，随着来学习中文的人日益', '歌舞比赛的场景。', '增多，阿斯马拉大学教学点已难以满足教学', '中国和厄立特里亚传统友谊深厚。近年', '需要。2024年4月，由中企蜀道集团所属四', '来，在高质量共建“一带一路”框架下，中厄两', '川路桥承建的孔院教学楼项目在阿斯马拉开', '国人文交流不断深化，互利合作的民意基础', '工建设，预计今年上半年竣工，建成后将为危', '日益深厚。', '特孔院提供全新的办学场地。', '“学好中文，我们的', '“在中国学习的经历', '未来不是梦”', '让我看到更广阔的世界”', '“鲜花曾告诉我你怎样走过，大地知道你', '多年来，厄立特里亚广大赴华留学生和', '心中的每一个角落……"厄立特里亚阿斯马拉', '培训人员积极投身国家建设，成为助力该国', '大学综合楼二层，一阵优美的歌声在走廊里回', '发展的人才和厄中友好的见证者和推动者。', '响。循着熟悉的旋律轻轻推开一间教室的门，', '在厄立特里亚全国妇女联盟工作的约翰', '学生们正跟着老师学唱中文歌曲《同一首歌》。', '娜·特韦尔德·凯莱塔就是其中一位。她曾在', '这是厄特孔院阿斯马拉大学教学点的一', '中华女子学院攻读硕士学位，研究方向是女', '节中文歌曲课。为了让学生们更好地理解歌', '性领导力与社会发展。其间，她实地走访中国', '词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐', '多个地区，获得了观察中国社会发展的第一', '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。', '字翻译和解释歌词。随着伴奏声响起，学生们', '手资料。', '中国驻厄立特里亚大使馆供图', '边唱边随着节拍摇动身体，现场气氛热烈。', '谈起在中国求学的经历，约翰娜记忆犹', '“这是中文歌曲初级班，共有32人。学', '新：“中国的发展在当今世界是独一无二的。', '“不管远近都是客人，请不用客气；相约', '瓦的北红海省博物馆。', '生大部分来自首都阿斯马拉的中小学，年龄', '沿着中国特色社会主义道路坚定前行，中国', '好了在一起，我们欢迎你……”在一场中厄青', '博物馆二层陈列着一个发掘自阿杜利', '最小的仅有6岁。”尤斯拉告诉记者。', '创造了发展奇迹，这一切都离不开中国共产党', '年联谊活动上，四川路桥中方员工同当地大', '斯古城的中国古代陶制酒器，罐身上写着', '尤斯拉今年23岁，是厄立特里亚一所公立', '的领导。中国的发展经验值得许多国家学习', '学生合唱《北京欢迎你》。厄立特里亚技术学', '“万”“和”“禅”“山”等汉字。“这件文物证', '学校的艺术老师。她12岁开始在厄特孔院学', '借鉴。”', '院计算机科学与工程专业学生鲁夫塔·谢拉', '明，很早以前我们就通过海上丝绸之路进行', '习中文，在2017年第十届“汉语桥"世界中学生', '正在西南大学学习的厄立特里亚博士生', '是其中一名演唱者，她很早便在孔院学习中', '贸易往来与文化交流。这也是厄立特里亚', '中文比赛中获得厄立特里亚赛区第一名，并和', '穆卢盖塔·泽穆伊对中国怀有深厚感情。8', '文，一直在为去中国留学作准备。“这句歌词', '与中国友好交往历史的有力证明。”北红海', '同伴代表厄立特里亚前往中国参加决赛，获得', '年前，在北京师范大学获得硕士学位后，穆卢', '是我们两国人民友谊的生动写照。无论是投', '省博物馆研究与文献部负责人伊萨亚斯·特', '团体优胜奖。2022年起，尤斯拉开始在厄特孔', '盖塔在社交媒体上写下这样一段话：“这是我', '身于厄立特里亚基础设施建设的中企员工，', '斯法兹吉说。', '院兼职教授中文歌曲，每周末两个课时。“中国', '人生的重要一步，自此我拥有了一双坚固的', '还是在中国留学的厄立特里亚学子，两国人', '厄立特里亚国家博物馆考古学和人类学', '文化博大精深，我希望我的学生们能够通过中', '鞋子，赋予我穿越荆棘的力量。”', '民携手努力，必将推动两国关系不断向前发', '研究员菲尔蒙·特韦尔德十分喜爱中国文', '文歌曲更好地理解中国文化。”她说。', '穆卢盖塔密切关注中国在经济、科技、教', '展。”鲁夫塔说。', '化。他表示：“学习彼此的语言和文化，将帮', '“姐姐，你想去中国吗?”“非常想！我想', '育等领域的发展，“中国在科研等方面的实力', '厄立特里亚高等教育委员会主任助理萨', '助厄中两国人民更好地理解彼此，助力双方', '去看故宫、爬长城。”尤斯拉的学生中有一对', '与日俱增。在中国学习的经历让我看到更广', '马瑞表示：“每年我们都会组织学生到中国访', '交往，搭建友谊桥梁。”', '能歌善舞的姐妹，姐姐露娅今年15岁，妹妹', '阔的世界，从中受益匪浅。”', '问学习，自前有超过5000名厄立特里亚学生', '厄立特里亚国家博物馆馆长塔吉丁·努', '莉娅14岁，两人都已在厄特孔院学习多年，', '23岁的莉迪亚·埃斯蒂法诺斯已在厄特', '在中国留学。学习中国的教育经验，有助于', '重达姆·优素福曾多次访问中国，对中华文明', '中文说得格外流利。', '孔院学习3年，在中国书法、中国画等方面表', '提升厄立特里亚的教育水平。”', '的传承与创新、现代化博物馆的建设与发展', '露娅对记者说：“这些年来，怀着对中文', '现干分优秀，在2024年厄立特里亚赛区的', '印象深刻。“中国博物馆不仅有许多保存完好', '“共同向世界展示非', '和中国文化的热爱，我们姐妹俩始终相互鼓', '“汉语桥”比赛中获得一等奖。莉迪亚说：“学', '的文物，还充分运用先进科技手段进行展示，', '励，一起学习。我们的中文一天比一天好，还', '习中国书法让我的内心变得安宁和纯粹。我', '洲和亚洲的灿烂文明”', '帮助人们更好理解中华文明。”塔吉丁说，“危', '学会了中文歌和中国舞。我们一定要到中国', '也喜欢中国的服饰，希望未来能去中国学习，', '立特里亚与中国都拥有悠久的文明，始终相', '去。学好中文，我们的未来不是梦！”', '把中国不同民族元素融入服装设计中，创作', '从阿斯马拉出发，沿着蜿蜒曲折的盘山', '互理解、相互尊重。我希望未来与中国同行', '据厄特孔院中方院长黄鸣飞介绍，这所', '出更多精美作品，也把厄特文化分享给更多', '公路一路向东寻找丝路印迹。驱车两个小', '加强合作，共同向世界展示非洲和亚洲的灿', '孔院成立于2013年3月，由贵州财经大学和', '的中国朋友。”', '时，记者来到位于厄立特里亚港口城市马萨', '烂文明。”'], 'rec_scores': array([0.99113536, ..., 0.95110023]), 'rec_polys': array([[[ 129,   42],
        ...,
        [ 129,  140]],

       ...,

       [[1156, 1330],
        ...,
        [1156, 1351]]], dtype=int16), 'rec_boxes': array([[ 129, ...,  140],
       ...,
       [1156, ..., 1351]], dtype=int16)}}}

</code></pre></details>

The result parameter description can be found in the result interpretation in [2.2.2 Python Script Integration](#222-python-script-integration).

<b>Note:</b> Since the default model of the pipeline is relatively large, the inference speed may be slow. You can refer to the model list in Section 1 and replace it with a model that has faster inference speed.

### 2.2 Python Script Integration
Just a few lines of code can complete the quick inference of the pipeline. Taking the PP-StructureV3 pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-StructureV3")

output = pipeline.predict(
    input="./pp_structure_v3_demo.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print() ## Print the structured prediction output
    res.save_to_json(save_path="output") ## Save the structured JSON result of the current image
    res.save_to_markdown(save_path="output") ## Save the result of the current image in Markdown format
    res.save_to_word(save_path="output") ## Save the result of the current image in Word format
```
If it is a PDF file, each page of the PDF will be processed separately, and each page will have its own corresponding Markdown file. If you want to convert the entire PDF file into a Markdown file, it is recommended to run it in the following way:

```python
from pathlib import Path
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-StructureV3")

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
    md_info = res.markdown
    markdown_list.append(md_info)
    markdown_images.append(md_info.get("markdown_images", {}))

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

**Note:**

- The default text recognition model used by PP-StructureV3 is the **Chinese-English recognition model**. Limited recognition capability for purely English text. For fully English scenarios, you can modify the `model_name` under the `TextRecognition` configuration item in the [PP-StructureV3 configuration file](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-rc/paddlex/configs/pipelines/PP-StructureV3.yaml) to `en_PP-OCRv4_mobile_rec` or other English recognition models for better recognition results. For other language scenarios, you can also refer to the model list mentioned earlier and choose the corresponding language recognition model for replacement.

- In the example code, the parameters `use_doc_orientation_classify`, `use_doc_unwarping`, and `use_textline_orientation` are all set to False by default. These parameters respectively control the document orientation classification, document unwarping, and text line orientation classification functions. If you need to use these features, you can manually set them to True.

- PP-StructureV3 provides flexible parameter configuration, allowing you to adjust parameters for layout detection, text detection, text recognition, etc., based on the characteristics of the document for better results. For more detailed configurations, please refer to the [PP-StructureV3 configuration file](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-rc/paddlex/configs/pipelines/PP-StructureV3.yaml).

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` instance to create a pipeline object. The specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is a pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>The path to the pipeline configuration file.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The inference device for the pipeline. It supports specifying the specific GPU card number, such as "gpu:0", other hardware card numbers, such as "npu:0", or CPU, such as "cpu". Supports specifying multiple devices simultaneously for parallel inference. For details, please refer to <a href="../../instructions/parallel_inference.en.md#specifying-multiple-inference-devices">Pipeline Parallel Inference</a>.</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable the high-performance inference plugin. If set to <code>None</code>, the setting from the configuration file or <code>config</code> will be used.</td>
<td><code>bool</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>High-performance inference configuration</td>
<td><code>dict</code> | <code>None</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>engine</code></td>
<td>Inference engine</td>
<td><code>str | None</code></td>
<td>Optional <code>paddle</code>, <code>paddle_static</code>, <code>paddle_dynamic</code>, <code>hpi</code>, <code>flexible</code>, <code>transformers</code>, <code>genai_client</code>.</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>engine_config</code></td>
<td>Inference engine configuration</td>
<td><code>dict | None</code></td>
<td>Different engines support different fields, please refer to <a href="../../instructions/pipeline_python_API.en.md#4-inference-engine-and-configuration">Inference Engine and Configuration</a>.</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>pp_option</code></td>
<td>Used for changing runtime mode and other configuration items</td>
<td><code>PaddlePredictorOption</code></td>
<td>For detailed inference configuration, please refer to <a href="../../instructions/pipeline_python_API.en.md#5-compatibility-configuration-paddlepredictoroption">Compatible Configuration (PaddlePredictorOption)</a>.</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the pipeline object to perform inference prediction. This method will return a `generator`. Below are the parameters and their descriptions for the `predict()` method:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supports multiple input types, required</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
<li><b>str</b>: Local path of an image file or PDF file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the web URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">example</a>; <b>Local directory</b>, the directory should contain images to be predicted, such as the local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories; PDF files must be specified with an exact file path)</li>
<li><b>List</b>: Elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use the document orientation classification module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the document unwarping module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to use the text line orientation classification module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_general_ocr</code></td>
<td>Whether to use the OCR sub-line</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>Whether to use the seal recognition sub-line</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, initialized as <code>False</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>Whether to use the table recognition sub-line</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td>Whether to use the formula recognition sub-line</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>Whether to use the chart recognition sub-production line</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>False</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td>Whether to use the document region detection production line</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>format_block_content</code></td>
<td>Whether to format the content in <code>block_content</code> as Markdown. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence. When set to <code>True</code>, the <code>block_content</code> of image-type blocks will contain image path information (e.g., <code>&lt;img src="..." /&gt;</code>). When set to <code>False</code> (default), the <code>block_content</code> of image-type blocks will only contain OCR-recognized text content without image paths. To include image paths in JSON output, set this parameter to <code>True</code>.</td>
<td><code>bool|None</code></td>
<td></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Layout model score threshold</td>
<td><code>float|dict|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number between <code>0</code> and <code>1</code>;</li>
<li><b>dict</b>: <code>{0:0.1}</code> where the key is the category ID and the value is the threshold for that category;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>0.5</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether the layout area detection model uses NMS post-processing</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Expansion ratio of the detection box for the layout area detection model</td>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>;</li>
<li><b>Tuple[float,float]</b>: The expansion ratios in the horizontal and vertical directions, respectively;</li>
<li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code>, values as float scaling factors, e.g., <code>{0: (1.1, 2.0)}</code> means cls_id 0 expanding the width by 1.1 times and the height by 2.0 times while keeping the center unchanged</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>1.0</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Overlap box filtering method for layout area detection</td>
<td><code>str|dict|None</code></td>
<td>
<ul>
<li><b>str</b>: <code>large</code>, <code>small</code>, <code>union</code>, representing whether to retain the larger box, the smaller box, or both during overlap box filtering;</li>
<li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code> and values as merging modes, e.g., <code>{0: "large", 2: "small"}</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>large</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>Image side length limit for text detection</td>
<td><code>int|None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized for this parameter in the pipeline will be used, initialized as <code>960</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>Type of image side length limit for text detection</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> ensures that the shortest side of the image is not less than <code>det_limit_side_len</code>, and <code>max</code> ensures that the longest side of the image is not greater than <code>limit_side_len</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, initialized as <code>max</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Detection pixel threshold. In the output probability map, only pixels with scores greater than this threshold will be considered as text pixels.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, which is <code>0.3</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Detection box threshold. The detection result will be considered as a text area only if the average score of all pixels within the bounding box is greater than this threshold.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, which is <code>0.6</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Text detection expansion ratio. This method is used to expand the text area. The larger this value, the larger the expansion area.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized in the pipeline will be used, which is <code>2.0</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Text recognition threshold, text results with scores greater than this threshold will be retained</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter <code>0.0</code>. That is, no threshold is set</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Image side length limit for seal detection</td>
<td><code>int|None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter, initialized as <code>960</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>Type of image side length limit for seal detection</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>, <code>min</code> indicates that the shortest side of the image is not less than <code>det_limit_side_len</code>, <code>max</code> indicates that the longest side of the image is not greater than <code>limit_side_len</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter, initialized as <code>max</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Detection pixel threshold, in the output probability map, pixels with scores greater than this threshold will be considered as seal pixels</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter <code>0.3</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Detection box threshold, within the detection result bounding box, if the average score of all pixels is greater than this threshold, the result will be considered as a seal area</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter <code>0.6</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Expansion ratio for seal detection, this method is used to expand the text area, the larger this value, the larger the expanded area</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter <code>2.0</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Seal recognition threshold, text results with scores greater than this threshold will be retained</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the production-initialized value of this parameter <code>0.0</code>. That is, no threshold is set</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_wired_table_cells_trans_to_html</code></td>
<td>Whether to enable direct conversion of wired table cell detection results to HTML. Default is False. If enabled, HTML will be constructed directly based on the geometric relationship of wired table cell detection results.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>False</code>;</li>
</ul>
<td><code>False</code></td>
</td>
</tr>
<tr>
<td><code>use_wired_table_cells_trans_to_html</code></td>
<td>Whether to enable direct conversion of wireless table cell detection results to HTML. Default is False. If enabled, HTML will be constructed directly based on the geometric relationship of wireless table cell detection results.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>False</code>;</li>
</ul>
<td><code>False</code></td>
</td>
</tr>
<tr>
<td><code>use_table_orientation_classify</code></td>
<td>Whether to enable table orientation classification. When enabled, it can correct the orientation and correctly complete table recognition if the table in the image is rotated by 90/180/270 degrees.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>use_ocr_results_with_table_cells</code></td>
<td>Whether to enable OCR within cell segmentation. When enabled, OCR detection results will be segmented and re-recognized based on cell prediction results to avoid text loss.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>use_e2e_wired_table_rec_model</code></td>
<td>Whether to enable end-to-end wired table recognition mode. If enabled, the cell detection model will not be used, and only the table structure recognition model will be used.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>False</code>;</li>
</ul>
</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_e2e_wireless_table_rec_model</code></td>
<td>Whether to enable end-to-end wireless table recognition mode. If enabled, the cell detection model will not be used, and only the table structure recognition model will be used.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the initialized parameter value, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>markdown_ignore_labels</code></td>
<td>Layout labels that need to be ignored in Markdown. If set to <code>None</code>, the initialized default value will be used: <code>['number','footnote','header','header_image','footer','footer_image','aside_text']</code></td>
<td><code>list|None</code></td>
<td></td>
</tr>
</table>

(3) Process the prediction results: The prediction result of each sample is a corresponding Result object, and it supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file will have the same name as the input file</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the visualization images of each module in PNG format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_markdown()</code></td>
<td>Saves each page of the image or PDF file as a markdown formatted file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Save the table in the file as an HTML file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Save the table in the file as an XLSX file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_word()</code></td>
<td>Save the layout parsing result as a Word (.docx) format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal. The content printed to the terminal is explained as follows:
    - `input_path`: `(str)` The input path of the image to be predicted.

    - `page_index`: `(Union[int, None])` If the input is a PDF file, this indicates which page of the PDF it is; otherwise, it is `None`.

    - `page_count`: `(Union[int, None])` If the input is a PDF file, it indicates the total number of pages in the PDF; otherwise, it is `None`.

    - `width`: `(int)` The width of the original input image.

    - `height`: `(int)` The height of the original input image.

    - `model_settings`: `(Dict[str, bool])` Model parameters required for configuring the pipeline.

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessor sub-line.
        - `use_general_ocr`: `(bool)` Controls whether to enable the OCR sub-line.
        - `use_seal_recognition`: `(bool)` Controls whether to enable the seal recognition sub-line.
        - `use_table_recognition`: `(bool)` Controls whether to enable the table recognition sub-line.
        - `use_formula_recognition`: `(bool)` Controls whether to enable the formula recognition sub-line.
        - `format_block_content`: `(bool)` Controls whether to format the `block_content` into Markdown format. When set to `True`, the `block_content` of image-type blocks will contain image path information (e.g., `<img src="..." />`). When set to `False` (default), the `block_content` of image-type blocks will only contain OCR-recognized text content without image paths. To include image paths in JSON output, set this parameter to `True`.
        - `markdown_ignore_labels`: `(List[str])` Labels of layout regions that need to be ignored in Markdown, defaulting to `['number','footnote','header','header_image','footer','footer_image','aside_text']`

    - `parsing_res_list`: `(List[Dict])` A list of parsing results, where each element is a dictionary. The order of the list is the reading order after parsing.
        - `block_bbox`: `(np.ndarray)` The bounding box of the layout area.
        - `block_label`: `(str)` The label of the layout area, such as `text`, `table`, etc.
        - `block_content`: `(str)` The content within the layout area.
        - `block_id`: `(int)` The index of the layout area, used to display the layout sorting result.
        - `block_order`: `(int)` The order of the layout area, used to display the reading order of the layout. For non-ordered parts, the default value is `None`.

    - `overall_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` A dictionary of global OCR results
      -  `input_path`: `(Union[str, None])` The image path accepted by the image OCR sub-line. When the input is a `numpy.ndarray`, it is saved as `None`.
      - `model_settings`: `(Dict)` Model configuration parameters for the OCR sub-line.
      - `dt_polys`: `(List[numpy.ndarray])` A list of polygon boxes for text detection. Each detection box is represented by a numpy array of 4 vertex coordinates, with a shape of (4, 2) and a data type of int16.
      - `dt_scores`: `(List[float])` A list of confidence scores for text detection boxes.
      - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the text detection module.
        - `limit_side_len`: `(int)` The side length limit value during image preprocessing.
        - `limit_type`: `(str)` The processing method for side length limits.
        - `thresh`: `(float)` The confidence threshold for text pixel classification.
        - `box_thresh`: `(float)` The confidence threshold for text detection boxes.
        - `unclip_ratio`: `(float)` The expansion ratio for text detection boxes.
        - `text_type`: `(str)` The type of text detection, currently fixed as "general".

      - `text_type`: `(str)` The type of text detection, currently fixed as "general".
      - `textline_orientation_angles`: `(List[int])` The prediction results for text line orientation classification. When enabled, it returns the actual angle values (e.g., [0,0,1]).
      - `text_rec_score_thresh`: `(float)` The filtering threshold for text recognition results.
      - `rec_texts`: `(List[str])` A list of text recognition results, containing only texts with confidence scores above `text_rec_score_thresh`.
      - `rec_scores`: `(List[float])` A list of confidence scores for text recognition, filtered by `text_rec_score_thresh`.
      - `rec_polys`: `(List[numpy.ndarray])` A list of text detection boxes filtered by confidence score, in the same format as `dt_polys`.

    - `text_paragraphs_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` OCR results for paragraphs, excluding paragraphs of layout types such as tables, seals, and formulas.
        - `rec_polys`: `(List[numpy.ndarray])` A list of text detection boxes, in the same format as `dt_polys`.
        - `rec_texts`: `(List[str])` A list of text recognition results.
        - `rec_scores`: `(List[float])` A list of confidence scores for text recognition results.
        - `rec_boxes`: `(numpy.ndarray)` An array of rectangular bounding boxes for detection boxes, with a shape of (n, 4) and a dtype of int16. Each row represents a rectangle.

    - `formula_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` A list of formula recognition results, where each element is a dictionary.
        - `rec_formula`: `(str)` The result of formula recognition.
        - `rec_polys`: `(numpy.ndarray)` The detection box for the formula, with a shape of (4, 2) and a dtype of int16.
        - `formula_region_id`: `(int)` The region ID where the formula is located.

    - `seal_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` A list of seal recognition results, where each element is a dictionary.
        - `input_path`: `(str)` The input path of the seal image.
        - `model_settings`: `(Dict)` Model configuration parameters for the seal recognition sub-line.
        - `dt_polys`: `(List[numpy.ndarray])` A list of seal detection boxes, in the same format as `dt_polys`.
        - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the seal detection module, with the same specific parameter meanings as above.
        - `text_type`: `(str)` The type of seal detection, currently fixed as "seal".
        - `text_rec_score_thresh`: `(float)` The filtering threshold for seal recognition results.
        - `rec_texts`: `(List[str])` A list of seal recognition results, containing only texts with confidence scores above `text_rec_score_thresh`.
        - `rec_scores`: `(List[float])` A list of confidence scores for seal recognition, filtered by `text_rec_score_thresh`.
        - `rec_polys`: `(List[numpy.ndarray])` A list of seal detection boxes filtered by confidence score, in the same format as `dt_polys`.
        - `rec_boxes`: `(numpy.ndarray)` An array of rectangular bounding boxes for detection boxes, with a shape of (n, 4) and a dtype of int16. Each row represents a rectangle.

    - `table_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` A list of table recognition results, where each element is a dictionary.
        - `cell_box_list`: `(List[numpy.ndarray])` A list of bounding boxes for table cells.
        - `pred_html`: `(str)` The HTML format string of the table.
        - `table_ocr_pred`: `(dict)` The OCR recognition results of the table.
            - `rec_polys`: `(List[numpy.ndarray])` A list of detection boxes for cells.
            - `rec_texts`: `(List[str])` The recognition results of cells.
            - `rec_scores`: `(List[float])` The recognition confidence scores of cells.
            - `rec_boxes`: `(numpy.ndarray)` An array of rectangular bounding boxes for detection boxes, with a shape of (n, 4) and a dtype of int16. Each row represents a rectangle.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the save path will be `save_path/{your_img_basename}_res.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving NumPy arrays, `numpy.array` types will be converted to lists.
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, it will save images such as layout detection visualization, global OCR visualization, and layout reading order visualization. If a file is specified, it will be saved directly to that file. (The pipeline usually contains many result images, so it is not recommended to specify a specific file path directly; otherwise, multiple images will be overwritten, and only the last image will be retained.)
- Calling the `save_to_markdown()` method will save the converted Markdown file to the specified `save_path`. The save path will be `save_path/{your_img_basename}.md`. If the input is a PDF file, it is recommended to specify a directory directly; otherwise, multiple Markdown files will be overwritten.

In addition, it also supports obtaining visualized images with results and prediction results through attributes, as follows:
<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get the visualized image in <code>dict</code> format</td>
</tr>
<thead>
<tr>
<th>Property</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>json</code></td>
<td>Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get the visualization image in <code>dict</code> format</td>
</tr>
<tr>
</tr>
<tr>
<td rowspan="3"><code>markdown</code></td>
<td rowspan="3">Get the markdown result in <code>dict</code> format</td>
</tr>
<tr>
</tr>
<tr>
</tr>
</tbody>
</table>

- The prediction result obtained through the `json` attribute is data of the dict type, and its content is consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is data of the dictionary type. The keys are `layout_det_res`, `overall_ocr_res`, `text_paragraphs_ocr_res`, `formula_res_region1`, `table_cell_img`, and `seal_res_region1`, and the corresponding values are `Image.Image` objects: used to display the visualized images of layout detection, OCR, OCR text paragraphs, formulas, tables, and seal results. If the optional module is not used, the dictionary only contains `layout_det_res`.
- - The `markdown` property returns the prediction result as a dictionary. The keys are `markdown_texts` and `markdown_images`, corresponding to markdown text and images for display in Markdown (`Image.Image` objects).

In addition, you can obtain the PP-StructureV3 pipeline configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config PP-StructureV3 --save_path ./my_path
```

If you have obtained the configuration file, you can customize the PP-StructureV3 pipeline configuration. You just need to modify the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file. The example is as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/PP-StructureV3.yaml")
output = pipeline.predict(
    input="./pp_structure_v3_demo.png",,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print()
    res.print() ## Print the structured prediction output
    res.save_to_json(save_path="output") ## Save the structured JSON result of the current image
    res.save_to_markdown(save_path="output") ## Save the result of the current image in Markdown format
```

<b>Note:</b> The parameters in the configuration file are the pipeline initialization parameters. If you wish to change the initialization parameters of the PP-StructureV3 pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in a configuration file, simply specify the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the pipeline directly into your Python project, you can refer to the example code in [2.2 Python Script Method](#22-python脚本方式集成).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have strict performance requirements (especially response speed) for deployment strategies to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin aimed at deeply optimizing the performance of model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Serving Deployment</b>: Serving Deployment is a common form of deployment in actual production environments. By encapsulating the inference functionality into a service, clients can access these services through network requests to obtain inference results. PaddleX supports various serving deployment solutions for pipelines. For detailed procedures, please refer to the [PaddleX Serving Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below is the API reference for basic serving deployment and examples of service calls in multiple languages:

<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>The result of the operation.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Same as the response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Perform layout parsing.</p>
<p><code>POST /layout-parsing</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>file</code></td>
<td><code>string</code></td>
<td>The URL of an image or PDF file accessible by the server, or the Base64-encoded content of the above file types. By default, for PDF files exceeding 10 pages, only the content of the first 10 pages will be processed.<br />
To remove the page limit, please add the following configuration to the pipeline configuration file:
<pre><code>Serving:
  extra:
    max_num_input_imgs: null
</code></pre></td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code>｜<code>null</code></td>
<td>File type. <code>0</code> represents a PDF file, and <code>1</code> represents an image file. If this attribute is missing from the request body, the file type will be inferred based on the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_orientation_classify</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_unwarping</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTextlineOrientation</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_textline_orientation</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useSealRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_seal_recognition</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_table_recognition</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useFormulaRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_formula_recognition</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useChartRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_chart_recognition</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useRegionDetection</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_region_detection</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>formatBlockContent</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>format_block_content</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>object</code> | </code><code>null</code></td>
<td>Please refer to the description of the <code>layout_threshold</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_nms</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_unclip_ratio</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_merge_bboxes_mode</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_limit_side_len</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_limit_type</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_box_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_unclip_ratio</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_rec_score_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_limit_side_len</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_limit_type</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_box_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_unclip_ratio</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_rec_score_thresh</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useWiredTableCellsTransToHtml</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_wired_table_cells_trans_to_html</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useWirelessTableCellsTransToHtml</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_wireless_table_cells_trans_to_html</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableOrientationClassify</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_table_orientation_classify</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useOcrResultsWithTableCells</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_ocr_results_with_table_cells</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useE2eWiredTableRecModel</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_e2e_wired_table_rec_model</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useE2eWirelessTableRecModel</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_e2e_wireless_table_rec_model</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>markdownIgnoreLabels</code></td>
<td><code>array</code> | <code>null</code></td>
<td>Please refer to the description of the <code>markdown_ignore_labels</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>prettifyMarkdown</code></td>
<td><code>boolean</code></td>
<td>Whether to output beautified Markdown text. The default is <code>true</code>.</td>
<td>No</td>
</tr>
<tr>
<td><code>showFormulaNumber</code></td>
<td><code>boolean</code></td>
<td>Whether to include formula numbers in the output Markdown text. The default is <code>false</code>.</td>
<td>No</td>
</tr>
<tr>
<td><code>outputFormats</code></td>
<td><code>array</code> | <code>null</code></td>
<td>Optional list of extra formats to return. Currently only <code>"docx"</code> is supported.</td>
<td>No</td>
</tr>
<tr>
<td><code>visualize</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>
Whether to return the final visualization image and intermediate images during the processing.<br/>
<ul style="margin: 0 0 0 1em; padding-left: 0em;">
<li>If <code>true</code> is provided: return images.</li>
<li>If <code>false</code> is provided: do not return any images.</li>
<li>If this parameter is omitted from the request body, or if <code>null</code> is explicitly passed, the behavior will follow the value of <code>Serving.visualize</code> in the pipeline configuration.</li>
</ul>
<br/>
For example, adding the following setting to the pipeline config file:<br/>
<pre><code>Serving:
  visualize: False
</code></pre>
will disable image return by default. This behavior can be overridden by explicitly setting the <code>visualize</code> parameter in the request.<br/>
If neither the request body nor the configuration file is set (If <code>visualize</code> is set to <code>null</code> in the request and  not defined in the configuration file), the image is returned by default.
</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> in the response body has the following attributes:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>layoutParsingResults</code></td>
<td><code>array</code></td>
<td>The layout parsing results. The array length is 1 (for image input) or the actual number of document pages processed (for PDF input). For PDF input, each element in the array represents the result of each page actually processed in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>layoutParsingResults</code> is an <code>object</code> with the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>A simplified version of the <code>res</code> field in the JSON representation of the result generated by the <code>predict</code> method of the pipeline object, with the <code>input_path</code> and the <code>page_index</code> fields removed.</td>
</tr>
<tr>
<td><code>markdown</code></td>
<td><code>object</code></td>
<td>The Markdown result.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>See the description of the <code>img</code> attribute of the result of the pipeline prediction. The images are in JPEG format and are Base64-encoded.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>The input image. The image is in JPEG format and is Base64-encoded.</td>
</tr>
<tr>
<td><code>exports</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Optional additional exports when <code>outputFormats</code> is present—for example, <code>{"docx": {"content": "..."}}</code>, where <code>content</code> is the Base64-encoded file content.</td>
</tr>
</tbody>
</table>
<p><code>markdown</code> is an <code>object</code> with the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>The Markdown text.</td>
</tr>
<tr>
<td><code>images</code></td>
<td><code>object</code></td>
<td>A key-value pair of relative paths of Markdown images and Base64-encoded images.</td>
</tr>
<tr>
<td><code>isStart</code></td>
<td><code>boolean</code></td>
<td>Whether the first element on the current page is the start of a segment.</td>
</tr>
<tr>
<td><code>isEnd</code></td>
<td><code>boolean</code></td>
<td>Whether the last element on the current page is the end of a segment.</td>
</tr>
</tbody>
</table>
</details>

<details><summary>Multi-language Service Call Example</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">
import base64
import requests
import pathlib

API_URL = "http://localhost:8080/layout-parsing" # Service URL

image_path = "./demo.jpg"

# Encode the local image with Base64
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "file": image_data, # Base64-encoded file content or file URL
    "fileType": 1, # file type, 1 represents image file
}

# Call the API
response = requests.post(API_URL, json=payload)

# Process the response data
assert response.status_code == 200
result = response.json()["result"]
print("\nDetected layout elements:")
for i, res in enumerate(result["layoutParsingResults"]):
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

    auto response = client.Post("/layout-parsing", jsonObj.dump(), "application/json");

    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        if (!result.is_object() || !result.contains("layoutParsingResults")) {
            std::cerr << "Unexpected response format." << std::endl;
            return 1;
        }

        const auto& results = result["layoutParsingResults"];
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& res = results[i];

            if (res.contains("prunedResult")) {
                std::cout << "Layout result [" << i << "]: " << res["prunedResult"].dump() << std::endl;
            }

            if (res.contains("outputImages") && res["outputImages"].is_object()) {
                for (auto& [imgName, imgBase64] : res["outputImages"].items()) {
                    std::string outputPath = imgName + "_" + std::to_string(i) + ".jpg";
                    std::string decodedImage = base64::from_base64(imgBase64.get<std::string>());

                    std::ofstream outFile(outputPath, std::ios::binary);
                    if (outFile.is_open()) {
                        outFile.write(decodedImage.c_str(), decodedImage.size());
                        outFile.close();
                        std::cout << "Saved image: " << outputPath << std::endl;
                    } else {
                        std::cerr << "Failed to save image: " << outputPath << std::endl;
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
        String API_URL = "http://localhost:8080/layout-parsing";
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

                JsonNode layoutParsingResults = result.get("layoutParsingResults");
                for (int i = 0; i < layoutParsingResults.size(); i++) {
                    JsonNode item = layoutParsingResults.get(i);
                    int finalI = i;
                    JsonNode prunedResult = item.get("prunedResult");
                    System.out.println("Pruned Result [" + i + "]: " + prunedResult.toString());

                    JsonNode outputImages = item.get("outputImages");
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
    "path/filepath"
)

func main() {
    API_URL := "http://localhost:8080/layout-parsing"
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
        fmt.Printf("Error reading response: %v\n", err)
        return
    }

    type Markdown struct {
        Text   string            `json:"text"`
        Images map[string]string `json:"images"`
    }

    type LayoutResult struct {
        PrunedResult map[string]interface{} `json:"prunedResult"`
        Markdown     Markdown               `json:"markdown"`
        OutputImages map[string]string      `json:"outputImages"`
        InputImage   *string                `json:"inputImage"`
    }

    type Response struct {
        Result struct {
            LayoutParsingResults []LayoutResult `json:"layoutParsingResults"`
            DataInfo             interface{}    `json:"dataInfo"`
        } `json:"result"`
    }

    var respData Response
    if err := json.Unmarshal(body, &respData); err != nil {
        fmt.Printf("Error parsing response: %v\n", err)
        return
    }

    for i, res := range respData.Result.LayoutParsingResults {
        fmt.Printf("Result %d - prunedResult: %+v\n", i, res.PrunedResult)

        mdDir := fmt.Sprintf("markdown_%d", i)
        os.MkdirAll(mdDir, 0755)
        mdFile := filepath.Join(mdDir, "doc.md")
        if err := os.WriteFile(mdFile, []byte(res.Markdown.Text), 0644); err != nil {
            fmt.Printf("Error writing markdown file: %v\n", err)
        } else {
            fmt.Printf("Markdown document saved at %s\n", mdFile)
        }

        for path, imgBase64 := range res.Markdown.Images {
            fullPath := filepath.Join(mdDir, path)
            os.MkdirAll(filepath.Dir(fullPath), 0755)
            imgBytes, err := base64.StdEncoding.DecodeString(imgBase64)
            if err != nil {
                fmt.Printf("Error decoding markdown image: %v\n", err)
                continue
            }
            if err := os.WriteFile(fullPath, imgBytes, 0644); err != nil {
                fmt.Printf("Error saving markdown image: %v\n", err)
            }
        }

        for name, imgBase64 := range res.OutputImages {
            imgBytes, err := base64.StdEncoding.DecodeString(imgBase64)
            if err != nil {
                fmt.Printf("Error decoding output image %s: %v\n", name, err)
                continue
            }
            filename := fmt.Sprintf("%s_%d.jpg", name, i)
            if err := os.WriteFile(filename, imgBytes, 0644); err != nil {
                fmt.Printf("Error saving output image %s: %v\n", filename, err)
            } else {
                fmt.Printf("Output image saved at %s\n", filename)
            }
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
    static readonly string API_URL = "http://localhost:8080/layout-parsing";
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

        JArray layoutParsingResults = (JArray)jsonResponse["result"]["layoutParsingResults"];
        for (int i = 0; i < layoutParsingResults.Count; i++)
        {
            var res = layoutParsingResults[i];
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

const API_URL = 'http://localhost:8080/layout-parsing';
const imagePath = './demo.jpg';
const fileType = 1;

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

const payload = {
  file: encodeImageToBase64(imagePath),
  fileType: fileType
};

axios.post(API_URL, payload)
  .then(response => {
    const results = response.data.result.layoutParsingResults;
    results.forEach((res, index) => {
      console.log(`\n[${index}] prunedResult:`);
      console.log(res.prunedResult);

      const outputImages = res.outputImages;
      if (outputImages) {
        Object.entries(outputImages).forEach(([imgName, base64Img]) => {
          const imgPath = `${imgName}_${index}.jpg`;
          fs.writeFileSync(imgPath, Buffer.from(base64Img, 'base64'));
          console.log(`Output image saved at ${imgPath}`);
        });
      } else {
        console.log(`[${index}] No outputImages.`);
      }
    });
  })
  .catch(error => {
    console.error('Error during API request:', error.message || error);
  });
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/layout-parsing";
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

$result = json_decode($response, true)["result"]["layoutParsingResults"];

foreach ($result as $i => $item) {
    echo "[$i] prunedResult:\n";
    print_r($item["prunedResult"]);

    if (!empty($item["outputImages"])) {
        foreach ($item["outputImages"] as $img_name => $img_base64) {
            $output_image_path = "{$img_name}_{$i}.jpg";
            file_put_contents($output_image_path, base64_decode($img_base64));
            echo "Output image saved at $output_image_path\n";
        }
    } else {
        echo "No outputImages found for item $i\n";
    }
}
?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>On-Device Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on user devices, allowing the device to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX On-Device Deployment Guide](../../../pipeline_deploy/on_device_deployment.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model into your pipeline and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the PP-StructureV3 pipeline do not meet your requirements in terms of accuracy or speed, you can try to <b>fine-tune</b> the existing model using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the PP-StructureV3 pipeline in your scenario.

### 4.1 Model Fine-Tuning
Since the PP-StructureV3 pipeline includes several modules, the unsatisfactory performance of the pipeline may originate from any one of these modules. You can analyze the cases with poor extraction results, identify which module is problematic through visualizing the images, and refer to the corresponding fine-tuning tutorial links in the table below to fine-tune the model.

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Fine-tuning Module</th>
<th>Fine-tuning Reference Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Inaccurate layout region detection, such as missed detection of seals, tables, etc.</td>
<td>Layout Region Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate table structure recognition</td>
<td>Table Structure Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_structure_recognition.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate formula recognition</td>
<td>Formula Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/formula_recognition.html">Link</a></td>
</tr>
<tr>
<td>Missed detection of seal text</td>
<td>Seal Text Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/seal_text_detection.html">Link</a></td>
</tr>
<tr>
<td>Missed detection of text</td>
<td>Text Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_detection.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate text content</td>
<td>Text Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_recognition.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of vertical or rotated text lines</td>
<td>Text Line Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of whole-image rotation</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of image distortion</td>
<td>Text Image Correction Module</td>
<td>Fine-tuning not supported</td>
</tr>
</tbody>
</table>

### 4.2 Model Application
After you complete fine-tuning with your private dataset, you will obtain a local model weight file.

If you need to use the fine-tuned model weights, simply modify the production configuration file by replacing the local path of the fine-tuned model weights to the corresponding position in the production configuration file:

```yaml
......
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayout_plus-L
    model_dir: null
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
        model_name: PP-OCRv5_server_rec
        model_dir: null
        limit_side_len: 960
        limit_type: max
        max_side_limit: 4000
        thresh: 0.3
        box_thresh: 0.6
        unclip_ratio: 1.5

      TextRecognition:
        module_name: text_recognition
        model_name: PP-OCRv5_server_rec
        model_dir: null
        batch_size: 1
        score_thresh: 0
......
```

Subsequently, refer to the command-line method or Python script method in the local experience to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices such as NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware devices.

For example, if you use Ascend NPU for PP-StructureV3 pipeline inference, the CLI command you use is:

```bash
paddlex --pipeline PP-StructureV3 \
        --input pp_structure_v3_demo.png  \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --use_e2e_wireless_table_rec_model True \
        --save_path ./output \
        --device npu:0
```

Of course, you can also specify the hardware device when calling `create_pipeline()` or `predict()` in your Python script.

If you want to use the PP-StructureV3 pipeline on a wider range of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
