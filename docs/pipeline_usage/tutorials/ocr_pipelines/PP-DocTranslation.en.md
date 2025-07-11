---
comments: true
---

# PP-DocTranslation Pipeline Tutorial

## 1. Introduction to PP-DocTranslation Pipeline
The General Document Translation Pipeline (PP-DocTranslation) is a document intelligent translation solution provided by PaddlePaddle. It integrates advanced general layout analysis technology with the capabilities of large language models (LLMs) to offer you efficient document intelligent translation services. This solution can accurately identify and extract various elements within documents, including text blocks, headings, paragraphs, images, tables, and other complex layout structures, and on this basis, achieve high-quality multilingual translation. PP-DocTranslation supports mutual translation among multiple mainstream languages, and is particularly adept at handling documents with complex layouts and strong contextual dependencies, striving to deliver accurate, natural, fluent, and professional translation results. This pipeline also provides flexible Serving deployment options, supporting the use of multiple programming languages across various hardware. Moreover, it offers the capability for secondary development, allowing you to train and fine-tune models on your own datasets, with the trained models being seamlessly integrable.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/doc_translation/pp_doctranslation.png"/>

<b>The general document translation pipeline uses PP-StructureV3 sub-pipeline, and thus has all the functions of PP-StructureV3. For more information on the functions and usage details of PP-StructureV3, click on the [PP-StructureV3 Documentation](./PP-StructureV3.en.md) page.</b>

<b>If you prioritize model accuracy, choose a high-accuracy model; if you prioritize model inference speed, choose a faster inference model; if you prioritize model storage size, choose a smaller storage model.</b>

<details><summary>👉Details of model list</summary>
<p><b>Document image orientation classification module:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model download link</th>
<th>Top-1 Acc (%)</th>
<th>GPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>CPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>Model storage size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training model</a></td>
<td>99.06</td>
<td>2.62 / 0.59</td>
<td>3.24 / 1.19</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, with four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees</td>
</tr>
</tbody>
</table>
<p><b>Text image rectification module:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model download link</th>
<th>CER</th>
<th>Model storage size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training model</a></td>
<td>0.179</td>
<td>30.3 M</td>
<td>A high-precision text image rectification model</td>
</tr>
</tbody>
</table>
<p><b>Layout region detection module model:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model download link</th>
<th>mAP(0.5) (%)</th>
<th>GPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>CPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>Model storage size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout_plus-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">Training model</a></td>
<td>83.2</td>
<td>53.03 / 17.23</td>
<td>634.62 / 378.32</td>
<td>126.01 M</td>
<td>A higher-precision layout region localization model trained on a self-built dataset based on RT-DETR-L, covering scenarios such as Chinese and English papers, multi-column magazines, newspapers, PPTs, contracts, books, examination papers, research reports, ancient books, Japanese documents, and documents with vertical text.</td>
</tr>
<td>PP-DocLayout-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Training model</a></td>
<td>90.4</td>
<td>33.59 / 33.59</td>
<td>503.01 / 251.08</td>
<td>123.76 M</td>
<td>A high-precision layout region localization model trained on a self-built dataset based on RT-DETR-L, covering scenarios such as Chinese and English papers, magazines, contracts, books, examination papers, and research reports.</td>
<tr>
<td>PP-DocLayout-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training model</a></td>
<td>75.2</td>
<td>13.03 / 4.72</td>
<td>43.39 / 24.44</td>
<td>22.578</td>
<td>A layout region localization model with balanced precision and efficiency trained on a self-built dataset based on PicoDet-L, covering scenarios such as Chinese and English papers, magazines, contracts, books, examination papers, and research reports.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training model</a></td>
<td>70.9</td>
<td>11.54 / 3.86</td>
<td>18.53 / 6.29</td>
<td>4.834</td>
<td>A highly efficient layout region localization model trained on a self-built dataset based on PicoDet-S, covering scenarios such as Chinese and English papers, magazines, contracts, books, examination papers, and research reports.</td>
</tr>
</tbody>
</table>
<p><b>Table structure recognition module:</b></p>
<table>
<tr>
<th>Model</th><th>Model download link</th>
<th>Accuracy (%)</th>
<th>GPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>CPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>Model storage size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wired_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">Training model</a></td>
<td rowspan="2">69.65</td>
<td rowspan="2">85.92 / 85.92</td>
<td rowspan="2">- / 501.66</td>
<td rowspan="2">351M</td>
<td rowspan="2">The SLANeXt series is a new generation of table structure recognition models independently developed by Baidu PaddlePaddle's vision team. Compared to SLANet and SLANet_plus, SLANeXt focuses on recognizing table structures and has trained dedicated weights for wired and wireless tables separately. Its recognition capabilities for various types of tables have been significantly improved, especially for wired tables.</td>
</tr>
<tr>
<td>SLANeXt_wireless</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wireless_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wireless_pretrained.pdparams">Training model</a></td>
</tr>
</table>
<p><b>Table classification module model:</b></p>
<table>
<tr>
<th>Model</th><th>Model download link</th>
<th>Top1 Acc(%)</th>
<th>GPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>CPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>Model storage size (M)</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CLIP_vit_base_patch16_224_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">Training model</a></td>
<td>94.2</td>
<td>2.62 / 0.60</td>
<td>3.17 / 1.14</td>
<td>6.6M</td>
</tr>
</table>
<p><b>Table cell detection module model:</b></p>
<table>
<tr>
<th>Model</th><th>Model download link</th>
<th>mAP(%)</th>
<th>GPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>CPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>Model storage size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wired_table_cell_det_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">Training model</a></td>
<td rowspan="2">82.7</td>
<td rowspan="2">33.47 / 27.02</td>
<td rowspan="2">402.55 / 256.56</td>
<td rowspan="2">124M</td>
<td rowspan="2">RT-DETR is the first real-time end-to-end object detection model. The Baidu PaddlePaddle Vision team completed pre-training on a self-built table cell detection dataset using RT-DETR-L as the base model, achieving table cell detection with good performance for both wired and wireless tables.</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wireless_table_cell_det_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams">Training model</a></td>
</tr>
</table>
<p><b>Text detection module:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model download link</th>
<th>Detection Hmean (%)</th>
<th>GPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>CPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>Model storage size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv5_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams">Training model</a></td>
<td>83.8</td>
<td>89.55 / 70.19</td>
<td>383.15 / 383.15</td>
<td>84.3</td>
<td>The server-side text detection model of PP-OCRv5, with higher accuracy, suitable for deployment on servers with better performance</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams">Training model</a></td>
<td>79.0</td>
<td>10.67 / 6.36</td>
<td>57.77 / 28.15</td>
<td>4.7</td>
<td>The mobile-side text detection model of PP-OCRv5, with higher efficiency, suitable for deployment on end-side devices</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Training model</a></td>
<td>69.2</td>
<td>127.82 / 98.87</td>
<td>585.95 / 489.77</td>
<td>109</td>
<td>PP-OCRv4 server-side text detection model with higher accuracy, suitable for deployment on servers with better performance</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Training model</a></td>
<td>63.8</td>
<td>9.87 / 4.17</td>
<td>56.60 / 20.79</td>
<td>4.7</td>
<td>PP-OCRv4 mobile-side text detection model with higher efficiency, suitable for deployment on edge devices</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_det_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_det_pretrained.pdparams">Training model</a></td>
<td>Accuracy close to PP-OCRv4_mobile_det</td>
<td>9.90 / 3.60</td>
<td>41.93 / 20.76</td>
<td>2.1</td>
<td>PP-OCRv3 mobile-side text detection model with higher efficiency, suitable for deployment on edge devices</td>
</tr>
<tr>
<td>PP-OCRv3_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_server_det_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_server_det_pretrained.pdparams">Training model</a></td>
<td>Accuracy close to PP-OCRv4_server_det</td>
<td>119.50 / 75.00</td>
<td>379.35 / 318.35</td>
<td>102.1</td>
<td>PP-OCRv3 server-side text detection model with higher accuracy, suitable for deployment on servers with better performance</td>
</tr>
</tbody>
</table>
<p><b>Text recognition module models:</b></p>*<b>Chinese recognition model</b>
<table>
<tr>
<th>Model</th><th>Model download link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>CPU inference time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">Training Model</a></td>
<td>86.38</td>
<td>8.46 / 2.36</td>
<td>31.21 / 31.21</td>
<td>81 M</td>
<td rowspan="2">PP-OCRv5_rec is a new-generation text recognition model. This model is dedicated to efficiently and accurately supporting the recognition of four major languages, namely Simplified Chinese, Traditional Chinese, English, and Japanese, as well as complex text scenarios such as handwriting, vertical text, pinyin, and rare characters, all within a single model. While maintaining recognition accuracy, it also balances inference speed and model robustness, providing efficient and accurate technical support for document understanding in various scenarios.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>81.29</td>
<td>5.43 / 1.46</td>
<td>21.20 / 5.32</td>
<td>16 M</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">Training Model</a></td>
<td>86.58</td>
<td>8.69 / 2.78</td>
<td>37.93 / 37.93</td>
<td>74.7 M</td>
<td>PP-OCRv4_server_rec_doc is trained on a mixed dataset of more Chinese document data and PP-OCR training data, based on PP-OCRv4_server_rec. It has enhanced the ability to recognize some Traditional Chinese characters, Japanese characters, and special characters, supporting the recognition of over 15,000 characters. In addition to improving the document-related text recognition capabilities, it has also enhanced the recognition capabilities for general text.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>78.74</td>
<td>5.26 / 1.12</td>
<td>17.48 / 3.61</td>
<td>10.6 M</td>
<td>Lightweight recognition model of PP-OCRv4 with high inference efficiency, which can be deployed on various hardware devices including edge devices</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training model</a></td>
<td>80.61</td>
<td>8.75 / 2.49</td>
<td>36.93 / 36.93</td>
<td>71.2 M</td>
<td>Server-side model of PP-OCRv4 with high inference accuracy, which can be deployed on various servers</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv3_mobile_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_rec_pretrained.pdparams">Training model</a></td>
<td>72.96</td>
<td>3.89 / 1.16</td>
<td>8.72 / 3.56</td>
<td>9.2 M</td>
<td>Lightweight recognition model of PP-OCRv3 with high inference efficiency, which can be deployed on various hardware devices including edge devices</td>
</tr>
</table>
<table>
<tr>
<th>Model</th><th>Model download link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>CPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>Model storage size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training model</a></td>
<td>68.81</td>
<td>10.38 / 8.31</td>
<td>66.52 / 30.83</td>
<td>73.9 M</td>
<td rowspan="1">SVTRv2 is a server-side text recognition model developed by the OpenOCR team of the Vision and Learning Lab (FVL) at Fudan University. It won the first prize in the PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task. The end-to-end recognition accuracy on Leaderboard A improved by 6% compared to PP-OCRv4.</td>
</tr>
</table>
<table>
<tr>
<th>Model</th><th>Model download link</th>
<th>Identify Avg Accuracy(%)</th>
<th>GPU inference time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU inference time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model storage size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training model</a></td>
<td>65.07</td>
<td>6.29 / 1.57</td>
<td>20.64 / 5.40</td>
<td>22.1 M</td>
<td rowspan="1">The RepSVTR text recognition model is a mobile-end text recognition model based on SVTRv2. It won the first prize in the PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task. Compared with PP-OCRv4, its end-to-end recognition accuracy on Leaderboard B has increased by 2.5%, while the inference speed remains the same.</td>
</tr>
</table>*<b>English recognition model</b>
<table>
<tr>
<th>Model</th><th>Model download link</th>
<th>Identify Avg Accuracy(%)</th>
<th>GPU inference time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU inference time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model storage size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">Training model</a></td>
<td>70.39</td>
<td>4.81 / 1.23</td>
<td>17.20 / 4.18</td>
<td>6.8 M</td>
<td>An ultra-lightweight English recognition model trained based on the PP-OCRv4 recognition model, supporting English and numeric recognition</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv3_mobile_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv3_mobile_rec_pretrained.pdparams">Training model</a></td>
<td>70.69</td>
<td>3.56 / 0.78</td>
<td>8.44 / 5.78</td>
<td>7.8 M</td>
<td>An ultra-lightweight English recognition model trained based on the PP-OCRv3 recognition model, supporting English and number recognition</td>
</tr>
</table>*<b>Multilingual recognition model</b>
<table>
<tr>
<th>Model</th><th>Model download link</th>
<th>Avg Accuracy of recognition (%)</th>
<th>GPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>CPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>Model storage size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
korean_PP-OCRv3_mobile_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv3_mobile_rec_pretrained.pdparams">Training model</a></td>
<td>60.21</td>
<td>3.73 / 0.98</td>
<td>8.76 / 2.91</td>
<td>8.6 M</td>
<td>An ultra-lightweight Korean recognition model trained based on the PP-OCRv3 recognition model, supporting Korean and number recognition</td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
japan_PP-OCRv3_mobile_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/japan_PP-OCRv3_mobile_rec_pretrained.pdparams">Training model</a></td>
<td>45.69</td>
<td>3.86 / 1.01</td>
<td>8.62 / 2.92</td>
<td>8.8 M</td>
<td>An ultra-lightweight Japanese recognition model trained based on the PP-OCRv3 recognition model, supporting Japanese and number recognition</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/chinese_cht_PP-OCRv3_mobile_rec_pretrained.pdparams">Training model</a></td>
<td>82.06</td>
<td>3.90 / 1.16</td>
<td>9.24 / 3.18</td>
<td>9.7 M</td>
<td>An ultra-lightweight traditional Chinese recognition model trained based on the PP-OCRv3 recognition model, supporting traditional Chinese and number recognition</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
te_PP-OCRv3_mobile_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/te_PP-OCRv3_mobile_rec_pretrained.pdparams">Training model</a></td>
<td>95.88</td>
<td>3.59 / 0.81</td>
<td>8.28 / 6.21</td>
<td>7.8 M</td>
<td>An ultra-lightweight Telugu recognition model trained based on the PP-OCRv3 recognition model, supporting Telugu and digit recognition</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ka_PP-OCRv3_mobile_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ka_PP-OCRv3_mobile_rec_pretrained.pdparams">Training model</a></td>
<td>96.96</td>
<td>3.49 / 0.89</td>
<td>8.63 / 2.77</td>
<td>8.0 M</td>
<td>An ultra-lightweight Kannada recognition model trained based on the PP-OCRv3 recognition model, supporting Kannada and digit recognition</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ta_PP-OCRv3_mobile_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv3_mobile_rec_pretrained.pdparams">Training model</a></td>
<td>76.83</td>
<td>3.49 / 0.86</td>
<td>8.35 / 3.41</td>
<td>8.0 M</td>
<td>An ultra-lightweight Tamil recognition model trained based on the PP-OCRv3 recognition model, supporting Tamil and digit recognition</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
latin_PP-OCRv3_mobile_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv3_mobile_rec_pretrained.pdparams">Training model</a></td>
<td>76.93</td>
<td>3.53 / 0.78</td>
<td>8.50 / 6.83</td>
<td>7.8 M</td>
<td>An ultra-lightweight Latin recognition model trained based on the PP-OCRv3 recognition model, supporting Latin and digit recognition</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
arabic_PP-OCRv3_mobile_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv3_mobile_rec_pretrained.pdparams">Training model</a></td>
<td>73.55</td>
<td>3.60 / 0.83</td>
<td>8.44 / 4.69</td>
<td>7.8 M</td>
<td>An ultra-lightweight Arabic letter recognition model trained based on the PP-OCRv3 recognition model, supporting Arabic letter and digit recognition</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/cyrillic_PP-OCRv3_mobile_rec_pretrained.pdparams">Training model</a></td>
<td>94.28</td>
<td>3.56 / 0.79</td>
<td>8.22 / 2.76</td>
<td>7.9 M</td>
<td>An ultra-lightweight Slavic letter recognition model trained based on the PP-OCRv3 recognition model, supporting Slavic letter and digit recognition</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/devanagari_PP-OCRv3_mobile_rec_pretrained.pdparams">Training model</a></td>
<td>96.44</td>
<td>3.60 / 0.78</td>
<td>6.95 / 2.87</td>
<td>7.9 M</td>
<td>An ultra-lightweight Devanagari letter recognition model trained based on the PP-OCRv3 recognition model, supporting Devanagari letter and digit recognition</td>
</tr>
</table>
<p><b>Text line orientation classification module (optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th>
<th>Model download link</th>
<th>Top-1 Acc (%)</th>
<th>GPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>CPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>Model storage size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">Training model</a></td>
<td>95.54</td>
<td>2.16 / 0.41</td>
<td>2.37 / 0.73</td>
<td>0.32</td>
<td>A text line classification model based on PP-LCNet_x0_25, with two categories, namely 0 degree and 180 degrees</td>
</tr>
</tbody>
</table>
<p><b>Formula recognition module:</b></p>
<table>
<tr>
<th>Model</th><th>Model download link</th>
<th>Avg-BLEU(%)</th>
<th>GPU inference latency (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU inference latency (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model storage size (M)</th>
<th>Introduction</th>
</tr>
<td>UniMERNet</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UniMERNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">Training Model</a></td>
<td>86.13</td>
<td>2266.96/-</td>
<td>-/-</td>
<td>1.4 G</td>
<td>UniMERNet is a formula recognition model developed by Shanghai AI Lab. It employs Donut Swin as the encoder and MBartDecoder as the decoder. By training on a dataset of one million entries, including simple formulas, complex formulas, scanned formulas, and handwritten formulas, the model significantly enhances its recognition accuracy for formulas in real-world scenarios.</td>
<td>PP-FormulaNet-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">Training Model</a></td>
<td>87.12</td>
<td>1311.84 / 1311.84</td>
<td>- / 8288.07</td>
<td>167.9 M</td>
<td rowspan="2">PP-FormulaNet is an advanced formula recognition model developed by Baidu PaddlePaddle Vision Team, supporting the recognition of 50,000 common LateX source code vocabulary. The PP-FormulaNet-S version adopts PP-HGNetV2-B4 as its backbone network. Through techniques such as parallel masking and model distillation, it significantly improves the model's inference speed while maintaining high recognition accuracy, suitable for scenarios such as simple printed formulas and cross-line simple printed formulas. The PP-FormulaNet-L version, on the other hand, is based on Vary_VIT_B as the backbone network and has undergone in-depth training on a large-scale formula dataset. It shows significant improvement in recognizing complex formulas compared to PP-FormulaNet-S and is suitable for scenarios such as simple printed formulas, complex printed formulas, and handwritten formulas.</td>
<td>PP-FormulaNet-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">Training Model</a></td>
<td>92.13</td>
<td>1976.52/-</td>
<td>-/-</td>
<td>535.2 M</td>
<td>LaTeX_OCR_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Training model</a></td>
<td>71.63</td>
<td>1088.89 / 1088.89</td>
<td>- / -</td>
<td>89.7 M</td>
<td>LaTeX-OCR is a formula recognition algorithm based on an autoregressive large model. By adopting Hybrid ViT as the backbone network and transformer as the decoder, it significantly improves the accuracy of formula recognition.</td>
</table>
<p><b>Seal text detection module:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model download link</th>
<th>Detection Hmean (%)</th>
<th>GPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>CPU inference time (ms)<br/>[Normal mode / High-performance mode]</th>
<th>Model storage size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Training model</a></td>
<td>98.21</td>
<td>124.64 / 91.57</td>
<td>545.68 / 439.86</td>
<td>109</td>
<td>PP-OCRv4's server-side seal text detection model with higher accuracy, suitable for deployment on better servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">Inference model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Training model</a></td>
<td>96.47</td>
<td>9.70 / 3.56</td>
<td>50.38 / 19.64</td>
<td>4.6</td>
<td>PP-OCRv4's mobile-side seal text detection model with higher efficiency, suitable for deployment on the end side</td>
</tr>
</tbody>
</table>
<strong>Test environment description:</strong>
<ul>
<li><b>Performance test environment</b>
<ul>
<li><strong>Test dataset:</strong>
<ul>
<li>Document image orientation classification model: A self-built dataset by PaddleX, covering multiple scenarios such as certificates and documents, including 1000 images.</li>
<li>Text image rectification model: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a>.</li>
<li>Layout region detection model: A self-built layout region analysis dataset by PaddleOCR, including 10,000 common document-type images such as Chinese and English papers, magazines, and research reports.</li>
<li>PP-DocLayout_plus-L: A self-built layout region detection dataset by PaddleOCR, including 1300 document-type images such as Chinese and English papers, magazines, newspapers, research reports, PPTs, exam papers, and textbooks.</li>
<li>Table structure recognition model: An internally self-built English table recognition dataset by PaddleX.</li>
<li>Text detection model: A self-built Chinese dataset by PaddleOCR, covering multiple scenarios such as street views, web images, documents, and handwriting, including 500 images for detection.</li>
<li>Chinese recognition model: A self-built Chinese dataset by PaddleOCR, covering multiple scenarios such as street views, web images, documents, and handwriting, including 11,000 images for text recognition.</li>
<li>ch_SVTRv2_rec: <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task</a>Phase A evaluation set.</li>
<li>ch_RepSVTR_rec: <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task</a>Phase B evaluation set.</li>
<li>English recognition model: A self-built English dataset by PaddleX.</li>
<li>Multilingual recognition model: A self-built multilingual dataset by PaddleX.</li>
<li>Text line orientation classification model: A self-built dataset by PaddleX, covering multiple scenarios such as certificates and documents, including 1000 images.</li>
<li>Seal text detection model: A self-built dataset by PaddleX, including 500 round seal images.</li>
</ul>
</li>
<li><strong>Hardware configuration: </strong>
<ul>
<li>GPU: NVIDIA Tesla T4</li>
<li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
<li>Other environments: Ubuntu 20.04 / CUDA 11.8 / cuDNN 8.9 / TensorRT 8.6.1.6</li>
</ul>
</li>
</ul>
</li>
<li><b>Description of inference modes</b></li>
</ul>
<table border="1">
<thead>
<tr>
<th>Mode</th>
<th>GPU configuration</th>
<th>CPU configuration</th>
<th>Combination of acceleration technologies</th>
</tr>
</thead>
<tbody>
<tr>
<td>Regular mode</td>
<td>FP32 precision / No TRT acceleration</td>
<td>FP32 precision / 8 threads</td>
<td>PaddleInference</td>
</tr>
<tr>
<td>High-performance mode</td>
<td>Select the optimal combination of prior precision type and acceleration strategy</td>
<td>FP32 precision / 8 threads</td>
<td>Select the prior optimal backend (Paddle/OpenVINO/TRT, etc.)</td>
</tr>
</tbody>
</table>
</details>

## 2. Quick Start
### 2.1 Local Experience
Before using the PP-DocTranslation pipeline locally, ensure that you have completed the installation of PaddleX (refer to the [Installation Guide](../../../installation/installation.en.md)). This pipeline depends on the group named `translation`.

Before use, you need to prepare the API key for a large language model, which supports the [Baidu Cloud Qianfan Platform](https://console.bce.baidu.com/qianfan/ais/console/onlineService) or local large model services that comply with the OpenAI interface standards.

```python
from paddlex import create_pipeline
# Create a translation pipeline
pipeline = create_pipeline(pipeline="PP-DocTranslation")

# Document path
input_path = "document_sample.pdf"

# Output directory
output_path = "./output"

# Large model configuration
chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

if input_path.lower().endswith(".md"):
    # Read markdown documents, supporting passing in directories and url links with the .md suffix
    ori_md_info_list = pipeline.load_from_markdown(input_path)
else:
    # Use PP-StructureV3 to perform layout analysis on PDF/image documents to obtain markdown information
    visual_predict_res = pipeline.visual_predict(
        input_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_common_ocr=True,
        use_seal_recognition=True,
        use_table_recognition=True,
)

    ori_md_info_list = []
    for res in visual_predict_res:
        layout_parsing_result = res["layout_parsing_result"]
        ori_md_info_list.append(layout_parsing_result.markdown)
        layout_parsing_result.save_to_img(output_path)
        layout_parsing_result.save_to_markdown(output_path)

    # Concatenate the markdown information of multi-page documents into a single markdown file, and save the merged original markdown
    if input_path.lower().endswith(".pdf"):
        ori_md_info = pipeline.concatenate_markdown_pages(ori_md_info_list)
        ori_md_info.save_to_markdown(output_path)

# Perform document translation (target language: English)
tgt_md_info_list = pipeline.translate(
    ori_md_info_list=ori_md_info_list,
    target_language="en",
    chunk_size=5000,
    chat_bot_config=chat_bot_config,
)
# Save the translation results
for tgt_md_info in tgt_md_info_list:
    tgt_md_info.save_to_markdown(output_path)
```

After executing the above code, you will obtain the parsed results of the original document to be translated, the Markdown file of the original document to be translated, and the Markdown file of the translated document, all saved in the `output` folder.

# PP-DocTranslation Prediction Process, API Description, and Output Description

<details><summary>(1) Instantiation <code>create_pipeline</code> Instantiate a pipeline object, with specific parameter descriptions as follows:</summary>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>Pipeline name or configuration file path (set to "PP-DocTranslation")</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>Inference device (such as "gpu:0", "npu:0", "cpu", etc.)</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference plugin</td>
<td><code>bool</code>|<code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>High-performance inference configuration</td>
<td><code>dict</code>|<code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>initial_predictor</code></td>
<td>Whether to initialize the inference module</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
</tbody>
</table>
</details>
<details><summary>(2) Call the <code>visual_predict()</code> method of the PP-DocTranslation pipeline object to obtain visual prediction results. This method will return a <code>generator</code>. Below are the parameters and descriptions of the <code>visual_predict()</code>method:</summary>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types, required</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: such as <code>numpy.ndarray</code>representing image data</li>
<li><b>str</b>: such as the local path of an image file or PDF file: <code>/root/data/img.jpg</code>;<b>such as a URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">example</a>;<b>such as a local directory</b>, which should contain the images to be predicted, such as the local path: <code>/root/data/</code>(Currently, prediction for PDF files contained in directories is not supported. For PDF files, a specific file path needs to be specified.)</li>
<li><b>List</b>: The list elements should be of the above-mentioned data types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
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
<li><b>bool</b>:<code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the parameter value initialized by the pipeline will be used by default, which is initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the document unwarping module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>:<code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized to<code>True</code>;</li>
</ul>
</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to use the text line orientation classification module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>:<code>True</code>or<code>False</code>;</li>
<li><b>None</b>: If set to<code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized to<code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_general_ocr</code></td>
<td>Whether to use the OCR sub-pipeline</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>:<code>True</code>or<code>False</code>;</li>
<li><b>None</b>: If set to<code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized to<code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>Whether to use the seal recognition sub-pipeline</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>:<code>True</code>or<code>False</code>;</li>
<li><b>None</b>: If set to<code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized to<code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>Whether to use the table recognition sub-pipeline</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>:<code>True</code>or<code>False</code>;</li>
<li><b>None</b>: If set to<code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized to<code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_formula_recognition</code></td>
<td>Whether to use the sub-pipeline for formula recognition</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>:<code>True</code>or<code>False</code>;</li>
<li><b>None</b>: If set to<code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized to<code>True</code>;</li>
</ul>
</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>Whether to use the sub-pipeline for chart recognition</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>:<code>True</code>or<code>False</code>;</li>
<li><b>None</b>: If set to<code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized to<code>True</code>;</li>
</ul>
</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_region_detection</code></td>
<td>Whether to use the pipeline for document region detection</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>:<code>True</code>or<code>False</code>;</li>
<li><b>None</b>: If set to<code>None</code>The parameter value initialized by the pipeline will be used by default, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Score threshold for layout model</td>
<td><code>float|dict|None</code></td>
<td>
<ul>
<li><b>float</b>:<code>Any floating-point number between </code>0-1</li>
<li><b>;</b>dict<code>:</code>{0:0.1}</li>
<li><b>The key is the class ID, and the value is the threshold for that class;</b>None<code>: If set to </code>None<code>, the parameter value initialized by the pipeline will be used by default, initialized to </code>0.5</li>
</ul>
</td>
<td><code>;</code></td>
</tr>
<tr>
<td><code>None</code></td>
<td>layout_nms</td>
<td><code>Whether the layout region detection model uses NMS post-processing</code></td>
<td>
<ul>
<li><b>bool|None</b>bool<code>:</code>True<code> or </code>False</li>
<li><b>;</b>None<code>: If set to </code>None<code>, the parameter value initialized by the pipeline will be used by default, initialized to </code>True</li>
</ul>
</td>
<td><code>;</code></td>
</tr>
<tr>
<td><code>None</code></td>
<td>layout_unclip_ratio</td>
<td><code>Expansion coefficient of the detection box for the layout region detection model</code></td>
<td>
<ul>
<li><b>float|Tuple[float,float]|dict|None</b>float<code>: Any floating-point number greater than </code>0</li>
<li><b>;</b>Tuple[float,float]</li>
<li><b>: The respective expansion coefficients in the horizontal and vertical directions;</b>Dictionary, <b>the key of the dictionary is of </b>int<code> type, representing </code>, the value is of <b>tuple</b> type, such as <code>{0: (1.1, 2.0)}</code>, which means that the center of the detection box for the 0th category output by the model remains unchanged, the width is expanded by 1.1 times, and the height is expanded by 2.0 times.</li>
<li><b>None</b>: If set to <code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized to <code>1.0</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>The filtering method for overlapping boxes in layout area detection</td>
<td><code>str|dict|None</code></td>
<td>
<ul>
<li><b>str</b>:<code>large</code>,<code>small</code>,<code>union</code>, which respectively indicate that when filtering overlapping boxes, the large box, the small box, or both are retained.</li>
<li><b>dict</b>, the key of the dictionary is of <b>int</b> type, representing <code>cls_id</code>, and the value is of <b>str</b> type, such as <code>{0: "large", 2: "small"}</code>, which means that the large mode is used for the detection box of the 0th category and the small mode is used for the detection box of the 2nd category.</li>
<li><b>None</b>: If set to <code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized to <code>large</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>The image side length limit for text detection</td>
<td><code>int|None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>The parameter value initialized by the pipeline will be used by default, initialized to <code>960</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>Image side length limit type for text detection</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: supports <code>min</code> and <code>max</code>, <code>min</code> means ensuring that the shortest side of the image is not less than <code>det_limit_side_len</code>, <code>max</code> means ensuring that the longest side of the image is not greater than <code>limit_side_len</code></li>
<li><b>None</b>: if set to <code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized to <code>max</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Detection pixel threshold. Only pixels with scores greater than this threshold in the output probability map will be considered as text pixels</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: any floating-point number greater than <code>0</code>None<li><b>: if set to </b>None<code>, the parameter value initialized by the pipeline will be used by default</code>0.3<code>None</code></li></li></ul></td>
<td><code>text_det_box_thresh</code></td>
</tr>
<tr>
<td><code>Detection box threshold. When the average score of all pixels within the detection result border is greater than this threshold, the result will be considered as a text area</code></td>
<td>float|None</td>
<td><code>float</code></td>
<td>
<ul>
<li><b>: any floating-point number greater than </b>0<code>None</code>: if set to <li><b>None</b>, the parameter value initialized by the pipeline will be used by default<code>None</code>, 将默认使用产线初始化的该参数值 <code>0.6</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Text detection expansion coefficient. This method is used to expand the text area. The larger the value, the larger the expanded area.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: any floating-point number <code>greater than </code>0<li><b>None</b>: if set to <code>None</code>, the parameter value initialized by the pipeline will be used by default<code>2.0</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Text recognition threshold. Text results with scores greater than this threshold will be retained.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: any floating-point number <code>greater than </code>0<li><b>None</b>: if set to <code>None</code>, the parameter value initialized by the pipeline will be used by default<code>0.0</code>. That is, no threshold is set</li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Image side length limit for seal detection</td>
<td><code>int|None</code></td>
<td>
<ul>
<li><b>int</b>: any integer <code>greater than </code>0</li>
<li><b>; </b>None<code>: if set to </code>None<code>, the parameter value initialized by the pipeline will be used by default, initialized to </code>960</li>
</ul>
</td>
<td><code>; </code></td>
</tr>
<tr>
<td><code>None</code></td>
<td>seal_det_limit_type</td>
<td><code>Image side length limit type for seal detection</code></td>
<td>
<ul>
<li><b>str|None</b>str<code>: supports </code>min<code> and </code>max<code>, </code>Indicates that the shortest side of the image is guaranteed to be no less than<code>det_limit_side_len</code>,<code>max</code>Indicates that the longest side of the image is guaranteed to be no greater than<code>limit_side_len</code></li>
<li><b>None</b>: If set to<code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized to<code>max</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Detection pixel threshold. Only pixels with scores greater than this threshold in the output probability map are considered as seal pixels.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than<code>0</code>None<li><b>: If set to</b>None<code>, the parameter value initialized by the pipeline will be used by default.</code>0.3<code>None</code></li></li></ul></td>
<td><code>seal_det_box_thresh</code></td>
</tr>
<tr>
<td><code>Detection box threshold. When the average score of all pixels within the bounding box of the detection result is greater than this threshold, the result is considered as a seal area.</code></td>
<td>float|None</td>
<td><code>float</code></td>
<td>
<ul>
<li><b>: Any floating-point number greater than</b>0<code>None</code>: If set to<li><b>None</b>, the parameter value initialized by the pipeline will be used by default.<code>0.6</code>None<code>seal_det_unclip_ratio</code></li></li></ul></td>
<td><code>Seal detection expansion coefficient. This method is used to expand the text area. The larger the value, the larger the expanded area.</code></td>
</tr>
<tr>
<td><code>float|None</code></td>
<td>float</td>
<td><code>: Any floating-point number greater than</code></td>
<td>
<ul>
<li><b>0</b>None<code>: If set to</code>None<li><b>None</b>：如果设置为 <code>None</code>The parameter value initialized by the pipeline will be used by default.<code>2.0</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Seal recognition threshold. Text results with scores greater than this threshold will be retained.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>:<code>Any</code>floating-point number greater than<li><b>0</b>None<code>: If set to</code>None<code>, the parameter value initialized by the pipeline will be used by default.</code>0.0</li></li></ul></td>
<td><code>. That is, no threshold is set.</code></td>
</tr>
<tr>
<td><code>None</code></td>
<td>use_wired_table_cells_trans_to_html</td>
<td><code>Whether to enable direct conversion of wired table cell detection results to HTML. The default is False. If enabled, HTML will be constructed directly based on the geometric relationships of wired table cell detection results.</code></td>
<td>
<ul>
<li><b>float|None</b>bool<code>:</code>True<code>or</code>False</li>
<li><b>;</b>None<code>: If set to</code>None<code>, the parameter value initialized by the pipeline will be used by default, initialized to</code>False</li>
</ul>
<td><code>;</code></td>
</td>
</tr>
<tr>
<td><code>False</code></td>
<td>use_wired_table_cells_trans_to_html (Note: This line seems to be a duplicate in the original text, but translated as per the context it would be the same as above. However, if it's intended to be a different parameter, it might need clarification. Here, I've translated it as if it's a repetition for the sake of completeness.)</td>
<td><code>Whether to enable direct conversion of detection results (without wired table cells) to HTML. The default is False. If enabled, HTML will be constructed directly based on the geometric relationships of detection results (without wired table cells).</code></td>
<td>
<ul>
<li><b>float|None</b>bool<code>:</code>True<code>or</code>False</li>
<li><b>;</b>None<code>: If set to</code>None<code>False</code>;</li>
</ul>
<td><code>False</code></td>
</td>
</tr>
<tr>
<td><code>use_table_orientation_classify</code></td>
<td>Whether to enable table orientation classification. When enabled, if the table in the image is rotated by 90/180/270 degrees, the orientation can be corrected and table recognition can be completed correctly.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>:<code>True</code>or<code>False</code>;</li>
<li><b>None</b>: If set to<code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized to<code>True</code>;</li>
</ul>
</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>use_ocr_results_with_table_cells</code></td>
<td>Whether to enable cell segmentation OCR. When enabled, OCR detection results will be segmented and re-recognized based on cell prediction results to avoid missing text.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>:<code>True</code>or<code>False</code>;</li>
<li><b>None</b>: If set to<code>None</code>, the parameter value initialized by the pipeline will be used by default, initialized to<code>True</code>;</li>
</ul>
</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>use_e2e_wired_table_rec_model</code></td>
<td>Whether to enable the end-to-end wired table recognition mode. When enabled, the cell detection model will not be used, and only the table structure recognition model will be used.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>:<code>True</code>or<code>False</code>;</li>
<li><b>None</b>: If set to<code>None</code>The parameter value will be initialized by default using the pipeline initialization, initialized to <code>False</code>;</li>
</ul>
</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_e2e_wireless_table_rec_model</code></td>
<td>Whether to enable the wireless table end-to-end table recognition mode. If enabled, the cell detection model will not be used, and only the table structure recognition model will be used.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>:<code>True</code>or<code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the parameter value will be initialized by default using the pipeline initialization, initialized to <code>False</code>;</li>
</ul>
</td>
<td><code>True</code></td>
</tr>
</table>
</details>
<details><summary>(3) Processing visual prediction results: The prediction result for each sample is a corresponding Result object, and it supports operations such as printing, saving as an image, and saving as a <code>json</code> file:</summary>
<table>
<thead>
<tr>
<th>Method</th>
<th>Method Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to use JSON indentation formatting for the output content<code>JSON</code>indentation formatting</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable. This is only valid when <code>format_json</code> is <code>True</code>4</td>
<td>ensure_ascii</td>
</tr>
<tr>
<td><code>bool</code></td>
<td><code>Controls whether non-</code></td>
<td>ASCII<code> characters are escaped to </code> 字符转义为 <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped;<code>False</code> will retain the original characters, valid only when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Saves the result as a file in JSON format.</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file type name.</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output <code>JSON</code> data and make it more readable, valid only when <code>format_json</code> is <code>True</code>.</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped;<code>False</code> will retain the original characters, valid only when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Saves the visualization images of each intermediate module in PNG format.</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. Supports directory or file path.</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_markdown()</code></td>
<td>Save each page in an image or PDF file as a file in markdown format separately</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The saved file path, which supports directory or file path</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Save the tables in the file as a file in html format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The saved file path, which supports directory or file path</td>
<td>None</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Save the tables in the file as a file in xlsx format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The saved file path, which supports directory or file path</td>
<td>None</td>
</tr>
</table>- Calling the `print()` method will print the results to the terminal. The content printed to the terminal is explained as follows:
    - `input_path`: `(str)` The input path of the image or PDF to be predicted

    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates which page of the PDF it is; otherwise, it is `None`

    - `model_settings`: `(Dict[str, bool])` Configure the model parameters required for the pipeline

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing sub-pipeline
        - `use_general_ocr`: `(bool)` Controls whether to enable the OCR sub-pipeline
        - `use_seal_recognition`: `(bool)` Controls whether to enable the seal recognition sub-pipeline
        - `use_table_recognition`: `(bool)` Controls whether to enable the table recognition sub-pipeline
        - `use_formula_recognition`: `(bool)` Controls whether to enable the formula recognition sub-pipeline

    - `doc_preprocessor_res`: `(Dict[str, Union[List[float], str]])` A dictionary of document preprocessing results, which only exists when `use_doc_preprocessor=True`
        - `input_path`: `(str)` The image path accepted by the document preprocessing sub-pipeline. When the input is `numpy.ndarray`, it is saved as `None`, and it is `None` here
        - `page_index`: `None`, as the input here is `numpy.ndarray`, so the value is `None`
        - `model_settings`: `(Dict[str, bool])` The model configuration parameters for the document preprocessing sub-pipeline
- `use_doc_orientation_classify`: `(bool)` Controls whether to enable the submodule for document image orientation classification.
          - `use_doc_unwarping`: `(bool)` Controls whether to enable the submodule for text image distortion correction.
        - `angle`: `(int)` The prediction result of the submodule for document image orientation classification. Returns the actual angle value when enabled.

    - `parsing_res_list`: `(List[Dict])` A list of parsing results, where each element is a dictionary. The list is ordered according to the reading sequence after parsing.
        - `block_bbox`: `(np.ndarray)` The bounding box of the layout area.
        - `block_label`: `(str)` The label of the layout area, such as `text`, `table`, etc.
        - `block_content`: `(str)` The content within the layout area.
        - `seg_start_flag`: `(bool)` Indicates whether this layout area is the start of a paragraph.
        - `seg_end_flag`: `(bool)` Indicates whether this layout area is the end of a paragraph.
        - `sub_label`: `(str)` The sub-label of the layout area. For example, the sub-label of `text` might be `title_text`.
        - `sub_index`: `(int)` The sub-index of the layout area, used for restoring Markdown.
        - `index`: `(int)` The index of the layout area, used for displaying the layout sorting results.

    - `overall_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` A dictionary of global OCR results.
      - `input_path`: `(Union[str, None])` The image path accepted by the image OCR sub-pipeline. When the input is `numpy.ndarray`, it is saved as `None`.
- `page_index`: `None`, the input here is `numpy.ndarray`, so the value is `None`.
      - `model_settings`: `(Dict)` Model configuration parameters for the OCR sub-pipeline.
      - `dt_polys`: `(List[numpy.ndarray])` List of polygon bounding boxes for text detection. Each bounding box is represented by a numpy array of 4 vertex coordinates, with an array shape of (4, 2) and a data type of int16.
      - `dt_scores`: `(List[float])` List of confidence scores for text detection bounding boxes.
      - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the text detection module.
        - `limit_side_len`: `(int)` Side length limit during image preprocessing.
        - `limit_type`: `(str)` Method for handling side length limits.
        - `thresh`: `(float)` Confidence threshold for text pixel classification.
        - `box_thresh`: `(float)` Confidence threshold for text detection bounding boxes.
        - `unclip_ratio`: `(float)` Expansion coefficient for text detection bounding boxes.
        - `text_type`: `(str)` Type of text detection, currently fixed as "general".

      - `text_type`: `(str)` Type of text detection, currently fixed as "general".
      - `textline_orientation_angles`: `(List[int])` Prediction results for text line orientation classification. Returns actual angle values when enabled (e.g., [0,0,1]).
      - `text_rec_score_thresh`: `(float)` Filtering threshold for text recognition results.
      - `rec_texts`: `(List[str])` List of text recognition results, containing only texts with confidence scores exceeding `text_rec_score_thresh`.
- `rec_scores`: `(List[float])` List of confidence scores for text recognition, filtered by `text_rec_score_thresh`
      - `rec_polys`: `(List[numpy.ndarray])` List of text detection bounding boxes after confidence filtering, with the same format as `dt_polys`

    - `formula_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` List of formula recognition results, with each element being a dictionary
        - `rec_formula`: `(str)` Formula recognition result
        - `rec_polys`: `(numpy.ndarray)` Formula detection bounding box with a shape of (4, 2) and dtype of int16
        - `formula_region_id`: `(int)` Region number where the formula is located

    - `seal_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` List of seal recognition results, with each element being a dictionary
        - `input_path`: `(str)` Input path of the seal image
        - `page_index`: `None`, since the input here is `numpy.ndarray`, the value is `None`
        - `model_settings`: `(Dict)` Model configuration parameters for the seal recognition sub-pipeline
        - `dt_polys`: `(List[numpy.ndarray])` List of seal detection bounding boxes, with the same format as `dt_polys`
        - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the seal detection module, with the same meanings of specific parameters as above
        - `text_type`: `(str)` Type of seal detection, currently fixed as "seal"
- `text_rec_score_thresh`: `(float)` The filtering threshold for seal recognition results
        - `rec_texts`: `(List[str])` A list of seal recognition results, containing only texts with confidence scores exceeding `text_rec_score_thresh`
        - `rec_scores`: `(List[float])` A list of confidence scores for seal recognition, filtered by `text_rec_score_thresh`
        - `rec_polys`: `(List[numpy.ndarray])` A list of seal detection boxes filtered by confidence scores, with the same format as `dt_polys`
        - `rec_boxes`: `(numpy.ndarray)` An array of rectangular bounding boxes for detection boxes, with a shape of (n, 4) and dtype of int16. Each row represents a rectangle

    - `table_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` A list of table recognition results, with each element being a dictionary
        - `cell_box_list`: `(List[numpy.ndarray])` A list of bounding boxes for table cells
        - `pred_html`: `(str)` An HTML-formatted string for the table
        - `table_ocr_pred`: `(dict)` OCR recognition results for the table
            - `rec_polys`: `(List[numpy.ndarray])` A list of detection boxes for cells
            - `rec_texts`: `(List[str])` Recognition results for cells
            - `rec_scores`: `(List[float])` Recognition confidence scores for cells
- `rec_boxes`: `(numpy.ndarray)` An array of rectangular bounding boxes for detection, with a shape of (n, 4) and dtype of int16. Each row represents a rectangle.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, the `numpy.array` type will be converted to a list format.
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, it will save the layout region detection visualization image, global OCR visualization image, layout reading order visualization image, etc. If a file is specified, it will be saved directly to that file. (The pipeline usually contains many result images, so it is not recommended to directly specify a specific file path; otherwise, multiple images will be overwritten, and only the last image will be retained.)
- Calling the `save_to_markdown()` method will save the converted Markdown file to the specified `save_path`. The saved file path will be `save_path/{your_img_basename}.md`. If the input is a PDF file, it is recommended to directly specify a directory; otherwise, multiple markdown files will be overwritten.
- Calling the `concatenate_markdown_pages()` method will merge the multi-page Markdown content `markdown_list` output by the PP-DocTranslation pipeline into a single complete document and return the merged Markdown content.</details>
<details><summary>(4) Call <code>translate()</code> method to perform document translation. This method will return the original markdown text and the translated text, which is a markdown object. You can save the required parts locally by executing the <code>save_to_markdown()</code> method. The following are the descriptions of the relevant parameters of the <code>translate()</code> method:</summary>
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
<tbody>
<tr>
<td><code>ori_md_info_list</code></td>
<td>A list of data in the original markdown format, containing the content to be translated</td>
<td><code>List[Dict]</code></td>
<td>It must be a list composed of dictionaries, with each dictionary representing a document block</td>
<td>No default value (required)</td>
</tr>
<tr>
<td><code>target_language</code></td>
<td>Target translation language code</td>
<td><code>str</code></td>
<td>ISO 639-1 language code (such as "en"/"ja"/"fr")</td>
<td><code>"zh"</code></td>
</tr>
<tr>
<td><code>chunk_size</code></td>
<td>The character threshold for chunking the text to be translated</td>
<td><code>int</code></td>
<td>An integer greater than 0</td>
<td><code>5000</code></td>
</tr>
<tr>
<td><code>task_description</code></td>
<td>Custom task description prompt</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: Custom text for translation task instructions</li>
<li><b>None</b>: Use default task description</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>output_format</code></td>
<td>Specify output format requirements</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: Format specifications (e.g., "Maintain original Markdown structure")</li>
<li><b>None</b>: Do not add additional format constraints</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>rules_str</code></td>
<td>Custom translation rule description</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: Terminology/style rule text</li>
<li><b>None</b>: Do not use additional rules</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>few_shot_demo_text_content</code></td>
<td>Example text content for few-shot learning</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: Example text string</li>
<li><b>None</b>: Do not provide text examples</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>few_shot_demo_key_value_list</code></td>
<td>Structured few-shot example data</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: Example data in key-value pair format</li>
<li><b>None</b>: Do not provide structured examples</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>glossary</code></td>
<td>Glossary of technical terms</td>
<td><code>dict|None</code></td>
<td>
<ul>
<li><b>dict</b>: Dictionary for glossary mapping</li>
<li><b>None</b>: Use default configuration</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>llm_request_interval</code></td>
<td>Time interval in seconds for sending requests to the large language model. This parameter can be used to prevent overly frequent calls to the large language model.</td>
<td><code>float</code></td>
<td>A floating-point number greater than or equal to 0</td>
<td><code>0.0</code></td>
</tr>
<tr>
<td><code>chat_bot_config</code></td>
<td>Large language model configuration</td>
<td><code>dict|None</code></td>
<td>
<ul>
<li><b>dict</b>: Dictionary of model parameter configurations</li>
<li><b>None</b>: Use default configuration</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>
</details>

In addition, you can obtain the configuration file for the layout analysis pipeline and load the configuration file for prediction. You can execute the following command to save the results in `my_path`:

```
paddlex --get_pipeline_config PP-DocTranslation --save_path ./my_path
```

If you obtain the configuration file, you can customize the configurations of the layout parsing pipeline by simply modifying the value of the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file. Here is an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/PP-DocTranslation.yaml")

# Document path
img_path = "document_sample.pdf"

# Large model configuration
chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # Replace with the actual API key
}

# Perform layout analysis
visual_predict_res = pipeline.visual_predict(
    img_path,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_common_ocr=True,
    use_seal_recognition=True,
    use_table_recognition=True,
)

# Extract the original document structure information
ori_md_info_list = []
for res in visual_predict_res:
    layout_parsing_result = res["layout_parsing_result"]
    ori_md_info_list.append(layout_parsing_result.markdown)
    layout_parsing_result.print()
    layout_parsing_result.save_to_img("./output")
    layout_parsing_result.save_to_json("./output")

# Document translation
tgt_md_info_list = pipeline.translate(
    ori_md_info_list=ori_md_info_list,
    target_language="en",
    chunk_size=5000,
    chat_bot_config=chat_bot_config,
)

# Save the translation results
for tgt_md_info in tgt_md_info_list:
    tgt_md_info.save_to_markdown(output_path)
```

<b>Note:</b>The parameters in the configuration file are pipeline initialization parameters. If you want to change the initialization parameters of the general layout parsing v3 pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Meanwhile, CLI prediction also supports passing in the configuration file, and you can specify the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment
If the pipeline can meet your requirements for inference speed and accuracy, you can directly proceed with development integration/deployment.

If you need to directly apply the pipeline to your Python project, you can refer to the sample code in [2.2 Integration via Python Script](#22-python脚本方式集成).

In addition, PaddleX also provides three other deployment methods, which are described in detail below:

🚀<b>High-performance inference</b>: In real production environments, many applications have stringent performance metrics (especially response speed) for deployment strategies to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin designed to deeply optimize model inference and pre/post-processing, achieving significant acceleration of the end-to-end process. For detailed information on the high-performance inference process, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️<b>Serving</b>: Serving is a common form of deployment in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports multiple pipeline Serving deployment solutions. For detailed pipeline Serving deployment procedures, please refer to the [PaddleX Serving Deployment Guide](../../../pipeline_deploy/serving.en.md).

The following is the API reference for basic Serving and examples of multilingual service invocation:

<details><summary>API reference</summary>
<p>Main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is<code>200</code>, and the attributes of the response body are as follows:</li>
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
<td>Error code. Fixed as<code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed as<code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
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
<td>Error description.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>analyzeImages</code></b></li>
</ul>
<p>Analyze images using computer vision models to obtain OCR, table recognition results, etc.</p>
<p><code>POST /doctrans-visual</code></p>
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
<td>The URL of an image file or PDF file accessible by the server, or the Base64-encoded result of the content of the aforementioned file types. By default, for PDF files with more than 10 pages, only the first 10 pages will be processed.<br/>To remove the page limit, add the following configuration to the pipeline configuration file:<pre><code>Serving:
  extra:
    max_num_input_imgs: null</code></pre>
</td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code>｜<code>null</code></td>
<td>File type.<code>0</code>Indicates a PDF file,<code>1</code>Indicates an image file. If this attribute is not present in the request body, the file type will be inferred from the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>Refer to the description of the <code>use_doc_orientation_classify</code>parameter in the <code>visual_predict</code>method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>Refer to the description of the <code>use_doc_unwarping</code>parameter in the <code>visual_predict</code>method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTextlineOrientation</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>Refer to the description of the <code>use_textline_orientation</code>parameter in the <code>visual_predict</code>method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useSealRecognition</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>See the description of the <code>use_seal_recognition</code> parameter for the <code>visual_predict</code> method in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableRecognition</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>See the description of the <code>use_table_recognition</code> parameter for the <code>visual_predict</code> method in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useFormulaRecognition</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>See the description of the <code>use_formula_recognition</code> parameter for the <code>visual_predict</code> method in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useChartRecognition</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>See the description of the <code>use_chart_recognition</code> parameter for the <code>visual_predict</code> method in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useRegionDetection</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>See the description of the <code>use_region_detection</code> parameter for the <code>visual_predict</code> method in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code>|<code>object</code>|<code>null</code></td>
<td>Refer to the description of the <code>visual_predict</code> method's <code>layout_threshold</code> parameter in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>Refer to the description of the <code>visual_predict</code> method's <code>layout_nms</code> parameter in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code>|<code>array</code>|<code>object</code>|<code>null</code></td>
<td>Refer to the description of the <code>visual_predict</code> method's <code>layout_unclip_ratio</code> parameter in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code>|<code>object</code>|<code>null</code></td>
<td>Refer to the description of the <code>visual_predict</code> method's <code>layout_merge_bboxes_mode</code> parameter in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code>|<code>null</code></td>
<td>Refer to the description of the <code>visual_predict</code> method's <code>text_det_limit_side_len</code> parameter in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code>|<code>null</code></td>
<td>Refer to the description of the <code>text_det_limit_type</code> parameter of the <code>visual_predict</code> method in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Refer to the description of the <code>text_det_thresh</code> parameter of the <code>visual_predict</code> method in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Refer to the description of the <code>text_det_box_thresh</code> parameter of the <code>visual_predict</code> method in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Refer to the description of the <code>text_det_unclip_ratio</code> parameter of the <code>visual_predict</code> method in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Refer to the description of the <code>text_rec_score_thresh</code> parameter of the <code>visual_predict</code> method in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitSideLen</code></td>
<td><code>integer</code>|<code>null</code></td>
<td>Refer to the description in the pipeline object<code>visual_predict</code>method<code>seal_det_limit_side_len</code>parameter description.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitType</code></td>
<td><code>string</code>|<code>null</code></td>
<td>Refer to the parameter description of <code>visual_predict</code>method's<code>seal_det_limit_type</code> in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetThresh</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Refer to the parameter description of <code>visual_predict</code>method's<code>seal_det_thresh</code> in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetBoxThresh</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Refer to the parameter description of <code>visual_predict</code>method's<code>seal_det_box_thresh</code> in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetUnclipRatio</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Refer to the parameter description of <code>visual_predict</code>method's<code>seal_det_unclip_ratio</code> in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealRecScoreThresh</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Refer to the parameter description of <code>visual_predict</code>method's<code>seal_rec_score_thresh</code>Parameter description.</td>
<td>No</td>
</tr>
<tr>
<td><code>useWiredTableCellsTransToHtml</code></td>
<td><code>boolean</code></td>
<td>Refer to the parameter description of <code>use_wired_table_cells_trans_to_html</code> in the <code>visual_predict</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useWirelessTableCellsTransToHtml</code></td>
<td><code>boolean</code></td>
<td>Refer to the parameter description of <code>use_wireless_table_cells_trans_to_html</code> in the <code>visual_predict</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableOrientationClassify</code></td>
<td><code>boolean</code></td>
<td>Refer to the parameter description of <code>use_table_orientation_classify</code> in the <code>visual_predict</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useOcrResultsWithTableCells</code></td>
<td><code>boolean</code></td>
<td>Refer to the parameter description of <code>use_ocr_results_with_table_cells</code> in the <code>visual_predict</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useE2eWiredTableRecModel</code></td>
<td><code>boolean</code></td>
<td>Refer to the parameter description of <code>use_e2e_wired_table_rec_model</code> in the <code>visual_predict</code> method of the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useE2eWirelessTableRecModel</code></td>
<td><code>boolean</code></td>
<td>Refer to the <code>visual_predict</code>method's <code>use_e2e_wireless_table_rec_model</code>parameter description in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>visualize</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>Whether to return visualization result graphs and intermediate images during processing, etc.<ul style="margin: 0 0 0 1em; padding-left: 0em;">
<li>Pass in <code>true</code>: Return images.</li>
<li>Pass in <code>false</code>: Do not return images.</li>
<li>If this parameter is not provided in the request body or <code>null</code>is passed in: Follow the setting in the <code>Serving.visualize</code>of the pipeline configuration file.</li>
</ul>
<br/>For example, add the following field in the pipeline configuration file:<br/>
<pre><code>Serving:
  visualize: False</code></pre>Images will not be returned by default, and the default behavior can be overridden by the <code>visualize</code>parameter in the request body. If neither the request body nor the configuration file is set (or <code>null</code>is passed in the request body and nothing is set in the configuration file), images will be returned by default.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code>in the response body has the following attributes:</li>
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
<td>Layout parsing results. The array length is 1 (for image input) or the actual number of document pages processed (for PDF input). For PDF input, each element in the array represents the result of each page actually processed in the PDF file in sequence.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Enter the data information.</td>
</tr>
</tbody>
</table>
<p><code>Each element in </code>layoutParsingResults<code> is an </code>object</p>
<table>
<thead>
<tr>
<th> with the following attributes:</th>
<th>Name</th>
<th>Type</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>Meaning</code></td>
<td><code>prunedResult</code></td>
<td>object<code>A simplified version of the </code>res<code> field in the JSON representation of the </code>layout_parsing_result<code> generated by the </code>visual_predict<code> method of the pipeline object, with the </code>input_path<code> and </code>page_index</td>
</tr>
<tr>
<td><code> fields removed.</code></td>
<td><code>markdown</code></td>
<td>object</td>
</tr>
<tr>
<td><code>Markdown result.</code></td>
<td><code>outputImages</code>object<code>|</code></td>
<td>null<code>Refer to the description of the </code>img</td>
</tr>
<tr>
<td><code> attribute in the pipeline prediction results. The image is in JPEG format and encoded with Base64.</code></td>
<td><code>inputImage</code>string<code>|</code></td>
<td>null</td>
</tr>
</tbody>
</table>
<p><code>Input image. The image is in JPEG format and encoded with Base64.</code>markdown<code> is an </code>object</p>
<table>
<thead>
<tr>
<th> with the following attributes:</th>
<th>Name</th>
<th>Type</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>Meaning</code></td>
<td><code>text</code></td>
<td>string</td>
</tr>
<tr>
<td><code>Markdown text.</code></td>
<td><code>images</code></td>
<td>object</td>
</tr>
<tr>
<td><code>A key-value pair of relative paths of Markdown images and Base64-encoded images.</code></td>
<td><code>isStart</code></td>
<td>Whether the first element of the current page is the start of a paragraph.</td>
</tr>
<tr>
<td><code>isEnd</code></td>
<td><code>boolean</code></td>
<td>Whether the last element of the current page is the end of a paragraph.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>translate</code></b></li>
</ul>
<p>Translate documents using large models.</p>
<p><code>POST /doctrans-translate</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>name</th>
<th>type</th>
<th>meaning</th>
<th>required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>markdownList</code></td>
<td><code>array</code></td>
<td>A list of Markdown to be translated. It can be obtained from the result of the <code>analyzeImages</code>operation.<code>The </code>images</td>
<td>attribute will not be used.</td>
</tr>
<tr>
<td><code>Yes</code></td>
<td><code>targetLanguage</code></td>
<td>string<code>Refer to the description of the </code>target_language<code>parameter in the </code>translate</td>
<td>method of the pipeline object.</td>
</tr>
<tr>
<td><code>No</code></td>
<td><code>chunkSize</code></td>
<td>integer<code>Refer to the description of the </code>chunk_size<code>parameter in the </code>translate</td>
<td>method of the pipeline object.</td>
</tr>
<tr>
<td><code>No</code></td>
<td><code>taskDescription</code>string<code>|</code></td>
<td>null<code>Refer to the description of the </code>task_description<code>parameter in the </code>translate</td>
<td>method of the pipeline object.</td>
</tr>
<tr>
<td><code>No</code></td>
<td><code>outputFormat</code>string<code>|</code></td>
<td>null<code>Refer to the description of the </code>method<code>output_format</code>Description of the relevant parameters.</td>
<td>No</td>
</tr>
<tr>
<td><code>rulesStr</code></td>
<td><code>string</code>|<code>null</code></td>
<td>Refer to the description of the <code>translate</code>method's<code>rules_str</code>parameter in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>fewShotDemoTextContent</code></td>
<td><code>string</code>|<code>null</code></td>
<td>Refer to the description of the <code>translate</code>method's<code>few_shot_demo_text_content</code>parameter in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>fewShotDemoKeyValueList</code></td>
<td><code>string</code>|<code>null</code></td>
<td>Refer to the description of the <code>translate</code>method's<code>few_shot_demo_key_value_list</code>parameter in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>llmRequestInterval</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Refer to the description of the <code>translate</code>method's<code>llm_request_interval</code>parameter in the pipeline object.</td>
<td>No</td>
</tr>
<tr>
<td><code>chatBotConfig</code></td>
<td><code>object</code>|<code>null</code></td>
<td>Refer to the description of the <code>translate</code>method's<code>chat_bot_config</code>parameter in the pipeline object.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When request processing is successful, the response body's <code>result</code> has the following attributes:</li>
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
<td><code>translationResults</code></td>
<td><code>array</code></td>
<td>Translation results.</td>
</tr>
</tbody>
</table>
<p><code>Each element in </code>translationResults<code> is an </code>object</p>
<table>
<thead>
<tr>
<th> with the following attributes:</th>
<th>Name</th>
<th>Type</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>Meaning</code></td>
<td><code>language</code></td>
<td>string</td>
</tr>
<tr>
<td><code>Target language.</code></td>
<td><code>markdown</code></td>
<td>object<code>Markdown result. The object definition is consistent with the </code>markdown<code> returned by the </code>analyzeImages</td>
</tr>
</tbody>
</table>
<li><b> operation.</b></li>Note:<br/><br/>
</details>
<details><summary>Including sensitive parameters such as the API key for large model calls in the request body may pose security risks. If not necessary, set these parameters in the configuration file and do not pass them during the request.</summary>
<details>
<summary>Example of multilingual service invocation</summary>
<pre><code class="language-python">import base64
import pathlib
import pprint
import sys

import requests


API_BASE_URL = "http://127.0.0.1:8080"

file_path = "./demo.jpg"
target_language = "en"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {
    "file": file_data,
    "fileType": 1,
}
resp_visual = requests.post(url=f"{API_BASE_URL}/doctrans-visual", json=payload)
if resp_visual.status_code != 200:
    print(
        f"Request to doctrans-visual failed with status code {resp_visual.status_code}."
    )
    pprint.pp(resp_visual.json())
    sys.exit(1)
result_visual = resp_visual.json()["result"]

markdown_list = []
for i, res in enumerate(result_visual["layoutParsingResults"]):
    md_dir = pathlib.Path(f"markdown_{i}")
    md_dir.mkdir(exist_ok=True)
    (md_dir / "doc.md").write_text(res["markdown"]["text"])
for img_path, img in res["markdown"]["images"].items():
        img_path = md_dir / img_path
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(base64.b64decode(img))
    print(f"The Markdown document to be translated is saved at {md_dir / 'doc.md'}")
    del res["markdown"]["images"]
    markdown_list.append(res["markdown"])
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")

payload = {
    "markdownList": markdown_list,
    "targetLanguage": target_language,
}
resp_translate = requests.post(url=f"{API_BASE_URL}/doctrans-translate", json=payload)
if resp_translate.status_code != 200:
    print(
        f"Request to doctrans-translate failed with status code {resp_translate.status_code}."
    )
    pprint.pprint(resp_translate.json())
    sys.exit(1)
result_translate = resp_translate.json()["result"]

for i, res in enumerate(result_translate["translationResults"]):
    md_dir = pathlib.Path(f"markdown_{i}")
    (md_dir / "doc_translated.md").write_text(res["markdown"]["text"])
    print(f"Translated markdown document saved at {md_dir / 'doc_translated.md'}")</code></pre></details>
</details>
<br/>

📱<b>On-device deployment</b>: On-device deployment is a method that places computing and data processing functions on the user's device itself, allowing the device to process data directly without relying on a remote server. PaddleX supports deploying models on on-device devices such as Android. For detailed on-device deployment procedures, please refer to the [PaddleX On-device Deployment Guide](../../../pipeline_deploy/on_device_deployment.en.md).

You can choose an appropriate way to deploy the model pipeline according to your needs, and then proceed with subsequent AI application integration.

## 4. Secondary Development
If the default model weights provided by the Layout Analysis v3 sub-pipeline in the general document translation pipeline do not meet your accuracy or speed requirements in your scenario, you can try to use<b>the data from your own specific domain or application scenario</b>to further<b>fine-tune</b>the existing model to improve the recognition performance of the general Layout Analysis v3 sub-pipeline in your scenario.

### 4.1 Model Fine-tuning

Since the General Layout Analysis v3 sub-pipeline contains several modules, the subpar performance of the model pipeline may stem from any one of these modules. You can analyze the cases with poor extraction results, use visualized images to identify which module is problematic, and refer to the corresponding fine-tuning tutorial links in the following table to fine-tune the model.

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Fine-tuning module</th>
<th>Fine-tuning reference link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Inaccurate detection of layout areas, such as failure to detect seals and tables</td>
<td>Layout area detection module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/layout_detection.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate recognition of table structures</td>
<td>Table structure recognition module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/table_structure_recognition.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate recognition of formulas</td>
<td>Formula recognition module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/formula_recognition.html">Link</a></td>
</tr>
<tr>
<td>Omission in detecting seal text</td>
<td>Seal text detection module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/seal_text_detection.html">Link</a></td>
</tr>
<tr>
<td>Omission in detecting text</td>
<td>Text detection module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_detection.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate text content</td>
<td>Text recognition module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_recognition.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of vertical or rotated text lines</td>
<td>Text line orientation classification module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/textline_orientation_classification.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of whole image rotation</td>
<td>Document image orientation classification module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate correction of image distortion</td>
<td>Text image correction module</td>
<td>Fine-tuning is temporarily not supported</td>
</tr>
</tbody>
</table>

### 4.2 Model Application
After completing fine-tuning training with your private dataset, you can obtain a local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights into the corresponding location in the pipeline configuration file:

```yaml
......
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayout_plus-L
    model_dir: null # Replace with the path to the weights of the fine-tuned layout region detection model
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
        model_name: PP-OCRv5_server_det
        model_dir: null # Replace with the path to the weights of the fine-tuned text detection model
        limit_side_len: 960
        limit_type: max
        max_side_limit: 4000
        thresh: 0.3
        box_thresh: 0.6
        unclip_ratio: 1.5

      TextRecognition:
        module_name: text_recognition
        model_name: PP-OCRv5_server_rec
        model_dir: null # Replace with the path to the weights of the fine-tuned text recognition model
        batch_size: 1
        score_thresh: 0
......
```

Then, refer to the command-line method or Python script method in the local experience to load the modified pipeline configuration file.

## 5. Multi-hardware Support
PaddleX supports multiple mainstream hardware devices such as NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambrian MLU, <b>and only requires setting the `device` parameter </b>to achieve seamless switching between different hardware.

For example, when using the document scenario information extraction v4 pipeline, to change the running device from NVIDIA GPU to Ascend NPU, you only need to modify `device` to npu in the script:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(
    pipeline="PP-DocTranslation",
    device="npu:0" # change gpu:0 to npu:0
    )
```

If you want to use the general document translation pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
