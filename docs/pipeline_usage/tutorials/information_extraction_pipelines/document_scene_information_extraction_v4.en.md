---
comments: true
---

# PP-ChatOCRv4-doc Pipeline Tutorial

## 1. Introduction to PP-ChatOCRv4-doc Pipeline
PP-ChatOCRv4-doc is a unique document and image intelligent analysis solution from PaddlePaddle, combining LLM, MLLM, and OCR technologies to address complex document information extraction challenges such as layout analysis, rare characters, multi-page PDFs, tables, and seal recognition. Integrated with ERNIE Bot, it fuses massive data and knowledge, achieving high accuracy and wide applicability. This pipeline also provides flexible service deployment options, supporting deployment on various hardware. Furthermore, it offers custom development capabilities, allowing you to train and fine-tune models on your own datasets, with seamless integration of trained models.

<img src="https://github.com/user-attachments/assets/0870cdec-1909-4247-9004-d9efb4ab9635">

The Document Scene Information Extraction v4 pipeline includes modules for **Layout Region Detection**, **Table Structure Recognition**, **Table Classification**, **Table Cell Localization**, **Text Detection**, **Text Recognition**, **Seal Text Detection**, **Text Image Rectification**, and **Document Image Orientation Classification**. The relevant models are integrated as sub-pipelines, and you can view the model configurations of different modules through the [pipeline configuration](../../../../paddlex/configs/pipelines/PP-ChatOCRv4-doc.yaml).

<b>If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.</b> Benchmarks for some models are as follows:

<details><summary> 👉Model List Details</summary>
<p><b>Document Image Orientation Classification Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
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
<td>A document image classification model based on PP-LCNet_x1_0, with four categories: 0°, 90°, 180°, 270°</td>
</tr>
</tbody>
</table>
<p><b>Text Image Rectification Module Models</b>:</p>
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
<p><b>Table Structure Recognition Module Models</b>:</p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANet</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">Training Model</a></td>
<td>59.52</td>
<td>23.96 / 21.75</td>
<td>- / 43.12</td>
<td>6.9</td>
<td>SLANet is a table structure recognition model developed by Baidu PaddleX Team. The model significantly improves the accuracy and inference speed of table structure recognition by adopting a CPU-friendly lightweight backbone network PP-LCNet, a high-low-level feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structural and positional information.</td>
</tr>
<tr>
<td>SLANet_plus</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_plus_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">Training Model</a></td>
<td>63.69</td>
<td>23.43 / 22.16</td>
<td>- / 41.80</td>
<td>6.9</td>
<td>SLANet_plus is an enhanced version of SLANet, the table structure recognition model developed by Baidu PaddleX Team. Compared to SLANet, SLANet_plus significantly improves the recognition ability for wireless and complex tables and reduces the model's sensitivity to the accuracy of table positioning, enabling more accurate recognition even with offset table positioning.</td>
</tr>
</table>

<p><b>Layout Detection Module Models</b>:</p>
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
<td>PP-DocLayout-L</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Training Model</a></td>
<td>90.4</td>
<td>33.59 / 33.59</td>
<td>503.01 / 251.08</td>
<td>123.76</td>
<td>A high-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using RT-DETR-L.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>13.03 / 4.72</td>
<td>43.39 / 24.44</td>
<td>22.578</td>
<td>A layout area localization model with balanced precision and efficiency, trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training Model</a></td>
<td>70.9</td>
<td>11.54 / 3.86</td>
<td>18.53 / 6.29</td>
<td>4.834</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-S.</td>
</tr>
</tbody>
</table>
<b>Note: The evaluation dataset for the above precision metrics is a self-built layout area detection dataset by PaddleOCR, containing 500 common document-type images of Chinese and English papers, magazines, contracts, books, exams, and research reports. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

> ❗ The above list includes the <b>3 core models</b> that are key supported by the text recognition module. The module actually supports a total of <b>11 full models</b>, including several predefined models with different categories. The complete model list is as follows:

* <b>Table Layout Detection Model</b>
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
<td>PicoDet_layout_1x_table</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Training Model</a></td>
<td>97.5</td>
<td>9.57 / 6.63</td>
<td>27.66 / 16.75</td>
<td>7.4</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset using PicoDet-1x, capable of detecting table regions.</td>
</tr>
</tbody></table>
<b>Note: The evaluation dataset for the above precision metrics is a self-built layout table area detection dataset by PaddleOCR, containing 7835 Chinese and English document images with tables. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

* <b>3-Class Layout Detection Model, including Table, Image, and Stamp</b>
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
<td>PicoDet-S_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>88.2</td>
<td>8.43 / 3.44</td>
<td>17.60 / 6.51</td>
<td>4.8</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>12.80 / 9.57</td>
<td>45.04 / 23.86</td>
<td>22.6</td>
<td>A balanced efficiency and precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>95.8</td>
<td>114.80 / 25.65</td>
<td>924.38 / 924.38</td>
<td>470.1</td>
<td>A high-precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using RT-DETR-H.</td>
</tr>
</tbody></table>
<b>Note: The evaluation dataset for the above precision metrics is a self-built layout area detection dataset by PaddleOCR, containing 1154 common document images of Chinese and English papers, magazines, and research reports. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

* <b>5-Class English Document Area Detection Model, including Text, Title, Table, Image, and List</b>
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
<td>PicoDet_layout_1x</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Training Model</a></td>
<td>97.8</td>
<td>9.62 / 6.75</td>
<td>26.96 / 12.77</td>
<td>7.4</td>
<td>A high-efficiency English document layout area localization model trained on the PubLayNet dataset using PicoDet-1x.</td>
</tr>
</tbody></table>
<b>Note: The evaluation dataset for the above precision metrics is the [PubLayNet](https://developer.ibm.com/exchanges/data/all/publaynet/) dataset, containing 11245 English document images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

* <b>17-Class Area Detection Model, including 17 common layout categories: Paragraph Title, Image, Text, Number, Abstract, Content, Figure Caption, Formula, Table, Table Caption, References, Document Title, Footnote, Header, Algorithm, Footer, and Stamp</b>
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
<td>PicoDet-S_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>87.4</td>
<td>8.80 / 3.62</td>
<td>17.51 / 6.35</td>
<td>4.8</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>12.60 / 10.27</td>
<td>43.70 / 24.42</td>
<td>22.6</td>
<td>A balanced efficiency and precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>98.3</td>
<td>115.29 / 101.18</td>
<td>964.75 / 964.75</td>
<td>470.2</td>
<td>A high-precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using RT-DETR-H.</td>
</tr>
</tbody>
</table>

<p><b>Text Detection Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
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

<p><b>Text Recognition Module Models</b>:</p>
* <b>Chinese Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
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
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training Model</a></td>
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
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
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
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
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
<p><b>Formula Recognition Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>BLEU Score</th>
<th>Normed Edit Distance</th>
<th>ExpRate (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
</tr>
</thead>
<tbody>
<tr>
<td>LaTeX_OCR_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Training Model</a></td>
<td>0.8821</td>
<td>0.0823</td>
<td>40.01</td>
<td>1088.89 / 1088.89</td>
<td>- / -</td>
<td>99</td>
</tr>
</tbody>
</table>

<p><b>Seal Text Detection Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
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
<td>PP-OCRv4's server-side seal text detection model, featuring higher accuracy, suitable for deployment on better-equipped servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Training Model</a></td>
<td>96.36</td>
<td>9.70 / 3.56</td>
<td>50.38 / 19.64</td>
<td>4.7</td>
<td>PP-OCRv4's mobile seal text detection model, offering higher efficiency, suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
                    <li><strong>Test Dataset：</strong>
                        <ul>
                          <li>Text Image Rectification Model: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a></li>
                          <li>Layout Region Detection Model: A self-built layout analysis dataset using PaddleOCR, containing 10,000 images of common document types such as Chinese and English papers, magazines, and research reports.</li>
                          <li>Table Structure Recognition Model: A self-built English table recognition dataset using PaddleX.</li>
                          <li>Text Detection Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 500 images for detection.</li>
                          <li>Chinese Recognition Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 11,000 images for text recognition.</li>
                          <li>ch_SVTRv2_rec: Evaluation set A for "OCR End-to-End Recognition Task" in the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a></li>
                          <li>ch_RepSVTR_rec: Evaluation set B for "OCR End-to-End Recognition Task" in the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a></li>
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

## 2. Quick Start
The pre-trained pipelines provided by PaddleX allow for quick experience of their effects. You can locally use Python to experience the effects of the PP-ChatOCRv4-doc pipeline.

### 2.1 Local Experience
Before using the PP-ChatOCRv4-doc pipeline locally, ensure you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md). If you wish to selectively install dependencies, please refer to the relevant instructions in the installation guide. The dependency group corresponding to this pipeline is `ie`.

Before performing model inference, you first need to prepare the API key for the large language model. PP-ChatOCRv4 supports large model services on the [Baidu Cloud Qianfan Platform](https://console.bce.baidu.com/qianfan/ais/console/onlineService) or the locally deployed standard OpenAI interface. If using the Baidu Cloud Qianfan Platform, refer to [Authentication and Authorization](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Um2wxbaps_en) to obtain the API key. If using a locally deployed large model service, refer to the [PaddleNLP Large Model Deployment Documentation](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm) for deployment of the dialogue interface and vectorization interface for large models, and fill in the corresponding `base_url` and `api_key`. If you need to use a multimodal large model for data fusion, refer to the OpenAI service deployment in the [PaddleMIX Model Documentation](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee) for multimodal large model deployment, and fill in the corresponding `base_url` and `api_key`.

After updating the configuration file, you can complete quick inference using just a few lines of Python code. You can use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png) for testing:

**Note**: If local deployment of a multimodal large model is restricted due to the local environment, you can comment out the lines containing the `mllm` variable in the code and only use the large language model for information extraction.

```python
from paddlex import create_pipeline

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

retriever_config = {
    "module_name": "retriever",
    "model_name": "embedding-v1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "qianfan",
    "api_key": "api_key",  # your api_key
}

mllm_chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "PP-DocBee2",
    "base_url": "http://172.0.0.1:8080/v1/chat/completions",  # your local mllm service url
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

pipeline = create_pipeline(pipeline="PP-ChatOCRv4-doc", initial_predictor=False)

visual_predict_res = pipeline.visual_predict(
    input="vehicle_certificate-1.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_common_ocr=True,
    use_seal_recognition=True,
    use_table_recognition=True,
)

visual_info_list = []
for res in visual_predict_res:
    visual_info_list.append(res["visual_info"])
    layout_parsing_result = res["layout_parsing_result"]

vector_info = pipeline.build_vector(
    visual_info_list, flag_save_bytes_vector=True, retriever_config=retriever_config
)
mllm_predict_res = pipeline.mllm_pred(
    input="vehicle_certificate-1.png",
    key_list=["驾驶室准乘人数"],
    mllm_chat_bot_config=mllm_chat_bot_config,
)
mllm_predict_info = mllm_predict_res["mllm_res"]
chat_result = pipeline.chat(
    key_list=["驾驶室准乘人数"],
    visual_info=visual_info_list,
    vector_info=vector_info,
    mllm_predict_info=mllm_predict_info,
    chat_bot_config=chat_bot_config,
    retriever_config=retriever_config,
)
print(chat_result)

```

After running, the output result is as follows:

```
{'chat_res': {'驾驶室准乘人数': '2'}}
```

PP-ChatOCRv4 Prediction Process, API Description, and Output Description:

<details><summary>(1) Instantiate the PP-ChatOCRv4 Pipeline Object by Calling the <code>create_pipeline</code> Method.</summary>

The following are the parameter descriptions:

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
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the name of the pipeline, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for pipeline inference. Supports specifying specific GPU card numbers, such as "gpu:0", other hardware card numbers, such as "npu:0", and CPU as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
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
<td><code>initial_predictor</code></td>
<td>Whether to initialize the inference module (if <code>False</code>, it will be initialized when the relevant inference module is used for the first time).</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
</tbody>
</table>
</details>

<details><summary>(2) Call the <code>visual_predict()</code> Method of the PP-ChatOCRv4 Pipeline Object to Obtain Visual Prediction Results. This method returns a generator.</summary>

The following are the parameters and descriptions of the `visual_predict()` method:

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
<td><code>input</code></td>
<td>The data to be predicted, supporting multiple input types, required.</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
  <li><b>Python Var</b>: Such as <code>numpy.ndarray</code> representing image data.</li>
  <li><b>str</b>: Such as the local path of an image file or PDF file: <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">Example</a>; <b>Local directory</b>, which should contain images to be predicted, such as the local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories, PDF files need to be specified to the specific file path).</li>
  <li><b>List</b>: List elements need to be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use the document orientation classification module.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the document distortion correction module.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to use the text line orientation classification module.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_general_ocr</code></td>
<td>Whether to use the OCR sub-pipeline.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td>Whether to use the seal recognition sub-pipeline.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_table_recognition</code></td>
<td>Whether to use the table recognition sub-pipeline.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>The score threshold for the layout model.</td>
<td><code>float|dict|None</code></td>
<td>
<ul>
  <li><b>float</b>: Any floating-point number between <code>0-1</code>;</li>
  <li><b>dict</b>: <code>{0:0.1}</code> where the key is the category ID and the value is the threshold for that category;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.5</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS.</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>The expansion coefficient for layout detection.</td>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td>
<ul>
  <li><b>float</b>: Any floating-point number greater than <code>0</code>;</li>
  <li><b>Tuple[float,float]</b>: The expansion coefficients in the horizontal and vertical directions, respectively;</li>
  <li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code>, values as float scaling factors for each category.</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>1.0</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>The method for filtering overlapping bounding boxes.</td>
<td><code>str|dict|None</code></td>
<td>
<ul>
  <li><b>str</b>: large, small, union. Respectively representing retaining the larger box, smaller box, or both when overlapping boxes are filtered.</li>
  <li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code> and values as merging modes for each category.</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>large</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>The side length limit for text detection images.</td>
<td><code>int|None</code></td>
<td>
<ul>
  <li><b>int</b>: Any integer greater than <code>0</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>960</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>The type of side length limit for text detection images.</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>str</b>: Supports <code>min</code> and <code>max</code>, where <code>min</code> ensures that the shortest side of the image is not less than <code>det_limit_side_len</code>, and <code>max</code> ensures that the longest side of the image is not greater than <code>limit_side_len</code>.</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>max</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>The pixel threshold for detection. In the output probability map, pixel points with scores greater than this threshold will be considered as text pixels.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.3</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>The bounding box threshold for detection. When the average score of all pixel points within the detection result bounding box is greater than this threshold, the result will be considered as a text region.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.6</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>The expansion coefficient for text detection. This method is used to expand the text region, and the larger the value, the larger the expansion area.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>2.0</code>.</li>
</ul>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>The text recognition threshold. Text results with scores greater than this threshold will be retained.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.0</code>. I.e., no threshold is set.</li>
</ul>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>The side length limit for seal detection images.</td>
<td><code>int|None</code></td>
<td>
<ul>
  <li><b>int</b>: Any integer greater than <code>0</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>960</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_type</code></td>
<td>The type of side length limit for seal detection images.</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>str</b>: Supports <code>min</code> and <code>max</code>, where <code>min</code> ensures that the shortest side of the image is not less than <code>det_limit_side_len</code>, and <code>max</code> ensures that the longest side of the image is not greater than <code>limit_side_len</code>.</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>max</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>The pixel threshold for detection. In the output probability map, pixel points with scores greater than this threshold will be considered as seal pixels.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.3</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>The bounding box threshold for detection. When the average score of all pixel points within the detection result bounding box is greater than this threshold, the result will be considered as a seal region.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.6</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>The expansion coefficient for seal detection. This method is used to expand the seal region, and the larger the value, the larger the expansion area.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>2.0</code>.</li>
</ul>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>The seal recognition threshold. Text results with scores greater than this threshold will be retained.</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.0</code>. I.e., no threshold is set.</li>
</ul>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>
</details>

<details><summary>(3) Process the Visual Prediction Results.</summary>

The prediction result for each sample is of `dict` type, containing two fields: `visual_info` and `layout_parsing_result`. You can obtain visual information through `visual_info` (including `normal_text_dict`, `table_text_list`, `table_html_list`, etc.), and place the information for each sample into the `visual_info_list` list, which will be fed into the large language model later.

Of course, you can also obtain the layout parsing results through `layout_parsing_result`, which includes tables, text, images, and other content contained in the document or image. It supports operations such as printing, saving as an image, and saving as a `json` file:

```python
......
for res in visual_predict_res:
    visual_info_list.append(res["visual_info"])
    layout_parsing_result = res["layout_parsing_result"]
    layout_parsing_result.print()
    layout_parsing_result.save_to_img("./output")
    layout_parsing_result.save_to_json("./output")
    layout_parsing_result.save_to_xlsx("./output")
    layout_parsing_result.save_to_html("./output")
......
```

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
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">Prints the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content with JSON indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output JSON data for better readability, only valid when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-ASCII characters to Unicode. When set to <code>True</code>, all non-ASCII characters will be escaped; <code>False</code> retains the original characters, only valid when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Saves the result as a json file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. When it is a directory, the saved file name is consistent with the input file type</td>
<td>N/A</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output JSON data for better readability, only valid when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-ASCII characters to Unicode. When set to <code>True</code>, all non-ASCII characters will be escaped; <code>False</code> retains the original characters, only valid when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Saves the visual images of each intermediate module in png format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file, supports directory or file path</td>
<td>N/A</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Saves the tables in the file as html files</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file, supports directory or file path</td>
<td>N/A</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Saves the tables in the file as xlsx files</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file, supports directory or file path</td>
<td>N/A</td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal. The content printed to the terminal is explained as follows:
    - `input_path`: `(str)` The input path of the image to be predicted

    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF; otherwise, it is `None`

    - `model_settings`: `(Dict[str, bool])` Model parameters required for the pipeline

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing pipeline
        - `use_general_ocr`: `(bool)` Controls whether to enable the OCR pipeline
        - `use_seal_recognition`: `(bool)` Controls whether to enable the seal recognition pipeline
        - `use_table_recognition`: `(bool)` Controls whether to enable the table recognition pipeline
        - `use_formula_recognition`: `(bool)` Controls whether to enable the formula recognition pipeline

    - `parsing_res_list`: `(List[Dict])` A list of parsing results, each element is a dictionary, and the list order is the reading order after parsing.
        - `block_bbox`: `(np.ndarray)` The bounding box of the layout area.
        - `block_label`: `(str)` The label of the layout area, such as `text`, `table`, etc.
        - `block_content`: `(str)` The content within the layout area.

    - `overall_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` A dictionary of global OCR results
      - `input_path`: `(Union[str, None])` The image path accepted by the OCR pipeline, when the input is `numpy.ndarray`, it is saved as `None`
      - `model_settings`: `(Dict)` Model configuration parameters for the OCR pipeline
      - `dt_polys`: `(List[numpy.ndarray])` A list of polygon boxes for text detection. Each detection box is represented by a numpy array of 4 vertex coordinates, with a shape of (4, 2) and a data type of int16
      - `dt_scores`: `(List[float])` A list of confidence scores for text detection boxes
      - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the text detection module
        - `limit_side_len`: `(int)` The side length limit for image preprocessing
        - `limit_type`: `(str)` The processing method for the side length limit
        - `thresh`: `(float)` The confidence threshold for text pixel classification
        - `box_thresh`: `(float)` The confidence threshold for text detection boxes
        - `unclip_ratio`: `(float)` The inflation coefficient for text detection boxes
        - `text_type`: `(str)` The type of text detection, currently fixed as "general"

      - `text_type`: `(str)` The type of text detection, currently fixed as "general"
      - `textline_orientation_angles`: `(List[int])` The prediction results of text line orientation classification. When enabled, it returns actual angle values (e.g., [0,0,1])
      - `text_rec_score_thresh`: `(float)` The filtering threshold for text recognition results
      - `rec_texts`: `(List[str])` A list of text recognition results, only including texts with confidence exceeding `text_rec_score```markdown
- Calling the `save_to_json()` method will save the aforementioned content to the specified `save_path`. If a directory is specified, the save path will be `save_path/{your_img_basename}.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, `numpy.array` types will be converted to list form.
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, the save path will be `save_path/{your_img_basename}_ocr_res_img.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (Production pipelines often involve numerous result images, so it is not recommended to specify a specific file path directly, as multiple images will be overwritten, leaving only the last one.)

In addition, it is also supported to obtain visualized images with results and prediction results through attributes, as detailed below:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Obtain prediction results in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Obtain visualized images in <code>dict</code> format</td>
</tr>
</table>

- The prediction result obtained by the `json` attribute is data of type `dict`, with content consistent with that saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is data of type `dict`. The keys are `layout_det_res`, `overall_ocr_res`, `text_paragraphs_ocr_res`, `formula_res_region1`, `table_cell_img`, and `seal_res_region1`, with corresponding values being `Image.Image` objects: used for displaying visualized images of layout detection, OCR, OCR text paragraphs, formulas, tables, and seal results, respectively. If optional modules are not used, only `layout_det_res` will be included in the dictionary.
</details>

<details><summary>(4) Call the <code>build_vector()</code> method of the PP-ChatOCRv4 pipeline object to construct vectors for text content.</summary>

Below are the parameters and their descriptions for the `build_vector()` method:

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
<td><code>visual_info</code></td>
<td>Visual information, which can be a dictionary containing visual information or a list composed of such dictionaries</td>
<td><code>list|dict</code></td>
<td>
<code>None</code>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>min_characters</code></td>
<td>Minimum number of characters</td>
<td><code>int</code></td>
<td>
A positive integer greater than 0, determined based on the token length supported by the large language model
</td>
<td><code>3500</code></td>
</tr>
<tr>
<td><code>block_size</code></td>
<td>Chunk size for establishing a vector library for long text</td>
<td><code>int</code></td>
<td>
A positive integer greater than 0, determined based on the token length supported by the large language model
</td>
<td><code>300</code></td>
</tr>
<tr>
<td><code>flag_save_bytes_vector</code></td>
<td>Whether to save text as a binary file</td>
<td><code>bool</code></td>
<td>
<code>True|False</code>
</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>retriever_config</code></td>
<td>Configuration parameters for the vector retrieval large model, referring to the "LLM_Retriever" field in the configuration file</td>
<td><code>dict</code></td>
<td>
<code>None</code>
</td>
<td><code>None</code></td>
</tr>
</table>

This method returns a dictionary containing visual text information, with the following content:

- `flag_save_bytes_vector`: `(bool)` Whether the result is saved as a binary file
- `flag_too_short_text`: `(bool)` Whether the text length is less than the minimum number of characters
- `vector`: `(str|list)` Binary content or text content of the text, depending on the values of `flag_save_bytes_vector` and `min_characters`. If `flag_save_bytes_vector=True` and the text length is greater than or equal to the minimum number of characters, binary content is returned; otherwise, the original text is returned.
</details>

<details><summary>(5) Call the <code>mllm_pred()</code> method of the PP-ChatOCRv4 pipeline object to obtain multimodal large model extraction results.</summary>

Below are the parameters and their descriptions for the `mllm_pred()` method:

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
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types, required</td>
<td><code>Python Var|str</code></td>
<td>
<ul>
  <li><b>Python Var</b>: Such as <code>numpy.ndarray</code> representing image data</li>
  <li><b>str</b>: Local path of an image file or a single-page PDF file, e.g., <code>/root/data/img.jpg</code>; <b>or URL link</b>, such as the network URL of an image file or a single-page PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">Example</a>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>key_list</code></td>
<td>A single key or a list of keys used to extract information</td>
<td><code>Union[str, List[str]]</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>mllm_chat_bot_config</code></td>
<td>Configuration parameters for the multimodal large model, referring to the "MLLM_Chat" field in the configuration file</td>
<td><code>dict</code></td>
<td>
<code>None</code>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

</details>

<details><summary>(6) Call the <code>chat()</code> method of the PP-ChatOCRv4 pipeline object to extract key information.</summary>

Below are the parameters and their descriptions for the `chat()` method:

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
<td><code>key_list</code></td>
<td>A single key or a list of keys used to extract information</td>
<td><code>Union[str, List[str]]</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>visual_info</code></td>
<td>Visual information results</td>
<td><code>List[dict]</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_vector_retrieval</code></td>
<td>Whether to use vector retrieval</td>
<td><code>bool</code></td>
<td><code>True|False</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>vector_info</code></td>
<td>Vector information for retrieval</td>
<td><code>dict</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>min_characters</code></td>
<td>Minimum number of characters required</td>
<td><code>int</code></td>
<td>A positive integer greater than 0</td>
<td><code>3500</code></td>
</tr>
<tr>
<td><code>text_task_description</code></td>
<td>Description of the text task</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_output_format</code></td>
<td>Output format of the text result</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rules_str</code></td>
<td>Rules for generating text results</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_few_shot_demo_text_content</code></td>
<td>Text content for few-shot demonstration</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_few_shot_demo_key_value_list</code></td>
<td>Key-value list for few-shot demonstration</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_task_description</code></td>
<td>Description of the table task</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_output_format</code></td>
<td>Output format of the table result</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_rules_str</code></td>
<td>Rules for generating table results</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_few_shot_demo_text_content</code></td>
<td>Text content for table few-shot demonstration</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_few_shot_demo_key_value_list</code></td>
<td>Key-value list for table few-shot demonstration</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>mllm_predict_info</code></td>
<td>Results from the multimodal large language model</td>
<td><code>dict</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>mllm_integration_strategy</code></td>
<td>Integration strategy for multimodal large language model and large language model data, supporting the use of either alone or the fusion of both results</td>
<td><code>str</code></td>
<td><code>"integration"</code></td>
<td><code>"integration", "llm_only", and "mllm_only"</code></td>
</tr>
<tr>
<td><code>chat_bot_config</code></td>
<td>Configuration information for the large language model, with content referring to the "LLM_Chat" field in the pipeline configuration file</td>
<td><code>dict</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>retriever_config</code></td>
<td>Configuration parameters for the vector retrieval large model, with content referring to the "LLM_Retriever" field in the configuration file</td>
<td><code>dict</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

This method will print the results to the terminal. The content printed to the terminal is explained as follows:
  - `chat_res`: `(dict)` The result of information extraction, which is a dictionary containing the keys to be extracted and their corresponding values.

</details>

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy in production, you can proceed directly with development integration/deployment.

If you need to apply the pipeline directly in your Python project, you can refer to the sample code in [2.2 Local Experience](#22-local-experience).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Inference**: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed instructions on high-performance inference, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

☁️ **Serving**: Serving is a common deployment form in actual production environments. By encapsulating the inference functionality as a service, clients can access these services through network requests to obtain inference results. PaddleX supports multiple serving solutions for pipelines. For detailed instructions on serving, please refer to the [PaddleX Serving Guide](../../../pipeline_deploy/serving.md).

Below are the API references for basic serving and multi-language service invocation examples:

<details><summary>API Reference</summary>

<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is successfully processed, the response status code is <code>200</code>, and the response body has the following properties:</li>
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
<td>UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed at <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed at <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not successfully processed, the response body has the following properties:</li>
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
<td>UUID of the request.</td>
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
<p>Uses computer vision models to analyze images, obtain OCR, table recognition results, etc., and extract key information from the images.</p>
<p><code>POST /chatocr-visual</code></p>
<ul>
<li>Properties of the request body:</li>
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
<td>URL of an image file or PDF file accessible to the server, or Base64 encoded result of the content of the above file types. By default, for PDF files exceeding 10 pages, only the content of the first 10 pages will be processed.<br />
To remove the page limit, please add the following configuration to the pipeline configuration file:
<pre><code>Serving:
  extra:
    max_num_input_imgs: null
</code></pre></td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>File type. <code>0</code> represents a PDF file, <code>1</code> represents an image file. If this property is not present in the request body, the file type will be inferred based on the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_unwarping</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTextlineOrientation</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_textline_orientation</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useSealRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_seal_recognition</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_table_recognition</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>object</code> | </code><code>null</code></td>
<td>Please refer to the description of the <code>layout_threshold</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_nms</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_unclip_ratio</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_merge_bboxes_mode</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_limit_side_len</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_limit_type</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_thresh</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_box_thresh</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_det_unclip_ratio</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_rec_score_thresh</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_limit_side_len</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_limit_type</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_thresh</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_box_thresh</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_det_unclip_ratio</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>seal_rec_score_thresh</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
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
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_orientation_classify</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> of the response body has the following properties:</li>
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
<td>Analysis results obtained using computer vision models. The array length is 1 (for image input) or the actual number of document pages processed (for PDF input). For PDF input, each element in the array represents the result of each page actually processed in the PDF file.</td>
</tr>
<tr>
<td><code>visualInfo</code></td>
<td><code>array</code></td>
<td>Key information in the image, which can be used as input for other operations.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Input data information.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>layoutParsingResults</code> is an <code>object</code> with the following properties:</p>
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
<td>A simplified version of the <code>res</code> field in the JSON representation of the <code>layout_parsing_result</code> generated by the pipeline object's <code>visual_predict</code> method, with the <code>input_path</code> and <code>page_index</code> fields removed.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Refer to the description of <code>img</code> attribute of the pipeline's visual prediction result.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Input image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>buildVectorStore</code></b></li>
</ul>
<p>Builds a vector database.</p>
<p><code>POST /chatocr-vector</code></p>
<ul>
<li>Properties of the request body:</li>
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
<td><code>visualInfo</code></td>
<td><code>array</code></td>
<td>Key information in the image. Provided by the <code>analyzeImages</code> operation.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>minCharacters</code></td>
<td><code>integer</code></td>
<td>Please refer to the description of the <code>min_characters</code> parameter of the pipeline object's <code>build_vector</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>blockSize</code></td>
<td><code>integer</code></td>
<td>Please refer to the description of the <code>block_size</code> parameter of the pipeline object's <code>build_vector</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>retrieverConfig</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>retriever_config</code> parameter of the pipeline object's <code>build_vector</code> method.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> of the response body has the following properties:</li>
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
<td><code>vectorInfo</code></td>
<td><code>object</code></td>
<td>Serialized result of the vector database, which can be used as input for other operations.</td>
</tr>
</tbody>
</table>
<li><b><code>invokeMLLM</code></b></li>
</ul>
<p>Invoke the MLLM.</p>
<p><code>POST /chatocr-mllm</code></p>
<ul>
<li>Properties of the request body:</li>
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
<td><code>image</code></td>
<td><code>string</code></td>
<td>URL of an image file accessible by the server or the Base64-encoded content of the image file.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>keyList</code></td>
<td><code>array</code></td>
<td>List of keys.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>mllmChatBotConfig</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>mllm_chat_bot_config</code> parameter of the pipeline object's <code>mllm_pred</code> method.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> of the response body has the following property:</li>
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
<td><code>mllmPredictInfo</code></td>
<td><code>object</code></td>
<td>MLLM invocation result.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>chat</code></b></li>
</ul>
<p>Interacts with large language models to extract key information using them.</p>
<p><code>POST /chatocr-chat</code></p>
<ul>
<li>Properties of the request body:</li>
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
<td><code>keyList</code></td>
<td><code>array</code></td>
<td>List of keys.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>visualInfo</code></td>
<td><code>object</code></td>
<td>Key information in the image. Provided by the <code>analyzeImages</code> operation.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>useVectorRetrieval</code></td>
<td><code>boolean</code></td>
<td>Please refer to the description of the <code>use_vector_retrieval</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>vectorInfo</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Serialized result of the vector database. Provided by the <code>buildVectorStore</code> operation. Please note that the deserialization process involves performing an unpickle operation. To prevent malicious attacks, be sure to use data from trusted sources.</td>
<td>No</td>
</tr>
<tr>
<td><code>minCharacters</code></td>
<td><code>integer</code></td>
<td>Please refer to the description of the <code>block_size</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textTaskDescription</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_task_description</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textOutputFormat</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_output_format</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRulesStr</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_rules_str</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textFewShotDemoTextContent</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_few_shot_demo_text_content</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>textFewShotDemoKeyValueList</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>text_few_shot_demo_key_value_list</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>tableTaskDescription</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>table_task_description</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>tableOutputFormat</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>table_output_format</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>tableRulesStr</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>table_rules_str</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>tableFewShotDemoTextContent</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>table_few_shot_demo_text_content</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>tableFewShotDemoKeyValueList</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>table_few_shot_demo_key_value_list</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>mllmPredictInfo</code></td>
<td><code>object</code> | <code>null</code></td>
<td>MLLM invocation result. Provided by the <code>invokeMllm</code> operation.</td>
<td>No</td>
</tr>
<tr>
<td><code>mllmIntegrationStrategy</code></td>
<td><code>string</code></td>
<td>Please refer to the description of the <code>mllm_integration_strategy</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>chatBotConfig</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>chat_bot_config</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>retrieverConfig</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Please refer to the description of the <code>retriever_config</code> parameter of the pipeline object's <code>chat</code> method.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> of the response body has the following properties:</li>
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
<td><code>chatResult</code></td>
<td><code>object</code></td>
<td>Key information extraction result.</td>
</tr>
</tbody>
</table>
<li><b>Note:</b></li>
Including sensitive parameters such as API key for large model calls in the request body can be a security risk. If not necessary, set these parameters in the configuration file and do not pass them on request.
<br/><br/>
</details>

<details><summary>Multi-language Service Invocation Examples</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">
# This script only shows the use case for images. For calling with other file types, please read the API reference and make adjustments.

import base64
import pprint
import sys
import requests


API_BASE_URL = "http://127.0.0.1:8080"

image_path = "./demo.jpg"
keys = ["name"]

with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "file": image_data,
    "fileType": 1,
}

resp_visual = requests.post(url=f"{API_BASE_URL}/chatocr-visual", json=payload)
if resp_visual.status_code != 200:
    print(
        f"Request to chatocr-visual failed with status code {resp_visual.status_code}."
    )
    pprint.pp(resp_visual.json())
    sys.exit(1)
result_visual = resp_visual.json()["result"]

for i, res in enumerate(result_visual["layoutParsingResults"]):
    print(res["prunedResult"])
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")

payload = {
    "visualInfo": result_visual["visualInfo"],
}
resp_vector = requests.post(url=f"{API_BASE_URL}/chatocr-vector", json=payload)
if resp_vector.status_code != 200:
    print(
        f"Request to chatocr-vector failed with status code {resp_vector.status_code}."
    )
    pprint.pp(resp_vector.json())
    sys.exit(1)
result_vector = resp_vector.json()["result"]

payload = {
    "image": image_data,
    "keyList": keys,
}
resp_mllm = requests.post(url=f"{API_BASE_URL}/chatocr-mllm", json=payload)
if resp_mllm.status_code != 200:
    print(
        f"Request to chatocr-mllm failed with status code {resp_mllm.status_code}."
    )
    pprint.pp(resp_mllm.json())
    sys.exit(1)
result_mllm = resp_mllm.json()["result"]

payload = {
    "keyList": keys,
    "visualInfo": result_visual["visualInfo"],
    "useVectorRetrieval": True,
    "vectorInfo": result_vector["vectorInfo"],
    "mllmPredictInfo": result_mllm["mllmPredictInfo"],
}
resp_chat = requests.post(url=f"{API_BASE_URL}/chatocr-chat", json=payload)
if resp_chat.status_code != 200:
    print(
        f"Request to chatocr-chat failed with status code {resp_chat.status_code}."
    )
    pprint.pp(resp_chat.json())
    sys.exit(1)
result_chat = resp_chat.json()["result"]
print("Final result:")
print(result_chat["chatResult"])
</code></pre>
</details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &lt;fstream&gt;
#include &lt;vector&gt;
#include &lt;string&gt;
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

using json = nlohmann::json;

std::string encode_image(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("File open error.");
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    file.read(buf.data(), size);
    return base64::to_base64(std::string(buf.data(), buf.size()));
}

int main() {
    httplib::Client client("localhost", 8080);
    std::string imagePath = "./demo.jpg";
    std::string imageData = encode_image(imagePath);
    json keys = { "Name" };

    json payload_visual = { {"file", imageData}, {"fileType", 1} };
    auto resp1 = client.Post("/chatocr-visual", payload_visual.dump(), "application/json");
    if (!resp1 || resp1->status != 200) {
        std::cerr << "chatocr-visual failed.\n"; return 1;
    }
    json result_visual = json::parse(resp1->body)["result"];

    for (size_t i = 0; i < result_visual["layoutParsingResults"].size(); ++i) {
        auto& res = result_visual["layoutParsingResults"][i];
        std::cout << "prunedResult: " << res["prunedResult"].dump() << "\n";
        if (res.contains("outputImages")) {
            for (auto& [name, b64] : res["outputImages"].items()) {
                std::string outPath = name + "_" + std::to_string(i) + ".jpg";
                std::string decoded = base64::from_base64(b64.get<std::string>());
                std::ofstream out(outPath, std::ios::binary);
                out.write(decoded.data(), decoded.size());
                out.close();
                std::cout << "Saved: " << outPath << "\n";
            }
        }
    }

    json payload_vector = { {"visualInfo", result_visual["visualInfo"]} };
    auto resp2 = client.Post("/chatocr-vector", payload_vector.dump(), "application/json");
    if (!resp2 || resp2->status != 200) {
        std::cerr << "chatocr-vector failed.\n"; return 1;
    }
    json result_vector = json::parse(resp2->body)["result"];

    json payload_mllm = { {"image", imageData}, {"keyList", keys} };
    auto resp3 = client.Post("/chatocr-mllm", payload_mllm.dump(), "application/json");
    if (!resp3 || resp3->status != 200) {
        std::cerr << "chatocr-mllm failed.\n"; return 1;
    }
    json result_mllm = json::parse(resp3->body)["result"];

    json payload_chat = {
        {"keyList", keys},
        {"visualInfo", result_visual["visualInfo"]},
        {"useVectorRetrieval", true},
        {"vectorInfo", result_vector["vectorInfo"]},
        {"mllmPredictInfo", result_mllm["mllmPredictInfo"]}
    };
    auto resp4 = client.Post("/chatocr-chat", payload_chat.dump(), "application/json");
    if (!resp4 || resp4->status != 200) {
        std::cerr << "chatocr-chat failed.\n"; return 1;
    }

    json result_chat = json::parse(resp4->body)["result"];
    std::cout << "Final chat result: " << result_chat["chatResult"] << std::endl;

    return 0;
}
</code></pre></details>

<details><summary>Java</summary>

<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.Iterator;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_BASE_URL = "http://localhost:8080";
        String imagePath = "./demo.jpg";
        String[] keys = {"Name"};

        OkHttpClient client = new OkHttpClient();
        ObjectMapper objectMapper = new ObjectMapper();
        MediaType JSON = MediaType.parse("application/json; charset=utf-8");

        byte[] imageBytes = java.nio.file.Files.readAllBytes(new File(imagePath).toPath());
        String base64Image = Base64.getEncoder().encodeToString(imageBytes);

        ObjectNode visualPayload = objectMapper.createObjectNode();
        visualPayload.put("file", base64Image);
        visualPayload.put("fileType", 1);

        Request requestVisual = new Request.Builder()
                .url(API_BASE_URL + "/chatocr-visual")
                .post(RequestBody.create(JSON, visualPayload.toString()))
                .build();

        Response responseVisual = client.newCall(requestVisual).execute();
        if (!responseVisual.isSuccessful()) {
            System.err.println("chatocr-visual failed: " + responseVisual.code());
            return;
        }

        JsonNode resultVisual = objectMapper.readTree(responseVisual.body().string()).get("result");

        JsonNode layoutResults = resultVisual.get("layoutParsingResults");
        for (int i = 0; i < layoutResults.size(); i++) {
            JsonNode res = layoutResults.get(i);
            System.out.println("prunedResult [" + i + "]: " + res.get("prunedResult").toString());

            JsonNode outputImages = res.get("outputImages");
            if (outputImages != null && outputImages.isObject()) {
                Iterator<String> names = outputImages.fieldNames();
                while (names.hasNext()) {
                    String imgName = names.next();
                    String imgBase64 = outputImages.get(imgName).asText();
                    byte[] imgBytes = Base64.getDecoder().decode(imgBase64);
                    String imgPath = imgName + "_" + i + ".jpg";
                    try (FileOutputStream fos = new FileOutputStream(imgPath)) {
                        fos.write(imgBytes);
                        System.out.println("Saved image: " + imgPath);
                    }
                }
            }
        }

        ObjectNode vectorPayload = objectMapper.createObjectNode();
        vectorPayload.set("visualInfo", resultVisual.get("visualInfo"));

        Request requestVector = new Request.Builder()
                .url(API_BASE_URL + "/chatocr-vector")
                .post(RequestBody.create(JSON, vectorPayload.toString()))
                .build();

        Response responseVector = client.newCall(requestVector).execute();
        if (!responseVector.isSuccessful()) {
            System.err.println("chatocr-vector failed: " + responseVector.code());
            return;
        }

        JsonNode resultVector = objectMapper.readTree(responseVector.body().string()).get("result");

        ObjectNode mllmPayload = objectMapper.createObjectNode();
        mllmPayload.put("image", base64Image);
        mllmPayload.putArray("keyList").add(keys[0]);

        Request requestMllm = new Request.Builder()
                .url(API_BASE_URL + "/chatocr-mllm")
                .post(RequestBody.create(JSON, mllmPayload.toString()))
                .build();

        Response responseMllm = client.newCall(requestMllm).execute();
        if (!responseMllm.isSuccessful()) {
            System.err.println("chatocr-mllm failed: " + responseMllm.code());
            return;
        }

        JsonNode resultMllm = objectMapper.readTree(responseMllm.body().string()).get("result");

        ObjectNode chatPayload = objectMapper.createObjectNode();
        chatPayload.putArray("keyList").add(keys[0]);
        chatPayload.set("visualInfo", resultVisual.get("visualInfo"));
        chatPayload.put("useVectorRetrieval", true);
        chatPayload.set("vectorInfo", resultVector.get("vectorInfo"));
        chatPayload.set("mllmPredictInfo", resultMllm.get("mllmPredictInfo"));

        Request requestChat = new Request.Builder()
                .url(API_BASE_URL + "/chatocr-chat")
                .post(RequestBody.create(JSON, chatPayload.toString()))
                .build();

        Response responseChat = client.newCall(requestChat).execute();
        if (!responseChat.isSuccessful()) {
            System.err.println("chatocr-chat failed: " + responseChat.code());
            return;
        }

        JsonNode resultChat = objectMapper.readTree(responseChat.body().string()).get("result");
        System.out.println("Final result:");
        System.out.println(resultChat.get("chatResult").toString());
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

func sendPostRequest(url string, payload map[string]interface{}) (map[string]interface{}, error) {
    bodyBytes, err := json.Marshal(payload)
    if err != nil {
        return nil, fmt.Errorf("error marshaling payload: %v", err)
    }

    req, err := http.NewRequest("POST", url, bytes.NewBuffer(bodyBytes))
    if err != nil {
        return nil, fmt.Errorf("error creating request: %v", err)
    }
    req.Header.Set("Content-Type", "application/json")

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return nil, fmt.Errorf("error sending request: %v", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("status code error: %d", resp.StatusCode)
    }

    respBytes, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        return nil, fmt.Errorf("error reading response: %v", err)
    }

    var result map[string]interface{}
    if err := json.Unmarshal(respBytes, &result); err != nil {
        return nil, fmt.Errorf("error unmarshaling response: %v", err)
    }
    return result["result"].(map[string]interface{}), nil
}

func main() {
    apiBase := "http://localhost:8080"
    imagePath := "./demo.jpg"
    keys := []string{"Name"}

    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Printf("read image failed : %v\n", err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    visualPayload := map[string]interface{}{
        "file":     imageData,
        "fileType": 1,
    }
    visualResult, err := sendPostRequest(apiBase+"/chatocr-visual", visualPayload)
    if err != nil {
        fmt.Printf("chatocr-visual request error: %v\n", err)
        return
    }

    layoutResults := visualResult["layoutParsingResults"].([]interface{})
    for i, res := range layoutResults {
        layout := res.(map[string]interface{})
        fmt.Println("PrunedResult:", layout["prunedResult"])
        outputImages := layout["outputImages"].(map[string]interface{})
        for name, img := range outputImages {
            imgBytes, _ := base64.StdEncoding.DecodeString(img.(string))
            filename := fmt.Sprintf("%s_%d.jpg", name, i)
            if err := os.WriteFile(filename, imgBytes, 0644); err == nil {
                fmt.Printf("save image：%s\n", filename)
            }
        }
    }

    vectorPayload := map[string]interface{}{
        "visualInfo": visualResult["visualInfo"],
    }
    vectorResult, err := sendPostRequest(apiBase+"/chatocr-vector", vectorPayload)
    if err != nil {
        fmt.Printf("chatocr-vector request error: %v\n", err)
        return
    }

    mllmPayload := map[string]interface{}{
        "image":   imageData,
        "keyList": keys,
    }
    mllmResult, err := sendPostRequest(apiBase+"/chatocr-mllm", mllmPayload)
    if err != nil {
        fmt.Printf("chatocr-mllm request error: %v\n", err)
        return
    }

    chatPayload := map[string]interface{}{
        "keyList":           keys,
        "visualInfo":        visualResult["visualInfo"],
        "useVectorRetrieval": true,
        "vectorInfo":        vectorResult["vectorInfo"],
        "mllmPredictInfo":   mllmResult["mllmPredictInfo"],
    }
    chatResult, err := sendPostRequest(apiBase+"/chatocr-chat", chatPayload)
    if err != nil {
        fmt.Printf("chatocr-chat request error: %v\n", err)
        return
    }

    fmt.Println("final result：", chatResult["chatResult"])
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
    static readonly string API_BASE_URL = "http://localhost:8080";
    static readonly string inputFilePath = "./demo.jpg";
    static readonly string[] keys = { "Name" };

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] imageBytes = File.ReadAllBytes(inputFilePath);
        string imageData = Convert.ToBase64String(imageBytes);

        var payloadVisual = new JObject
        {
            { "file", imageData },
            { "fileType", 1 }
        };

        var respVisual = await httpClient.PostAsync($"{API_BASE_URL}/chatocr-visual",
            new StringContent(payloadVisual.ToString(), Encoding.UTF8, "application/json"));

        if (!respVisual.IsSuccessStatusCode)
        {
            Console.Error.WriteLine($"Request to chatocr-visual failed: {respVisual.StatusCode}");
            Console.Error.WriteLine(await respVisual.Content.ReadAsStringAsync());
            return;
        }

        JObject resultVisual = JObject.Parse(await respVisual.Content.ReadAsStringAsync())["result"] as JObject;

        var layoutParsingResults = (JArray)resultVisual["layoutParsingResults"];
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
                        File.WriteAllBytes(imgPath, Convert.FromBase64String(base64Img));
                        Console.WriteLine($"Output image saved at {imgPath}");
                    }
                }
            }
        }

        var payloadVector = new JObject
        {
            { "visualInfo", resultVisual["visualInfo"] }
        };

        var respVector = await httpClient.PostAsync($"{API_BASE_URL}/chatocr-vector",
            new StringContent(payloadVector.ToString(), Encoding.UTF8, "application/json"));

        if (!respVector.IsSuccessStatusCode)
        {
            Console.Error.WriteLine($"Request to chatocr-vector failed: {respVector.StatusCode}");
            Console.Error.WriteLine(await respVector.Content.ReadAsStringAsync());
            return;
        }

        JObject resultVector = JObject.Parse(await respVector.Content.ReadAsStringAsync())["result"] as JObject;

        var payloadMllm = new JObject
        {
            { "image", imageData },
            { "keyList", new JArray(keys) }
        };

        var respMllm = await httpClient.PostAsync($"{API_BASE_URL}/chatocr-mllm",
            new StringContent(payloadMllm.ToString(), Encoding.UTF8, "application/json"));

        if (!respMllm.IsSuccessStatusCode)
        {
            Console.Error.WriteLine($"Request to chatocr-mllm failed: {respMllm.StatusCode}");
            Console.Error.WriteLine(await respMllm.Content.ReadAsStringAsync());
            return;
        }

        JObject resultMllm = JObject.Parse(await respMllm.Content.ReadAsStringAsync())["result"] as JObject;

        var payloadChat = new JObject
        {
            { "keyList", new JArray(keys) },
            { "visualInfo", resultVisual["visualInfo"] },
            { "useVectorRetrieval", true },
            { "vectorInfo", resultVector["vectorInfo"] },
            { "mllmPredictInfo", resultMllm["mllmPredictInfo"] }
        };

        var respChat = await httpClient.PostAsync($"{API_BASE_URL}/chatocr-chat",
            new StringContent(payloadChat.ToString(), Encoding.UTF8, "application/json"));

        if (!respChat.IsSuccessStatusCode)
        {
            Console.Error.WriteLine($"Request to chatocr-chat failed: {respChat.StatusCode}");
            Console.Error.WriteLine(await respChat.Content.ReadAsStringAsync());
            return;
        }

        JObject resultChat = JObject.Parse(await respChat.Content.ReadAsStringAsync())["result"] as JObject;
        Console.WriteLine("Final result:");
        Console.WriteLine(resultChat["chatResult"]);
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');
const path = require('path');

const API_BASE_URL = 'http://localhost:8080';
const imagePath = './demo.jpg';
const keys = ['Name'];

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

(async () => {
  try {
    const imageData = encodeImageToBase64(imagePath);

    const respVisual = await axios.post(`${API_BASE_URL}/chatocr-visual`, {
      file: imageData,
      fileType: 1
    });

    const resultVisual = respVisual.data.result;
    resultVisual.layoutParsingResults.forEach((res, i) => {
      console.log(`\n[${i}] prunedResult:\n`, res.prunedResult);
      const outputImages = res.outputImages || {};
      for (const [imgName, base64Img] of Object.entries(outputImages)) {
        const fileName = `${imgName}_${i}.jpg`;
        fs.writeFileSync(fileName, Buffer.from(base64Img, 'base64'));
        console.log(`Output image saved at ${fileName}`);
      }
    });

    const respVector = await axios.post(`${API_BASE_URL}/chatocr-vector`, {
      visualInfo: resultVisual.visualInfo
    });
    const resultVector = respVector.data.result;

    const respMllm = await axios.post(`${API_BASE_URL}/chatocr-mllm`, {
      image: imageData,
      keyList: keys
    });
    const resultMllm = respMllm.data.result;

    const respChat = await axios.post(`${API_BASE_URL}/chatocr-chat`, {
      keyList: keys,
      visualInfo: resultVisual.visualInfo,
      useVectorRetrieval: true,
      vectorInfo: resultVector.vectorInfo,
      mllmPredictInfo: resultMllm.mllmPredictInfo
    });

    const resultChat = respChat.data.result;
    console.log('\nFinal result:\n', resultChat.chatResult);

  } catch (error) {
    if (error.response) {
      console.error(`❌ Request failed: ${error.response.status}`);
      console.error(error.response.data);
    } else {
      console.error('❌ Error occurred:', error.message);
    }
  }
})();
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_BASE_URL = "http://localhost:8080";
$image_path = "./demo.jpg";
$keys = ["Name"];

$image_data = base64_encode(file_get_contents($image_path));

$payload_visual = [
    "file" => $image_data,
    "fileType" => 1
];
$response_visual_raw = send_post_raw("$API_BASE_URL/chatocr-visual", $payload_visual);
$response_visual = json_decode($response_visual_raw, true);
if (!isset($response_visual["result"])) {
    echo "chatocr-visual request error\n";
    print_r($response_visual);
    exit(1);
}
$result_visual_raw = json_decode($response_visual_raw, false)->result;
$result_visual_arr = $response_visual["result"];

foreach ($result_visual_arr["layoutParsingResults"] as $i => $res) {
    echo "[$i] prunedResult:\n";
    print_r($res["prunedResult"]);
    if (!empty($res["outputImages"])) {
        foreach ($res["outputImages"] as $img_name => $base64_img) {
            $img_path = "{$img_name}_{$i}.jpg";
            file_put_contents($img_path, base64_decode($base64_img));
            echo "Output image saved at $img_path\n";
        }
    }
}

$payload_vector = [
    "visualInfo" => $result_visual_raw->visualInfo
];
$response_vector_raw = send_post_raw("$API_BASE_URL/chatocr-vector", $payload_vector);
$response_vector = json_decode($response_vector_raw, true);
if (!isset($response_vector["result"])) {
    echo "chatocr-vector request error\n";
    print_r($response_vector);
    exit(1);
}
$result_vector_raw = json_decode($response_vector_raw, false)->result;

$payload_mllm = [
    "image" => $image_data,
    "keyList" => $keys
];
$response_mllm_raw = send_post_raw("$API_BASE_URL/chatocr-mllm", $payload_mllm);
$response_mllm = json_decode($response_mllm_raw, true);
if (!isset($response_mllm["result"])) {
    echo "chatocr-mllm request error\n";
    print_r($response_mllm);
    exit(1);
}
$result_mllm_raw = json_decode($response_mllm_raw, false)->result;

$payload_chat = [
    "keyList" => $keys,
    "visualInfo" => $result_visual_raw->visualInfo,
    "useVectorRetrieval" => true,
    "vectorInfo" => $result_vector_raw->vectorInfo,
    "mllmPredictInfo" => $result_mllm_raw->mllmPredictInfo
];
$response_chat_raw = send_post_raw("$API_BASE_URL/chatocr-chat", $payload_chat);
$response_chat = json_decode($response_chat_raw, true);
if (!isset($response_chat["result"])) {
    echo "chatocr-chat request error\n";
    print_r($response_chat);
    exit(1);
}

echo "Final result:\n";
echo json_encode($response_chat["result"]["chatResult"], JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT) . "\n";


function send_post_raw($url, $data) {
    $json_str = json_encode($data, JSON_UNESCAPED_UNICODE);
    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $json_str);
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    $response = curl_exec($ch);
    if ($response === false) {
        echo "cURL error: " . curl_error($ch) . "\n";
    }
    curl_close($ch);
    return $response;
}
?&gt;
</code></pre></details>
</details>
<br/>

📱 **On-Device Deployment**: Edge deployment is a method where computing and data processing functions are placed on the user's device itself. The device can directly process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed instructions on edge deployment, please refer to the [PaddleX On-Device Deployment Guide](../../../pipeline_deploy/on_device_deployment.md).
You can choose an appropriate deployment method for your pipeline based on your needs and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the Document Scene Information Extraction v4 Pipeline do not meet your expectations in terms of accuracy or speed in your specific scenario, you can try to further **fine-tune** the existing models using **data from your specific domain or application scenario** to enhance the recognition performance in your context.

### 4.1 Model Fine-Tuning
Since the Document Scene Information Extraction v4 Pipeline consists of several modules, suboptimal performance may stem from any of these modules. You can analyze cases with poor extraction results, identify which module is problematic through visual image inspection, and refer to the fine-tuning tutorial links in the table below for model fine-tuning.

<table>
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Module to Fine-Tune</th>
      <th>Fine-Tuning Reference Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Inaccurate layout area detection, such as missed detection of seals or tables</td>
      <td>Layout Area Detection Module</td>
      <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html">Link</a></td>
    </tr>
    <tr>
      <td>Inaccurate table structure recognition</td>
      <td>Table Structure Recognition</td>
      <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_structure_recognition.html">Link</a></td>
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
      <td>Inaccurate correction of overall image rotation</td>
      <td>Document Image Orientation Classification Module</td>
      <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html">Link</a></td>
    </tr>
    <tr>
      <td>Inaccurate correction of image distortion</td>
      <td>Text Image Rectification Module</td>
      <td>Fine-tuning Not Supported Yet</td>
    </tr>
  </tbody>
</table>

### 4.2 Model Deployment
After fine-tuning using your private dataset, you will obtain local model weights files.

To use the fine-tuned model weights, you only need to modify the pipeline configuration file by replacing the path to the default model weights with the path to your fine-tuned model weights in the corresponding location:

```yaml
......
SubModules:
    TextDetection:
    module_name: text_detection
    model_name: PP-OCRv5_server_det
    model_dir: null # Replace with the path to the fine-tuned text detection model weights
    limit_side_len: 960
    limit_type: max
    max_side_limit: 4000
    thresh: 0.3
    box_thresh: 0.6
    unclip_ratio: 1.5

    TextRecognition:
    module_name: text_recognition
    model_name: PP-OCRv5_server_rec
    model_dir: null # Replace with the path to the fine-tuned text recognition model weights
    batch_size: 1
    score_thresh: 0
......
```

Subsequently, refer to the command line method or Python script method in [2.2 Local Experience](#22-local-experience) to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU, allowing seamless switching between different hardware **by simply setting the `device` parameter**.

For example, when using the Document Scene Information Extraction v4 Pipeline, to change the running device from an NVIDIA GPU to an Ascend NPU, you only need to modify the `device` in the script to npu:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(
    pipeline="PP-ChatOCRv4-doc",
    device="npu:0" # gpu:0 --> npu:0
    )
```

If you want to use the General Document Scene Information Extraction v4 Pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
