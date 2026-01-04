---
comments: true
---

# Seal Text Recognition Pipeline Tutorial

## 1. Introduction to Seal Text Recognition Pipeline
Seal text recognition is a technology that automatically extracts and recognizes the content of seals from documents or images. The recognition of seal text is part of document processing and has many applications in various scenarios, such as contract comparison, warehouse entry and exit review, and invoice reimbursement review.

The seal text recognition pipeline is used to recognize the text content of seals, extracting the text information from seal images and outputting it in text form. This pipeline integrates the industry-renowned end-to-end OCR system PP-OCRv4, supporting the detection and recognition of curved seal text. Additionally, this pipeline integrates an optional layout region localization module, which can accurately locate the layout position of the seal within the entire document. It also includes optional document image orientation correction and distortion correction functions. Based on this pipeline, millisecond-level accurate text content prediction can be achieved on a CPU. This pipeline also provides flexible service deployment methods, supporting the use of multiple programming languages on various hardware. Moreover, it offers custom development capabilities, allowing you to train and fine-tune on your own dataset based on this pipeline, and the trained model can be seamlessly integrated.

<img src="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/PP-ChatOCRv3_doc_seal/01.png" style="width: 70%"/>
<b>The seal text recognition</b> pipeline includes a seal text detection module and a text recognition module, as well as optional layout detection module, document image orientation classification module, and text image correction module. Each module contains multiple models, and you can choose the model based on the benchmark test data below.

### 1.1 Model benchmark data

<b>If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, choose a model with faster inference speed. If you prioritize model storage size, choose a model with smaller storage size.</b>

> The inference time only includes the model inference time and does not include the time for pre- or post-processing.

<p><b>Layout Region Detection Module (Optional):</b></p>

* <b>Layout detection model, including 23 common categories: document title, paragraph title, text, page number, abstract, table of contents, references, footnotes, header, footer, algorithm, formula, formula number, image, chart title, table, table title, seal, chart title, chart, header image, footer image, sidebar text</b>
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


> ❗ The above list includes the <b>3 core models</b> that are key supported by the text recognition module. The module actually supports a total of <b>11 full models</b>, including several predefined models with different categories. The complete model list is as follows:

<details><summary> 👉Details of the Model List</summary>

* <b>3-class layout detection model, including table, image, seal</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
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
<td>A highly efficient layout region localization model based on the lightweight PicoDet-S model trained on a self-built dataset including Chinese and English papers, magazines, and research reports</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>12.80 / 9.57</td>
<td>45.04 / 23.86</td>
<td>22.6</td>
<td>An efficiency-accuracy balanced layout region localization model based on PicoDet-L trained on a self-built dataset including Chinese and English papers, magazines, and research reports</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>95.8</td>
<td>114.80 / 25.65</td>
<td>924.38 / 924.38</td>
<td>470.1</td>
<td>A high precision layout region localization model based on RT-DETR-H trained on a self-built dataset including Chinese and English papers, magazines, and research reports</td>
</tr>
</tbody></table>

* <b>17-class region detection model, including 17 common layout categories: paragraph title, image, text, number, abstract, content, chart title, formula, table, table title, references, document title, footnote, header, algorithm, footer, seal</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Regular Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
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
<td>A highly efficient layout region localization model based on the lightweight PicoDet-S model trained on a self-built dataset including Chinese and English papers, magazines, and research reports</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>12.60 / 10.27</td>
<td>43.70 / 24.42</td>
<td>22.6</td>
<td>An efficiency-accuracy balanced layout region localization model based on PicoDet-L trained on a self-built dataset including Chinese and English papers, magazines, and research reports</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>98.3</td>
<td>115.29 / 101.18</td>
<td>964.75 / 964.75</td>
<td>470.2</td>
<td>A high precision layout region localization model based on RT-DETR-H trained on a self-built dataset including Chinese and English papers, magazines, and research reports</td>
</tr>
</tbody>
</table>
</details>
</details>

<p><b>Document Image Orientation Classification Module (Optional):</b></p>
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
<td>A document image classification model based on PP-LCNet_x1_0, containing four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees</td>
</tr>
</tbody>
</table>
<p><b>Text Image Correction Module (Optional):</b></p>
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

<p><b>Text Detection Module:</b></p>
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
<td>PP-OCRv4 server-side seal text detection model, with higher accuracy, suitable for deployment on better servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Training Model</a></td>
<td>96.36</td>
<td>9.70 / 3.56</td>
<td>50.38 / 19.64</td>
<td>4.7</td>
<td>PP-OCRv4 mobile-side seal text detection model, with higher efficiency, suitable for deployment on the edge</td>
</tr>
</tbody>
</table>

<p><b>Text Recognition Module:</b></p>

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
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">Training Model</a></td>
<td>86.38</td>
<td>8.46 / 2.36</td>
<td>31.21 / 31.21</td>
<td>81</td>
<td rowspan="2">PP-OCRv5_rec is a next-generation text recognition model. This model is dedicated to efficiently and accurately supporting four major languages—Simplified Chinese, Traditional Chinese, English, and Japanese—with a single model. It supports complex text scenarios, including handwritten, vertical text, pinyin, and rare characters. While maintaining recognition accuracy, it also balances inference speed and model robustness, providing efficient and precise technical support for document understanding in various scenarios.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">Training Model</a></td>
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
<td>PP-OCRv4_server_rec_doc is trained on a mixed dataset of more Chinese document data and PP-OCR training data based on PP-OCRv4_server_rec. It has added the ability to recognize some traditional Chinese characters, Japanese, and special characters, and can support the recognition of more than 15,000 characters. In addition to improving the text recognition capability related to documents, it also enhances the general text recognition capability.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>78.74</td>
<td>5.26 / 1.12</td>
<td>17.48 / 3.61</td>
<td>10.5</td>
<td>
The lightweight recognition model of PP-OCRv4 has high inference efficiency and can be deployed on various hardware devices, including edge devices.</td>
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
<td>en_PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>70.39</td>
<td>4.81 / 1.23</td>
<td>17.20 / 4.18</td>
<td>7.5</td>
<td>The ultra-lightweight English recognition model, trained based on the PP-OCRv4 recognition model, supports the recognition of English letters and numbers.</td>
</tr>
</table>

> ❗ The above list features the <b>4 core models</b> that the text recognition module primarily supports. In total, this module supports <b>18 models</b>. The complete list of models is as follows:

<details><summary> 👉Model List Details</summary>

* <b>PP-OCRv5 Multi-Scenario Model</b>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Chinese Recognition Avg Accuracy (%)</th>
<th>English Recognition Avg Accuracy (%)</th>
<th>Traditional Chinese Recognition Avg Accuracy (%)</th>
<th>Japanese Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">Training Model</a></td>
<td>86.38</td>
<td>64.70</td>
<td>93.29</td>
<td>60.35</td>
<td>8.46 / 2.36</td>
<td>31.21 / 31.21</td>
<td>81</td>
<td rowspan="2">PP-OCRv5_rec is a next-generation text recognition model. This model efficiently and accurately supports four major languages with a single model: Simplified Chinese, Traditional Chinese, English, and Japanese. It recognizes complex text scenarios including handwritten, vertical text, pinyin, and rare characters. While maintaining recognition accuracy, it balances inference speed and model robustness, providing efficient and precise technical support for document understanding in various scenarios.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>81.29</td>
<td>66.00</td>
<td>83.55</td>
<td>54.65</td>
<td>5.43 / 1.46</td>
<td>21.20 / 5.32</td>
<td>16</td>
</tr>
</table>

* <b>Chinese Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
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
</details>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
            <li><strong>Test Dataset：</strong>
                        <ul>
                         <li>Document Image Orientation Classification Module: A self-built dataset using PaddleX, covering multiple scenarios such as ID cards and documents, containing 1000 images.</li>
                          <li>Text Image Rectification Model: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a></li>
                          <li>Layout Detection Model: A self-built layout detection dataset using PaddleOCR, including 500 images of common document types such as Chinese and English papers, magazines, contracts, books, exam papers, and research reports.</li>
                          <li>3-Category Layout Detection Model: A self-built layout detection dataset using PaddleOCR, containing 1154 images of common document types such as Chinese and English papers, magazines, and research reports.</li>
                          <li>17-Category Region Detection Model: A self-built layout detection dataset using PaddleOCR, including 892 images of common document types such as Chinese and English papers, magazines, and research reports.</li>
                          <li>Text Detection Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 500 images for detection.</li>
                          <li> Chinese Recognition Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 11,000 images for text recognition.</li>
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

### 1.2 Pipeline benchmark data

<details>
<summary>Click to expand/collapse the table</summary>

<table border="1">
<tr><th>Pipeline configuration</th><th>Hardware</th><th>Avg. inference time (s)</th><th>Peak CPU utilization (%)</th><th>Avg. CPU utilization (%)</th><th>Peak host memory (MB)</th><th>Avg. host memory (MB)</th><th>Peak GPU utilization (%)</th><th>Avg. GPU utilization (%)</th><th>Peak device memory (MB)</th><th>Avg. device memory (MB)</th></tr>
<tr>
<td rowspan="9">seal_recognition-default</td>
<td>Intel 6271C</td>
<td>1.44</td>
<td>1013.50</td>
<td>1005.77</td>
<td>2850.27</td>
<td>2593.21</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>1.30</td>
<td>1010.40</td>
<td>970.29</td>
<td>2797.28</td>
<td>2550.80</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.10</td>
<td>115.80</td>
<td>112.72</td>
<td>1647.39</td>
<td>1636.58</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.07</td>
<td>117.90</td>
<td>115.03</td>
<td>1519.09</td>
<td>1501.98</td>
<td>42</td>
<td>36.22</td>
<td>2508.00</td>
<td>2245.78</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.12</td>
<td>117.80</td>
<td>115.16</td>
<td>1425.80</td>
<td>1402.56</td>
<td>49</td>
<td>40.27</td>
<td>1862.00</td>
<td>1700.93</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.08</td>
<td>117.40</td>
<td>111.64</td>
<td>1721.26</td>
<td>1706.75</td>
<td>64</td>
<td>54.90</td>
<td>2502.00</td>
<td>2316.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.08</td>
<td>125.80</td>
<td>118.96</td>
<td>1727.16</td>
<td>1711.59</td>
<td>60</td>
<td>48.55</td>
<td>2212.00</td>
<td>2042.91</td>
</tr>
<tr>
<td>M4</td>
<td>1.54</td>
<td>104.70</td>
<td>101.57</td>
<td>4526.11</td>
<td>2082.03</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.16</td>
<td>110.20</td>
<td>108.01</td>
<td>1690.79</td>
<td>1667.58</td>
<td>82</td>
<td>70.95</td>
<td>1564.00</td>
<td>1397.90</td>
</tr>
<tr>
<td rowspan="9">seal_recognition-nopp</td>
<td>Intel 6271C</td>
<td>1.11</td>
<td>1014.70</td>
<td>1009.28</td>
<td>3062.18</td>
<td>2793.37</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>1.20</td>
<td>1012.60</td>
<td>1006.68</td>
<td>3079.39</td>
<td>2766.63</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.12</td>
<td>116.90</td>
<td>115.51</td>
<td>1608.80</td>
<td>1608.61</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.08</td>
<td>118.90</td>
<td>117.49</td>
<td>1440.84</td>
<td>1433.75</td>
<td>51</td>
<td>45.50</td>
<td>2586.00</td>
<td>2302.80</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.12</td>
<td>119.80</td>
<td>117.69</td>
<td>1451.77</td>
<td>1440.18</td>
<td>53</td>
<td>46.24</td>
<td>2078.00</td>
<td>1900.35</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.10</td>
<td>117.90</td>
<td>111.53</td>
<td>1722.02</td>
<td>1711.42</td>
<td>72</td>
<td>67.92</td>
<td>2438.00</td>
<td>2247.23</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.10</td>
<td>120.80</td>
<td>118.38</td>
<td>1642.00</td>
<td>1628.48</td>
<td>63</td>
<td>59.62</td>
<td>2430.00</td>
<td>2239.23</td>
</tr>
<tr>
<td>M4</td>
<td>2.34</td>
<td>104.30</td>
<td>101.47</td>
<td>4436.23</td>
<td>2145.62</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.20</td>
<td>109.80</td>
<td>107.97</td>
<td>1628.87</td>
<td>1614.33</td>
<td>82</td>
<td>76.56</td>
<td>1844.00</td>
<td>1662.80</td>
</tr>
<tr>
<td rowspan="9">seal_recognition-nopp-nolayout</td>
<td>Intel 6271C</td>
<td>0.66</td>
<td>1021.40</td>
<td>1014.05</td>
<td>2541.84</td>
<td>2317.52</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>0.77</td>
<td>1013.70</td>
<td>1009.71</td>
<td>2558.11</td>
<td>2321.48</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.10</td>
<td>113.90</td>
<td>112.78</td>
<td>1556.05</td>
<td>1555.78</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.06</td>
<td>122.90</td>
<td>120.53</td>
<td>1323.04</td>
<td>1314.25</td>
<td>51</td>
<td>44.89</td>
<td>1814.00</td>
<td>1814.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.10</td>
<td>123.80</td>
<td>121.95</td>
<td>1349.10</td>
<td>1344.00</td>
<td>55</td>
<td>48.31</td>
<td>1208.00</td>
<td>1208.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.10</td>
<td>116.90</td>
<td>113.28</td>
<td>1620.22</td>
<td>1601.19</td>
<td>74</td>
<td>65.77</td>
<td>1850.00</td>
<td>1850.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.08</td>
<td>123.50</td>
<td>120.70</td>
<td>1477.95</td>
<td>1465.51</td>
<td>68</td>
<td>56.82</td>
<td>1570.00</td>
<td>1570.00</td>
</tr>
<tr>
<td>M4</td>
<td>2.05</td>
<td>103.60</td>
<td>101.33</td>
<td>4115.47</td>
<td>2480.96</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.17</td>
<td>112.10</td>
<td>109.61</td>
<td>1477.02</td>
<td>1466.66</td>
<td>88</td>
<td>78.00</td>
<td>976.00</td>
<td>976.00</td>
</tr>
<tr>
<td rowspan="9">seal_recognition-nopp-lightweight</td>
<td>Intel 6271C</td>
<td>0.65</td>
<td>1015.50</td>
<td>1010.90</td>
<td>1701.61</td>
<td>1682.21</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>0.60</td>
<td>1011.70</td>
<td>1004.39</td>
<td>1794.69</td>
<td>1702.02</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.07</td>
<td>120.90</td>
<td>118.70</td>
<td>1601.37</td>
<td>1592.34</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.06</td>
<td>123.90</td>
<td>121.47</td>
<td>1483.47</td>
<td>1471.42</td>
<td>25</td>
<td>22.38</td>
<td>926.00</td>
<td>926.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.09</td>
<td>125.80</td>
<td>124.22</td>
<td>1410.70</td>
<td>1399.48</td>
<td>27</td>
<td>24.00</td>
<td>724.00</td>
<td>724.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.05</td>
<td>131.90</td>
<td>124.07</td>
<td>1820.23</td>
<td>1809.12</td>
<td>29</td>
<td>25.00</td>
<td>940.00</td>
<td>940.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.06</td>
<td>131.80</td>
<td>129.06</td>
<td>1668.08</td>
<td>1656.94</td>
<td>38</td>
<td>31.33</td>
<td>680.00</td>
<td>680.00</td>
</tr>
<tr>
<td>M4</td>
<td>0.70</td>
<td>106.60</td>
<td>104.71</td>
<td>1176.22</td>
<td>1155.02</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.09</td>
<td>120.80</td>
<td>117.79</td>
<td>1492.52</td>
<td>1481.60</td>
<td>49</td>
<td>44.25</td>
<td>490.00</td>
<td>490.00</td>
</tr>
<tr>
<td rowspan="9">seal_recognition-nopp-lightweightlayout</td>
<td>Intel 6271C</td>
<td>0.26</td>
<td>1018.90</td>
<td>1012.30</td>
<td>2759.20</td>
<td>2417.41</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>0.29</td>
<td>1017.60</td>
<td>1009.56</td>
<td>2718.50</td>
<td>2400.95</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.06</td>
<td>136.30</td>
<td>128.47</td>
<td>1565.15</td>
<td>1564.68</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.04</td>
<td>115.90</td>
<td>115.12</td>
<td>1313.98</td>
<td>1305.40</td>
<td>42</td>
<td>31.33</td>
<td>2896.00</td>
<td>2896.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.06</td>
<td>119.80</td>
<td>117.14</td>
<td>1268.04</td>
<td>1257.68</td>
<td>54</td>
<td>32.00</td>
<td>1956.00</td>
<td>1956.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.05</td>
<td>119.00</td>
<td>111.67</td>
<td>1735.01</td>
<td>1709.83</td>
<td>66</td>
<td>55.43</td>
<td>2394.00</td>
<td>2394.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.05</td>
<td>118.90</td>
<td>118.48</td>
<td>1596.27</td>
<td>1585.00</td>
<td>63</td>
<td>42.83</td>
<td>2376.00</td>
<td>2376.00</td>
</tr>
<tr>
<td>M4</td>
<td>0.79</td>
<td>112.40</td>
<td>102.80</td>
<td>6106.27</td>
<td>2249.03</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.08</td>
<td>111.90</td>
<td>108.66</td>
<td>1510.57</td>
<td>1502.12</td>
<td>83</td>
<td>53.82</td>
<td>1714.00</td>
<td>1714.00</td>
</tr>
<tr>
<td rowspan="9">seal_recognition-nopp-nolayout-lightweight</td>
<td>Intel 6271C</td>
<td>0.15</td>
<td>1027.50</td>
<td>1022.43</td>
<td>1245.17</td>
<td>1206.22</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>0.13</td>
<td>1024.70</td>
<td>1014.78</td>
<td>1211.13</td>
<td>1194.93</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.06</td>
<td>131.70</td>
<td>128.79</td>
<td>1532.17</td>
<td>1513.78</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.04</td>
<td>133.00</td>
<td>129.92</td>
<td>1169.25</td>
<td>1167.00</td>
<td>13</td>
<td>12.17</td>
<td>610.00</td>
<td>610.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.07</td>
<td>132.70</td>
<td>128.90</td>
<td>1269.38</td>
<td>1264.15</td>
<td>13</td>
<td>11.44</td>
<td>488.00</td>
<td>488.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.04</td>
<td>134.90</td>
<td>128.48</td>
<td>1530.35</td>
<td>1520.96</td>
<td>12</td>
<td>11.00</td>
<td>690.00</td>
<td>681.67</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.05</td>
<td>141.90</td>
<td>139.48</td>
<td>1458.07</td>
<td>1454.14</td>
<td>16</td>
<td>13.57</td>
<td>398.00</td>
<td>398.00</td>
</tr>
<tr>
<td>M4</td>
<td>0.28</td>
<td>110.30</td>
<td>109.19</td>
<td>1100.36</td>
<td>1097.53</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.07</td>
<td>125.90</td>
<td>123.15</td>
<td>1389.43</td>
<td>1379.54</td>
<td>32</td>
<td>30.44</td>
<td>254.00</td>
<td>254.00</td>
</tr>
</table>


<table border="1">
<tr><th>Pipeline configuration</th><th>description</th></tr>
<tr>
<td>seal_recognition-default</td>
<td>Default configuration</td>
</tr>
<tr>
<td>seal_recognition-nopp</td>
<td>Based on the default configuration, document image preprocessing is disabled</td>
</tr>
<tr>
<td>seal_recognition-nopp-nolayout</td>
<td>Based on the default configuration, document image preprocessing is disabled</td>
</tr>
<tr>
<td>seal_recognition-nopp-lightweight</td>
<td>Based on the default configuration, document image preprocessing is disabled, and lightweight models PP-OCRv4_mobile_seal_det and PP-OCRv4_mobile_rec are used</td>
</tr>
<tr>
<td>seal_recognition-nopp-lightweightlayout</td>
<td>Based on the default configuration, document image preprocessing is disabled, and the lightweight layout region detection model PP-DocLayout-S is used</td>
</tr>
<tr>
<td>seal_recognition-nopp-nolayout-lightweight</td>
<td>Based on the default configuration, document image preprocessing and layout region detection are disabled, and lightweight models PP-OCRv4_mobile_seal_det and PP-OCRv4_mobile_rec are used</td>
</tr>
</table>
</details>


* Test environment:
    * PaddlePaddle 3.1.0、CUDA 11.8、cuDNN 8.9
    * PaddleX @ develop (f1eb28e23cfa54ce3e9234d2e61fcb87c93cf407)
    * Docker image: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda11.8-cudnn8.9
* Test data:
    * Test data containing 130 seal images.
* Test strategy:
    * Warm up with 20 samples, then repeat the full dataset once for performance testing.
* Note:
    * Since we did not collect device memory data for NPU and XPU, the corresponding entries in the table are marked as N/A.

## 2. Quick Start
All model pipelines provided by PaddleX can be quickly experienced. You can experience the effect of the seal text recognition pipeline on the community platform, or you can use the command line or Python locally to experience the effect of the seal text recognition pipeline.

### 2.1 Online Experience
You can [experience the seal text recognition pipeline online](https://aistudio.baidu.com/community/app/387977/webUI?source=appCenter) by recognizing the demo images provided by the official platform, for example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/seal_recognition/seal_aistudio.png"/>

If you are satisfied with the performance of the pipeline, you can directly integrate and deploy it. You can choose to download the deployment package from the cloud, or refer to the methods in [Section 2.2 Local Experience](#22-local-experience) for local deployment. If you are not satisfied with the effect, you can <b>fine-tune the models in the pipeline using your private data</b>. If you have local hardware resources for training, you can start training directly on your local machine; if not, the Star River Zero-Code platform provides a one-click training service. You don't need to write any code—just upload your data and start the training task with one click.

### 2.2 Local Experience
> ❗ Before using the seal text recognition pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Installation Guide](../../../installation/installation.en.md). If you wish to selectively install dependencies, please refer to the relevant instructions in the installation guide. The dependency group corresponding to this pipeline is `ocr`.

#### 2.2.1 Command Line Experience
You can quickly experience the seal text recognition pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png), and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline seal_recognition \
    --input seal_text_det.png \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --device gpu:0 \
    --save_path ./output
```

<b>Note: </b>The official models would be download from HuggingFace by first. PaddleX also support to specify the preferred source by setting the environment variable `PADDLE_PDX_MODEL_SOURCE`. The supported values are `huggingface`, `aistudio`, `bos`, and `modelscope`. For example, to prioritize using `bos`, set: `PADDLE_PDX_MODEL_SOURCE="bos"`.

The relevant parameter descriptions can be referred to in the parameter explanations of [2.1.2 Integration via Python Script](#212-integration-via-python-script). Supports specifying multiple devices simultaneously for parallel inference. For details, please refer to the documentation on pipeline parallel inference.

After running, the results will be printed to the terminal, as follows:

<details><summary> 👉Click to Expand</summary>

```bash
{'res': {'input_path': 'seal_text_det.png', 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 16, 'label': 'seal', 'score': 0.975531280040741, 'coordinate': [6.195526, 0.1579895, 634.3982, 628.84595]}]}, 'seal_res_list': [{'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': [array([[320,  38],
       ...,
       [315,  38]]), array([[461, 347],
       ...,
       [456, 346]]), array([[439, 445],
       ...,
       [434, 444]]), array([[158, 468],
       ...,
       [154, 466]])], 'text_det_params': {'limit_side_len': 736, 'limit_type': 'min', 'thresh': 0.2, 'box_thresh': 0.6, 'unclip_ratio': 0.5}, 'text_type': 'seal', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0, 'rec_texts': ['天津君和缘商贸有限公司', '发票专用章', '吗繁物', '5263647368706'], 'rec_scores': array([0.9934051 , ..., 0.99139398]), 'rec_polys': [array([[320,  38],
       ...,
       [315,  38]]), array([[461, 347],
       ...,
       [456, 346]]), array([[439, 445],
       ...,
       [434, 444]]), array([[158, 468],
       ...,
       [154, 466]])], 'rec_boxes': array([], dtype=float64)}]}}
```

</details>

The explanation of the result parameters can be found in [2.1.2 Python Script Integration](#212-python-script-integration).

The visualized results are saved under `save_path`, and the visualized result of seal OCR is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/seal_recognition/03.png"/>

#### 2.2.2 Python Script Integration

* The above command line is for quickly experiencing and viewing the effect. Generally, in a project, you often need to integrate through code. You can complete the quick inference of the pipeline with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="seal_recognition")

output = pipeline.predict(
    "seal_text_det.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

In the above Python script, the following steps were executed:

(1) The seal recognition pipeline object was instantiated via `create_pipeline()`, with the specific parameters described as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is a pipeline name, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the pipeline (if set simultaneously with <code>pipeline</code>, it has higher priority than <code>pipeline</code>, and the pipeline name must be consistent with <code>pipeline</code>).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for pipeline inference. It supports specifying the specific card number of the GPU, such as "gpu:0", other hardware card numbers, such as "npu:0", or CPU, such as "cpu". Supports specifying multiple devices simultaneously for parallel inference. For details, please refer to <a href="../../instructions/parallel_inference.en.md#specifying-multiple-inference-devices">Pipeline Parallel Inference</a>.</td>
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
</tbody>
</table>

(2) Call the `predict()` method of the Seal Text Recognition pipeline object for inference prediction. This method will return a `generator`. Below are the parameters and their descriptions for the `predict()` method:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supports multiple input types (required)</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
<li><b>str</b>: Local path of an image or PDF file, e.g., <code>/root/data/img.jpg</code>; <b>URL link</b>, e.g., the network URL of an image or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png">Example</a>; <b>Local directory</b>, containing images to be predicted, e.g., <code>/root/data/</code> (currently does not support prediction of PDF files in directories; PDF files must be specified with an exact file path)</li>
<li><b>List</b>: Elements of the list must be of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
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
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, initialized as <code>True</code>.</li>
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
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, initialized as <code>True</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>Whether to use the layout detection module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, initialized as <code>True</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Confidence threshold for layout detection; only scores above this threshold will be output</td>
<td><code>float|dict|None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than <code>0</code></li>
<li><b>dict</b>: Key is the int category ID, value is any float greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, initialized as <code>0.5</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use Non-Maximum Suppression (NMS) for layout detection post-processing</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, initialized as <code>True</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Expansion ratio of detection box edges; if not specified, the default value from the PaddleX official model configuration will be used</td>
<td><code>float|list|None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than 0, e.g., 1.1, which means expanding the width and height of the detection box by 1.1 times while keeping the center unchanged</li>
<li><b>list</b>: e.g., [1.2, 1.5], which means expanding the width of the detection box by 1.2 times and the height by 1.5 times while keeping the center unchanged</li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, initialized as 1.0</li>
</ul>
</td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Merging mode for detection boxes in layout detection output; if not specified, the default value from the PaddleX official model configuration will be used</td>
<td><code>string|None</code></td>
<td>
<ul>
<li><b>large</b>: When set to <code>large</code>, only the largest external box will be retained for overlapping detection boxes, and the internal overlapping boxes will be removed.</li>
<li><b>small</b>: When set to <code>small</code>, only the smallest internal box will be retained for overlapping detection boxes, and the external overlapping boxes will be removed.</li>
<li><b>union</b>: No filtering of boxes will be performed; both internal and external boxes will be retained.</li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, initialized as <code>large</code>.</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Side length limit for seal text detection</td>
<td><code>int|None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, initialized as <code>736</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Text recognition threshold; text results with scores above this threshold will be retained</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any float greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, initialized as <code>0.0</code>. This means no threshold is applied.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</table>

(3) Process the prediction results. The prediction result for each sample is of `dict` type and supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data for better readability, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters, effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save results as a json file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save the results. When it is a directory, the saved file name will be consistent with the input file type</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data for better readability, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters, effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save results as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save the results, supports directory or file path</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal, and the explanations of the printed content are as follows:

    - `input_path`: `(str)` The input path of the image to be predicted.

    - `model_settings`: `(Dict[str, bool])` The model parameters required for pipeline configuration.

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing sub-pipeline.
        - `use_layout_detection`: `(bool)` Controls whether to enable the layout detection sub-module.

    - `layout_det_res`: `(Dict[str, Union[List[numpy.ndarray], List[float]]])` The output result of the layout detection sub-module. Only exists when `use_layout_detection=True`.

        - `input_path`: `(Union[str, None])` The image path accepted by the layout detection module. Saved as `None` when the input is a `numpy.ndarray`.
        - `page_index`: `(Union[int, None])` Indicates the current page number of the PDF if the input is a PDF file; otherwise, it is `None`.
        - `boxes`: `(List[Dict])` A list of detected layout seal regions, with each element containing the following fields:
            - `cls_id`: `(int)` The class ID of the detected seal region.
            - `score`: `(float)` The confidence score of the detected region.
            - `coordinate`: `(List[float])` The coordinates of the four corners of the detection box, in the order of x1, y1, x2, y2, representing the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner.

    - `seal_res_list`: `List[Dict]` A list of seal text recognition results, with each element containing the following fields:

        - `input_path`: `(Union[str, None])` The image path accepted by the seal text recognition pipeline. Saved as `None` when the input is a `numpy.ndarray`.
        - `page_index`: `(Union[int, None])` Indicates the current page number of the PDF if the input is a PDF file; otherwise, it is `None`.
        - `model_settings`: `(Dict[str, bool])` The model configuration parameters for the seal text recognition pipeline.
          - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing sub-pipeline.
          - `use_textline_orientation`: `(bool)` Controls whether to enable the text line orientation classification sub-module.

    - `doc_preprocessor_res`: `(Dict[str, Union[str, Dict[str, bool], int]])` The output result of the document preprocessing sub-pipeline. Only exists when `use_doc_preprocessor=True`.

        - `input_path`: `(Union[str, None])` The image path accepted by the document preprocessing sub-pipeline. Saved as `None` when the input is a `numpy.ndarray`.
        - `model_settings`: `(Dict)` The model configuration parameters for the preprocessing sub-pipeline.
            - `use_doc_orientation_classify`: `(bool)` Controls whether to enable document orientation classification.
            - `use_doc_unwarping`: `(bool)` Controls whether to enable document unwarping.
        - `angle`: `(int)` The predicted result of document orientation classification. When enabled, it takes values [0, 1, 2, 3], corresponding to [0°, 90°, 180°, 270°]; when disabled, it is -1.

    - `dt_polys`: `(List[numpy.ndarray])` A list of polygon boxes for seal text detection. Each detection box is represented by a numpy array of multiple vertex coordinates, with the array shape being (n, 2).

    - `dt_scores`: `(List[float])` A list of confidence scores for text detection boxes.

    - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the text detection module.
        - `limit_side_len`: `(int)` The side length limit value during image preprocessing.
        - `limit_type`: `(str)` The handling method for side length limits.
        - `thresh`: `(float)` The confidence threshold for text pixel classification.
        - `box_thresh`: `(float)` The confidence threshold for text detection boxes.
        - `unclip_ratio`: `(float)` The expansion ratio for text detection boxes.
        - `text_type`: `(str)` The type of seal text detection, currently fixed as "seal".

    - `text_rec_score_thresh`: `(float)` The filtering threshold for text recognition results.

    - `rec_texts`: `(List[str])` A list of text recognition results, containing only texts with confidence scores above `text_rec_score_thresh`.

    - `rec_scores`: `(List[float])` A list of confidence scores for text recognition, filtered by `text_rec_score_thresh`.

    - `rec_polys`: `(List[numpy.ndarray])` A list of text detection boxes filtered by confidence score, in the same format as `dt_polys`.

    - `rec_boxes`: `(numpy.ndarray)` An array of rectangular bounding boxes for detection boxes; the seal recognition pipeline returns an empty array.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, `numpy.array` types will be converted to list format.

- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_seal_res_region1.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (The pipeline usually contains multiple result images, so it is not recommended to specify a specific file path directly, as multiple images will be overwritten, and only the last image will be retained.)

* Additionally, you can obtain visualized images with results and prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction results in <code>json</code> format.</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get the visualization results in <code>dict</code> format.</td>
</tr>
</table>

- The prediction results obtained through the `json` attribute are of dict type, with content consistent with what is saved by calling the `save_to_json()` method.
- The prediction results returned by the `img` attribute are of dict type. The keys are `layout_det_res`, `seal_res_region1`, and `preprocessed_img`, corresponding to three `Image.Image` objects: one for visualizing layout detection, one for visualizing seal text recognition results, and one for visualizing image preprocessing. If the image preprocessing sub-module is not used, `preprocessed_img` will not be included in the dictionary. If the layout region detection module is not used, `layout_det_res` will not be included.

Additionally, you can obtain the configuration file for the seal text recognition pipeline and load the configuration file for prediction. You can execute the following command to save the results in `my_path`:

```
paddlex --get_pipeline_config seal_recognition --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the seal text recognition pipeline by simply modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file. The example is as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/seal_recognition.yaml")
output = pipeline.predict("seal_text_det.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存可视化结果
    res.save_to_json("./output/") ## 保存预测结果的json文件
```

<b>Note:</b> The parameters in the configuration file are the pipeline initialization parameters. If you wish to change the initialization parameters of the seal text recognition pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in a configuration file. Simply specify the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the pipeline into your Python project, you can refer to the example code in [2.2.2 Python Script Method](#222-python脚本方式集成).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Deployment</b>: In practical production environments, many applications have strict performance requirements (especially response speed) for deployment strategies to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin that aims to deeply optimize the performance of model inference and pre/post-processing, significantly speeding up the end-to-end process. For detailed high-performance deployment procedures, please refer to the [PaddleX High-Performance Deployment Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Serving Deployment</b>: Serving Deployment is a common form of deployment in practical production environments. By encapsulating inference capabilities as services, clients can access these services via network requests to obtain inference results. PaddleX supports various pipeline serving deployment solutions. For detailed pipeline serving deployment procedures, please refer to the [PaddleX Serving Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references for basic serving deployment and multi-language service invocation examples:

<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>The request body and response body are both JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
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
<th>Description</th>
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
<p>Obtain the seal text recognition result.</p>
<p><code>POST /seal-recognition</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>file</code></td>
<td><code>string</code></td>
<td>The URL of an image or PDF file accessible by the server, or the Base64-encoded content of the file. By default, for PDF files exceeding 10 pages, only the content of the first 10 pages will be processed.<br />
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
<td>The type of file. <code>0</code> indicates a PDF file, <code>1</code> indicates an image file. If this attribute is not present in the request body, the file type will be inferred from the URL.</td>
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
<td><code>useLayoutDetection</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_layout_detection</code> parameter of the pipeline object's <code>predict</code> method.</td>
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
<td><code>number</code> | <code>array</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_unclip_ratio</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_merge_bboxes_mode</code> parameter of the pipeline object's <code>predict</code> method.</td>
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
<li>When the request is processed successfully, the <code>result</code> in the response body has the following properties:</li>
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
<td><code>sealRecResults</code></td>
<td><code>object</code></td>
<td>The seal text recognition result. The array length is 1 (for image input) or the actual number of document pages processed (for PDF input). For PDF input, each element in the array represents the result of each page actually processed in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>sealRecResults</code> is an <code>object</code> with the following properties:</p>
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
<td>A simplified version of the <code>res</code> field in the JSON representation generated by the <code>predict</code> method of the production object, where the <code>input_path</code> and the <code>page_index</code> fields are removed.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>See the description of the <code>img</code> attribute of the result of the pipeline prediction. The images are in JPEG format and encoded in Base64.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>The input image. The image is in JPEG format and encoded in Base64.</td>
</tr>
</tbody>
</table></details>
<details><summary>Multi-language Service Invocation Example</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/seal-recognition"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["sealRecResults"]):
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

    auto response = client.Post("/seal-recognition", jsonObj.dump(), "application/json");

    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        if (!result.is_object() || !result["sealRecResults"].is_array()) {
            std::cerr << "Unexpected response format." << std::endl;
            return 1;
        }

        for (size_t i = 0; i < result["sealRecResults"].size(); ++i) {
            auto res = result["sealRecResults"][i];

            if (res.contains("prunedResult")) {
                std::cout << "Recognized seal result: " << res["prunedResult"].dump() << std::endl;
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
        String API_URL = "http://localhost:8080/seal-recognition";
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

                JsonNode sealRecResults = result.get("sealRecResults");
                for (int i = 0; i < sealRecResults.size(); i++) {
                    JsonNode item = sealRecResults.get(i);
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
)

func main() {
    API_URL := "http://localhost:8080/seal-recognition"
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

    resp, err := client.Do(req)
    if err != nil {
        fmt.Printf("Error sending request: %v\n", err)
        return
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        fmt.Printf("Unexpected status code: %d\n", resp.StatusCode)
        return
    }

    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        fmt.Printf("Error reading response body: %v\n", err)
        return
    }

    type SealResult struct {
        PrunedResult map[string]interface{}   `json:"prunedResult"`
        OutputImages map[string]string        `json:"outputImages"`
        InputImage   *string                  `json:"inputImage"`
    }

    type Response struct {
        Result struct {
            SealRecResults []SealResult  `json:"sealRecResults"`
            DataInfo       interface{}   `json:"dataInfo"`
        } `json:"result"`
    }

    var respData Response
    if err := json.Unmarshal(body, &respData); err != nil {
        fmt.Printf("Error unmarshaling response: %v\n", err)
        return
    }

    for i, res := range respData.Result.SealRecResults {
        fmt.Printf("Pruned Result %d: %+v\n", i, res.PrunedResult)

        for name, imgBase64 := range res.OutputImages {
            imgBytes, err := base64.StdEncoding.DecodeString(imgBase64)
            if err != nil {
                fmt.Printf("Error decoding image %s: %v\n", name, err)
                continue
            }

            filename := fmt.Sprintf("%s_%d.jpg", name, i)
            if err := ioutil.WriteFile(filename, imgBytes, 0644); err != nil {
                fmt.Printf("Error saving image %s: %v\n", filename, err)
                continue
            }
            fmt.Printf("Output image saved at %s\n", filename)
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
    static readonly string API_URL = "http://localhost:8080/seal-recognition";
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

        JArray sealRecResults = (JArray)jsonResponse["result"]["sealRecResults"];
        for (int i = 0; i < sealRecResults.Count; i++)
        {
            var res = sealRecResults[i];
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

const API_URL = 'http://localhost:8080/seal-recognition';
const imagePath = './demo.jpg';

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

const payload = {
  file: encodeImageToBase64(imagePath),
  fileType: 1
};

axios.post(API_URL, payload)
  .then((response) => {
    const result = response.data["result"];
    const sealRecResults = result["sealRecResults"];

    sealRecResults.forEach((res, i) => {
      console.log(`\n[${i}] prunedResult:\n`, res["prunedResult"]);

      const outputImages = res["outputImages"];
      if (outputImages) {
        for (const [imgName, base64Img] of Object.entries(outputImages)) {
          const imgBuffer = Buffer.from(base64Img, 'base64');
          const fileName = `${imgName}_${i}.jpg`;
          fs.writeFileSync(fileName, imgBuffer);
          console.log(`Output image saved at ${fileName}`);
        }
      } else {
        console.log(`[${i}] No outputImages found.`);
      }
    });
  })
  .catch((error) => {
    console.error('Error occurred while calling the API:', error.message);
  });
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/seal-recognition";
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

$result = json_decode($response, true)["result"]["sealRecResults"];

foreach ($result as $i => $item) {
    echo "[$i] prunedResult:\n";
    print_r($item["prunedResult"]);

    if (!empty($item["outputImages"])) {
        foreach ($item["outputImages"] as $img_name => $base64_img) {
            if (!empty($base64_img)) {
                $output_path = "{$img_name}_{$i}.jpg";
                file_put_contents($output_path, base64_decode($base64_img));
                echo "Output image saved at $output_path\n";
            }
        }
    } else {
        echo "No outputImages found for item $i\n";
    }
}
?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>On-Device Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on user devices, allowing devices to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX On-Device Deployment Guide](../../../pipeline_deploy/on_device_deployment.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model pipeline into subsequent AI applications.

## 4. Custom Development
If the default model weights provided by the seal text recognition pipeline do not meet your requirements in terms of accuracy or speed, you can try to <b>fine-tune</b> the existing models using <b>your own domain-specific or application data</b> to improve the recognition performance of the seal text recognition pipeline in your scenario.

### 4.1 Model Fine-Tuning
Since the seal text recognition pipeline consists of several modules, if the pipeline's performance does not meet expectations, the issue may arise from any one of these modules. You can analyze images with poor recognition results to identify which module is problematic and refer to the corresponding fine-tuning tutorial links in the table below for model fine-tuning.

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Fine-Tuning Module</th>
<th>Fine-Tuning Reference Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Inaccurate or missing seal position detection</td>
<td>Layout Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html">Link</a></td>
</tr>
<tr>
<td>Missing text detection</td>
<td>Text Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/seal_text_detection.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate text content</td>
<td>Text Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_recognition.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate full-image rotation correction</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate image distortion correction</td>
<td>Text Image Correction Module</td>
<td>Not supported for fine-tuning</td>
</tr>
</tbody>
</table>

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain the local model weight files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights in the corresponding position of the pipeline configuration file:

```python
......
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayout-L
    model_dir: null # 修改此处为微调后的版面检测模型权重的本地路径
    ...

SubPipelines:
  DocPreprocessor:
    ...
    SubModules:
      DocOrientationClassify:
        module_name: doc_text_orientation
        model_name: PP-LCNet_x1_0_doc_ori
        model_dir: null # 修改此处为微调后的文档图像方向分类模型权重的本地路径
    ...
    SubModules:
      TextDetection:
        module_name: seal_text_detection
        model_name: PP-OCRv4_server_seal_det
        model_dir: null # Modify this to the local path of the fine-tuned text detection model weights
        ...
        TextRecognition:
          module_name: text_recognition
          model_name: PP-OCRv5_server_rec
          model_dir: null # Modify this to the local path of the fine-tuned text recognition model weights
        ...
```

Then, refer to the command-line or Python script methods in [2.2 Local Experience](#2-quick-start) to load the modified pipeline configuration file.

## 5. Multi-Hardware Support

PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware devices.

For example, if you use Ascend NPU for inference on the seal text recognition pipeline, the Python command would be:

```bash
paddlex --pipeline seal_recognition \
    --input seal_text_det.png \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --device npu:0 \
    --save_path ./output
```

If you wish to use the seal text recognition pipeline on a wider variety of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
