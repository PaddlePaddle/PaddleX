---
comments: true
---

# General OCR Pipeline Tutorial

## 1. Introduction to the OCR pipeline
OCR (Optical Character Recognition) is a technology that converts text in images into editable text. It is widely used in document digitization, information extraction, and data processing. OCR can recognize printed text, handwritten text, and even certain types of fonts and symbols.

The General OCR pipeline is designed to solve text recognition tasks, extracting text information from images and outputting it in text form. This pipeline integrates the end-to-end OCR series systems, PP-OCRv5 and PP-OCRv4, supporting recognition of over 80 languages. Additionally, it includes functions for image orientation correction and distortion correction. Based on this pipeline, precise text content prediction at the millisecond level on CPUs can be achieved, covering a wide range of applications including general, manufacturing, finance, and transportation sectors. The pipeline also provides flexible deployment options, supporting calls in various programming languages on multiple hardware platforms. Moreover, it offers the capability for custom development, allowing you to train and optimize on your own dataset. The trained models can also be seamlessly integrated.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/ocr/01.png"/>

<b>The General OCR pipeline includes mandatory text detection and text recognition modules, as well as optional document image orientation classification, text image correction, and text line orientation classification modules.</b> The document image orientation classification and text image correction modules are integrated as a document preprocessing sub-line into the General OCR pipeline. Each module contains multiple models, and you can choose the model based on the benchmark test data below.

### 1.1 Model benchmark data

<b>If you prioritize model accuracy, choose a high-accuracy model; if you prioritize inference speed, choose a faster inference model; if you care about model storage size, choose a smaller model.</b>

> The inference time only includes the model inference time and does not include the time for pre- or post-processing.

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
<td>A document image classification model based on PP-LCNet_x1_0, with four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees.</td>
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
<p><b>Text Detection Module:</b></p>
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
<p><b>Text Recognition Module:</b></p>
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
<td>en_PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv5_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td> 85.25</td>
<td>-</td>
<td>-</td>
<td>7.5</td>
<td>The ultra-lightweight English recognition model trained based on the PP-OCRv5 recognition model supports the recognition of English and numbers.</td>
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
<td>korean_PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
korean_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv5_mobile_rec_pretrained.pdparams">Pre-trained Model</a></td>
<td>90.45</td>
<td>5.43 / 1.46</td>
<td>21.20 / 5.32</td>
<td>14</td>
<td>An ultra-lightweight Korean text recognition model trained based on the PP-OCRv5 recognition framework. Supports Korean, English and numeric text recognition.</td>
</tr>
<tr>
<td>latin_PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
latin_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv5_mobile_rec_pretrained.pdparams">Pre-trained Model</a></td>
<td>84.7</td>
<td>5.43 / 1.46</td>
<td>21.20 / 5.32</td>
<td>14</td>
<td>A Latin-script text recognition model trained based on the PP-OCRv5 recognition framework. Supports most Latin alphabet languages and numeric text recognition.</td>
</tr>
<tr>
<td>eslav_PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
eslav_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/eslav_PP-OCRv5_mobile_rec_pretrained.pdparams">Pre-trained Model</a></td>
<td>85.8</td>
<td>5.43 / 1.46</td>
<td>21.20 / 5.32</td>
<td>14</td>
<td>An East Slavic language recognition model trained based on the PP-OCRv5 recognition framework. Supports East Slavic languages, English and numeric text recognition.</td>
</tr>
<tr>
<td>th_PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
th_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/th_PP-OCRv5_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>82.68</td>
<td>-</td>
<td>-</td>
<td>7.5</td>
<td>The Thai recognition model trained based on the PP-OCRv5 recognition model supports recognition of Thai, English, and numbers.</td>
</tr>
<tr>
<td>el_PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
el_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/el_PP-OCRv5_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>89.28</td>
<td>-</td>
<td>-</td>
<td>7.5</td>
<td>The Greek recognition model trained based on the PP-OCRv5 recognition model supports recognition of Greek, English, and numbers.</td>
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
<td>
<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">Inference Model</a>/
<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x0_25_textline_ori_pretrained.pdparams">Training Model</a>
</td>
<td>98.85</td>
<td>2.16 / 0.41</td>
<td>2.37 / 0.73</td>
<td>0.96</td>
<td>A text line orientation classification model based on PP-LCNet_x0_25, with two categories: 0 degrees and 180 degrees.</td>
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
                          <li>Text Detection Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 500 images for detection.</li>
                          <li>Chinese Recognition Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 11,000 images for text recognition.</li>
                          <li>ch_SVTRv2_rec: Evaluation set A for "OCR End-to-End Recognition Task" in the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a></li>
                          <li>ch_RepSVTR_rec: Evaluation set B for "OCR End-to-End Recognition Task" in the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a>.</li>
                          <li>English Recognition Model: A self-built English dataset using PaddleX.</li>
                          <li>Multilingual Recognition Model: A self-built multilingual dataset using PaddleX.</li>
                          <li>Text Line Orientation Classification Model: A self-built dataset using PaddleX, covering various scenarios such as ID cards and documents, containing 1000 images.</li>
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
<td rowspan="7">OCR-default</td>
<td>Intel 6271C</td>
<td>3.97</td>
<td>1015.40</td>
<td>917.61</td>
<td>4381.22</td>
<td>3457.78</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>3.79</td>
<td>1022.50</td>
<td>921.68</td>
<td>4675.46</td>
<td>3585.96</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.65</td>
<td>113.50</td>
<td>102.48</td>
<td>2240.15</td>
<td>1868.44</td>
<td>47</td>
<td>19.60</td>
<td>7612.00</td>
<td>6634.15</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>1.06</td>
<td>114.90</td>
<td>103.05</td>
<td>2142.66</td>
<td>1791.43</td>
<td>72</td>
<td>20.01</td>
<td>5516.00</td>
<td>4812.81</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.65</td>
<td>108.90</td>
<td>101.95</td>
<td>2456.05</td>
<td>2080.26</td>
<td>100</td>
<td>36.52</td>
<td>6736.00</td>
<td>6017.05</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.74</td>
<td>115.90</td>
<td>102.22</td>
<td>2352.88</td>
<td>1993.39</td>
<td>100</td>
<td>25.56</td>
<td>6762.00</td>
<td>6039.93</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>1.17</td>
<td>107.10</td>
<td>101.78</td>
<td>2361.88</td>
<td>1986.61</td>
<td>100</td>
<td>51.11</td>
<td>5282.00</td>
<td>4585.10</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-mobile</td>
<td>Intel 6271C</td>
<td>1.39</td>
<td>1019.60</td>
<td>1007.69</td>
<td>2178.12</td>
<td>1873.73</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>1.15</td>
<td>1015.70</td>
<td>1006.87</td>
<td>2184.91</td>
<td>1916.85</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.35</td>
<td>110.80</td>
<td>103.77</td>
<td>2022.49</td>
<td>1808.11</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.27</td>
<td>110.90</td>
<td>103.80</td>
<td>1762.36</td>
<td>1525.04</td>
<td>31</td>
<td>19.30</td>
<td>4328.00</td>
<td>3356.30</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.55</td>
<td>113.80</td>
<td>103.68</td>
<td>1728.02</td>
<td>1470.52</td>
<td>38</td>
<td>18.59</td>
<td>4198.00</td>
<td>3199.12</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.22</td>
<td>111.90</td>
<td>103.99</td>
<td>2073.88</td>
<td>1876.14</td>
<td>32</td>
<td>20.25</td>
<td>4386.00</td>
<td>3435.86</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.31</td>
<td>119.90</td>
<td>104.24</td>
<td>2037.38</td>
<td>1771.06</td>
<td>52</td>
<td>32.74</td>
<td>3446.00</td>
<td>2733.21</td>
</tr>
<tr>
<td>M4</td>
<td>6.51</td>
<td>147.30</td>
<td>106.24</td>
<td>3550.58</td>
<td>3236.75</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.46</td>
<td>111.90</td>
<td>103.11</td>
<td>2035.38</td>
<td>1742.39</td>
<td>65</td>
<td>46.77</td>
<td>3968.00</td>
<td>2991.91</td>
</tr>
<tr>
<td rowspan="7">OCR-nopp-server</td>
<td>Intel 6271C</td>
<td>3.00</td>
<td>1016.00</td>
<td>1004.87</td>
<td>4445.46</td>
<td>3179.86</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>3.23</td>
<td>1010.70</td>
<td>1002.63</td>
<td>4175.39</td>
<td>3137.58</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.34</td>
<td>110.90</td>
<td>103.30</td>
<td>1904.99</td>
<td>1591.10</td>
<td>57</td>
<td>32.29</td>
<td>7494.00</td>
<td>6551.47</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.69</td>
<td>108.90</td>
<td>102.95</td>
<td>1808.30</td>
<td>1568.64</td>
<td>72</td>
<td>35.30</td>
<td>5410.00</td>
<td>4741.18</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.38</td>
<td>109.40</td>
<td>102.34</td>
<td>2100.00</td>
<td>1863.73</td>
<td>100</td>
<td>50.18</td>
<td>6614.00</td>
<td>5926.51</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.41</td>
<td>109.00</td>
<td>103.18</td>
<td>2055.21</td>
<td>1845.14</td>
<td>100</td>
<td>47.15</td>
<td>6654.00</td>
<td>5951.22</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.82</td>
<td>104.40</td>
<td>101.73</td>
<td>1906.88</td>
<td>1689.69</td>
<td>100</td>
<td>76.41</td>
<td>5178.00</td>
<td>4502.64</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-min736-mobile</td>
<td>Intel 6271C</td>
<td>1.41</td>
<td>1020.10</td>
<td>1008.14</td>
<td>2184.16</td>
<td>1911.86</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>1.20</td>
<td>1015.70</td>
<td>1007.08</td>
<td>2254.04</td>
<td>1935.18</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.36</td>
<td>112.90</td>
<td>104.29</td>
<td>2174.58</td>
<td>1827.67</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.27</td>
<td>113.90</td>
<td>104.48</td>
<td>1717.55</td>
<td>1529.77</td>
<td>30</td>
<td>19.54</td>
<td>4328.00</td>
<td>3388.44</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.57</td>
<td>118.80</td>
<td>104.45</td>
<td>1693.10</td>
<td>1470.74</td>
<td>40</td>
<td>19.83</td>
<td>4198.00</td>
<td>3206.91</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.22</td>
<td>113.40</td>
<td>104.66</td>
<td>2037.13</td>
<td>1797.10</td>
<td>31</td>
<td>20.64</td>
<td>4384.00</td>
<td>3427.91</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.31</td>
<td>119.30</td>
<td>106.05</td>
<td>1879.15</td>
<td>1732.39</td>
<td>49</td>
<td>30.40</td>
<td>3446.00</td>
<td>2751.08</td>
</tr>
<tr>
<td>M4</td>
<td>6.39</td>
<td>124.90</td>
<td>107.16</td>
<td>3578.98</td>
<td>3209.90</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.47</td>
<td>109.60</td>
<td>103.26</td>
<td>1961.40</td>
<td>1742.95</td>
<td>60</td>
<td>44.26</td>
<td>3968.00</td>
<td>3002.81</td>
</tr>
<tr>
<td rowspan="7">OCR-nopp-min736-server</td>
<td>Intel 6271C</td>
<td>3.26</td>
<td>1068.50</td>
<td>1004.96</td>
<td>4582.52</td>
<td>3135.68</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>3.52</td>
<td>1010.70</td>
<td>1002.33</td>
<td>4723.23</td>
<td>3209.27</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.35</td>
<td>108.90</td>
<td>103.94</td>
<td>1703.65</td>
<td>1485.50</td>
<td>60</td>
<td>35.54</td>
<td>7492.00</td>
<td>6576.97</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.71</td>
<td>110.80</td>
<td>103.54</td>
<td>1800.06</td>
<td>1559.28</td>
<td>78</td>
<td>36.65</td>
<td>5410.00</td>
<td>4741.55</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.40</td>
<td>110.20</td>
<td>102.75</td>
<td>2012.64</td>
<td>1843.45</td>
<td>100</td>
<td>55.74</td>
<td>6614.00</td>
<td>5940.44</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.44</td>
<td>114.90</td>
<td>103.87</td>
<td>2002.72</td>
<td>1773.17</td>
<td>100</td>
<td>49.28</td>
<td>6654.00</td>
<td>5980.68</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.89</td>
<td>105.00</td>
<td>101.91</td>
<td>2149.31</td>
<td>1795.35</td>
<td>100</td>
<td>76.39</td>
<td>5176.00</td>
<td>4528.77</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-max640-mobile</td>
<td>Intel 6271C</td>
<td>1.00</td>
<td>1033.70</td>
<td>1005.95</td>
<td>2021.88</td>
<td>1743.27</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>0.88</td>
<td>1043.60</td>
<td>1006.77</td>
<td>1980.82</td>
<td>1724.51</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.28</td>
<td>125.70</td>
<td>101.56</td>
<td>1962.27</td>
<td>1782.68</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.21</td>
<td>122.50</td>
<td>101.87</td>
<td>1772.39</td>
<td>1569.55</td>
<td>29</td>
<td>18.74</td>
<td>2360.00</td>
<td>2039.07</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.43</td>
<td>133.80</td>
<td>101.82</td>
<td>1636.93</td>
<td>1464.10</td>
<td>37</td>
<td>20.94</td>
<td>2386.00</td>
<td>2055.30</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.18</td>
<td>119.90</td>
<td>102.12</td>
<td>2119.93</td>
<td>1889.49</td>
<td>29</td>
<td>20.92</td>
<td>2636.00</td>
<td>2321.11</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.24</td>
<td>126.80</td>
<td>101.78</td>
<td>1905.14</td>
<td>1739.93</td>
<td>48</td>
<td>30.71</td>
<td>2232.00</td>
<td>1911.18</td>
</tr>
<tr>
<td>M4</td>
<td>7.08</td>
<td>137.80</td>
<td>104.83</td>
<td>2931.08</td>
<td>2658.25</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.36</td>
<td>124.80</td>
<td>101.70</td>
<td>1983.21</td>
<td>1729.43</td>
<td>61</td>
<td>46.10</td>
<td>2162.00</td>
<td>1836.63</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-max960-mobile</td>
<td>Intel 6271C</td>
<td>1.21</td>
<td>1020.00</td>
<td>1008.49</td>
<td>2200.30</td>
<td>1800.74</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>1.01</td>
<td>1024.10</td>
<td>1007.32</td>
<td>2038.80</td>
<td>1800.05</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.32</td>
<td>107.50</td>
<td>102.00</td>
<td>2001.21</td>
<td>1799.01</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.23</td>
<td>107.70</td>
<td>102.33</td>
<td>1727.89</td>
<td>1490.18</td>
<td>30</td>
<td>20.19</td>
<td>2646.00</td>
<td>2385.40</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.49</td>
<td>109.90</td>
<td>102.26</td>
<td>1726.01</td>
<td>1504.90</td>
<td>38</td>
<td>20.11</td>
<td>2498.00</td>
<td>2227.73</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.20</td>
<td>109.90</td>
<td>102.52</td>
<td>1959.46</td>
<td>1798.35</td>
<td>28</td>
<td>19.38</td>
<td>2712.00</td>
<td>2450.10</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.27</td>
<td>102.90</td>
<td>101.19</td>
<td>1938.48</td>
<td>1741.19</td>
<td>47</td>
<td>29.27</td>
<td>3344.00</td>
<td>2585.02</td>
</tr>
<tr>
<td>M4</td>
<td>5.44</td>
<td>122.10</td>
<td>105.91</td>
<td>3094.72</td>
<td>2686.52</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.41</td>
<td>106.00</td>
<td>101.81</td>
<td>1859.88</td>
<td>1722.62</td>
<td>68</td>
<td>47.05</td>
<td>2264.00</td>
<td>2001.07</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-max640-server</td>
<td>Intel 6271C</td>
<td>2.16</td>
<td>1026.30</td>
<td>1005.10</td>
<td>3467.93</td>
<td>3074.06</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>2.30</td>
<td>1008.70</td>
<td>1003.32</td>
<td>3435.54</td>
<td>3042.62</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.35</td>
<td>104.70</td>
<td>101.27</td>
<td>1948.85</td>
<td>1779.77</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.25</td>
<td>104.90</td>
<td>101.42</td>
<td>1833.93</td>
<td>1560.71</td>
<td>41</td>
<td>27.61</td>
<td>4480.00</td>
<td>3955.14</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.56</td>
<td>106.20</td>
<td>101.47</td>
<td>1669.73</td>
<td>1500.87</td>
<td>58</td>
<td>31.78</td>
<td>3160.00</td>
<td>2838.78</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.23</td>
<td>109.40</td>
<td>101.45</td>
<td>1968.77</td>
<td>1800.81</td>
<td>58</td>
<td>30.81</td>
<td>2602.00</td>
<td>2588.77</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.30</td>
<td>106.10</td>
<td>101.55</td>
<td>2027.13</td>
<td>1749.07</td>
<td>69</td>
<td>39.10</td>
<td>3318.00</td>
<td>2795.54</td>
</tr>
<tr>
<td>M4</td>
<td>7.26</td>
<td>133.90</td>
<td>104.48</td>
<td>5473.38</td>
<td>3472.28</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.58</td>
<td>103.90</td>
<td>100.86</td>
<td>1884.23</td>
<td>1714.48</td>
<td>84</td>
<td>63.50</td>
<td>2852.00</td>
<td>2540.37</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-max960-server</td>
<td>Intel 6271C</td>
<td>2.53</td>
<td>1014.50</td>
<td>1005.22</td>
<td>3625.57</td>
<td>3151.73</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>2.66</td>
<td>1010.60</td>
<td>1003.39</td>
<td>3580.64</td>
<td>3197.09</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.40</td>
<td>105.90</td>
<td>101.76</td>
<td>2040.65</td>
<td>1810.97</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.29</td>
<td>108.90</td>
<td>102.12</td>
<td>1821.03</td>
<td>1620.02</td>
<td>44</td>
<td>30.38</td>
<td>4290.00</td>
<td>2928.79</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.60</td>
<td>109.90</td>
<td>101.98</td>
<td>1797.75</td>
<td>1544.96</td>
<td>61</td>
<td>32.48</td>
<td>2936.00</td>
<td>2117.71</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.28</td>
<td>108.80</td>
<td>101.92</td>
<td>2016.22</td>
<td>1811.74</td>
<td>73</td>
<td>41.82</td>
<td>2636.00</td>
<td>2241.23</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.34</td>
<td>111.00</td>
<td>102.75</td>
<td>1964.21</td>
<td>1750.21</td>
<td>68</td>
<td>41.25</td>
<td>2722.00</td>
<td>2293.74</td>
</tr>
<tr>
<td>M4</td>
<td>6.28</td>
<td>129.10</td>
<td>103.74</td>
<td>7780.70</td>
<td>3571.92</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.67</td>
<td>116.90</td>
<td>101.33</td>
<td>1941.09</td>
<td>1693.39</td>
<td>88</td>
<td>65.48</td>
<td>2714.00</td>
<td>1923.06</td>
</tr>
<tr>
<td rowspan="7">OCR-nopp-min1280-server</td>
<td>Intel 6271C</td>
<td>4.13</td>
<td>1043.40</td>
<td>1005.45</td>
<td>5993.70</td>
<td>3454.00</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>4.46</td>
<td>1011.70</td>
<td>996.72</td>
<td>5633.51</td>
<td>3489.79</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.42</td>
<td>113.90</td>
<td>106.08</td>
<td>1747.88</td>
<td>1546.18</td>
<td>85</td>
<td>43.73</td>
<td>13558.00</td>
<td>11297.98</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.82</td>
<td>116.80</td>
<td>105.18</td>
<td>1873.38</td>
<td>1609.55</td>
<td>100</td>
<td>39.57</td>
<td>10376.00</td>
<td>8427.30</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.55</td>
<td>114.80</td>
<td>103.14</td>
<td>2036.36</td>
<td>1864.45</td>
<td>100</td>
<td>69.67</td>
<td>13224.00</td>
<td>11411.31</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.55</td>
<td>105.90</td>
<td>101.86</td>
<td>1931.35</td>
<td>1764.44</td>
<td>100</td>
<td>56.16</td>
<td>12418.00</td>
<td>10510.77</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>1.13</td>
<td>105.90</td>
<td>102.35</td>
<td>2066.73</td>
<td>1787.78</td>
<td>100</td>
<td>83.50</td>
<td>10142.00</td>
<td>8338.80</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-min1280-mobile</td>
<td>Intel 6271C</td>
<td>1.59</td>
<td>1019.90</td>
<td>1008.39</td>
<td>2366.86</td>
<td>1992.03</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>1.29</td>
<td>1017.70</td>
<td>1007.28</td>
<td>2501.24</td>
<td>2059.99</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.43</td>
<td>120.90</td>
<td>107.02</td>
<td>2108.87</td>
<td>1821.91</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.29</td>
<td>117.90</td>
<td>107.19</td>
<td>1847.97</td>
<td>1570.89</td>
<td>31</td>
<td>18.98</td>
<td>3746.00</td>
<td>3321.86</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.61</td>
<td>122.80</td>
<td>107.07</td>
<td>1789.25</td>
<td>1542.56</td>
<td>39</td>
<td>20.52</td>
<td>4058.00</td>
<td>3487.46</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.24</td>
<td>116.80</td>
<td>106.80</td>
<td>2092.63</td>
<td>1882.77</td>
<td>28</td>
<td>18.67</td>
<td>3902.00</td>
<td>3444.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.34</td>
<td>125.80</td>
<td>106.79</td>
<td>1959.45</td>
<td>1783.97</td>
<td>49</td>
<td>32.66</td>
<td>3532.00</td>
<td>3094.29</td>
</tr>
<tr>
<td>M4</td>
<td>6.64</td>
<td>139.40</td>
<td>107.63</td>
<td>4283.97</td>
<td>3112.59</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.51</td>
<td>116.90</td>
<td>105.06</td>
<td>1927.22</td>
<td>1675.34</td>
<td>68</td>
<td>45.78</td>
<td>3828.00</td>
<td>3283.78</td>
</tr>
</table>


<table border="1">
<tr><th>Pipeline configuration</th><th>description</th></tr>
<tr>
<td>OCR-default</td>
<td>Default configuration</td>
</tr>
<tr>
<td>OCR-nopp-mobile</td>
<td>Based on the default configuration, document image preprocessing is disabled and mobile det and rec models are used</td>
</tr>
<tr>
<td>OCR-nopp-server</td>
<td>Based on the default configuration, document image preprocessing is disabled</td>
</tr>
<tr>
<td>OCR-nopp-min736-mobile</td>
<td>Based on the default configuration, document image preprocessing is disabled, det model input resizing strategy is set to min+736, and mobile det and rec models are used</td>
</tr>
<tr>
<td>OCR-nopp-min736-server</td>
<td>Based on the default configuration, document image preprocessing is disabled, and the det model input resizing strategy is set to min+736</td>
</tr>
<tr>
<td>OCR-nopp-max640-mobile</td>
<td>Based on the default configuration, document image preprocessing is disabled, det model input resizing strategy is set to max+640, and mobile det and rec models are used</td>
</tr>
<tr>
<td>OCR-nopp-max960-mobile</td>
<td>Based on the default configuration, document image preprocessing is disabled, det model input resizing strategy is set to max+960, and mobile det and rec models are used</td>
</tr>
<tr>
<td>OCR-nopp-max640-server</td>
<td>Based on the default configuration, document image preprocessing is disabled, and the det model input resizing strategy is set to max+640</td>
</tr>
<tr>
<td>OCR-nopp-max960-server</td>
<td>Based on the default configuration, document image preprocessing is disabled, and the det model input resizing strategy is set to max+960</td>
</tr>
<tr>
<td>OCR-nopp-min1280-server</td>
<td>Based on the default configuration, document image preprocessing is disabled, and the det model input resizing strategy is set to min+1280</td>
</tr>
<tr>
<td>OCR-nopp-min1280-mobile</td>
<td>Based on the default configuration, document image preprocessing is disabled, det model input resizing strategy is set to min+1280, and mobile det and rec models are used</td>
</tr>
</table>
</details>


* Test environment:
    * PaddlePaddle 3.1.0、CUDA 11.8、cuDNN 8.9
    * PaddleX @ develop (f1eb28e23cfa54ce3e9234d2e61fcb87c93cf407)
    * Docker image: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda11.8-cudnn8.9
* Test data:
    * Test data containing 200 images from document and general scenarios.
* Test strategy:
    * Warm up with 20 samples, then repeat the full dataset once for performance testing.
* Note:
    * Since we did not collect device memory data for NPU and XPU, the corresponding entries in the table are marked as N/A.

## 2. Quick Start
All model pipelines provided by PaddleX can be quickly experienced. You can experience the effect of the general OCR pipeline on the community platform, or you can use the command line or Python locally to experience the effect of the general OCR pipeline.

### 2.1 Online Experience
You can [experience the general OCR pipeline online](https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent) by recognizing the demo images provided by the official platform, for example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/ocr/02.png"/>

If you are satisfied with the performance of the pipeline, you can directly integrate and deploy it. You can choose to download the deployment package from the cloud, or refer to the methods in [Section 2.2 Local Experience](#22-local-experience) for local deployment. If you are not satisfied with the effect, you can <b>fine-tune the models in the pipeline using your private data</b>. If you have local hardware resources for training, you can start training directly on your local machine; if not, the Star River Zero-Code platform provides a one-click training service. You don't need to write any code—just upload your data and start the training task with one click.

### 2.2 Local Experience
> ❗ Before using the general OCR pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Installation Guide](../../../installation/installation.en.md). If you wish to selectively install dependencies, please refer to the relevant instructions in the installation guide. The dependency group corresponding to this pipeline is `ocr`.

#### 2.2.1 Command Line Experience
* You can quickly experience the OCR pipeline with a single command. Use the [test image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png), and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline OCR \
        --input general_ocr_002.png \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --save_path ./output \
        --device gpu:0
```

<b>Note: </b>The official models would be download from HuggingFace by first. PaddleX also support to specify the preferred source by setting the environment variable `PADDLE_PDX_MODEL_SOURCE`. The supported values are `huggingface`, `aistudio`, `bos`, and `modelscope`. For example, to prioritize using `bos`, set: `PADDLE_PDX_MODEL_SOURCE="bos"`.

For details on the relevant parameter descriptions, please refer to the parameter descriptions in [2.2.2 Python Script Integration](#222-python-script-integration). Supports specifying multiple devices simultaneously for parallel inference. For details, please refer to the documentation on pipeline parallel inference.

After running, the results will be printed to the terminal as follows:

```bash
{'res': {'input_path': './general_ocr_002.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[  3,  10],
        ...,
        [  4,  30]],

       ...,

       [[ 99, 456],
        ...,
        [ 99, 479]]], dtype=int16), 'text_det_params': {'limit_side_len': 736, 'limit_type': 'min', 'thresh': 0.3, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 1.5}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['www.997700', '', 'Cm', '登机牌', 'BOARDING', 'PASS', 'CLASS', '序号SERIAL NO.', '座位号', 'SEAT NO.', '航班FLIGHT', '日期DATE', '舱位', '', 'W', '035', '12F', 'MU2379', '03DEc', '始发地', 'FROM', '登机口', 'GATE', '登机时间BDT', '目的地TO', '福州', 'TAIYUAN', 'G11', 'FUZHOU', '身份识别IDNO.', '姓名NAME', 'ZHANGQIWEI', '票号TKT NO.', '张祺伟', '票价FARE', 'ETKT7813699238489/1', '登机口于起飞前10分钟关闭 GATESCL0SE10MINUTESBEFOREDEPARTURETIME'], 'rec_scores': array([0.67634439, ..., 0.97416091]), 'rec_polys': array([[[  3,  10],
        ...,
        [  4,  30]],

       ...,

       [[ 99, 456],
        ...,
        [ 99, 479]]], dtype=int16), 'rec_boxes': array([[  3, ...,  30],
       ...,
       [ 99, ..., 479]], dtype=int16)}}
```
The explanation of the running result parameters can refer to the result interpretation in [2.2.2 Python Script Integration](#222-python-script-integration).

The visualized results are saved under `save_path`, and the OCR visualization results are as follows:
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/ocr/03.png"/>

#### 2.2.2 Python Script Integration
* The above command line is for quick experience and effect checking. Generally, in a project, integration through code is often required. You can complete the quick inference of the pipeline with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="OCR")

output = pipeline.predict(
    input="./general_ocr_002.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")

```

In the above Python script, the following steps are executed:

(1) The OCR pipeline object is instantiated via `create_pipeline()`, with specific parameter descriptions as follows:

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
<td>Specific configuration information for the pipeline (if set simultaneously with the <code>pipeline</code>, it takes precedence over the <code>pipeline</code>, and the pipeline name must match the <code>pipeline</code>).
</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for pipeline inference. It supports specifying specific GPU card numbers, such as "gpu:0", other hardware card numbers, such as "npu:0", or CPU, such as "cpu". Supports specifying multiple devices simultaneously for parallel inference. For details, please refer to <a href="../../instructions/parallel_inference.en.md#specifying-multiple-inference-devices">Pipeline Parallel Inference</a>.</td>
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

(2) The `predict()` method of the OCR pipeline object is called to perform inference. This method returns a `generator`. Below are the parameters and their descriptions for the `predict()` method:

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
<td>Data to be predicted, supports multiple input types (required).</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
<li><b>str</b>: Local path of an image file or PDF file, e.g., <code>/root/data/img.jpg</code>; <b>URL link</b>, e.g., network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png">example</a>; <b>local directory</b>, which must contain images to be predicted, e.g., <code>/root/data/</code> (prediction of PDF files in directories is currently not supported; PDF files must specify the exact file path)</li>
<li><b>List</b>: Elements of the list must be of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
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
<li><b>bool</b>: <code>True</code> or <code>False</code></li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, which is <code>True</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the document unwarping module.</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code></li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, which is <code>True</code></li>
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
<li><b>bool</b>: <code>True</code> or <code>False</code></li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, which is <code>True</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>The limit on the side length of the image for text detection.</td>
<td><code>int|None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, which is <code>64</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>The type of limit on the side length of the image for text detection.</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>. <code>min</code> ensures that the shortest side of the image is not less than <code>det_limit_side_len</code>, while <code>max</code> ensures that the longest side is not greater than <code>limit_side_len</code></li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, which is <code>min</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>The detection pixel threshold. Pixels with scores greater than this threshold in the output probability map will be considered as text pixels.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, which is <code>0.3</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>The detection box threshold. A detection result will be considered as a text region if the average score of all pixels within the bounding box is greater than this threshold.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, which is <code>0.6</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>The text detection expansion ratio. The larger this value, the larger the expanded area.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, which is <code>2.0</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>The text recognition score threshold. Text results with scores greater than this threshold will be retained.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, the default value from the pipeline initialization will be used, which is <code>0.0</code> (i.e., no threshold)</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</table>

(3) Process the prediction results. The prediction result for each sample is of type `dict`, and supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When a directory is specified, the saved file name will match the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>None</td>
</tr>
<tr>
<td><code>return_word_box</code></td>
<td>Whether to return the position coordinates of each character</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>：<code>True</code> 或者 <code>False</code>；</li>
<li><b>None</b>：If set to<code>None</code>, the default value initialized by the pipeline will be used, which is initialized as<code>False</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</table>

- Calling the `print()` method will print the result to the terminal. The printed content is explained as follows:

    - `input_path`: `(str)` The input path of the image to be predicted

    - `page_index`: `(Union[int, None])` If the input is a PDF file, this indicates the current page number of the PDF. Otherwise, it is `None`

    - `model_settings`: `(Dict[str, bool])` The model parameters required for the pipeline configuration

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing sub-line
        - `use_textline_orientation`: `(bool)` Controls whether to enable text line orientation classification

    - `doc_preprocessor_res`: `(Dict[str, Union[str, Dict[str, bool], int]])` The output result of the document preprocessing sub-line. This exists only when `use_doc_preprocessor=True`
        - `input_path`: `(Union[str, None])` The image path accepted by the preprocessing sub-line. When the input is `numpy.ndarray`, it is saved as `None`
        - `model_settings`: `(Dict)` The model configuration parameters for the preprocessing sub-line
            - `use_doc_orientation_classify`: `(bool)` Controls whether to enable document orientation classification
            - `use_doc_unwarping`: `(bool)` Controls whether to enable document unwarping
        - `angle`: `(int)` The prediction result of document orientation classification. When enabled, it takes values [0,1,2,3], corresponding to [0°,90°,180°,270°]; when disabled, it is -1

    - `dt_polys`: `(List[numpy.ndarray])` A list of polygon boxes for text detection. Each detection box is represented by a numpy array of 4 vertex coordinates, with a shape of (4, 2) and data type int16

    - `dt_scores`: `(List[float])` A list of confidence scores for text detection boxes

    - `text_det_params`: `(Dict[str, Dict[str, int, float]])` The configuration parameters for the text detection module
        - `limit_side_len`: `(int)` The side length limit value for image preprocessing
        - `limit_type`: `(str)` The processing method for side length limits
        - `thresh`: `(float)` The confidence threshold for text pixel classification
        - `box_thresh`: `(float)` The confidence threshold for text detection boxes
        - `unclip_ratio`: `(float)` The expansion ratio for text detection boxes
        - `text_type`: `(str)` The type of text detection, currently fixed as "general"

    - `textline_orientation_angles`: `(List[int])` The prediction results for text line orientation classification. When enabled, it returns actual angle values (e.g., [0,0,1]); when disabled, it returns [-1,-1,-1]

    - `text_rec_score_thresh`: `(float)` The filtering threshold for text recognition results

    - `rec_texts`: `(List[str])` A list of text recognition results, containing only texts with confidence scores above `text_rec_score_thresh`

    - `rec_scores`: `(List[float])` A list of confidence scores for text recognition, filtered by `text_rec_score_thresh`

    - `rec_polys`: `(List[numpy.ndarray])` A list of text detection boxes filtered by confidence score, in the same format as `dt_polys`

    - `rec_boxes`: `(numpy.ndarray)` An array of rectangular bounding boxes for detection boxes, with a shape of (n, 4) and dtype int16. Each row represents the [x_min, y_min, x_max, y_max] coordinates of a rectangle, where (x_min, y_min) is the top-left corner and (x_max, y_max) is the bottom-right corner

    - `text_word`: `(List[str])` When `return_word_box` is set to `True`, returns a list of the recognized text for each character.

    - `text_word_boxes`: `(List[numpy.ndarray])` When `return_word_box` is set to `True`, returns a list of bounding box coordinates for each recognized character.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, the `numpy.array` type will be converted to a list format.

- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_ocr_res_img.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (Since the pipeline usually contains multiple result images, it is not recommended to specify a specific file path directly, as multiple images will be overwritten and only the last image will be retained)

* Additionally, it also supports obtaining the visualization image with results and the prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction results in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get the visualization image in <code>dict</code> format</td>
</tr>
</table>

- The prediction results obtained through the `json` attribute are of type `dict`, and the content is consistent with the data saved by calling the `save_to_json()` method.
- The prediction results returned by the `img` attribute are of type `dict`. The keys are `ocr_res_img` and `preprocessed_img`, and the corresponding values are two `Image.Image` objects: one for displaying the visualization image of OCR results, and the other for showing the visualization image of image preprocessing. If the image preprocessing sub-module is not used, the dictionary will only contain `ocr_res_img`.

Additionally, you can obtain the OCR pipeline configuration file and load the configuration file for prediction. You can execute the following command to save the results in `my_path`:

```
paddlex --get_pipeline_config OCR --save_path ./my_path
```

If you have obtained the configuration file, you can customize the configurations of the OCR pipeline. You just need to modify the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file. The example is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/OCR.yaml")

output = pipeline.predict(
    input="./general_ocr_002.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

<b>Note:</b> The parameters in the configuration file are initialization parameters for the pipeline. If you want to change the general OCR pipeline initialization parameters, you can directly modify the parameters in the configuration file and load the configuration file for prediction. In addition, CLI prediction also supports passing in a configuration file, just specify the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment
If the general OCR pipeline meets your requirements for inference speed and accuracy, you can proceed with development integration/deployment directly.

If you need to apply the general OCR pipeline directly in your Python project, you can refer to the example code in [2.2.2 Python Script Method](#222-python-script-method).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have strict performance requirements for deployment strategies, especially response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin, which aims to deeply optimize the performance of model inference and pre/post-processing, significantly speeding up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).


 ☁️ <b>Serving</b>: Serving is a common deployment strategy in real-world production environments. By encapsulating inference functions into services, clients can access these services via network requests to obtain inference results. PaddleX supports various solutions for serving pipelines. For detailed pipeline serving procedures, please refer to the [PaddleX Pipeline Serving Guide](../../../pipeline_deploy/serving.en.md).

Below are the API reference and multi-language service invocation examples for the basic serving solution:

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
<p>Obtain OCR results from images.</p>
<p><code>POST /ocr</code></p>
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
<td>The URL of an image or PDF file accessible by the server, or the Base64-encoded content of the file. By default, for PDF files exceeding 10 pages, only the first 10 pages will be processed.<br />
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
<td>The type of the file. <code>0</code> for PDF files, <code>1</code> for image files. If this attribute is missing, the file type will be inferred from the URL.</td>
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
<li>When the request is processed successfully, the response body has the following properties for <code>result</code>:</li>
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
<td><code>ocrResults</code></td>
<td><code>object</code></td>
<td>OCR results. The array length is 1 (for image input) or the actual number of document pages processed (for PDF input). For PDF input, each element in the array represents the result of each page actually processed in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>ocrResults</code> is an <code>object</code> with the following properties:</p>
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
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>The simplified version of the <code>res</code> field in the JSON representation generated by the <code>predict</code> method of the production object, with the <code>input_path</code> and the <code>page_index</code> fields removed.</td>
</tr>
<tr>
<td><code>ocrImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>The OCR result image, which marks the detected text positions. The image is in JPEG format and encoded in Base64.</td>
</tr>
<tr>
<td><code>docPreprocessingImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>The visualization result image. The image is in JPEG format and encoded in Base64.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>The input image. The image is in JPEG format and encoded in Base64.</td>
</tr>
</tbody>
</table>
</details>
<details><summary>Multi-language Service Call Example</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/ocr"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["ocrResults"]):
    print(res["prunedResult"])
    ocr_img_path = f"ocr_{i}.jpg"
    with open(ocr_img_path, "wb") as f:
        f.write(base64.b64decode(res["ocrImage"]))
    print(f"Output image saved at {ocr_img_path}")
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
        std::cerr << "Error opening file." << std::endl;
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

    auto response = client.Post("/ocr", jsonObj.dump(), "application/json");

    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        if (!result.is_object() || !result["ocrResults"].is_array()) {
            std::cerr << "Unexpected response structure." << std::endl;
            return 1;
        }

        for (size_t i = 0; i < result["ocrResults"].size(); ++i) {
            auto ocrResult = result["ocrResults"][i];
            std::cout << ocrResult["prunedResult"] << std::endl;

            std::string ocrImgPath = "ocr_" + std::to_string(i) + ".jpg";
            std::string encodedImage = ocrResult["ocrImage"];
            std::string decodedImage = base64::from_base64(encodedImage);

            std::ofstream outputImage(ocrImgPath, std::ios::binary);
            if (outputImage.is_open()) {
                outputImage.write(decodedImage.c_str(), static_cast<std::streamsize>(decodedImage.size()));
                outputImage.close();
                std::cout << "Output image saved at " << ocrImgPath << std::endl;
            } else {
                std::cerr << "Unable to open file for writing: " << ocrImgPath << std::endl;
            }
        }
    } else {
        std::cerr << "Failed to send HTTP request." << std::endl;
        if (response) {
            std::cerr << "HTTP status code: " << response->status << std::endl;
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
        String API_URL = "http://localhost:8080/ocr";
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

                JsonNode ocrResults = result.get("ocrResults");
                for (int i = 0; i < ocrResults.size(); i++) {
                    JsonNode item = ocrResults.get(i);

                    JsonNode prunedResult = item.get("prunedResult");
                    System.out.println("Pruned Result [" + i + "]: " + prunedResult.toString());

                    String ocrImageBase64 = item.get("ocrImage").asText();
                    byte[] ocrImageBytes = Base64.getDecoder().decode(ocrImageBase64);
                    String ocrImgPath = "ocr_result_" + i + ".jpg";
                    try (FileOutputStream fos = new FileOutputStream(ocrImgPath)) {
                        fos.write(ocrImageBytes);
                        System.out.println("Saved OCR image to: " + ocrImgPath);
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
    API_URL := "http://localhost:8080/ocr"
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

    type OcrResult struct {
        PrunedResult map[string]interface{} `json:"prunedResult"`
        OcrImage     *string                `json:"ocrImage"`
    }

    type Response struct {
        Result struct {
            OcrResults []OcrResult `json:"ocrResults"`
            DataInfo   interface{} `json:"dataInfo"`
        } `json:"result"`
    }

    var respData Response
    if err := json.Unmarshal(body, &respData); err != nil {
        fmt.Printf("Error unmarshaling response: %v\n", err)
        return
    }

    for i, res := range respData.Result.OcrResults {

        if res.OcrImage != nil {
            imgBytes, err := base64.StdEncoding.DecodeString(*res.OcrImage)
            if err != nil {
                fmt.Printf("Error decoding image %d: %v\n", i, err)
                continue
            }

            filename := fmt.Sprintf("ocr_%d.jpg", i)
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
    static readonly string API_URL = "http://localhost:8080/ocr";
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

        JArray ocrResults = (JArray)jsonResponse["result"]["ocrResults"];
        for (int i = 0; i < ocrResults.Count; i++)
        {
            var res = ocrResults[i];
            Console.WriteLine($"[{i}] prunedResult:\n{res["prunedResult"]}");

            string base64Image = res["ocrImage"]?.ToString();
            if (!string.IsNullOrEmpty(base64Image))
            {
                string outputPath = $"ocr_{i}.jpg";
                byte[] imageBytes = Convert.FromBase64String(base64Image);
                File.WriteAllBytes(outputPath, imageBytes);
                Console.WriteLine($"OCR image saved to {outputPath}");
            }
            else
            {
                Console.WriteLine($"OCR image at index {i} is null.");
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

$API_URL = "http://localhost:8080/ocr";
$image_path = "./demo.jpg";

$image_data = base64_encode(file_get_contents($image_path));
$payload = array(
    "file" => $image_data,
    "fileType" => 1
);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)["result"]["ocrResults"];

foreach ($result as $i => $item) {
    echo "[$i] prunedResult:\n";
    print_r($item["prunedResult"]);

    if (!empty($item["ocrImage"])) {
        $output_img_path = "ocr_{$i}.jpg";
        file_put_contents($output_img_path, base64_decode($item["ocrImage"]));
        echo "OCR image saved at $output_img_path\n";
    } else {
        echo "No ocrImage found for item $i\n";
    }
}
?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>On-Device Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on user devices, allowing them to process data locally without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed instructions, please refer to the [PaddleX On-Device Deployment Guide](../../../pipeline_deploy/on_device_deployment.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model pipeline into your AI applications.

## 4. Custom Development
If the default model weights provided by the General OCR pipeline do not meet your requirements in terms of accuracy or speed, you can attempt to <b>fine-tune</b> the existing models using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the General OCR pipeline in your scenario.

### 4.1 Model Fine-Tuning
Since the General OCR pipeline consists of several modules, the unsatisfactory performance of the pipeline may originate from any one of these modules. You can analyze the images with poor recognition results to identify which module is problematic and refer to the corresponding fine-tuning tutorial links in the table below for model fine-tuning.

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
<td>Text is missed in detection</td>
<td>Text Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_detection.html">Link</a></td>
</tr>
<tr>
<td>Text content is inaccurate</td>
<td>Text Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_recognition.html">Link</a></td>
</tr>
<tr>
<td>Vertical or rotated text line correction is inaccurate</td>
<td>Text Line Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html">Link</a></td>
</tr>
<tr>
<td>Whole-image rotation correction is inaccurate</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html">Link</a></td>
</tr>
<tr>
<td>Image distortion correction is inaccurate</td>
<td>Text Image Correction Module</td>
<td>Fine-tuning not supported yet</td>
</tr>
</tbody>
</table>

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain the local model weight files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local paths of the fine-tuned model weights into the corresponding positions in the configuration file:

```yaml
SubPipelines:
  DocPreprocessor:
    ...
    SubModules:
      DocOrientationClassify:
        module_name: doc_text_orientation
        model_name: PP-LCNet_x1_0_doc_ori
        model_dir: null # Replace with the path to the fine-tuned document image orientation classification model weights.
    ...

SubModules:
  TextDetection:
    module_name: text_detection
    model_name: PP-OCRv5_mobile_det
    model_dir: null # Replace with the path to the fine-tuned text detection model weights.
    ...
  TextLineOrientation:
    module_name: textline_orientation
    model_name: PP-LCNet_x0_25_textline_ori
    model_dir: null  # Replace with the path to the fine-tuned textline orientation classification model weights.
    batch_size: 1
  TextRecognition:
    module_name: text_recognition
    model_name: PP-OCRv5_mobile_rec
    model_dir: null  # Replace with the path to the fine-tuned text recognition model weights.
    batch_size: 1
```

Subsequently, refer to the command-line or Python script methods in [2.2 Local Experience](#22-local-experience) to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPUs, Kunlunxin XPUs, Ascend NPUs, and Cambricon MLUs. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware devices.

For example, if you are using an NVIDIA GPU for OCR pipeline inference, the Python command is:

```bash
paddlex --pipeline OCR \
        --input general_ocr_002.png \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --save_path ./output \
        --device npu:0
```
Of course, you can also specify the hardware device when calling `create_pipeline()` or `predict()` in a Python script.

If you want to use the General OCR pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
</details>
