---
comments: true
---

# General Layout Parsing Pipeline Tutorial

## 1. Introduction to the General Layout Parsing Pipeline
Layout parsing is a technology that extracts structured information from document images, primarily used to convert complex document layouts into machine-readable data formats. This technology has extensive applications in document management, information extraction, and data digitization. By combining Optical Character Recognition (OCR), image processing, and machine learning algorithms, layout parsing can identify and extract text blocks, titles, paragraphs, images, tables, and other layout elements from documents. The process typically involves three main steps: layout analysis, element analysis, and data formatting, ultimately generating structured document data to improve data processing efficiency and accuracy.

The <b>General Layout Parsing Pipeline</b> includes modules for table structure recognition, layout region analysis, text detection, text recognition, formula recognition, seal text detection, text image rectification, and document image orientation classification.

<b>If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, choose a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.</b>
<details><summary> 👉Model List Details</summary>
<p><b>Table Structure Recognition Module Models</b>:</p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">Training Model</a></td>
<td>59.52</td>
<td>103.08 / 103.08</td>
<td>197.99 / 197.99</td>
<td>6.9 M</td>
<td>SLANet is a table structure recognition model developed by Baidu PaddleX Team. The model significantly improves the accuracy and inference speed of table structure recognition by adopting a CPU-friendly lightweight backbone network PP-LCNet, a high-low-level feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structural and positional information.</td>
</tr>
<tr>
<td>SLANet_plus</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_plus_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">Training Model</a></td>
<td>63.69</td>
<td>140.29 / 140.29</td>
<td>195.39 / 195.39</td>
<td>6.9 M</td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Training Model</a></td>
<td>90.4</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / -</td>
<td>123.76 M</td>
<td>A high-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using RT-DETR-L.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>13.3259 / 4.8685</td>
<td>44.0680 / 44.0680</td>
<td>22.578</td>
<td>A layout area localization model with balanced precision and efficiency, trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training Model</a></td>
<td>70.9</td>
<td>8.3008 / 2.3794</td>
<td>10.0623 / 9.9296</td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Training Model</a></td>
<td>97.5</td>
<td>8.02 / 3.09</td>
<td>23.70 / 20.41</td>
<td>7.4 M</td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>88.2</td>
<td>8.99 / 2.22</td>
<td>16.11 / 8.73</td>
<td>4.8</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>13.05 / 4.50</td>
<td>41.30 / 41.30</td>
<td>22.6</td>
<td>A balanced efficiency and precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>95.8</td>
<td>114.93 / 27.71</td>
<td>947.56 / 947.56</td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Training Model</a></td>
<td>97.8</td>
<td>9.03 / 3.10</td>
<td>25.82 / 20.70</td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>87.4</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>13.50 / 4.69</td>
<td>43.32 / 43.32</td>
<td>22.6</td>
<td>A balanced efficiency and precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>98.3</td>
<td>115.29 / 104.09</td>
<td>995.27 / 995.27</td>
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
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Training Model</a></td>
<td>82.69</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>109</td>
<td>PP-OCRv4's server-side text detection model, featuring higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>77.79</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>PP-OCRv4's mobile text detection model, optimized for efficiency, suitable for deployment on edge devices</td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>81.53</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>74.7 M</td>
<td>PP-OCRv4_server_rec_doc is trained on a mixed dataset of more Chinese document data and PP-OCR training data based on PP-OCRv4_server_rec. It has added the recognition capabilities for some traditional Chinese characters, Japanese, and special characters. The number of recognizable characters is over 15,000. In addition to the improvement in document-related text recognition, it also enhances the general text recognition capability.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>78.74</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td>The lightweight recognition model of PP-OCRv4 has high inference efficiency and can be deployed on various hardware devices, including edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training Model</a></td>
<td>80.61 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
<td>The server-side model of PP-OCRv4 offers high inference accuracy and can be deployed on various types of servers.</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>72.96</td>
<td>5.87 / 1.19</td>
<td>9.07 / 4.28</td>
<td>9.2 M</td>
<td>PP-OCRv3’s lightweight recognition model is designed for high inference efficiency and can be deployed on a variety of hardware devices, including edge devices.</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
<td>68.81</td>
<td>8.08 / 2.74</td>
<td>50.17 / 42.50</td>
<td>73.9 M</td>
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
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
<td>65.07</td>
<td>5.93 / 1.62</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td> 70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>6.8 M</td>
<td>The ultra-lightweight English recognition model trained based on the PP-OCRv4 recognition model supports the recognition of English and numbers.</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/en_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>70.69</td>
<td>5.44 / 0.75</td>
<td>8.65 / 5.57</td>
<td>7.8 M </td>
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
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/korean_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>60.21</td>
<td>5.40 / 0.97</td>
<td>9.11 / 4.05</td>
<td>8.6 M</td>
<td>The ultra-lightweight Korean recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Korean and numbers. </td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/japan_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>45.69</td>
<td>5.70 / 1.02</td>
<td>8.48 / 4.07</td>
<td>8.8 M </td>
<td>The ultra-lightweight Japanese recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Japanese and numbers.</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>82.06</td>
<td>5.90 / 1.28</td>
<td>9.28 / 4.34</td>
<td>9.7 M </td>
<td>The ultra-lightweight Traditional Chinese recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Traditional Chinese and numbers.</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/te_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>95.88</td>
<td>5.42 / 0.82</td>
<td>8.10 / 6.91</td>
<td>7.8 M </td>
<td>The ultra-lightweight Telugu recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Telugu and numbers.</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ka_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>96.96</td>
<td>5.25 / 0.79</td>
<td>9.09 / 3.86</td>
<td>8.0 M </td>
<td>The ultra-lightweight Kannada recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Kannada and numbers.</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ta_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>76.83</td>
<td>5.23 / 0.75</td>
<td>10.13 / 4.30</td>
<td>8.0 M </td>
<td>The ultra-lightweight Tamil recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Tamil and numbers.</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/latin_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>76.93</td>
<td>5.20 / 0.79</td>
<td>8.83 / 7.15</td>
<td>7.8 M</td>
<td>The ultra-lightweight Latin recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Latin script and numbers.</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/arabic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>73.55</td>
<td>5.35 / 0.79</td>
<td>8.80 / 4.56</td>
<td>7.8 M</td>
<td>The ultra-lightweight Arabic script recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Arabic script and numbers.</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>94.28</td>
<td>5.23 / 0.76</td>
<td>8.89 / 3.88</td>
<td>7.9 M  </td>
<td>
The ultra-lightweight cyrillic alphabet recognition model trained based on the PP-OCRv3 recognition model supports the recognition of cyrillic letters and numbers.</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td>96.44</td>
<td>5.22 / 0.79</td>
<td>8.56 / 4.06</td>
<td>7.9 M  </td>
<td>The ultra-lightweight Devanagari script recognition model trained based on the PP-OCRv3 recognition model supports the recognition of Devanagari script and numbers.</td>
</tr>
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
<th>Model Size</th>
</tr>
</thead>
<tbody>
<tr>
<td>LaTeX_OCR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Training Model</a></td>
<td>0.8821</td>
<td>0.0823</td>
<td>40.01</td>
<td>2047.13 / 2047.13</td>
<td>10582.73 / 10582.73</td>
<td>89.7 M</td>
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
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Training Model</a></td>
<td>98.21</td>
<td>74.75 / 67.72</td>
<td>382.55 / 382.55</td>
<td>109</td>
<td>PP-OCRv4's server-side seal text detection model, featuring higher accuracy, suitable for deployment on better-equipped servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Training Model</a></td>
<td>96.47</td>
<td>7.82 / 3.09</td>
<td>48.28 / 23.97</td>
<td>4.6</td>
<td>PP-OCRv4's mobile seal text detection model, offering higher efficiency, suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>

**Test Environment Description**:

- **Performance Test Environment**
  - **Test Dataset**:
    - Text Image Rectification Model: [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html).
    - Layout Region Detection Model: A self-built layout analysis dataset using PaddleOCR, containing 10,000 images of common document types such as Chinese and English papers, magazines, and research reports.
    - Table Structure Recognition Model: A self-built English table recognition dataset using PaddleX.
    - Text Detection Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 500 images for detection.
    - Chinese Recognition Model: A self-built Chinese dataset using PaddleOCR, covering multiple scenarios such as street scenes, web images, documents, and handwriting, with 11,000 images for text recognition.
    - ch_SVTRv2_rec: Evaluation set A for "OCR End-to-End Recognition Task" in the [PaddleOCR Algorithm Model Challenge](https://aistudio.baidu.com/competition/detail/1131/0/introduction).
    - ch_RepSVTR_rec: Evaluation set B for "OCR End-to-End Recognition Task" in the [PaddleOCR Algorithm Model Challenge](https://aistudio.baidu.com/competition/detail/1131/0/introduction).
    - English Recognition Model: A self-built English dataset using PaddleX.
    - Multilingual Recognition Model: A self-built multilingual dataset using PaddleX.
    - Text Line Orientation Classification Model: A self-built dataset using PaddleX, covering various scenarios such as ID cards and documents, containing 1000 images.
    - Seal Text Detection Model: A self-built dataset using PaddleX, containing 500 images of circular seal textures.
  - **Hardware Configuration**:
    - GPU: NVIDIA Tesla T4
    - CPU: Intel Xeon Gold 6271C @ 2.60GHz
    - Other Environments: Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2

- **Inference Mode Description**

| Mode        | GPU Configuration                        | CPU Configuration | Acceleration Technology Combination                   |
|-------------|----------------------------------------|-------------------|---------------------------------------------------|
| Normal Mode | FP32 Precision / No TRT Acceleration   | FP32 Precision / 8 Threads | PaddleInference                                 |
| High-Performance Mode | Optimal combination of pre-selected precision types and acceleration strategies | FP32 Precision / 8 Threads | Pre-selected optimal backend (Paddle/OpenVINO/TRT, etc.) |

</details>

## 2. Quick Start
The pipelines provided by PaddleX allow for quick experience of their effects. You can use the command line or Python to experience the effects of the General Layout Parsing pipeline locally.

Before using the General Layout Parsing pipeline locally, ensure you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Tutorial](../../../installation/installation.md).

### 2.1 Experience via Command Line
You can quickly experience the effects of the Layout Parsing pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_parsing_demo.png) and replace `--input` with the local path for prediction:

```
paddlex --pipeline layout_parsing \
        --input layout_parsing_demo.png \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --save_path ./output \
        --device gpu:0
```
For parameter descriptions, refer to the parameter explanations in [2.2.2 Integration via Python Script](#222-integration-via-python-script).

After running, the results will be printed to the terminal, as shown below:

<details><summary> 👉Click to expand</summary>
<pre><code>{'res': {'input_path': 'layout_parsing_demo.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_general_ocr': True, 'use_seal_recognition': True, 'use_table_recognition': True, 'use_formula_recognition': False}, 'parsing_res_list': [{'block_bbox': [133.37144, 40.12515, 1383.7618, 123.51433], 'block_label': 'text', 'block_content': '助力双方交往\n搭建友谊桥梁'}, {'block_bbox': [587.43024, 160.58405, 927.63995, 179.2846], 'block_label': 'figure_title', 'block_content': '本报记者沈小晓任彦黄培昭'}, {'block_bbox': [773.798, 200.63779, 1505.5233, 687.11847], 'block_label': 'image', 'block_content': ''}, {'block_bbox': [390.42462, 201.87276, 741.41675, 292.5969], 'block_label': 'text', 'block_content': '厄立特里亚高等教育与研究院合作建立，开\n设了中国语言课程和中国文化课程，注册学\n生2万余人次。10余年来，厄特孔院已成为\n当地民众了解中国的一扇窗口。'}, {'block_bbox': [9.70394, 202.7036, 359.6133, 340.30905], 'block_label': 'text', 'block_content': '身着中国传统民族服装的厄立特里亚青\n年依次登台表演中国民族舞、现代舞、扇子舞\n等，曼妙的舞姿赢得现场观众阵阵掌声。这\n是日前厄立特里亚高等教育与研究院孔子学\n院(以下简称"厄特孔院"举办“喜迎新年"中国\n歌舞比赛的场景。'}, {'block_bbox': [390.74887, 298.432, 740.7994, 436.79953], 'block_label': 'text', 'block_content': '黄鸣飞表示，随着来学习中文的人日益\n增多，阿斯马拉大学教学点已难以满足教学\n需要。2024年4月，由中企蜀道集团所属四\n川路桥承建的孔院教学楼项目在阿斯马拉开\n工建设，预计今年上半年峻工，建成后将为厄\n特孔院提供全新的办学场地。'}, {'block_bbox': [10.5880165, 346.2769, 359.125, 436.1819], 'block_label': 'text', 'block_content': '中国和厄立特里亚传统友谊深厚。近年\n来,在高质量共建“一带一路"框架下，中厄两\n国人文交流不断深化，互利合作的民意基础\n日益深厚。'}, {'block_bbox': [410.5304, 457.0797, 722.77606, 516.7847], 'block_label': 'text', 'block_content': '“在中国学习的经历\n让我看到更广阔的世界”'}, {'block_bbox': [30.340591, 457.54282, 341.95337, 516.82825], 'block_label': 'paragraph_title', 'block_content': '“学好中文，我们的\n未来不是梦"'}, {'block_bbox': [390.90765, 538.18097, 742.19904, 604.67365], 'block_label': 'text', 'block_content': '多年来，厄立特里亚广大赴华留学生和\n培训人员积极投身国家建设，成为助力该国\n发展的人才和厄中友好的见证者和推动者。'}, {'block_bbox': [9.953403, 538.3851, 359.45145, 652.02905], 'block_label': 'text', 'block_content': '“鲜花曾告诉我你怎样走过，大地知道你\n心中的每一个角落……"厄立特里亚阿斯马拉\n大学综合楼二层，一阵优美的歌声在走廊里回\n响。循着熟悉的旋律轻轻推开一间教室的门，\n学生们正跟着老师学唱中文歌曲《同一首歌》。'}, {'block_bbox': [390.89615, 610.6184, 741.1807, 747.9165], 'block_label': 'text', 'block_content': '在厄立特里亚全国妇女联盟工作的约翰\n娜·特韦尔德·凯莱塔就是其中一位。她曾在\n中华女子学院攻读硕士学位，研究方向是女\n性领导力与社会发展。其间，她实地走访中国\n多个地区，获得了观察中国社会发展的第一\n手资料。'}, {'block_bbox': [10.181939, 658.8049, 359.41302, 771.31146], 'block_label': 'text', 'block_content': '这是厄特孔院阿斯马拉大学教学点的一\n节中文歌曲课。为了让学生们更好地理解歌\n词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐\n字翻译和解释歌词。随着伴奏声响起，学生们\n边唱边随着节拍摇动身体，现场气氛热烈。'}, {'block_bbox': [809.68475, 705.4048, 1485.5435, 747.4364], 'block_label': 'figure_title', 'block_content': '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。\n中国驻厄立特里亚大使馆供图'}, {'block_bbox': [389.63492, 753.45245, 742.05634, 890.96497], 'block_label': 'text', 'block_content': '谈起在中国求学的经历，约翰娜记忆犹\n新：“中国的发展在当今世界是独一无二的。\n沿着中国特色社会主义道路坚定前行，中国\n创造了发展奇迹，这一切都离不开中国共产党\n的领导。中国的发展经验值得许多国家学习\n借鉴。”'}, {'block_bbox': [9.884867, 777.39636, 360.3998, 843.4287], 'block_label': 'text', 'block_content': '“这是中文歌曲初级班，共有32人。学\n生大部分来自首都阿斯马拉的中小学，年龄\n最小的仅有6岁。"尤斯拉告诉记者。'}, {'block_bbox': [9.801341, 850.1048, 359.61642, 1059.8444], 'block_label': 'text', 'block_content': '尤斯拉今年23岁，是厄立特里亚一所公立\n学校的艺术老师。她12岁开始在厄特孔院学\n习中文，在2017年第十届"汉语桥"世界中学生\n中文比赛中获得厄立特里亚赛区第一名，并和\n同伴代表厄立特里亚前往中国参加决赛，获得\n团体优胜奖。2022年起，尤斯拉开始在厄特孔\n院兼职教授中文歌曲，每周末两个课时。“中国\n文化博大精深，我希望我的学生们能够通过中\n文歌曲更好地理解中国文化。"她说。'}, {'block_bbox': [772.0007, 777.06, 1124.396, 1059.2354], 'block_label': 'text', 'block_content': '“不管远近都是客人，请不用客气；相约\n好了在一起，我们欢迎你…"在一场中厄青\n年联谊活动上，四川路桥中方员工同当地大\n学生合唱《北京欢迎你》。厄立特里亚技术学\n院计算机科学与工程专业学生鲁夫塔·谢拉\n是其中一名演唱者，她很早便在孔院学习中\n文，一直在为去中国留学作准备。“这句歌词\n是我们两国人民友谊的生动写照。无论是投\n身于厄立特里亚基础设施建设的中企员工，\n还是在中国留学的厄立特里亚学子，两国人\n民携手努力，必将推动两国关系不断向前发\n展。"鲁夫塔说。'}, {'block_bbox': [1155.9297, 777.71344, 1331.4728, 795.6411], 'block_label': 'text', 'block_content': '瓦的北红海省博物馆。'}, {'block_bbox': [1153.7091, 801.56256, 1504.5591, 987.63544], 'block_label': 'text', 'block_content': '博物馆二层陈列着一个发掘自阿杜利\n斯古城的中国古代陶制酒器，罐身上写着\n“万”“和"“禅"“山"等汉字。“这件文物证\n明，很早以前我们就通过海上丝绸之路进行\n贸易往来与文化交流。这也是厄立特里亚\n与中国友好交往历史的有力证明。"北红海\n省博物馆研究与文献部负责人伊萨亚斯·特\n斯法兹吉说。'}, {'block_bbox': [390.203, 897.60095, 742.03674, 1035.7938], 'block_label': 'text', 'block_content': '正在西南大学学习的厄立特里亚博士生\n穆卢盖塔·泽穆伊对中国怀有深厚感情。8\n盖塔在社交媒体上写下这样一段话：“这是我\n人生的重要一步，自此我拥有了一双坚固的\n鞋子，赋予我穿越荆棘的力量。"'}, {'block_bbox': [1154.4471, 993.4835, 1503.8441, 1107.7363], 'block_label': 'text', 'block_content': '厄立特里亚国家博物馆考古学和人类学\n研究员菲尔蒙·特韦尔德十分喜爱中国文\n化。他表示：“学习彼此的语言和文化，将帮\n助厄中两国人民更好地理解彼此，助力双方\n交往，搭建友谊桥梁。"'}, {'block_bbox': [391.17816, 1041.2622, 740.8725, 1131.4589], 'block_label': 'text', 'block_content': '穆卢盖塔密切关注中国在经济、科技、教\n育等领域的发展，“中国在科研等方面的实力\n与日俱增。在中国学习的经历让我看到更广\n阔的世界，从中受益匪浅。”'}, {'block_bbox': [9.486691, 1065.2955, 360.2089, 1180.0446], 'block_label': 'text', 'block_content': '“姐姐，你想去中国吗？"“非常想！我想\n去看故宫、爬长城。"尤斯拉的学生中有一对\n能歌善舞的姐妹，姐姐露娅今年15岁，妹妹\n莉娅14岁，两人都已在厄特孔院学习多年，\n中文说得格外流利。'}, {'block_bbox': [771.51514, 1065.1091, 1123.4568, 1179.5624], 'block_label': 'text', 'block_content': '厄立特里亚高等教育委员会主任助理萨\n马瑞表示：“每年我们都会组织学生到中国访\n问学习，目前有超过5000名厄立特里亚学生\n在中国留学。学习中国的教育经验，有助于\n提升厄立特里亚的教育水平。"'}, {'block_bbox': [1153.9272, 1114.0178, 1503.9585, 1347.0802], 'block_label': 'text', 'block_content': '厄立特里亚国家博物馆馆长塔吉丁·努\n里达姆·优素福曾多次访问中国，对中华文明\n的传承与创新、现代化博物馆的建设与发展\n印象深刻。“中国博物馆不仅有许多保存完好\n的文物，还充分运用先进科技手段进行展示，\n帮助人们更好理解中华文明。"塔吉丁说，“厄\n立特里亚与中国都拥有悠久的文明，始终相\n互理解、相互尊重。我希望未来与中国同行\n加强合作，共同向世界展示非洲和亚洲的灿\n烂文明。”'}, {'block_bbox': [390.8594, 1137.4973, 741.0567, 1346.7653], 'block_label': 'text', 'block_content': '23岁的莉迪亚·埃斯蒂法诺斯已在厄特\n孔院学习3年，在中国书法、中国画等方面表\n现十分优秀，在2024年厄立特里亚赛区的\n“汉语桥"比赛中获得一等奖。莉迪亚说：“学\n习中国书法让我的内心变得安宁和纯粹。我\n也喜欢中国的服饰，希望未来能去中国学习，\n把中国不同民族元素融入服装设计中，创作\n出更多精美作品，也把厄特文化分享给更多\n的中国朋友。”'}, {'block_bbox': [8.70449, 1186.1178, 359.8176, 1299.481], 'block_label': 'text', 'block_content': '露娅对记者说：“这些年来，怀着对中文\n和中国文化的热爱，我们姐妹俩始终相互鼓\n励，一起学习。我们的中文一天比一天好，还\n学会了中文歌和中国舞。我们一定要到中国\n去。学好中文，我们的未来不是梦！”'}, {'block_bbox': [9.666538, 1305.0905, 359.62704, 1347.939], 'block_label': 'text', 'block_content': '据厄特孔院中方院长黄鸣飞介绍，这所\n孔院成立于2013年3月，由贵州财经大学和'}, {'block_bbox': [791.9397, 1201.0502, 1104.4906, 1260.1833], 'block_label': 'text', 'block_content': '“共同向世界展示非\n洲和亚洲的灿烂文明”'}, {'block_bbox': [772.51917, 1281.01, 1123.4009, 1348.0028], 'block_label': 'text', 'block_content': '从阿斯马拉出发，沿着蜿蜓曲折的盘山\n公路一路向东寻找丝路印迹。驱车两个小\n时，记者来到位于厄立特里亚港口城市马萨'}], 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 1, 'label': 'image', 'score': 0.9853348731994629, 'coordinate': [773.798, 200.63779, 1505.5233, 687.11847]}, {'cls_id': 2, 'label': 'text', 'score': 0.9780634045600891, 'coordinate': [772.0007, 777.06, 1124.396, 1059.2354]}, {'cls_id': 2, 'label': 'text', 'score': 0.9771724343299866, 'coordinate': [1153.9272, 1114.0178, 1503.9585, 1347.0802]}, {'cls_id': 2, 'label': 'text', 'score': 0.9763692021369934, 'coordinate': [390.74887, 298.432, 740.7994, 436.79953]}, {'cls_id': 2, 'label': 'text', 'score': 0.9752321839332581, 'coordinate': [9.70394, 202.7036, 359.6133, 340.30905]}, {'cls_id': 2, 'label': 'text', 'score': 0.9751048684120178, 'coordinate': [1153.7091, 801.56256, 1504.5591, 987.63544]}, {'cls_id': 2, 'label': 'text', 'score': 0.9741119742393494, 'coordinate': [9.801341, 850.1048, 359.61642, 1059.8444]}, {'cls_id': 2, 'label': 'text', 'score': 0.9722761511802673, 'coordinate': [390.42462, 201.87276, 741.41675, 292.5969]}, {'cls_id': 2, 'label': 'text', 'score': 0.9718317985534668, 'coordinate': [390.8594, 1137.4973, 741.0567, 1346.7653]}, {'cls_id': 2, 'label': 'text', 'score': 0.9703624844551086, 'coordinate': [390.89615, 610.6184, 741.1807, 747.9165]}, {'cls_id': 2, 'label': 'text', 'score': 0.9677473306655884, 'coordinate': [8.70449, 1186.1178, 359.8176, 1299.481]}, {'cls_id': 2, 'label': 'text', 'score': 0.9674075841903687, 'coordinate': [390.203, 897.60095, 742.03674, 1035.7938]}, {'cls_id': 2, 'label': 'text', 'score': 0.9671176075935364, 'coordinate': [389.63492, 753.45245, 742.05634, 890.96497]}, {'cls_id': 2, 'label': 'text', 'score': 0.9656032919883728, 'coordinate': [10.5880165, 346.2769, 359.125, 436.1819]}, {'cls_id': 2, 'label': 'text', 'score': 0.9655402898788452, 'coordinate': [771.51514, 1065.1091, 1123.4568, 1179.5624]}, {'cls_id': 2, 'label': 'text', 'score': 0.96494060754776, 'coordinate': [1154.4471, 993.4835, 1503.8441, 1107.7363]}, {'cls_id': 2, 'label': 'text', 'score': 0.9630844593048096, 'coordinate': [772.51917, 1281.01, 1123.4009, 1348.0028]}, {'cls_id': 2, 'label': 'text', 'score': 0.9615732431411743, 'coordinate': [9.486691, 1065.2955, 360.2089, 1180.0446]}, {'cls_id': 2, 'label': 'text', 'score': 0.9598038792610168, 'coordinate': [10.181939, 658.8049, 359.41302, 771.31146]}, {'cls_id': 2, 'label': 'text', 'score': 0.9591749310493469, 'coordinate': [391.17816, 1041.2622, 740.8725, 1131.4589]}, {'cls_id': 2, 'label': 'text', 'score': 0.9563097953796387, 'coordinate': [9.953403, 538.3851, 359.45145, 652.02905]}, {'cls_id': 2, 'label': 'text', 'score': 0.95261549949646, 'coordinate': [390.90765, 538.18097, 742.19904, 604.67365]}, {'cls_id': 2, 'label': 'text', 'score': 0.9493226408958435, 'coordinate': [9.884867, 777.39636, 360.3998, 843.4287]}, {'cls_id': 2, 'label': 'text', 'score': 0.9399433135986328, 'coordinate': [9.666538, 1305.0905, 359.62704, 1347.939]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.9254537224769592, 'coordinate': [809.68475, 705.4048, 1485.5435, 747.4364]}, {'cls_id': 2, 'label': 'text', 'score': 0.9046457409858704, 'coordinate': [1155.9297, 777.71344, 1331.4728, 795.6411]}, {'cls_id': 2, 'label': 'text', 'score': 0.8674532771110535, 'coordinate': [410.5304, 457.0797, 722.77606, 516.7847]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.7949447631835938, 'coordinate': [30.340591, 457.54282, 341.95337, 516.82825]}, {'cls_id': 2, 'label': 'text', 'score': 0.7313820719718933, 'coordinate': [791.9397, 1201.0502, 1104.4906, 1260.1833]}, {'cls_id': 6, 'label': 'figure_title', 'score': 0.6073322892189026, 'coordinate': [587.43024, 160.58405, 927.63995, 179.2846]}, {'cls_id': 2, 'label': 'text', 'score': 0.5846534967422485, 'coordinate': [133.37144, 40.12515, 1383.7618, 123.51433]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': array([[[ 122,   28],
        ...,
        [ 122,  135]],

       ...,

       [[1156, 1330],
        ...,
        [1156, 1351]]], dtype=int16), 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0, 'rec_texts': ['助力双方交往', '搭建友谊桥梁', '西', '本报记者沈小晓任彦黄培昭', '身着中国传统民族服装的厄立特里亚青', '厄立特里亚高等教育与研究院合作建立，开', '年依次登台表演中国民族舞、现代舞、扇子舞', '设了中国语言课程和中国文化课程，注册学', '等，曼妙的舞姿赢得现场观众阵阵掌声。这', '生2万余人次。10余年来，厄特孔院已成为', '是日前厄立特里亚高等教育与研究院孔子学', '当地民众了解中国的一扇窗口。', '院(以下简称"厄特孔院"举办“喜迎新年"中国', '黄鸣飞表示，随着来学习中文的人日益', '歌舞比赛的场景。', '增多，阿斯马拉大学教学点已难以满足教学', '中国和厄立特里亚传统友谊深厚。近年', '需要。2024年4月，由中企蜀道集团所属四', '来,在高质量共建“一带一路"框架下，中厄两', '川路桥承建的孔院教学楼项目在阿斯马拉开', '国人文交流不断深化，互利合作的民意基础', '工建设，预计今年上半年峻工，建成后将为厄', '日益深厚。', '特孔院提供全新的办学场地。', '“学好中文，我们的', '“在中国学习的经历', '未来不是梦"', '让我看到更广阔的世界”', '“鲜花曾告诉我你怎样走过，大地知道你', '多年来，厄立特里亚广大赴华留学生和', '心中的每一个角落……"厄立特里亚阿斯马拉', '培训人员积极投身国家建设，成为助力该国', '大学综合楼二层，一阵优美的歌声在走廊里回', '发展的人才和厄中友好的见证者和推动者。', '响。循着熟悉的旋律轻轻推开一间教室的门，', '在厄立特里亚全国妇女联盟工作的约翰', '学生们正跟着老师学唱中文歌曲《同一首歌》。', '娜·特韦尔德·凯莱塔就是其中一位。她曾在', '这是厄特孔院阿斯马拉大学教学点的一', '中华女子学院攻读硕士学位，研究方向是女', '节中文歌曲课。为了让学生们更好地理解歌', '性领导力与社会发展。其间，她实地走访中国', '词大意，老师尤斯拉·穆罕默德萨尔·侯赛因逐', '多个地区，获得了观察中国社会发展的第一', '在厄立特里亚不久前举办的第六届中国风筝文化节上，当地小学生体验风筝制作。', '字翻译和解释歌词。随着伴奏声响起，学生们', '手资料。', '中国驻厄立特里亚大使馆供图', '边唱边随着节拍摇动身体，现场气氛热烈。', '谈起在中国求学的经历，约翰娜记忆犹', '“这是中文歌曲初级班，共有32人。学', '新：“中国的发展在当今世界是独一无二的。', '“不管远近都是客人，请不用客气；相约', '瓦的北红海省博物馆。', '生大部分来自首都阿斯马拉的中小学，年龄', '沿着中国特色社会主义道路坚定前行，中国', '好了在一起，我们欢迎你………"在一场中厄青', '博物馆二层陈列着一个发掘自阿杜利', '最小的仅有6岁。"尤斯拉告诉记者。', '创造了发展奇迹，这一切都离不开中国共产党', '年联谊活动上，四川路桥中方员工同当地大', '斯古城的中国古代陶制酒器，罐身上写着', '尤斯拉今年23岁，是厄立特里亚一所公立', '的领导。中国的发展经验值得许多国家学习', '学生合唱《北京欢迎你》。厄立特里亚技术学', '“万”“和”“禅”“山"等汉字。“这件文物证', '学校的艺术老师。她12岁开始在厄特孔院学', '借鉴。”', '院计算机科学与工程专业学生鲁夫塔·谢拉', '明，很早以前我们就通过海上丝绸之路进行', '习中文，在2017年第十届"汉语桥"世界中学生', '正在西南大学学习的厄立特里亚博士生', '是其中一名演唱者，她很早便在孔院学习中', '贸易往来与文化交流。这也是厄立特里亚', '中文比赛中获得厄立特里亚赛区第一名，并和', '穆卢盖塔·泽穆伊对中国怀有深厚感情。8', '文，一直在为去中国留学作准备。“这句歌词', '与中国友好交往历史的有力证明。"北红海', '同伴代表厄立特里亚前往中国参加决赛，获得', '是我们两国人民友谊的生动写照。无论是投', '省博物馆研究与文献部负责人伊萨亚斯·特', '团体优胜奖。2022年起，尤斯拉开始在厄特孔', '盖塔在社交媒体上写下这样一段话：“这是我', '身于厄立特里亚基础设施建设的中企员工，', '斯法兹吉说。', '院兼职教授中文歌曲，每周末两个课时。“中国', '人生的重要一步，自此我拥有了一双坚固的', '还是在中国留学的厄立特里亚学子，两国人', '厄立特里亚国家博物馆考古学和人类学', '文化博大精深，我希望我的学生们能够通过中', '鞋子，赋予我穿越荆棘的力量。"', '民携手努力，必将推动两国关系不断向前发', '研究员菲尔蒙·特韦尔德十分喜爱中国文', '文歌曲更好地理解中国文化。"她说。', '穆卢盖塔密切关注中国在经济、科技、教', '展。"鲁夫塔说。', '化。他表示：“学习彼此的语言和文化，将帮', '“姐姐，你想去中国吗？"“非常想！我想', '育等领域的发展，“中国在科研等方面的实力', '厄立特里亚高等教育委员会主任助理萨', '助厄中两国人民更好地理解彼此，助力双方', '去看故宫、爬长城。"尤斯拉的学生中有一对', '与日俱增。在中国学习的经历让我看到更广', '马瑞表示：“每年我们都会组织学生到中国访', '交往，搭建友谊桥梁。"', '能歌善舞的姐妹，姐姐露娅今年15岁，妹妹', '阔的世界，从中受益匪浅。”', '问学习，目前有超过5000名厄立特里亚学生', '厄立特里亚国家博物馆馆长塔吉丁·努', '莉娅14岁，两人都已在厄特孔院学习多年，', '23岁的莉迪亚·埃斯蒂法诺斯已在厄特', '在中国留学。学习中国的教育经验，有助于', '里达姆·优素福曾多次访问中国，对中华文明', '中文说得格外流利。', '孔院学习3年，在中国书法、中国画等方面表', '提升厄立特里亚的教育水平。"', '的传承与创新、现代化博物馆的建设与发展', '露娅对记者说：“这些年来，怀着对中文', '现十分优秀，在2024年厄立特里亚赛区的', '印象深刻。“中国博物馆不仅有许多保存完好', '“共同向世界展示非', '和中国文化的热爱，我们姐妹俩始终相互鼓', '“汉语桥"比赛中获得一等奖。莉迪亚说：“学', '的文物，还充分运用先进科技手段进行展示，', '励，一起学习。我们的中文一天比一天好，还', '习中国书法让我的内心变得安宁和纯粹。我', '洲和亚洲的灿烂文明”', '帮助人们更好理解中华文明。"塔吉丁说，“厄', '学会了中文歌和中国舞。我们一定要到中国', '也喜欢中国的服饰，希望未来能去中国学习，', '立特里亚与中国都拥有悠久的文明，始终相', '去。学好中文，我们的未来不是梦！”', '把中国不同民族元素融入服装设计中，创作', '从阿斯马拉出发，沿着蜿蜓曲折的盘山', '互理解、相互尊重。我希望未来与中国同行', '据厄特孔院中方院长黄鸣飞介绍，这所', '出更多精美作品，也把厄特文化分享给更多', '公路一路向东寻找丝路印迹。驱车两个小', '加强合作，共同向世界展示非洲和亚洲的灿', '孔院成立于2013年3月，由贵州财经大学和', '的中国朋友。”', '时，记者来到位于厄立特里亚港口城市马萨', '烂文明。”'], 'rec_scores': array([0.99982363, ..., 0.93620157]), 'rec_polys': array([[[ 122,   28],
        ...,
        [ 122,  135]],

       ...,

       [[1156, 1330],
        ...,
        [1156, 1351]]], dtype=int16), 'rec_boxes': array([[ 122, ...,  135],
       ...,
       [1156, ..., 1351]], dtype=int16)}}}
</code></pre></details>

### 2.2 Integrating via Python Script
A few lines of code suffice for rapid inference on the pipeline, taking the general layout parsing pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="layout_parsing")

output = pipeline.predict(
    input="./layout_parsing_demo.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print()  ## Print the structured output of the prediction
    res.save_to_img(save_path="./output/")  ## Save the visualized image results of all submodules for the current image
    res.save_to_json(save_path="./output/")  ## Save the structured JSON results for the current image
    res.save_to_xlsx(save_path="./output/")  ## Save the sub-table results in XLSX format for the current image
    res.save_to_html(save_path="./output/")  ## Save the sub-table results in HTML format for the current image
```

In the above Python script, the following steps are executed:

（1）Instantiate the `create_pipeline` to create a pipeline object: The specific parameter descriptions are as follows:

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
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the pipeline name, it must be a pipeline supported by PaddleX.</td>
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
<td>The inference device for the pipeline. Supports specifying the specific card number for GPUs, such as "gpu:0", specific card numbers for other hardware, such as "npu:0", and CPUs as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available if the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

### (2) Invoke the `predict()` method of the Layout Analysis Pipeline object for inference prediction. This method will return a `generator`. Below are the parameters of the `predict()` method and their descriptions:

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
<td>The data to be predicted, supporting multiple input types, required</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Such as <code>numpy.ndarray</code> representing image data</li>
<li><b>str</b>: Such as the local path of an image file or PDF file: <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">Example</a>; <b>Local directory</b>, which should contain images to be predicted, such as the local path: <code>/root/data/</code> (currently does not support prediction of PDF files within directories, PDF files need to be specified to the specific file path)</li>
<li><b>List</b>: List elements need to be of the above types of data, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The inference device for the pipeline</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>CPU</b>: Such as <code>cpu</code> to use CPU for inference;</li>
<li><b>GPU</b>: Such as <code>gpu:0</code> to use the first GPU for inference;</li>
<li><b>NPU</b>: Such as <code>npu:0</code> to use the first NPU for inference;</li>
<li><b>XPU</b>: Such as <code>xpu:0</code> to use the first XPU for inference;</li>
<li><b>MLU</b>: Such as <code>mlu:0</code> to use the first MLU for inference;</li>
<li><b>DCU</b>: Such as <code>dcu:0</code> to use the first DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline. During initialization, it will prioritize using the local GPU 0 device, and if not available, it will use the CPU device;</li>
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
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the document distortion correction module</td>
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
<td>Whether to use the text line orientation classification module</td>
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
<td>Whether to use the OCR sub-pipeline</td>
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
<td>Whether to use the seal recognition sub-pipeline</td>
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
<td>Whether to use the table recognition sub-pipeline</td>
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
<td><code>use_formula_recognition</code></td>
<td>Whether to use the formula recognition sub-pipeline</td>
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
<td>Score threshold for the layout model</td>
<td><code>float|dict|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number between <code>0-1</code>;</li>
<li><b>dict</b>: <code>{0:0.1}</code> where the key is the class ID and the value is the threshold for that class;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>0.5</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS post-processing for the layout region detection model</td>
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
<td>Expansion coefficient for the detection box of the layout region detection model</td>
<td><code>float|Tuple[float,float]|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>;</li>
<li><b>Tuple[float,float]</b>: Expansion coefficients in the horizontal and vertical directions respectively;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>1.0</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Overlapping box filtering method for layout region detection</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: <code>large</code>, <code>small</code>, <code>union</code>, indicating whether to keep the larger box, smaller box, or both when overlapping boxes are filtered</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>large</code>;</li>
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
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>960</code>;</li>
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
<li><b>str</b>: Supports <code>min</code> and <code>max</code>, where <code>min</code> ensures that the shortest side of the image is not less than <code>det_limit_side_len</code>, and <code>max</code> ensures that the longest side of the image is not greater than <code>limit_side_len</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>max</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>Detection pixel threshold, where pixels with scores greater than this threshold in the output probability map are considered text pixels</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>0.3</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>Detection box threshold, where detection results with an average score of all pixels within the border greater than this threshold are considered text regions</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>0.6</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>Text detection expansion coefficient, which expands the text region. The larger this value, the larger the expansion area</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>2.0</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>Text recognition threshold, where text results with scores greater than this threshold are retained</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>0.0</code>. I.e., no threshold is set</li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_limit_side_len</code></td>
<td>Image side length limit for seal detection</td>
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
<td>Image side length limit type for seal detection</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>, where <code>min</code> ensures that the shortest side of the image is not less than <code>det_limit_side_len</code>, and <code>max</code> ensures that the longest side of the image is not greater than <code>limit_side_len</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>max</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_thresh</code></td>
<td>Detection pixel threshold, where pixels with scores greater than this threshold in the output probability map are considered seal pixels</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>0.3</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>Detection box threshold, where detection results with an average score of all pixels within the border greater than this threshold are considered seal regions</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>0.6</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>Seal detection expansion coefficient, which expands the seal region. The larger this value, the larger the expansion area</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>2.0</code></li></li></ul></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>Seal recognition threshold, where text results with scores greater than this threshold are retained</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>
    <li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>0.0</code>. I.e., no threshold is set</li></li></ul></td>
<td><code>None</code></td>
</tr>
</table>

(3) Processing Prediction Results: Each sample's prediction result is encapsulated in a corresponding Result object, supporting operations such as printing, saving as an image, and saving as a `json` file:


<table>
<thead>
<tr>
<th>Method</th>
<th>Method Description</th>
<th>Parameters</th>
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
<td>Whether to format the output content with JSON indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output JSON data for better readability, only valid when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-ASCII characters to Unicode. When set to <code>True</code>, all non-ASCII characters will be escaped; <code>False</code> retains the original characters, only valid when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save results as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, when it is a directory, the saved file name will be consistent with the input file type</td>
<td>N/A</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output JSON data for better readability, only valid when <code>format_json</code> is <code>True</code></td>
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
<td>Save the visual images of intermediate modules in PNG format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supports directories or file paths</td>
<td>N/A</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Save the tables in the file as an HTML file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supports directories or file paths</td>
<td>N/A</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Save the tables in the file as an XLSX file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supports directories or file paths</td>
<td>N/A</td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal. The content printed to the terminal is explained as follows:
    - `input_path`: `(str)` The input path of the image to be predicted

    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF; otherwise, it is `None`

    - `model_settings`: `(Dict[str, bool])` Model parameters required for configuring the pipeline

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing pipeline
        - `use_general_ocr`: `(bool)` Controls whether to enable the OCR pipeline
        - `use_seal_recognition`: `(bool)` Controls whether to enable the seal recognition pipeline
        - `use_table_recognition`: `(bool)` Controls whether to enable the table recognition pipeline
        - `use_formula_recognition`: `(bool)` Controls whether to enable the formula recognition pipeline

    - `parsing_res_list`: `(List[Dict])` A list of parsing results, each element is a dictionary, and the list order follows the reading order after parsing.
        - `block_bbox`: `(np.ndarray)` The bounding box of the layout area.
        - `block_label`: `(str)` The label of the layout area, such as `text`, `table`, etc.
        - `block_content`: `(str)` The content within the layout area.

    - `overall_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` A dictionary of global OCR results
      - `input_path`: `(Union[str, None])` The image path received by the OCR pipeline, when the input is `numpy.ndarray`, it is saved as `None`
      - `model_settings`: `(Dict)` Model configuration parameters for the OCR pipeline
      - `dt_polys`: `(List[numpy.ndarray])` A list of polygon boxes for text detection. Each detection box is represented by a numpy array consisting of 4 vertex coordinates, with a shape of (4, 2) and a data type of int16
      - `dt_scores`: `(List[float])` A list of confidence scores for text detection boxes
      - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the text detection module
        - `limit_side_len`: `(int)` The side length limit for image preprocessing
        - `limit_type`: `(str)` The processing method for the side length limit
        - `thresh`: `(float)` The confidence threshold for text pixel classification
        - `box_thresh`: `(float)` The confidence threshold for text detection boxes
        - `unclip_ratio`: `(float)` The inflation coefficient for text detection boxes
        - `text_type`: `(str)` The type of text detection, currently fixed as "general"

      - `text_type`: `(str)` The type of text detection, currently fixed as "general"
      - `textline_orientation_angles`: `(List[int])` The prediction results for text line orientation classification. When enabled, it returns actual angle values (e.g., [0,0,1])
      - `text_rec_score_thresh`: `(float)` The filtering threshold for text recognition results
      - `rec_texts`: `(List[str])` A list of text recognition results, only including texts with confidence scores exceeding `text_rec_score_thresh`
      - `rec_scores`: `(List[float])` A list of confidence scores for text recognition, already filtered by `text_rec_score_thresh`
      - `rec_polys`: `(List[numpy.ndarray])` A list of text detection boxes after confidence filtering, with the same format as `dt_polys`

    - `formula_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` A list of formula recognition results, each element is a dictionary
        - `rec_formula`: `(str)` The formula recognition result
        - `rec_polys`: `(numpy.ndarray)` The formula detection box, with a shape of (4, 2) and a dtype of int16
        - `formula_region_id`: `(int)` The region ID where the formula is located

    - `seal_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` A list of seal recognition results, each element is a dictionary
        - `input_path`: `(str)` The input path of the seal image
        - `model_settings`: `(Dict)` Model configuration parameters for```markdown
**AI and Computer Vision Tutorial**

- Calling the `save_to_json()` method will save the aforementioned content to the specified `save_path`. If a directory is specified, the save path will be `save_path/{your_img_basename}.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, `numpy.array` types will be converted to list form.
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, the save path will be `save_path/{your_img_basename}_ocr_res_img.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (pipelines often contain many result images, so it is not recommended to specify a specific file path directly, as multiple images will be overwritten, leaving only the last one.)

In addition, attributes are also supported for obtaining visual images with results and prediction results, specifically as follows:
<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Obtain the predicted results in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Obtain the visual image in <code>dict</code> format</td>
</tr>
</table>

- The prediction results obtained by the `json` attribute are data of the `dict` type, with content consistent with that saved by calling the `save_to_json()` method.
- The prediction results returned by the `img` attribute are data of the `dict` type. The keys are `layout_det_res`, `overall_ocr_res`, `text_paragraphs_ocr_res`, `formula_res_region1`, `table_cell_img`, and `seal_res_region1`, and the corresponding values are `Image.Image` objects: used to display the visual images of layout area detection, OCR, OCR text paragraphs, formulas, tables, and seal results, respectively. If optional modules are not used, only `layout_det_res` will be included in the dictionary.

Furthermore, you can obtain the layout parsing pipeline configuration file and load it for prediction. Execute the following command to save the results in `my_path`:
```
paddlex --get_pipeline_config layout_parsing --save_path ./my_path
```
Once you have obtained the configuration file, you can customize the configurations of the layout parsing pipeline by modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file. An example is as follows:
```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/layout_parsing.yaml")
output = pipeline.predict(
    input="./demo_paper.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```
**Note**: The parameters in the configuration file are pipeline initialization parameters. If you wish to change the initialization parameters of the general layout parsing pipeline, you can directly modify the parameters in the configuration file and load it for prediction. Additionally, CLI prediction also supports passing in configuration files by specifying the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment

If the pipeline meets your requirements in terms of inference speed and accuracy, you can proceed with development integration or deployment.

To directly apply the pipeline in your Python project, refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

Additionally, PaddleX offers three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In production environments, many applications require stringent performance metrics, especially response speed, to ensure efficient operation and smooth user experience. PaddleX provides a high-performance inference plugin that deeply optimizes model inference and pre/post-processing for significant end-to-end speedups. For detailed instructions on high-performance inference, refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Serving</b>: Serving is a common deployment strategy in real-world production environments. By encapsulating inference functions into services, clients can access these services via network requests to obtain inference results. PaddleX supports various solutions for serving pipelines. For detailed pipeline serving procedures, please refer to the [PaddleX Pipeline Serving Guide](../../../pipeline_deploy/serving.md).

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
<td>The URL of an image or PDF file accessible by the server, or the Base64-encoded content of the above file types. For PDF files with more than 10 pages, only the content of the first 10 pages will be used.</td>
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
<td>See the description of the <code>use_doc_orientation_classify</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the description of the <code>use_doc_unwarping</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTextlineOrientation</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the description of the <code>use_textline_orientation</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>useGeneralOcr</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the description of the <code>use_general_ocr</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>useSealRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the description of the <code>use_seal_recognition</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>useTableRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the description of the <code>use_table_recognition</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>useFormulaRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the description of the <code>use_formula_recognition</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>See the description of the <code>text_det_limit_side_len</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>See the description of the <code>text_det_limit_type</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>text_det_thresh</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>text_det_box_thresh</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>textDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>text_det_unclip_ratio</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>textRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>text_rec_score_thresh</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitSideLen</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>See the description of the <code>seal_det_limit_side_len</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetLimitType</code></td>
<td><code>string</code> | <code>null</code></td>
<td>See the description of the <code>seal_det_limit_type</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>seal_det_thresh</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetBoxThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>seal_det_box_thresh</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealDetUnclipRatio</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>seal_det_unclip_ratio</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>sealRecScoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>seal_rec_score_thresh</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>See the description of the <code>layout_threshold</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>See the description of the <code>layout_nms</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>null</code></td>
<td>See the description of the <code>layout_unclip_ratio</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>null</code></td>
<td>See the description of the <code>layout_merge_bboxes_mode</code> parameter in the <code>predict</code> method of the pipeline.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the response body's <code>result</code> has the following attributes:</li>
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
<td>The layout parsing results. The length of the array is 1 (for image input) or the smaller of the document page count and 10 (for PDF input). For PDF input, each element in the array represents the processing result of each page in the PDF file.</td>
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
<td>A simplified version of the <code>res</code> field in the JSON representation generated by the <code>predict</code> method of the production object, with the <code>input_path</code> field removed.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>See the description of the <code>img</code> attribute in the result of the pipeline prediction. The images are in JPEG format and encoded in Base64.</td>
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

API_URL = "http://localhost:8080/layout-parsing" # Service URL
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {
    "file": file_data, # Base64-encoded file content or file URL
    "fileType": 1,
}

# Call the API
response = requests.post(API_URL, json=payload)

# Process the response data
assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["layoutParsingResults"]):
    print(res["prunedResult"])
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment refers to placing computational and data processing capabilities directly on user devices, enabling them to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).

You can choose an appropriate method to deploy your model pipeline based on your needs, and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the general layout parsing pipeline do not meet your requirements in terms of accuracy or speed for your specific scenario, you can try to further fine-tune the existing models using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the general layout parsing pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the general layout parsing pipeline consists of 7 modules, unsatisfactory performance may stem from any of these modules.

You can analyze images with poor recognition results and follow the guidelines below for analysis and model fine-tuning:

* Incorrect table structure detection (e.g., wrong row/column recognition, incorrect cell positions) may indicate deficiencies in the table structure recognition module. You need to refer to the <b>Customization</b> section in the [Table Structure Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md) and fine-tune the table structure recognition model using your private dataset.
* Misplaced layout elements (e.g., incorrect positioning of tables, seals) may suggest issues with the layout detection module. You should consult the <b>Customization</b> section in the [Layout Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/layout_detection.md) and fine-tune the layout detection model with your private dataset.
* Frequent undetected texts (i.e., text missing detection) indicate potential weaknesses in the text detection model. Follow the <b>Customization</b> section in the [Text Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_detection.md) to fine-tune the text detection model using your private dataset.
* High text recognition errors (i.e., recognized text content does not match the actual text) suggest further improvements to the text recognition model. Refer to the <b>Customization</b> section in the [Text Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_recognition.md) to fine-tune the text recognition model.
* Frequent recognition errors in detected seal texts indicate the need for improvements to the seal text detection model. Consult the <b>Customization</b> section in the [Seal Text Detection Module Development Tutorials](../../../module_usage/tutorials/ocr_modules/) to fine-tune the seal text detection model.
* High recognition errors in detected formulas (i.e., recognized formula content does not match the actual formula) suggest further enhancements to the formula recognition model. Follow the [Customization](../../../module_usage/tutorials/ocr_modules/formula_recognition.md#Customization) section in the [Formula Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/formula_recognition.md) to fine-tune the formula recognition model.
* Frequent misclassifications of document or certificate orientations with text areas indicate the need for improvements to the document image orientation classification model. Refer to the <b>Customization</b> section in the [Document Image Orientation Classification Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md) to fine-tune the model.

### 4.2 Model Application
After fine-tuning your model with a private dataset, you will obtain local model weights files.

To use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local paths of the fine-tuned model weights to the corresponding positions in the configuration file:

```python
......
 Pipeline:
  layout_model: PicoDet_layout_1x  # Can be modified to the local path of the fine-tuned model
  table_model: SLANet_plus  # Can be modified to the local path of the fine-tuned model
  text_det_model: PP-OCRv4_server_det  # Can be modified to the local path of the fine-tuned model
  text_rec_model: PP-OCRv4_server_rec  # Can be modified to the local path of the fine-tuned model
  formula_rec_model: LaTeX_OCR_rec  # Can be modified to the local path of the fine-tuned model
  seal_text_det_model: PP-OCRv4_server_seal_det   # Can be modified to the local path of the fine-tuned model
  doc_image_unwarp_model: UVDoc  # Can be modified to the local path of the fine-tuned model
  doc_image_ori_cls_model: PP-LCNet_x1_0_doc_ori  # Can be modified to the local path of the fine-tuned model
  layout_batch_size: 1
  text_rec_batch_size: 1
  table_batch_size: 1
  device: "gpu:0"
......
```
Subsequently, refer to the command line or Python script methods in the local experience to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for inference in the layout parsing pipeline, the Python command is:

```bash
paddlex --pipeline layout_parsing --input layout_parsing.jpg --device gpu:0
```
At this point, if you want to switch the hardware to Ascend NPU, simply modify `--device` to npu in the Python command:

```bash
paddlex --pipeline layout_parsing --input layout_parsing.jpg --device npu:0
```
If you want to use the general layout parsing pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.md).
