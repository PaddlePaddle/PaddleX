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
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">Trained Model</a></td>
<td>59.52</td>
<td>103.08 / 103.08</td>
<td>197.99 / 197.99</td>
<td>6.9 M</td>
<td>SLANet is a table structure recognition model developed by Baidu PaddleX Team. The model significantly improves the accuracy and inference speed of table structure recognition by adopting a CPU-friendly lightweight backbone network PP-LCNet, a high-low-level feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structural and positional information.</td>
</tr>
<tr>
<td>SLANet_plus</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANet_plus_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">Trained Model</a></td>
<td>63.69</td>
<td>140.29 / 140.29</td>
<td>195.39 / 195.39</td>
<td>6.9 M</td>
<td>SLANet_plus is an enhanced version of SLANet, the table structure recognition model developed by Baidu PaddleX Team. Compared to SLANet, SLANet_plus significantly improves the recognition ability for wireless and complex tables and reduces the model's sensitivity to the accuracy of table positioning, enabling more accurate recognition even with offset table positioning.</td>
</tr>
</table>
<p><b>Note: The above accuracy metrics are measured on PaddleX's internally built English table recognition dataset. All GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Layout Detection Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Trained Model</a></td>
<td>86.8</td>
<td>9.03 / 3.10</td>
<td>25.82 / 20.70</td>
<td>7.4</td>
<td>An efficient layout area localization model trained on the PubLayNet dataset based on PicoDet-1x can locate five types of areas, including text, titles, tables, images, and lists.</td>
</tr>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Trained Model</a></td>
<td>95.7</td>
<td>8.02 / 3.09</td>
<td>23.70 / 20.41</td>
<td>7.4 M</td>
<td>An efficient layout area localization model trained on the PubLayNet dataset based on PicoDet-1x can locate one type of tables.</td>
</tr>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Trained Model</a></td>
<td>87.1</td>
<td>8.99 / 2.22</td>
<td>16.11 / 8.73</td>
<td>4.8</td>
<td>An high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Trained Model</a></td>
<td>70.3</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td>A high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Trained Model</a></td>
<td>89.3</td>
<td>13.05 / 4.50</td>
<td>41.30 / 41.30</td>
<td>22.6</td>
<td>An efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Trained Model</a></td>
<td>79.9</td>
<td>13.50 / 4.69</td>
<td>43.32 / 43.32</td>
<td>22.6</td>
<td>A efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Trained Model</a></td>
<td>95.9</td>
<td>114.93 / 27.71</td>
<td>947.56 / 947.56</td>
<td>470.1</td>
<td>A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Trained Model</a></td>
<td>92.6</td>
<td>115.29 / 104.09</td>
<td>995.27 / 995.27</td>
<td>470.2</td>
<td>A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
</tbody>
</table>
<p><b>Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built layout region analysis dataset, containing 10,000 images of common document types, including English and Chinese papers, magazines, research reports, etc. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Text Detection Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Trained Model</a></td>
<td>82.69</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>109</td>
<td>PP-OCRv4's server-side text detection model, featuring higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Trained Model</a></td>
<td>77.79</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>PP-OCRv4's mobile text detection model, optimized for efficiency, suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>
<p><b>Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built Chinese dataset, covering street scenes, web images, documents, and handwritten texts, with 500 images for detection. All GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Text Recognition Module Models</b>:</p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Trained Model</a></td>
<td>78.20</td>
<td>4.82 / 4.82</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td rowspan="2">PP-OCRv4 is the next version of Baidu PaddlePaddle's self-developed text recognition model PP-OCRv3. By introducing data augmentation schemes and GTC-NRTR guidance branches, it further improves text recognition accuracy without compromising inference speed. The model offers both server (server) and mobile (mobile) versions to meet industrial needs in different scenarios.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Trained Model</a></td>
<td>79.20</td>
<td>6.58 / 6.58</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
</tr>
</table>
<p><b>Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built Chinese dataset, covering street scenes, web images, documents, and handwritten texts, with 11,000 images for text recognition. All GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Trained Model</a></td>
<td>68.81</td>
<td>8.08 / 8.08</td>
<td>50.17 / 42.50</td>
<td>73.9 M</td>
<td rowspan="1">
SVTRv2 is a server-side text recognition model developed by the OpenOCR team at the Vision and Learning Lab (FVL) of Fudan University. It won the first prize in the OCR End-to-End Recognition Task of the PaddleOCR Algorithm Model Challenge, with a 6% improvement in end-to-end recognition accuracy compared to PP-OCRv4 on the A-list.
</td>
</tr>
</table>
<p><b>Note: The evaluation set for the above accuracy metrics is the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task</a> A-list. GPU inference time is based on NVIDIA Tesla T4 with FP32 precision. CPU inference speed is based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Trained Model</a></td>
<td>65.07</td>
<td>5.93 / 5.93</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
<td rowspan="1">
The RepSVTR text recognition model is a mobile-oriented text recognition model based on SVTRv2. It won the first prize in the OCR End-to-End Recognition Task of the PaddleOCR Algorithm Model Challenge, with a 2.5% improvement in end-to-end recognition accuracy compared to PP-OCRv4 on the B-list, while maintaining similar inference speed.
</td>
</tr>
</table>
<p><b>Note: The evaluation set for the above accuracy metrics is the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task</a> B-list. GPU inference time is based on NVIDIA Tesla T4 with FP32 precision. CPU inference speed is based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Formula Recognition Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>BLEU Score</th>
<th>Normed Edit Distance</th>
<th>ExpRate (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size</th>
</tr>
</thead>
<tbody>
<tr>
<td>LaTeX_OCR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Trained Model</a></td>
<td>0.8821</td>
<td>0.0823</td>
<td>40.01</td>
<td>2047.13 / 2047.13</td>
<td>10582.73 / 10582.73</td>
<td>89.7 M</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are measured on the <a href="https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO">LaTeX-OCR Formula Recognition Test Set</a>. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Seal Text Detection Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Trained Model</a></td>
<td>98.21</td>
<td>74.75 / 67.72</td>
<td>382.55 / 382.55</td>
<td>109</td>
<td>PP-OCRv4's server-side seal text detection model, featuring higher accuracy, suitable for deployment on better-equipped servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Trained Model</a></td>
<td>96.47</td>
<td>7.82 / 3.09</td>
<td>48.28 / 23.97</td>
<td>4.6</td>
<td>PP-OCRv4's mobile seal text detection model, offering higher efficiency, suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are evaluated on a self-built dataset containing 500 circular seal images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Text Image Rectification Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>MS-SSIM (%)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Trained Model</a></td>
<td>54.40</td>
<td>30.3 M</td>
<td>High-precision text image rectification model</td>
</tr>
</tbody>
</table>
<p><b>The accuracy metrics of the models are measured from the <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet benchmark</a>.</b></p>
<p><b>Document Image Orientation Classification Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Trained Model</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, with four categories: 0°, 90°, 180°, 270°</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are evaluated on a self-built dataset covering various scenarios such as certificates and documents, containing 1000 images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p></details>

## 2. Quick Start

PaddleX provides pre-trained model pipelines that can be quickly experienced. You can experience the effect of the General Image Classification pipeline online, or locally using command line or Python.

Before using the General Layout Parsing pipeline locally, please ensure you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.md).

### 2.1 Experience via Command Line
One command is all you need to quickly experience the effect of the Layout Parsing pipeline. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png) and replace `--input` with your local path to make predictions.

```bash
paddlex --pipeline layout_parsing --input demo_paper.png --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it is the Layout Parsing pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 indicates using the first GPU, gpu:1,2 indicates using the second and third GPUs). You can also choose to use CPU (--device cpu).
```

When executing the above command, the default Layout Parsing pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details><summary> 👉Click to expand</summary>
<pre><code class="language-bash">paddlex --get_pipeline_config layout_parsing
</code></pre>
<p>After execution, the layout parsing pipeline configuration file will be saved in the current directory. If you wish to customize the save location, you can execute the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code class="language-bash">paddlex --get_pipeline_config layout_parsing --save_path ./my_path
</code></pre>
<p>After obtaining the pipeline configuration file, you can replace <code>--pipeline</code> with the saved path of the configuration file to make it take effect. For example, if the configuration file is saved as <code>./layout_parsing.yaml</code>, simply execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./layout_parsing.yaml --input layout_parsing.jpg
</code></pre>
<p>Here, parameters such as <code>--model</code> and <code>--device</code> do not need to be specified, as they will use the parameters in the configuration file. If these parameters are still specified, the specified parameters will take precedence.</p></details>

After running, the result will be:

<details><summary> 👉Click to expand</summary>
<pre><code>{'input_path': PosixPath('/root/.paddlex/temp/tmp5jmloefs.png'), 'parsing_result': [{'input_path': PosixPath('/root/.paddlex/temp/tmpshsq8_w0.png'), 'layout_bbox': [51.46833, 74.22329, 542.4082, 232.77504], 'image': {'img': array([[[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [213, 221, 238],
        [217, 223, 240],
        [233, 234, 241]],

       [[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8), 'image_text': ''}, 'layout': 'single'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpcd2q9uyu.png'), 'layout_bbox': [47.68295, 243.08054, 546.28253, 295.71045], 'figure_title': 'Overview of RT-DETR, We feed th', 'layout': 'single'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpr_iqa8b3.png'), 'layout_bbox': [58.416977, 304.1531, 275.9134, 400.07513], 'image': {'img': array([[[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8), 'image_text': ''}, 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmphpblxl3p.png'), 'layout_bbox': [100.62961, 405.97458, 234.79774, 414.77414], 'figure_title': 'Figure 5. The fusion block in CCFF.', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmplgnczrsf.png'), 'layout_bbox': [47.81724, 421.9041, 288.01566, 550.538], 'text': 'D, Ds, not only significantly reduces latency (35% faster),\nRut\nnproves accuracy (0.4% AP higher), CCFF is opti\nased on the cross-scale fusion module, which\nnsisting of convolutional lavers intc\npath.\nThe role of the fusion block is t\n into a new feature, and its\nFigure 5. The f\nblock contains tw\n1 x1\nchannels, /V RepBlock\n. anc\n: two-path outputs are fused by element-wise add. We\ntormulate the calculation ot the hvbrid encoder as:', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpsq0ey9md.png'), 'layout_bbox': [94.60716, 558.703, 288.04193, 600.19434], 'formula': '\\begin{array}{l}{{\\Theta=K=\\mathrm{p.s.sp{\\pm}}\\mathrm{i.s.s.}(\\mathrm{l.s.}(\\mathrm{l.s.}(\\mathrm{l.s.}}),{\\qquad\\mathrm{{a.s.}}\\mathrm{s.}}}\\\\ {{\\tau_{\\mathrm{{s.s.s.s.s.}}(\\mathrm{l.s.},\\mathrm{l.s.},\\mathrm{s.s.}}\\mathrm{s.}\\mathrm{s.}}\\end{array}),}}\\\\ {{\\bar{\\mathrm{e-c.c.s.s.}(\\mathrm{s.},\\mathrm{s.s.},\\ s_{s}}\\mathrm{s.s.},\\tau),}}\\end{array}', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpv30qy0v4.png'), 'layout_bbox': [47.975555, 607.12024, 288.5776, 629.1252], 'text': 'tened feature to the same shape as Ss.\nwhere Re shape represents restoring the shape of the flat-', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmp0jejzwwv.png'), 'layout_bbox': [48.383354, 637.581, 245.96404, 648.20496], 'paragraph_title': '4.3. Uncertainty-minimal Query Selection', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpushex416.png'), 'layout_bbox': [47.80134, 656.002, 288.50192, 713.24994], 'text': 'To reduce the difficulty of optimizing object queries in\nDETR, several subsequent works [42, 44, 45] propose query\nselection schemes, which have in common that they use the\nconfidence score to select the top K’ features from the en-\ncoder to initialize object queries (or just position queries).', 'layout': 'left'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpki7e_6wc.png'), 'layout_bbox': [306.6371, 302.1026, 546.3772, 419.76724], 'text': 'The confidence score represents the likelihood that the fea\nture includes foreground objects. Nevertheless, the \nare required to simultaneously model the category\nojects, both of which determine the quality of the\npertor\ncore of the fes\nBased on the analysis, the current query\n considerable level of uncertainty in the\nresulting in sub-optimal initialization for\nand hindering the performance of the detector.', 'layout': 'right'}, {'input_path': PosixPath('/root/.paddlex/temp/tmppbxrfehp.png'), 'layout_bbox': [306.0642, 422.7347, 546.9216, 539.45734], 'text': 'To address this problem, we propose the uncertainty mini\nmal query selection scheme, which explicitly const\noptim\n the epistemic uncertainty to model the\nfeatures, thereby providing \nhigh-quality\nr the decoder. Specifically,\n the discrepancy between i\nalization P\nand classificat\n.(2\ntunction for the gradie', 'layout': 'right'}, {'input_path': PosixPath('/root/.paddlex/temp/tmp1mgiyd21.png'), 'layout_bbox': [331.52808, 549.32635, 546.5229, 586.15546], 'formula': '\\begin{array}{c c c}{{}}&amp;{{}}&amp;{{\\begin{array}{c}{{i\\langle X\\rangle=({\\bar{Y}}({\\bar{X}})+{\\bar{Z}}({\\bar{X}})\\mid X\\in{\\bar{\\pi}}^{\\prime}}}&amp;{{}}\\\\ {{}}&amp;{{}}&amp;{{}}\\end{array}}}&amp;{{\\emptyset}}\\\\ {{}}&amp;{{}}&amp;{{C(\\bar{X},{\\bar{X}})=C..\\scriptstyle(\\bar{0},{\\bar{Y}})+{\\mathcal{L}}_{{\\mathrm{s}}}({\\bar{X}}),\\ 6)}}&amp;{{}}\\end{array}', 'layout': 'right'}, {'input_path': PosixPath('/root/.paddlex/temp/tmp8t73dpym.png'), 'layout_bbox': [306.44016, 592.8762, 546.84314, 630.60126], 'text': 'where  and y denote the prediction and ground truth,\n= (c, b), c and b represent the category and bounding\nbox respectively, X represent the encoder feature.', 'layout': 'right'}, {'input_path': PosixPath('/root/.paddlex/temp/tmpftnxeyjm.png'), 'layout_bbox': [306.15652, 632.3142, 546.2463, 713.19073], 'text': 'Effectiveness analysis. To analyze the effectiveness of the\nuncertainty-minimal query selection, we visualize the clas-\nsificatior\nscores and IoU scores of the selected fe\nCOCO\na 12017, Figure 6. We draw the scatterplo\nt with\ndots\nrepresent the selected features from the model trained\nwith uncertainty-minimal query selection and vanilla query', 'layout': 'right'}]}
</code></pre></details>

### 2.2 Python Script Integration
A few lines of code are all you need to quickly perform inference on your production line. Taking the general layout parsing pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="layout_parsing")

output = pipeline.predict("demo_paper.png")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_img("./output/")  # Save the result as an image file
    res.save_to_xlsx("./output/")  # Save the result as an Excel file
    res.save_to_html("./output/")  # Save the result as an HTML file
```
The results obtained are the same as those from the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: Specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it's a pipeline name, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for pipeline model inference. Supports: "gpu", "cpu".</td>
<td><code>str</code></td>
<td>"gpu"</td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available if the pipeline supports it.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
(2) Call the `predict` method of the pipeline object to perform inference: The `predict` method takes `x` as a parameter, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

<table>
<thead>
<tr>
<th>Parameter Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>Supports directly passing Python variables, such as numpy.ndarray representing image data.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing the path of the file to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing the URL of the file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">Example</a>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing a local directory, which should contain files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td><code>dict</code></td>
<td>Supports passing a dictionary type, where the key needs to correspond to the specific task, e.g., "img" for image classification tasks, and the value of the dictionary supports the above data types, e.g., <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td><code>list</code></td>
<td>Supports passing a list, where the list elements need to be of the above data types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>, <code>[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>
(3) Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

(4) Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving as files, with the supported file types depending on the specific pipeline, such as:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Method Parameters</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>save_to_img</code></td>
<td>Saves the result as an image file.</td>
<td><code>- save_path</code>: <code>str</code> type, the path to save the file. When it's a directory, the saved file name is consistent with the input file name.</td>
</tr>
<tr>
<td><code>save_to_html</code></td>
<td>Saves the result as an HTML file.</td>
<td><code>- save_path</code>: <code>str</code> type, the path to save the file. When it's a directory, the saved file name is consistent with the input file name.</td>
</tr>
</tbody>
</table>
| `save_to_xlsx` | Saves the result as an Excel file. | `- save_path`: `str` type, the path to save the file. When it's a directory, the saved file name is consistent with the input file name.

Within this tutorial on Artificial Intelligence and Computer Vision, we will explore the capabilities of saving and exporting results from various processes, including OCR (Optical Character Recognition), layout analysis, and table structure recognition. Specifically, the `save_to_img` function enables saving visualization results, `save_to_html` converts tables directly into HTML files, and `save_to_xlsx` exports tables as Excel files.

Upon obtaining the configuration file, you can customize various settings for the layout parsing pipeline by simply modifying the `pipeline` parameter within the `create_pipeline` method to point to your configuration file path.

For instance, if your configuration file is saved at `./my_path/layout_parsing.yaml`, you can execute the following code:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/layout_parsing.yaml")
output = pipeline.predict("layout_parsing.jpg")
for res in output:
    res.print()  # Prints the structured output of the layout parsing prediction
    res.save_to_img("./output/")  # Saves the img format results from each submodule of the pipeline
    res.save_to_xlsx("./output/")  # Saves the xlsx format results from the table recognition module
    res.save_to_html("./output/")  # Saves the html results from the table recognition module
```

## 3. Development Integration/Deployment

If the pipeline meets your requirements in terms of inference speed and accuracy, you can proceed with development integration or deployment.

To directly apply the pipeline in your Python project, refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

Additionally, PaddleX offers three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In production environments, many applications require stringent performance metrics, especially response speed, to ensure efficient operation and smooth user experience. PaddleX provides a high-performance inference plugin that deeply optimizes model inference and pre/post-processing for significant end-to-end speedups. For detailed instructions on high-performance inference, refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Serving</b>: Serving is a common deployment strategy in real-world production environments. By encapsulating inference functions into services, clients can access these services via network requests to obtain inference results. PaddleX supports various solutions for serving pipelines. For detailed pipeline serving procedures, please refer to the [PaddleX Pipeline Serving Guide](../../../pipeline_deploy/serving.md).

Below are the API reference and multi-language service invocation examples for the basic serving solution:

<details><summary>API Reference</summary>
<p>For primary operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>The request body and the response body are both JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body properties are as follows:</li>
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
<td>UUID for the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the response body properties are as follows:</li>
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
<td>UUID for the request.</td>
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
<p>Primary operations provided by the service:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Performs layout parsing.</p>
<p><code>POST /layout-parsing</code></p>
<ul>
<li>Request body properties:</li>
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
<td>The URL of an image file or PDF file accessible by the server, or the Base64 encoded result of the content of the above-mentioned file types. For PDF files with more than 10 pages, only the content of the first 10 pages will be used.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code></td>
<td>File type. <code>0</code> indicates a PDF file, and <code>1</code> indicates an image file. If this property is not present in the request body, the file type will be inferred based on the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useImgOrientationCls</code></td>
<td><code>boolean</code></td>
<td>Whether to enable document image orientation classification. This function is enabled by default.</td>
<td>No</td>
</tr>
<tr>
<td><code>useImgUnwarping</code></td>
<td><code>boolean</code></td>
<td>Whether to enable text image rectification. This function is enabled by default.</td>
<td>No</td>
</tr>
<tr>
<td><code>useSealTextDet</code></td>
<td><code>boolean</code></td>
<td>Whether to enable seal text detection. This function is enabled by default.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> of the response body has the following properties:</li>
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
<td><code>layoutParsingResults</code></td>
<td><code>array</code></td>
<td>Layout parsing results. The array length is 1 (for image input) or the smaller of the number of document pages and 10 (for PDF input). For PDF input, each element in the array represents the processing result of each page in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>layoutParsingResults</code> is an <code>object</code> with the following properties:</p>
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
<td><code>layoutElements</code></td>
<td><code>array</code></td>
<td>Layout element information.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>layoutElements</code> is an <code>object</code> with the following properties:</p>
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
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>Position of the layout element. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner of the bounding box, respectively.</td>
</tr>
<tr>
<td><code>label</code></td>
<td><code>string</code></td>
<td>Label of the layout element.</td>
</tr>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>Text contained in the layout element.</td>
</tr>
<tr>
<td><code>layoutType</code></td>
<td><code>string</code></td>
<td>Arrangement of the layout element.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>Image of the layout element, in JPEG format, encoded using Base64.</td>
</tr>
</tbody>
</table></details>

<details><summary>Multi-language Service Invocation Examples</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/layout-parsing"

with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "file": image_data,
    "fileType": 1,
    "useImgOrientationCls": True,
    "useImgUnwarping": True,
    "useSealTextDet": True,
}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
print("\nDetected layout elements:")
for res in result["layoutParsingResults"]:
    for ele in res["layoutElements"]:
        print("===============================")
        print("bbox:", ele["bbox"])
        print("label:", ele["label"])
        print("text:", repr(ele["text"]))
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

To use the fine-tuned model weights, simply modify the production line configuration file by replacing the local paths of the fine-tuned model weights to the corresponding positions in the configuration file:

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
Subsequently, refer to the command line or Python script methods in the local experience to load the modified production line configuration file.

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
