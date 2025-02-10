---
comments: true
---

# General Table Recognition Pipeline v2 User Guide

## 1. Introduction to General Table Recognition Pipeline v2
Table recognition is a technology that automatically identifies and extracts table content and its structure from documents or images. It is widely used in data entry, information retrieval, and document analysis. By using computer vision and machine learning algorithms, table recognition can convert complex table information into an editable format, making it easier for users to further process and analyze data.

The General Table Recognition Pipeline v2 is designed to solve table recognition tasks by identifying tables in images and outputting them in HTML format. Unlike the General Table Recognition Pipeline, this pipeline introduces two additional modules: table classification and table cell detection, which are linked with the table structure recognition module to complete the table recognition task. This pipeline can achieve accurate table predictions and is applicable in various fields such as general, manufacturing, finance, and transportation. It also provides flexible service deployment options, supporting multiple programming languages on various hardware. Additionally, it offers secondary development capabilities, allowing you to train and fine-tune models on your own dataset, with seamless integration of the trained models.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/01.png"/>
<b>The General Table Recognition Pipeline v2 includes mandatory modules such as table structure recognition, table classification, table cell localization, text detection, and text recognition, as well as optional modules like layout area detection, document image orientation classification, and text image correction.</b>
<b>If you prioritize model accuracy, choose a model with higher accuracy; if you care more about inference speed, choose a model with faster inference speed; if you are concerned about model storage size, choose a model with a smaller storage size.</b>
<p><b>Table Recognition Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Accuracy (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANeXt_wired_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">Training Model</a></td>
<td rowspan="2">69.65</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">351M</td>
<td rowspan="2">The SLANeXt series is a new generation of table structure recognition models developed by the Baidu PaddlePaddle Vision Team. Compared to SLANet and SLANet_plus, SLANeXt focuses on recognizing table structures and has trained dedicated weights for both wired and wireless tables, significantly improving the recognition capabilities for all types of tables, especially for wired tables.</td>
</tr>
<tr>
<td>SLANeXt_wireless</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SLANeXt_wireless_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wireless_pretrained.pdparams">Training Model</a></td>
<td>63.69</td>
<td>522.536</td>
<td>1845.37</td>
<td>6.9 M</td>
</tr>
</table>
<b>Note: The above accuracy metrics are measured from the high-difficulty Chinese table recognition dataset internally built by PaddleX. The GPU inference time for all models is based on an NVIDIA Tesla T4 machine with FP32 precision type. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision type.</b>
<p><b>Table Classification Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top1 Acc(%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/CLIP_vit_base_patch16_224_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">Training Model</a></td>
<td>--</td>
<td>--</td>
<td>--</td>
<td>6.6M</td>
</tr>
</table>
<p><b>Note: The above accuracy metrics are measured on PaddleX's internally built table classification dataset. All model GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision, and CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Table Cell Detection Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wired_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">Training Model</a></td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
`<td rowspan="2">124M</td>
<td rowspan="2">RT-DETR is the first real-time end-to-end object detection model. The Baidu PaddlePaddle Vision Team used RT-DETR-L as the base model and pre-trained it on a self-built table cell detection dataset, achieving good performance in detecting both wired and wireless tables.</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wireless_table_cell_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">Training Model</a></td>
</tr>
</table>
<p><b>Note: The above accuracy metrics are measured on PaddleX's internally built table cell detection dataset. All model GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision, and CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Text Detection Module Models:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Training Model</a></td>
<td>82.69</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>109</td>
<td>The server-side text detection model of PP-OCRv4, with higher accuracy, suitable for deployment on high-performance servers.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>77.79</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>The mobile text detection model of PP-OCRv4, with higher efficiency, suitable for deployment on edge devices.</td>
</tr>
</tbody>
</table>
<p><b>Note: The evaluation set for the above accuracy metrics is a self-built Chinese dataset by PaddleOCR, covering multiple scenarios such as street view, web images, documents, and handwriting, with 500 images for detection. All models' GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Text Recognition Module Models:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>78.20</td>
<td>4.82 / 4.82</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td rowspan="2">PP-OCRv4 is the next version of the self-developed text recognition model PP-OCRv3 by Baidu PaddlePaddle Vision Team. By introducing data augmentation schemes and GTC-NRTR guidance branches, it further improves text recognition accuracy without changing the model inference speed. This model provides both server and mobile versions to meet industrial needs in different scenarios.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training Model</a></td>
<td>79.20</td>
<td>6.58 / 6.58</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
</tr>
</table>
<p><b>Note: The evaluation set for the above accuracy metrics is a self-built Chinese dataset by PaddleOCR, covering multiple scenarios such as street view, web images, documents, and handwriting, with 11,000 images for text recognition. All models' GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
<td>68.81</td>
<td>8.08 / 8.08</td>
<td>50.17 / 42.50</td>
<td>73.9 M</td>
<td rowspan="1">
SVTRv2 is a server-side text recognition model developed by the OpenOCR team from Fudan University's Vision and Learning Laboratory (FVL). It won the first prize in the PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task, with a 6% improvement in end-to-end recognition accuracy compared to PP-OCRv4.
</td>
</tr>
</table>
<p><b>Note: The evaluation set for the above accuracy metrics is the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task</a> leaderboard A. All models' GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
<td>65.07</td>
<td>5.93 / 5.93</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
<td rowspan="1">RepSVTR is a mobile text recognition model based on SVTRv2. It won the first prize in the PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task, with a 2.5% improvement in end-to-end recognition accuracy compared to PP-OCRv4 and comparable inference speed.</td>
</tr>
</table>
<p><b>Note: The evaluation set for the above accuracy metrics is the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task</a> leaderboard B. All models' GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Layout Region Detection Module Models (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Training Model</a></td>
<td>90.4</td>
<td>34.5252</td>
<td>1454.27</td>
<td>123.76 M</td>
<td>A high-precision layout region localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports, based on RT-DETR-L.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>15.9</td>
<td>160.1</td>
<td>22.578</td>
<td>A balanced precision and efficiency layout region localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports, based on PicoDet-L.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training Model</a></td>
<td>70.9</td>
<td>13.8</td>
<td>46.7</td>
<td>4.834</td>
<td>A high-efficiency layout region localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports, based on PicoDet-S.</td>
</tr>
</tbody>
</table>
<b>Note: The evaluation dataset for the above accuracy metrics is the layout region detection dataset built by PaddleOCR, containing 500 common document-type images of Chinese and English papers, magazines, contracts, books, exams, and research reports. The GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision type. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision type.</b>

> ❗ The above list includes the <b>3 core models</b> that are the focus of the layout detection module. The module supports a total of <b>11 full models</b>, including multiple predefined models with different categories. The complete list of models is as follows:

* <b>Table Layout Detection Models</b>
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
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Training Model</a></td>
<td>97.5</td>
<td>8.02 / 3.09</td>
<td>23.70 / 20.41</td>
<td>7.4 M</td>
<td>A high-efficiency layout region localization model trained on a self-built dataset using PicoDet-1x, capable of locating 1 type of region: tables</td>
</tr>
</tbody></table>
<b>Note: The evaluation dataset for the above accuracy metrics is the layout table region detection dataset built by PaddleOCR, containing 7,835 document-type images of Chinese and English papers with tables. The GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision type. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision type.</b>

* <b>3-category layout detection model, including tables, images, and seals</b>
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
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>88.2</td>
<td>8.99 / 2.22</td>
<td>16.11 / 8.73</td>
<td>4.8</td>
<td>A high-efficiency layout region localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using the lightweight PicoDet-S model</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>13.05 / 4.50</td>
<td>41.30 / 41.30</td>
<td>22.6</td>
<td>A layout region localization model with balanced efficiency and accuracy, trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-L</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>95.8</td>
<td>114.93 / 27.71</td>
<td>947.56 / 947.56</td>
<td>470.1</td>
<td>A high-precision layout region localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using RT-DETR-H</td>
</tr>
</tbody></table>
<b>Note: The evaluation dataset for the above accuracy metrics is the layout region detection dataset built by PaddleOCR, containing 1,154 common document-type images of Chinese and English papers, magazines, and research reports. The GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision type. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision type.</b>

* <b>5-category English document region detection model, including text, title, table, image, and list</b>
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
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Training Model</a></td>
<td>97.8</td>
<td>9.03 / 3.10</td>
<td>25.82 / 20.70</td>
<td>7.4</td>
<td>A high-efficiency English document layout region localization model trained on the PubLayNet dataset using PicoDet-1x</td>
</tr>
</tbody></table>
<b>Note: The evaluation dataset for the above accuracy metrics is the [PubLayNet](https://developer.ibm.com/exchanges/data/all/publaynet/) evaluation dataset, containing 11,245 images of English documents. The GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision type. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision type.</b>

* <b>17-category region detection model, including 17 common layout categories: paragraph title, image, text, number, abstract, content, figure title, formula, table, table title, reference, document title, footnote, header, algorithm, footer, seal</b>
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
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>87.4</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td>A high-efficiency layout region localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using the lightweight PicoDet-S model</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>13.50 / 4.69</td>
<td>43.32 / 43.32</td>
<td>22.6</td>
<td>A layout region localization model with balanced efficiency and accuracy, trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-L</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>98.3</td>
<td>115.29 / 104.09</td>
<td>995.27 / 995.27</td>
<td>470.2</td>
<td>A high-precision layout region localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using RT-DETR-H</td>
</tr>
</tbody>
</table>
<p><b>Note: The evaluation dataset for the above accuracy metrics is the layout region detection dataset built by PaddleOCR, containing 892 common document-type images of Chinese and English papers, magazines, and research reports. The GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision type. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision type.</b></p>
<p><b>Text Image Correction Module Model (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>MS-SSIM (%)</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
<td>54.40</td>
<td>30.3 M</td>
<td>High-precision text image correction model</td>
</tr>
</tbody>
</table>
<p><b>The accuracy metrics of the model are measured from the <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet benchmark</a>.</b></p>
<p><b>Document Image Orientation Classification Module Model (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>Document image classification model based on PP-LCNet_x1_0, containing four categories: 0 degrees, 90 degrees, 180 degrees, 270 degrees</td>
</tr>
</tbody>
</table>
<p><b>Note: The accuracy metrics above are evaluated on a self-built dataset covering multiple scenarios such as documents and certificates, including 1000 images. GPU inference time is based on NVIDIA Tesla T4 machines with FP32 precision, and CPU inference speed is based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>


## 2. Quick Start
All model production lines provided by PaddleX can be quickly experienced. You can use the command line or Python locally to experience the effect of the general table recognition production line v2.

### 2.1 Online Experience
Online experience is not supported at the moment.

### 2.2 Local Experience
Before using the General Table Recognition Production Line v2 locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 2.1 Command Line Experience
You can quickly experience the effect of the table recognition production line with one command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg), and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline table_recognition_v2 \
        --input table_recognition.jpg \
        --save_path ./output \
        --device gpu:0
```

The relevant parameter descriptions can be referred to in the [2.2.2 Integration via Python Script](#222-python-script-integration) for parameter descriptions.

<details><summary>👉 <b>After running, the result is: (Click to expand)</b></summary>

```bash
{'res': {'input_path': 'table_recognition.jpg', 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True, 'use_ocr_model': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'Table', 'score': 0.9922188520431519, 'coordinate': [3.0127392, 0.14648987, 547.5102, 127.72023]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': [array([[234,   6],
       [316,   6],
       [316,  25],
       [234,  25]], dtype=int16), array([[38, 39],
       [73, 39],
       [73, 57],
       [38, 57]], dtype=int16), array([[122,  32],
       [201,  32],
       [201,  58],
       [122,  58]], dtype=int16), array([[227,  34],
       [346,  34],
       [346,  57],
       [227,  57]], dtype=int16), array([[351,  34],
       [391,  34],
       [391,  58],
       [351,  58]], dtype=int16), array([[417,  35],
       [534,  35],
       [534,  58],
       [417,  58]], dtype=int16), array([[34, 70],
       [78, 70],
       [78, 90],
       [34, 90]], dtype=int16), array([[287,  70],
       [328,  70],
       [328,  90],
       [287,  90]], dtype=int16), array([[454,  69],
       [496,  69],
       [496,  90],
       [454,  90]], dtype=int16), array([[ 17, 101],
       [ 95, 101],
       [ 95, 124],
       [ 17, 124]], dtype=int16), array([[144, 101],
       [178, 101],
       [178, 122],
       [144, 122]], dtype=int16), array([[278, 101],
       [338, 101],
       [338, 124],
       [278, 124]], dtype=int16), array([[448, 101],
       [503, 101],
       [503, 121],
       [448, 121]], dtype=int16)], 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 'text_rec_score_thresh': 0, 'rec_texts': ['CRuncover', 'Dres', '连续工作3', '取出来放在网上', '没想', '江、整江等八大', 'Abstr', 'rSrivi', '$709.', 'cludingGiv', '2.72', 'Ingcubic', '$744.78'], 'rec_scores': [0.9951260685920715, 0.9943379759788513, 0.9968608021736145, 0.9978817105293274, 0.9985721111297607, 0.9616036415100098, 0.9977153539657593, 0.987593948841095, 0.9906861186027527, 0.9959743618965149, 0.9970152378082275, 0.9977849721908569, 0.9984450936317444], 'rec_polys': [array([[234,   6],
       [316,   6],
       [316,  25],
       [234,  25]], dtype=int16), array([[38, 39],
       [73, 39],
       [73, 57],
       [38, 57]], dtype=int16), array([[122,  32],
       [201,  32],
       [201,  58],
       [122,  58]], dtype=int16), array([[227,  34],
       [346,  34],
       [346,  57],
       [227,  57]], dtype=int16), array([[351,  34],
       [391,  34],
       [391,  58],
       [351,  58]], dtype=int16), array([[417,  35],
       [534,  35],
       [534,  58],
       [417,  58]], dtype=int16), array([[34, 70],
       [78, 70],
       [78, 90],
       [34, 90]], dtype=int16), array([[287,  70],
       [328,  70],
       [328,  90],
       [287,  90]], dtype=int16), array([[454,  69],
       [496,  69],
       [496,  90],
       [454,  90]], dtype=int16), array([[ 17, 101],
       [ 95, 101],
       [ 95, 124],
       [ 17, 124]], dtype=int16), array([[144, 101],
       [178, 101],
       [178, 122],
       [144, 122]], dtype=int16), array([[278, 101],
       [338, 101],
       [338, 124],
       [278, 124]], dtype=int16), array([[448, 101],
       [503, 101],
       [503, 121],
       [448, 121]], dtype=int16)], 'rec_boxes': array([[234,   6, 316,  25],
       [ 38,  39,  73,  57],
       [122,  32, 201,  58],
       [227,  34, 346,  57],
       [351,  34, 391,  58],
       [417,  35, 534,  58],
       [ 34,  70,  78,  90],
       [287,  70, 328,  90],
       [454,  69, 496,  90],
       [ 17, 101,  95, 124],
       [144, 101, 178, 122],
       [278, 101, 338, 124],
       [448, 101, 503, 121]], dtype=int16)}, 'table_res_list': [{'cell_box_list': [array([3.18822289e+00, 1.46489874e-01, 5.46996138e+02, 3.08782365e+01]), array([  3.21032453,  31.1510637 , 110.20750237,  65.14108063]), array([110.18174553,  31.13076188, 213.00813103,  65.02860047]), array([212.96108818,  31.09959008, 404.19618034,  64.99535157]), array([404.08112907,  31.18304802, 547.00864983,  65.0847223 ]), array([  3.21772957,  65.0738733 , 110.33685875,  96.07921387]), array([110.23703575,  65.02486207, 213.08839226,  96.01378419]), array([213.06095695,  64.96230103, 404.28425407,  95.97141816]), array([404.23704338,  65.04879548, 547.01273918,  96.03654267]), array([  3.22793937,  96.08334137, 110.38572502, 127.08698823]), array([110.40586662,  96.10539795, 213.19943047, 127.07002045]), array([213.12627983,  96.0539148 , 404.42686272, 127.02842499]), array([404.33042717,  96.07251526, 547.01273918, 126.45088746])], 'pred_html': '<html><body><table><tr><td colspan="4">CRuncover</td></tr><tr><td>Dres</td><td>连续工作3</td><td>取出来放在网上 没想</td><td>江、整江等八大</td></tr><tr><td>Abstr</td><td></td><td>rSrivi</td><td>$709.</td></tr><tr><td>cludingGiv</td><td>2.72</td><td>Ingcubic</td><td>$744.78</td></tr></table></body></html>', 'table_ocr_pred': {'rec_polys': [array([[234,   6],
       [316,   6],
       [316,  25],
       [234,  25]], dtype=int16), array([[38, 39],
       [73, 39],
       [73, 57],
       [38, 57]], dtype=int16), array([[122,  32],
       [201,  32],
       [201,  58],
       [122,  58]], dtype=int16), array([[227,  34],
       [346,  34],
       [346,  57],
       [227,  57]], dtype=int16), array([[351,  34],
       [391,  34],
       [391,  58],
       [351,  58]], dtype=int16), array([[417,  35],
       [534,  35],
       [534,  58],
       [417,  58]], dtype=int16), array([[34, 70],
       [78, 70],
       [78, 90],
       [34, 90]], dtype=int16), array([[287,  70],
       [328,  70],
       [328,  90],
       [287,  90]], dtype=int16), array([[454,  69],
       [496,  69],
       [496,  90],
       [454,  90]], dtype=int16), array([[ 17, 101],
       [ 95, 101],
       [ 95, 124],
       [ 17, 124]], dtype=int16), array([[144, 101],
       [178, 101],
       [178, 122],
       [144, 122]], dtype=int16), array([[278, 101],
       [338, 101],
       [338, 124],
       [278, 124]], dtype=int16), array([[448, 101],
       [503, 101],
       [503, 121],
       [448, 121]], dtype=int16)], 'rec_texts': ['CRuncover', 'Dres', '连续工作3', '取出来放在网上', '没想', '江、整江等八大', 'Abstr', 'rSrivi', '$709.', 'cludingGiv', '2.72', 'Ingcubic', '$744.78'], 'rec_scores': [0.9951260685920715, 0.9943379759788513, 0.9968608021736145, 0.9978817105293274, 0.9985721111297607, 0.9616036415100098, 0.9977153539657593, 0.987593948841095, 0.9906861186027527, 0.9959743618965149, 0.9970152378082275, 0.9977849721908569, 0.9984450936317444], 'rec_boxes': [array([234,   6, 316,  25], dtype=int16), array([38, 39, 73, 57], dtype=int16), array([122,  32, 201,  58], dtype=int16), array([227,  34, 346,  57], dtype=int16), array([351,  34, 391,  58], dtype=int16), array([417,  35, 534,  58], dtype=int16), array([34, 70, 78, 90], dtype=int16), array([287,  70, 328,  90], dtype=int16), array([454,  69, 496,  90], dtype=int16), array([ 17, 101,  95, 124], dtype=int16), array([144, 101, 178, 122], dtype=int16), array([278, 101, 338, 124], dtype=int16), array([448, 101, 503, 121], dtype=int16)]}}]}}
```

The explanation of the running result parameters can refer to the result interpretation in [2.2.2 Python Script Integration](#222-python-script-integration).

</details>

The visualization results are saved under `save_path`, where the visualization result of table recognition is as follows:
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition_v2/03.png"/>

### 2.2 Python Script Integration
* The above command line is for a quick experience to view the effect. Generally, in a project, integration through code is often required. You can complete the pipeline's fast inference with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline_name="table_recognition_v2")

output = pipeline.predict(
    input="table_recognition.jpg",
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

In the above Python script, the following steps are executed:

(1) The `create_pipeline()` function is used to instantiate a General Table Recognition Pipeline v2 object. The specific parameter descriptions are as follows:

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
<td>The name of the pipeline or the path to the pipeline configuration file. If it is a pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the production line (if set simultaneously with <code>pipeline</code>, it has higher priority than <code>pipeline</code>, and the production line name must be consistent with <code>pipeline</code>).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for pipeline inference. It supports specifying specific GPU card numbers, such as "gpu:0", specific card numbers for other hardware, such as "npu:0", or CPU as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference. This is only available if the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the general table recognition pipeline v2 object for inference prediction. This method will return a `generator`. The parameters of the `predict()` method and their descriptions are as follows:

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
<td>Data to be predicted, supports multiple input types, required</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Such as <code>numpy.ndarray</code> representing image data</li>
<li><b>str</b>: Such as the local path of an image file or PDF file: <code>/root/data/img.jpg</code>; <b>such as URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg">Example</a>; <b>such as local directory</b>, the directory must contain the images to be predicted, such as the local path: <code>/root/data/</code> (currently does not support prediction of PDF files in the directory, PDF files need to be specified to a specific file path)</li>
<li><b>List</b>: The list elements must be the above types of data, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Pipeline inference device</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>CPU</b>: Such as <code>cpu</code> indicating using CPU for inference;</li>
<li><b>GPU</b>: Such as <code>gpu:0</code> indicating using the 1st GPU for inference;</li>
<li><b>NPU</b>: Such as <code>npu:0</code> indicating using the 1st NPU for inference;</li>
<li><b>XPU</b>: Such as <code>xpu:0</code> indicating using the 1st XPU for inference;</li>
<li><b>MLU</b>: Such as <code>mlu:0</code> indicating using the 1st MLU for inference;</li>
<li><b>DCU</b>: Such as <code>dcu:0</code> indicating using the 1st DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to using the parameter value initialized by the pipeline. During initialization, it will preferentially use the local GPU 0 device, if not available, it will use the CPU device;</li>
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
<li><b>None</b>: If set to <code>None</code>, it will default to using the parameter value initialized by the pipeline, initialized to <code>True</code>;</li>
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
<li><b>None</b>: If set to <code>None</code>, it will default to using the parameter value initialized by the pipeline, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<td><code>text_det_limit_side_len</code></td>
<td>Image side length limit for text detection</td>
<td><code>int|None</code></td>
<td>
<ul>
<li><b>int</b>: Any integer greater than <code>0</code>;</li>
<li><b>None</b>: If set to <code>None</code>, it will default to using the parameter value initialized by the pipeline, initialized to <code>960</code>;</li>
</ul>
</td>
<td><code>None</code></td>

<td><code>text_det_limit_type</code></td>
<td>Image side length limit type for text detection</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>str</b>: Supports <code>min</code> and <code>max</code>, <code>min</code> indicates ensuring the shortest side of the image is not less than <code>det_limit_side_len</code>, <code>max</code> indicates ensuring the longest side of the image is not greater than <code>limit_side_len</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to using the parameter value initialized by the pipeline, initialized to <code>max</code>;</li>
</ul>
</td>
<td><code>None</code></td>

<td><code>text_det_thresh</code></td>
<td>Detection pixel threshold, only pixels with scores greater than this threshold in the output probability map will be considered as text pixels</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to using the parameter value initialized by the pipeline <code>0.3</code></li>
</ul>
</td>
<td><code>None</code></td>

<td><code>text_det_box_thresh</code></td>
<td>Detection box threshold, the result will be considered as a text region if the average score of all pixels within the detection box is greater than this threshold</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to using the parameter value initialized by the pipeline <code>0.6</code></li>
</ul>
</td>
<td><code>None</code></td>

<td><code>text_det_unclip_ratio</code></td>
<td>Text detection expansion ratio, this method is used to expand the text region, the larger the value, the larger the expanded area</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to using the parameter value initialized by the pipeline <code>2.0</code></li>
</ul>

</td>
<td><code>None</code></td>

<td><code>text_rec_score_thresh</code></td>
<td>Text recognition threshold, text results with scores greater than this threshold will be retained</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, it will default to using the parameter value initialized by the pipeline <code>0.0</code>, meaning no threshold</li>
</ul>

</td>
<td><code>None</code></td>

<tr>
<td><code>use_layout_detection</code></td>
<td>Whether to use the layout detection module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the production line will be used, initialized as <code>True</code>;</li>
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
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>dict</b>: Key is an integer category ID, value is any floating-point number greater than <code>0</code></li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the production line will be used, initialized as <code>0.5</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS post-processing for layout detection</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the production line will be used, initialized as <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Scale factor for the side length of detection boxes; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>float|list|None</code></td>
<td>
<ul>
<li><b>float</b>: A floating-point number greater than 0, e.g., 1.1, indicating that the center of the detection box remains unchanged, and both the width and height are scaled by 1.1 times</li>
<li><b>list</b>: e.g., [1.2, 1.5], indicating that the center of the detection box remains unchanged, the width is scaled by 1.2 times, and the height is scaled by 1.5 times</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the production line will be used, initialized as 1.0</li>
</ul>
</td>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Merging mode for detection boxes output by the model; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>string|None</code></td>
<td>
<ul>
<li><b>large</b>: When set to large, only the outermost box will be retained for overlapping detection boxes, and the inner overlapping boxes will be removed.</li>
<li><b>small</b>: When set to small, only the innermost boxes will be retained for overlapping detection boxes, and the outer overlapping boxes will be removed.</li>
<li><b>union</b>: No filtering of boxes will be performed; both inner and outer boxes will be retained.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the production line will be used, initialized as <code>large</code></li>
</ul>
</td>
<td>None</td>
</tr>
</tr></table>

(3) Process the prediction results. The prediction result for each sample is of type `dict`, and supports operations such as printing, saving as an image, saving as an `xlsx` file, saving as an `HTML` file, and saving as a `json` file:

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
<td>The file path for saving. When it is a directory, the saved file name will match the input file name</td>
<td>N/A</td>
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
<td>N/A</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Save the result as an xlsx file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>N/A</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Save the result as an HTML file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>N/A</td>
</tr>
</table>

- Calling the `print()` method will print the result to the terminal, with the printed content explained as follows:

    - `input_path`: `(str)` The input path of the image to be predicted

    - `model_settings`: `(Dict[str, bool])` Configuration parameters required for the production line model

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing sub-production line
    - `use_layout_detection`: `(bool)` Controls whether to enable the layout detection sub-production line
    - `use_ocr_model`: `(bool)` Controls whether to enable the OCR sub-production line
    - `layout_det_res`: `(Dict[str, Union[List[numpy.ndarray], List[float]]])` Output result of the layout detection sub-module. Only exists when `use_layout_detection=True`
        - `input_path`: `(Union[str, None])` The image path accepted by the layout detection module, saved as `None` when the input is a `numpy.ndarray`
        - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF, otherwise it is `None`
        - `boxes`: `(List[Dict])` List of detection boxes for layout seal regions, each element in the list contains the following fields
            - `cls_id`: `(int)` The class ID of the detection box
            - `score`: `(float)` The confidence score of the detection box
            - `coordinate`: `(List[float])` The coordinates of the four corners of the detection box, in the order of x1, y1, x2, y2, representing the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner  
    - `doc_preprocessor_res`: `(Dict[str, Union[str, Dict[str, bool], int]])` The output result of the document preprocessing sub-production line. Exists only when `use_doc_preprocessor=True`
        - `input_path`: `(Union[str, None])` The image path accepted by the image preprocessing sub-production line, saved as `None` when the input is `numpy.ndarray`
        - `model_settings`: `(Dict)` Model configuration parameters for the preprocessing sub-production line
            - `use_doc_orientation_classify`: `(bool)` Controls whether to enable document orientation classification
            - `use_doc_unwarping`: `(bool)` Controls whether to enable document unwarping
        - `angle`: `(int)` The prediction result of document orientation classification. When enabled, the values are [0,1,2,3], corresponding to [0°,90°,180°,270°] respectively; when not enabled, it is -1

    - `dt_polys`: `(List[numpy.ndarray])` List of polygons for text detection. Each detection box is represented by a numpy array of 4 vertex coordinates, with the array shape being (4, 2) and the data type being int16

    - `dt_scores`: `(List[float])` List of confidence scores for text detection boxes

    - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the text detection module
        - `limit_side_len`: `(int)` The side length limit value during image preprocessing
        - `limit_type`: `(str)` The processing method for the side length limit
        - `thresh`: `(float)` Confidence threshold for text pixel classification
        - `box_thresh`: `(float)` Confidence threshold for text detection boxes
        - `unclip_ratio`: `(float)` Expansion coefficient for text detection boxes
        - `text_type`: `(str)` Type of text detection, currently fixed as "general"

    - `text_rec_score_thresh`: `(float)` Filtering threshold for text recognition results

    - `rec_texts`: `(List[str])` List of text recognition results, only includes texts with confidence scores exceeding `text_rec_score_thresh`

    - `rec_scores`: `(List[float])` List of confidence scores for text recognition, filtered by `text_rec_score_thresh`

    - `rec_polys`: `(List[numpy.ndarray])` List of text detection boxes after confidence filtering, same format as `dt_polys`

    - `rec_boxes`: `(numpy.ndarray)` Array of rectangular bounding boxes for detection boxes, with shape (n, 4) and dtype int16. Each row represents the [x_min, y_min, x_max, y_max] coordinates of a rectangle, where (x_min, y_min) is the top-left coordinate and (x_max, y_max) is the bottom-right coordinate

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}.json`; if specified as a file, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, the `numpy.array` types will be converted to lists.

- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_ocr_res_img.{your_img_extension}`; if specified as a file, it will be saved directly to that file. (The production line usually contains many result images, it is not recommended to specify a specific file path directly, otherwise multiple images will be overwritten, leaving only the last image)

- Calling the `save_to_html()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}.html`; if specified as a file, it will be saved directly to that file. In the general table recognition production line v2, the HTML form of the table in the image will be written to the specified HTML file.

- Calling the `save_to_xlsx()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}.xlsx`; if specified as a file, it will be saved directly to that file. In the general table recognition production line v2, the Excel form of the table in the image will be written to the specified XLSX file.

* Additionally, it also supports obtaining visualized images and prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the predicted <code>json</code> format result</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get the visualized image in <code>dict</code> format</td>
</tr>
</table>

- The prediction result obtained by the `json` attribute is a dict type of data, with content consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is a dictionary type of data. The keys are `table_res_img`, `ocr_res_img`, `layout_res_img`, and `preprocessed_img`, and the corresponding values are four `Image.Image` objects, in order: visualized image of table recognition result, visualized image of OCR result, visualized image of layout region detection result, and visualized image of image preprocessing. If a sub-module is not used, the corresponding result image is not included in the dictionary.

In addition, you can obtain the general table recognition production line v2 configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config table_recognition_v2 --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the General Table Recognition Pipeline v2. Simply modify the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file. The example is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline_name="./my_path/table_recognition_v2.yaml")

output = pipeline.predict(
    input="table_recognition.jpg",
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

<b>Note:</b> The parameters in the configuration file are the initialization parameters for the pipeline. If you want to change the initialization parameters of the General Table Recognition Pipeline v2, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in the configuration file by specifying the path with `--pipeline`.

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, you can refer to the example code in [2.2 Integration via Python Script](#22-integration-via-python-script).

In addition, PaddleX also provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent performance requirements (especially response speed) for deployment strategies to ensure efficient system operation and smooth user experience. Therefore, PaddleX provides a high-performance inference plugin designed to deeply optimize the performance of model inference and pre/post-processing, significantly speeding up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service Deployment</b>: Service deployment is a common form of deployment in actual production environments. By encapsulating inference functions as services, clients can access these services via network requests to obtain inference results. PaddleX supports multiple pipeline service deployment solutions. For detailed pipeline service deployment procedures, please refer to the [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references and multi-language service call examples for basic service deployment:

<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is successfully processed, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
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
<td>Error code. Fixed to <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed to <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>The result of the operation.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not successfully processed, the attributes of the response body are as follows:</li>
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
<p>Locate and recognize tables in the image.</p>
<p><code>POST /table-recognition</code></p>
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
<td>The URL of an image or PDF file accessible by the server, or the Base64-encoded content of the file. For PDF files with more than 10 pages, only the first 10 pages will be used.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code></td>
<td>The type of the file. <code>0</code> indicates a PDF file, and <code>1</code> indicates an image file. If this attribute is not provided in the request body, the file type will be inferred from the URL.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> in the response body has the following attributes:</li>
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
<td><code>tableRecResults</code></td>
<td><code>object</code></td>
<td>The result of table recognition. The length of the array is 1 (for image input) or the smaller of the number of document pages and 10 (for PDF input). For PDF input, each element in the array represents the processing result of each page in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>tableRecResults</code> is an <code>object</code> with the following attributes:</p>
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
<td><code>tables</code></td>
<td><code>array</code></td>
<td>The location and content of the tables.</td>
</tr>
<tr>
<td><code>layoutImage</code></td>
<td><code>string</code></td>
<td>The result image of layout region detection. The image is in JPEG format and encoded using Base64.</td>
</tr>
<tr>
<td><code>ocrImage</code></td>
<td><code>string</code></td>
<td>The OCR result image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>tables</code> is an <code>object</code> with the following attributes:</p>
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
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>The location of the table. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner.</td>
</tr>
<tr>
<td><code>html</code></td>
<td><code>string</code></td>
<td>The table recognition result in HTML format.</td>
</tr>
</tbody>
</table></details>
<details><summary>Multi-Language Service Invocation Examples</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/table-recognition"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["tableRecResults"]):
    print("Detected tables:")
    print(res["tables"])
    layout_img_path = f"layout_{i}.jpg"
    with open(layout_img_path, "wb") as f:
        f.write(base64.b64decode(res["layoutImage"]))
    ocr_img_path = f"ocr_{i}.jpg"
    with open(ocr_img_path, "wb") as f:
        f.write(base64.b64decode(res["ocrImage"]))
    print(f"Output images saved at {layout_img_path} and {ocr_img_path}")
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on user devices, allowing them to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model production line into subsequent AI applications.

## 4. Custom Development
If the default model weights provided by the General Table Recognition Production Line v2 do not meet your requirements in terms of accuracy or speed, you can try to further <b>fine-tune</b> the existing models using <b>your own domain-specific or application data</b> to improve the recognition performance of the General Table Recognition Production Line v2 in your specific scenario.

### 4.1 Model Fine-Tuning
Since the General Table Recognition Production Line v2 consists of several modules, if the overall performance is not satisfactory, the issue may lie in any one of these modules. You can analyze the images with poor recognition results to identify which module is problematic and refer to the corresponding fine-tuning tutorial links in the table below.

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
<td>Table classification errors</td>
<td>Table Classification Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/table_classification.en.md">Link</a></td>
</tr>
<tr>
<td>Table cell localization errors</td>
<td>Table Cell Detection Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/table_cells_detection.en.md">Link</a></td>
</tr>
<tr>
<td>Table structure recognition errors</td>
<td>Table Structure Recognition Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/table_structure_recognition.en.md">Link</a></td>
</tr>
<tr>
<td>Failure to detect table regions</td>
<td>Layout Region Detection Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/layout_detection.en.md">Link</a></td>
</tr>
<tr>
<td>Missing text detection</td>
<td>Text Detection Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/text_detection.en.md">Link</a></td>
</tr>
<tr>
<td>Inaccurate text content</td>
<td>Text Recognition Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/text_recognition.en.md">Link</a></td>
</tr>
<tr>
<td>Inaccurate whole-image rotation correction</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.en.md">Link</a></td>
</tr>
<tr>
<td>Inaccurate image distortion correction</td>
<td>Text Image Correction Module</td>
<td>Fine-tuning not supported</td>
</tr>
</tbody>
</table>

### 4.2 Model Application
After fine-tuning with your private dataset, you can obtain the local model weight file.

To use the fine-tuned model weights, simply modify the production line configuration file by replacing the local path of the fine-tuned model weights in the corresponding position in the configuration file:

```yaml
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PicoDet_layout_1x_table
    model_dir: null # 替换为微调后的版面区域检测模型权重路径

  TableClassification:
    module_name: table_classification
    model_name: PP-LCNet_x1_0_table_cls
    model_dir: null # 替换为微调后的表格分类模型权重路径

  WiredTableStructureRecognition:
    module_name: table_structure_recognition
    model_name: SLANeXt_wired
    model_dir: null # 替换为微调后的有线表格结构识别模型权重路径
  
  WirelessTableStructureRecognition:
    module_name: table_structure_recognition
    model_name: SLANeXt_wireless
    model_dir: null # 替换为微调后的无线表格结构识别模型权重路径
  
  WiredTableCellsDetection:
    module_name: table_cells_detection
    model_name: RT-DETR-L_wired_table_cell_det
    model_dir: null # 替换为微调后的有线表格单元格检测模型权重路径
  
  WirelessTableCellsDetection:
    module_name: table_cells_detection
    model_name: RT-DETR-L_wireless_table_cell_det
    model_dir: null # 替换为微调后的无线表格单元格检测模型权重路径

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

      DocUnwarping:
        module_name: image_unwarping
        model_name: UVDoc
        model_dir: null

  GeneralOCR:
    pipeline_name: OCR
    text_type: general
    use_doc_preprocessor: False
    use_textline_orientation: False
    SubModules:
      TextDetection:
        module_name: text_detection
        model_name: PP-OCRv4_server_det
        model_dir: null # 替换为微调后的文本检测模型权重路径
        limit_side_len: 960
        limit_type: max
        thresh: 0.3
        box_thresh: 0.6
        unclip_ratio: 2.0
        
      TextRecognition:
        module_name: text_recognition
        model_name: PP-OCRv4_server_rec
        model_dir: null # 替换为微调后文本识别的模型权重路径
        batch_size: 1
        score_thresh: 0
```

Subsequently, refer to the command line method or Python script method in [2.2 Local Experience](#22-本地体验) to load the modified production line configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPU, Kunlun Chip XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to achieve seamless switching between different hardware.

For example, if you use Ascend NPU for OCR production line inference, the Python command used is:

```bash
paddlex --pipeline table_recognition_v2 \
        --input table_recognition.jpg \
        --save_path ./output \
        --device npu:0
```

If you want to use the General Table Recognition Production Line v2 on a wider variety of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).

