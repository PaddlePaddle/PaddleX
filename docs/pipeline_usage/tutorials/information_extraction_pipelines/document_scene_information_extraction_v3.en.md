---
comments: true
---

# PP-ChatOCRv3-doc Pipeline utorial

## 1. Introduction to PP-ChatOCRv3-doc Pipeline
PP-ChatOCRv3-doc is a unique intelligent analysis solution for documents and images developed by PaddlePaddle. It combines Large Language Models (LLM) and OCR technology to provide a one-stop solution for complex document information extraction challenges such as layout analysis, rare characters, multi-page PDFs, tables, and seal recognition. By integrating with ERNIE Bot, it fuses massive data and knowledge to achieve high accuracy and wide applicability.

<img src="https://github.com/user-attachments/assets/90cb740b-7741-4383-bc4c-663f9d042d02"/>

The <b>PP-ChatOCRv3-doc</b> pipeline includes modules for <b>Table Structure Recognition</b>, <b>Layout Region Detection</b>, <b>Text Detection</b>, <b>Text Recognition</b>, <b>Seal Text Detection</b>, <b>Text Image Rectification</b>, and <b>Document Image Orientation Classification</b>.

<b>If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, choose a model with faster inference speed. If you prioritize model storage size, choose a model with a smaller storage size.</b> Some benchmarks for these models are as follows:

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
PaddleX's pre-trained model pipelines can be quickly experienced. You can experience the effect of the Document Scene Information Extraction v3 pipeline online or locally using Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/182491/webUI) the effect of the Document Scene Information Extraction v3 pipeline, using the official demo images for recognition, for example:

<img src="https://github.com/user-attachments/assets/aa261b2b-b79c-4487-9323-dfcc43c3d581"/>

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to <b>fine-tune the models in the pipeline online</b>.

### 2.2 Local Experience
Before using the PP-ChatOCRv3-doc pipeline locally, please ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

A few lines of code are all you need to complete the quick inference of the pipeline. Using the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf), taking the PP-ChatOCRv3-doc pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="PP-ChatOCRv3-doc",
    llm_name="ernie-3.5",
    llm_params={"api_type": "qianfan", "ak": "", "sk": ""} # Please enter your ak and sk; otherwise, the large model cannot be invoked.
    # llm_params={"api_type": "aistudio", "access_token": ""} # Please enter your access_token; otherwise, the large model cannot be invoked.
    )

visual_result, visual_info = pipeline.visual_predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf")

for res in visual_result:
    res.save_to_img("./output")
    res.save_to_html('./output')
    res.save_to_xlsx('./output')

vector = pipeline.build_vector(visual_info=visual_info)

chat_result = pipeline.chat(
    key_list=["乙方", "手机号"],
    visual_info=visual_info,
    vector=vector,
    )
chat_result.print()
```
<b>Note</b>: Currently, the large language model only supports Ernie. You can obtain the relevant ak/sk (access_token) on the [Baidu Cloud Qianfan Platform](https://console.bce.baidu.com/qianfan/ais/console/onlineService) or [Baidu AIStudio Community](https://aistudio.baidu.com/). If you use the Baidu Cloud Qianfan Platform, you can refer to the [AK and SK Authentication API Calling Process](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Hlwerugt8) to obtain ak/sk. If you use Baidu AIStudio Community, you can obtain the access_token from the [Baidu AIStudio Community Access Token](https://aistudio.baidu.com/account/accessToken).

After running, the output is as follows:

```
{'chat_res': {'乙方': '股份测试有限公司', '手机号': '19331729920'}, 'prompt': ''}
```

In the above Python script, the following steps are executed:

(1) Call the `create_pipeline` to instantiate a PP-ChatOCRv3-doc pipeline object, related parameters descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Type</th>
<th>Default</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>str</td>
<td>None</td>
<td>Pipeline name or pipeline configuration file path. If it's a pipeline name, it must be supported by PaddleX;</td>
</tr>
<tr>
<td><code>llm_name</code></td>
<td>str</td>
<td>"ernie-3.5"</td>
<td>Large Language Model name, we support <code>ernie-4.0</code> and <code>ernie-3.5</code>, with more models on the way.</td>
</tr>
<tr>
<td><code>llm_params</code></td>
<td>dict</td>
<td><code>{}</code></td>
<td>API configuration;</td>
</tr>
<tr>
<td><code>device(kwargs)</code></td>
<td>str/<code>None</code></td>
<td><code>None</code></td>
<td>Running device, support <code>cpu</code>, <code>gpu</code>, <code>gpu:0</code>, etc. <code>None</code> meaning automatic selection;</td>
</tr>
</tbody>
</table>
(2) Call the `visual_predict` of the PP-ChatOCRv3-doc pipeline object to visual predict, related parameters descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Type</th>
<th>Default</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>Python Var</td>
<td>-</td>
<td>Support to pass Python variables directly, such as <code>numpy.ndarray</code> representing image data;</td>
</tr>
<tr>
<td><code>input</code></td>
<td>str</td>
<td>-</td>
<td>Support to pass the path of the file to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>;</td>
</tr>
<tr>
<td><code>input</code></td>
<td>str</td>
<td>-</td>
<td>Support to pass the URL of the file to be predicted, such as: <code>https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf</code>;</td>
</tr>
<tr>
<td><code>input</code></td>
<td>str</td>
<td>-</td>
<td>Support to pass the local directory, which should contain files to be predicted, such as: <code>/root/data/</code>;</td>
</tr>
<tr>
<td><code>input</code></td>
<td>dict</td>
<td>-</td>
<td>Support to pass a dictionary, where the key needs to correspond to the specific pipeline, such as: <code>{"img": "/root/data1"}</code>；</td>
</tr>
<tr>
<td><code>input</code></td>
<td>list</td>
<td>-</td>
<td>Support to pass a list, where the elements must be of the above types of data, such as: <code>[numpy.ndarray, numpy.ndarray]</code>，<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>，<code>["/root/data1", "/root/data2"]</code>，<code>[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>；</td>
</tr>
<tr>
<td><code>use_doc_image_ori_cls_model</code></td>
<td>bool</td>
<td><code>True</code></td>
<td>Whether or not to use the orientation classification model;</td>
</tr>
<tr>
<td><code>use_doc_image_unwarp_model</code></td>
<td>bool</td>
<td><code>True</code></td>
<td>Whether or not to use the unwarp model;</td>
</tr>
<tr>
<td><code>use_seal_text_det_model</code></td>
<td>bool</td>
<td><code>True</code></td>
<td>Whether or not to use the seal text detection model;</td>
</tr>
</tbody>
</table>
(3) Call the relevant functions of prediction object to save the prediction results. The related functions are as follows:

<table>
<thead>
<tr>
<th>Function</th>
<th>Parameter</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>save_to_img</code></td>
<td><code>save_path</code></td>
<td>Save OCR prediction results, layout results, and table recognition results as image files, with the parameter <code>save_path</code> used to specify the save path;</td>
</tr>
<tr>
<td><code>save_to_html</code></td>
<td><code>save_path</code></td>
<td>Save the table recognition results as an HTML file, with the parameter 'save_path' used to specify the save path;</td>
</tr>
<tr>
<td><code>save_to_xlsx</code></td>
<td><code>save_path</code></td>
<td>Save the table recognition results as an Excel file, with the parameter 'save_path' used to specify the save path;</td>
</tr>
</tbody>
</table>
(4) Call the `chat` of PP-ChatOCRv3-doc pipeline object to query information with LLM, related parameters are described as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Type</th>
<th>Default</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>key_list</code></td>
<td>str</td>
<td>-</td>
<td>Keywords used to query. A string composed of multiple keywords with "," as separators, such as "Party B, phone number";</td>
</tr>
<tr>
<td><code>key_list</code></td>
<td>list</td>
<td>-</td>
<td>Keywords used to query. A list composed of multiple keywords.</td>
</tr>
</tbody>
</table>
(3) Obtain prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through calls. The `predict` method predicts data in batches, so the prediction results are represented as a list of prediction results.

(4) Interact with the large model by calling the `predict.chat` method, which takes as input keywords (multiple keywords are supported) for information extraction. The prediction results are represented as a list of information extraction results.

(5) Process the prediction results: The prediction result for each sample is in the form of a dict, which supports printing or saving to a file. The supported file types depend on the specific pipeline, such as:

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
<td>save_to_img</td>
<td>Saves layout analysis, table recognition, etc. results as image files.</td>
<td><code>save_path</code>: str, the file path to save.</td>
</tr>
<tr>
<td>save_to_html</td>
<td>Saves table recognition results as HTML files.</td>
<td><code>save_path</code>: str, the file path to save.</td>
</tr>
<tr>
<td>save_to_xlsx</td>
<td>Saves table recognition results as Excel files.</td>
<td><code>save_path</code>: str, the file path to save.</td>
</tr>
</tbody>
</table>
When executing the above command, the default Pipeline configuration file is loaded. If you need to customize the configuration file, you can use the following command to obtain it:

```bash
paddlex --get_pipeline_config PP-ChatOCRv3-doc
```

After execution, the configuration file for the PP-ChatOCRv3-doc pipeline will be saved in the current path. If you wish to customize the save location, you can execute the following command (assuming the custom save location is `./my_path`):

```bash
paddlex --get_pipeline_config PP-ChatOCRv3-doc --save_path ./my_path
```
After obtaining the configuration file, you can customize the various configurations of the PP-ChatOCRv3-doc pipeline:

```yaml
Pipeline:
  layout_model: RT-DETR-H_layout_3cls
  table_model: SLANet_plus
  text_det_model: PP-OCRv4_server_det
  text_rec_model: PP-OCRv4_server_rec
  seal_text_det_model: PP-OCRv4_server_seal_det
  doc_image_ori_cls_model: null
  doc_image_unwarp_model: null
  llm_name: "ernie-3.5"
  llm_params:
    api_type: qianfan
    ak:
    sk:
```

In the above configuration, you can modify the models loaded by each module of the pipeline, as well as the large language model used. Please refer to the module documentation for the list of supported models for each module, and the list of supported large language models includes: ernie-4.0, ernie-3.5, ernie-3.5-8k, ernie-lite, ernie-tiny-8k, ernie-speed, ernie-speed-128k, ernie-char-8k.

After making modifications, simply update the `pipeline` parameter value in the `create_pipeline` method to the path of your pipeline configuration file to apply the configuration.

For example, if your configuration file is saved at `./my_path/PP-ChatOCRv3-doc.yaml`, you would execute:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="./my_path/PP-ChatOCRv3-doc.yaml",
    llm_name="ernie-3.5",
    llm_params={"api_type": "qianfan", "ak": "", "sk": ""} # Please enter your ak and sk; otherwise, the large model cannot be invoked.
    # llm_params={"api_type": "aistudio", "access_token": ""} # Please enter your access_token; otherwise, the large model cannot be invoked.
    )

visual_result, visual_info = pipeline.visual_predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf")

for res in visual_result:
    res.save_to_img("./output")
    res.save_to_html('./output')
    res.save_to_xlsx('./output')

vector = pipeline.build_vector(visual_info=visual_info)

chat_result = pipeline.chat(
    key_list=["乙方", "手机号"],
    visual_info=visual_info,
    vector=vector,
    )
chat_result.print()
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to directly apply the pipeline in your Python project, you can refer to the example code in [2.2 Local Experience](#22-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics (especially response speed) of deployment strategies to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

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
<p>Primary operations provided by the service are as follows:</p>
<ul>
<li><b><code>analyzeImages</code></b></li>
</ul>
<p>Analyze images using computer vision models to obtain OCR, table recognition results, and extract key information from the images.</p>
<p><code>POST /chatocr-visual</code></p>
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
<td>The URL of an accessible image file or PDF file, or the Base64 encoded content of the above file types. For PDF files with more than 10 pages, only the first 10 pages will be used.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code></td>
<td>File type. <code>0</code> represents PDF files, <code>1</code> represents image files. If this property is not present in the request body, the file type will be inferred based on the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useImgOrientationCls</code></td>
<td><code>boolean</code></td>
<td>Whether to enable document image orientation classification. This feature is enabled by default.</td>
<td>No</td>
</tr>
<tr>
<td><code>useImgUnwarping</code></td>
<td><code>boolean</code></td>
<td>Whether to enable text image correction. This feature is enabled by default.</td>
<td>No</td>
</tr>
<tr>
<td><code>useSealTextDet</code></td>
<td><code>boolean</code></td>
<td>Whether to enable seal text detection. This feature is enabled by default.</td>
<td>No</td>
</tr>
<tr>
<td><code>inferenceParams</code></td>
<td><code>object</code></td>
<td>Inference parameters.</td>
<td>No</td>
</tr>
</tbody>
</table>
<p>Properties of <code>inferenceParams</code>:</p>
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
<td><code>maxLongSide</code></td>
<td><code>integer</code></td>
<td>During inference, if the length of the longer side of the input image for the text detection model is greater than <code>maxLongSide</code>, the image will be scaled so that the length of the longer side equals <code>maxLongSide</code>.</td>
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
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>visualResults</code></td>
<td><code>array</code></td>
<td>Analysis results obtained using the computer vision model. The array length is 1 (for image input) or the smaller of the number of document pages and 10 (for PDF input). For PDF input, each element in the array represents the processing result of each page in the PDF file in sequence.</td>
</tr>
<tr>
<td><code>visualInfo</code></td>
<td><code>object</code></td>
<td>Key information in the image, which can be used as input for other operations.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>visualResults</code> is an <code>object</code> with the following properties:</p>
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
<td><code>texts</code></td>
<td><code>array</code></td>
<td>Text locations, contents, and scores.</td>
</tr>
<tr>
<td><code>tables</code></td>
<td><code>array</code></td>
<td>Table locations and contents.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code></td>
<td>Input image. The image is in JPEG format and encoded in Base64.</td>
</tr>
<tr>
<td><code>ocrImage</code></td>
<td><code>string</code></td>
<td>OCR result image. The image is in JPEG format and encoded in Base64.</td>
</tr>
<tr>
<td><code>layoutImage</code></td>
<td><code>string</code></td>
<td>Layout area detection result image. The image is in JPEG format and encoded in Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>texts</code> is an <code>object</code> with the following properties:</p>
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
<td><code>poly</code></td>
<td><code>array</code></td>
<td>Text location. The elements in the array are the vertex coordinates of the polygon enclosing the text in sequence.</td>
</tr>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>Text content.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>Text recognition score.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>tables</code> is an <code>object</code> with the following properties:</p>
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
<td>Table location. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner of the bounding box in sequence.</td>
</tr>
<tr>
<td><code>html</code></td>
<td><code>string</code></td>
<td>Table recognition result in HTML format.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>buildVectorStore</code></b></li>
</ul>
<p>Builds a vector database.</p>
<p><code>POST /chatocr-vector</code></p>
<ul>
<li>The request body properties are as follows:</li>
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
<td><code>visualInfo</code></td>
<td><code>object</code></td>
<td>Key information from the image. Provided by the <code>analyzeImages</code> operation.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>minChars</code></td>
<td><code>integer</code></td>
<td>Minimum data length to enable the vector database.</td>
<td>No</td>
</tr>
<tr>
<td><code>llmRequestInterval</code></td>
<td><code>number</code></td>
<td>Interval time for calling the large language model API.</td>
<td>No</td>
</tr>
<tr>
<td><code>llmName</code></td>
<td><code>string</code></td>
<td>Name of the large language model.</td>
<td>No</td>
</tr>
<tr>
<td><code>llmParams</code></td>
<td><code>object</code></td>
<td>Parameters for the large language model API.</td>
<td>No</td>
</tr>
</tbody>
</table>
<p>Currently, <code>llmParams</code> can take one of the following forms:</p>
<pre><code class="language-json">{
"apiType": "qianfan",
"apiKey": "{Qianfan Platform API key}",
"secretKey": "{Qianfan Platform secret key}"
}
</code></pre>
<pre><code class="language-json">{
"apiType": "aistudio",
"accessToken": "{AI Studio access token}"
}
</code></pre>
<ul>
<li>When the request is processed successfully, the <code>result</code> in the response body has the following properties:</li>
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
<td><code>vectorStore</code></td>
<td><code>string</code></td>
<td>Serialized result of the vector database, which can be used as input for other operations.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>retrieveKnowledge</code></b></li>
</ul>
<p>Perform knowledge retrieval.</p>
<p><code>POST /chatocr-retrieval</code></p>
<ul>
<li>The request body properties are as follows:</li>
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
<td><code>keys</code></td>
<td><code>array</code></td>
<td>List of keywords.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>vectorStore</code></td>
<td><code>string</code></td>
<td>Serialized result of the vector database. Provided by the <code>buildVectorStore</code> operation.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>llmName</code></td>
<td><code>string</code></td>
<td>Name of the large language model.</td>
<td>No</td>
</tr>
<tr>
<td><code>llmParams</code></td>
<td><code>object</code></td>
<td>API parameters for the large language model.</td>
<td>No</td>
</tr>
</tbody>
</table>
<p>Currently, <code>llmParams</code> can take one of the following forms:</p>
<pre><code class="language-json">{
"apiType": "qianfan",
"apiKey": "{Qianfan Platform API key}",
"secretKey": "{Qianfan Platform secret key}"
}
</code></pre>
<pre><code class="language-json">{
"apiType": "aistudio",
"accessToken": "{AI Studio access token}"
}
</code></pre>
<ul>
<li>When the request is processed successfully, the <code>result</code> in the response body has the following properties:</li>
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
<td><code>retrievalResult</code></td>
<td><code>string</code></td>
<td>The result of knowledge retrieval, which can be used as input for other operations.</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>chat</code></b></li>
</ul>
<p>Interact with large language models to extract key information.</p>
<p><code>POST /chatocr-chat</code></p>
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
<td><code>keys</code></td>
<td><code>array</code></td>
<td>List of keywords.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>visualInfo</code></td>
<td><code>object</code></td>
<td>Key information from images. Provided by the <code>analyzeImages</code> operation.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>vectorStore</code></td>
<td><code>string</code></td>
<td>Serialized result of the vector database. Provided by the <code>buildVectorStore</code> operation.</td>
<td>No</td>
</tr>
<tr>
<td><code>retrievalResult</code></td>
<td><code>string</code></td>
<td>Results of knowledge retrieval. Provided by the <code>retrieveKnowledge</code> operation.</td>
<td>No</td>
</tr>
<tr>
<td><code>taskDescription</code></td>
<td><code>string</code></td>
<td>Task prompt.</td>
<td>No</td>
</tr>
<tr>
<td><code>rules</code></td>
<td><code>string</code></td>
<td>Custom extraction rules, e.g., for output formatting.</td>
<td>No</td>
</tr>
<tr>
<td><code>fewShot</code></td>
<td><code>string</code></td>
<td>Example prompts.</td>
<td>No</td>
</tr>
<tr>
<td><code>llmName</code></td>
<td><code>string</code></td>
<td>Name of the large language model.</td>
<td>No</td>
</tr>
<tr>
<td><code>llmParams</code></td>
<td><code>object</code></td>
<td>API parameters for the large language model.</td>
<td>No</td>
</tr>
<tr>
<td><code>returnPrompts</code></td>
<td><code>boolean</code></td>
<td>Whether to return the prompts used. Enabled by default.</td>
<td>No</td>
</tr>
</tbody>
</table>
<p>Currently, <code>llmParams</code> can take one of the following forms:</p>
<pre><code class="language-json">{
"apiType": "qianfan",
"apiKey": "{Qianfan Platform API key}",
"secretKey": "{Qianfan Platform secret key}"
}
</code></pre>
<pre><code class="language-json">{
"apiType": "aistudio",
"accessToken": "{AI Studio access token}"
}
</code></pre>
<ul>
<li>On successful request processing, the <code>result</code> in the response body has the following properties:</li>
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
<td><code>chatResult</code></td>
<td><code>object</code></td>
<td>Extracted key information.</td>
</tr>
<tr>
<td><code>prompts</code></td>
<td><code>object</code></td>
<td>Prompts used.</td>
</tr>
</tbody>
</table>
<p>Properties of <code>prompts</code>:</p>
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
<td><code>ocr</code></td>
<td><code>array</code></td>
<td>OCR prompts.</td>
</tr>
<tr>
<td><code>table</code></td>
<td><code>array</code></td>
<td>Table prompts.</td>
</tr>
<tr>
<td><code>html</code></td>
<td><code>array</code></td>
<td>HTML prompts.</td>
</tr>
</tbody>
</table></details>
<details><summary>Multi-Language Service Invocation Examples</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import pprint
import sys

import requests

API_BASE_URL = "http://0.0.0.0:8080"
API_KEY = "{Qianfan Platform API key}"
SECRET_KEY = "{Qianfan Platform secret key}"
LLM_NAME = "ernie-3.5"
LLM_PARAMS = {
    "apiType": "qianfan",
    "apiKey": API_KEY,
    "secretKey": SECRET_KEY,
}

file_path = "./demo.jpg"
keys = ["电话"]

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {
    "file": file_data,
    "fileType": 1,
    "useImgOrientationCls": True,
    "useImgUnwarping": True,
    "useSealTextDet": True,
}
resp_visual = requests.post(url=f"{API_BASE_URL}/chatocr-visual", json=payload)
if resp_visual.status_code != 200:
    print(
        f"Request to chatocr-visual failed with status code {resp_visual.status_code}.",
        file=sys.stderr,
    )
    pprint.pp(resp_visual.json())
    sys.exit(1)
result_visual = resp_visual.json()["result"]

for i, res in enumerate(result_visual["visualResults"]):
    print("Texts:")
    pprint.pp(res["texts"])
    print("Tables:")
    pprint.pp(res["tables"])
    layout_img_path = f"layout_{i}.jpg"
    with open(layout_img_path, "wb") as f:
        f.write(base64.b64decode(res["layoutImage"]))
    ocr_img_path = f"ocr_{i}.jpg"
    with open(ocr_img_path, "wb") as f:
        f.write(base64.b64decode(res["ocrImage"]))
    print(f"Output images saved at {layout_img_path} and {ocr_img_path}")

payload = {
    "visualInfo": result_visual["visualInfo"],
    "minChars": 200,
    "llmRequestInterval": 1000,
    "llmName": LLM_NAME,
    "llmParams": LLM_PARAMS,
}
resp_vector = requests.post(url=f"{API_BASE_URL}/chatocr-vector", json=payload)
if resp_vector.status_code != 200:
    print(
        f"Request to chatocr-vector failed with status code {resp_vector.status_code}.",
        file=sys.stderr,
    )
    pprint.pp(resp_vector.json())
    sys.exit(1)
result_vector = resp_vector.json()["result"]

payload = {
    "keys": keys,
    "vectorStore": result_vector["vectorStore"],
    "llmName": LLM_NAME,
    "llmParams": LLM_PARAMS,
}
resp_retrieval = requests.post(url=f"{API_BASE_URL}/chatocr-retrieval", json=payload)
if resp_retrieval.status_code != 200:
    print(
        f"Request to chatocr-retrieval failed with status code {resp_retrieval.status_code}.",
        file=sys.stderr,
    )
    pprint.pp(resp_retrieval.json())
    sys.exit(1)
result_retrieval = resp_retrieval.json()["result"]

payload = {
    "keys": keys,
    "visualInfo": result_visual["visualInfo"],
    "vectorStore": result_vector["vectorStore"],
    "retrievalResult": result_retrieval["retrievalResult"],
    "taskDescription": "",
    "rules": "",
    "fewShot": "",
    "llmName": LLM_NAME,
    "llmParams": LLM_PARAMS,
    "returnPrompts": True,
}
resp_chat = requests.post(url=f"{API_BASE_URL}/chatocr-chat", json=payload)
if resp_chat.status_code != 200:
    print(
        f"Request to chatocr-chat failed with status code {resp_chat.status_code}.",
        file=sys.stderr,
    )
    pprint.pp(resp_chat.json())
    sys.exit(1)
result_chat = resp_chat.json()["result"]
print("\nPrompts:")
pprint.pp(result_chat["prompts"])
print("Final result:")
print(result_chat["chatResult"])
</code></pre>
<b>Note</b>: Please fill in your API key and secret key at `API_KEY` and `SECRET_KEY`.</details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computing and data processing functions on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).

## 4. Custom Development

If the default model weights provided by the PP-ChatOCRv3-doc Pipeline do not meet your requirements in terms of accuracy or speed for your specific scenario, you can attempt to further <b>fine-tune</b> the existing models using <b>your own domain-specific or application-specific data</b> to enhance the recognition performance of the general table recognition pipeline in your scenario.

### 4.1 Model Fine-tuning

Since the PP-ChatOCRv3-doc Pipeline comprises six modules, unsatisfactory performance may stem from any of these modules (note that the text image rectification module does not support customization at this time).

You can analyze images with poor recognition results and follow the guidelines below for analysis and model fine-tuning:

* Incorrect table structure detection (e.g., row/column misidentification, cell position errors) may indicate deficiencies in the table structure recognition module. You need to refer to the <b>Customization</b> section in the [Table Structure Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.en.md) and fine-tune the table structure recognition model using your private dataset.

* Misplaced layout elements (e.g., incorrect positioning of tables or seals) may suggest issues with the layout detection module. Consult the <b>Customization</b> section in the [Layout Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/layout_detection.en.md) and fine-tune the layout detection model with your private dataset.

* Frequent undetected text (i.e., text leakage) may indicate limitations in the text detection model. Refer to the <b>Customization</b> section in the [Text Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_detection.en.md) and fine-tune the text detection model using your private dataset.

* High text recognition errors (i.e., recognized text content does not match the actual text) suggest that the text recognition model requires improvement. Follow the <b>Customization</b> section in the [Text Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_recognition.en.md) to fine-tune the text recognition model.

* Frequent recognition errors in detected seal text indicate that the seal text detection model needs further refinement. Consult the <b>Customization</b> section in the [Seal Text Detection Module Development Tutorials](../../../module_usage/tutorials/ocr_modules/seal_text_detection.en.md) to fine-tune the seal text detection model.

* Frequent misidentifications of document or certificate orientations with text regions suggest that the document image orientation classification model requires improvement. Refer to the <b>Customization</b> section in the [Document Image Orientation Classification Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.en.md) to fine-tune the document image orientation classification model.

### 4.2 Model Deployment

After fine-tuning your models using your private dataset, you will obtain local model weights files.

To use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local paths of the default model weights with those of your fine-tuned models:

```yaml
......
Pipeline:
  layout_model: RT-DETR-H_layout_3cls  # Replace with the local path of your fine-tuned model
  table_model: SLANet_plus  # Replace with the local path of your fine-tuned model
  text_det_model: PP-OCRv4_server_det  # Replace with the local path of your fine-tuned model
  text_rec_model: PP-OCRv4_server_rec  # Replace with the local path of your fine-tuned model
  seal_text_det_model: PP-OCRv4_server_seal_det  # Replace with the local path of your fine-tuned model
  doc_image_ori_cls_model: null   # Replace with the local path of your fine-tuned model if applicable
  doc_image_unwarp_model: null   # Replace with the local path of your fine-tuned model if applicable
......
```

Subsequently, load the modified pipeline configuration file using the command-line interface or Python script as described in the local experience section.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Seamless switching between different hardware can be achieved by simply setting the `--device` parameter</b>.

For example, to perform inference using the PP-ChatOCRv3-doc Pipeline on an NVIDIA GPU.
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the script to `npu`:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(
    pipeline="PP-ChatOCRv3-doc",
    llm_name="ernie-3.5",
    llm_params={"api_type": "qianfan", "ak": "", "sk": ""},
    device="npu:0" # gpu:0 --> npu:0
    )
```

If you want to use the PP-ChatOCRv3-doc Pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
