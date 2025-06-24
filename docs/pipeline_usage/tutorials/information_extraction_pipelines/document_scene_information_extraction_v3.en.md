---
comments: true
---

# PP-ChatOCRv3-doc Pipeline Tutorial

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
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SLANet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">Training Model</a></td>
<td>59.52</td>
<td>103.08 / 103.08</td>
<td>197.99 / 197.99</td>
<td>6.9 M</td>
<td>SLANet is a table structure recognition model developed by Baidu PaddleX Team. The model significantly improves the accuracy and inference speed of table structure recognition by adopting a CPU-friendly lightweight backbone network PP-LCNet, a high-low-level feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structural and positional information.</td>
</tr>
<tr>
<td>SLANet_plus</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_plus_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">Training Model</a></td>
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
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Training Model</a></td>
<td>86.8</td>
<td>9.03 / 3.10</td>
<td>25.82 / 20.70</td>
<td>7.4</td>
<td>An efficient layout area localization model trained on the PubLayNet dataset based on PicoDet-1x can locate five types of areas, including text, titles, tables, images, and lists.</td>
</tr>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Training Model</a></td>
<td>95.7</td>
<td>8.02 / 3.09</td>
<td>23.70 / 20.41</td>
<td>7.4 M</td>
<td>An efficient layout area localization model trained on the PubLayNet dataset based on PicoDet-1x can locate one type of tables.</td>
</tr>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>87.1</td>
<td>8.99 / 2.22</td>
<td>16.11 / 8.73</td>
<td>4.8</td>
<td>An high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>70.3</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td>A high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>89.3</td>
<td>13.05 / 4.50</td>
<td>41.30 / 41.30</td>
<td>22.6</td>
<td>An efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>79.9</td>
<td>13.50 / 4.69</td>
<td>43.32 / 43.32</td>
<td>22.6</td>
<td>A efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Training Model</a></td>
<td>95.9</td>
<td>114.93 / 27.71</td>
<td>947.56 / 947.56</td>
<td>470.1</td>
<td>A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>92.6</td>
<td>115.29 / 104.09</td>
<td>995.27 / 995.27</td>
<td>470.2</td>
<td>A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
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
<td>PP-OCRv5_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams">Training Model</a></td>
<td>83.8</td>
<td>89.55 / 70.19</td>
<td>371.65 / 371.65</td>
<td>84.3</td>
<td>PP-OCRv5 server-side text detection model with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>79.0</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>PP-OCRv5 mobile-side text detection model with higher efficiency, suitable for deployment on edge devices</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Training Model</a></td>
<td>69.2</td>
<td>83.34 / 80.91</td>
<td>442.58 / 442.58</td>
<td>109</td>
<td>PP-OCRv4 server-side text detection model with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Training Model</a></td>
<td>63.8</td>
<td>8.79 / 3.13</td>
<td>51.00 / 28.58</td>
<td>4.7</td>
<td>PP-OCRv4 mobile-side text detection model with higher efficiency, suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>

<p><b>Text Recognition Module Models</b>:</p>
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
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>78.20</td>
<td>4.82 / 4.82</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td rowspan="2">PP-OCRv4 is the next version of Baidu PaddlePaddle's self-developed text recognition model PP-OCRv3. By introducing data augmentation schemes and GTC-NRTR guidance branches, it further improves text recognition accuracy without compromising inference speed. The model offers both server (server) and mobile (mobile) versions to meet industrial needs in different scenarios.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training Model</a></td>
<td>79.20</td>
<td>6.58 / 6.58</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
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
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
<td>68.81</td>
<td>8.08 / 8.08</td>
<td>50.17 / 42.50</td>
<td>73.9 M</td>
<td rowspan="1">
SVTRv2 is a server-side text recognition model developed by the OpenOCR team at the Vision and Learning Lab (FVL) of Fudan University. It won the first prize in the OCR End-to-End Recognition Task of the PaddleOCR Algorithm Model Challenge, with a 6% improvement in end-to-end recognition accuracy compared to PP-OCRv4 on the A-list.
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
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
<td>65.07</td>
<td>5.93 / 5.93</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
<td rowspan="1">
The RepSVTR text recognition model is a mobile-oriented text recognition model based on SVTRv2. It won the first prize in the OCR End-to-End Recognition Task of the PaddleOCR Algorithm Model Challenge, with a 2.5% improvement in end-to-end recognition accuracy compared to PP-OCRv4 on the B-list, while maintaining similar inference speed.
</td>
</tr>
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
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Training Model</a></td>
<td>98.21</td>
<td>74.75 / 67.72</td>
<td>382.55 / 382.55</td>
<td>109</td>
<td>PP-OCRv4's server-side seal text detection model, featuring higher accuracy, suitable for deployment on better-equipped servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Training Model</a></td>
<td>96.47</td>
<td>7.82 / 3.09</td>
<td>48.28 / 23.97</td>
<td>4.6</td>
<td>PP-OCRv4's mobile seal text detection model, offering higher efficiency, suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>

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
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
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
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, with four categories: 0°, 90°, 180°, 270°</td>
</tr>
</tbody>
</table>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
                    <li><strong>Test Dataset：</strong>
                        <ul>
                          <li>Table Structure Recognition Model: PaddleX internally built English table recognition dataset.</li>
                          <li>Layout Detection Model: PaddleOCR's self-built layout analysis dataset, containing 10,000 images of common document types such as Chinese and English papers, magazines, and research reports.</li>
                          <li>Text Detection Model: PaddleOCR's self-built Chinese dataset, covering multiple scenarios including street scenes, web images, documents, and handwriting, with 500 images for detection.</li>
                          <li>Text Recognition Model: PaddleOCR's self-built Chinese dataset, covering multiple scenarios including street scenes, web images, documents, and handwriting, with 11,000 images for text recognition.</li>
                          <li>ch_SVTRv2_rec: <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task</a> A-rank evaluation set.</li>
                          <li> ch_RepSVTR_rec: <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition Task</a> B-rank evaluation set.</li>
                          <li>English Recognition Model: PaddleX self-built English dataset.</li>
                          <li>Multilingual Recognition Model: PaddleX self-built multilingual dataset.</li>
                          <li>Text Line Direction Classification Model: PaddleX self-built dataset, covering multiple scenarios such as certificates and documents, containing 1,000 images.</li>
                          <li>Text Image Rectification Model: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a></li>
                        </ul>
                    </li>
              <li><strong>Hardware Configuration：</strong>
                  <ul>
                      <li>GPU: NVIDIA Tesla T4</li>
                      <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>Other Environments: Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
                  </ul>
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
The pre-trained model pipelines provided by PaddleX allow for quick experience of their effects. You can experience the effect of the Document Scene Information Extraction v3 pipeline online, or use Python to experience it locally.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/182491/webUI) the effect of the Document Scene Information Extraction v3 pipeline, using the demo images provided by the official. For example:

<img src="https://github.com/user-attachments/assets/aa261b2b-b79c-4487-9323-dfcc43c3d581"/>

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use private data to **fine-tune the models in the pipeline online**.

### 2.2 Local Experience
Before using the Document Scene Information Extraction v3 pipeline locally, ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.md). If you wish to selectively install dependencies, please refer to the relevant instructions in the installation guide. The dependency group corresponding to this pipeline is `ie`.

Before performing model inference, you need to prepare the API key for the large language model. PP-ChatOCRv3 supports calling the large model inference service provided by the [Baidu Cloud Qianfan Platform](https://console.bce.baidu.com/qianfan/ais/console/onlineService). You can refer to [Authentication and Authorization](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Um2wxbaps) to obtain the API key from the Qianfan Platform.

After updating the configuration file, you can use a few lines of Python code to complete the quick inference. You can use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png) for testing:

```python
from paddlex import create_pipeline

chat_bot_config={
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key" # your api_key
}

retriever_config={
    "module_name": "retriever",
    "model_name": "embedding-v1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "qianfan",
    "api_key": "api_key" # your api_key
}

pipeline = create_pipeline(pipeline="PP-ChatOCRv3-doc", initial_predictor=False)

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
    visual_info_list,
    flag_save_bytes_vector=True,
    retriever_config=retriever_config,
)
chat_result = pipeline.chat(
    key_list=["驾驶室准乘人数"],
    visual_info=visual_info_list,
    vector_info=vector_info,
    chat_bot_config=chat_bot_config,
    retriever_config=retriever_config,
)
print(chat_result)

```

After running, the output will be as follows:

```
{'chat_res': {'驾驶室准乘人数': '2'}}
```

The prediction process, API descriptions, and output descriptions of PP-ChatOCRv3-doc are as follows:

<details><summary>(1) Call the <code>create_pipeline</code> method to instantiate the PP-ChatOCRv3 pipeline object.</summary>

The relevant parameter descriptions are as follows:

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
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the pipeline (if set simultaneously with <code>pipeline</code>, it has higher priority than <code>pipeline</code>, and the pipeline name must be consistent).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for pipeline inference. Supports specifying specific GPU card numbers, such as "gpu:0", specific card numbers for other hardware, such as "npu:0", and CPU such as "cpu".</td>
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

<details><summary>(2) Call the <code>visual_predict()</code> method of the PP-ChatOCRv3-doc pipeline object to obtain visual prediction results. This method will return a generator.</summary>

The following are the parameters and their descriptions for the `visual_predict()` method:

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
<td>The data to be predicted, supporting multiple input types, required.</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Such as <code>numpy.ndarray</code> representing image data.</li>
<li><b>str</b>: Such as the local path of an image file or PDF file: <code>/root/data/img.jpg</code>; <b>URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">Example</a>; <b>Local directory</b>, which should contain images to be predicted, such as the local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories, PDF files need to be specified to the specific file path).</li>
<li><b>List</b>: List elements need to be of the above types of data, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
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
<li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code>, values as float scaling factors, e.g., <code>{0: (1.1, 2.0)}</code> means cls_id 0 expanding the width by 1.1 times and the height by 2.0 times while keeping the center unchanged</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, initialized to <code>1.0</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>The overlapping box filtering method.</td>
<td><code>str|dict|None</code></td>
<td>
<ul>
<li><b>str</b>: large, small, union. Respectively representing retaining the large box, small box, or both when filtering overlapping boxes.</li>
<li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code> and values as merging modes, e.g., <code>{0: "large", 2: "small"}</li>
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
<td>The side length limit type for text detection images.</td>
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
<td>The detection pixel threshold, where pixels with scores greater than this threshold in the output probability map are considered text pixels.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>0.3</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>The detection box threshold, where a detection result is considered a text region if the average score of all pixels within the border of the result is greater than this threshold.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>0.6</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>The text detection expansion coefficient, which expands the text region using this method. The larger the value, the larger the expansion area.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>2.0</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>The text recognition threshold, where text results with scores greater than this threshold are retained.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>0.0</code>. I.e., no threshold is set.</li>
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
<td>The side length limit type for seal detection images.</td>
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
<td>The detection pixel threshold, where pixels with scores greater than this threshold in the output probability map are considered seal pixels.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>0.3</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_box_thresh</code></td>
<td>The detection box threshold, where a detection result is considered a seal region if the average score of all pixels within the border of the result is greater than this threshold.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>0.6</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_det_unclip_ratio</code></td>
<td>The seal detection expansion coefficient, which expands the seal region using this method. The larger the value, the larger the expansion area.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>2.0</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>seal_rec_score_thresh</code></td>
<td>The seal recognition threshold, where text results with scores greater than this threshold are retained.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code>.</li>
<li><b>None</b>: If set to <code>None</code>, it will default to the value initialized by the pipeline, <code>0.0</code>. I.e., no threshold is set.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</table>
</details>
<details><summary>(3) Process the visual prediction results.</summary>

The prediction result for each sample is of type `dict`, containing two fields: `visual_info` and `layout_parsing_result`. Obtain visual information (including `normal_text_dict`, `table_text_list`, `table_html_list`, etc.) through `visual_info`, and place the information for each sample into the `visual_info_list` list, which will be sent to the large language model later.

Of course, you can also obtain the layout parsing results through `layout_parsing_result`, which contains tables, text, images, etc., contained in the file or image, and supports printing, saving as an image, and saving as a `json` file:

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
<th>Parameters</th>
<th>Parameter Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Prints the result to the terminal</td>
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
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Saves the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, when it is a directory, the saved file name will be consistent with the input file type</td>
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
<td>Saves the visual images of each module in PNG format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supports directory or file path</td>
<td>N/A</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Saves the tables in the file as an HTML file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supports directory or file path</td>
<td>N/A</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Saves the tables in the file as an XLSX file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supports directory or file path</td>
<td>N/A</td>
</tr>
</table>

- Calling the `print()` method will print the result to the terminal. The content printed to the terminal is explained as follows:
    - `input_path`: `(str)` The input path of the image to be predicted

    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF, otherwise it is `None`

    - `model_settings`: `(Dict[str, bool])` Model parameters required for configuring the pipeline

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
      - `input_path`: `(Union[str, None])` The image path received by the image OCR pipeline, saved as `None` when the input is `numpy.ndarray`
      - `model_settings`: `(Dict)` Model configuration parameters for the OCR pipeline
      - `dt_polys`: `(List[numpy.ndarray])` A list of polygon boxes for text detection. Each detection box is represented by a numpy array of 4 vertex coordinates, with a shape of (4, 2) and a data type of int16
      - `dt_scores`: `(List[float])` A list of confidence scores for text detection boxes
      - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the text detection module
        - `limit_side_len`: `(int)` The side length limit during image preprocessing
        - `limit_type`: `(str)` The processing method for the side length limit
        - `thresh`: `(float)` The confidence threshold for text pixel classification
        - `box_thresh`: `(float)` The confidence threshold for text detection boxes
        - `unclip_ratio`: `(float)` The expansion coefficient for text detection boxes
        - `text_type`: `(str)` The type of text detection, currently fixed as "general"

      - `text_type`: `(str)` The type of text detection, currently fixed as "general"
      - `textline_orientation_angles`: `(List[int])` The prediction results of text line orientation classification. Actual angle values are returned when enabled (e.g., [0,0,1])
      - `text_rec_score_thresh`: `(float)` The filtering threshold for text recognition results
      - `rec_texts`: `(List[str])` A list of text recognition results, only including texts with confidence exceeding `text_rec_score_thresh`
      - `rec_scores`: `(List[float])` A list of confidence scores for text recognition, already filtered by `text_rec_score_thresh`
      - `rec_polys`: `(List[numpy.ndarray])` A list of text detection boxes filtered by confidence, with the same format as `dt_polys`

    - `formula_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` A list of formula recognition results, each element is a dictionary
        - `rec_formula`: `(str)` The formula recognition result
        - `rec_polys`: `(numpy.ndarray)` The formula detection box, with a shape of (4, 2) and a dtype of int16
        - `formula_region_id`: `(int)` The region```markdown
- Calling the `save_to_json()` method will save the aforementioned content to the specified `save_path`. If a directory is specified, the save path will be `save_path/{your_img_basename}_res.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, `numpy.array` types will be converted to list form.
- Invoking the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, the layout detection visualization image, global OCR visualization image, reading order visualization image, and other contents will be saved. If a file is specified, it will be saved directly to that file. (Pipelines often involve multiple result images, so it is not recommended to specify a specific file path directly, as multiple images will be overwritten, leaving only the last one.)

In addition, attributes are also supported to obtain visualized images with results and prediction results, as detailed below:

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

- The prediction result obtained by the `json` attribute is data of type `dict`, and its content is consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is data of type `dict`. The keys are `layout_det_res`, `overall_ocr_res`, `text_paragraphs_ocr_res`, `formula_res_region1`, `table_cell_img`, and `seal_res_region1`, and the corresponding values are `Image.Image` objects: used to display the visualized images of layout detection, OCR, OCR text paragraphs, formulas, tables, and seal results, respectively. If optional modules are not used, only `layout_det_res` is included in the dictionary.
</details>

<details><summary>(4) Call the <code>build_vector()</code> method of the PP-ChatOCRv3-doc Pipeline object to construct vectors for text content.</summary>

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
<td>Block size for vector library creation of long texts</td>
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
<td>Configuration parameters for the vector retrieval large model, refer to the "LLM_Retriever" field in the configuration file</td>
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
- `vector`: `(str|list)` The binary content or text content of the text, depending on the values of `flag_save_bytes_vector` and `min_characters`. If `flag_save_bytes_vector=True` and the text length is greater than or equal to the minimum number of characters, binary content is returned; otherwise, the original text is returned.
</details>

<details><summary>(5) Call the <code>chat()</code> method of the PP-ChatOCRv3-doc Pipeline object to extract key information.</summary>

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
<td>Vector information used for retrieval</td>
<td><code>dict</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>min_characters</code></td>
<td>Required minimum number of characters</td>
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
<td>Output format of text results</td>
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
<td>表结果的输出格式</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_rules_str</code></td>
<td>生成表结果的规则</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_few_shot_demo_text_content</code></td>
<td>表少样本演示的文本内容</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>table_few_shot_demo_key_value_list</code></td>
<td>表少样本演示的键值列表</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>chat_bot_config</code></td>
<td>大语言模型配置信息，内容参考产线配置文件“LLM_Chat”字段</td>
<td><code>dict</code></td>
<td>
<code>None</code>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>retriever_config</code></td>
<td>向量检索大模型配置参数,内容参考配置文件中的“LLM_Retriever”字段</td>
<td><code>dict</code></td>
<td>
<code>None</code>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

该方法会将结果打印到终端，打印到终端的内容解释如下：
  - `chat_res`: `(dict)` 提取信息的结果，是一个字典，包含了待抽取的键和对应的值。

</details>

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy in production, you can proceed directly with development integration/deployment.

If you need to apply the pipeline directly in your Python project, you can refer to the sample code in [2.2 Local Experience](#22-local-experience).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Inference**: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin designed to deeply optimize model inference and pre/post-processing, achieving significant speedups in the end-to-end process. For detailed instructions on high-performance inference, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

☁️ **Serving**: Serving is a common deployment form in actual production environments. By encapsulating the inference functionality as a service, clients can access these services through network requests to obtain inference results. PaddleX supports multiple serving solutions for pipelines. For detailed instructions on serving, please refer to the [PaddleX Serving Guide](../../../pipeline_deploy/serving.md).

Below are the API references for basic serving and multi-language service invocation examples:

<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is successfully processed, the response status code is <code>200</code>, and the response body has the following attributes:</li>
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
<li>When the request is not successfully processed, the response body has the following attributes:</li>
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
<li>Attributes of the request body:</li>
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
<td>File type. <code>0</code> represents a PDF file, <code>1</code> represents an image file. If this attribute is not present in the request body, the file type will be inferred based on the URL.</td>
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
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_unwarping</code> parameter of the pipeline object's <code>visual_predict</code> method.</td>
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
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> of the response body has the following attributes:</li>
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
<td>A simplified version of the <code>res</code> field in the JSON representation of the results generated by the pipeline's <code>visual_predict</code> method, with the <code>input_path</code> and the <code>page_index</code> fields removed.</td>
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
<li>Attributes of the request body:</li>
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
<td><code>integer</code> | <code>null</code></td>
<td>Minimum data length to enable the vector database.</td>
<td>No</td>
</tr>
<tr>
<td><code>blockSize</code></td>
<td><code>int</code> | <code>null</code></td>
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
<li>When the request is successfully processed, the <code>result</code> of the response body has the following attributes:</li>
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
<ul>
<li><b><code>chat</code></b></li>
</ul>
<p>Interacts with large language models to extract key information using them.</p>
<p><code>POST /chatocr-chat</code></p>
<ul>
<li>Attributes of the request body:</li>
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
<td><code>boolean</code> | <code>null</code></td>
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
<td>Minimum data length to enable the vector database.</td>
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
<li>When the request is successfully processed, the <code>result</code> of the response body has the following attributes:</li>
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

<pre><code class="language-python">import base64
import pprint
import sys

import requests


API_BASE_URL = "http://0.0.0.0:8080"

file_path = "./demo.jpg"
keys = ["Name"]

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {
    "file": file_data,
    "fileType": 1,
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
        f"Request to chatocr-vector failed with status code {resp_vector.status_code}.",
        file=sys.stderr,
    )
    pprint.pp(resp_vector.json())
    sys.exit(1)
result_vector = resp_vector.json()["result"]

payload = {
    "keyList": keys,
    "visualInfo": result_visual["visualInfo"],
    "useVectorRetrieval": True,
    "vectorInfo": result_vector["vectorInfo"],
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
print("Final result:")
print(result_chat["chatResult"])
</code></pre></details>

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
    std::string imagePath = " ./demo.jpg";
    std::string imageData = encode_image(imagePath);
    json keys = { "Name" };

    json payload_visual = {
        {"file", imageData},
        {"fileType", 1}
    };
    auto resp1 = client.Post("/chatocr-visual", payload_visual.dump(), "application/json");
    if (!resp1 || resp1->status != 200) {
        std::cerr << "chatocr-visual failed: " << (resp1 ? resp1->status : 0) << std::endl;
        return 1;
    }
    json result_visual = json::parse(resp1->body)["result"];

    auto layoutParsingResults = result_visual["layoutParsingResults"];
    for (size_t i = 0; i < layoutParsingResults.size(); ++i) {
        const auto& res = layoutParsingResults[i];
        std::cout << "prunedResult [" << i << "]: " << res["prunedResult"].dump() << "\n";

        if (res.contains("outputImages")) {
            for (auto& [imgName, base64Data] : res["outputImages"].items()) {
                std::string outPath = imgName + "_" + std::to_string(i) + ".jpg";
                std::string decoded = base64::from_base64(base64Data.get<std::string>());
                std::ofstream out(outPath, std::ios::binary);
                out.write(decoded.data(), decoded.size());
                out.close();
                std::cout << "Saved image: " << outPath << std::endl;
            }
        }
    }

    json payload_vector = {
        {"visualInfo", result_visual["visualInfo"]}
    };
    auto resp2 = client.Post("/chatocr-vector", payload_vector.dump(), "application/json");
    if (!resp2 || resp2->status != 200) {
        std::cerr << "chatocr-vector failed: " << (resp2 ? resp2->status : 0) << std::endl;
        return 1;
    }
    json result_vector = json::parse(resp2->body)["result"];

    json payload_chat = {
        {"keyList", keys},
        {"visualInfo", result_visual["visualInfo"]},
        {"useVectorRetrieval", true},
        {"vectorInfo", result_vector["vectorInfo"]}
    };
    auto resp3 = client.Post("/chatocr-chat", payload_chat.dump(), "application/json");
    if (!resp3 || resp3->status != 200) {
        std::cerr << "chatocr-chat failed: " << (resp3 ? resp3->status : 0) << std::endl;
        return 1;
    }

    json result_chat = json::parse(resp3->body)["result"];
    std::cout << "\nFinal result:\n" << result_chat["chatResult"] << std::endl;

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

        ObjectNode chatPayload = objectMapper.createObjectNode();
        chatPayload.putArray("keyList").add(keys[0]);
        chatPayload.set("visualInfo", resultVisual.get("visualInfo"));
        chatPayload.put("useVectorRetrieval", true);
        chatPayload.set("vectorInfo", resultVector.get("vectorInfo"));

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
)

func postJSON(url string, payload interface{}) ([]byte, error) {
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        return nil, fmt.Errorf("marshal error: %v", err)
    }

    req, err := http.NewRequest("POST", url, bytes.NewBuffer(payloadBytes))
    if err != nil {
        return nil, fmt.Errorf("request creation error: %v", err)
    }
    req.Header.Set("Content-Type", "application/json")

    client := &http.Client{}
    res, err := client.Do(req)
    if err != nil {
        return nil, fmt.Errorf("http request error: %v", err)
    }
    defer res.Body.Close()

    if res.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("unexpected status code: %d", res.StatusCode)
    }

    return ioutil.ReadAll(res.Body)
}

func main() {
    apiBase := "http://localhost:8080"
    filePath := "./demo.jpg"
    keys := []string{"Name"}

    imageBytes, err := ioutil.ReadFile(filePath)
    if err != nil {
        fmt.Println("Error reading file:", err)
        return
    }
    imageBase64 := base64.StdEncoding.EncodeToString(imageBytes)

    visualPayload := map[string]interface{}{
        "file":     imageBase64,
        "fileType": 1,
    }

    visualResp, err := postJSON(apiBase+"/chatocr-visual", visualPayload)
    if err != nil {
        fmt.Println("chatocr-visual failed:", err)
        return
    }

    var visualResult struct {
        Result struct {
            LayoutParsingResults []struct {
                PrunedResult interface{}            `json:"prunedResult"`
                OutputImages map[string]string      `json:"outputImages"`
            } `json:"layoutParsingResults"`
            VisualInfo interface{} `json:"visualInfo"`
        } `json:"result"`
    }
    json.Unmarshal(visualResp, &visualResult)

    for i, res := range visualResult.Result.LayoutParsingResults {
        fmt.Println("PrunedResult:", res.PrunedResult)
        for name, img := range res.OutputImages {
            imgBytes, _ := base64.StdEncoding.DecodeString(img)
            filename := fmt.Sprintf("%s_%d.jpg", name, i)
            ioutil.WriteFile(filename, imgBytes, 0644)
            fmt.Println("Saved image:", filename)
        }
    }

    vectorPayload := map[string]interface{}{
        "visualInfo": visualResult.Result.VisualInfo,
    }
    vectorResp, err := postJSON(apiBase+"/chatocr-vector", vectorPayload)
    if err != nil {
        fmt.Println("chatocr-vector failed:", err)
        return
    }

    var vectorResult struct {
        Result struct {
            VectorInfo interface{} `json:"vectorInfo"`
        } `json:"result"`
    }
    json.Unmarshal(vectorResp, &vectorResult)


    mllmPayload := map[string]interface{}{
        "image":   imageBase64,
        "keyList": keys,
    }

    mllmResp, err := postJSON(apiBase+"/chatocr-mllm", mllmPayload)
    if err != nil {
        fmt.Println("chatocr-mllm failed:", err)
        return
    }

    var mllmResult struct {
        Result struct {
            MllmPredictInfo interface{} `json:"mllmPredictInfo"`
        } `json:"result"`
    }
    json.Unmarshal(mllmResp, &mllmResult)


    chatPayload := map[string]interface{}{
        "keyList":            keys,
        "visualInfo":         visualResult.Result.VisualInfo,
        "useVectorRetrieval": true,
        "vectorInfo":         vectorResult.Result.VectorInfo,
        "mllmPredictInfo":    mllmResult.Result.MllmPredictInfo,
    }

    chatResp, err := postJSON(apiBase+"/chatocr-chat", chatPayload)
    if err != nil {
        fmt.Println("chatocr-chat failed:", err)
        return
    }

    var chatResult struct {
        Result struct {
            ChatResult string `json:"chatResult"`
        } `json:"result"`
    }
    json.Unmarshal(chatResp, &chatResult)
    fmt.Println("Final result:")
    fmt.Println(chatResult.Result.ChatResult)
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

        byte[] fileBytes = File.ReadAllBytes(inputFilePath);
        string fileData = Convert.ToBase64String(fileBytes);
        var payloadVisual = new JObject
        {
            { "file", fileData },
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

        var resultVisual = JObject.Parse(await respVisual.Content.ReadAsStringAsync())["result"];
        Console.WriteLine("Step 1: Layout parsing result:");
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

        var resultVector = JObject.Parse(await respVector.Content.ReadAsStringAsync())["result"];

        var payloadChat = new JObject
        {
            { "keyList", new JArray(keys) },
            { "visualInfo", resultVisual["visualInfo"] },
            { "useVectorRetrieval", true },
            { "vectorInfo", resultVector["vectorInfo"] }
        };

        var respChat = await httpClient.PostAsync($"{API_BASE_URL}/chatocr-chat",
            new StringContent(payloadChat.ToString(), Encoding.UTF8, "application/json"));

        if (!respChat.IsSuccessStatusCode)
        {
            Console.Error.WriteLine($"Request to chatocr-chat failed: {respChat.StatusCode}");
            Console.Error.WriteLine(await respChat.Content.ReadAsStringAsync());
            return;
        }

        var resultChat = JObject.Parse(await respChat.Content.ReadAsStringAsync())["result"];
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
      for (const [imgName, imgBase64] of Object.entries(outputImages)) {
        const fileName = `${imgName}_${i}.jpg`;
        fs.writeFileSync(fileName, Buffer.from(imgBase64, 'base64'));
        console.log(`Output image saved at ${fileName}`);
      }
    });

    const respVector = await axios.post(`${API_BASE_URL}/chatocr-vector`, {
      visualInfo: resultVisual.visualInfo
    });
    const resultVector = respVector.data.result;

    const respChat = await axios.post(`${API_BASE_URL}/chatocr-chat`, {
      keyList: keys,
      visualInfo: resultVisual.visualInfo,
      useVectorRetrieval: true,
      vectorInfo: resultVector.vectorInfo
    });

    const resultChat = respChat.data.result;
    console.log('\nFinal result:\n', resultChat.chatResult);

  } catch (error) {
    if (error.response) {
      console.error(`Request failed with status ${error.response.status}`);
      console.error(error.response.data);
    } else {
      console.error('Error occurred:', error.message);
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
    echo "chatocr-visual request failed\n";
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
    echo "chatocr-vector request failed\n";
    print_r($response_vector);
    exit(1);
}
$result_vector_raw = json_decode($response_vector_raw, false)->result;

$payload_chat = [
    "keyList" => $keys,
    "visualInfo" => $result_visual_raw->visualInfo,
    "useVectorRetrieval" => true,
    "vectorInfo" => $result_vector_raw->vectorInfo
];
$response_chat = send_post_raw("$API_BASE_URL/chatocr-chat", $payload_chat);
$response_chat_data = json_decode($response_chat, true);

if (!isset($response_chat_data["result"])) {
    echo "chatocr-chat request failed\n";
    print_r($response_chat_data);
    exit(1);
}

echo "Final result:\n";
#echo $response_chat_data["result"]["chatResult"] . "\n";
print_r($response_chat_data["result"]["chatResult"]);


function send_post_raw($url, $payload) {
    $json_str = json_encode($payload, JSON_UNESCAPED_UNICODE);
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

📱 **On-Device Deployment**: Edge deployment is a method where computing and data processing functions are placed on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed instructions on edge deployment, please refer to the [PaddleX On-Device Deployment Guide](../../../pipeline_deploy/on_device_deployment.md).
You can choose the appropriate deployment method for your pipeline based on your needs, and proceed with subsequent AI application integration.

## 4. Custom Development

If the default model weights provided by the PP-ChatOCRv3-doc Pipeline do not meet your requirements in terms of accuracy or speed for your specific scenario, you can attempt to further <b>fine-tune</b> the existing models using <b>your own domain-specific or application-specific data</b> to enhance the recognition performance of the general table recognition pipeline in your scenario.

### 4.1 Model Fine-tuning

The document scenario information extraction V3 pipeline consists of several modules. If the performance of the model pipeline does not meet expectations, the issue may originate from any of these modules. You can analyze cases with poor extraction results by visualizing images to determine which module has the problem. Then, refer to the corresponding fine-tuning tutorial links in the table below to fine-tune the model：

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Module to Fine-tune</th>
<th>Fine-tuning Reference Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Inaccurate layout detection, such as undetected stamps or tables</td>
<td>Layout Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html">Link</a></td>
</tr>
<tr>
<td>Inaccurate table structure recognition</td>
<td>Table Structure Recognition</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_structure_recognition.html">Link</a></td>
</tr>
<tr>
<td>Seal text missed</td>
<td>Seal Text Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/seal_text_detection.html">Link</a></td>
</tr>
<tr>
<td>Text missed</td>
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
<td>Whole image rotation correction is inaccurate</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html">Link</a></td>
</tr>
<tr>
<td>Image distortion correction is inaccurate</td>
<td>Text Image Correction Module</td>
<td>Not supported for fine-tuning</td>
</tr>
</tbody>
</table>

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
    device="npu:0" # gpu:0 -->npu:0
    )
```

If you want to use the PP-ChatOCRv3-doc Pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
