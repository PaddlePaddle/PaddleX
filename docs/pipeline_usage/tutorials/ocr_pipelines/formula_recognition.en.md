---
comments: true
---

# Formula Recognition Pipeline Tutorial

## 1. Introduction to Formula Recognition Pipeline

Formula recognition is a technology that automatically identifies and extracts LaTeX formula content and structure from documents or images. It is widely used in fields such as mathematics, physics, and computer science for document editing and data analysis. By using computer vision and machine learning algorithms, formula recognition can convert complex mathematical formula information into editable LaTeX format, facilitating further processing and analysis of data.

The formula recognition pipeline is designed to solve formula recognition tasks by extracting formula information from images and outputting it in LaTeX source code format. This pipeline integrates the advanced formula recognition model PP-FormulaNet developed by the PaddlePaddle Vision Team and the well-known formula recognition model UniMERNet. It is an end-to-end formula recognition system that supports the recognition of simple printed formulas, complex printed formulas, and handwritten formulas. Additionally, it includes functions for image orientation correction and distortion correction. Based on this pipeline, precise formula content prediction can be achieved, covering various application scenarios in education, research, finance, manufacturing, and other fields. The pipeline also provides flexible deployment options, supporting multiple hardware devices and programming languages. Moreover, it offers the capability for custom development. You can train and optimize the pipeline on your own dataset, and the trained model can be seamlessly integrated.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/03.png" style="width: 70%"/>

<b>The formula recognition pipeline includes a mandatory formula recognition module,</b> as well as optional layout detection, document image orientation classification, and text image unwarping modules. The document image orientation classification module and the text image unwarping module are integrated into the formula recognition pipeline as a document preprocessing sub-pipeline. Each module contains multiple models, and you can choose the model based on the benchmark test data below.

### 1.1 Model benchmarkBenchmark dataData

<b>If you prioritize model accuracy, choose a model with higher precision; if you care more about inference speed, choose a faster model; if you are concerned about model storage size, choose a smaller model.</b>

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

<p><b>Layout Detection Module (Optional):</b></p>

* <b>The layout detection model includes 20 common categories: document title, paragraph title, text, page number, abstract, table, references, footnotes, header, footer, algorithm, formula, formula number, image, table, seal, figure_table title, chart, and sidebar text and lists of references</b>
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
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">Training Model</a></td>
<td>83.2</td>
<td>53.03 / 17.23</td>
<td>634.62 / 378.32</td>
<td>126.01</td>
<td>A higher-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, PPT, multi-layout magazines, contracts, books, exams, ancient books and research reports using RT-DETR-L</td>
</tr>
<tr>
</tbody>
</table>

* <b>The layout detection model includes 23 common categories: document title, paragraph title, text, page number, abstract, table of contents, references, footnotes, header, footer, algorithm, formula, formula number, image, figure caption, table, table caption, seal, figure title, figure, header image, footer image, and sidebar text</b>
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

> ❗ The above list includes the <b>4 core models</b> that are key supported by the layout detection module. The module actually supports a total of <b>7 full models</b>, including several predefined models with different categories. The complete model list is as follows:

<details><summary> 👉 Details of Model List</summary>


* <b>17-Class Layout Detection Model, including 17 common layout categories: Paragraph Title, Image, Text, Number, Abstract, Content, Figure Caption, Formula, Table, Table Caption, References, Document Title, Footnote, Header, Algorithm, Footer, and Stamp</b>

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
</tbody></table>

* <b>23-Class Layout Detection Model, including 23 common layout categories: Document Title, Section Title, Text, Page Number, Abstract, Table of Contents, References, Footnotes, Header, Footer, Algorithm, Formula, Formula Number, Image, Figure Caption, Table, Table Caption, Seal, Chart Caption, Chart, Header Image, Footer Image, Sidebar Text</b>

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

* <b>The layout detection model includes 20 common categories: document title, paragraph title, text, page number, abstract, table, references, footnotes, header, footer, algorithm, formula, formula number, image, table, seal, figure_table title, chart, and sidebar text and lists of references</b>

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
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">Training Model</a></td>
<td>83.2</td>
<td>53.03 / 17.23</td>
<td>634.62 / 378.32</td>
<td>126.01</td>
<td>A higher-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, PPT, multi-layout magazines, contracts, books, exams, ancient books and research reports using RT-DETR-L</td>
</tr>
<tr>
</tbody>
</table>

</details>

<p><b>Formula Recognition Module </b></p>
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

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
                    <li><strong>Test Dataset：</strong>
                        <ul>
                          <li>Document Image Orientation Classification Module: A self-built dataset using PaddleX, covering multiple scenarios such as ID cards and documents, containing 1000 images.</li>
                          <li> Text Image Rectification Module: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a></li>
                          <li>Layout Region Detection Module: A self-built layout region detection dataset using PaddleOCR, including 500 images of common document types such as Chinese and English papers, magazines, contracts, books, exam papers, and research reports.</li>
                          <li>17-Class Region Detection Model: A self-built layout region detection dataset using PaddleOCR, including 892 images of common document types such as Chinese and English papers, magazines, and research reports.</li>
                          <li>Formula Recognition Module: A self-built formula recognition test set using PaddleX.</li>
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
<td rowspan="9">formula_recognition-default</td>
<td>Intel 6271C</td>
<td>26.27</td>
<td>1039.70</td>
<td>1003.86</td>
<td>3093.11</td>
<td>2675.83</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>18.24</td>
<td>1048.60</td>
<td>893.64</td>
<td>3029.64</td>
<td>2676.89</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>5.97</td>
<td>120.80</td>
<td>101.63</td>
<td>2255.59</td>
<td>2140.24</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>1.44</td>
<td>111.60</td>
<td>103.08</td>
<td>2145.61</td>
<td>1976.60</td>
<td>44</td>
<td>36.36</td>
<td>2422.00</td>
<td>2422.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>2.03</td>
<td>129.80</td>
<td>104.93</td>
<td>2231.39</td>
<td>2058.09</td>
<td>53</td>
<td>37.64</td>
<td>1888.00</td>
<td>1888.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>1.17</td>
<td>109.90</td>
<td>102.93</td>
<td>2509.55</td>
<td>2355.77</td>
<td>55</td>
<td>44.37</td>
<td>2406.00</td>
<td>2406.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>1.58</td>
<td>115.90</td>
<td>103.61</td>
<td>2442.22</td>
<td>2269.11</td>
<td>64</td>
<td>42.06</td>
<td>1874.00</td>
<td>1874.00</td>
</tr>
<tr>
<td>M4</td>
<td>10.16</td>
<td>126.60</td>
<td>107.78</td>
<td>3135.38</td>
<td>2932.62</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>2.42</td>
<td>115.00</td>
<td>102.81</td>
<td>2350.60</td>
<td>2223.92</td>
<td>79</td>
<td>57.36</td>
<td>1638.00</td>
<td>1638.00</td>
</tr>
<tr>
<td rowspan="9">formula_recognition-nopp</td>
<td>Intel 6271C</td>
<td>24.86</td>
<td>1050.30</td>
<td>1009.22</td>
<td>2715.87</td>
<td>2405.00</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>17.00</td>
<td>1046.60</td>
<td>891.66</td>
<td>2701.11</td>
<td>2371.54</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>2.43</td>
<td>112.90</td>
<td>102.75</td>
<td>1972.57</td>
<td>1933.25</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>1.36</td>
<td>106.90</td>
<td>102.16</td>
<td>2167.37</td>
<td>2100.08</td>
<td>44</td>
<td>36.64</td>
<td>2264.00</td>
<td>2264.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>1.93</td>
<td>112.30</td>
<td>103.38</td>
<td>1878.44</td>
<td>1819.13</td>
<td>52</td>
<td>37.44</td>
<td>1766.00</td>
<td>1766.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>1.09</td>
<td>108.80</td>
<td>102.04</td>
<td>2276.06</td>
<td>2221.90</td>
<td>56</td>
<td>44.67</td>
<td>2270.00</td>
<td>2270.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>1.46</td>
<td>106.30</td>
<td>102.23</td>
<td>1972.24</td>
<td>1916.99</td>
<td>65</td>
<td>43.27</td>
<td>1760.00</td>
<td>1760.00</td>
</tr>
<tr>
<td>M4</td>
<td>13.52</td>
<td>131.60</td>
<td>107.52</td>
<td>2502.56</td>
<td>2295.11</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>2.27</td>
<td>108.00</td>
<td>101.84</td>
<td>1942.30</td>
<td>1887.73</td>
<td>80</td>
<td>58.32</td>
<td>1514.00</td>
<td>1514.00</td>
</tr>
<tr>
<td rowspan="8">formula_recognition-nolayout</td>
<td>Intel 8350C</td>
<td>65.37</td>
<td>1068.60</td>
<td>1022.44</td>
<td>2566.70</td>
<td>2306.92</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>16.89</td>
<td>118.90</td>
<td>100.37</td>
<td>2239.15</td>
<td>2128.04</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>10.31</td>
<td>109.90</td>
<td>100.34</td>
<td>2139.89</td>
<td>1953.87</td>
<td>53</td>
<td>39.47</td>
<td>1520.00</td>
<td>1520.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>15.87</td>
<td>118.80</td>
<td>100.76</td>
<td>2280.21</td>
<td>2110.58</td>
<td>57</td>
<td>30.56</td>
<td>26706.00</td>
<td>26706.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>6.45</td>
<td>109.90</td>
<td>100.38</td>
<td>2348.35</td>
<td>2155.24</td>
<td>47</td>
<td>42.99</td>
<td>2440.00</td>
<td>2440.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>10.48</td>
<td>113.90</td>
<td>100.44</td>
<td>2348.49</td>
<td>2151.79</td>
<td>53</td>
<td>41.80</td>
<td>3852.00</td>
<td>3852.00</td>
</tr>
<tr>
<td>M4</td>
<td>42.12</td>
<td>149.50</td>
<td>128.97</td>
<td>2437.94</td>
<td>2264.82</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>13.95</td>
<td>114.90</td>
<td>100.56</td>
<td>2346.36</td>
<td>2148.19</td>
<td>64</td>
<td>49.19</td>
<td>9816.00</td>
<td>9816.00</td>
</tr>
<tr>
<td rowspan="8">formula_recognition-nopp-nolayout</td>
<td>Intel 8350C</td>
<td>56.83</td>
<td>1067.80</td>
<td>1023.31</td>
<td>2173.72</td>
<td>2059.64</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>16.58</td>
<td>104.90</td>
<td>100.18</td>
<td>2017.30</td>
<td>1968.04</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>9.08</td>
<td>102.90</td>
<td>100.19</td>
<td>1747.94</td>
<td>1690.36</td>
<td>42</td>
<td>39.01</td>
<td>1460.00</td>
<td>1460.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>13.97</td>
<td>116.80</td>
<td>100.45</td>
<td>1864.97</td>
<td>1796.86</td>
<td>36</td>
<td>30.08</td>
<td>5382.00</td>
<td>5382.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>6.45</td>
<td>107.00</td>
<td>100.19</td>
<td>2163.71</td>
<td>2125.32</td>
<td>45</td>
<td>41.34</td>
<td>4488.00</td>
<td>4488.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>9.15</td>
<td>103.00</td>
<td>100.24</td>
<td>1998.28</td>
<td>1941.02</td>
<td>53</td>
<td>41.47</td>
<td>3760.00</td>
<td>3760.00</td>
</tr>
<tr>
<td>M4</td>
<td>39.77</td>
<td>149.80</td>
<td>129.49</td>
<td>2098.33</td>
<td>1915.70</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>11.57</td>
<td>103.00</td>
<td>100.37</td>
<td>1995.27</td>
<td>1925.57</td>
<td>66</td>
<td>51.18</td>
<td>6534.00</td>
<td>6534.00</td>
</tr>
<tr>
<td rowspan="9">formula_recognition-nopp-lightweight</td>
<td>Intel 6271C</td>
<td>6.54</td>
<td>1023.60</td>
<td>1007.63</td>
<td>2097.84</td>
<td>1974.29</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>4.72</td>
<td>1028.60</td>
<td>926.98</td>
<td>2089.03</td>
<td>1951.58</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.76</td>
<td>116.90</td>
<td>107.99</td>
<td>1909.78</td>
<td>1844.97</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.49</td>
<td>109.90</td>
<td>105.51</td>
<td>1789.37</td>
<td>1739.76</td>
<td>45</td>
<td>25.68</td>
<td>1302.00</td>
<td>1302.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.67</td>
<td>116.80</td>
<td>109.35</td>
<td>1751.75</td>
<td>1695.62</td>
<td>43</td>
<td>26.31</td>
<td>1058.00</td>
<td>1058.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.42</td>
<td>112.90</td>
<td>105.16</td>
<td>2249.38</td>
<td>2190.95</td>
<td>41</td>
<td>28.32</td>
<td>1364.00</td>
<td>1364.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.52</td>
<td>108.90</td>
<td>105.69</td>
<td>1903.01</td>
<td>1847.40</td>
<td>46</td>
<td>29.18</td>
<td>1040.00</td>
<td>1040.00</td>
</tr>
<tr>
<td>M4</td>
<td>6.15</td>
<td>116.00</td>
<td>105.42</td>
<td>2048.00</td>
<td>1927.59</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.72</td>
<td>110.50</td>
<td>105.53</td>
<td>1912.62</td>
<td>1841.16</td>
<td>61</td>
<td>42.81</td>
<td>874.00</td>
<td>874.00</td>
</tr>
<tr>
<td rowspan="9">formula_recognition-nopp-lightweightlayout</td>
<td>Intel 6271C</td>
<td>7.56</td>
<td>1052.60</td>
<td>1015.57</td>
<td>2214.38</td>
<td>2009.53</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>5.46</td>
<td>1056.60</td>
<td>940.86</td>
<td>2217.84</td>
<td>1946.67</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>1.26</td>
<td>115.80</td>
<td>103.72</td>
<td>1937.91</td>
<td>1888.03</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.70</td>
<td>107.90</td>
<td>102.54</td>
<td>1740.03</td>
<td>1693.03</td>
<td>44</td>
<td>36.96</td>
<td>2018.00</td>
<td>2018.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>1.01</td>
<td>114.00</td>
<td>104.15</td>
<td>1760.52</td>
<td>1723.06</td>
<td>47</td>
<td>33.37</td>
<td>1542.00</td>
<td>1542.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.56</td>
<td>107.80</td>
<td>102.41</td>
<td>2073.04</td>
<td>1998.66</td>
<td>51</td>
<td>43.06</td>
<td>2058.00</td>
<td>2058.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.73</td>
<td>108.90</td>
<td>102.77</td>
<td>1952.25</td>
<td>1911.61</td>
<td>59</td>
<td>41.03</td>
<td>1524.00</td>
<td>1524.00</td>
</tr>
<tr>
<td>M4</td>
<td>2.87</td>
<td>124.10</td>
<td>113.46</td>
<td>2467.97</td>
<td>2009.42</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>1.04</td>
<td>108.00</td>
<td>102.42</td>
<td>1928.40</td>
<td>1886.86</td>
<td>71</td>
<td>53.18</td>
<td>1298.00</td>
<td>1298.00</td>
</tr>
<tr>
<td rowspan="9">formula_recognition-nopp-nolayout-lightweight</td>
<td>Intel 6271C</td>
<td>2.80</td>
<td>1047.60</td>
<td>1031.25</td>
<td>1534.17</td>
<td>1473.93</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>2.32</td>
<td>1052.60</td>
<td>1034.39</td>
<td>1523.69</td>
<td>1465.91</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>1.00</td>
<td>104.90</td>
<td>101.83</td>
<td>1859.66</td>
<td>1824.77</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.59</td>
<td>103.90</td>
<td>102.00</td>
<td>1633.84</td>
<td>1594.35</td>
<td>44</td>
<td>38.61</td>
<td>864.00</td>
<td>864.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.79</td>
<td>108.90</td>
<td>103.25</td>
<td>1658.45</td>
<td>1622.73</td>
<td>38</td>
<td>33.39</td>
<td>718.00</td>
<td>717.35</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.47</td>
<td>108.00</td>
<td>102.07</td>
<td>2023.42</td>
<td>1991.38</td>
<td>51</td>
<td>44.90</td>
<td>918.00</td>
<td>918.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.62</td>
<td>103.00</td>
<td>101.00</td>
<td>1918.49</td>
<td>1888.83</td>
<td>43</td>
<td>38.38</td>
<td>622.00</td>
<td>622.00</td>
</tr>
<tr>
<td>M4</td>
<td>1.57</td>
<td>119.50</td>
<td>117.72</td>
<td>1489.78</td>
<td>1430.09</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.83</td>
<td>105.00</td>
<td>101.98</td>
<td>1934.31</td>
<td>1848.85</td>
<td>46</td>
<td>39.28</td>
<td>472.00</td>
<td>472.00</td>
</tr>
</table>


<table border="1">
<tr><th>Pipeline configuration</th><th>description</th></tr>
<tr>
<td>formula_recognition-default</td>
<td>Default configuration</td>
</tr>
<tr>
<td>formula_recognition-nopp</td>
<td>Based on the default configuration, document image preprocessing is disabled</td>
</tr>
<tr>
<td>formula_recognition-nolayout</td>
<td>Based on the default configuration, the layout region detection model is disabled</td>
</tr>
<tr>
<td>formula_recognition-nopp-nolayout</td>
<td>Based on the default configuration, document image preprocessing and the layout region detection model are disabled</td>
</tr>
<tr>
<td>formula_recognition-nopp-lightweight</td>
<td>Based on the default configuration, document image preprocessing is disabled, and the lightweight formula model PP-FormulaNet_plus-S is used</td>
</tr>
<tr>
<td>formula_recognition-nopp-lightweightlayout</td>
<td>Based on the default configuration, document image preprocessing is disabled, and the lightweight layout detection model PP-DocLayout-S is used</td>
</tr>
<tr>
<td>formula_recognition-nopp-nolayout-lightweight</td>
<td>Based on the default configuration, document image preprocessing and the layout region detection model are disabled, and the lightweight formula model PP-FormulaNet_plus-S is used</td>
</tr>
</table>
</details>


* Test environment:
    * PaddlePaddle 3.1.0、CUDA 11.8、cuDNN 8.9
    * PaddleX @ develop (f1eb28e23cfa54ce3e9234d2e61fcb87c93cf407)
    * Docker image: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda11.8-cudnn8.9
* Test data:
    * Test data containing 4 images and 1 PDF, including Chinese document formulas, English document formulas, standalone Chinese formulas, and tables with embedded formulas.
* Test strategy:
    * Warm up with 3 samples, then repeat the full dataset 20 times for performance testing.
* Note:
    * Since we did not collect device memory data for NPU and XPU, the corresponding entries in the table are marked as N/A.

## 2. Quick Start
All model pipelines provided by PaddleX can be quickly experienced. You can experience the effect of the formula recognition pipeline on the community platform, or you can use the command line or Python locally to experience the effect of the formula recognition pipeline.

### 2.1 Online Experience
You can [experience the formula recognition pipeline online](https://aistudio.baidu.com/community/app/387976/webUI?source=appCenter) by recognizing the demo images provided by the official platform, for example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/formula_aistudio.png"/>

If you are satisfied with the performance of the pipeline, you can directly integrate and deploy it. You can choose to download the deployment package from the cloud, or refer to the methods in [Section 2.2 Local Experience](#22-local-experience) for local deployment. If you are not satisfied with the effect, you can <b>fine-tune the models in the pipeline using your private data</b>. If you have local hardware resources for training, you can start training directly on your local machine; if not, the Star River Zero-Code platform provides a one-click training service. You don't need to write any code—just upload your data and start the training task with one click.

### 2.2 Local Experience
> ❗ Before using the formula recognition pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Installation Guide](../../../installation/installation.en.md). If you wish to selectively install dependencies, please refer to the relevant instructions in the installation guide. The dependency group corresponding to this pipeline is `ocr`.

#### 2.2.1 Command Line Experience
You can quickly experience the effect of the formula recognition pipeline with one command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/pipelines/general_formula_recognition_001.png), and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline formula_recognition \
        --input general_formula_recognition_001.png \
        --use_layout_detection True \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --layout_threshold 0.5 \
        --layout_nms True \
        --layout_unclip_ratio  1.0 \
        --layout_merge_bboxes_mode "'large'"\
        --save_path ./output \
        --device gpu:0
```

<b>Note: </b>The official models would be download from HuggingFace by first. PaddleX also support to specify the preferred source by setting the environment variable `PADDLE_PDX_MODEL_SOURCE`. The supported values are `huggingface`, `aistudio`, `bos`, and `modelscope`. For example, to prioritize using `bos`, set: `PADDLE_PDX_MODEL_SOURCE="bos"`.

The relevant parameter descriptions can be referenced from [2.2 Integration via Python Script](#22-integration-via-python-script). Supports specifying multiple devices simultaneously for parallel inference. For details, please refer to the documentation on pipeline parallel inference.

After running, the results will be printed to the terminal, as shown below:

```bash
{'res': {'input_path': 'general_formula_recognition_001.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9849682450294495, 'coordinate': [112.44934, 1074.9535, 663.01184, 1527.8729]}, {'cls_id': 2, 'label': 'text', 'score': 0.9809818267822266, 'coordinate': [112.761505, 160.65752, 662.4872, 414.6153]}, {'cls_id': 2, 'label': 'text', 'score': 0.9776389002799988, 'coordinate': [697.2594, 597.79034, 1246.6486, 746.98676]}, {'cls_id': 2, 'label': 'text', 'score': 0.9713516235351562, 'coordinate': [696.8539, 751.34827, 1246.7255, 875.0985]}, {'cls_id': 2, 'label': 'text', 'score': 0.9704778790473938, 'coordinate': [696.94653, 118.34517, 1247.156, 215.88304]}, {'cls_id': 2, 'label': 'text', 'score': 0.9666076898574829, 'coordinate': [113.78964, 795.92883, 662.4155, 893.1124]}, {'cls_id': 2, 'label': 'text', 'score': 0.9665274620056152, 'coordinate': [697.57666, 310.11346, 1247.8262, 408.27625]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9658663272857666, 'coordinate': [725.58777, 454.7463, 1183.6041, 577.70825]}, {'cls_id': 2, 'label': 'text', 'score': 0.9638860821723938, 'coordinate': [697.0996, 1481.4896, 1246.7573, 1606.951]}, {'cls_id': 2, 'label': 'text', 'score': 0.9625014066696167, 'coordinate': [698.0481, 945.31726, 1246.6857, 1017.86316]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9616876244544983, 'coordinate': [723.08606, 1304.9329, 1221.5212, 1433.7351]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9603673219680786, 'coordinate': [171.7144, 915.73035, 606.7387, 1026.4806]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9600085020065308, 'coordinate': [803.9297, 1040.9861, 1142.0847, 1099.2834]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9595950245857239, 'coordinate': [767.97546, 236.60234, 1180.0568, 291.77338]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9594774842262268, 'coordinate': [753.474, 1189.0759, 1154.2972, 1242.22]}, {'cls_id': 2, 'label': 'text', 'score': 0.9578917622566223, 'coordinate': [112.63562, 1532.9048, 663.50165, 1606.6626]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9559630155563354, 'coordinate': [179.8501, 566.4032, 599.8784, 620.08655]}, {'cls_id': 2, 'label': 'text', 'score': 0.953528642654419, 'coordinate': [113.30719, 490.1695, 662.4507, 546.1473]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9524251222610474, 'coordinate': [133.87987, 713.4751, 617.22253, 772.76074]}, {'cls_id': 2, 'label': 'text', 'score': 0.9439663290977478, 'coordinate': [114.23682, 642.90356, 662.42474, 692.64404]}, {'cls_id': 2, 'label': 'text', 'score': 0.9418545961380005, 'coordinate': [696.9155, 1120.0913, 1246.4368, 1168.2529]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9261635541915894, 'coordinate': [208.07648, 440.37366, 569.06274, 464.80005]}, {'cls_id': 7, 'label': 'formula', 'score': 0.920579731464386, 'coordinate': [843.95044, 899.7913, 1102.5901, 922.502]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.9113483428955078, 'coordinate': [1208.2693, 1061.5608, 1245.5154, 1085.906]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.910507321357727, 'coordinate': [1208.536, 1204.4053, 1245.2667, 1228.9385]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.909659206867218, 'coordinate': [1207.9521, 899.5998, 1245.6323, 923.64777]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.9072737097740173, 'coordinate': [1208.1934, 505.28503, 1245.4221, 529.67957]}, {'cls_id': 2, 'label': 'text', 'score': 0.9037376046180725, 'coordinate': [698.28625, 1262.6298, 1059.2609, 1285.3319]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9015905857086182, 'coordinate': [183.00662, 161.18907, 512.1726, 185.87654]}, {'cls_id': 2, 'label': 'text', 'score': 0.9007592797279358, 'coordinate': [721.2688, 412.6385, 1216.6045, 436.58807]}, {'cls_id': 2, 'label': 'text', 'score': 0.896611750125885, 'coordinate': [113.65086, 1048.154, 451.65906, 1070.5525]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8945134878158569, 'coordinate': [1208.4277, 1436.2947, 1245.581, 1459.6746]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.892361044883728, 'coordinate': [1220.0835, 251.3541, 1245.6968, 275.61313]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8889164924621582, 'coordinate': [116.8533, 265.05603, 307.09003, 289.93555]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8836347460746765, 'coordinate': [635.03656, 441.60486, 660.72943, 465.45752]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8798341751098633, 'coordinate': [635.3544, 928.30615, 660.801, 952.7964]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8789814710617065, 'coordinate': [635.7049, 582.2196, 660.6303, 606.67004]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.878730833530426, 'coordinate': [635.7882, 987.47, 660.4442, 1011.64307]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8710389137268066, 'coordinate': [635.75195, 731.1045, 660.3816, 754.42114]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8667231798171997, 'coordinate': [540.2363, 489.12628, 661.61475, 517.5715]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8634350299835205, 'coordinate': [117.003365, 1302.1862, 277.30377, 1326.8046]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8547449707984924, 'coordinate': [1072.7319, 1558.7657, 1242.1775, 1581.1473]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8485350608825684, 'coordinate': [373.27258, 290.88776, 518.7126, 316.89764]}, {'cls_id': 3, 'label': 'number', 'score': 0.8004439473152161, 'coordinate': [1236.0953, 62.784798, 1247.5856, 79.25625]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7683551907539368, 'coordinate': [258.48035, 189.60883, 325.27173, 213.70544]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7660759091377258, 'coordinate': [909.1803, 1584.3768, 999.6908, 1606.9323]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7568713426589966, 'coordinate': [701.779, 1532.1399, 891.00665, 1556.2966]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7310451865196228, 'coordinate': [701.32715, 1583.2532, 857.19336, 1605.7214]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7173376083374023, 'coordinate': [116.34981, 518.7822, 268.02765, 544.4822]}, {'cls_id': 7, 'label': 'formula', 'score': 0.6384682059288025, 'coordinate': [768.198, 601.2766, 793.38574, 622.52]}, {'cls_id': 7, 'label': 'formula', 'score': 0.6290246248245239, 'coordinate': [583.8933, 1253.2761, 609.42676, 1276.0557]}, {'cls_id': 7, 'label': 'formula', 'score': 0.6267108917236328, 'coordinate': [1095.0366, 1533.7931, 1163.5054, 1555.3956]}, {'cls_id': 7, 'label': 'formula', 'score': 0.6116171479225159, 'coordinate': [1214.5708, 337.85333, 1244.5303, 359.7912]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5208110809326172, 'coordinate': [701.8009, 336.1151, 729.7782, 357.85132]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5141622424125671, 'coordinate': [189.37997, 368.27167, 253.99014, 393.74164]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5111705660820007, 'coordinate': [778.9128, 365.22208, 801.6574, 385.84518]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5065054893493652, 'coordinate': [267.81232, 341.93176, 336.70538, 366.84973]}]}, 'formula_res_list': [{'rec_formula': '\\begin{aligned}{\\psi_{0}(M)-\\psi(M,z)=}&{{}\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}}\\frac{\\lambda^{2}c^{2}}{t_{\\operatorname{E}}^{2}\\operatorname{l n}(10)}\\times}\\\\ {}&{{}\\left.\\int_{0}^{z}d z^{\\prime}\\frac{d t}{d z^{\\prime}}\\;\\frac{\\partial\\phi}{\\partial L}\\right|_{L=\\lambda M c^{2}/t_{\\operatorname{E}}},}\\\\ \\end{aligned}', 'formula_region_id': 1, 'dt_polys': ([725.58777, 454.7463, 1183.6041, 577.70825],)}, {'rec_formula': '\\begin{aligned}{p(\\operatorname{l o g}_{10}}&{{}M|\\operatorname{l o g}_{10}\\sigma)=\\frac{1}{\\sqrt{2\\pi}\\epsilon_{0}}}\\\\ {}&{{}\\times\\operatorname{e x p}\\left[-\\frac{1}{2}\\left(\\frac{\\operatorname{l o g}_{10}M-a_{\\bullet}-b_{\\bullet}\\operatorname{l o g}_{10}\\sigma}{\\epsilon_{0}}\\right)^{2}\\right].}\\\\ \\end{aligned}', 'formula_region_id': 2, 'dt_polys': ([723.08606, 1304.9329, 1221.5212, 1433.7351],)}, {'rec_formula': '\\begin{aligned}{\\rho_{\\operatorname{B H}}}&{{}=\\int d M\\psi(M)M}\\\\ {}&{{}=\\frac{1-\\epsilon_{r}}{\\epsilon_{r}c^{2}}\\int_{0}^{\\infty}d z\\frac{d t}{d z}\\int d\\operatorname{l o g}_{10}L\\phi(L,z)L,}\\\\ \\end{aligned}', 'formula_region_id': 3, 'dt_polys': ([171.7144, 915.73035, 606.7387, 1026.4806],)}, {'rec_formula': '\\frac{d n}{d\\sigma}d\\sigma=\\psi_{*}\\left(\\frac{\\sigma}{\\sigma_{*}}\\right)^{\\alpha}\\frac{e^{-(\\sigma/\\sigma_{*})^{\\beta}}}{\\Gamma(\\alpha/\\beta)}\\beta\\frac{d\\sigma}{\\sigma}.', 'formula_region_id': 4, 'dt_polys': ([803.9297, 1040.9861, 1142.0847, 1099.2834],)}, {'rec_formula': '\\phi(L)\\equiv\\frac{d n}{d\\log_{10}L}=\\frac{\\phi_{*}}{(L/L_{*})^{\\gamma_{1}}+(L/L_{*})^{\\gamma_{2}}}.', 'formula_region_id': 5, 'dt_polys': ([767.97546, 236.60234, 1180.0568, 291.77338],)}, {'rec_formula': '\\psi_{0}(M)=\\int d\\sigma\\frac{p(\\log_{10}M|\\log_{10}\\sigma)}{M\\log(10)}\\frac{d n}{d\\sigma}(\\sigma),', 'formula_region_id': 6, 'dt_polys': ([753.474, 1189.0759, 1154.2972, 1242.22],)}, {'rec_formula': '\\langle\\dot{M}(M,t)\\rangle\\psi(M,t)=\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}c^{2}\\operatorname{l n}(10)}\\phi(L,t)\\frac{d L}{d M}.', 'formula_region_id': 7, 'dt_polys': ([179.8501, 566.4032, 599.8784, 620.08655],)}, {'rec_formula': '\\frac{\\partial\\psi}{\\partial t}(M,t)+\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}}\\frac{\\lambda^{2}c^{2}}{t_{\\operatorname{E}}^{2}\\operatorname{l n}(10)}\\left.\\frac{\\partial\\phi}{\\partial L}\\right|_{L=\\lambda M c^{2}/t_{\\operatorname{E}}}=0,', 'formula_region_id': 8, 'dt_polys': ([133.87987, 713.4751, 617.22253, 772.76074],)}, {'rec_formula': '\\phi(L,t)d\\log_{10}L=\\delta(M,t)\\psi(M,t)d M.', 'formula_region_id': 9, 'dt_polys': ([208.07648, 440.37366, 569.06274, 464.80005],)}, {'rec_formula': '\\log_{10}M=a_{\\bullet}+b_{\\bullet}\\log_{10}X.', 'formula_region_id': 10, 'dt_polys': ([843.95044, 899.7913, 1102.5901, 922.502],)}, {'rec_formula': 't_{E}=\\sigma_{T}c/4\\pi G m_{p}=4.5\\times10^{8}\\mathrm{y r}', 'formula_region_id': 11, 'dt_polys': ([183.00662, 161.18907, 512.1726, 185.87654],)}, {'rec_formula': '\\dot{M}\\:=\\:(1\\:-\\:\\epsilon_{r})\\dot{M}_{\\mathrm{a c c}}', 'formula_region_id': 12, 'dt_polys': ([116.8533, 265.05603, 307.09003, 289.93555],)}, {'rec_formula': '\\langle\\dot{M}(M,t)\\rangle=', 'formula_region_id': 13, 'dt_polys': ([540.2363, 489.12628, 661.61475, 517.5715],)}, {'rec_formula': 'M_{*}=L_{*}t_{E}/\\lambda c^{2}', 'formula_region_id': 14, 'dt_polys': ([117.003365, 1302.1862, 277.30377, 1326.8046],)}, {'rec_formula': 'a_{\\bullet}=8.32\\pm0.05', 'formula_region_id': 15, 'dt_polys': ([1072.7319, 1558.7657, 1242.1775, 1581.1473],)}, {'rec_formula': '\\phi(L,t)d\\operatorname{l o g}_{10}L', 'formula_region_id': 16, 'dt_polys': ([373.27258, 290.88776, 518.7126, 316.89764],)}, {'rec_formula': '\\epsilon_{r}\\dot{M}_{\\mathrm{a c c}}', 'formula_region_id': 17, 'dt_polys': ([258.48035, 189.60883, 325.27173, 213.70544],)}, {'rec_formula': '\\epsilon_{0}=0.38', 'formula_region_id': 18, 'dt_polys': ([909.1803, 1584.3768, 999.6908, 1606.9323],)}, {'rec_formula': 'X=\\sigma/200\\mathrm{k m}\\mathrm{s}^{-1}', 'formula_region_id': 19, 'dt_polys': ([701.779, 1532.1399, 891.00665, 1556.2966],)}, {'rec_formula': 'b_{\\bullet}=5.64\\pm\\widetilde{0.32}', 'formula_region_id': 20, 'dt_polys': ([701.32715, 1583.2532, 857.19336, 1605.7214],)}, {'rec_formula': '\\delta(M,t)\\dot{M}(M,t)', 'formula_region_id': 21, 'dt_polys': ([116.34981, 518.7822, 268.02765, 544.4822],)}, {'rec_formula': '\\psi_{0}', 'formula_region_id': 22, 'dt_polys': ([768.198, 601.2766, 793.38574, 622.52],)}, {'rec_formula': 'L_{*}', 'formula_region_id': 23, 'dt_polys': ([583.8933, 1253.2761, 609.42676, 1276.0557],)}, {'rec_formula': 'M-\\sigma', 'formula_region_id': 24, 'dt_polys': ([1095.0366, 1533.7931, 1163.5054, 1555.3956],)}, {'rec_formula': 'L_{*}', 'formula_region_id': 25, 'dt_polys': ([1214.5708, 337.85333, 1244.5303, 359.7912],)}, {'rec_formula': '\\mathrm{A^{\\prime\\prime}}', 'formula_region_id': 26, 'dt_polys': ([701.8009, 336.1151, 729.7782, 357.85132],)}, {'rec_formula': '\\phi(L,t)', 'formula_region_id': 27, 'dt_polys': ([189.37997, 368.27167, 253.99014, 393.74164],)}, {'rec_formula': '\\gamma_{2}', 'formula_region_id': 28, 'dt_polys': ([778.9128, 365.22208, 801.6574, 385.84518],)}, {'rec_formula': '\\delta(\\bar{M,t})', 'formula_region_id': 29, 'dt_polys': ([267.81232, 341.93176, 336.70538, 366.84973],)}]}}
```

The explanation of the running result parameters can refer to the result interpretation in [2.2 Integration via Python Script](#22-integration-via-python-script).

The visualization results are saved under `save_path`, where the visualization result of formula recognition is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/04_paddleocr3.png" style="width: 70%"/>

<b>If you need to visualize the formula recognition pipeline, you need to run the following command to install the LaTeX rendering environment. Currently, visualization of the formula recognition pipeline only supports the Ubuntu environment, and other environments are not supported. For complex formulas, the LaTeX result may contain some advanced representations that may not be successfully displayed in environments such as Markdown:</b>

```bash
sudo apt-get update
sudo apt-get install texlive texlive-latex-base texlive-xetex latex-cjk-all texlive-latex-extra -y
```

<b>Note</b>: Due to the need to render each formula image during the formula recognition visualization process, the process takes a long time. Please be patient.

#### 2.2.2 Python Script Integration
A few lines of code can quickly complete the pipeline inference. Taking the formula recognition pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="formula_recognition")

output = pipeline.predict(
    input="./general_formula_recognition_001.png",
    use_layout_detection=True ,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    layout_threshold=0.5,
    layout_nms=True,
    layout_unclip_ratio=1.0,
    layout_merge_bboxes_mode="large"
)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")
```

In the above Python script, the following steps are executed:

(1) Instantiate the formula recognition pipeline object through `create_pipeline()`, with specific parameters as follows:

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
<td>Pipeline name or path to pipeline config file, if it's set as a pipeline name, it must be a pipeline supported by PaddleX.</td>
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
<td>Pipeline inference device. Supports specifying the specific GPU card number, such as "gpu:0", other hardware specific card numbers, such as "npu:0", CPU such as "cpu". Supports specifying multiple devices simultaneously for parallel inference. For details, please refer to <a href="../../instructions/parallel_inference.en.md#specifying-multiple-inference-devices">Pipeline Parallel Inference</a>.</td>
<td><code>str</code></td>
<td><code>None</code></td>
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

(2) Call the `predict()` method of the formula recognition pipeline object for inference prediction. This method will return a `generator`. Below are the parameters of the `predict()` method and their descriptions:
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
<tbody>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types, required</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
<li><b>str</b>: Local path of image or PDF file, e.g., <code>/root/data/img.jpg</code>; <b>URL link</b>, e.g., network URL of image or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/pipelines/general_formula_recognition_001.png">Example</a>; <b>Local directory</b>, the directory should contain images to be predicted, e.g., local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories; PDF files must be specified with a specific file path)</li>
<li><b>List</b>: Elements of the list must be of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>Whether to use the document layout detection module</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized as <code>True</code>.</li>
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
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized as <code>True</code>.</li>
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
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized as <code>True</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Threshold for filtering low-confidence prediction results; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>float/dict/None</code></td>
<td>
<ul>
<li><b>float</b>, e.g., 0.2, indicating filtering out all bounding boxes with confidence scores below 0.2</li>
<li><b>Dictionary</b>, with <b>int</b> keys representing <code>cls_id</code> and <b>float</b> values as thresholds. For example, <code>{0: 0.45, 2: 0.48, 7: 0.4}</code> indicates applying a threshold of 0.45 for class ID 0, 0.48 for class ID 2, and 0.4 for class ID 7</li>
<li><b>None</b>: If not specified, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS post-processing to filter overlapping bounding boxes; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>bool/None</code></td>
<td>
<ul>
<li><b>bool</b>: <code>True</code> or <code>False</code>, indicating whether to use NMS for post-processing to filter overlapping bounding boxes</li>
<li><b>None</b>: If not specified, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Scaling factor for the side length of bounding boxes; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>float/list/None</code></td>
<td>
<ul>
<li><b>float</b>: A positive float number, e.g., 1.1, indicating that the center of the bounding box remains unchanged while the width and height are both scaled up by a factor of 1.1</li>
<li><b>List</b>: e.g., [1.2, 1.5], indicating that the center of the bounding box remains unchanged while the width is scaled up by a factor of 1.2 and the height by a factor of 1.5</li>
<li><b>None</b>: If not specified, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<td><code>None</code></td>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Merging mode for the bounding boxes output by the model; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>string/None</code></td>
<td>
<ul>
<li><b>large</b>: When set to "large", only the largest outer bounding box will be retained for overlapping bounding boxes, and the inner overlapping boxes will be removed.</li>
<li><b>small</b>: When set to "small", only the smallest inner bounding boxes will be retained for overlapping bounding boxes, and the outer overlapping boxes will be removed.</li>
<li><b>union</b>: No filtering of bounding boxes will be performed, and both inner and outer boxes will be retained.</li>
<li><b>None</b>: If not specified, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tr></tbody>
</table>

(3) Process the prediction results. The prediction result of each sample is of `dict` type, and supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td rowspan="3">Print results to terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save results as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. If it is a directory, the saved file will be named the same as the input file type</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save results as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file, supports directory or file path</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal. The content printed to the terminal is explained as follows:

    - `input_path`: `(str)` The input path of the image to be predicted.

    - `page_index`: `(Union[int, None])` If the input is a PDF file, this indicates the current page number of the PDF. Otherwise, it is `None`

    - `model_settings`: `(Dict[str, bool])` The model parameters required for the pipeline configuration.

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing sub-pipeline.
        - `use_layout_detection`: `(bool)` Controls whether to enable the layout area detection module.

    - `doc_preprocessor_res`: `(Dict[str, Union[str, Dict[str, bool], int]])` The output result of the document preprocessing sub-pipeline. It exists only when `use_doc_preprocessor=True`.
        - `input_path`: `(Union[str, None])` The image path accepted by the image preprocessing sub-pipeline. When the input is a `numpy.ndarray`, it is saved as `None`.
        - `model_settings`: `(Dict)` The model configuration parameters of the preprocessing sub-pipeline.
            - `use_doc_orientation_classify`: `(bool)` Controls whether to enable document orientation classification.
            - `use_doc_unwarping`: `(bool)` Controls whether to enable document distortion correction.
        - `angle`: `(int)` The prediction result of document orientation classification. When enabled, it takes values from [0,1,2,3], corresponding to [0°,90°,180°,270°]; when disabled, it is -1.
    - `layout_det_res`: `(Dict[str, List[Dict]])` The output result of the layout area detection module. It exists only when `use_layout_detection=True`.
        - `input_path`: `(Union[str, None])` The image path accepted by the layout area detection module. When the input is a `numpy.ndarray`, it is saved as `None`.
        - `boxes`: `(List[Dict[int, str, float, List[float]]])` A list of layout area detection prediction results.
            - `cls_id`: `(int)` The class ID predicted by layout area detection.
            - `label`: `(str)` The class label predicted by layout area detection.
            - `score`: `(float)` The confidence score of the predicted class.
            - `coordinate`: `(List[float])` The bounding box coordinates predicted by layout area detection, in the format [x_min, y_min, x_max, y_max], where (x_min, y_min) is the top-left corner and (x_max, y_max) is the bottom-right corner.
    - `formula_res_list`: `(List[Dict[str, int, List[float]]])` A list of formula recognition prediction results.
        - `rec_formula`: `(str)` The LaTeX source code predicted by formula recognition.
        - `formula_region_id`: `(int)` The ID number predicted by formula recognition.
        - `dt_polys`: `(List[float])` The bounding box coordinates predicted by formula recognition, in the format [x_min, y_min, x_max, y_max], where (x_min, y_min) is the top-left corner and (x_max, y_max) is the bottom-right corner.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, `numpy.array` types will be converted to list format.
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_formula_res_img.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (The pipeline usually contains many result images, so it is not recommended to specify a specific file path directly, otherwise multiple images will be overwritten and only the last one will be retained.)

* In addition, you can also obtain the visualization image with results and the prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
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

- The prediction results obtained through the `json` attribute are of the dict type, with content consistent with what is saved using the `save_to_json()` method.
- The prediction results returned by the `img` attribute are of the dictionary type. The keys are `preprocessed_img`, `layout_det_res`, and `formula_res_img`, corresponding to three `Image.Image` objects: the first one displays the visualization image of image preprocessing, the second one displays the visualization image of layout area detection, and the third one displays the visualization image of formula recognition. If the image preprocessing sub-module is not used, the dictionary does not contain `preprocessed_img`; if the layout area detection sub-module is not used, the dictionary does not contain `layout_det_res`.

In addition, you can obtain the configuration file of the formula recognition pipeline and load the configuration file for prediction. You can execute the following command to save the results in `my_path`:

```
paddlex --get_pipeline_config formula_recognition --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the formula recognition pipeline by simply modifying the value of the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file. An example is shown below:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/formula_recognition.yaml")

output = pipeline.predict(
    input="./general_formula_recognition_001.png",
    use_layout_detection=True ,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    layout_threshold=0.5,
    layout_nms=True,
    layout_unclip_ratio=1.0,
    layout_merge_bboxes_mode="large"
)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")
```

<b>Note:</b> The parameters in the configuration file are initialization parameters for the pipeline. If you want to change the initialization parameters for the formula recognition pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in a configuration file, simply specify the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment
If the formula recognition pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the formula recognition pipeline into your Python project, you can refer to the example code in [ 2.2 Integration via Python Script](#22-integration-via-python-script).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have strict performance requirements for deployment strategies, especially in terms of response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin, which aims to deeply optimize the performance of model inference and pre/post-processing, significantly speeding up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service-Based Deployment</b>: Service-based deployment is a common form of deployment in actual production environments. By encapsulating inference capabilities into services, clients can access these services via network requests to obtain inference results. PaddleX supports multiple pipeline service-based deployment solutions. For detailed pipeline service-based deployment procedures, please refer to the [PaddleX Service-Based Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references for basic service-based deployment and multi-language service invocation examples:

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
<p>Obtain the formula recognition results from images.</p>
<p><code>POST /formula-recognition</code></p>
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
<td><code>formulaRecResults</code></td>
<td><code>object</code></td>
<td>The formula recognition results. The array length is 1 (for image input) or the actual number of document pages processed (for PDF input). For PDF input, each element in the array represents the result of each page actually processed in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>formulaRecResults</code> is an <code>object</code> with the following attributes:</p>
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
<td>A simplified version of the <code>res</code> field in the JSON representation of the result generated by the pipeline object's <code>predict</code> method, excluding the <code>input_path</code> and the <code>page_index</code> fields.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>See the description of the <code>img</code> attribute of the result of the pipeline prediction. The images are in JPEG format and are Base64-encoded.</td>
</tr>
<tr>
<td><code>inputImage</code> | <code>null</code></td>
<td><code>string</code></td>
<td>The input image. The image is in JPEG format and is Base64-encoded.</td>
</tr>
</tbody>
</table>
</details>
<details><summary>Multi-language Service Invocation Example</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/formula-recognition"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["formulaRecResults"]):
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

    auto response = client.Post("/formula-recognition", jsonObj.dump(), "application/json");

    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        if (!result.is_object() || !result["formulaRecResults"].is_array()) {
            std::cerr << "Unexpected response format." << std::endl;
            return 1;
        }

        for (size_t i = 0; i < result["formulaRecResults"].size(); ++i) {
            auto res = result["formulaRecResults"][i];

            if (res.contains("prunedResult")) {
                std::cout << "Recognized formula: " << res["prunedResult"].dump() << std::endl;
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
        String API_URL = "http://localhost:8080/formula-recognition";
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

                JsonNode formulaRecResults = result.get("formulaRecResults");
                for (int i = 0; i < formulaRecResults.size(); i++) {
                    JsonNode item = formulaRecResults.get(i);
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
    "os"
)

func main() {
    API_URL := "http://localhost:8080/formula-recognition"
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

    type FormulaRecResult struct {
        PrunedResult map[string]interface{} `json:"prunedResult"`
        OutputImages map[string]string      `json:"outputImages"`
        InputImage   *string                `json:"inputImage"`
    }

    type Response struct {
        Result struct {
            FormulaRecResults []FormulaRecResult `json:"formulaRecResults"`
            DataInfo          interface{}        `json:"dataInfo"`
        } `json:"result"`
    }

    var respData Response
    if err := json.Unmarshal(body, &respData); err != nil {
        fmt.Printf("Error unmarshaling response: %v\n", err)
        return
    }

    for i, res := range respData.Result.FormulaRecResults {
        fmt.Printf("Result %d - prunedResult: %+v\n", i, res.PrunedResult)

        for imgName, imgData := range res.OutputImages {
            imgBytes, err := base64.StdEncoding.DecodeString(imgData)
            if err != nil {
                fmt.Printf("Error decoding image %s_%d: %v\n", imgName, i, err)
                continue
            }

            filename := fmt.Sprintf("%s_%d.jpg", imgName, i)
            if err := os.WriteFile(filename, imgBytes, 0644); err != nil {
                fmt.Printf("Error saving image %s: %v\n", filename, err)
                continue
            }
            fmt.Printf("Saved image to %s\n", filename)
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
    static readonly string API_URL = "http://localhost:8080/formula-recognition";
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

        JArray formulaRecResults = (JArray)jsonResponse["result"]["formulaRecResults"];
        for (int i = 0; i < formulaRecResults.Count; i++)
        {
            var res = formulaRecResults[i];
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

const API_URL = 'http://localhost:8080/formula-recognition';
const inputFilePath = './demo.jpg';
const fileType = 1;

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

const payload = {
  file: encodeImageToBase64(inputFilePath),
  fileType: fileType
};

axios.post(API_URL, payload)
  .then((response) => {
    const resultArray = response.data.result.formulaRecResults;

    resultArray.forEach((res, index) => {
      console.log(`\n[${index}] prunedResult:`);
      console.log(res.prunedResult);

      const outputImages = res.outputImages;
      if (outputImages) {
        Object.entries(outputImages).forEach(([imgName, base64Img]) => {
          const outputPath = `${imgName}_${index}.jpg`;
          fs.writeFileSync(outputPath, Buffer.from(base64Img, 'base64'));
          console.log(`Saved output image: ${outputPath}`);
        });
      } else {
        console.log(`[${index}] outputImages is null`);
      }
    });
  })
  .catch((error) => {
    console.error('API error：', error.message);
  });
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/formula-recognition";
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

$result = json_decode($response, true)["result"]["formulaRecResults"];

foreach ($result as $i => $item) {
    echo "[$i] prunedResult:\n";
    print_r($item["prunedResult"]);

    if (!empty($item["outputImages"])) {
        foreach ($item["outputImages"] as $img_name => $base64_img) {
            $img_path = "{$img_name}_{$i}.jpg";
            file_put_contents($img_path, base64_decode($base64_img));
            echo "Output image saved at $img_path\n";
        }
    } else {
        echo "No outputImages found for item $i\n";
    }
}
?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>On-Device Deployment</b>: Edge deployment is a method that places computing and data processing capabilities directly on user devices, allowing the device to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed deployment procedures, please refer to the [PaddleX On-Device Deployment Guide](../../../pipeline_deploy/on_device_deployment.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model pipeline into subsequent AI applications.


## 4. Custom Development
If the default model weights provided by the formula recognition pipeline do not meet your requirements in terms of accuracy or speed, you can try to <b>fine-tune</b> the existing models using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the formula recognition pipeline in your scenario.

### 4.1 Model Fine-Tuning
Since the formula recognition pipeline consists of several modules, if the pipeline's performance is not satisfactory, the issue may arise from any one of these modules. You can analyze the poorly recognized images to determine which module is problematic and refer to the corresponding fine-tuning tutorial links in the table below for model fine-tuning.


<table>
<thead>
<tr>
<th>Scenario</th>
<th>Fine-Tuning Module</th>
<th>Reference Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Formulas are missing</td>
<td>Layout Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html">Link</a></td>
</tr>
<tr>
<td>Formula content is inaccurate</td>
<td>Formula Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/formula_recognition.html">Link</a></td>
</tr>
<tr>
<td>Whole-image rotation correction is inaccurate</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html">Link</a></td>
</tr>
<tr>
<td>Image distortion correction is inaccurate</td>
<td>Text Image Correction Module</td>
<td>Fine-tuning not supported</td>
</tr>
</tbody>
</table>


### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain the local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file and replace the local path of the fine-tuned model weights into the corresponding position in the pipeline configuration file:

```yaml
...
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayout-L
    model_dir: null # Replace with the fine-tuned layout detection model weights path
...

  FormulaRecognition:
    module_name: formula_recognition
    model_name: PP-FormulaNet-L
    model_dir: null # Replace with the fine-tuned formula recognition model weights path
    batch_size: 5

SubPipelines:
  DocPreprocessor:
    pipeline_name: doc_preprocessor
    use_doc_orientation_classify: True
    use_doc_unwarping: True
    SubModules:
      DocOrientationClassify:
        module_name: doc_text_orientation
        model_name: PP-LCNet_x1_0_doc_ori
        model_dir: null # Replace with the fine-tuned document image orientation classification model weights path
        batch_size: 1
...
```

Then, refer to the command-line or Python script methods in [2. Quick Start](#2-Quick-Start) to load the modified pipeline configuration file.

##  5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. You can seamlessly switch between different hardware devices by simply modifying the `--device` parameter.

For example, if you use Ascend NPU for formula recognition pipeline inference, the CLI command is:

```bash
paddlex --pipeline formula_recognition \
        --input general_formula_recognition_001.png \
        --use_layout_detection True \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --layout_threshold 0.5 \
        --layout_nms True \
        --layout_unclip_ratio  1.0 \
        --layout_merge_bboxes_mode "'large'"\
        --save_path ./output \
        --device npu:0

```
Of course, you can also specify the hardware device when calling `create_pipeline()` or `predict()` in a Python script.

If you want to use the formula recognition pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
