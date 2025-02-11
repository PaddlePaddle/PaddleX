---
comments: true
---

# Formula Recognition Pipeline User Guide

## 1. Introduction to Formula Recognition Pipeline

Formula recognition is a technology that automatically identifies and extracts LaTeX formula content and structure from documents or images. It is widely used in fields such as mathematics, physics, and computer science for document editing and data analysis. By using computer vision and machine learning algorithms, formula recognition can convert complex mathematical formula information into editable LaTeX format, facilitating further processing and analysis of data.

The formula recognition pipeline is designed to solve formula recognition tasks by extracting formula information from images and outputting it in LaTeX source code format. This pipeline integrates the advanced formula recognition model PP-FormulaNet developed by the PaddlePaddle Vision Team and the well-known formula recognition model UniMERNet. It is an end-to-end formula recognition system that supports the recognition of simple printed formulas, complex printed formulas, and handwritten formulas. Additionally, it includes functions for image orientation correction and distortion correction. Based on this pipeline, precise formula content prediction can be achieved, covering various application scenarios in education, research, finance, manufacturing, and other fields. The pipeline also provides flexible deployment options, supporting multiple hardware devices and programming languages. Moreover, it offers the capability for secondary development. You can train and optimize the pipeline on your own dataset, and the trained model can be seamlessly integrated.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/03.png" style="width: 70%"/>
<b>The formula recognition pipeline includes a mandatory formula recognition module,</b> as well as optional layout detection, document image orientation classification, and text image unwarping modules. The document image orientation classification module and the text image unwarping module are integrated into the formula recognition pipeline as a document preprocessing sub-pipeline. Each module contains multiple models, and you can choose the model based on the benchmark test data below.

<b>If you prioritize model accuracy, choose a model with higher precision; if you care more about inference speed, choose a faster model; if you are concerned about model storage size, choose a smaller model.</b>
<p><b>Document Image Orientation Classification Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, with four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees.</td>
</tr>
</tbody>
</table>
<b>Note: The evaluation dataset for the above accuracy metrics is a self-built dataset covering multiple scenarios such as certificates and documents, with 1,000 images. The GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>
<p><b>Text Image Correction Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>CER</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
<td>0.179</td>
<td>30.3 M</td>
<td>High-precision text image correction model</td>
</tr>
</tbody>
</table>
<b>Note: The accuracy metrics of the model are measured from the <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet benchmark</a>.</b>
<p><b>Layout Detection Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Training Model</a></td>
<td>90.4</td>
<td>34.5252</td>
<td>1454.27</td>
<td>123.76 M</td>
<td>A high-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using RT-DETR-L.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>15.9</td>
<td>160.1</td>
<td>22.578</td>
<td>A layout area localization model with balanced precision and efficiency, trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training Model</a></td>
<td>70.9</td>
<td>13.8</td>
<td>46.7</td>
<td>4.834</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-S.</td>
</tr>
</tbody>
</table>
<b>Note: The evaluation dataset for the above precision metrics is a self-built layout area detection dataset by PaddleOCR, containing 500 common document-type images of Chinese and English papers, magazines, contracts, books, exams, and research reports. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

> ❗ The above list includes the <b>3 core models</b> that are key supported by the text recognition module. The module actually supports a total of <b>11 full models</b>, including several predefined models with different categories. The complete model list is as follows:

<details><summary> 👉 Details of Model List</summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
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
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
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
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
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
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
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
<b>Note: The evaluation dataset for the above precision metrics is a self-built layout area detection dataset by PaddleOCR, containing 892 common document images of Chinese and English papers, magazines, and research reports. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>
</details>
<p><b>Formula Recognition Module </b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Avg-BLEU(%)</th>
<th>GPU Inference Time (ms)</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<td>UniMERNet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UniMERNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">Training Model</a></td>
<td>86.13</td>
<td>2266.96</td>
<td>1.4 G</td>
<td>UniMERNet is a formula recognition model developed by Shanghai AI Lab. It uses Donut Swin as the encoder and MBartDecoder as the decoder. The model is trained on a dataset of one million samples, including simple formulas, complex formulas, scanned formulas, and handwritten formulas, significantly improving the recognition accuracy of real-world formulas.</td>
<tr>
<td>PP-FormulaNet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-FormulaNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">Training Model</a></td>
<td>87.12</td>
<td>202.25</td>
<td>167.9 M</td>
<td rowspan="2">PP-FormulaNet is an advanced formula recognition model developed by the Baidu PaddlePaddle Vision Team. The PP-FormulaNet-S version uses PP-HGNetV2-B4 as its backbone network. Through parallel masking and model distillation techniques, it significantly improves inference speed while maintaining high recognition accuracy, making it suitable for applications requiring fast inference. The PP-FormulaNet-L version, on the other hand, uses Vary_VIT_B as its backbone network and is trained on a large-scale formula dataset, showing significant improvements in recognizing complex formulas compared to PP-FormulaNet-S.</td>
</tr>
<td>PP-FormulaNet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-FormulaNet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">Training Model</a></td>
<td>92.13</td>
<td>1976.52</td>
<td>535.2 M</td>
<tr>
<td>LaTeX_OCR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Training Model</a></td>
<td>71.63</td>
<td>-</td>
<td>89.7 M</td>
<td>LaTeX-OCR is a formula recognition algorithm based on an autoregressive large model. It uses Hybrid ViT as the backbone network and a transformer as the decoder, significantly improving the accuracy of formula recognition.</td>
</tr>
</table>
<b>Note: The above accuracy metrics are measured using an internally built formula recognition test set within PaddleX. The BLEU score of LaTeX_OCR_rec on the LaTeX-OCR formula recognition test set is 0.8821. All model GPU inference times are based on machines with Tesla V100 GPUs, with precision type FP32.</b>

## 2. Quick Start
PaddleX supports experiencing the formula recognition pipeline locally using the command line or Python.

Before using the formula recognition pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 2.1 Command Line Experience
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
        --layout_merge_bboxes_mode large \
        --save_path ./output \
        --device gpu:0
```

The relevant parameter descriptions can be referenced from [2.2 Integration via Python Script](#22-integration-via-python-script).

After running, the results will be printed to the terminal, as shown below:

```bash
{'res': {'input_path': 'general_formula_recognition.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False,'use_layout_detection': True}, 'layout_det_res': {'input_path': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9778407216072083, 'coordinate': [271.257, 648.50824, 1040.2291, 774.8482]}, ...]}, 'formula_res_list': [{'rec_formula': '\\small\\begin{aligned}{p(\\mathbf{x})=c(\\mathbf{u})\\prod_{i}p(x_{i}).}\\\\ \\end{aligned}', 'formula_region_id': 1, 'dt_polys': ([553.0718, 802.0996, 758.75635, 853.093],)}, ...]}}
```

The explanation of the running result parameters can refer to the result interpretation in [2.2 Integration via Python Script](#22-integration-via-python-script).

The visualization results are saved under `save_path`, where the visualization result of formula recognition is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/04.png" style="width: 70%"/>
<b>If you need to visualize the formula recognition pipeline, you need to run the following command to install the LaTeX rendering environment. Currently, visualization of the formula recognition pipeline only supports the Ubuntu environment, and other environments are not supported. For complex formulas, the LaTeX result may contain some advanced representations that may not be successfully displayed in environments such as Markdown:</b>

```bash
sudo apt-get update
sudo apt-get install texlive texlive-latex-base texlive-latex-extra -y
```

<b>Note</b>: Due to the need to render each formula image during the formula recognition visualization process, the process takes a long time. Please be patient.

### 2.2 Integration via Python Script
A few lines of code can quickly complete the production line inference. Taking the formula recognition production line as an example:

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
<td>Pipeline inference device. Supports specifying the specific GPU card number, such as "gpu:0", other hardware specific card numbers, such as "npu:0", CPU such as "cpu".</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available when the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the formula recognition production line object for inference prediction. This method will return a `generator`. Below are the parameters of the `predict()` method and their descriptions:
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
<td><code>device</code></td>
<td>Production line inference device</td>
<td><code>str|None</code></td>
<td>
<ul>
<li><b>CPU</b>: e.g., <code>cpu</code> indicates using CPU for inference;</li>
<li><b>GPU</b>: e.g., <code>gpu:0</code> indicates using the 1st GPU for inference;</li>
<li><b>NPU</b>: e.g., <code>npu:0</code> indicates using the 1st NPU for inference;</li>
<li><b>XPU</b>: e.g., <code>xpu:0</code> indicates using the 1st XPU for inference;</li>
<li><b>MLU</b>: e.g., <code>mlu:0</code> indicates using the 1st MLU for inference;</li>
<li><b>DCU</b>: e.g., <code>dcu:0</code> indicates using the 1st DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the production line will be used. During initialization, the local GPU 0 will be prioritized; if unavailable, the CPU will be used.</li>
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
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the production line will be used, initialized as <code>True</code>.</li>
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
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the production line will be used, initialized as <code>True</code>.</li>
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
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the production line will be used, initialized as <code>True</code>.</li>
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

    - `model_settings`: `(Dict[str, bool])` The model parameters required for the production line configuration.

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing sub-production line.
        - `use_layout_detection`: `(bool)` Controls whether to enable the layout area detection module.

    - `doc_preprocessor_res`: `(Dict[str, Union[str, Dict[str, bool], int]])` The output result of the document preprocessing sub-production line. It exists only when `use_doc_preprocessor=True`.
        - `input_path`: `(Union[str, None])` The image path accepted by the image preprocessing sub-production line. When the input is a `numpy.ndarray`, it is saved as `None`.
        - `model_settings`: `(Dict)` The model configuration parameters of the preprocessing sub-production line.
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
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_formula_res_img.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (The production line usually contains many result images, so it is not recommended to specify a specific file path directly, otherwise multiple images will be overwritten and only the last one will be retained.)

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

In addition, you can obtain the configuration file of the formula recognition production line and load the configuration file for prediction. You can execute the following command to save the results in `my_path`:

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

<b>Note:</b> The parameters in the configuration file are initialization parameters for the production line. If you want to change the initialization parameters for the formula recognition production line, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in a configuration file, simply specify the path of the configuration file with `--pipeline`.

## 3. Development Integration/Deployment
If the formula recognition production line meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the formula recognition production line into your Python project, you can refer to the example code in [ 2.2 Integration via Python Script](#22-integration-via-python-script).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have strict performance requirements for deployment strategies, especially in terms of response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin, which aims to deeply optimize the performance of model inference and pre/post-processing, significantly speeding up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service-Based Deployment</b>: Service-based deployment is a common form of deployment in actual production environments. By encapsulating inference capabilities into services, clients can access these services via network requests to obtain inference results. PaddleX supports multiple production line service-based deployment solutions. For detailed production line service-based deployment procedures, please refer to the [PaddleX Service-Based Deployment Guide](../../../pipeline_deploy/serving.en.md).

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
<p>Get the formula recognition result of an image.</p>
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
<td>The URL of an image or PDF file accessible by the server, or the Base64-encoded content of the above file types. For PDF files exceeding 10 pages, only the content of the first 10 pages will be used.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code></td>
<td>The type of the file. <code>0</code> indicates a PDF file, and <code>1</code> indicates an image file. If this attribute is missing in the request body, the file type will be inferred from the URL.</td>
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
<td>The formula recognition result. The length of the array is 1 (for image input) or the smaller of the number of document pages and 10 (for PDF input). For PDF input, each element in the array represents the processing result of each page in the PDF file.</td>
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
<td><code>formulas</code></td>
<td><code>array</code></td>
<td>The positions and contents of the formulas.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code></td>
<td>The input image. The image is in JPEG format and is Base64-encoded.</td>
</tr>
<tr>
<td><code>layoutImage</code></td>
<td><code>string</code></td>
<td>The layout detection result image. The image is in JPEG format and is Base64-encoded.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>formulas</code> is an <code>object</code> with the following attributes:</p>
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
<td><code>poly</code></td>
<td><code>array</code></td>
<td>The position of the formula. The elements in the array are the coordinates of the vertices of the polygon surrounding the text.</td>
</tr>
<tr>
<td><code>latex</code></td>
<td><code>string</code></td>
<td>The content of the formula.</td>
</tr>
</tbody>
</table>
</details>
<details><summary>Multi-language Service Call Examples</summary>
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
    print("Detected formulas:")
    print(res["formulas"])
    layout_img_path = f"layout_{i}.jpg"
    with open(layout_img_path, "wb") as f:
        f.write(base64.b64decode(res["layoutImage"]))
    print(f"Output image saved at {layout_img_path}")
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computing and data processing capabilities directly on user devices, allowing the device to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model pipeline into subsequent AI applications.


## 4. Secondary Development
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
<td><a href="../../../module_usage/tutorials/ocr_modules/layout_detection.en.md">Link</a></td>
</tr>
<tr>
<td>Formula content is inaccurate</td>
<td>Formula Recognition Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/formula_recognition.en.md">Link</a></td>
</tr>
<tr>
<td>Whole-image rotation correction is inaccurate</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.en.md">Link</a></td>
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
        --layout_merge_bboxes_mode large \
        --save_path ./output \
        --device npu:0

```
Of course, you can also specify the hardware device when calling `create_pipeline()` or `predict()` in a Python script.

If you want to use the formula recognition production line on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
