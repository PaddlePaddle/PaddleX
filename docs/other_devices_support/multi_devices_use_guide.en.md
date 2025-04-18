---
comments: true
---

# PaddleX Multi-Devices Usage Guide

This document focuses on the usage guide of PaddleX for Huawei Ascend NPU, Cambricon MLU, Kunlun XPU, and Hygon DCU hardware platforms.

## 1. Installation
### 1.1 PaddlePaddle Installation
First, please complete the installation of PaddlePaddle according to your hardware platform. The installation tutorials for each hardware are as follows:

Ascend NPU: [Ascend NPU PaddlePaddle Installation Guide](./paddlepaddle_install_NPU.en.md)

Cambricon MLU: [Cambricon MLU PaddlePaddle Installation Guide](./paddlepaddle_install_MLU.en.md)

Kunlun XPU: [Kunlun XPU PaddlePaddle Installation Guide](./paddlepaddle_install_XPU.en.md)

Hygon DCU: [Hygon DCU PaddlePaddle Installation Guide](./paddlepaddle_install_DCU.en.md)

Enflame GCU: [Enflame GCU PaddlePaddle Installation Guide](./paddlepaddle_install_GCU.en.md)

### 1.2 PaddleX Installation
Welcome to use PaddlePaddle's low-code development tool, PaddleX. Before we officially start the local installation, please clarify your development needs and choose the appropriate installation mode based on your requirements.

PaddleX offers two installation modes: Wheel Package Installation and Plugin Installation. The following details the application scenarios and installation methods for these two modes.

#### 1.2.1 Wheel Package Installation Mode
If your application scenario for PaddleX is <b>model inference and integration</b>, we recommend using the more <b>convenient</b> and <b>lightweight</b> Wheel Package Installation Mode.

After installing PaddlePaddle, you can directly execute the following commands to quickly install the PaddleX Wheel package:

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0b1-py3-none-any.whl
```

#### 1.2.2 Plugin Installation Mode
If your application scenario for PaddleX is <b>secondary development</b>, we recommend using the more <b>powerful</b> Plugin Installation Mode.

After installing the PaddleX plugins you need, you can not only perform inference and integration on the models supported by the plugins but also conduct more advanced operations such as model training for secondary development.

The plugins supported by PaddleX are as follows. Please determine the name(s) of the plugin(s) you need based on your development requirements:

<details><summary>👉 <b>Plugin and Pipeline Correspondence (Click to Expand)</b></summary>

<table>
<thead>
<tr>
<th>Pipeline</th>
<th>Module</th>
<th>Corresponding Plugin</th>
</tr>
</thead>
<tbody>
<tr>
<td>General Image Classification</td>
<td>Image Classification</td>
<td><code>PaddleClas</code></td>
</tr>
<tr>
<td>General Object Detection</td>
<td>Object Detection</td>
<td><code>PaddleDetection</code></td>
</tr>
<tr>
<td>General Semantic Segmentation</td>
<td>Semantic Segmentation</td>
<td><code>PaddleSeg</code></td>
</tr>
<tr>
<td>General Instance Segmentation</td>
<td>Instance Segmentation</td>
<td><code>PaddleDetection</code></td>
</tr>
<tr>
<td>General OCR</td>
<td>Text Detection<br>Text Recognition</td>
<td><code>PaddleOCR</code></td>
</tr>
<tr>
<td>General Table Recognition</td>
<td>Layout Region Detection<br>Table Structure Recognition<br>Text Detection<br>Text Recognition</td>
<td><code>PaddleOCR</code><br><code>PaddleDetection</code></td>
</tr>
<tr>
<td>Document Scene Information Extraction v3</td>
<td>Table Structure Recognition<br>Layout Region Detection<br>Text Detection<br>Text Recognition<br>Seal Text Detection<br>Document Image Correction<br>Document Image Orientation Classification</td>
<td><code>PaddleOCR</code><br><code>PaddleDetection</code><br><code>PaddleClas</code></td>
</tr>
<tr>
<td>Time Series Prediction</td>
<td>Time Series Prediction Module</td>
<td><code>PaddleTS</code></td>
</tr>
<tr>
<td>Time Series Anomaly Detection</td>
<td>Time Series Anomaly Detection Module</td>
<td><code>PaddleTS</code></td>
</tr>
<tr>
<td>Time Series Classification</td>
<td>Time Series Classification Module</td>
<td><code>PaddleTS</code></td>
</tr>
<tr>
<td>General Multi-label Classification</td>
<td>Image Multi-label Classification</td>
<td><code>PaddleClas</code></td>
</tr>
<tr>
<td>Small Object Detection</td>
<td>Small Object Detection</td>
<td><code>PaddleDetection</code></td>
</tr>
<tr>
<td>Image Anomaly Detection</td>
<td>Unsupervised Anomaly Detection</td>
<td><code>PaddleSeg</code></td>
</tr>
</tbody>
</table></details>


If the plugin(s) you need to install is/are PaddleXXX (can be multiple), after installing PaddlePaddle, you can directly execute the following commands to quickly install the corresponding PaddleX plugin(s):

```bash
# obtain PaddleX source code
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX

# Install PaddleX whl
# -e: Install in editable mode, so changes to the current project's code will directly affect the installed PaddleX Wheel
pip install -e .

# Install PaddleX Plugins
paddlex --install PaddleXXX
```
For example, if you need to install the PaddleOCR and PaddleClas plugins, you can execute the following command:

```bash
# Install PaddleOCR and PaddleClas Plugins
paddlex --install PaddleOCR PaddleClas
```

If you wish to install all plugins, you do not need to specify the plugin names. Simply execute the following command:

```bash
# Install All PaddleX Plugins
paddlex --install
```

The default clone source for plugins is github.com, but it also supports gitee.com. You can specify the clone source using `--platform`.

For instance, if you want to install all PaddleX plugins using the gitee.com clone source, execute the following command:

```bash
# Install PaddleX Plugins using gitee.com
paddlex --install --platform gitee.com
```

Upon successful installation, you will see the following prompt:

```
All packages are installed.
```

## 2. Usage

The usage of PaddleX model pipeline development tool on hardware platforms such as Ascend NPU, Cambricon MLU, Kunlun XPU, Hygon DCU and Enflame GCU is identical to that on GPU. You only need to modify the device configuration parameters according to your hardware platform.
Taking the general OCR pipeline as an example to introduce the pipeline development tools.The General OCR Pipeline is designed to solve text recognition tasks, extracting text information from images and outputting it in text form. PP-OCRv4 is an end-to-end OCR system that achieves millisecond-level text content prediction on CPUs, reaching state-of-the-art (SOTA) performance in open-source projects for general scenarios.You can directly use the pre-trained models provided by OCR pipeline for inference.
* Command line

```bash
paddlex --pipeline OCR --input general_ocr_002.png --device npu:0 # change the device name to npu, mlu, xpu, dcu or gcu
```
* Python Script

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="OCR", device="npu:0") # change the device name to npu, mlu, xpu, dcu or gcu

output = pipeline.predict("general_ocr_002.png")
for res in output:
    res.print()
    res.save_to_img("./output/")
```
If you are not satisfied with the performance of the pre-trained model, you can fine-tune it. For example, let's discuss single model development using the PP-OCRv4 mobile text detection model (PP-OCRv4_mobile_det).

```bash
# train
python main.py -c paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ocr_det_dataset_examples \ # change the dataset_dir to your own dataset path
    -o Global.device=npu:0,1,2,3 # change the device name to npu, mlu, xpu, dcu or gcu

# predict
python main.py -c paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_accuracy/inference" \
    -o Predict.input="general_ocr_001.png"
    -o Global.device=npu # change the device name to npu, mlu, xpu, dcu or gcu
```
For more detailed usage tutorials, please refer to [PaddleX Pipeline Development Tool Local Usage Guide](../pipeline_usage/pipeline_develop_guide.en.md).
