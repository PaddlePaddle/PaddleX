[简体中文](multi_devices_use_guide.md) | English

# PaddleX Multi-Hardware Usage Guide

This document focuses on the usage guide of PaddleX for Huawei Ascend NPU, Cambricon MLU, Kunlun XPU, and Hygon DCU hardware platforms.

## 1. Installation
### 1.1 PaddlePaddle Installation
First, please complete the installation of PaddlePaddle according to your hardware platform. The installation tutorials for each hardware are as follows:

Ascend NPU: [Ascend NPU PaddlePaddle Installation Guide](./paddlepaddle_install_NPU_en.md)

Cambricon MLU: [Cambricon MLU PaddlePaddle Installation Guide](./paddlepaddle_install_MLU_en.md)

Kunlun XPU: [Kunlun XPU PaddlePaddle Installation Guide](./paddlepaddle_install_XPU_en.md)

Hygon DCU: [Hygon DCU PaddlePaddle Installation Guide](./paddlepaddle_install_DCU_en.md)

### 1.2 PaddleX Installation
Welcome to use PaddlePaddle's low-code development tool, PaddleX. Before we officially start the local installation, please clarify your development needs and choose the appropriate installation mode based on your requirements.

PaddleX offers two installation modes: Wheel Package Installation and Plugin Installation. The following details the application scenarios and installation methods for these two modes.

#### 1.2.1 Wheel Package Installation Mode
If your application scenario for PaddleX is **model inference and integration**, we recommend using the more **convenient** and **lightweight** Wheel Package Installation Mode.

After installing PaddlePaddle, you can directly execute the following commands to quickly install the PaddleX Wheel package:

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0b1-py3-none-any.whl
```

#### 1.2.2 Plugin Installation Mode
If your application scenario for PaddleX is **secondary development**, we recommend using the more **powerful** Plugin Installation Mode.

After installing the PaddleX plugins you need, you can not only perform inference and integration on the models supported by the plugins but also conduct more advanced operations such as model training for secondary development.

The plugins supported by PaddleX are as follows. Please determine the name(s) of the plugin(s) you need based on your development requirements:

<details>
  <summary>👉 <b>Plugin and Pipeline Correspondence (Click to Expand)</b></summary>

| Pipeline | Module | Corresponding Plugin |
|-|-|-|
| General Image Classification | Image Classification | `PaddleClas` |
| General Object Detection | Object Detection | `PaddleDetection` |
| General Semantic Segmentation | Semantic Segmentation | `PaddleSeg` |
| General Instance Segmentation | Instance Segmentation | `PaddleDetection` |
| General OCR | Text Detection<br>Text Recognition | `PaddleOCR` |
| General Table Recognition | Layout Region Detection<br>Table Structure Recognition<br>Text Detection<br>Text Recognition | `PaddleOCR`<br>`PaddleDetection` |
| Document Scene Information Extraction v3 | Table Structure Recognition<br>Layout Region Detection<br>Text Detection<br>Text Recognition<br>Seal Text Detection<br>Document Image Correction<br>Document Image Orientation Classification | `PaddleOCR`<br>`PaddleDetection`<br>`PaddleClas` |
| Time Series Prediction | Time Series Prediction Module | `PaddleTS` |
| Time Series Anomaly Detection | Time Series Anomaly Detection Module | `PaddleTS` |
| Time Series Classification | Time Series Classification Module | `PaddleTS` |
| General Multi-label Classification | Image Multi-label Classification | `PaddleClas` |
| Small Object Detection | Small Object Detection | `PaddleDetection` |
| Image Anomaly Detection | Unsupervised Anomaly Detection | `PaddleSeg` |

</details>


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

The usage of PaddleX model pipeline development tool on hardware platforms such as Ascend NPU, Cambricon MLU, Kunlun XPU, and Hygon DCU is identical to that on GPU. You only need to modify the device configuration parameters according to your hardware platform. For detailed usage tutorials, please refer to [PaddleX Pipeline Development Tool Local Usage Guide](../pipeline_usage/pipeline_develop_guide_en.md).
