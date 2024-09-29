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

#### 1.2.1 Obtain PaddleX Source Code
Please use the following command to obtain the latest source code of PaddleX from GitHub:

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git
```
If accessing GitHub is slow, you can download from Gitee instead, using the following command:

```bash
git clone https://gitee.com/paddlepaddle/PaddleX.git
```

#### 1.2.2 Wheel Package Installation Mode
If your application scenario for PaddleX is **model inference and integration**, we recommend using the more **convenient** and **lightweight** Wheel Package Installation Mode.

After installing PaddlePaddle, you can directly execute the following commands to quickly install the PaddleX Wheel package:

```bash
cd PaddleX

# Install PaddleX whl
# -e: Install in editable mode, so changes to the current project's code will directly affect the installed PaddleX Wheel
pip install -e .
```

#### 1.2.3 Plugin Installation Mode
If your application scenario for PaddleX is **secondary development**, we recommend using the more **powerful** Plugin Installation Mode.

After installing the PaddleX plugins you need, you can not only perform inference and integration on the models supported by the plugins but also conduct more advanced operations such as model training for secondary development.

The plugins supported by PaddleX are as follows. Please determine the name(s) of the plugin(s) you need based on your development requirements:

| Plugin Name | Basic Plugin Functions | Supported Pipelines | Reference Documentation |
|-|-|-|-|
| PaddleClas | Image Classification, Feature Extraction | General Image Classification Pipeline, General Multi-label Image Classification Pipeline, General Image Recognition Pipeline, Document Scene Information Extraction v3 Pipeline | General Image Classification Pipeline Usage Guide |
| PaddleDetection | Object Detection, Instance Segmentation | General Object Detection Pipeline, Small Object Detection Pipeline, Document Scene Information Extraction v3 Pipeline | General Object Detection Pipeline Usage Guide |
| PaddleOCR | OCR (Text Detection, Text Recognition), Table Recognition, Formula Recognition | General OCR Pipeline, General Table Recognition Pipeline, Document Scene Information Extraction v3 Pipeline | General OCR Pipeline Usage Guide |
| PaddleSeg | Semantic Segmentation, Image Anomaly Detection | General Instance Segmentation Pipeline, General Semantic Segmentation Pipeline | General Semantic Segmentation Pipeline Usage Guide |
| PaddleTS | Time Series Forecasting, Time Series Classification, Time Series Anomaly Detection | Time Series Forecasting Pipeline, Time Series Classification Pipeline, Time Series Anomaly Detection Pipeline | Time Series Forecasting Pipeline Usage Guide |

If the plugin(s) you need to install is/are PaddleXXX (can be multiple), after installing PaddlePaddle, you can directly execute the following commands to quickly install the corresponding PaddleX plugin(s):

```bash
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