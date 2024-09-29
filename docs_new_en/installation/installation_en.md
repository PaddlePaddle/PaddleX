# PaddleX Local Installation Tutorial

## 1. Quick Installation
Welcome to PaddleX, Baidu's low-code development tool for AI. Before we dive into the local installation process, please clarify your development needs and choose the appropriate installation mode.

PaddleX offers two installation modes: **Wheel Package Installation** and **Plugin Installation**. Below, we introduce their respective application scenarios:

### 1.1 Wheel Package Installation Mode
If your use case for PaddleX involves **model inference and integration**, we recommend the more **convenient** and **lightweight** Wheel package installation mode.

After installing PaddlePaddle (refer to the [PaddlePaddle Local Installation Tutorial](paddlepaddle_install_en.md)), you can quickly install the PaddleX Wheel package by executing the following commands:

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
pip install -e .
```

### 1.2 Plugin Installation Mode
If your use case for PaddleX involves **secondary development**, we recommend the more **powerful** plugin installation mode.

After installing the PaddleX plugins you need, you can not only perform inference and integration with the supported models but also conduct advanced operations such as model training for secondary development.

The plugins supported by PaddleX are listed below. Please determine the name(s) of the plugin(s) you need based on your development requirements:

| Plugin Name       | Basic Plugin Functions                         | Supported Pipelines                                                                  |
|-------------------|------------------------------------------------|------------------------------------------------------------------------------------|
| `PaddleClas`      | Image Classification, Feature Extraction       | General Image Classification Pipeline, General Multi-label Image Classification Pipeline, General Image Recognition Pipeline, Document Scene Information Extraction v3 Pipeline | 
| `PaddleDetection` | Object Detection, Instance Segmentation        | General Object Detection Pipeline, Small Object Detection Pipeline, Document Scene Information Extraction v3 Pipeline | 
| `PaddleOCR`       | OCR (Text Detection, Text Recognition), Table Recognition, Formula Recognition | General OCR Pipeline, General Table Recognition Pipeline, Document Scene Information Extraction v3 Pipeline | 
| `PaddleSeg`       | Semantic Segmentation, Image Anomaly Detection | General Instance Segmentation Pipeline, General Semantic Segmentation Pipeline | 
| `PaddleTS`        | Time Series Forecasting, Time Series Classification, Time Series Anomaly Detection | Time Series Forecasting Pipeline, Time Series Classification Pipeline, Time Series Anomaly Detection Pipeline | 

If the plugin you need to install is `PaddleXXX`, after installing PaddlePaddle (refer to the [PaddlePaddle Local Installation Tutorial](paddlepaddle_install_en.md)), you can quickly install the corresponding PaddleX plugin by executing the following commands:

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
pip install -e .
paddlex --install PaddleXXX
```

> ❗ Note: The two installation methods are not mutually exclusive, and you can install both simultaneously.

Next, we provide detailed installation tutorials for your reference. If you are using a Linux operating system, please refer to [2. Detailed Tutorial for Installing PaddleX on Linux](#2-detailed-tutorial-for-installing-paddlex-on-linux).

## 2. Detailed Tutorial for Installing PaddleX on Linux
When installing PaddleX on Linux, we **strongly recommend using the official PaddleX Docker image**. Alternatively, you can use other custom installation methods.

When using the official Docker image, **PaddlePaddle, PaddleX (including the wheel package and all plugins), and the corresponding CUDA environment are already pre-installed**. You can simply obtain the Docker image and start the container to begin using it.

When using custom installation methods, you need to first install the PaddlePaddle framework, then obtain the PaddleX source code, and finally choose the PaddleX installation mode.
```markdown
* If you want to delve deeper into the principles or usage of Docker, please refer to the [Docker Official Website](https://www.docker.com/) or the [Docker Official Tutorial](https://docs.docker.com/get-started/).
* If you are a CUDA 11.8 user, please ensure your Docker version is >= 19.03; if you are a CUDA 12.3 user, please ensure your Docker version is >= 20.10.

### 2.2 Custom Installation of PaddleX
Before installation, please ensure you have completed the local installation of PaddlePaddle by referring to the [PaddlePaddle Local Installation Tutorial](paddlepaddle_install_en.md).

#### 2.2.1 Obtain PaddleX Source Code
Next, use the following command to obtain the latest PaddleX source code from GitHub:

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git
```
If accessing GitHub is slow, you can download from Gitee instead, using the following command:

```bash
git clone https://gitee.com/paddlepaddle/PaddleX.git
```

#### 2.2.2 Install PaddleX
After obtaining the latest PaddleX source code, you can choose between Wheel package installation mode or plugin installation mode.

* **If you choose Wheel package installation mode**, execute the following commands:

```bash
cd PaddleX

# Install PaddleX whl
# -e: Install in editable mode, so changes to the current project's code will directly affect the installed PaddleX Wheel
pip install -e .
```

* **If you choose plugin installation mode** and the plugin you need is named PaddleXXX (there can be multiple), execute the following commands:

```bash
cd PaddleX

# Install PaddleX whl
# -e: Install in editable mode, so changes to the current project's code will directly affect the installed PaddleX Wheel
pip install -e .

# Install PaddleX plugins
paddlex --install PaddleXXX
```

For example, if you need to install the PaddleOCR and PaddleClas plugins, execute the following commands to install the plugins:

```bash
# Install PaddleOCR and PaddleClas plugins
paddlex --install PaddleOCR PaddleClas
```

If you need to install all plugins, you do not need to specify the plugin names, just execute the following command:

```bash
# Install all PaddleX plugins
paddlex --install
```

The default clone source for plugins is github.com, but it also supports gitee.com as a clone source. You can specify the clone source using `--platform`.

For example, if you need to use gitee.com as the clone source to install all PaddleX plugins, just execute the following command:

```bash
# Install PaddleX plugins
paddlex --install --platform gitee.com
```

After installation, you will see the following prompt:

```
All packages are installed.
```

For PaddleX installation on more hardware environments, please refer to the [PaddleX Multi-hardware Usage Guide](installation_other_devices_en.md)
