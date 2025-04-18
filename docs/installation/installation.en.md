---
comments: true
---

# PaddleX Local Installation Tutorial
> ❗Before installing PaddleX, please ensure you have a basic <b>Python environment</b> (Note: Currently supports Python 3.8 to Python 3.12, with more Python versions being adapted).
## 1. Quick Installation
Welcome to PaddleX, Baidu's low-code development tool for AI. Before we dive into the local installation process, please clarify your development needs and choose the appropriate installation mode.

PaddleX offers two installation modes: <b>Wheel Package Installation</b> and <b>Plugin Installation</b>. Below, we introduce their respective application scenarios:

### 1.1 Wheel Package Installation Mode
If your use case for PaddleX involves <b>model inference and integration</b>, we recommend the more <b>convenient</b> and <b>lightweight</b> Wheel package installation mode.

After installing PaddlePaddle (refer to the [PaddlePaddle Local Installation Tutorial](paddlepaddle_install.en.md)), you can quickly install the PaddleX Wheel package by executing the following commands:

> ❗ <b>Note</b>: Please ensure that PaddlePaddle is successfully installed before proceeding to the next step.

```bash
pip install "https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0rc0-py3-none-any.whl[base]"
```

### 1.2 Plugin Installation Mode
If your use case for PaddleX involves <b>custom development</b> (e.g. retraining models, fine-tuning models, customizing model structures, customizing inference codes, etc.), we recommend the more <b>powerful</b> plugin installation mode.

After installing the PaddleX plugins you need, you can not only perform inference and integration with the supported models but also conduct advanced operations such as model training for custom development.

The model training related plugins supported by PaddleX are listed below. Please determine the name(s) of the plugin(s) you need based on your development requirements:

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
<td>Table Recognition</td>
<td>Layout Region Detection<br>Table Structure Recognition<br>Text Detection<br>Text Recognition</td>
<td><code>PaddleOCR</code><br><code>PaddleDetection</code></td>
</tr>
<tr>
<td>PP-ChatOCRv3-doc</td>
<td>Table Structure Recognition<br>Layout Region Detection<br>Text Detection<br>Text Recognition<br>Seal Text Detection<br>Text Image Correction<br>Document Image Orientation Classification</td>
<td><code>PaddleOCR</code><br><code>PaddleDetection</code><br><code>PaddleClas</code></td>
</tr>
<tr>
<td>Time Series Forecasting</td>
<td>Time Series Forecasting Module</td>
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
<td>Image Multi-Label Classification</td>
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

If the plugin you need to install is `PaddleXXX`, after installing PaddlePaddle (refer to the [PaddlePaddle Local Installation Tutorial](paddlepaddle_install.en.md)), you can quickly install the corresponding PaddleX plugin by executing the following commands:

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
pip install -e ".[base]"
paddlex --install PaddleXXX
```

> ❗ Note: The two installation methods are not mutually exclusive, and you can install both simultaneously.

Next, we provide detailed installation tutorials for your reference. If you are using a Linux operating system, please refer to [2. Detailed Tutorial for Installing PaddleX on Linux](#2-detailed-tutorial-for-installing-paddlex-on-linux).

## 2. Detailed Tutorial for Installing PaddleX on Linux
When installing PaddleX on Linux, we <b>strongly recommend using the official PaddleX Docker image</b>. Alternatively, you can use other custom installation methods.

When using the official Docker image, <b>PaddlePaddle, PaddleX (including the wheel package and all plugins), and the corresponding CUDA environment are already pre-installed</b>. You can simply obtain the Docker image and start the container to begin using it. <b>Please note that the official Docker image of PaddleX is different from the official Docker image of the PaddlePaddle framework, as the latter does not come with PaddleX pre-installed.</b>

When using custom installation methods, you need to first install the PaddlePaddle framework, then obtain the PaddleX source code, and finally choose the PaddleX installation mode.
### 2.1 Get PaddleX based on Docker
Using the PaddleX official Docker image, create a container called 'paddlex' and map the current working directory to the '/paddle' directory inside the container by following the command.

If your Docker version >= 19.03, please use:

```bash
# For CPU
docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc0-paddlepaddle3.0.0rc0-cpu /bin/bash

# For GPU
# For CUDA11.8
docker run --gpus all --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc0-paddlepaddle3.0.0rc0-gpu-cuda11.8-cudnn8.6-trt8.5 /bin/bash

# For CUDA12.3
docker run --gpus all --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc0-paddlepaddle3.0.0rc0-gpu-cuda12.3-cudnn9.0-trt8.6 /bin/bash
```

* If your Docker version <= 19.03 and >= 17.06, please use:

<details><summary> Click Here</summary>

<pre><code class="language-bash"># For CPU
docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc0-paddlepaddle3.0.0rc0-cpu /bin/bash

# For GPU
# For CUDA11.8
nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc0-paddlepaddle3.0.0rc0-gpu-cuda11.8-cudnn8.6-trt8.5 /bin/bash

# For CUDA12.3
nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc0-paddlepaddle3.0.0rc0-gpu-cuda12.3-cudnn9.0-trt8.6 /bin/bash
</code></pre></details>

* If your Docker version <= 17.06, please update your Docker.

* If you want to delve deeper into the principles or usage of Docker, please refer to the [Docker Official Website](https://www.docker.com/) or the [Docker Official Tutorial](https://docs.docker.com/get-started/).

### 2.2 Custom Installation of PaddleX
Before installation, please ensure you have completed the local installation of PaddlePaddle by referring to the [PaddlePaddle Local Installation Tutorial](paddlepaddle_install.en.md).

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

* <b>If you choose Wheel package installation mode</b>, execute the following commands:

```bash
cd PaddleX

# Install PaddleX whl
# -e: Install in editable mode, so changes to the current project's code will directly affect the installed PaddleX Wheel
pip install -e ".[base]"
```

* <b>If you choose plugin installation mode</b> and the plugin you need is named PaddleXXX (there can be multiple), execute the following commands:

```bash
cd PaddleX

# Install PaddleX whl
# -e: Install in editable mode, so changes to the current project's code will directly affect the installed PaddleX Wheel
pip install -e ".[base]"

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

For PaddleX installation on more hardware environments, please refer to the [PaddleX Multi-hardware Usage Guide](../other_devices_support/multi_devices_use_guide.en.md)

### 2.3 Selective Installation of Dependencies

PaddleX offers a wide range of features, and different features require different dependencies. The features in PaddleX that can be used without installing plugins are categorized as "basic features." The official PaddleX Docker images have all dependencies required for these basic features preinstalled. Similarly, using the installation method introduced earlier—`pip install "...[base]"`—will install all dependencies needed for the basic features.

If you are only focused on a specific feature of PaddleX and want to minimize the size of the installed dependencies, you can selectively install them by specifying a "dependency group":

```bash
# For example, to install only the basic OCR features
# Install the precompiled wheel package
pip install "/url/of/wheel[ocr]"
# Install from source
pip install -e ".[ocr]"

# You can also specify multiple dependency groups at once
pip install -e ".[ocr,cv]"
```

PaddleX currently provides the following dependency groups:

| Dependency Group | Corresponding Features |
| - | - |
| `base` | All basic features of PaddleX. |
| `cv` | Basic features of computer vision pipelines. |
| `multimodal` | Basic features of multimodal pipelines. |
| `ie` | Basic features of information extraction pipelines. |
| `ocr` | Basic features of OCR-related pipelines. |
| `speech` | Basic features of speech pipeline.s |
| `ts` | Basic features of time series pipelines. |
| `video` | Basic features of video pipelines. |
| `serving` | The serving feature. Installing this group is equivalent to installing the PaddleX serving plugin; the plugin can also be installed via the PaddleX CLI. |
| `plugins` | All plugin-provided features that support installation via dependency groups. |
| `all` | All basic features of PaddleX, as well as all plugin-provided features installable via dependency groups. |

Each pipeline belongs to exactly one dependency group. You can refer to the tutorial of each pipeline to find out which dependency group it belongs to. For modules, you can access the related basic features by installing any dependency group that includes the module.
