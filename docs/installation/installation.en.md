---
comments: true
---

# PaddleX Local Installation Tutorial
> ❗Before installing PaddleX, please ensure you have a basic <b>Python environment</b> (Note: Python 3.8 through 3.13 are currently supported).

## 1. Quick Installation
Welcome to PaddleX, Baidu's low-code development tool for Paddle. Before we start local installation, please clarify your development needs and choose an appropriate installation mode.

PaddleX offers two installation modes: <b>Wheel package installation</b> and <b>plugin installation</b>. Their use cases are described below.

### 1.1 Wheel Package Installation Mode
If your use case for PaddleX is <b>model inference and integration</b>, we recommend the more <b>convenient</b> and <b>lightweight</b> wheel package installation mode.

After installing the lightweight wheel package, you can run inference with all models supported by PaddleX and integrate them directly into your project.

Install the PaddleX wheel package with:

```bash
# Only required dependencies (optional dependencies can be installed later as needed)
pip install paddlex
```

Install optional dependencies as needed (see [3 Selective Installation of Dependencies](#3-selective-installation-of-dependencies)):

Install all dependencies for PaddleX “basic features”:

```bash
pip install "paddlex[base]"
```

Install dependencies for a single feature only:

```bash
pip install "paddlex[ocr]"
```

### 1.2 Plugin Installation Mode
If your use case for PaddleX is <b>custom development</b> (e.g. retraining models, fine-tuning, custom model structures, custom inference code, etc.), we recommend the more <b>capable</b> plugin installation mode.

After installing the PaddleX plugins you need, you can still run inference and integration for supported models, and perform advanced tasks such as model training.

The model-training-related plugins supported by PaddleX are listed below. Choose one or more plugin names according to your needs:

<details><summary>👉 <b>Plugin and pipeline correspondence (click to expand)</b></summary>

<table>
<thead>
<tr>
<th>Pipeline</th>
<th>Module</th>
<th>Corresponding plugin</th>
</tr>
</thead>
<tbody>
<tr>
<td>General image classification</td>
<td>Image classification</td>
<td><code>PaddleClas</code></td>
</tr>
<tr>
<td>General object detection</td>
<td>Object detection</td>
<td><code>PaddleDetection</code></td>
</tr>
<tr>
<td>General semantic segmentation</td>
<td>Semantic segmentation</td>
<td><code>PaddleSeg</code></td>
</tr>
<tr>
<td>General instance segmentation</td>
<td>Instance segmentation</td>
<td><code>PaddleDetection</code></td>
</tr>
<tr>
<td>General OCR</td>
<td>Document image orientation classification<br>Text image unwarping<br>Text detection<br>Text line orientation classification<br>Text recognition</td>
<td><code>PaddleOCR</code><br><code>PaddleClas</code></td>
</tr>
<tr>
<td>General table recognition</td>
<td>Layout region detection<br>Table structure recognition<br>Text detection<br>Text recognition</td>
<td><code>PaddleOCR</code><br><code>PaddleDetection</code></td>
</tr>
<tr>
<td>Document scene information extraction v3</td>
<td>Table structure recognition<br>Layout region detection<br>Text detection<br>Text recognition<br>Seal text detection<br>Text image unwarping<br>Document image orientation classification</td>
<td><code>PaddleOCR</code><br><code>PaddleDetection</code><br><code>PaddleClas</code></td>
</tr>
<tr>
<td>Time series forecasting</td>
<td>Time series forecasting module</td>
<td><code>PaddleTS</code></td>
</tr>
<tr>
<td>Time series anomaly detection</td>
<td>Time series anomaly detection module</td>
<td><code>PaddleTS</code></td>
</tr>
<tr>
<td>Time series classification</td>
<td>Time series classification module</td>
<td><code>PaddleTS</code></td>
</tr>
<tr>
<td>General multi-label classification</td>
<td>Image multi-label classification</td>
<td><code>PaddleClas</code></td>
</tr>
<tr>
<td>Small object detection</td>
<td>Small object detection</td>
<td><code>PaddleDetection</code></td>
</tr>
<tr>
<td>Image anomaly detection</td>
<td>Unsupervised anomaly detection</td>
<td><code>PaddleSeg</code></td>
</tr>
</tbody>
</table></details>

If the plugin you need is `PaddleXXX`, install the corresponding PaddleX plugin with:

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
pip install -e .
paddlex --install PaddleXXX  # e.g. PaddleOCR
```

> ❗ Note: This installs PaddleX in editable mode; changes under the project directory apply directly to the installed PaddleX wheel.

If the steps above succeed, you can skip the rest of this section.

If you use Linux, see [2. Detailed guide for installing PaddleX on Linux](#2-detailed-guide-for-installing-paddlex-on-linux). Installation on other operating systems will be documented later.

## 2. Detailed guide for installing PaddleX on Linux
When installing PaddleX on Linux, we <b>strongly recommend using the official PaddleX Docker image</b>. You may also use a custom installation path.

With the official Docker image, <b>PaddlePaddle, PaddleX (wheel and all plugins), and the CUDA stack are preconfigured</b>; <b>pull the image and start the container to begin</b>. <b>Note: the PaddleX official image is not the same as the PaddlePaddle framework official image—the latter does not include PaddleX.</b>

With a custom installation, you usually install the PaddlePaddle framework or other dependencies per Sections 4 and 5 first, then obtain the PaddleX source code, and finally choose an installation mode.

> ❗ You do not need to match the host CUDA version; only the GPU driver version matters.

### 2.1 Get PaddleX via Docker
Create a container named `paddlex` and mount the current working directory to `/paddle` in the container using the PaddleX official image.

If your Docker version is >= 19.03, run:

```bash
# CPU
docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.3.11-paddlepaddle3.2.0-cpu /bin/bash

# GPU — GPU driver >= 450.80.02 (Linux) or >= 452.39 (Windows)
docker run --gpus all --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.3.11-paddlepaddle3.2.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash

# GPU — GPU driver >= 545.23.06 (Linux) or >= 545.84 (Windows)
docker run --gpus all --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.3.11-paddlepaddle3.2.0-gpu-cuda12.6-cudnn9.5 /bin/bash

# GPU — GPU driver >= 550.xx
docker run --gpus all --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.3.11-paddlepaddle3.2.0-gpu-cuda12.9-cudnn9.9 /bin/bash
```

* If your Docker version is <= 19.03 and >= 17.06, run:

<details><summary> Click to expand</summary>

<pre><code class="language-bash"># CPU
docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.3.11-paddlepaddle3.2.0-cpu /bin/bash

# GPU — GPU driver >= 450.80.02 (Linux) or >= 452.39 (Windows)
nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.3.11-paddlepaddle3.2.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash

# GPU — GPU driver >= 545.23.06 (Linux) or >= 545.84 (Windows)
nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.3.11-paddlepaddle3.2.0-gpu-cuda12.6-cudnn9.5 /bin/bash

# GPU — GPU driver >= 550.xx
nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.3.11-paddlepaddle3.2.0-gpu-cuda12.9-cudnn9.9 /bin/bash

</code></pre></details>

* If your Docker version is <= 17.06, upgrade Docker.

* For more on Docker, see the [Docker website](https://www.docker.com/) or the [Docker getting started guide](https://docs.docker.com/get-started/).

### 2.2 Custom installation of PaddleX

#### 2.2.1 Obtain the PaddleX source code
Clone the latest PaddleX source from GitHub:

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git
```

If GitHub is slow, use Gitee:

```bash
git clone https://gitee.com/paddlepaddle/PaddleX.git
```

#### 2.2.2 Install PaddleX
After cloning, choose wheel installation or plugin installation.

<b>Wheel installation mode</b> — run:

```bash
cd PaddleX

# Install the PaddleX wheel
# -e: editable install; local changes apply to the installed wheel
pip install -e ".[base]"
```

<b>Plugin installation mode</b> — if you need plugins named `PaddleXXX` (one or more), run:

```bash
cd PaddleX

# Install the PaddleX wheel
# -e: editable install; local changes apply to the installed wheel
pip install -e ".[base]"

# Install PaddleX plugins
paddlex --install PaddleXXX
```

For example, to install PaddleOCR and PaddleClas:

```bash
# Install PaddleOCR and PaddleClas plugins
paddlex --install PaddleOCR PaddleClas
```

To install all plugins, omit names:

```bash
# Install all PaddleX plugins
paddlex --install
```

The default clone host is github.com; you can use gitee.com with `--platform`.

For example, to install all plugins from Gitee:

```bash
# Install PaddleX plugins
paddlex --install --platform gitee.com
```

When finished, you should see:

```
All packages are installed.
```

For more hardware targets, see the [PaddleX multi-hardware guide](../other_devices_support/multi_devices_use_guide.en.md).

## 3 Selective installation of dependencies

PaddleX has many features, each with different dependencies. Features that work without plugins are called “basic features.” The official PaddleX Docker images include all basic-feature dependencies; `pip install "...[base]"` does the same. To keep the install small, install only the dependency groups you need:

```bash
# Example: OCR basic features only

# Prebuilt wheel
pip install "paddlex[ocr]"
# From source
pip install -e ".[ocr]"

# Multiple groups at once
pip install -e ".[ocr,cv]"
```

Available dependency groups:

| Dependency group | Features |
| - | - |
| `base` | All basic PaddleX features. |
| `cv` | Basic computer vision pipelines. |
| `multimodal` | Basic multimodal pipelines. |
| `ie` | Basic information-extraction pipelines. |
| `ocr` | Basic OCR-related pipelines. |
| `speech` | Basic speech pipelines. |
| `ts` | Basic time-series pipelines. |
| `video` | Basic video pipelines. |
| `trans` | Basic translation pipelines. |
| `genai-client` | Generative AI client. Installing this group is equivalent to installing the PaddleX generative AI client plugin; you can also install that plugin via the PaddleX CLI. |
| `genai-sglang-server` | SGLang server. Installing this group is equivalent to installing the PaddleX SGLang server plugin; you can also install it via the PaddleX CLI. |
| `genai-vllm-server` | vLLM server. Installing this group is equivalent to installing the PaddleX vLLM server plugin; you can also install it via the PaddleX CLI. |
| `serving` | Serving deployment. Installing this group is equivalent to installing the PaddleX serving plugin; you can also install it via the PaddleX CLI. |
| `paddle2onnx` | Paddle2ONNX. Installing this group is equivalent to installing the PaddleX Paddle2ONNX plugin; you can also install it via the PaddleX CLI. |

Each pipeline belongs to exactly one dependency group; each pipeline’s doc states its group. For a single module, install any dependency group that covers the pipelines using that module to enable the related basic features.

## 4 Installing the PaddlePaddle framework

When you run inference with the PaddlePaddle framework, install it following the [PaddlePaddle local installation tutorial](paddlepaddle_install.en.md).

## 5 Additional dependencies for non-Paddle engines

When you use non-Paddle engines for inference, install the matching dependencies:

- `engine="transformers"`: install the `transformers` library (e.g. `pip install transformers`) and configure the environment per the [Transformers installation guide](https://huggingface.co/docs/transformers/installation).
