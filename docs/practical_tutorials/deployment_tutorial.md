---
comments: true
---

# PaddleX 3.0 产线部署教程

在使用本教程之前，您首先需要安装 PaddleX，安装方式请参考[ PaddleX 安装](../installation/installation.md)。

PaddleX 的三种部署方式详细说明如下：

* 高性能推理：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考 [PaddleX 高性能推理指南](../pipeline_deploy/high_performance_inference.md)。

* 服务化部署：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考 [PaddleX 服务化部署指南](../pipeline_deploy/serving.md)。

* 端侧部署：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考 [PaddleX端侧部署指南](../pipeline_deploy/edge_deploy.md)。

本教程将举三个实际应用例子，来依次介绍 PaddleX 的三种部署方式。

## 1 高性能推理示例

### 1.1 安装高性能推理插件

根据设备类型，执行如下指令，安装高性能推理插件：

如果你的设备是 CPU，请使用以下命令安装 PaddleX 的 CPU 版本：

```bash
paddlex --install hpi-cpu
```

如果你的设备是 GPU，请使用以下命令安装 PaddleX 的 GPU 版本。请注意，GPU 版本包含了 CPU 版本的所有功能，因此无需单独安装 CPU 版本：

```bash
paddlex --install hpi-gpu
```

目前高性能推理支持的处理器架构、操作系统、设备类型和 Python 版本如下表所示：

<table>
  <tr>
    <th>处理器架构</th>
    <th>操作系统</th>
    <th>设备类型</th>
    <th>Python 版本</th>
  </tr>
  <tr>
    <td rowspan="7">x86-64</td>
    <td rowspan="7">Linux</td>
    <td rowspan="4">CPU</td>
  </tr>
  <tr>
    <td>3.8</td>
  </tr>
  <tr>
    <td>3.9</td>
  </tr>
  <tr>
    <td>3.10</td>
  </tr>
  <tr>
    <td rowspan="3">GPU&nbsp;（CUDA&nbsp;11.8&nbsp;+&nbsp;cuDNN&nbsp;8.6）</td>
    <td>3.8</td>
  </tr>
  <tr>
    <td>3.9</td>
  </tr>
  <tr>
    <td>3.10</td>
  </tr>
</table>

### 1.2 启用高性能推理插件

对于 PaddleX CLI，指定 `--use_hpip`，即可启用高性能推理插件。以通用图像分类产线为例：

```bash
paddlex \
    --pipeline image_classification \
    --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
    --device gpu:0 \
    --use_hpip
```

对于 PaddleX Python API，启用高性能推理插件的方法类似。以通用图像分类产线和图像分类模块为例：

通用图像分类产线：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="image_classification",
    device="gpu",
    use_hpip=True
)

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
```

图像分类模块：

```python
from paddlex import create_model

model = create_model(
    model_name="ResNet18",
    device="gpu",
    use_hpip=True
)

output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
```

启用高性能推理插件得到的推理结果与未启用插件时一致。对于部分模型，在首次启用高性能推理插件时，可能需要花费较长时间完成推理引擎的构建。PaddleX 将在推理引擎的第一次构建完成后将相关信息缓存在模型目录，并在后续复用缓存中的内容以提升初始化速度。

### 1.3 推理步骤

本推理步骤基于 <b>PaddleX CLI、联网激活序列号、Python 3.10.0、设备类型为CPU</b> 的方式使用高性能推理插件，其他使用方式（如不同 Python 版本、设备类型或 PaddleX Python API）可参考 [PaddleX 高性能推理指南](../pipeline_deploy/high_performance_inference.md) 替换相应的指令。

```bash
# 安装高性能推理插件
paddlex --install hpi-gpu
# 确保当前环境的 `LD_LIBRARY_PATH` 没有指定 TensorRT 的共享库目录 可以使用下面命令去除或手动去除
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v TensorRT | tr '\n' ':' | sed 's/:*$//')
# 执行推理
paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0 --save_path ./output
```

运行结果：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/deployment/01.png"  width="700" />
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/deployment/02.png"  width="700" />

### 1.4 更换产线或模型

- 更换产线：

  若想更换其他产线使用高性能推理插件，则替换 `--pipeline` 传入的值即可，以下以通用目标检测产线为例：

  ```bash
  paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device gpu:0 --use_hpip --save_path ./output
  ```

- 更换模型：

  OCR 产线默认使用 PP-OCRv4_mobile_det、PP-OCRv4_mobile_rec 模型，若想更换其他模型，如 PP-OCRv4_server_det、PP-OCRv4_server_rec 模型，可参考 [通用OCR产线使用教程](../pipeline_usage/tutorials/ocr_pipelines/OCR.md)，具体操作如下：

  ```bash
  # 1. 修改 OCR 产线配置文件
  #    将 SubModules.TextDetection.model_name 的值改为 PP-OCRv4_server_det
  #    将 SubModules.TextDetection.model_dir 的值改为 PP-OCRv4_server_det 模型所在路径
  #    将 SubModules.TextRecognition.model_name 的值改为 PP-OCRv4_server_rec
  #    将 SubModules.TextRecognition.model_dir 的值改为 PP-OCRv4_server_rec 模型所在路径

  # 2. 执行推理
  paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0 --use_hpip --save_path ./output
  ```

  通用目标检测产线默认使用 PicoDet-S 模型，若想更换其他模型，如 RT-DETR 模型，可参考 [通用目标检测产线使用教程](../pipeline_usage/tutorials/cv_pipelines/object_detection.md)，具体操作如下：

  ```bash
  # 1. 修改 object_detection 产线配置文件
  #    将 SubModules.ObjectDetection.model_name 的值改为 RT-DETR
  #    将 SubModules.ObjectDetection.model_dir 的值改为 RT-DETR 模型所在路径

  # 2. 执行推理
  paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0 --use_hpip --save_path ./output
  ```

  其他产线的操作与上述两条产线的操作类似，更多细节可参考产线使用教程。


## 2 服务化部署示例

### 2.1 安装服务化部署插件

执行如下指令，安装服务化部署插件：

```
paddlex --install serving
```

### 2.2 启动服务

通过 PaddleX CLI 启动服务，指令格式为：

```shell
paddlex --serve --pipeline {产线名称或产线配置文件路径} [{其他命令行选项}]
```

以通用OCR产线为例：

```shell
paddlex --serve --pipeline OCR
```

服务启动成功后，可以看到类似如下展示的信息：

```
INFO:     Started server process [63108]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

`--pipeline` 可指定为官方产线名称或本地产线配置文件路径。PaddleX 以此构建产线并部署为服务。如需调整配置（如模型路径、batch_size、部署设备等），请参考[通用OCR产线使用教程](../pipeline_usage/tutorials/ocr_pipelines/OCR.md)中的 <b>“模型应用”</b> 部分。
与服务化部署相关的命令行选项如下：

<table>
<thead>
<tr>
<th>名称</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>--pipeline</code></td>
<td>产线名称或产线配置文件路径。</td>
</tr>
<tr>
<td><code>--device</code></td>
<td>产线部署设备。默认为 <code>cpu</code>（如 GPU 不可用）或 <code>gpu</code>（如 GPU 可用）。</td>
</tr>
<tr>
<td><code>--host</code></td>
<td>服务器绑定的主机名或 IP 地址。默认为0.0.0.0。</td>
</tr>
<tr>
<td><code>--port</code></td>
<td>服务器监听的端口号。默认为8080。</td>
</tr>
<tr>
<td><code>--use_hpip</code></td>
<td>如果指定，则启用高性能推理插件。</td>
</tr>
<tr>
<td><code>--serial_number</code></td>
<td>高性能推理插件使用的序列号。只在启用高性能推理插件时生效。 请注意，并非所有产线、模型都支持使用高性能推理插件，详细的支持情况请参考<a href="../pipeline_deploy/high_performance_inference.md">PaddleX 高性能推理指南</a>。</td>
</tr>
<tr>
<td><code>--update_license</code></td>
<td>如果指定，则进行联网激活。只在启用高性能推理插件时生效。</td>
</tr>
</tbody>
</table>
### 2.3 调用服务

此处只展示 Python 调用示例，API参考和其他语言服务调用示例可参考 [PaddleX服务化部署指南](../pipeline_deploy/serving.md) 的 <b>1.3 调用服务</b> 中各产线使用教程的 <b>“开发集成/部署”</b> 部分。

```python
import base64
import requests

API_URL = "http://localhost:8080/ocr" # 服务URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64编码的文件内容或者图像URL

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected texts:")
print(result["texts"])
```

### 2.4 部署步骤

```bash
# 安装服务化部署插件
paddlex --install serving
# 启动服务
paddlex --serve --pipeline OCR
# 调用服务 | fast_test.py 中代码为上一节的 Python 调用示例
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png -O demo.jpg
python fast_test.py
```

运行结果：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/deployment/03.png"  width="700" />
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/deployment/04.png"  width="700" />

### 2.5 更换产线或模型

- 更换产线：

  若想更换其他产线进行服务化部署，则替换 `--pipeline` 传入的值即可，以下以通用目标检测产线为例：

  ```bash
  paddlex --serve --pipeline object_detection
  ```

- 更换模型：

  OCR 产线默认使用 PP-OCRv4_mobile_det、PP-OCRv4_mobile_rec 模型，若想更换其他模型，如 PP-OCRv4_server_det、PP-OCRv4_server_rec 模型，可参考 [通用OCR产线使用教程](../pipeline_usage/tutorials/ocr_pipelines/OCR.md)，具体操作如下：

  ```bash
  # 1. 获取 OCR 产线配置文件并保存到 ./OCR.yaml
  paddlex --get_pipeline_config OCR --save_path ./OCR.yaml

  # 2. 修改 ./OCR.yaml 配置文件
  #    将 Pipeline.text_det_model 的值改为 PP-OCRv4_server_det 模型所在路径
  #    将 Pipeline.text_rec_model 的值改为 PP-OCRv4_server_rec 模型所在路径

  # 3. 启动服务时使用修改后的配置文件
  paddlex --serve --pipeline ./OCR.yaml
  # 4. 调用服务
  wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png -O demo.jpg
  python fast_test.py
  ```

  通用目标检测产线默认使用 PicoDet-S 模型，若想更换其他模型，如 RT-DETR 模型，可参考 [通用目标检测产线使用教程](../pipeline_usage/tutorials/cv_pipelines/object_detection.md)，具体操作如下：

  ```bash
  # 1. 获取 OCR 产线配置文件并保存到 ./object_detection.yaml
  paddlex --get_pipeline_config object_detection --save_path ./object_detection.yaml

  # 2. 修改 ./object_detection.yaml 配置文件
  #    将 Pipeline.model 的值改为 RT-DETR 模型所在路径

  # 3. 启动服务时使用修改后的配置文件
  paddlex --serve --pipeline ./object_detection.yaml
  # 4. 调用服务 | fast_test.py 需要替换为通用目标检测产线使用教程中的 Python 调用示例
  wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png -O demo.jpg
  python fast_test.py
  ```

  其他产线的操作与上述两条产线的操作类似，更多细节可参考产线使用教程。

## 3 端侧部署示例

### 3.1 环境准备

1. 在本地环境安装好 CMake 编译工具，并在 [Android NDK 官网](https://developer.android.google.cn/ndk/downloads)下载当前系统符合要求的版本的 NDK 软件包。例如，在 Mac 上开发，需要在 Android NDK 官网下载 Mac 平台的 NDK 软件包。

    <b>环境要求</b>
    -  `CMake >= 3.10`（最低版本未经验证，推荐 3.20 及以上）
    -  `Android NDK >= r17c`（最低版本未经验证，推荐 r20b 及以上）

    <b>本指南所使用的测试环境：</b>
    -  `cmake == 3.20.0`
    -  `android-ndk == r20b`

2. 准备一部 Android 手机，并开启 USB 调试模式。开启方法: `手机设置 -> 查找开发者选项 -> 打开开发者选项和 USB 调试模式`。

3. 电脑上安装 ADB 工具，用于调试。ADB 安装方式如下：

    3.1. Mac 电脑安装 ADB

    ```shell
     brew cask install android-platform-tools
    ```

    3.2. Linux 安装 ADB

    ```shell
     # debian系linux发行版的安装方式
     sudo apt update
     sudo apt install -y wget adb

     # redhat系linux发行版的安装方式
     sudo yum install adb
    ```

    3.3. Windows 安装 ADB

    win 上安装需要去谷歌的安卓平台下载 ADB 软件包进行安装：[链接](https://developer.android.com/studio)

    打开终端，手机连接电脑，在终端中输入

    ```shell
     adb devices
    ```

    如果有 device 输出，则表示安装成功。

    ```shell
     List of devices attached
     744be294    device
    ```

### 3.2 物料准备

1. 克隆 `Paddle-Lite-Demo` 仓库的 `feature/paddle-x` 分支到 `PaddleX-Lite-Deploy` 目录。

    ```shell
    git clone -b feature/paddle-x https://github.com/PaddlePaddle/Paddle-Lite-Demo.git PaddleX-Lite-Deploy
    ```

2. 填写 [问卷](https://paddle.wjx.cn/vm/eaaBo0H.aspx#) 下载压缩包，将压缩包放到指定解压目录，切换到指定解压目录后执行解压命令。
      ```shell
      # 1. 切换到指定解压目录
      cd PaddleX-Lite-Deploy/ocr/android/shell/ppocr_demo

      # 2. 执行解压命令
      unzip ocr.zip
      ```

### 3.3 部署步骤

1. 将工作目录切换到 `PaddleX-Lite-Deploy/libs` 目录，运行 `download.sh` 脚本，下载需要的 Paddle Lite 预测库。此步骤只需执行一次，即可支持每个 demo 使用。

2. 将工作目录切换到 `PaddleX-Lite-Deploy/ocr/assets` 目录，运行 `download.sh` 脚本，下载 [paddle_lite_opt 工具](https://www.paddlepaddle.org.cn/lite/v2.10/user_guides/model_optimize_tool.html) 优化后的模型文件。

3. 将工作目录切换到 `PaddleX-Lite-Deploy/ocr/android/shell/cxx/ppocr_demo` 目录，运行 `build.sh` 脚本，完成可执行文件的编译。

4. 将工作目录切换到 `PaddleX-Lite-Deploy/ocr/android/shell/cxx/ppocr_demo`，运行 `run.sh` 脚本，完成在端侧的预测。

<b>注意事项：</b>
  - 在运行 `build.sh` 脚本前，需要更改 `NDK_ROOT` 指定的路径为实际安装的 NDK 路径。
  - 在 Windows 系统上可以使用 Git Bash 执行部署步骤。
  - 若在 Windows 系统上编译，需要将 `CMakeLists.txt` 中的 `CMAKE_SYSTEM_NAME` 设置为 `windows`。
  - 若在 Mac 系统上编译，需要将 `CMakeLists.txt` 中的 `CMAKE_SYSTEM_NAME` 设置为 `darwin`。
  - 在运行 `run.sh` 脚本时需保持 ADB 连接。
  - `download.sh` 和 `run.sh` 支持传入参数来指定模型，若不指定则默认使用 `PP-OCRv4_mobile` 模型。目前适配了 2 个模型：
    - `PP-OCRv3_mobile`
    - `PP-OCRv4_mobile`

以下为实际操作时的示例：
```shell
 # 1. 下载需要的 Paddle Lite 预测库
 cd PaddleX-Lite-Deploy/libs
 sh download.sh

 # 2. 下载 paddle_lite_opt 工具优化后的模型文件
 cd ../ocr/assets
 sh download.sh PP-OCRv4_mobile

 # 3. 完成可执行文件的编译
 cd ../android/shell/ppocr_demo
 sh build.sh

# 4. 预测
 sh run.sh PP-OCRv4_mobile
```

检测结果：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/deployment/05.png"  width="500" />

识别结果：

```text
The detection visualized image saved in ./test_img_result.jpg
0       纯臻营养护发素  0.993706
1       产品信息/参数   0.991224
2       （45元/每公斤，100公斤起订）    0.938893
3       每瓶22元，1000瓶起订）  0.988353
4       【品牌】：代加工方式/OEMODM     0.97557
5       【品名】：纯臻营养护发素        0.986914
6       ODMOEM  0.929891
7       【产品编号】：YM-X-3011 0.964156
8       【净含量】：220ml       0.976404
9       【适用人群】：适合所有肤质      0.987942
10      【主要成分】：鲸蜡硬脂醇、燕麦β-葡聚    0.968315
11      糖、椰油酰胺丙基甜菜碱、泛醒    0.941537
12      （成品包材）    0.974796
13      【主要功能】：可紧致头发磷层，从而达到  0.988799
14      即时持久改善头发光泽的效果，给干燥的头  0.989547
15      发足够的滋养    0.998413
```
