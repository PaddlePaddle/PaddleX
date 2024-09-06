# PaddleX 端侧部署 demo 使用指南

- [安装流程与使用方式](#安装流程与使用方式)
  - [环境准备](#环境准备)
  - [部署步骤](#部署步骤)
- [参考资料](#参考资料)

本指南主要介绍 PaddleX 端侧部署 demo 在 Android shell 上的运行方法。
本指南适用于下列 4 个任务的 8 个模型：

<table>
  <tr>
    <th>任务名</th>
    <th>模型名</th>
    <th>CPU</th>
    <th>GPU</th>
  </tr>
  <tr>
    <td>object_detection（目标检测）</td>
    <td>PicoDet-S<br/>PicoDet-L<br/>PicoDet_layout_1x</td>
    <td>✅<br/>✅<br/>✅</td>
    <td>✅<br/>✅<br/>✅</td>
  </tr>
  <tr>
    <td>semantic_segmentation（语义分割）</td>
    <td>PP-LiteSeg-T</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>image_classification（图像分类）</td>
    <td>PP-LCNet_x1_0<br/>MobileNetV3_small_x1_0</td>
    <td>✅<br/>✅</td>
    <td>✅<br/>✅</td>
  </tr>
  <tr>
    <td>ocr（文字识别）</td>
    <td>PP-OCRv4_mobile：<br/>PP-OCRv4_mobile_det，<br/>PP-OCRv4_mobile_rec</td>
    <td>✅</td>
    <td></td>
  </tr>
</table>

**备注**
- `GPU` 指的是 [使用 OpenCL 将计算映射到 GPU 上执行](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/opencl.html) ，以充分利用 GPU 硬件算力，提高推理性能。

## 安装流程与使用方式

### 环境准备

1. 在本地环境安装好 CMake 编译工具，并在 [Android NDK 官网](https://developer.android.google.cn/ndk/downloads)下载当前系统符合要求的版本的 NDK 软件包。例如，在 Mac 上开发，需要在 Android NDK 官网下载 Mac 平台的 NDK 软件包。

    **环境要求**
    -  `CMake >= 3.10`（最低版本未经验证，推荐 3.20 及以上）
    -  `Android NDK >= r17c`（最低版本未经验证，推荐 r20b 及以上）

    **本指南所使用的测试环境：**
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

### 部署步骤

克隆 `Paddle-Lite-Demo` 仓库的 `feature/paddle-x` 分支到 `PaddleX-Lite-Deploy` 目录。

```shell
git clone -b feature/paddle-x https://github.com/PaddlePaddle/Paddle-Lite-Demo.git PaddleX_Lite_Deploy
```

1. 将工作目录切换到 `PaddleX_Lite_Deploy/libs`，运行 `download.sh` 脚本，下载所需要的 Paddle Lite 预测库。此步骤只需执行一次，即可支持每个 Demo 使用。

2. 将工作目录切换到 `PaddleX_Lite_Deploy/{Task_Name}/assets`，运行 `download.sh` 脚本，下载 [paddle_lite_opt 工具](https://www.paddlepaddle.org.cn/lite/v2.10/user_guides/model_optimize_tool.html) 优化后的模型、测试图片和标签文件等。

3. 将工作目录切换到 `PaddleX_Lite_Deploy/{Task_Name}/android/shell/cxx/{Demo_Name}`，运行 `build.sh` 脚本，完成可执行文件的编译和运行。

4. 将工作目录切换到 `PaddleX-Lite-Deploy/{Task_Name}/android/shell/cxx/{Demo_Name}`，运行 `run.sh` 脚本，完成在端侧的预测。

    **注意：**
    - `Task_Name` 和 `Demo_Name` 为占位符，具体值可参考本节最后的表格。
    - `download.sh` 和 `run.sh` 支持传入模型名来指定模型，若不指定则使用默认模型。目前适配的模型可参考本节最后表格的 `Model_Name` 列。
    - 在运行 `build.sh` 脚本前，需要更改 `NDK_ROOT` 指定的路径为实际安装的 NDK 路径。
    - 在运行 `build.sh` 脚本时需保持 ADB 连接。
    - 若在 Mac 系统上编译，需要将 `CMakeLists.txt` 中的 `CMAKE_SYSTEM_NAME` 设置为 `darwin`。

以下为 object_detection 的示例，其他 demo 需按参考本节最后的表格改变第二步和第三步所切换的目录。

```shell
 # 1. 下载需要的 Paddle Lite 预测库
 cd PaddleX_Lite_Deploy/libs
 sh download.sh

 # 2. 下载 paddle_lite_opt 工具优化后的模型、测试图片、标签文件
 cd ../object_detection/assets
 sh download.sh
 # 支持传入模型名来指定下载的模型 支持的模型列表可参考本节最后表格的 Model_Name 列
 # sh download.sh PicoDet-L

 # 3. 完成可执行文件的编译
 cd ../android/app/shell/cxx/picodet_detection
 sh build.sh

 # 4. 预测
 sh run.sh
 # 支持传入模型名来指定预测的模型 支持的模型列表可参考本节最后表格的 Model_Name 列
 # sh run.sh PicoDet-L
```

运行结果如下所示,并生成一张名叫 `dog_picodet_detection_result.jpg` 的结果图：

```text
======= benchmark summary =======
input_shape(s) (NCHW): {1, 3, 320, 320}
model_dir:./models/PicoDet-S/model.nb
warmup:1
repeats:10
power_mode:1
thread_num:0
*** time info(ms) ***
1st_duration:320.086
max_duration:277.331
min_duration:272.67
avg_duration:274.91

====== output summary ======
detection, image size: 768, 576, detect object: bicycle, score: 0.905929, location: x=125, y=120, width=441, height=304
detection, image size: 768, 576, detect object: truck, score: 0.653789, location: x=465, y=72, width=230, height=98
detection, image size: 768, 576, detect object: dog, score: 0.731584, location: x=128, y=222, width=182, height=319
```

![预测结果](https://github.com/PaddlePaddle/Paddle-Lite-Demo/blob/feature/paddle-x/docs_img/object_detection/PicoDet-S.jpg?raw=true)

本节描述的部署步骤适用于下表中列举的 demo：

  <table>
    <tr>
      <th>Task_Name</th>
      <th>Demo_Name</th>
      <th>Model_Name</th>
    </tr>
    <tr>
      <td>object_detection</td>
      <td>picodet_detection</td>
      <td>PicoDet-S（default）<br/>PicoDet-L<br/>PicoDet_layout_1x<br/>PicoDet-S_gpu<br/>PicoDet-L_gpu<br/>PicoDet_layout_1x_gpu</td>
    </tr>
    <tr>
      <td>semantic_segmentation</td>
      <td>semantic_segmentation</td>
      <td>PP-LiteSeg-T（default）<br/>PP-LiteSeg-T_gpu</td>
    </tr>
    <tr>
      <td>image_classification</td>
      <td>image_classification</td>
      <td>PP-LCNet_x1_0（default）<br/>MobileNetV3_small_x1_0<br/>PP-LCNet_x1_0_gpu<br/>MobileNetV3_small_x1_0_gpu</td>
    </tr>
    <tr>
      <td>ocr</td>
      <td>ppocr_demo</td>
      <td>PP-OCRv4_mobile（default）：<br/>PP-OCRv4_mobile_det，<br/>PP-OCRv4_mobile_rec</td>PP-OCRv4_mobile_rec</td>
    </tr>
  </table>

## 参考资料
本指南仅介绍端侧部署 demo 的基本安装、使用流程，若想要了解更细致的信息，如代码介绍、代码讲解、更新模型、更新输入和输出预处理、更新预测库等，可参考下列文档：

- [object_detection（目标检测）](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/feature/paddle-x/object_detection/android/shell/cxx/picodet_detection)
- [semantic_segmentation（语义分割）](https://github.com/PaddlePaddle/Paddle-Lite-Demo/blob/feature/paddle-x/semantic_segmentation/android/shell/cxx/semantic_segmentation/README.md)
- [image_classification（图像分类）](https://github.com/PaddlePaddle/Paddle-Lite-Demo/blob/feature/paddle-x/image_classification/android/shell/cxx/image_classification/README.md)
- [ocr（文字识别）](https://github.com/PaddlePaddle/Paddle-Lite-Demo/blob/feature/paddle-x/ocr/android/shell/ppocr_demo/README.md)
