# PaddleX 端侧部署 demo 使用指南

- [安装流程与使用方式](#安装流程与使用方式)
  - [环境准备](#环境准备)
  - [部署步骤](#部署步骤)
- [参考资料](#参考资料)

本指南主要介绍 PaddleX 端侧部署 demo 在 Android shell 上的运行方法。
本指南适用于下列 7 个任务的 10 个模型：
- face_detection（人脸检测）
- face_keypoints_detection（人脸关键点检测）
- mask_detection（口罩检测）
- object_detection（目标检测）
- human_segmentation（人像分割）
- image_classification（图像分类）
- PP-shitu（PP识图）

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

1. 克隆 Paddle-Lite-Demo 仓库。

    ```shell
    git clone https://github.com/PaddlePaddle/Paddle-Lite-Demo.git
    ```

2. 将工作目录切换到 `Paddle-Lite-Demo/libs`，运行 `download.sh` 脚本，下载所需要的 Paddle Lite 预测库。此步骤只需执行一次，即可支持每个 Demo 使用。

3. 将工作目录切换到 `Paddle-Lite-Demo/{Demo Name}/assets`，运行 `download.sh` 脚本，下载 [paddle_lite_opt 工具](https://www.paddlepaddle.org.cn/lite/v2.10/user_guides/model_optimize_tool.html) 优化后的模型、测试图片和标签文件等。

4. 将工作目录切换到 `Paddle-Lite-Demo/{Demo Name}/android/shell/cxx/{Model Name}`，运行 `build.sh` 脚本，完成可执行文件的编译和运行。

    **注意：**
    - `Demo Name` 和 `Model Name` 为占位符，具体值可参考本节最后的表格。
    - 在运行 `build.sh` 脚本前，需要更改 `NDK_ROOT` 指定的路径为实际安装的 NDK 路径。
    - 在运行 `build.sh` 脚本时需保持 ADB 连接。
    - 若在 Mac 系统上编译，需要将 `CMakeLists.txt` 中的 `CMAKE_SYSTEM_NAME` 设置为 `darwin`。

以下为 face_detection 的示例，其他 demo 需按参考本节最后的表格改变第二步和第三步所切换的目录。

```shell
 # 1. 下载需要的 Paddle Lite 预测库
 cd Paddle-Lite-Demo/libs
 sh download.sh

 # 2. 下载 paddle_lite_opt 工具优化后的模型、测试图片、标签文件
 cd ../face_detection/assets
 sh download.sh

 # 3. 完成可执行文件的编译和运行
 cd ../android/app/shell/cxx/face_detection
 sh build.sh
```

运行结果如下所示,并生成一张名叫 `face_detection.jpg` 的人脸检测结果图：

```text
======= benchmark summary =======
input_shape(s) (NCHW): {1, 3, 240, 320}
model_dir:models/model.nb
warmup:2
repeats:10
power_mode:0
thread_num:1
*** time info(ms) ***
1st_duration:30.485
max_duration:23.106
min_duration:22.217
avg_duration:22.6233

====== output summary ======

```
本节描述的部署步骤适用于下表中列举的 demo：

  <table>
    <tr>
      <th>Demo Name</th>
      <th>Model Name</th>
    </tr>
    <tr>
      <td>face_detection</td>
      <td>face_detection</td>
    </tr>
    <tr>
      <td>face_keypoints_detection</td>
      <td>face_keypoints_detection</td>
    </tr>
    <tr>
      <td>mask_detection</td>
      <td>mask_detection</td>
    </tr>
    <tr>
      <td>object_detection</td>
      <td>picodet_detection<br/>ssd_mobilenetv1_detection<br/>yolov3_mobilenet_v3<br/>yolov5n_detection</td>
    </tr>
    <tr>
      <td>human_segmentation</td>
      <td>human_segmentation</td>
    </tr>
    <tr>
      <td>image_classification</td>
      <td>image_classification</td>
    </tr>
    <tr>
      <td>PP-shitu</td>
      <td>shitu</td>
    </tr>
  </table>

## 参考资料
本指南仅介绍端侧部署 demo 的基本安装、使用流程，若想要了解更细致的信息，如代码介绍、代码讲解、更新模型、更新输入和输出预处理、更新预测库等，可参考下列文档：

- [face_detection（人脸检测）](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/face_detection/android/shell/cxx/face_detection/README.md)
- [face_keypoints_detection（人脸关键点检测）](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/face_keypoints_detection/android/shell/cxx/face_keypoints_detection/README.md)
- [mask_detection（口罩检测）](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/mask_detection/android/shell/cxx/mask_detection/README.md)
- [object_detection（目标检测）](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/object_detection/android/shell/cxx/picodet_detection/README.md)
- [human_segmentation（人像分割）](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/human_segmentation/android/shell/cxx/human_segmentation/README.md)
- [image_classification（图像分类）](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/image_classification/android/shell/cxx/image_classification/README.md)
- [PP-shitu（PP识图）](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/PP_shitu/android/shell/cxx/shitu/README.md)
