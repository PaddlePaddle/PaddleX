# PaddleX Edge Deployment Demo Usage Guide

- [Installation Process and Usage](#installation-process-and-usage)
  - [Environment Preparation](#environment-preparation)
  - [Material Preparation](#material-preparation)
  - [Deployment Steps](#deployment-steps)
- [Reference Materials](#reference-materials)
- [Feedback Section](#feedback-section)

This guide mainly introduces the operation method of the PaddleX edge deployment demo on the Android shell.
This guide applies to 8 models across 6 modules:

<table>
  <tr>
    <th>Module</th>
    <th>Specific Model</th>
    <th>CPU</th>
    <th>GPU</th>
  </tr>
  <tr>
    <td rowspan="2">Object Detection</td>
    <td>PicoDet-S</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>PicoDet-L</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>Layout Area Detection</td>
    <td>PicoDet_layout_1x</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>Semantic Segmentation</td>
    <td>PP-LiteSeg-T</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="2">Image Classification</td>
    <td>PP-LCNet_x1_0</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x1_0</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>Text Detection</td>
    <td>PP-OCRv4_mobile_det</td>
    <td>✅</td>
    <td></td>
  </tr>
  <tr>
    <td>Text Recognition</td>
    <td>PP-OCRv4_mobile_rec</td>
    <td>✅</td>
    <td></td>
  </tr>
</table>

**Note**
- `GPU` refers to [mapping computations to GPU execution using OpenCL](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/opencl.html) to fully utilize GPU hardware computing power and improve inference performance.

## Installation Process and Usage

### Environment Preparation

1. Install CMake build tool locally and download the required version of NDK software package from the [Android NDK official website](https://developer.android.google.cn/ndk/downloads?hl=en). For example, if developing on a Mac, download the NDK software package for the Mac platform from the Android NDK official website.

    **Environment Requirements**
    - `CMake >= 3.10` (Minimum version not verified, recommend 3.20 and above)
    - `Android NDK >= r17c` (Minimum version not verified, recommend r20b and above)

    **Tested Environment Used in This Guide**:
    - `cmake == 3.20.0`
    - `android-ndk == r20b`

2. Prepare an Android phone and enable USB debugging mode. Enable method: `Phone Settings -> Locate Developer Options -> Turn on Developer Options and USB Debugging Mode`.

3. Install ADB tool on your computer for debugging. ADB installation methods:

    3.1. For Mac:

    ```shell
    brew cask install android-platform-tools
    ```

    3.2. For Linux:

    ```shell
    # Debian-based Linux distributions
    sudo apt update
    sudo apt install -y wget adb

    # Red Hat-based Linux distributions
    sudo yum install adb
    ```

    3.3. For Windows:

    Install ADB by downloading the ADB software package from Google's Android platform: [Link](https://developer.android.com/studio?hl=en)

    Open a terminal, connect your phone to the computer, and enter in the terminal:

    ```shell
     adb devices
    ```

    If there is an output from the device, it indicates that the installation was successful.

    ```shell
     List of devices attached
     744be294    device
    ```

### Material Preparation

1. Clone the `feature/paddle-x` branch of the `Paddle-Lite-Demo` repository into the `PaddleX-Lite-Deploy` directory.

    ```shell
    git clone -b feature/paddle-x https://github.com/PaddlePaddle/Paddle-Lite-Demo.git PaddleX-Lite-Deploy
    ```

2. Fill out the **survey** to download the compressed package, place the compressed package in the specified unzip directory, switch to the specified unzip directory, and execute the unzip command.
    - [Object Detection Survey](https://paddle.wjx.cn/vm/OjV8gAb.aspx#)
    - [Semantic Segmentation Survey](https://paddle.wjx.cn/vm/Q2F1L37.aspx#)
    - [Image Classification Survey](https://paddle.wjx.cn/vm/rWPncBm.aspx#)
    - [OCR Survey](https://paddle.wjx.cn/vm/eaaBo0H.aspx#)

    Below is an example of the unzip operation for object_detection. Refer to the table below for other pipelines.

      ```shell
      # 1. Switch to the specified unzip directory
      cd PaddleX-Lite-Deploy/object_detection/android/shell/cxx/picodet_detection

      # 2. Execute the unzip command
      unzip object_detection.zip
      ```

      <table>
        <tr>
          <th>Pipeline Name</th>
          <th>Unzip Directory</th>
          <th>Unzip Command</th>
        </tr>
        <tr>
          <td>Object Detection</td>
          <td>PaddleX-Lite-Deploy/object_detection/android/shell/cxx/picodet_detection</td>
          <td>unzip object_detection.zip</td>
        </tr>
        <tr>
          <td>Semantic Segmentation</td>
          <td>PaddleX-Lite-Deploy/semantic_segmentation/android/shell/cxx/semantic_segmentation</td>
          <td>unzip semantic_segmentation.zip</td>
        </tr>
        <tr>
          <td>Image Classification</td>
          <td>PaddleX-Lite-Deploy/image_classification/android/shell/cxx/image_classification</td>
          <td>unzip image_classification.zip</td>
        </tr>
        <tr>
          <td>OCR</td>
          <td>PaddleX-Lite-Deploy/ocr/android/shell/ppocr_demo</td>
          <td>unzip ocr.zip</td>
        </tr>
      </table>

### Deployment Steps

1. Switch the working directory to `PaddleX_Lite_Deploy/libs` and run the `download.sh` script to download the necessary Paddle Lite prediction library. This step only needs to be executed once to support each demo.

2. Switch the working directory to `PaddleX_Lite_Deploy/{Task_Name}/assets`, run the `download.sh` script to download the [paddle_lite_opt tool](https://www.paddlepaddle.org.cn/lite/v2.10/user_guides/model_optimize_tool.html) optimized model, test images, label files, etc.

3. Switch the working directory to `PaddleX_Lite_Deploy/{Task_Name}/android/shell/cxx/{Demo_Name}`, run the `build.sh` script to complete the compilation and execution of the executable file.

4. Switch the working directory to `PaddleX-Lite-Deploy/{Task_Name}/android/shell/cxx/{Demo_Name}`, run the `run.sh` script to complete the prediction on the edge side.

    **Note**:
    - `{Pipeline_Name}` and `{Demo_Name}` are placeholders. Refer to the table at the end of this section for specific values.
    - `download.sh` and `run.sh` support passing in model names to specify models. If not specified, the default model will be used. Refer to the `Model_Name` column in the table at the end of this section for currently supported models.
    - To use your own trained model, refer to the [Model Conversion Method](https://paddlepaddle.github.io/Paddle-Lite/develop/model_optimize_tool/) to obtain the `.nb` model, place it in the `PaddleX_Lite_Deploy/{Pipeline_Name}/assets/{Model_Name}` directory, where `{Model_Name}` is the model name, e.g., `PaddleX_Lite_Deploy/object_detection/assets/PicoDet-L`.
    - Before running the `build.sh` script, change the path specified by `NDK_ROOT` to the actual installed NDK path.
    - Keep ADB connected when running the `build.sh` script.
    - On Windows systems, you can use Git Bash to execute the deployment steps.
    - If compiling on a Windows system, set `CMAKE_SYSTEM_NAME` to `windows` in `CMakeLists.txt`.
    - If compiling on a Mac system, set `CMAKE_SYSTEM_NAME` to `darwin` in `CMakeLists.txt`.

Below is an example for object_detection. For other demos, change the directories switched in steps 2 and 3 according to the table at the end of this section.

```shell
# 1. Download the necessary Paddle Lite prediction library
cd PaddleX_Lite_Deploy/libs
sh download.sh

# 2. Download the paddle_lite_opt tool optimized model, test images, and label files
cd ../object_detection/assets
sh download.sh
# Supports passing in model names to specify the downloaded model. Refer to the Model_Name column in the table at the end of this section for supported models.
# sh download.sh PicoDet-L

# 3. Complete the compilation of the executable file
cd ../android/app/shell/cxx/picodet_detection
sh build.sh

# 4. Prediction
sh run.sh
# Supports passing in model names to specify the prediction model. Refer to the Model_Name column in the table at the end of this section for supported models.
# sh run.sh PicoDet-L
```

The run results are shown below, and a result image named `dog_picodet_detection_result.jpg` is generated:

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
detection, image size: 768, 576, detect object: bicycle, score: 0.905929, location: x=125, y=1
```

![result](https://github.com/PaddlePaddle/Paddle-Lite-Demo/blob/feature/paddle-x/docs_img/object_detection/PicoDet-S.jpg?raw=true)

This section describes the deployment steps applicable to the demos listed in the following table:

<table>
  <tr>
    <th>Pipeline</th>
    <th>Pipeline_Name</th>
    <th>Module</th>
    <th>Demo_Name</th>
    <th>Specific Model</th>
    <th>Model_Name</th>
  </tr>
  <tr>
    <td rowspan="3">General Object Detection</td>
    <td rowspan="3">object_detection</td>
    <td rowspan="3">Object Detection</td>
    <td rowspan="3">picodet_detection</td>
    <td>PicoDet-S</td>
    <td>PicoDet-S（default）</br>PicoDet-S_gpu</td>
  </tr>
  <tr>
    <td>PicoDet-L</td>
    <td>PicoDet-L</br>PicoDet-L_gpu</td>
  </tr>
  <tr>
    <td>PicoDet_layout_1x</td>
    <td>PicoDet_layout_1x</br>PicoDet_layout_1x_gpu</td>
  </tr>
  <tr>
    <td>General Semantic Segmentation</td>
    <td>semantic_segmentation</td>
    <td>Semantic Segmentation</td>
    <td>semantic_segmentation</td>
    <td>PP-LiteSeg-T</td>
    <td>PP-LiteSeg-T（default）</br>PP-LiteSeg-T_gpu</td>
  </tr>
  <tr>
    <td rowspan="2">General Image Classification</td>
    <td rowspan="2">image_classification</td>
    <td rowspan="2">Image Classification</td>
    <td rowspan="2">image_classification</td>
    <td>PP-LCNet_x1_0</td>
    <td>PP-LCNet_x1_0（default）</br>PP-LCNet_x1_0_gpu</td>
  </tr>
  <tr>
    <td>MobileNetV3_small_x1_0</td>
    <td>MobileNetV3_small_x1_0</br>MobileNetV3_small_x1_0_gpu</td>
  </tr>
  <tr>
    <td rowspan="2">General OCR</td>
    <td rowspan="2">ocr</td>
    <td>Text Detection</td>
    <td rowspan="2">ppocr_demo</td>
    <td>PP-OCRv4_mobile_det</td>
    <td>PP-OCRv4_mobile_det</td>
  </tr>
  <tr>
    <td>Text Recognition</td>
    <td>PP-OCRv4_mobile_rec</td>
    <td>PP-OCRv4_mobile_rec</td>
  </tr>
</table>

**Note**
- Currently, there is no demo for deploying the Layout Area Detection module on the edge side, so the `picodet_detection` demo is reused to deploy the `PicoDet_layout_1x` model.

## Reference Materials

This guide only introduces the basic installation and usage process of the edge-side deployment demo. If you want to learn more detailed information, such as code introduction, code explanation, updating models, updating input and output preprocessing, updating prediction libraries, etc., please refer to the following documents:

- [Object Detection](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/feature/paddle-x/object_detection/android/shell/cxx/picodet_detection)
- [Semantic Segmentation](https://github.com/PaddlePaddle/Paddle-Lite-Demo/blob/feature/paddle-x/semantic_segmentation/android/shell/cxx/semantic_segmentation/README.md)
- [Image Classification](https://github.com/PaddlePaddle/Paddle-Lite-Demo/blob/feature/paddle-x/image_classification/android/shell/cxx/image_classification/README.md)
- [OCR](https://github.com/PaddlePaddle/Paddle-Lite-Demo/blob/feature/paddle-x/ocr/android/shell/ppocr_demo/README.md)

## Feedback Section

The edge-side deployment capabilities are continuously optimized. Welcome to submit [issue](https://github.com/PaddlePaddle/PaddleX/issues/new/choose) to report problems and needs, and we will follow up promptly.
