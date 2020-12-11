# Raspberry
PaddleX supports the prediction deployment on the raspberry through both Paddle-Lite and the OpenVINO-based Neural Compute Stick (NCS2).


## Hardware environment configuration

For a Raspberry without installing the system yet, you need to perform the system installation and environment configuration to initialize the hardware environment. The required software and hardware are as follows:

- Hardware: micro SD, monitor, keyboard, mouse
- Software: Raspbian OS
### Step1: System installation
- Format micro SD card as FAT. In the Windows and Mac systems, the [SD Memory Card Formatter](https://www.sdcard.org/downloads/formatter/) tool is recommended. In the Linux system, refer to [NOOBS For Raspberry Pi.](http://qdosmsq.dunbar-it.co.uk/blog/2013/06/noobs-for-raspberry-pi/)
- Download the NOOBS Raspbian OS [download link] (https://www.raspberrypi.org/downloads/). Copy the decompressed file to SD. After the SD is inserted, the Raspberry is powered on. The system is installed automatically.
### Step2: Environment configuration
- Start the VNC and SSH services: start the LX Terminal. Enter the following command, and select Interfacing Option. Then, select P2 SSH and P3 VNC to start the SSH and VNC respectively. After the startup, the Raspberry is connected through SSH or VNC.
```
sudo raspi-config
```
- Replace source: The official Raspberry source is very slow; therefore, it is recommended to check the official website of the domestic source[ Raspberry software](https://www.jianshu.com/p/67b9e6ebf8a0). After the replacement, run the following:
```
sudo apt-get update
sudo apt-get upgrade
```

## Paddle-Lite deployment
The Paddle-Lite-based deployment currently support PaddleX classification, segmentation and detection models. For the detection model, only YOLOV3 is supported.

Deployment process include: PaddleX model conversion and post-conversion model deployment

**Note**: For the PaddleX installation, refer to [PaddleX](https://paddlex.readthedocs.io/zh_CN/develop/install.html). For the details of the Paddle-Lite, refer to [Paddle-Lite](https://paddle-lite.readthedocs.io/zh/latest/index.html).

Make sure that the above basic software is installed on your system and that you have configured your environment accordingly. **The following examples are based on the `/root/projects/` directory**.

## Paddle-Lite model conversion
Convert the PaddleX model to Paddle-Lite model. For details, see [Paddle-Lite Model Conversions](./export_nb_model.md).

## Paddle-Lite predication
### Step1 Download the PaddleX prediction code.
```
mkdir -p /root/projects
cd /root/projects
git clone https://github.com/PaddlePaddle/PaddleX.git
```
**Note**: The C++ prediction code is in PaddleX/deploy/raspberry directory. This directory does not depend on any other directory under PaddleX. For the prediction deployment in the Python, refer to [Python prediction deployment] (./python.md).

### Step2: Download the Paddle-Lite pre-compiling library.
Provide the Paddle-Lite pre-compiling library under ArmLinux corresponding to the downloaded opt tool: [Paddle-Lite (ArmLinux) pre-compiling library] (https://bj.bcebos.com/paddlex/deploy/lite/inference_lite_2.6.1_armlinux.tar.bz2).
The pre-compiling library is recommended. If you compile it yourself, enter the following command in the LX terminal on the Raspberry.
```
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
sudo ./lite/tools/build.sh  --arm_os=armlinux --arm_abi=armv7hf --arm_lang=gcc  --build_extra=ON full_publish
```

Path of pre-compiling library: `/build.lite.armlinux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf/cxx`

**Note**: The prediction library version needs to be the same as the opt version. For more Paddle-Lite compiling contents, refer to [Paddle-Lite Compilaing](https://paddle-lite.readthedocs.io/zh/latest/user_guides/source_compile.html). For more pre-compiling Paddle-Lite prediction library, refer to [Paddle-Lite Release Note](https://github.com/PaddlePaddle/Paddle-Lite/releases).

### Step3 Software dependencies
Pre-compiling packages or one-key compilation of dependent software are provided. Users do not need to separately download or compile a third party dependent software. If you need to compile a third-party dependency software yourself, refer to:

- gflags: For compiling, refer to the [Compiling Documents](https://gflags.github.io/gflags/#download).

- opencv: For compiling, refer to [Compiling Documents] (https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html).
### Step4: Compile
Compile `cmake` in `scripts/build. sh`. Modify LITE_DIR to Paddle-Lite prediction library directory. If you compile a third-party dependency software, modify the main parameters as required in Step 1. The main content is described as follows:
```
# Path to the Paddle-Lite pre-compiling library
LITE_DIR=/path/to/Paddle-Lite/inference/lib
# Path to the gflags pre-compiling library
GFLAGS_DIR=$(pwd)/deps/gflags
# Path to the opencv pre-compiling library
OPENCV_DIR=$(pwd)/deps/opencv/
```
Run the `build` script:
```shell
sh . /scripts/build.sh
```


### Step3: Prediction

After successful compilation, the prediction executable program for the classification task is `classifier`, the prediction executable for the segmentation task is `segmenter`, and the prediction executable program for the detection task is `detector`. The main command parameters are as follows:

| Parameters | Description |
|  ----  | ----  |
| --model_dir | The path of the .xml file generated in the model conversion. Make sure that the three files generated in the model conversion are in the same path. |
| --image | The path of the image file to be predicted |
| --image_list | .txt file of storing image paths by line |
| --thread_num | Number of predicated threads, the default value is 1 |
| --cfg_file | .yml configuration file of PaddleX model. |
| --save_dir | Visualization results storage image. It is applicable to only detection and segmentation tasks. The default value is " ", that is, visualization results are not saved. |

### Example
`Example 1`:
Single image classification task
Test image `/path/to/test_img.jpeg`

```shell
./build/classifier --model_dir=/path/to/nb_model
--image=/path/to/test_img.jpeg --cfg_file=/path/to/PadlleX_model.yml  --thread_num=4
```


`Example 2`:
Multi-image segmentation task
Prediction of multiple images: `/path/to/image_list.txt`. The format of the image_list.txt content is as follows:
```
/path/to/images/test_img1.jpeg
/path/to/images/test_img2.jpeg
...
/path/to/images/test_imgn.jpeg
```

```shell
./build/segmenter --model_dir=/path/to/models/nb_model --image_list=/root/projects/images_list.txt --cfg_file=/path/to/PadlleX_model.yml --save_dir ./output --thread_num=4  
```

## Performance test
### Test environment:
Hardware: Raspberry Pi 3 Model B
System: raspbian OS
Software: paddle-lite 2.6.1
### Test results
Unit: ms. The num parameter indicates the number of threads used under paddle-lite.

| Model | lite(num=4) | Input Image Size |
| ----|  ---- | ----|
|mobilenet-v2|136.19|224*224|
|resnet-50|1131.42|224*224|
|deeplabv3|2162.03|512*512|
|hrnet|6118.23|512*512|
|yolov3-darknet53|4741.15|320*320|
|yolov3-mobilenet|1424.01|320*320|
|densenet121|1144.92|224*224|
|densenet161|2751.57|224*224|
|densenet201|1847.06|224*224|
|HRNet_W18|1753.06|224*224|
|MobileNetV1|177.63|224*224|
|MobileNetV3_large_ssld|133.99|224*224|
|MobileNetV3_small_ssld|53.99|224*224|
|ResNet101|2290.56|224*224|
|ResNet101_vd|2337.51|224*224|
|ResNet101_vd_ssld|3124.49|224*224|
|ShuffleNetV2|115.97|224*224|
|Xception41|1418.29|224*224|
|Xception65|2094.7|224*224|


From the test results, it is recommended that users use MobileNetV1-V3 and ShuffleNetV2, and other small networks on Raspberry.

## NCS2 deployment
Raspberry supports the running of PaddleX model prediction on NCS2 through OpenVINO. Currently, only PaddleX classification network is supported. The NCS2-based method includes two steps: Paddle model converted to OpenVINO IR and deployment of IR on NCS2 for prediction.
- For model conversion, refer to: [PaddleX model converted to OpenVINO IR]('./openvino/export_openvino_model.md'). OpenVINO on raspbian OS does not support model conversion. You need to convert FP16 IR on the host side first.
- For the prediction deployment, refer to the VPU deployment in raspbian OS in [OpenVINO deployment] (./openvino/linux.md).
