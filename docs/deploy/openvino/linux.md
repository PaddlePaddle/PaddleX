# Linux platform


## Pre-conditions

* OS: Ubuntu, Raspbian OS
* GCC* 5.4.0
* CMake 3.0+
* PaddleX 1.0+
* OpenVINO 2021.1+
* Hardware platform: CPU, VPU

**Note**: For PaddleX installation, see [PaddleX](https://paddlex.readthedocs.io/zh_CN/develop/install.html) description. For the installation of OpenVINO, see [OpenVINO-Linux](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html) or [OpenVINO-Raspbian](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_raspbian.html) description according to the corresponding systems.

Make sure that the above basic software is installed on your system and that you have configured your environment accordingly. **The following examples are based on the `/root/projects/` directory**.



## Inference deployment

This document provides prediction deployment methods under c++. To perform prediction deployment under python, see [python prediction deployment](./python.md).

### Step1 Download the PaddleX prediction code.
```
mkdir -p /root/projects
cd /root/projects
git clone https://github.com/PaddlePaddle/PaddleX.git
```
**Note**: The C++ prediction code is in PaddleX/deploy/openvino. The directory does not depend on any other directory in PaddleX.

### Step2 Software dependencies

For the compiled scripts in Step3, the pre-compiled package of a third-party dependent software is installed by pressing one key. Users do not need to download or compile these dependent software separately. If you need to compile a third-party dependency software yourself, refer to:

- gflags: For compiling, refer to [Compiling Documents] (https://gflags.github.io/gflags/#download).

- opencv: For compiling, refer to [Compiling Documents] (https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html).


### Step3: Compile
The command to compile `cmake` is in `scripts/build.sh`. If the compiling is performed in Raspberry (Raspbian OS), modify the ARCH parameters x86 to armv7. If you compile your own third-party dependency software, modify the main parameters as required, according to the software compiled in Step 1. The main content is described as follows:
```
# Path of the openvino pre-compiling library
OPENVINO_DIR=$INTEL_OPENVINO_DIR/inference_engine
# Path to the gflags pre-compiling library
GFLAGS_DIR=$(pwd)/deps/gflags
# Path to the ngraph lib pre-compiling library
NGRAPH_LIB=$INTEL_OPENVINO_DIR/deployment_tools/ngraph/lib
# Path to the opencv pre-compiling library 
OPENCV_DIR=$(pwd)/deps/opencv/
#cpu architecture (x86 or armv7)
ARCH=x86
```
Run the `build` script:
```shell
sh . /scripts/build.sh
```

### Step4: Prediction

After successful compilation, the prediction executable program for the classification task is `classifier`, the prediction executable program for the detection task is `detector`, and the prediction executable program for the segmentation task is `segmenter`. The main command parameters are described as follows:

| Parameters | Description |
|  ----  | ----  |
| --model_dir | The path of the .xml file generated in the model conversion. Make sure that the three files generated in the model conversion are in the same path. |
| --image | The path of the image file to be predicted |
| --image_list | .txt file of storing image paths by line |
| --device | Running platform. Options are {"CPU", "MYRIAD"}, and the default value is "CPU". For VPU, use "MYRIAD". |
| --cfg_file | .yml configuration file of PaddleX model. |
| --save_dir | Storage address of visualization result images. It is only for inspection tasks. The default value is " ". That is, the visualization result is not saved. |

### Example
`Example 1`:
Classification task prediction for a single image under CPU in Linux
Test image: `/path/to/test_img.jpeg`

```shell
. /build/classifier --model_dir=/path/to/openvino_model --image=/path/to/test_img.jpeg --cfg_file=/path/to/PadlleX_model.yml
```


`Example 2`:
The Linux system performs multiple image detection task predictions under the CPU, and saves the prediction visualization results
Predicted multiple images `/path/to/image_list.txt`, image_list.txt content in the following format.
```
/path/to/images/test_img1.jpeg
/path/to/images/test_img2.jpeg
...
/path/to/images/test_imgn.jpeg
```

```shell
./build/detector --model_dir=/path/to/models/openvino_model --image_list=/root/projects/images_list.txt --cfg_file=/path/to/PadlleX_model.yml --save_dir ./output
```

`Example 3`:
Raspbian OS Single image classification task prediction under VPU
Test image: `/path/to/test_img.jpeg`

```shell
. /build/classifier --model_dir=/path/to/openvino_model --image=/path/to/test_img.jpeg --cfg_file=/path/to/PadlleX_model.yml --device=MYRIAD
```

## Performance Test
`Test 1`:
The performance of OpenVINO acceleration on PaddleX deployments was tested at the server CPU.
- CPU: Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
- OpenVINO: 2020.4
- PaddleX: using Paddle prediction library (1.8), to enable the mkldnn acceleration and start the multithreading.
- The model is from PaddleX tutorials, the Batch Size is 1, the time consumption unit is ms/image. Only the model running time is calculated, not including the pre-processing and post-processing of the data, 20 images warmup, 100 images for testing the performance.

| Model | PaddleX | OpenVINO | Image Input Size |
|---|---|---|---|
|resnet-50 | 20.56 | 16.12 | 224*224 |
|mobilenet-V2 | 5.16 | 2.31 |224*224|
|yolov3-mobilnetv1 |76.63| 46.26|608*608 |
|unet| 276.40| 211.49| 512*512|  


`Test 2`:
Inserting a VPU architecture Neural Compute Stick (NCS2) into a PC to accelerate through Openvino.
- CPU：Intel(R) Core(TM) i5-4300U 1.90GHz
- VPU：Movidius Neural Compute Stick2
- OpenVINO： 2020.4
- The model is from PaddleX tutorials, the Batch Size is 1, the time consumption unit is ms/image. Only the model running time is calculated, not including the pre-processing and post-processing of the data, 20 images warmup, 100 images for testing the performance.

| Model | OpenVINO | Enter a Picture |
|---|---|---|
|mobilenetV2|24.00|224*224|
|resnet50_vd_ssld|58.53|224*224|  

`Test 3`:
Inserting a VPU architecture neural computation stick (NCS2) on the Raspberry 3B to accelerate through Openvino.
- CPU ：ARM Cortex-A72 1.2GHz 64bit
- VPU：Movidius Neural Compute Stick2
- OpenVINO 2020.4
- The model is from PaddleX tutorials, the Batch Size is 1, the time consumption unit is ms/image. Only the model running time is calculated, not including the pre-processing and post-processing of the data, 20 images warmup, 100 images for testing the performance.

| Model | OpenVINO | Input Image Size |
|---|---|---|
|mobilenetV2|43.15|224*224|
|resnet50|82.66|224*224|  
