# Nvidia Jetson Development Board

## Description
This document describes the test with `GCC 7.4` on the `Linux` platform based on Nvidia Jetpack 4.4. If you want to use a different G++ version, you need to recompile the Paddle prediction library. For details, see the compiling of [NVIDIA Jetson embedded hardware prediction library source codes] (https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html).

## Pre-conditions
* G++ 7.4
* CUDA 10.0 / CUDNN 8 (required only if using the prediction library in GPU version)
* CMake 3.0+

Make sure that the above basic software is installed on your system. **All the following examples are in the working directory `/root/projects/` **.

### Step1: Download the code

`git clone https://github.com/PaddlePaddle/PaddleX.git `

**Note**: The `C++` prediction code is in `PaddleX`/deploy/cpp`, which does not depend on any other directories in `PaddleX`.


### Step2: Download PaddlePaddle C++ prediction library: paddle_inference.

PaddlePaddle currently provides a C++ prediction library for Nvidia Jetson based on version 1.6.2.

| Release Notes | Prediction Library (version 1.6.2) |
|  ----  | ----  |
| nv-jetson-cuda10-cudnn7.5-trt5 | [paddle_inference](https://paddle-inference-lib.bj.bcebos.com/1.7.1-nv-jetson-cuda10-cudnn7.5-trt5/fluid_inference.tar.gz) |

The directory `/root/projects/fluid_inference` after downloading and decompression contains the following contents:
```
fluid_inference
├── paddle # paddle core library and header files
|
├──third_party # third-party dependency library and header files
|
└── version.txt # version and compilation information
```

### Step3: Compile

The command to compile `cmake` is in `scripts/jetson_build.sh`. Modify the main parameters as required. Its main content is described as follows:
```
# Whether GPU is used (i.e., whether CUDA is used)
WITH_GPU=OFF
# Use MKL or openblas
WITH_MKL=OFF
# Whether or not to integrate TensorRT (only WITH_GPU=ON available)
WITH_TENSORRT=OFF
# TensorRT path. If you need to integrate TensorRT, change the path to the actual TensorRT you installed.
TENSORRT_DIR=/root/projects/TensorRT/
# Paddle prediction library path. Change the path to the actual prediction library you installed.
PADDLE_DIR=/root/projects/fluid_inference
# Whether or not Paddle's prediction library is compiled using a static library
# When TensorRT is used, Paddle's prediction library is usually a dynamic library
WITH_STATIC_LIB=OFF
# CUDA's lib path
CUDA_LIB=/usr/local/cuda/lib64
# CUDNN's lib path
CUDNN_LIB=/usr/local/cuda/lib64

# You should not modify the following:
rm -rf build
mkdir -p build
cd build
cmake . . \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_MKL=${WITH_MKL} \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DWITH_ENCRYPTION=${WITH_ENCRYPTION} \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DWITH_STATIC_LIB=${WITH_STATIC_LIB} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB}
make
```
**Note: **When compiling in Linux environment, YAML is automatically downloaded. If the Internet is not available in the compiling environment, you can run the following to download manually:

- [yaml-cpp.zip ](https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip)

After downloading the yaml-cpp.zip file, you don't need to decompress it. In cmake/yaml.cmake, change the website in the `URL https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip` to the path of the downloading file.

After setting the main parameters of the modified script, run the `build` script.
```shell
sh . /scripts/jetson_build.sh
```

### Step4: Prediction and visualization

**Before loading the model, make sure that the files in your model directory should include `model.yml`, `__model__`, and `__params__`. If this condition is not met, refer to the [Model Export to Inference document](export_model.md) to export your model to the deployment format.**

* After successful compilation, the executable programs for the image prediction demo are `build/demo/detector`, `build/demo/classifier`, and `build/demo/segmenter`. Users can choose according to their model type. The main command parameters are as follows:

| Parameters | Description |
|  ----  | ----  |
| model_dir | The path of the exported prediction model |
| image | The path of the image file to be predicted |
| image_list | .txt file of storing image paths by line |
| use_gpu | Whether to use GPU prediction (value is 0 (default) or 1) |
| use_trt | Whether to use TensorRT for prediction, the value is 0 or 1 (the default value is 0) |
| gpu_id | GPU device ID (default value is 0) |
| save_dir | The path to save the visualization result, default is "output", classfier has no such parameter.**** |
| batch_size | Prediction batch size, default is 1 |
| thread_num | Number of predicted threads. By default, it is the number of CPU processors |

* After successful compilation, the executable programs of the video prediction demo are `build/demo/video_detector`, `build/demo/video_classifier`, and `build/demo/video_segmenter`. Users can choose according to the model type. The main command parameters are as follows:

| Parameters | Description |
|  ----  | ----  |
| model_dir | The path of the exported prediction model |
| use_camera | Whether to use the camera for prediction, the value is 0 or 1 (default value is 0) |
| camera_id | Camera device ID (default value is 0) |
| video_path | Path of video file |
| use_gpu | Whether to use GPU prediction (value is 0 (default) or 1) |
| use_trt | Whether to use TensorRT for prediction, the value is 0 or 1 (the default value is 0) |
| gpu_id | GPU device ID (default value is 0) |
| show_result | Whether or not to display the prediction visualization result in real time on the screen when making prediction on the video file (the result does not reflect the real frame rate because the delay process is added), the supported value is 0 or 1 (the default value is 0). |
| save_result | Whether to save the predicted visual result of each frame as a video file, the value is 0 or 1 (default value is 1) |
| save_dir | Path to save the visualization results (default value is "output") |

**Note: If the GUI is unavailable in the system, you should not set show_result to 1. When using a camera for prediction, press `ESC` to disable the camera and launch the prediction program.**


## Example

Predictions can be made using the [inference_model](export_model.md) and test images exported from the DUDU recognition model, to export to /root/projects. The model path is /root/projects/inference_model.``

`Example 1`:

Not using `GPU` test images: `/root/projects/images/xiaoduxiong.jpeg`

```shell
. /build/demo/detector --model_dir=/root/projects/inference_model --image=/root/projects/images/xiaoduxiong.jpeg --save_dir=output
```
The image file `visual predictions` are saved in the directory where the `save_dir` parameter is set.


`Example 2`:

Using the `GPU` to predict multiple images `/root/projects/image_list.txt`. The content of image_list.txt is in the following format:
```
/root/projects/images/xiaoduxiong1.jpeg
/root/projects/images/xiaoduxiong2.jpeg
...
/root/projects/images/xiaoduxiongn.jpeg
```
```shell
. /build/demo/detector --model_dir=/root/projects/inference_model --image_list=/root/projects/images_list. txt --use_gpu=1 --save_dir=output --batch_size=2 --thread_num=2
```
The image file `visual predictions` are saved in the directory where the `save_dir` parameter is set.

**Example 3: **

Using the camera prediction:

```shell
. /build/demo/video_detector --model_dir=/root/projects/inference_model --use_camera=1 --use_gpu=1 --save_dir=output --save_result=1
```
When `save_result` is set to 1, the `visual predictions` are saved in the directory where the `save_dir` parameter is set in the video file format.``

**Example 4: **

Predicting the video file:

```shell
./build/demo/video_detector --model_dir=/root/projects/inference_model --video_path=/path/to/video_file --use_gpu=1 --save_dir=output --show_result=1 --save_result=1
```
When `save_result` is set to 1, the `visual predictions` are saved in the directory where the `save_dir` parameter is set in the video file format. If the GUI is available in the system, view the visual prediciton results on the screen by setting `show_result` to 1.
