# Linux platform deployment

## Description
This document is tested with `GCC 4.8.5` and `GCC 4.9.4` in the `Linux`. To compile it with a later G++ version, you need to recompile the Paddle Prediction Library. Refer to [Compiling Paddle Prediction Library](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html#id12) from source code.

## Pre-conditions
* G++ 4.8.2 ~ 4.9.4
* CUDA 9.0 / CUDA 10.0, CUDNN 7+ (required only if using GPU version prediction library)
* CMake 3.0+

Make sure that the above basic software is installed on your system. **All the following examples are in the working directory `/root/projects/`**.

### Step1: Download the code.

`git clone https://github.com/PaddlePaddle/PaddleX.git `

**Note**: The C++ prediction code is in `/root/projects/`PaddleX`/deploy/cpp` directory. This directory does not depend on any other directory under `PaddleX`.


### Step2: Download PaddlePaddle C++ Prediction Library: paddle_inference.

The PaddlePaddle C++ prediction library provides different pre-compiled versions for different `CPUs`, `CUDAs`, and whether to support TensorRT. At present, PaddleX depends on the Paddle 1.8.4 version. The following provides a number of different versions of Paddle prediction libraries:

| Release Notes | Prediction Library (Version 1.8.4) |
|  ----  | ----  |
| ubuntu14.04_cpu_avx_mkl  | [paddle_inference](https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-mkl/fluid_inference.tgz) |
| ubuntu14.04_cpu_avx_openblas  | [paddle_inference](https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-openblas/fluid_inference.tgz) |
| ubuntu14.04_cpu_noavx_openblas  | [paddle_inference](https://paddle-inference-lib.bj.bcebos.com/latest-cpu-noavx-openblas/fluid_inference.tgz) |
| ubuntu14.04_cuda9.0_cudnn7_avx_mkl  | [paddle_inference](https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz) |
| ubuntu14.04_cuda10.0_cudnn7_avx_mkl  | [paddle_inference](https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda10-cudnn7-avx-mkl/fluid_inference.tgz) |

For more and later versions, you can download as required: [C++ prediction library download list] (https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)

The directory `/root/projects/fluid_inference` after downloading and decompression contains the following contents:
```
fluid_inference ├── paddle # paddle core library and header files
| 
├──third_party # third-party dependency library and header files
|
└── version.txt # Version and compilation information
```

**Note**: Except for `nv-jetson-cuda10-cudnn7.5-trt5`, other packages in the pre-compiled versions are based on `GCC 4.8.5`. There may be a `ABI` compatibility problem in the use of the later version of `GCC`. It is recommended to downgrade or [compile the prediction library yourself]. (https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html#id12)


### Step3: Compile

The command to compile `cmake` is in `scripts/build.sh`. Modify the main parameters as required. The main contents are described as follows:
```
# Whether GPU is used (i.e., whether CUDA is used)
WITH_GPU=OFF
# Use MKL or openblas
WITH_MKL=ON
# Whether or not to integrate TensorRT (only WITH_GPU=ON available)
WITH_TENSORRT=OFF
# TensorRT path. If you need to integrate TensorRT, change the path to the actual TensorRT you installed. TENSORRT_DIR=/root/projects/TensorRT/
# Paddle prediction library path. Change the path to the actual prediction library you installed.
PADDLE_DIR=/root/projects/fluid_inference
# Whether or not Paddle's prediction library is compiled using a static library
# When TensorRT is used, Paddle's prediction library is usually a dynamic library
WITH_STATIC_LIB=OFF 
# CUDA's lib path
CUDA_LIB=/usr/local/cuda/lib64
# CUDNN's lib path
CUDNN_LIB=/usr/local/cuda/lib64 

# Whether to load the encrypted model
WITH_ENCRYPTION=ON
# Path to the encryption tool. It may not be modified if you use the customized pre-compiling version.
sh $(pwd)/scripts/bootstrap.sh # Download the pre-compiling version of the encryption tool
ENCRYPTION_DIR=$(pwd)/paddlex-encryption 

# OPENCV path. It may not be modified if you use the customized pre-compiling version.
sh $(pwd)/scripts/bootstrap.sh # Download the pre-compiled version of opencv 
OPENCV_DIR=$(pwd)/deps/opencv3gcc4.8/ 

# You should not modify the following: 
rm -rf build
mkdir -p build
cd build
cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_MKL=${WITH_MKL} \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DWITH_ENCRYPTION=${WITH_ENCRYPTION} \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DWITH_STATIC_LIB=${WITH_STATIC_LIB} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DENCRYPTION_DIR=${ENCRYPTION_DIR} \
    -DOPENCV_DIR=${OPENCV_DIR}
make
```
**Note: **In the compiling in the Linux, the OPENCV, PaddleX-Encryption, and YAML are automatically downloaded. If the access to the Internet is unavailable in the compiling environment, you can download manually:

- [opencv3.4.6gcc4.8ffmpeg.tar.gz2](https://bj.bcebos.com/paddleseg/deploy/opencv3.4.6gcc4.8ffmpeg.tar.gz2)
- [paddlex-encryption.zip](https://bj.bcebos.com/paddlex/tools/1.2.0/paddlex-encryption.zip)
- [yaml-cpp.zip](https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip)

Download opencv3gcc4.8.tar.bz2. Then, unzip it. Specify `OPENCE_DIR` in script/build.sh as the path to unzip. 

Download paddlex-encryption.zip. Then unzip it. Specify `ENCRYPTION_DIR` in script/build.sh as the path to unzip.

After downloading the yaml-cpp.zip file, you don't need to decompress it. In cmake/yaml.cmake, change the website in the `URL https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip` to the path of the downloading file.

After setting the main parameters of the modified script, run the `build` script.
```shell
sh . /scripts/build.sh
```

### Step4: Prediction and visualization

**Before loading the model, make sure that the files in your model directory should include `model.yml`, `__model__`, and `__params__`.`If this condition is not met, refer to the [Model Export to Inference document](../../export_model.md) to export your model to the deployment format. **

* After successful compilation, the executable programs for the image prediction demo are `build/demo/detector`, `build/demo/classifier`, and `build/demo/segmenter`. Users can choose according to their model type. The main command parameters are as follows:

| Parameters | Description |
|  ----  | ----  |
| model_dir | The path of the exported prediction model |
| image | The path of the image file to be predicted |
| image_list | .txt file of storing image paths by line |
| use_gpu | Whether to use GPU prediction (value is 0 (default) or 1) |
| use_trt | Whether to use TensorRT for prediction, the value is 0 or 1 (the default value is 0) |
| use_mkl | Whether or not to use MKL to accelerate CPU prediction, the value is 0 or 1 (default value is 1). |
| mkl_thread_num | Number of threads for MKL inference. By default, it is the number of CPU processors |
| gpu_id | GPU device ID (default value is 0) |
| save_dir | The path to save the visualization result, default is "output", classfier has no such parameter.**** |
| key | Key information generated during the encryption process, default value is "" the unencrypted model is loaded. |
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
| use_mkl | Whether or not to use MKL to accelerate CPU prediction, the value is 0 or 1 (default value is 1). |
| mkl_thread_num | Number of threads for MKL inference. By default, it is the number of CPU processors |
| gpu_id | GPU device ID (default value is 0) |
| show_result | Whether or not to display the prediction visualization result in real time on the screen when making prediction on the video file (the result does not reflect the real frame rate because the delay process is added), the supported value is 0 or 1 (the default value is 0). |
| save_result | Whether to save the predicted visual result of each frame as a video file, the value is 0 or 1 (default value is 1) |
| save_dir | Path to save the visualization results (default value is "output") |
| key | Key information generated during the encryption process, default value is "" the unencrypted model is loaded. |

**Note: If the GUI is unavailable in the system, you should not set show_result to 1. When using a camera for prediction, press `ESC` to disable the camera and launch the prediction program.**

## Example

Predictions can be made using the `inference_model`  and test images exported from the [DUDU recognition model]>(../../export_model.md), to export to /root/projects. The model path is /root/projects/inference_model.

> Description about the prediction speed: The prediction speed of the first few images after loading the model is slow, because the initialization of the video card and memory is involved in the start-up. Generally, the prediction speed after predicting 20-30 images is stable.

**Example 1:**

Not using `GPU` test images: `/root/projects/images/xiaoduxiong.jpeg`

```shell
. /build/demo/detector --model_dir=/root/projects/inference_model --image=/root/projects/images/xiaoduxiong.jpeg --save_dir=output
```
The image `file visual predictions` are saved in the directory where the `save_dir` parameter is set.


**Example 2:**

Using the `GPU` to predict multiple images `/root/projects/image_list.txt`. The content of image_list.txt is in the following format:`
```
/root/projects/images/xiaoduxiong1.jpeg /root/projects/images/xiaoduxiong2.jpeg . . . /root/projects/images/xiaoduxiongn.jpeg
```
```shell
/root/projects/images/xiaoduxiong1.jpeg
/root/projects/images/xiaoduxiong2.jpeg
...
/root/projects/images/xiaoduxiongn.jpeg
```
```shell
./build/demo/detector --model_dir=/root/projects/inference_model --image_list=/root/projects/images_list.txt --use_gpu=1 --save_dir=output --batch_size=2 --thread_num=2
```
The image file `visual predictions` are saved in the directory where the `save_dir` parameter is set.

**Example 3:**

Using the camera prediction:

```shell
. /build/demo/video_detector --model_dir=/root/projects/inference_model --use_camera=1 --use_gpu=1 --save_dir=output --save_result=1
```
When `save_result` is set to 1, the `visual prediction results` are saved in the directory where the `save_dir` parameter is set in the video file format.

**Example 4:**

Predicting the video file:

```shell
. /build/demo/video_detector --model_dir=/root/projects/inference_model --video_path=/path/to/video_file --use_gpu=1 --save_dir=output --show_result=1 --save_result=1
```
When `save_result` is set to 1, the `visual prediction results` are saved in the directory where the `save_dir` parameter is set in the video file format.`If the GUI is available in the system, view the visual prediciton results on the screen by setting `show_result to 1.
