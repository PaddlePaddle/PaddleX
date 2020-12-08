# Windows platform deployments

## Description
On the Windows platform, use the `Visual Studio 2019 Community` for testing. Since 2017, Microsoft Visual Studio has supported the direct management of `CMake` cross-platform compilation projects. But it did not provide stable and complete support until `2019`. If you want to use CMake to manage project compilation and build, `Visual Studio 2019` is recommended.

## Pre-conditions
* Visual Studio 2019
* CUDA 9.0 / CUDA 10.0, CUDNN 7+ (required only if using GPU version prediction library)
* CMake 3.0+

Make sure that the above basic software is installed on your system. Here the `VS2019` Community Edition is used.

**All the examples below are shown in the working directory: `D:\projects`.**

### Step1: Download the PaddleX prediction code.

```shell
d:
mkdir projects
cd projects
git clone https://github.com/PaddlePaddle/PaddleX.git
```

**Note**: The `C++` prediction code is in `PaddleX`\deploy\cpp` directory, which does not depend on any other directory in `PaddleX`.


### Step2: Download PaddlePaddle C++ Prediction Library: paddle_inference.

PaddlePaddle C++ prediction Library provides the compiled prediction libraries for the use of GPU or not, whether or not support TensorRT, and different CUDA versions. At present, PaddleX depends on Paddle 1.8.4. The download link of Paddle prediction library based on Paddle 1.8.4 is as follows:

| Release Notes | Prediction Library (Version 1.8.4) | Compilers | Building Tools | cuDNN | CUDA |
|  ----  |  ----  |  ----  |  ----  | ---- | ---- |
| cpu_avx_mkl  | [paddle_inference](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/cpu/fluid_inference_install_dir.zip) | MSVC 2015 update 3 | CMake v3.16.0 |
| cpu_avx_openblas  | [paddle_inference](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/open/cpu/fluid_inference_install_dir.zip) | MSVC 2015 update 3 | CMake v3.16.0 |
| cuda9.0_cudnn7_avx_mkl  | [paddle_inference](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/post97/fluid_inference_install_dir.zip) | MSVC 2015 update 3 | CMake v3.16.0 | 7.4.1 | 9.0 |
| cuda9.0_cudnn7_avx_openblas  | [paddle_inference](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/open/post97/fluid_inference_install_dir.zip) | MSVC 2015 update 3 | CMake v3.16.0 | 7.4.1 | 9.0 |
| cuda10.0_cudnn7_avx_mkl  | [paddle_inference](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/post107/fluid_inference_install_dir.zip) | MSVC 2015 update 3 | CMake v3.16.0 | 7.5.0 | 10.0 |

Select the download as required. If the above version does not meet your needs, go to the [download list of C++ prediction library] and choose a suitable version (https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/advanced_guide/inference_deployment/inference/windows_cpp_inference.html).

After unzipping the prediction library, the directory is located in (for example, `D:\projects\fluid_inference\`) contains the following:
```
├── \paddle\ # paddle core library and header files 
|
├── \third_party\ # Third-party dependency library and header files
|
└── \version.txt # Version and compilation information
```

### Step3: Install and configure the OpenCV

1. Download version 3.4.6 for Windows from the OpenCV website. The [website for downloading] (https://bj.bcebos.com/paddleseg/deploy/opencv-3.4.6-vc14_vc15.exe)
2. Run the downloaded executable file and decompress the OpenCV to the specified directory, for example, `D:\projects\opencv`
3. Configure the environment variables:
   - My Computer->Properties->Advanced System Settings->Environmental Variables
   - Find Path in the system variables (if not, create one yourself and double-click to edit it.
   - Add a new file. Fill in the opencv path and save it. For example, `D:\projects\opencv\build\x64\vc14\bin`

### Step4: Compile CMake directly with Visual Studio 2019.

1. Open Visual Studio 2019 Community and click `Continue. No code is required` 
![](../../images/vs2019_step1.png)
2. Choose: `File` ->`Open`->`CMake`

![](../../images/vs2019_step2.png)

Choose the path where the C++ prediction code is located (for example, `D :\projects\PaddleX\deploy\cpp`), and open `CMakeList.txt`: 
![](../../images/vs2019_step3.png)
3. Choose `Project`->`CMake Settings`
![](../../images/vs2019_step4.png)
4. Click `Browse` to set the compiling options to specify the paths to the `CUDA`, `OpenCV`, and `Paddle prediction libraries`, respectively.
![](../../images/vs2019_step5.png)
Meaning of the dependency library path (with * means it is only specified when using the **GPU version** prediction library. For the CUDA library version, it should be aligned with the Paddle prediction library as much as possible. For example, if the Paddle prediction library is compiled with **versions 9.0 and 10.0**, the PaddleX prediction codes are compiled ** without using the CUDA libraries of V9.2, 10.1):

| Parameter Name | Meaning |
|  ----  | ----  |
| *CUDA_LIB | CUDA library path (Note that you should copy the cudnn.lib file from CUDNN to CUDA_LIB path) |
| OPENCV_DIR | the installation path of OpenCV. |
| PADDLE_DIR | Paddle c++ prediction library paths |

**Note:**
1. If you are using the `CPU` the prediction library, de-select `WITH_GPU`.
2. If you are using the `openblas` version, de-select `WITH_MKL`.
3. In the compiling in the windows environment, YAML is downloaded automatically. If you can't access the external network, you can download [yaml-cpp.zip](https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip)yaml-cpp.zip manually. After downloading the YAML file, you don't need to decompress it, just change the website in the `URL https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip`  to the path of the downloaded file in `cmake/yaml.cmake`.
4. If you use the model encryption function, you need to download the [Windows Prediction Model Encryption Tool](https://bj.bcebos.com/paddlex/tools/win/1.2.0/paddlex-encryption.zip) manually. For example, decompress it to `D:/projects`. The directory after decompression is `D:/projects/paddlex-encryption`. When compiling, select `WITH_EBNCRYPTION` and fill in `D:/projects/paddlex-encryption` in `ENCRTYPTION_DIR`.
![](../../images/vs2019_step_encryption.png)
![](../../images/vs2019_step6.png)
**After the settings are complete**, click `Save to generate CMake cache to load the variables`.
5. Choose `Generate`->`Generate All`

![step6](../../images/vs2019_step7.png)

### Step 5: Prediction and visualization

**Before loading the model, make sure that the files in your model directory should include `model.yml`, `__model__`, and `__params__`. If this condition is not met, refer to the [Deploy Model Export](../../export_model.md) to export your model to the deployment format.** 

The above compiled executable files in `Visual Studio 2019` are in the `out\build\x64-Release` directory. Run `cmd` to go to the directory:

```
D:
cd D:\projects\PaddleX\deploy\cpp\out\build\x64-Release
```

* After successful compilation`, the entry program of the image prediction demo is `paddlex_inference\detector.exe`, paddlex_inference\classifier.exe`, `paddlex_inference\segmenter.exe`, users can choose according to their own model types. Its main command parameters are described as follows:

| Parameters | Description |
|  ----  | ----  |
| model_dir | The path of the exported prediction model |
| image | The path of the image file to be predicted |
| image_list | .txt file of storing image paths by line |
| use_gpu | Whether to use GPU prediction (value is 0 (default) or 1) |
| use_mkl | Whether or not to use MKL to accelerate CPU prediction, the value is 0 or 1 (default value is 1). |
| mkl_thread_num | Number of threads for MKL inference. By default, it is the number of CPU processors |
| gpu_id | GPU device ID (default value is 0) |
| save_dir | The path to save the visualization result. The default value is "output", and classifier has no such parameter. |
| key | Key information generated during the encryption process, default value is "" the unencrypted model is loaded. |
| batch_size | Prediction batch size, default is 1 |
| thread_num | Number of predicted threads. By default, it is the number of CPU processors |

* After the successful compilation, the entry program for the video prediction demo is `paddlex_inference\video_detector.exe`,  `paddlex_inference\video_classifier.exe`, paddlex_inference\video_segmenter. exe`. Users can choose according to their model type. The main command parameters are described as follows:

| Parameters | Description |
|  ----  | ----  |
| model_dir | The path of the exported prediction model |
| use_camera | Whether to use the camera for prediction, the value is 0 or 1 (default value is 0) |
| camera_id | Camera device ID (default value is 0) |
| video_path | Path of video file |
| use_gpu | Whether to use GPU prediction (value is 0 (default) or 1) |
| use_mkl | Whether or not to use MKL to accelerate CPU prediction, the value is 0 or 1 (default value is 1). |
| mkl_thread_num | Number of threads for MKL inference. By default, it is the number of CPU processors |
| gpu_id | GPU device ID (default value is 0) |
| show_result | Whether or not to display the prediction visualization result in real time on the screen when making prediction on the video file (the result does not reflect the real frame rate because the delay process is added), the supported value is 0 or 1 (the default value is 0). |
| save_result | Whether to save the predicted visual result of each frame as a video file, the value is 0 or 1 (default value is 1) |
| save_dir | Path to save the visualization results (default value is "output") |
| key | Key information generated during the encryption process, default value is "" the unencrypted model is loaded. |

**Note: If the GUI is unavailable in the system, you should not set show_result to 1. When using a camera for prediction, press `ESC` to disable the camera and launch the prediction program.**


## Example

You can use the `inference_model` and test pictures exported from the [DUDU recognition model](../../export_model.md) to make predictions. For example, export to `D:\projects`. The model path is `D:\projects\inference_model`. 

> Description about the prediction speed: The prediction speed of the first few images after loading the model is slow, because the initialization of the video card and memory is involved in the start-up. Generally, the prediction speed after predicting 20-30 images is stable.


### Example 1: (Use the unencrypted model to predict a single image)

Test image without `GPU`: `D:\images\xiaoduxiong.jpeg`

```
. \paddlex_inference\detector.exe --model_dir=D:\projects\inference_model --image=D:\images\xiaoduxiong.jpeg --save_dir=output
```
The image file `visual predictions` are saved in the directory where the `save_dir` parameter is set.


### Example 2: (Use the unencrypted model to predict the image list)

Use `GPU` to predict multiple images `D:\images\image_list.txt`, the format of the content of image_list.txt is as follows: 
```
D:\images\xiaoduxiong1.jpeg
D:\images\xiaoduxiong2.jpeg
...
D:\images\xiaoduxiongn.jpeg
```
```
.\paddlex_inference\detector.exe --model_dir=D:\projects\inference_model --image_list=D:\images\image_list.txt --use_gpu=1 --save_dir=output --batch_size=2 --thread_num=2
```
The image file `visual predictions` are saved in the directory where the `save_dir` parameter is set.

### Example 3: (Use the encrypted model to predict a single picture)

If the model is not encrypted, please refer to the [encrypted PaddleX model](../encryption.html#paddlex) to encrypt the model. For example, the directory where the encrypted model is located at `D:\projects\encrypted_inference_model`.

```
. \paddlex_inference\detector.exe --model_dir=D:\projects\encrypted_inference_model --image=D:\images\xiaoduxiong. jpeg --save_dir=output --key=kLAl1qOs5uRbFt0/RrIDTZW2+tOf5bzvUIaHGF8lJ1c=
```

`--key`: pass in the key output from the encryption tool, for example, `kLAl1qOs5uRbFt0/RrIDTZW2+tOf5bzvUIaHGF8lJ1c=`, the image file visual prediction result will be saved in the directory where the `save_dir` parameter is set.

### Example 4: (Use an unencrypted model to enable camera prediction)

```shell
. \paddlex_inference\video_detector.exe --model_dir=D:\projects\inference_model --use_camera=1 --use_gpu=1 --save_dir=output
```
When `save_result` is set to 1, the `visual prediction results` are saved in the directory where the `save_dir` parameter is set in the video file format.

### Example 5: (Use an unencrypted model to make predictions on video files)


```shell
.\paddlex_inference\video_detector.exe --model_dir=D:\projects\inference_model --video_path=D:\projects\video_test.mp4 --use_gpu=1 --show_result=1 --save_dir=output
```
When `save_result` is set to 1, the visual prediction results are saved in the directory where the `save_dir` parameter is set in the video file format. If the GUI is available in the system, view the visual prediciton results on the screen by setting `show_result` to 1.
