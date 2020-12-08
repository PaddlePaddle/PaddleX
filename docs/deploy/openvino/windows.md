# Windows platform

## Description
On the Windows platform, use the `Visual Studio 2019 Community` for testing. Since 2017, Microsoft Visual Studio has supported the direct management of `CMake` cross-platform compilation projects. But it did not provide stable and complete support until `2019`. If you want to use CMake to manage project compilation and build, `Visual Studio 2019` is recommended.

## Pre-conditions
* Visual Studio 2019
* OpenVINO 2021.1+
* CMake 3.0+

**Note**: For PaddleX installation, refer to [PaddleX]. For OpenVINO installation, refer to [OpenVINO-Windows] (https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html)  

**Note**: After installing OpenVINO, you need to manually add the OpenVINO directory to the system environment variable. Otherwise, the dll may not be found when you run the program. For example, if you install OpenVINO without changing the OpenVINO installation directory, the process is as follows:
- My Computer->Properties->Advanced System Settings->Environmental Variables
   - Find Path in the system variables (if not, create one yourself and double-click to edit it.
   - To create a new one, fill in the following paths for OpenVINO respectively, and save it:

      `C:\Program File (x86)\IntelSWTools\openvino\inference_engine\bin\intel64\Release`

      `C:\Program File (x86)\IntelSWTools\openvino\inference_engine\external\tbb\bin`

      `C:\Program File (x86)\IntelSWTools\openvino\deployment_tools\ngraph\lib`

Make sure that you have installed the above basic software and configured your system accordingly. **All the examples below are based on the working directory `D:\projects`.**

## Inference deployment

This document provides prediction deployment methods under c++. To perform prediction deployment under python, see [python prediction deployment](./python.md).

### Step1: Download the PaddleX prediction code.

```shell
d:
mkdir projects
cd projects
git clone https://github.com/PaddlePaddle/PaddleX.git
```

**Note**: The C++` prediction code is in the `PaddleX`\deploy\openvino` directory. The directory does not depend on any other directory in PaddleX.

### Step2: Software dependencies
Pre-compiled libraries for dependent software are provided:
- [gflas](https://bj.bcebos.com/paddlex/deploy/windows/third-parts.zip)  
- [opencv](https://bj.bcebos.com/paddleseg/deploy/opencv-3.4.6-vc14_vc15.exe)  
Download the pre-compiled libraries for the two links above. If you need to download them yourself, please refer to:
- gflags: [download address](https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/gflags)
- opencv: [download address](https://opencv.org/releases/)
After downloading opencv, you need to configure the environment variables as follows:
   - My Computer->Properties->Advanced System Settings->Environmental Variables
   - Find Path in the system variables (if not, create one yourself) and double-click to edit it.
   - Add a new file. Fill in the opencv path and save it. For example, `D:\projects\opencv\build\x64\vc14\bin`

### Step3: Compile CMake directly by using Visual Studio 2019
1. Open Visual Studio 2019 Community and click `Continue, but no code is required`
2. Choose `File`->`Open`->`CMake` to select the path where the C++ prediction code is located (for example, `D:\projects\PaddleX\deploy\openvino`), and open `CMakeList.txt`.
3. Choose `Project`->`CMake Settings`
4. Click `Browse` to set the compiling options, and specify the paths to `OpenVINO`, `Gflags`, `NGRAPH`, and `OPENCV` respectively.

| Parameter Name | Meaning |
|  ----  | ----  |
| OPENCV_DIR | OpenCV library paths |
| OPENVINO_DIR | The OpenVINO inference library path is located in the deployment/inference_engine directory under the OpenVINO installation directory. If OpenVino is not modified, the installation directory should not be modified by default. |
| NGRAPH_LIB | The path of OpenVINO's ngraph library is located in the deployment/ngraph/lib directory under the OpenVINO installation directory. If OpenVino is not modified, the installation directory should not be modified by default. |
| GFLAGS_DIR | gflags library path |
| WITH_STATIC_LIB | Whether is static compiling. By default, it is True. |

**After the settings are complete**, click Save to generate the CMake cache to load the variables.`
5. Choose `Generate`-> `Generate All`
### Step 5: Prediction
The above compiled executable files in `Visual Studio 2019` are in the `out\build\x64-Release` directory. Run `cmd` to go to the directory:

```
D:
cd D:\projects\PaddleX\deploy\openvino\out\build\x64-Release
```

* After successful compilation, the entry program for the image prediction demo is `detector.exe`, `classifier.exe`, and `segmenter.exe`. You can choose according to the model types. Its main command parameters are described as follows:

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
Classification task prediction for a single image under the CPU
Test image `/path/to/test_img.jpeg`

```shell
. /classifier. exe --model_dir=/path/to/openvino_model --image=/path/to/test_img.jpeg --cfg_file=/path/to/PadlleX_model.yml
```

`Example 2`:
Detection task prediction of multiple images under CPU and saving of the prediction visualization results
Prediction of multiple images: `/path/to/image_list.txt`. The format of the image_list.txt content is as follows:
```
/path/to/images/test_img1.jpeg
/path/to/images/test_img2.jpeg
...
/path/to/images/test_imgn.jpeg
```

```shell
./detector.exe --model_dir=/path/to/models/openvino_model --image_list=/root/projects/images_list.txt --cfg_file=/path/to/PadlleX_model.yml --save_dir ./output
```

`Example 3`:
Classification task prediction for a single image under the VPU
Test image `/path/to/test_img.jpeg`  

```shell
.classifier.exe --model_dir=/path/to/openvino_model --image=/path/to/test_img.jpeg --cfg_file=/path/to/PadlleX_model.yml --device=MYRIAD
```
