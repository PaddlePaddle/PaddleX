# Windows平台

## 说明
Windows 平台下，我们使用`Visual Studio 2019 Community` 进行了测试。微软从`Visual Studio 2017`开始即支持直接管理`CMake`跨平台编译项目，但是直到`2019`才提供了稳定和完全的支持，所以如果你想使用CMake管理项目编译构建，我们推荐你使用`Visual Studio 2019`环境下构建。

## 前置条件
* Visual Studio 2019
* OpenVINO 2021.1+
* CMake 3.0+

**说明**：PaddleX安装请参考[PaddleX](https://paddlex.readthedocs.io/zh_CN/develop/install.html) ， OpenVINO安装请参考[OpenVINO-Windows](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html)  

**注意**：安装完OpenVINO后需要手动添加OpenVINO目录到系统环境变量，否则在运行程序时会出现找不到dll的情况。以安装OpenVINO时不改变OpenVINO安装目录情况下为示例，流程如下
- 我的电脑->属性->高级系统设置->环境变量
    - 在系统变量中找到Path（如没有，自行创建），并双击编辑
    - 新建，分别将OpenVINO以下路径填入并保存:  

      `C:\Program File (x86)\IntelSWTools\openvino\inference_engine\bin\intel64\Release`  

      `C:\Program File (x86)\IntelSWTools\openvino\inference_engine\external\tbb\bin`  

      `C:\Program File (x86)\IntelSWTools\openvino\deployment_tools\ngraph\lib`  

请确保系统已经安装好上述基本软件，并配置好相应环境，**下面所有示例以工作目录为 `D:\projects`演示。**

## 预测部署  

文档提供了c++下预测部署的方法，如果需要在python下预测部署请参考[python预测部署](./python.md)

### Step1: 下载PaddleX预测代码

```shell
d:
mkdir projects
cd projects
git clone https://github.com/PaddlePaddle/PaddleX.git
```

**说明**：其中`C++`预测代码在`PaddleX\deploy\openvino` 目录，该目录不依赖任何`PaddleX`下其他目录。

### Step2 软件依赖
提供了依赖软件预编译库:
- [gflas](https://bj.bcebos.com/paddlex/deploy/windows/third-parts.zip)  
- [opencv](https://bj.bcebos.com/paddleseg/deploy/opencv-3.4.6-vc14_vc15.exe)  

请下载上面两个连接的预编译库。若需要自行下载请参考：
- gflags:[下载地址](https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/gflags)
- opencv:[下载地址](https://opencv.org/releases/)  

下载完opencv后需要配置环境变量，如下流程所示  
    - 我的电脑->属性->高级系统设置->环境变量
    - 在系统变量中找到Path（如没有，自行创建），并双击编辑
    - 新建，将opencv路径填入并保存，如`D:\projects\opencv\build\x64\vc14\bin`

### Step3: 使用Visual Studio 2019直接编译CMake
1. 打开Visual Studio 2019 Community，点击`继续但无需代码`
2. 点击： `文件`->`打开`->`CMake` 选择C++预测代码所在路径（例如`D:\projects\PaddleX\deploy\openvino`），并打开`CMakeList.txt`  
3. 点击：`项目`->`CMake设置`
4. 点击`浏览`，分别设置编译选项指定`OpenVINO`、`Gflags`、`NGRAPH`、`OPENCV`的路径  

|  参数名   | 含义  |
|  ----  | ----  |
| OPENCV_DIR  | OpenCV库路径 |
| OPENVINO_DIR | OpenVINO推理库路径，在OpenVINO安装目录下的deployment/inference_engine目录，若未修改OpenVINO默认安装目录可以不用修改 |
| NGRAPH_LIB | OpenVINO的ngraph库路径，在OpenVINO安装目录下的deployment/ngraph/lib目录，若未修改OpenVINO默认安装目录可以不用修改 |
| GFLAGS_DIR | gflags库路径 |
| WITH_STATIC_LIB | 是否静态编译，默认为True |  

**设置完成后**, 点击`保存并生成CMake缓存以加载变量`。
5. 点击`生成`->`全部生成`
### Step5: 预测
上述`Visual Studio 2019`编译产出的可执行文件在`out\build\x64-Release`目录下，打开`cmd`，并切换到该目录：

```
D:
cd D:\projects\PaddleX\deploy\openvino\out\build\x64-Release
```

* 编译成功后，图片预测demo的入口程序为`detector.exe`，`classifier.exe`，`segmenter.exe`，用户可根据自己的模型类型选择，其主要命令参数说明如下：

|  参数   | 说明  |
|  ----  | ----  |
| --model_dir  | 模型转换生成的.xml文件路径，请保证模型转换生成的三个文件在同一路径下|
| --image  | 要预测的图片文件路径 |
| --image_list  | 按行存储图片路径的.txt文件 |
| --device  | 运行的平台，可选项{"CPU"，"MYRIAD"}，默认值为"CPU"，如在VPU上请使用"MYRIAD"|
| --cfg_file | PaddleX model 的.yml配置文件 |
| --save_dir | 可视化结果图片保存地址，仅适用于检测任务，默认值为" "，即不保存可视化结果 |

### 样例
`样例一`：
在CPU下做单张图片的分类任务预测  
测试图片 `/path/to/test_img.jpeg`  

```shell
./classifier.exe --model_dir=/path/to/openvino_model --image=/path/to/test_img.jpeg --cfg_file=/path/to/PadlleX_model.yml
```

`样例二`:
在CPU下做多张图片的检测任务预测，并保存预测可视化结果
预测多个图片`/path/to/image_list.txt`，image_list.txt内容的格式如下：
```
/path/to/images/test_img1.jpeg
/path/to/images/test_img2.jpeg
...
/path/to/images/test_imgn.jpeg
```

```shell
./detector.exe --model_dir=/path/to/models/openvino_model --image_list=/root/projects/images_list.txt --cfg_file=/path/to/PadlleX_model.yml --save_dir ./output
```

`样例三`:  
在VPU下做单张图片分类任务预测
测试图片 `/path/to/test_img.jpeg`  

```shell
.classifier.exe --model_dir=/path/to/openvino_model --image=/path/to/test_img.jpeg --cfg_file=/path/to/PadlleX_model.yml --device=MYRIAD
```
