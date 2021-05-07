# 基于PaddleInference的推理-Windows环境编译

本文档指引用户如何基于PaddleInference对飞桨模型进行推理，并编译执行。Windows 平台下，我们使用`Visual Studio 2019 Community` 进行了测试。微软从`Visual Studio 2017`开始即支持直接管理`CMake`跨平台编译项目，但是直到`2019`才提供了稳定和完全的支持，所以如果你想使用CMake管理项目编译构建，我们推荐你使用`Visual Studio 2019`环境下构建。

## 环境依赖

* Visual Studio 2019
* CUDA 10.0/CUDA 10.1/CUDA 10 .2等, CUDNN 7+ （仅在使用GPU版本的预测库时需要，需要与下载的预测库版本一致）
* CMake 3.0+

请确保系统已经安装好上述基本软件，我们使用的是`VS2019`的社区版。

## 编译步骤

**下面所有示例以工作目录为 `D:\projects`演示。**

### Step1: 下载PaddleX预测代码

```shell
d:
mkdir projects
cd projects
git clone https://github.com/PaddlePaddle/PaddleX.git
git checkout deploykit
```

**说明**：其中`C++`预测代码在`PaddleX\deploy\cpp` 目录，该目录不依赖任何`PaddleX`下其他目录。所有的公共实现代码在`model_deploy`目录下，而示例demo代码为`demo/model_infer.cpp`。


### Step2: 下载PaddlePaddle C++ 预测库 

PaddlePaddle C++ 预测库针对是否使用GPU、是否支持TensorRT、以及不同的CUDA版本提供了已经编译好的预测库，目前PaddleX支持Paddle预测库2.0+，最新2.0.2版本下载链接如下所示:

| 版本说明                     | 预测库(2.0.2)                                                                                                   | 编译器                | 构建工具      | cuDNN | CUDA |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------- | ------------- | ----- | ---- |
| cpu_avx_mkl                  | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.2/win-infer/mkl/cpu/paddle_inference.zip)         | Visual Studio 15 2017 | CMake v3.17.0 | -     | -    |
| cuda10.0_cudnn7_avx_mkl      | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.2/win-infer/mkl/post100/paddle_inference.zip)     | MSVC 2015 update 3    | CMake v3.17.0 | 7.4.1 | 10.0 |
| cuda10.0_cudnn7_avx_mkl_trt6 | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.2/win-infer/trt_mkl/post100/paddle_inference.zip) | MSVC 2015 update 3    | CMake v3.17.0 | 7.4.1 | 10.0 |
| cuda10.1_cudnn7_avx_mkl_trt6 | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.2/win-infer/trt_mkl/post101/paddle_inference.zip) | MSVC 2015 update 3    | CMake v3.17.0 | 7.6   | 10.1 |
| cuda10.2_cudnn7_avx_mkl_trt7 | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.2/win-infer/trt_mkl/post102/paddle_inference.zip) | MSVC 2015 update 3    | CMake v3.17.0 | 7.6   | 10.2 |
| cuda11.0_cudnn8_avx_mkl_trt7 | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.2/win-infer/trt_mkl/post11/paddle_inference.zip)  | MSVC 2015 update 3    | CMake v3.17.0 | 8.0   | 11.0 |

请根据实际情况选择下载，如若以上版本不满足您的需求，请至[C++预测库下载列表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/windows_cpp_inference.html)选择符合的版本。

将预测库解压后，其所在目录（例如`D:\projects\paddle_inference_install_dir\`）下主要包含的内容有：

```
├── \paddle\ # paddle核心库和头文件
|
├── \third_party\ # 第三方依赖库和头文件
|
└── \version.txt # 版本和编译信息
```

### Step3: 安装配置OpenCV

1. 在OpenCV官网下载适用于Windows平台的3.4.6版本， [下载地址](https://bj.bcebos.com/paddleseg/deploy/opencv-3.4.6-vc14_vc15.exe)  
2. 运行下载的可执行文件，将OpenCV解压至指定目录，例如`D:\projects\opencv`
3. 配置环境变量，如下流程所示  
   - 我的电脑->属性->高级系统设置->环境变量
   - 在系统变量中找到Path（如没有，自行创建），并双击编辑
   - 新建，将opencv路径填入并保存，如`D:\projects\opencv\build\x64\vc15\bin`
   - 在进行cmake构建时，会有相关提示，请注意vs2019的输出

### Step4: 使用Visual Studio 2019直接编译CMake

1. 打开Visual Studio 2019 Community，点击`继续但无需代码`
   ![](../../../../../docs/deploy/images/vs2019_step1.png)
2. 点击： `文件`->`打开`->`CMake`

![](../../../../../docs/deploy/images/vs2019_step2.png)

选择C++预测代码所在路径（例如`D:\projects\PaddleX\deploy\cpp`），并打开`CMakeList.txt`：
![](../../../../../docs/deploy/images/vs2019_step3.png)

3. 打开项目时，可能会自动构建。由于没有进行下面的依赖路径设置会报错，这个报错可以先忽略。

  点击：`项目`->`CMake设置`
  ![](../../../../../docs/deploy/images/vs2019_step4.png)

4. 点击`浏览`，分别设置编译选项指定`CUDA`、`OpenCV`、`Paddle预测库`的路径（也可以点击右上角的“编辑 JSON”，直接修改json文件，然后保存点 项目->生成缓存）
   ![](../../../../../docs/deploy/images/vs2019_step5.png)
   依赖库路径的含义说明如下（带*表示仅在使用**GPU版本**预测库时指定, 其中CUDA库版本尽量与Paddle预测库的对齐，例如Paddle预测库是**使用9.0、10.0版本**编译的，则编译PaddleX预测代码时**不使用9.2、10.1等版本**CUDA库）：

| 参数名     | 含义                                                                                                                                                |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| *CUDA_LIB  | CUDA的库路径, 注：请将CUDNN的cudnn.lib文件拷贝到CUDA_LIB路径下。<br />例如 `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\lib\\x64` |
| OPENCV_DIR | OpenCV的安装路径，例如`D:\\projects\\opencv`                                                                                                        |
| PADDLE_DIR | Paddle c++预测库的路径,例如 `D:\\projects\\paddle_inference_install_dir`                                                                            |

**注意：**

- 如果使用`CPU`版预测库，请把`WITH_GPU`的`值`去掉勾
- 如果使用的是`openblas`版本，请把`WITH_MKL`的`值`去掉勾
- Windows环境下编译会自动下载YAML，如果编译环境无法访问外网，可手动下载： [yaml-cpp.zip](https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip)。YAML文件下载后无需解压，在`cmake/yaml.cmake`中将`URL https://bj.bcebos.com/paddlex/deploy/deps/yaml-cpp.zip` 中的网址，改为下载文件的路径。

5. 保存并生成CMake缓存

![](../../../../../docs/deploy/images/vs2019_step6.png)
**设置完成后**, 点击上图中`保存并生成CMake缓存以加载变量`。然后我们可以看到vs的输出会打印CMake生成的过程，出现`CMake 生成完毕`且无报错代表生成完毕。

6. 点击`生成`->`全部生成`，生成demo里的可执行文件。

![step6](../../../../../docs/deploy/images/vs2019_step7.png)

### Step5: 预测

**在加载模型前，请检查你的模型是部署格式, 应该包括模型、参数、配置三个文件.比如是`__model__`、`__params__`、`infer_cfg.yml` 也可以是`model.pdmodel`、`model.pdiparams`、`deploy.yaml` 三个文件。如若不满足这个条件，请参考[部署模型导出](../../export_model.md)将模型导出为部署格式。**  

上述`Visual Studio 2019`编译产出的可执行文件在`out\build\x64-Release`目录下，打开`cmd`，并切换到该目录：

```
D:
cd D:\projects\PaddleX\deploy\cpp\out\build\x64-Release
```

* 编译成功后，图片预测demo的入口程序为`paddlex_inference\model_infer.exe`，用户可根据自己的需要以及模型类型，设置参数。其主要命令参数说明如下：

| 参数            | 说明                                                         |                    |
| --------------- | ------------------------------------------------------------ | ------------------ |
| model_filename  | 导出的预测模型 模型(model)的路径,例如`D:/ppyolo/__model__`   | 必填               |
| params_filename | 导出的预测模型 参数(params)的路径，例如`D:/ppyolo/__params__` | 必填               |
| cfg_file        | 导出的预测模型 配置文件(yml)的路径，例如`D:/ppyolo/infer_cfg.yml` | 必填               |
| model_type      | 导出预测模型所用的框架。当前支持的套件为[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)、[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)、[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)、[PaddleX](https://github.com/PaddlePaddle/PaddleX)对应的model_type分别为 det、seg、clas、paddlex | 必填               |
| image           | 要预测的图片文件路径                                         | 跟image_list二选一 |
| image_list      | 按行存储图片路径的.txt文件                                   | 跟image二选一      |
| use_gpu         | 是否使用 GPU 预测, 使用为1                                   | 默认为 0           |
| use_mkl         | 是否使用 MKL加速CPU预测, 使用为1                             | 默认为 1           |
| thread_num      | openmp对batch并行的线程数，默认为1                           | 默认为 1           |
| batch_size      | 预测的批量大小，默认为1                                      | 默认为 1           |
| gpu_id          | GPU 设备ID,默认为0                                           | 默认为0            |



## 推理运行样例

例如我们使用[[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)] release/0.5中导出的[yolov3_darknet](https://bj.bcebos.com/paddlex/deploy/models//yolov3_darknet.tar.gz)模型进行预测, 例如导出到`D:\projects`，模型解压路径为`D:\projects\yolov3_darknet`。用PaddleDetection套件导出的模型，所以model_type参数必须为det。

> 关于预测速度的说明：加载模型后前几张图片的预测速度会较慢，这是因为运行启动时涉及到内存显存初始化等步骤，通常在预测20-30张图片后模型的预测速度达到稳定。


### 样例一：(对单张图像做预测)

不使用`GPU`测试图片  `D:\images\xiaoduxiong.jpeg`  

```shell
.\paddlex_inference\model_infer.exe --model_filename=D:\projects\yolov3_darknet\__model__ --params_filename=D:\projects\yolov3_darknet\__params__ --cfg_file=D:\projects\yolov3_darknet\infer_cfg.yml --model_type=det --image=D:\images\xiaoduxiong.jpeg --use_gpu=0

```

图片的结果会打印出来，如果要获取结果的值，可以参照demo/model_infer.cpp里的代码拿到model->results_


### 样例二：(对图像列表做预测)

使用`GPU`预测多个图片，batch_size为2。假设有个`D:\images\image_list.txt`文件，image_list.txt内容的格式如下：

```
images/image1.jpeg
images/image2.jpeg
...
images/imagen.jpeg
```

```shell
.\paddlex_inference\model_infer.exe --model_filename=D:\projects\yolov3_darknet\__model__ --params_filename=D:\projects\yolov3_darknet\__params__ --cfg_file=D:\projects\yolov3_darknet\infer_cfg.yml --model_type=det --image_list=D:\images\image_list.txt --use_gpu=1 --gpu_id=0 --batch_size=2 --thread_num=2
```



## 多卡上运行

当前支持单机多卡部署，暂时不支持跨机器多卡。多卡部署必须使用`paddlex_inference\multi_gpu_model_infer.exe`进行预测，将每个batch的数据均摊到每张卡上进行并行加速。多卡的gpu_id设置几个GPU的id，就会在那几个GPU上进行预测,每个gpu_id之间用英文逗号隔开。注意：**单卡的`model_infer.exe`只能设置一个GPU的id。**

```c++
// 每次4张图片均摊在第4、5张卡进行推理计算
.\paddlex_inference\multi_gpu_model_infer.exe --model_filename=D:\projects\yolov3_darknet\__model__ --params_filename=D:\projects\yolov3_darknet\__params__ --cfg_file=D:\projects\yolov3_darknet\infer_cfg.yml --model_type=det --image_list=D:\images\image_list.txt --use_gpu=1 --gpu_id=4,5  --batch_size=4 --thread_num=2
  
// 每次4张图片均摊在第0、1、2、3张卡上并行推理计算
.\paddlex_inference\multi_gpu_model_infer.exe --model_filename=D:\projects\yolov3_darknet\__model__ --params_filename=D:\projects\yolov3_darknet\__params__ --cfg_file=D:\projects\yolov3_darknet\infer_cfg.yml --model_type=det --image_list=D:\images\image_list.txt --use_gpu=1 --gpu_id=0,1,2,3  --batch_size=4 --thread_num=2
```



## 部署支持

目前部署支持的模型包括

1. PaddleX 训练导出的模型
2. PaddleDetection release-0.5版本导出的模型（仅支持FasterRCNN/MaskRCNN/PPYOLO/YOLOv3)
3. PaddleSeg release-2.0版本导出的模型
4. PaddleClas release-2.0版本导出的模型

编译完成后,其他套件部署如下：

- [PaddleX部署指南](../../models/paddlex.md)
- [PaddleDetection部署指南](../../models/paddledetection.md)
- [PaddleSeg部署指南](../../models/paddleseg.md)
- [PaddleClas部署指南](../../models/paddleclas.md)