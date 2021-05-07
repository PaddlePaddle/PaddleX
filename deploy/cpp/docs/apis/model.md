# 部署C++ API说明

本文档主要根据部署步骤，讲解用到的一些API、数据结构，包括以下5个内容

1. [创建模型对象](#001)
2. [模型初始化](#002)
3. [模型预测](#003)
4. [预测结果字段](#004)
5. [代码示例](#005)

<span id="001"></span>

## 1. 创建模型对象

```c++
std::shared_ptr<PaddleDeploy::Model>  PaddleDeploy::ModelFactory::CreateObject(const std::string  &model_type)
```

> 根据模型来源的套件类型，创建相应的套件对象并返回基类智能指针。所有推理相关的操作，包括预处理、推理和后处理都在该对象中。

**参数**

> > **model_type** 套件类型，当前支持的套件为[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)、[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)、[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)、[PaddleX](https://github.com/PaddlePaddle/PaddleX)对应的model_type分别为 det、seg、clas、paddlex

**返回值**

>  指向套件对象的父类指针

**代码示例**

> ```c++
> std::shared_ptr<PaddleDeploy::Model> model = PaddleDeploy::ModelFactory::CreateObject("det")
> ```



<span id="002"></span>

## 2. 模型初始化

模型初始化包括2个步骤，第一步读取配置文件，初始化数据预处理和后处理相关操作；第二步初始化推理PaddleInference推理引擎；对应的接口分别为`PaddleDeploy::Model::Init()`和`PaddleDeploy::Model::PaddleEngineInit()`

### 2.1 模型前后处理初始化

```C++
bool Model::Init(const std::string& cfg_file)
```

> 读取模型的配置文件，初始化模型预测过程中的数据预处理和后处理等相关操作

**参数**

> > **cfg_file** 配置文件路径，如PaddleDetection导出的模型中的`infer_cfg.yml`

**返回值**

>  `true`或`false`，表示是否正确初始化

**代码示例**

> ```c++
> if (!model->Init("yolov3_mbv1/model/infer_cfg.yml")) {
>     std::cerr << "Fail to execute model->Init()" << std::endl;
> }
>```



### 2.2 推理引擎初始化

```c++
bool Model::PaddleEngineInit(const std::string& model_filename,
														 const std::string& params_filename,
														 bool use_gpu = false,
														 int gpu_id = 0, 
														 bool use_mkl = true,
														 int mkl_thread_num = 8);
```

> 初始化Paddle 推理引擎,  创建Model或者其子类对象后必须先调用它初始化，才能调推理接口。

**参数**

> >**model_filename** 模型结构文件路径，如`model.pdmodel`或`__model__`
> >
> >**params_filename** 模型权重文件路径，如`model.pdiparams`或`__params__`
> >
> >**use_gpu** 是否使用GPU
> >
> >**gpu_id** GPU设备ID
> >
> >**use_mkl** 是否使用mkldnn（CPU上推理加速引擎）
> >
> >**mkl_thread_num** 使用mkldnn时的推理线程数

**返回值**

>  `true`或`false`，表示是否正确初始化

**代码示例**

> ```c++
> if (!modle->PaddleEngineInit("yolov3_mbv1/model/model.pdmodel", 
>                              "yolov3_mbv1/model/model.pdiparams",
>                              true, 0)) {
>     std::cerr << "Fail to execute model->PaddleEngineInit()" << std::endl;
> }
> ```



<span id="003"></span>

## 3.模型预测

推理过程包括输入数据的预处理、推理引擎的inference、inference结果的后处理3个步骤，在部署代码中，三个步骤分别对应`PaddleDeploy::Model::PreProcess()`、`PaddleDeploy::Model::Infer()`、`PaddleDeploy::Model::PostProcess()`。

为了更便于开发者使用，此三个步骤被封装为`PaddleDeploy::Model::Predict()`一个接口内，用户可根据自行需求进行调用，一般情况下，推荐使用`PaddleDeploy::Model::Predict()`接口即可一步完成预测需求。



## 3.1 预测接口

```c++
 bool Model::Predict(const std::vector<cv::Mat>& imgs,
 						 				 vector<PaddleDeploy::Result>* results,
 						 				 int thread_num = 1)
```

> 对传入的图像进行预处理、inference、后处理，最终结果写回到`results`结构体内

**参数**

> **imgs** 传入的vector，元素为cv::Mat，预测时将会对vector中所有Mat进行预处理，并作为一个batch输入给推理引擎进行预测；开发者在调用时，需考虑硬件配置，vector的size过大时，可能会由于显存或内存不足导致程序出错
>
> **results** 预测结果vector，其与输入的imgs长度相同，vector中每个元素说明参考[预测结果字段说明](#004)
>
> **thread_num** 当输入vector的size大于1时，可通过thread_num来配置预处理和后处理的并行处理时的多线程数量

返回值

>`true`或`false`，表示是否预测成功

**代码示例**

> ```c++
> std::vector<cv::Mat> inputs;
> std::vector<PaddleDeploy::Result> results;
> inputs.push_back(cv::imread("test.jpg", 1));
> if (!model->Predict(inputs, &results)) {
>     std::cerr << "Fail to execute model->Predict()" << std::endl;
> }
> ```



### 3.2 预处理接口

```c++
bool Model::Preprocess(const std::vector<cv::Mat>& imgs,
                       std::vector<PaddleDeploy::DataBlob>* inputs,
                       std::vector<PaddleDeploy::ShapeInfo>* shape_infos,
                       int thread_num = 1)
```

>  对传入的图像进行预处理，得到推理引擎推理时所需的输入，此接口为`PaddleDeploy::Model::Predict`中的第一步

**参数**

> >**imgs** 传入的vector，元素为cv::Mat，预测时将会对vector中所有Mat进行预处理
> >
> >**inputs** 输入的imgs经过预处理后，将输给推理引擎进行推理使用的数据内存存在inputs中
> >
> >**shape_infos** 记录每张图像在预处理过程中的操作信息
> >
> >**thread_num** 当传入imgs的size大于1时，可通过thread_num来配置预处理的并行处理时的多线程数量

**返回值**

> `true`或`false`，表示是否预处理成功

**示例代码**

> ```c++
> std::vector<cv::Mat> imgs;
> std::vector<PaddleDeploy::DataBlob>  inputs;
> std::vector<PaddleDeploy::ShapeInfo> shape_infos;
> inputs.push_back(cv::imread("test.jpg"), 1);
> if (!model->Preprocess(imgs, &inputs, &shape_infos)) {
>     std::cerr << "Fail to execute model->Preprocess()" << std::endl;
> }
> ```



### 3.3 推理接口

```c++
bool Model::Infer(const std::vector<DataBlob>& inputs,  std::vector<DataBlob>* outputs)

> 接收预处理后的数据输入`std::vector<PaddleDeploy::DataBlob>`，使用推理引擎进行推理，此接口为`PaddleDeploy::Model::Predict`中的第二步

**参数**

> > **inputs** 预处理（即`Model::Preprocess`接口)后的输入数据
> >
> > **outputs** 推理引擎推理后的输出数据

**返回值**

> `true`或`false`表示是否推理成功

**示例代码**

```c++
std::vector<cv::Mat> imgs;
std::vector<PaddleDeploy::DataBlob>  inputs;
std::vector<PaddleDeploy::ShapeInfo> shape_infos;
inputs.push_back(cv::imread("test.jpg"), 1);
// 预处理
if (!model->Preprocess(imgs, &inputs, &shape_infos)) {
  std::cerr << "Fail to execute model->Preprocess()" << std::endl;
  return -1;
}
// 推理
std::vector<PaddleDeploy::DataBlob> outputs;
if (!model->Infer(inputs, &outputs)) {
  std::cerr << "Fail to execute model->Infer()" << std::endl;
  return -1;
}
```



### 3.4 后处理接口

```c++
bool Model::Postprocess(const std::vector<PaddleDeploy::DataBlob>& outputs,
												const std::vector<PaddleDeploy::ShapeInfo>& shape_infos,
                        std::vector<PaddleDeploy::Result>* results,
                        int thread_num = 1)
```

> 对推理引擎推理后的结果进行后处理，形成可供用户直接使用的结构化数据，存储在`results`字段中，此接口为`PaddleDeploy::Model::Predict`中的第三步

**参数**

> > **outputs** 推理引擎推理（即`Model::Infer`接口)后的结果
> >
> > **shape_infos** 预处理（即`Model::Preprocess`接口)过程中存储的相关操作信息
> >
> > **results** 后处理后结果写入到此字段，其长度与`Model::Preprocess`接收的图像数量一致
> >
> > **thread_num** 当后处理是在处理多张图像的结果时，通过thread_num可设置后处理的并行线程数

**返回值**

>  `true`或`false`表示是否后处理成功

**示例代码**

> ```c++
> std::vector<cv::Mat> imgs;
> std::vector<PaddleDeploy::DataBlob>  inputs;
> std::vector<PaddleDeploy::ShapeInfo> shape_infos;
> inputs.push_back(cv::imread("test.jpg"), 1);
> // 预处理
> if (!model->Preprocess(imgs, &inputs, &shape_infos)) {
>   std::cerr << "Fail to execute model->Preprocess()" << std::endl;
>   return -1;
> }
> // 推理
> std::vector<PaddleDeploy::DataBlob> outputs;
> if (!model->Infer(inputs, &outputs)) {
>   std::cerr << "Fail to execute model->Infer()" << std::endl;
>   return -1;
> }
> // 后处理
> std::vector<PaddleDeploy::Result> results;
> if (!model->Postprocess(outputs, &results)) {
>   std::cerr << "Fail to execute model->Postprocess()" << std::endl;
>   return -1;
> }
> ```



<span id="004"></span>

## 4. 预测结果字段

在开发者常调用的接口中，包括主要的`Model::Predict`，以及`Model::Predict`内部调用的`Model::Preprocess`、`Model::Infer`和`Model::Postprocess`接口，涉及到的结构体主要为`PaddleDeploy::Result`（预测结果），`PaddleDeploy::DataBlob`和`PaddleDeploy::ShapeInfo`。其中`DataBlob`和`ShapeInfo`开发者较少用到，可直接阅读其代码实现。

以下重点介绍`PaddleDeploy::Result`, 其结构定义如下

```c++
struct Result {
  std::string model_type;    // 结果类型，仅支持det/seg/clas三种
  union {
    // 当model_type为clas时，仅该指针为非Null，存储预测结果
    ClasResult* clas_result; 
    // 当model_type为det时，仅该指针为非Null，存储预测结果
    DetResult* det_result; 
    // 当model_type为seg时，仅该指针为非Null，存储预测结果
    SegResult* seg_result; 
  };
};
```



### 4.1 图像分类结果

```C++
struct ClasResult {
  int category_id;       // 类别id
  std::string category;  // 类别名
  double score;          // 置信度
};
```



### 4.2 目标检测结果

```C++
struct DetResult {
  // 单张图像中所有检测到的Box
  std::vector<Box> boxes;
};

struct Box {
  int category_id;       // 类别id
  std::string category;  // 类别名
  float score;           // 置信度
  // Box四维坐标[xmin, ymin, width, height]
  std::vector<float> coordinate;
  // MaskRCNN实例分割模型中，对应每个Box的掩膜信息
  Mask<uint8_t> mask;
};

template <class T>
struct Mask {
  // Mask数据
  std::vector<T> data;
  // Mask的Shape信息
  std::vector<int> shape;
};
```



### 4.3 语义分割结果

```c++
struct SegResult {
  Mask<uint8_t> label_map;   // 每个像素的分类id
  Mask<float> score_map;     // 每个像素的置信度
}

template <class T>
struct Mask {
  // Mask数据
  std::vector<T> data;
  // Mask的Shape信息
  std::vector<int> shape;
};
```



<span id="005"></span>

## 5. 部署代码示例

以下示例代码以目标检测模型为例

```c++
#include <iostream>
#include "model_deploy/common/include/paddle_deploy.h"
int main() {
  std::shared_ptr<PaddleDeploy::Model> model = 
    				PaddleDeploy::ModelFactory::CreateObject("det");
  model->Init("yolov3_mbv1/model/infer_cfg.yml");
  model->PaddleEngineInit("yolov3_mbv1/model/model.pdmodel",
                          "yolov3_mbv1/model/model.pdiparams",
                          true, 0);
  
  std::vector<cv::Mat> images;
  std::vector<PaddleDeploy::Result> results;
  images.push_back(cv::imread("test.jpg", 1));
  
  if (!model->Predict(images, &results)) {
    std::cerr << "Fail to execute model->Predict()" << std::endl;
    return -1;
  }
  
  // 第一层循环为样本数量，当前为1
  for (auto i = 0; i < results.size(); ++i) {
    // 第二层循环为每张样本中的box数量
    for (auto j = 0; j < (results[i].det_results->boxes).size(); ++j) {
      std::cout << (results[i].det_results->boxes)[j].coordinate[0] << " "
        				<< (results[i].det_results->boxes)[j].coordinate[1] << " "
        				<< (results[i].det_results->boxes)[j].coordinate[2] << " "
        				<< (results[i].det_results->boxes)[j].coordinate[3]
        				<< std::endl;
    }
  }
}
```

输出结果示例如下

```
419.252 155.974 48.2672 144.499
44.3458 213.229 15.759  29.256
372.985 152.078 90.556  144.341
384.818 172.375 17.1848 33.8331
```

