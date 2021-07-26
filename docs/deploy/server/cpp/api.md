# C++代码接口说明

## 头文件
`include/paddlex/paddlex.h`

## 类 PaddleX::Model

模型类，用于加载PaddleX训练的模型。

### 模型加载
```
PaddleX::Model::Init(const std::string& model_dir,
                     bool use_gpu = false,
                     bool use_trt = false,
                     bool use_mkl = true,
                     bool mkl_thread_num = 4,
                     int gpu_id = 0,
                     std::string key = "",
                     bool use_ir_optim = true)
```

**参数**  
- model_dir: 模型目录路径
- use_gpu: 是否使用gpu预测
- use_trt: 是否使用TensorRT
- use_mkl: 是否使用MKLDNN加速模型在CPU上的预测性能
- mkl_thread_num: 使用MKLDNN时，线程数量
- gpu_id: 使用gpu的id号
- key: 模型解密密钥，此参数用于加载加密的PaddleX模型时使用
- use_ir_optim: 是否加速模型后进行图优化

**返回值**
- 返回true或false，表示模型是否加载成功

### 模型预测推断

**分类模型单张图片预测**
```
PaddleX::Model::predict(const cv::Mat& im, ClsResult* result)
```
**分类模型多张图片批预测**
```
PaddleX::Model::predict(const std::vector<cv::Mat>& im_batch, std::vector<ClsResult>* results)
```
**目标检测/实例分割模型单张图片预测**
```
PaddleX::Model::predict(const cv::Mat& im, DetResult* result)
```
**目标检测/实例分割模型多张图片批预测**
```
PaddleX::Model::predict(const std::vector<cv::Mat>& im_batch, std::vector<DetResult>* results)
```
**语义分割模型单张图片预测**
```
PaddleX::Model::predict(const cv::Mat& im, SegResult* result)
```
**语义分割模型多张图片批预测**
```
PaddleX::Model::predict(const std::vector<cv::Mat>& im_batch, std::vector<SegResult>* results)
```
各接口返回值为true或false，用于表示是否预测成功

预测时，需传入cv::Mat结构体，结构需与如下示代码加载的结构体一致
```
cv::Mat im = cv::imread('test.jpg', 1);
```
当使用批预测时，注意会传入的vector中所有数据作为一个批次进行预测，因此vector越大，所需要使用的GPU显存会越高。

预测时，同时传入ClsResult/DetResult/SegResult结构体，用于存放模型的预测结果，各结构体说明如下
```
// 分类模型预测结果
class ClsResult {
 public:
  int category_id; // 类别id
  std::string category; // 类别标签
  float score; // 预测置信度
  std::string type = "cls";
}

// 目标检测/实例分割模型预测结果
class DetResult {
 public:
  std::vector<Box> boxes; // 预测结果中的各个目标框
  int mask_resolution;
  std::string type = "det";
}

// 语义分割模型预测结果
class SegResult : public BaseResult {
 public:
  Mask<int64_t> label_map; // 预测分割中各像素的类别
  Mask<float> score_map; // 预测分割中各像素的置信度
  std::string type = "seg";
}

struct Box {
  int category_id; // 类别id
  std::string category; // 类别标签
  float score; // 置信度
  std::vector<float> coordinate; // 4个元素值，表示xmin, ymin, width, height
  Mask<int> mask; // 实例分割中，用于表示Box内的分割结果
}

struct Mask {
  std::vector<T> data; // 分割中的label map或score map
  std::vector<int> shape; // 表示分割图的shape
}
```

## 预测结果可视化

### 目标检测/实例分割结果可视化
```
PaddleX::Visualize(const cv::Mat& img, // 原图
				   const DetResult& result, // 预测结果
				   const std::map<int, std::string>& labels // 各类别信息<id, label_name>
				  )
```
返回cv::Mat结构体，即为可视化后的结果

### 语义分割结果可视化
```
PaddleX::Visualize(const cv::Mat& img, // 原图
				   const SegResult& result, // 预测结果
                   const std::map<int, std::string>& labels // 各类别信息<id, label_name>
                  )
```
返回cv::Mat结构体，即为可视化后的结果


## 代码示例

- 图像分类 [PaddleX/deploy/cpp/demo/classifier.cpp](https://github.com/PaddlePaddle/PaddleX/tree/release/1.3/deploy/cpp/demo/classifier.cpp)  
- 目标检测/实例分割 [PaddleX/deploy/cpp/demo/detector.cpp](https://github.com/PaddlePaddle/PaddleX/tree/release/1.3/deploy/cpp/demo/detector.cpp)
- 语义分割 [PaddleX/deploy/cpp/demo/segmenter.cpp](https://github.com/PaddlePaddle/PaddleX/tree/release/1.3/deploy/cpp/demo/segmenter.cpp)
