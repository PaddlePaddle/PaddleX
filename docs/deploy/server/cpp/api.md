# C++ code interface description

## Header file
`include/paddlex/paddlex.h`

## Class PaddleX::Model

Model class is used to load models trained by PaddleX.

### Complete model loading
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

**Parameters**
- model_dir: path to the model directory
- use_gpu: whether to use GPU for prediction
- use_trt: whether or not to TensorRT
- use_mkl: whether or not to use MKLDNN to accelerate the predicted performance of the model on the CPU
- mkl_thread_num: the number of threads when MKLDNN is used
- gpu_id: ID of the GPU
- key: model decryption key. It is used when the encrypted PaddleX model is loaded.
- use_ir_optim: whether to speed up the model after image optimization

**Returned value**
- Returns true or false, indicating whether the model is loaded successfully.

### Model prediction inference

**Single image prediction of category model**
```
PaddleX::Model::predict(const cv::Mat& im, ClsResult* result)
```
**Batch prediction for multiple pictures in category model**
```
PaddleX::Model::predict(const std::vector<cv::Mat>& im_batch, std::vector<ClsResult>* results)
```
**Object detection/instance segmentation model single picture prediction**
```
PaddleX::Model::predict(const cv::Mat& im, DetResult* result)
```
**Object detection/instance segmentation model batch prediction for multiple images**
```
PaddleX::Model::predict(const std::vector<cv::Mat>& im_batch, std::vector<DetResult>* results)
```
**Single image prediction for semantic segmentation model**
```
PaddleX::Model::predict(const cv::Mat& im, SegResult* result)
```
**Multiple image batch prediction for semantic segmentation model**
```
PaddleX::Model::predict(const std::vector<cv::Mat>& im_batch, std::vector<SegResult>* results)
```
The return value of each interface is true or false, indicating whether the prediction is successful or not.

In the prediction, the cv::Mat structure should be passed in, and the structure should be identical to the one loaded by the following code
```
cv::Mat im = cv::imread('test.jpg', 1);
```
In the use of batch prediction, it should be noted that all the data in the incoming vector is predicted as a batch. Therefore, the larger the vector, the higher the GPU video memory required.

In the prediction, the ClsResult/DetResult/SegResult structure is passed in at the same time to store the prediction result of the model. The description of each structure is as follows:
```
// Prediction results of category models
class ClsResult {
public:
 int category_id; //Category_id
 std::string category; //Category label
 float score; // Prediction confidence level
 std::string type = "cls";
} 

//Prediction results of object detection/instance segmentation model
class DetResult {
 public:
 std::vector<Box> boxes; // Each object box in the prediction result
 int mask_resolution; std::string type = "det"; 
} 

// Prediction results of semantic segmentation model
class SegResult :public BaseResult {
 public:
  Mask<int64_t> label_map; // Category of each pixel in the prediction segmentation
  Mask<float> score_map; // Confidence level of each pixel in the prediction segmentation
  std::string type = "seg"; 
}

struct Box {
  int category_id; //Category_id
  std::string category; //Category label
  float score; // Confidence level
  std::vector<float> coordinate; // 4 element values, indicating xmin, ymin, width, height
  Mask<int> mask; // In instance segmentation, it represents the segmentation result of inside Box
} 

struct Mask {
  std::vector<T> data;// the label map or score map in the segmentation
  std::vector<int> shape; // represents the shape of the segmented graph.
 }
```

## Visualization of predicted results

### Object detection/instance segmentation result visualization
```
PaddleX::Visualize(const cv::Mat& img, // original image)
                                                    const DetResult& result, // prediction result
                                                    const std::map<int, std::string>& labels // each class of info <id, label_name>
                                                    )
```
Returns the cv::Mat structure, that is, the result of visualization

### Visualization of semantic segmentation results
```
PaddleX::Visualize(const cv::Mat& img, // original image
                                                   const SegResult& result, // prediction result
                   const std::map<int, std::string>& labels // each class of info <id, label_name>
                    )
```
Returns the cv::Mat structure, that is, the result of visualization


## Code example:

- Image classification
[PaddleX/deploy/cpp/demo/classifier.cpp](https://github.com/PaddlePaddle/PaddleX/blob/develop/deploy/cpp/demo/classifier.cpp)  
- Object detection/Instance segmentation
[PaddleX/deploy/cpp/demo/detector.cpp ](https://github.com/PaddlePaddle/PaddleX/blob/develop/deploy/cpp/demo/detector.cpp)
- Semantic segmentation
[PaddleX/deploy/cpp/demo/segmenter.cpp ](https://github.com/PaddlePaddle/PaddleX/blob/develop/deploy/cpp/demo/segmenter.cpp)
