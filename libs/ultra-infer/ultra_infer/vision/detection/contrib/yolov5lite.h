// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.  //NOLINT
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "ultra_infer/ultra_infer_model.h"
#include "ultra_infer/vision/common/processors/transform.h"
#include "ultra_infer/vision/common/result.h"

namespace ultra_infer {
namespace vision {
namespace detection {
/*! @brief YOLOv5Lite model object used when to load a YOLOv5Lite model exported
 * by YOLOv5Lite.
 */
class ULTRAINFER_DECL YOLOv5Lite : public UltraInferModel {
public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./yolov5lite.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in "valid_cpu_backends" \param[in] model_format
   * Model format of the loaded model, default is ONNX format
   */
  YOLOv5Lite(const std::string &model_file, const std::string &params_file = "",
             const RuntimeOption &custom_option = RuntimeOption(),
             const ModelFormat &model_format = ModelFormat::ONNX);

  ~YOLOv5Lite();

  virtual std::string ModelName() const { return "YOLOv5-Lite"; }
  /** \brief Predict the detection result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format \param[in] result The output detection result
   * will be written to this structure \param[in] conf_threshold confidence
   * threashold for postprocessing, default is 0.45 \param[in] nms_iou_threshold
   * iou threshold for NMS, default is 0.25 \return true if the prediction
   * successed, otherwise false
   */
  virtual bool Predict(cv::Mat *im, DetectionResult *result,
                       float conf_threshold = 0.45,
                       float nms_iou_threshold = 0.25);

  void UseCudaPreprocessing(int max_img_size = 3840 * 2160);

  /*! @brief
  Argument for image preprocessing step, tuple of (width, height), decide the
  target size after resize, size = {640, 640}
  */
  std::vector<int> size;
  // padding value, size should be the same as channels

  std::vector<float> padding_value;
  // only pad to the minimum rectangle which height and width is times of stride
  bool is_mini_pad;
  // while is_mini_pad = false and is_no_pad = true,
  // will resize the image to the set size
  bool is_no_pad;
  // if is_scale_up is false, the input image only can be zoom out,
  // the maximum resize scale cannot exceed 1.0
  bool is_scale_up;
  // padding stride, for is_mini_pad
  int stride;
  // for offsetting the boxes by classes when using NMS
  float max_wh;
  // downsample strides for YOLOv5Lite to generate anchors,
  // will take (8,16,32) as default values, might have stride=64.
  std::vector<int> downsample_strides;
  // anchors parameters, downsample_strides will take (8,16,32),
  // each stride has three anchors with width and height
  std::vector<std::vector<float>> anchor_config;
  /*! @brief
    whether the model_file was exported with decode module. The official
    YOLOv5Lite/export.py script will export ONNX file without
    decode module. Please set it 'true' manually if the model file
    was exported with decode module.
    false : ONNX files without decode module.
    true : ONNX file with decode module. default false.
  */
  bool is_decode_exported;

private:
  // necessary parameters for GenerateAnchors to generate anchors when ONNX file
  // without decode module.
  struct Anchor {
    int grid0;
    int grid1;
    int stride;
    float anchor_w;
    float anchor_h;
  };

  bool Initialize();

  bool Preprocess(Mat *mat, FDTensor *output,
                  std::map<std::string, std::array<float, 2>> *im_info);

  bool CudaPreprocess(Mat *mat, FDTensor *output,
                      std::map<std::string, std::array<float, 2>> *im_info);

  bool Postprocess(FDTensor &infer_result, DetectionResult *result,
                   const std::map<std::string, std::array<float, 2>> &im_info,
                   float conf_threshold, float nms_iou_threshold);

  // the official YOLOv5Lite/export.py will export ONNX file without decode
  // module.
  // this function support the postporocess for ONNX file without decode module.
  // set the `is_decode_exported = false`, this function will work.
  bool PostprocessWithDecode(
      FDTensor &infer_result, DetectionResult *result,
      const std::map<std::string, std::array<float, 2>> &im_info,
      float conf_threshold, float nms_iou_threshold);

  void LetterBox(Mat *mat, const std::vector<int> &size,
                 const std::vector<float> &color, bool _auto,
                 bool scale_fill = false, bool scale_up = true,
                 int stride = 32);

  // generate anchors for decoding when ONNX file without decode module.
  void GenerateAnchors(const std::vector<int> &size,
                       const std::vector<int> &downsample_strides,
                       std::vector<Anchor> *anchors, const int num_anchors = 3);

  // whether to inference with dynamic shape (e.g ONNX export with dynamic shape
  // or not.)
  // while is_dynamic_shape if 'false', is_mini_pad will force 'false'. This
  // value will
  // auto check by ultra_infer after the internal Runtime already initialized.
  bool is_dynamic_input_;
  // CUDA host buffer for input image
  uint8_t *input_img_cuda_buffer_host_ = nullptr;
  // CUDA device buffer for input image
  uint8_t *input_img_cuda_buffer_device_ = nullptr;
  // CUDA device buffer for TRT input tensor
  float *input_tensor_cuda_buffer_device_ = nullptr;
  // Whether to use CUDA preprocessing
  bool use_cuda_preprocessing_ = false;
  // CUDA stream
  void *cuda_stream_ = nullptr;
};
} // namespace detection
} // namespace vision
} // namespace ultra_infer
