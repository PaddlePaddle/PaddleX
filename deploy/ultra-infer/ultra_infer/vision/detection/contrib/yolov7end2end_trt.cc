// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "ultra_infer/vision/detection/contrib/yolov7end2end_trt.h"

#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/utils/utils.h"
#ifdef WITH_GPU
#include "ultra_infer/vision/utils/cuda_utils.h"
#endif // WITH_GPU

namespace ultra_infer {
namespace vision {
namespace detection {

void YOLOv7End2EndTRT::LetterBox(Mat *mat, const std::vector<int> &size,
                                 const std::vector<float> &color, bool _auto,
                                 bool scale_fill, bool scale_up, int stride) {
  float scale =
      std::min(size[1] * 1.0 / mat->Height(), size[0] * 1.0 / mat->Width());
  if (!scale_up) {
    scale = std::min(scale, 1.0f);
  }

  int resize_h = int(round(mat->Height() * scale));
  int resize_w = int(round(mat->Width() * scale));

  int pad_w = size[0] - resize_w;
  int pad_h = size[1] - resize_h;
  if (_auto) {
    pad_h = pad_h % stride;
    pad_w = pad_w % stride;
  } else if (scale_fill) {
    pad_h = 0;
    pad_w = 0;
    resize_h = size[1];
    resize_w = size[0];
  }
  if (resize_h != mat->Height() || resize_w != mat->Width()) {
    Resize::Run(mat, resize_w, resize_h);
  }
  if (pad_h > 0 || pad_w > 0) {
    float half_h = pad_h * 1.0 / 2;
    int top = int(round(half_h - 0.1));
    int bottom = int(round(half_h + 0.1));
    float half_w = pad_w * 1.0 / 2;
    int left = int(round(half_w - 0.1));
    int right = int(round(half_w + 0.1));
    Pad::Run(mat, top, bottom, left, right, color);
  }
}

YOLOv7End2EndTRT::YOLOv7End2EndTRT(const std::string &model_file,
                                   const std::string &params_file,
                                   const RuntimeOption &custom_option,
                                   const ModelFormat &model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {};             // NO CPU
    valid_gpu_backends = {Backend::TRT}; // NO ORT
  } else {
    valid_cpu_backends = {Backend::PDINFER};
    valid_gpu_backends = {Backend::PDINFER};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  if (runtime_option.device != Device::GPU) {
    FDWARNING << runtime_option.device
              << " is not support for YOLOv7End2EndTRT,"
              << "will fallback to Device::GPU." << std::endl;
    runtime_option.device = Device::GPU;
  }
  if (runtime_option.backend != Backend::UNKNOWN) {
    if (runtime_option.backend != Backend::TRT) {
      FDWARNING << runtime_option.backend
                << " is not support for YOLOv7End2EndTRT,"
                << "will fallback to Backend::TRT." << std::endl;
      runtime_option.backend = Backend::TRT;
    }
  }
#ifdef WITH_GPU
  cudaSetDevice(runtime_option.device_id);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cuda_stream_ = reinterpret_cast<void *>(stream);
  runtime_option.SetExternalStream(cuda_stream_);
#endif // WITH_GPU
  initialized = Initialize();
}

bool YOLOv7End2EndTRT::Initialize() {
  // parameters for preprocess
  size = {640, 640};
  padding_value = {114.0, 114.0, 114.0};
  is_mini_pad = false;
  is_no_pad = false;
  is_scale_up = false;
  stride = 32;
  reused_input_tensors_.resize(1);

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  // Check if the input shape is dynamic after Runtime already initialized,
  // Note that, We need to force is_mini_pad 'false' to keep static
  // shape after padding (LetterBox) when the is_dynamic_shape is 'false'.
  is_dynamic_input_ = false;
  auto shape = InputInfoOfRuntime(0).shape;
  for (int i = 0; i < shape.size(); ++i) {
    // if height or width is dynamic
    if (i >= 2 && shape[i] <= 0) {
      is_dynamic_input_ = true;
      break;
    }
  }
  if (!is_dynamic_input_) {
    is_mini_pad = false;
  }
  return true;
}

YOLOv7End2EndTRT::~YOLOv7End2EndTRT() {
#ifdef WITH_GPU
  if (use_cuda_preprocessing_) {
    CUDA_CHECK(cudaFreeHost(input_img_cuda_buffer_host_));
    CUDA_CHECK(cudaFree(input_img_cuda_buffer_device_));
    CUDA_CHECK(cudaFree(input_tensor_cuda_buffer_device_));
    CUDA_CHECK(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(cuda_stream_)));
  }
#endif // WITH_GPU
}

bool YOLOv7End2EndTRT::Preprocess(
    Mat *mat, FDTensor *output,
    std::map<std::string, std::array<float, 2>> *im_info) {
  float ratio = std::min(size[1] * 1.0f / static_cast<float>(mat->Height()),
                         size[0] * 1.0f / static_cast<float>(mat->Width()));
  if (std::fabs(ratio - 1.0f) > 1e-06) {
    int interp = cv::INTER_AREA;
    if (ratio > 1.0) {
      interp = cv::INTER_LINEAR;
    }
    int resize_h = int(mat->Height() * ratio);
    int resize_w = int(mat->Width() * ratio);
    Resize::Run(mat, resize_w, resize_h, -1, -1, interp);
  }
  YOLOv7End2EndTRT::LetterBox(mat, size, padding_value, is_mini_pad, is_no_pad,
                              is_scale_up, stride);
  BGR2RGB::Run(mat);
  std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  std::vector<float> beta = {0.0f, 0.0f, 0.0f};
  Convert::Run(mat, alpha, beta);

  (*im_info)["output_shape"] = {static_cast<float>(mat->Height()),
                                static_cast<float>(mat->Width())};

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");
  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1); // reshape to n, c, h, w
  return true;
}

void YOLOv7End2EndTRT::UseCudaPreprocessing(int max_image_size) {
#ifdef WITH_GPU
  use_cuda_preprocessing_ = true;
  is_scale_up = true;
  if (input_img_cuda_buffer_host_ == nullptr) {
    // prepare input data cache in GPU pinned memory
    CUDA_CHECK(cudaMallocHost((void **)&input_img_cuda_buffer_host_,
                              max_image_size * 3));
    // prepare input data cache in GPU device memory
    CUDA_CHECK(cudaMalloc((void **)&input_img_cuda_buffer_device_,
                          max_image_size * 3));
    CUDA_CHECK(cudaMalloc((void **)&input_tensor_cuda_buffer_device_,
                          3 * size[0] * size[1] * sizeof(float)));
  }
#else
  FDWARNING << "The UltraInfer didn't compile with WITH_GPU=ON." << std::endl;
  use_cuda_preprocessing_ = false;
#endif
}

bool YOLOv7End2EndTRT::CudaPreprocess(
    Mat *mat, FDTensor *output,
    std::map<std::string, std::array<float, 2>> *im_info) {
#ifdef WITH_GPU
  if (is_mini_pad != false || is_no_pad != false || is_scale_up != true) {
    FDERROR << "Preprocessing with CUDA is only available when the arguments "
               "satisfy (is_mini_pad=false, is_no_pad=false, is_scale_up=true)."
            << std::endl;
    return false;
  }

  // Record the shape of image and the shape of preprocessed image
  (*im_info)["input_shape"] = {static_cast<float>(mat->Height()),
                               static_cast<float>(mat->Width())};
  (*im_info)["output_shape"] = {static_cast<float>(mat->Height()),
                                static_cast<float>(mat->Width())};

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  int src_img_buf_size = mat->Height() * mat->Width() * mat->Channels();
  memcpy(input_img_cuda_buffer_host_, mat->Data(), src_img_buf_size);
  CUDA_CHECK(cudaMemcpyAsync(input_img_cuda_buffer_device_,
                             input_img_cuda_buffer_host_, src_img_buf_size,
                             cudaMemcpyHostToDevice, stream));
  utils::CudaYoloPreprocess(input_img_cuda_buffer_device_, mat->Width(),
                            mat->Height(), input_tensor_cuda_buffer_device_,
                            size[0], size[1], padding_value, stream);

  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {static_cast<float>(size[0]),
                                static_cast<float>(size[1])};

  output->SetExternalData({mat->Channels(), size[0], size[1]}, FDDataType::FP32,
                          input_tensor_cuda_buffer_device_);
  output->device = Device::GPU;
  output->shape.insert(output->shape.begin(), 1); // reshape to n, c, h, w
  return true;
#else
  FDERROR << "CUDA src code was not enabled." << std::endl;
  return false;
#endif // WITH_GPU
}

bool YOLOv7End2EndTRT::Postprocess(
    std::vector<FDTensor> &infer_results, DetectionResult *result,
    const std::map<std::string, std::array<float, 2>> &im_info,
    float conf_threshold) {
  FDASSERT(infer_results.size() == 4, "Output tensor size must be 4.");
  FDTensor &num_tensor = infer_results.at(0);     // INT32
  FDTensor &boxes_tensor = infer_results.at(1);   // FLOAT
  FDTensor &scores_tensor = infer_results.at(2);  // FLOAT
  FDTensor &classes_tensor = infer_results.at(3); // INT32
  FDASSERT(num_tensor.dtype == FDDataType::INT32,
           "The dtype of num_dets must be INT32.");
  FDASSERT(boxes_tensor.dtype == FDDataType::FP32,
           "The dtype of det_boxes_tensor must be FP32.");
  FDASSERT(scores_tensor.dtype == FDDataType::FP32,
           "The dtype of det_scores_tensor must be FP32.");
  FDASSERT(classes_tensor.dtype == FDDataType::INT32,
           "The dtype of det_classes_tensor must be INT32.");
  FDASSERT(num_tensor.shape[0] == 1, "Only support batch=1 now.");
  // post-process for end2end yolov7 after trt nms.
  float *boxes_data = static_cast<float *>(boxes_tensor.Data());   // (1,100,4)
  float *scores_data = static_cast<float *>(scores_tensor.Data()); // (1,100)
  int32_t *classes_data =
      static_cast<int32_t *>(classes_tensor.Data()); // (1,100)
  int32_t num_dets_after_trt_nms = static_cast<int32_t *>(num_tensor.Data())[0];
  if (num_dets_after_trt_nms == 0) {
    return true;
  }
  result->Clear();
  result->Reserve(num_dets_after_trt_nms);
  for (size_t i = 0; i < num_dets_after_trt_nms; ++i) {
    float confidence = scores_data[i];
    if (confidence <= conf_threshold) {
      continue;
    }
    int32_t label_id = classes_data[i];
    float x1 = boxes_data[(i * 4) + 0];
    float y1 = boxes_data[(i * 4) + 1];
    float x2 = boxes_data[(i * 4) + 2];
    float y2 = boxes_data[(i * 4) + 3];

    result->boxes.emplace_back(std::array<float, 4>{x1, y1, x2, y2});
    result->label_ids.push_back(label_id);
    result->scores.push_back(confidence);
  }

  if (result->boxes.size() == 0) {
    return true;
  }

  // scale the boxes to the origin image shape
  auto iter_out = im_info.find("output_shape");
  auto iter_ipt = im_info.find("input_shape");
  FDASSERT(iter_out != im_info.end() && iter_ipt != im_info.end(),
           "Cannot find input_shape or output_shape from im_info.");
  float out_h = iter_out->second[0];
  float out_w = iter_out->second[1];
  float ipt_h = iter_ipt->second[0];
  float ipt_w = iter_ipt->second[1];
  float scale = std::min(out_h / ipt_h, out_w / ipt_w);
  float pad_h = (out_h - ipt_h * scale) / 2.0f;
  float pad_w = (out_w - ipt_w * scale) / 2.0f;
  if (is_mini_pad) {
    pad_h = static_cast<float>(static_cast<int>(pad_h) % stride);
    pad_w = static_cast<float>(static_cast<int>(pad_w) % stride);
  }
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    int32_t label_id = (result->label_ids)[i];
    result->boxes[i][0] = std::max((result->boxes[i][0] - pad_w) / scale, 0.0f);
    result->boxes[i][1] = std::max((result->boxes[i][1] - pad_h) / scale, 0.0f);
    result->boxes[i][2] = std::max((result->boxes[i][2] - pad_w) / scale, 0.0f);
    result->boxes[i][3] = std::max((result->boxes[i][3] - pad_h) / scale, 0.0f);
    result->boxes[i][0] = std::min(result->boxes[i][0], ipt_w - 1.0f);
    result->boxes[i][1] = std::min(result->boxes[i][1], ipt_h - 1.0f);
    result->boxes[i][2] = std::min(result->boxes[i][2], ipt_w - 1.0f);
    result->boxes[i][3] = std::min(result->boxes[i][3], ipt_h - 1.0f);
  }
  return true;
}

bool YOLOv7End2EndTRT::Predict(cv::Mat *im, DetectionResult *result,
                               float conf_threshold) {
  Mat mat(*im);

  std::map<std::string, std::array<float, 2>> im_info;

  // Record the shape of image and the shape of preprocessed image
  im_info["input_shape"] = {static_cast<float>(mat.Height()),
                            static_cast<float>(mat.Width())};
  im_info["output_shape"] = {static_cast<float>(mat.Height()),
                             static_cast<float>(mat.Width())};

  if (use_cuda_preprocessing_) {
    if (!CudaPreprocess(&mat, &reused_input_tensors_[0], &im_info)) {
      FDERROR << "Failed to preprocess input image." << std::endl;
      return false;
    }
  } else {
    if (!Preprocess(&mat, &reused_input_tensors_[0], &im_info)) {
      FDERROR << "Failed to preprocess input image." << std::endl;
      return false;
    }
  }

  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  if (!Infer()) {
    FDERROR << "Failed to inference." << std::endl;
    return false;
  }

  if (!Postprocess(reused_output_tensors_, result, im_info, conf_threshold)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }

  return true;
}

} // namespace detection
} // namespace vision
} // namespace ultra_infer
