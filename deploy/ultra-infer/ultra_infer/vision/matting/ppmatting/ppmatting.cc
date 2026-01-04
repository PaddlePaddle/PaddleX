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
#include "ultra_infer/vision/matting/ppmatting/ppmatting.h"

#include "ultra_infer/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"

namespace ultra_infer {
namespace vision {
namespace matting {

PPMatting::PPMatting(const std::string &model_file,
                     const std::string &params_file,
                     const std::string &config_file,
                     const RuntimeOption &custom_option,
                     const ModelFormat &model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::ORT, Backend::PDINFER, Backend::LITE};
  valid_gpu_backends = {Backend::PDINFER, Backend::TRT};
  valid_kunlunxin_backends = {Backend::LITE};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool PPMatting::Initialize() {
  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
    return false;
  }
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

bool PPMatting::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  processors_.push_back(std::make_shared<BGR2RGB>());
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile &e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  FDASSERT((cfg["Deploy"]["input_shape"]),
           "The yaml file should include input_shape parameters");
  // input_shape
  // b c h w
  auto input_shape = cfg["Deploy"]["input_shape"].as<std::vector<int>>();
  FDASSERT(input_shape.size() == 4,
           "The input_shape in yaml file need to be 4-dimensions, but now its "
           "dimension is %zu.",
           input_shape.size());

  is_fixed_input_shape_ = false;
  if (input_shape[2] > 0 && input_shape[3] > 0) {
    is_fixed_input_shape_ = true;
  }
  if (input_shape[2] < 0 || input_shape[3] < 0) {
    FDWARNING << "Detected dynamic input shape of your model, only Paddle "
                 "Inference / OpenVINO support this model now."
              << std::endl;
  }
  if (cfg["Deploy"]["transforms"]) {
    auto preprocess_cfg = cfg["Deploy"]["transforms"];
    int long_size = -1;
    for (const auto &op : preprocess_cfg) {
      FDASSERT(op.IsMap(),
               "Require the transform information in yaml be Map type.");
      if (op["type"].as<std::string>() == "LimitShort") {
        int max_short = op["max_short"] ? op["max_short"].as<int>() : -1;
        int min_short = op["min_short"] ? op["min_short"].as<int>() : -1;
        if (is_fixed_input_shape_) {
          // if the input shape is fixed, will resize by scale, and the max
          // shape will not exceed input_shape
          long_size = max_short;
          std::vector<int> max_size = {input_shape[2], input_shape[3]};
          processors_.push_back(
              std::make_shared<ResizeByShort>(long_size, 1, true, max_size));
        } else {
          processors_.push_back(
              std::make_shared<LimitShort>(max_short, min_short));
        }
      } else if (op["type"].as<std::string>() == "ResizeToIntMult") {
        if (is_fixed_input_shape_) {
          std::vector<int> max_size = {input_shape[2], input_shape[3]};
          processors_.push_back(
              std::make_shared<ResizeByShort>(long_size, 1, true, max_size));
        } else {
          int mult_int = op["mult_int"] ? op["mult_int"].as<int>() : 32;
          processors_.push_back(std::make_shared<LimitByStride>(mult_int));
        }
      } else if (op["type"].as<std::string>() == "Normalize") {
        std::vector<float> mean = {0.5, 0.5, 0.5};
        std::vector<float> std = {0.5, 0.5, 0.5};
        if (op["mean"]) {
          mean = op["mean"].as<std::vector<float>>();
        }
        if (op["std"]) {
          std = op["std"].as<std::vector<float>>();
        }
        processors_.push_back(std::make_shared<Normalize>(mean, std));
      } else if (op["type"].as<std::string>() == "ResizeByShort") {
        long_size = op["short_size"].as<int>();
        if (is_fixed_input_shape_) {
          std::vector<int> max_size = {input_shape[2], input_shape[3]};
          processors_.push_back(
              std::make_shared<ResizeByShort>(long_size, 1, true, max_size));
        } else {
          processors_.push_back(std::make_shared<ResizeByShort>(long_size));
        }
      }
    }
    // the default padding value is {127.5,127.5,127.5} so after normalizing,
    // ((127.5/255)-0.5)/0.5 = 0.0
    std::vector<float> value = {0.0, 0.0, 0.0};
    processors_.push_back(std::make_shared<Cast>("float"));
    processors_.push_back(
        std::make_shared<PadToSize>(input_shape[3], input_shape[2], value));
    processors_.push_back(std::make_shared<HWC2CHW>());
  }

  return true;
}

bool PPMatting::Preprocess(Mat *mat, FDTensor *output,
                           std::map<std::string, std::array<int, 2>> *im_info) {
  (*im_info)["input_shape"] = {mat->Height(), mat->Width()};
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }
  (*im_info)["output_shape"] = {mat->Height(), mat->Width()};
  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);
  output->name = InputInfoOfRuntime(0).name;
  return true;
}

bool PPMatting::Postprocess(
    std::vector<FDTensor> &infer_result, MattingResult *result,
    const std::map<std::string, std::array<int, 2>> &im_info) {
  FDASSERT((infer_result.size() == 1),
           "The default number of output tensor must be 1 ");
  FDTensor &alpha_tensor = infer_result.at(0); // (1, 1, h, w)
  FDASSERT((alpha_tensor.shape[0] == 1), "Only support batch = 1 now.");
  if (alpha_tensor.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }
  std::vector<int64_t> dim{0, 2, 3, 1};
  function::Transpose(alpha_tensor, &alpha_tensor, dim);
  alpha_tensor.Squeeze(0);
  Mat mat = Mat::Create(alpha_tensor);

  auto iter_ipt = im_info.find("input_shape");
  auto iter_out = im_info.find("output_shape");
  if (is_fixed_input_shape_) {
    double scale_h = static_cast<double>(iter_out->second[0]) /
                     static_cast<double>(iter_ipt->second[0]);
    double scale_w = static_cast<double>(iter_out->second[1]) /
                     static_cast<double>(iter_ipt->second[1]);
    double actual_scale = std::min(scale_h, scale_w);

    int size_before_pad_h = round(actual_scale * iter_ipt->second[0]);
    int size_before_pad_w = round(actual_scale * iter_ipt->second[1]);

    Crop::Run(&mat, 0, 0, size_before_pad_w, size_before_pad_h);
  }

  Resize::Run(&mat, iter_ipt->second[1], iter_ipt->second[0], -1.0f, -1.0f, 1,
              false, ProcLib::OPENCV);

  result->Clear();
  // note: must be setup shape before Resize
  result->contain_foreground = false;
  result->shape = {iter_ipt->second[0], iter_ipt->second[1]};
  int numel = iter_ipt->second[0] * iter_ipt->second[1];
  int nbytes = numel * sizeof(float);
  result->Resize(numel);
  std::memcpy(result->alpha.data(), mat.Data(), nbytes);
  return true;
}

bool PPMatting::Predict(cv::Mat *im, MattingResult *result) {
  Mat mat(*im);
  std::vector<FDTensor> processed_data(1);

  std::map<std::string, std::array<int, 2>> im_info;

  if (!Preprocess(&mat, &(processed_data[0]), &im_info)) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }
  std::vector<FDTensor> infer_result(1);
  if (!Infer(processed_data, &infer_result)) {
    FDERROR << "Failed to inference while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  if (!Postprocess(infer_result, result, im_info)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

} // namespace matting
} // namespace vision
} // namespace ultra_infer
