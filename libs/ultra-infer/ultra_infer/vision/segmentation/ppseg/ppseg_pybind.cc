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
#include "ultra_infer/pybind/main.h"

namespace ultra_infer {
void BindPPSeg(pybind11::module &m) {
  pybind11::class_<vision::segmentation::PaddleSegPreprocessor,
                   vision::ProcessorManager>(m, "PaddleSegPreprocessor")
      .def(pybind11::init<std::string>())
      .def("run",
           [](vision::segmentation::PaddleSegPreprocessor &self,
              std::vector<pybind11::array> &im_list) {
             std::vector<vision::FDMat> images;
             for (size_t i = 0; i < im_list.size(); ++i) {
               images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
             }
             // Record the shape of input images
             std::map<std::string, std::vector<std::array<int, 2>>> imgs_info;
             std::vector<FDTensor> outputs;
             self.SetImgsInfo(&imgs_info);
             if (!self.Run(&images, &outputs)) {
               throw std::runtime_error(
                   "Failed to preprocess the input data in "
                   "PaddleSegPreprocessor.");
             }
             for (size_t i = 0; i < outputs.size(); ++i) {
               outputs[i].StopSharing();
             }
             return make_pair(outputs, imgs_info);
             ;
           })
      .def("disable_normalize",
           [](vision::segmentation::PaddleSegPreprocessor &self) {
             self.DisableNormalize();
           })
      .def("disable_permute",
           [](vision::segmentation::PaddleSegPreprocessor &self) {
             self.DisablePermute();
           })
      .def_property(
          "is_vertical_screen",
          &vision::segmentation::PaddleSegPreprocessor::GetIsVerticalScreen,
          &vision::segmentation::PaddleSegPreprocessor::SetIsVerticalScreen);

  pybind11::class_<vision::segmentation::PaddleSegModel, UltraInferModel>(
      m, "PaddleSegModel")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("clone",
           [](vision::segmentation::PaddleSegModel &self) {
             return self.Clone();
           })
      .def("predict",
           [](vision::segmentation::PaddleSegModel &self,
              pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::SegmentationResult res;
             self.Predict(&mat, &res);
             return res;
           })
      .def("batch_predict",
           [](vision::segmentation::PaddleSegModel &self,
              std::vector<pybind11::array> &data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<vision::SegmentationResult> results;
             self.BatchPredict(images, &results);
             return results;
           })
      .def_property_readonly(
          "preprocessor",
          &vision::segmentation::PaddleSegModel::GetPreprocessor)
      .def_property_readonly(
          "postprocessor",
          &vision::segmentation::PaddleSegModel::GetPostprocessor);

  pybind11::class_<vision::segmentation::PaddleSegPostprocessor>(
      m, "PaddleSegPostprocessor")
      .def(pybind11::init<std::string>())
      .def("run",
           [](vision::segmentation::PaddleSegPostprocessor &self,
              std::vector<FDTensor> &inputs,
              const std::map<std::string, std::vector<std::array<int, 2>>>
                  &imgs_info) {
             std::vector<vision::SegmentationResult> results;
             if (!self.Run(inputs, &results, imgs_info)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "PaddleSegPostprocessor.");
             }
             return results;
           })
      .def("run",
           [](vision::segmentation::PaddleSegPostprocessor &self,
              std::vector<pybind11::array> &input_array,
              const std::map<std::string, std::vector<std::array<int, 2>>>
                  &imgs_info) {
             std::vector<vision::SegmentationResult> results;
             std::vector<FDTensor> inputs;
             PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
             if (!self.Run(inputs, &results, imgs_info)) {
               throw std::runtime_error(
                   "Failed to postprocess the runtime result in "
                   "PaddleSegPostprocessor.");
             }
             return results;
           })
      .def_property(
          "apply_softmax",
          &vision::segmentation::PaddleSegPostprocessor::GetApplySoftmax,
          &vision::segmentation::PaddleSegPostprocessor::SetApplySoftmax)
      .def_property(
          "store_score_map",
          &vision::segmentation::PaddleSegPostprocessor::GetStoreScoreMap,
          &vision::segmentation::PaddleSegPostprocessor::SetStoreScoreMap);
}
} // namespace ultra_infer
