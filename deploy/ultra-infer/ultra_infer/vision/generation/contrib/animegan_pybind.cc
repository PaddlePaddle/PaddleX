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
void BindAnimeGAN(pybind11::module &m) {
  pybind11::class_<vision::generation::AnimeGAN, UltraInferModel>(m, "AnimeGAN")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::generation::AnimeGAN &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             cv::Mat res;
             self.Predict(mat, &res);
             auto ret = pybind11::array_t<unsigned char>(
                 {res.rows, res.cols, res.channels()}, res.data);
             return ret;
           })
      .def("batch_predict",
           [](vision::generation::AnimeGAN &self,
              std::vector<pybind11::array> &data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<cv::Mat> results;
             self.BatchPredict(images, &results);
             std::vector<pybind11::array_t<unsigned char>> ret;
             for (size_t i = 0; i < results.size(); ++i) {
               ret.push_back(pybind11::array_t<unsigned char>(
                   {results[i].rows, results[i].cols, results[i].channels()},
                   results[i].data));
             }
             return ret;
           })
      .def_property_readonly("preprocessor",
                             &vision::generation::AnimeGAN::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::generation::AnimeGAN::GetPostprocessor);

  pybind11::class_<vision::generation::AnimeGANPreprocessor>(
      m, "AnimeGANPreprocessor")
      .def(pybind11::init<>())
      .def("run", [](vision::generation::AnimeGANPreprocessor &self,
                     std::vector<pybind11::array> &im_list) {
        std::vector<vision::FDMat> images;
        for (size_t i = 0; i < im_list.size(); ++i) {
          images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
        }
        std::vector<FDTensor> outputs;
        if (!self.Run(images, &outputs)) {
          throw std::runtime_error(
              "Failed to preprocess the input data in PaddleClasPreprocessor.");
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
          outputs[i].StopSharing();
        }
        return outputs;
      });
  pybind11::class_<vision::generation::AnimeGANPostprocessor>(
      m, "AnimeGANPostprocessor")
      .def(pybind11::init<>())
      .def("run", [](vision::generation::AnimeGANPostprocessor &self,
                     std::vector<FDTensor> &inputs) {
        std::vector<cv::Mat> results;
        if (!self.Run(inputs, &results)) {
          throw std::runtime_error("Failed to postprocess the runtime result "
                                   "in YOLOv5Postprocessor.");
        }
        return results;
      });
}
} // namespace ultra_infer
