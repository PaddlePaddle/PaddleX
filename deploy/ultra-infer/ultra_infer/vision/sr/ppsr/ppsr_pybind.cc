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
void BindPPSR(pybind11::module &m) {
  pybind11::class_<vision::sr::PPMSVSR, UltraInferModel>(m, "PPMSVSR")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::sr::PPMSVSR &self, std::vector<pybind11::array> &datas) {
             std::vector<cv::Mat> inputs;
             for (auto &data : datas) {
               auto mat = PyArrayToCvMat(data);
               inputs.push_back(mat);
             }
             std::vector<cv::Mat> res;
             std::vector<pybind11::array> res_pyarray;
             self.Predict(inputs, res);
             for (auto &img : res) {
               auto ret = pybind11::array_t<unsigned char>(
                   {img.rows, img.cols, img.channels()}, img.data);
               res_pyarray.push_back(ret);
             }
             return res_pyarray;
           });
  pybind11::class_<vision::sr::EDVR, UltraInferModel>(m, "EDVR")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::sr::EDVR &self, std::vector<pybind11::array> &datas) {
             std::vector<cv::Mat> inputs;
             for (auto &data : datas) {
               auto mat = PyArrayToCvMat(data);
               inputs.push_back(mat);
             }
             std::vector<cv::Mat> res;
             std::vector<pybind11::array> res_pyarray;
             self.Predict(inputs, res);
             for (auto &img : res) {
               auto ret = pybind11::array_t<unsigned char>(
                   {img.rows, img.cols, img.channels()}, img.data);
               res_pyarray.push_back(ret);
             }
             return res_pyarray;
           });
  pybind11::class_<vision::sr::BasicVSR, UltraInferModel>(m, "BasicVSR")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::sr::BasicVSR &self, std::vector<pybind11::array> &datas) {
             std::vector<cv::Mat> inputs;
             for (auto &data : datas) {
               auto mat = PyArrayToCvMat(data);
               inputs.push_back(mat);
             }
             std::vector<cv::Mat> res;
             std::vector<pybind11::array> res_pyarray;
             self.Predict(inputs, res);
             for (auto &img : res) {
               auto ret = pybind11::array_t<unsigned char>(
                   {img.rows, img.cols, img.channels()}, img.data);
               res_pyarray.push_back(ret);
             }
             return res_pyarray;
           });
}
} // namespace ultra_infer
