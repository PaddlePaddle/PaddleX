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
#include <pybind11/stl.h>

#include "ultra_infer/pybind/main.h"

namespace ultra_infer {
void BindPPOCRModel(pybind11::module &m) {
  m.def("sort_boxes", [](std::vector<std::array<int, 8>> &boxes) {
    vision::ocr::SortBoxes(&boxes);
    return boxes;
  });

  // UVDoc
  pybind11::class_<vision::ocr::UVDocPreprocessor, vision::ProcessorManager>(
      m, "UVDocPreprocessor")
      .def(pybind11::init<>())
      .def("set_normalize",
           [](vision::ocr::UVDocPreprocessor &self,
              const std::vector<float> &mean, const std::vector<float> &std,
              bool is_scale) { self.SetNormalize(mean, std, is_scale); })
      .def("run",
           [](vision::ocr::UVDocPreprocessor &self,
              std::vector<pybind11::array> &im_list) {
             std::vector<vision::FDMat> images;
             for (size_t i = 0; i < im_list.size(); ++i) {
               images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
             }
             std::vector<FDTensor> outputs;
             if (!self.Run(&images, &outputs)) {
               throw std::runtime_error(
                   "Failed to preprocess the input data in "
                   "UVDocPreprocessor.");
             }
             for (size_t i = 0; i < outputs.size(); ++i) {
               outputs[i].StopSharing();
             }
             return outputs;
           })
      .def(
          "disable_normalize",
          [](vision::ocr::UVDocPreprocessor &self) { self.DisableNormalize(); })
      .def("disable_permute",
           [](vision::ocr::UVDocPreprocessor &self) { self.DisablePermute(); });

  pybind11::class_<vision::ocr::UVDocPostprocessor>(m, "UVDocPostprocessor")
      .def(pybind11::init<>())
      .def("run", [](vision::ocr::UVDocPostprocessor &self,
                     std::vector<FDTensor> &inputs) {
        std::vector<FDTensor> results;
        if (!self.Run(inputs, &results)) {
          throw std::runtime_error("Failed to preprocess the input data in "
                                   "UVDocPostprocessor.");
        }
        for (size_t i = 0; i < results.size(); ++i) {
          results[i].StopSharing();
        }
        return results;
      });

  pybind11::class_<vision::ocr::UVDocWarpper, UltraInferModel>(m,
                                                               "UVDocWarpper")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def(pybind11::init<>())
      .def_property_readonly("preprocessor",
                             &vision::ocr::UVDocWarpper::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::ocr::UVDocWarpper::GetPostprocessor)
      .def("clone",
           [](vision::ocr::UVDocWarpper &self) { return self.Clone(); })
      .def("predict",
           [](vision::ocr::UVDocWarpper &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             FDTensor res;
             self.Predict(mat, &res);
             res.StopSharing();
             return res;
           })
      .def("batch_predict", [](vision::ocr::UVDocWarpper &self,
                               std::vector<pybind11::array> &data) {
        std::vector<cv::Mat> images;
        for (size_t i = 0; i < data.size(); ++i) {
          images.push_back(PyArrayToCvMat(data[i]));
        }
        std::vector<FDTensor> results;
        self.BatchPredict(images, &results);
        for (size_t i = 0; i < results.size(); ++i) {
          results[i].StopSharing();
        }
        return results;
        // std::vector<cv::Mat> results;
        // self.BatchPredict(images, &results);
        // std::vector<pybind11::array_t<unsigned char>> ret;
        // for(size_t i = 0; i < results.size(); ++i){
        //   ret.push_back(pybind11::array_t<unsigned char>(
        //            {results[i].rows, results[i].cols, results[i].channels()},
        //            results[i].data));
        // }
        // return ret;
      });

  // DBDetector
  pybind11::class_<vision::ocr::DBDetectorPreprocessor,
                   vision::ProcessorManager>(m, "DBDetectorPreprocessor")
      .def(pybind11::init<>())
      .def_property("static_shape_infer",
                    &vision::ocr::DBDetectorPreprocessor::GetStaticShapeInfer,
                    &vision::ocr::DBDetectorPreprocessor::SetStaticShapeInfer)
      .def_property("max_side_len",
                    &vision::ocr::DBDetectorPreprocessor::GetMaxSideLen,
                    &vision::ocr::DBDetectorPreprocessor::SetMaxSideLen)
      .def("set_normalize",
           [](vision::ocr::DBDetectorPreprocessor &self,
              const std::vector<float> &mean, const std::vector<float> &std,
              bool is_scale) { self.SetNormalize(mean, std, is_scale); })
      .def("run",
           [](vision::ocr::DBDetectorPreprocessor &self,
              std::vector<pybind11::array> &im_list) {
             std::vector<vision::FDMat> images;
             for (size_t i = 0; i < im_list.size(); ++i) {
               images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
             }
             std::vector<FDTensor> outputs;
             self.Run(&images, &outputs);
             auto batch_det_img_info = self.GetBatchImgInfo();
             for (size_t i = 0; i < outputs.size(); ++i) {
               outputs[i].StopSharing();
             }
             return std::make_pair(outputs, *batch_det_img_info);
           })
      .def("disable_normalize",
           [](vision::ocr::DBDetectorPreprocessor &self) {
             self.DisableNormalize();
           })
      .def("disable_permute", [](vision::ocr::DBDetectorPreprocessor &self) {
        self.DisablePermute();
      });

  pybind11::class_<vision::ocr::DBDetectorPostprocessor>(
      m, "DBDetectorPostprocessor")
      .def(pybind11::init<>())
      .def_property("det_db_thresh",
                    &vision::ocr::DBDetectorPostprocessor::GetDetDBThresh,
                    &vision::ocr::DBDetectorPostprocessor::SetDetDBThresh)
      .def_property("det_db_box_thresh",
                    &vision::ocr::DBDetectorPostprocessor::GetDetDBBoxThresh,
                    &vision::ocr::DBDetectorPostprocessor::SetDetDBBoxThresh)
      .def_property("det_db_unclip_ratio",
                    &vision::ocr::DBDetectorPostprocessor::GetDetDBUnclipRatio,
                    &vision::ocr::DBDetectorPostprocessor::SetDetDBUnclipRatio)
      .def_property("det_db_score_mode",
                    &vision::ocr::DBDetectorPostprocessor::GetDetDBScoreMode,
                    &vision::ocr::DBDetectorPostprocessor::SetDetDBScoreMode)
      .def_property("use_dilation",
                    &vision::ocr::DBDetectorPostprocessor::GetUseDilation,
                    &vision::ocr::DBDetectorPostprocessor::SetUseDilation)

      .def("run",
           [](vision::ocr::DBDetectorPostprocessor &self,
              std::vector<FDTensor> &inputs,
              const std::vector<std::array<int, 4>> &batch_det_img_info) {
             std::vector<std::vector<std::array<int, 8>>> results;

             if (!self.Run(inputs, &results, batch_det_img_info)) {
               throw std::runtime_error(
                   "Failed to preprocess the input data in "
                   "DBDetectorPostprocessor.");
             }
             return results;
           })
      .def(
          "run", [](vision::ocr::DBDetectorPostprocessor &self,
                    std::vector<pybind11::array> &input_array,
                    const std::vector<std::array<int, 4>> &batch_det_img_info) {
            std::vector<std::vector<std::array<int, 8>>> results;
            std::vector<FDTensor> inputs;
            PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
            if (!self.Run(inputs, &results, batch_det_img_info)) {
              throw std::runtime_error("Failed to preprocess the input data in "
                                       "DBDetectorPostprocessor.");
            }
            return results;
          });

  pybind11::class_<vision::ocr::DBCURVEDetectorPostprocessor>(
      m, "DBCURVEDetectorPostprocessor")
      .def(pybind11::init<>())
      .def_property("det_db_thresh",
                    &vision::ocr::DBCURVEDetectorPostprocessor::GetDetDBThresh,
                    &vision::ocr::DBCURVEDetectorPostprocessor::SetDetDBThresh)
      .def_property(
          "det_db_box_thresh",
          &vision::ocr::DBCURVEDetectorPostprocessor::GetDetDBBoxThresh,
          &vision::ocr::DBCURVEDetectorPostprocessor::SetDetDBBoxThresh)
      .def_property(
          "det_db_unclip_ratio",
          &vision::ocr::DBCURVEDetectorPostprocessor::GetDetDBUnclipRatio,
          &vision::ocr::DBCURVEDetectorPostprocessor::SetDetDBUnclipRatio)
      .def_property(
          "det_db_score_mode",
          &vision::ocr::DBCURVEDetectorPostprocessor::GetDetDBScoreMode,
          &vision::ocr::DBCURVEDetectorPostprocessor::SetDetDBScoreMode)
      .def_property("det_db_box_type",
                    &vision::ocr::DBCURVEDetectorPostprocessor::GetDetDBBoxType,
                    &vision::ocr::DBCURVEDetectorPostprocessor::SetDetDBBoxType)
      .def_property("use_dilation",
                    &vision::ocr::DBCURVEDetectorPostprocessor::GetUseDilation,
                    &vision::ocr::DBCURVEDetectorPostprocessor::SetUseDilation)

      .def("run",
           [](vision::ocr::DBCURVEDetectorPostprocessor &self,
              std::vector<FDTensor> &inputs,
              const std::vector<std::array<int, 4>> &batch_det_img_info) {
             std::vector<std::vector<std::vector<int>>> results;

             if (!self.Run(inputs, &results, batch_det_img_info)) {
               throw std::runtime_error(
                   "Failed to preprocess the input data in "
                   "DBCURVEDetectorPostprocessor.");
             }
             return results;
           })
      .def(
          "run", [](vision::ocr::DBCURVEDetectorPostprocessor &self,
                    std::vector<pybind11::array> &input_array,
                    const std::vector<std::array<int, 4>> &batch_det_img_info) {
            std::vector<std::vector<std::vector<int>>> results;
            std::vector<FDTensor> inputs;
            PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
            if (!self.Run(inputs, &results, batch_det_img_info)) {
              throw std::runtime_error("Failed to preprocess the input data in "
                                       "DBCURVEDetectorPostprocessor.");
            }
            return results;
          });

  pybind11::class_<vision::ocr::DBDetector, UltraInferModel>(m, "DBDetector")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def(pybind11::init<>())
      .def_property_readonly("preprocessor",
                             &vision::ocr::DBDetector::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::ocr::DBDetector::GetPostprocessor)
      .def("predict",
           [](vision::ocr::DBDetector &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::OCRResult ocr_result;
             self.Predict(mat, &ocr_result);
             return ocr_result;
           })
      .def("batch_predict", [](vision::ocr::DBDetector &self,
                               std::vector<pybind11::array> &data) {
        std::vector<cv::Mat> images;
        for (size_t i = 0; i < data.size(); ++i) {
          images.push_back(PyArrayToCvMat(data[i]));
        }
        std::vector<vision::OCRResult> ocr_results;
        self.BatchPredict(images, &ocr_results);
        return ocr_results;
      });

  pybind11::class_<vision::ocr::DBCURVEDetector, UltraInferModel>(
      m, "DBCURVEDetector")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def(pybind11::init<>())
      .def_property_readonly("preprocessor",
                             &vision::ocr::DBCURVEDetector::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::ocr::DBCURVEDetector::GetPostprocessor)
      .def("predict",
           [](vision::ocr::DBCURVEDetector &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::OCRCURVEResult ocr_result;
             self.Predict(mat, &ocr_result);
             return ocr_result;
           })
      .def("batch_predict", [](vision::ocr::DBCURVEDetector &self,
                               std::vector<pybind11::array> &data) {
        std::vector<cv::Mat> images;
        for (size_t i = 0; i < data.size(); ++i) {
          images.push_back(PyArrayToCvMat(data[i]));
        }
        std::vector<vision::OCRCURVEResult> ocr_results;
        self.BatchPredict(images, &ocr_results);
        return ocr_results;
      });

  // Classifier
  pybind11::class_<vision::ocr::ClassifierPreprocessor,
                   vision::ProcessorManager>(m, "ClassifierPreprocessor")
      .def(pybind11::init<>())
      .def_property("cls_image_shape",
                    &vision::ocr::ClassifierPreprocessor::GetClsImageShape,
                    &vision::ocr::ClassifierPreprocessor::SetClsImageShape)
      .def("set_normalize",
           [](vision::ocr::ClassifierPreprocessor &self,
              const std::vector<float> &mean, const std::vector<float> &std,
              bool is_scale) { self.SetNormalize(mean, std, is_scale); })
      .def("run",
           [](vision::ocr::ClassifierPreprocessor &self,
              std::vector<pybind11::array> &im_list) {
             std::vector<vision::FDMat> images;
             for (size_t i = 0; i < im_list.size(); ++i) {
               images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
             }
             std::vector<FDTensor> outputs;
             if (!self.Run(&images, &outputs)) {
               throw std::runtime_error(
                   "Failed to preprocess the input data in "
                   "ClassifierPreprocessor.");
             }
             for (size_t i = 0; i < outputs.size(); ++i) {
               outputs[i].StopSharing();
             }
             return outputs;
           })
      .def("disable_normalize",
           [](vision::ocr::ClassifierPreprocessor &self) {
             self.DisableNormalize();
           })
      .def("disable_permute", [](vision::ocr::ClassifierPreprocessor &self) {
        self.DisablePermute();
      });

  pybind11::class_<vision::ocr::ClassifierPostprocessor>(
      m, "ClassifierPostprocessor")
      .def(pybind11::init<>())
      .def_property("cls_thresh",
                    &vision::ocr::ClassifierPostprocessor::GetClsThresh,
                    &vision::ocr::ClassifierPostprocessor::SetClsThresh)
      .def("run",
           [](vision::ocr::ClassifierPostprocessor &self,
              std::vector<FDTensor> &inputs) {
             std::vector<int> cls_labels;
             std::vector<float> cls_scores;
             if (!self.Run(inputs, &cls_labels, &cls_scores)) {
               throw std::runtime_error(
                   "Failed to preprocess the input data in "
                   "ClassifierPostprocessor.");
             }
             return std::make_pair(cls_labels, cls_scores);
           })
      .def("run", [](vision::ocr::ClassifierPostprocessor &self,
                     std::vector<pybind11::array> &input_array) {
        std::vector<FDTensor> inputs;
        PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
        std::vector<int> cls_labels;
        std::vector<float> cls_scores;
        if (!self.Run(inputs, &cls_labels, &cls_scores)) {
          throw std::runtime_error("Failed to preprocess the input data in "
                                   "ClassifierPostprocessor.");
        }
        return std::make_pair(cls_labels, cls_scores);
      });

  pybind11::class_<vision::ocr::Classifier, UltraInferModel>(m, "Classifier")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def(pybind11::init<>())
      .def_property_readonly("preprocessor",
                             &vision::ocr::Classifier::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::ocr::Classifier::GetPostprocessor)
      .def("predict",
           [](vision::ocr::Classifier &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::OCRResult ocr_result;
             self.Predict(mat, &ocr_result);
             return ocr_result;
           })
      .def("batch_predict", [](vision::ocr::Classifier &self,
                               std::vector<pybind11::array> &data) {
        std::vector<cv::Mat> images;
        for (size_t i = 0; i < data.size(); ++i) {
          images.push_back(PyArrayToCvMat(data[i]));
        }
        vision::OCRResult ocr_result;
        self.BatchPredict(images, &ocr_result);
        return ocr_result;
      });

  // Recognizer
  pybind11::class_<vision::ocr::RecognizerPreprocessor,
                   vision::ProcessorManager>(m, "RecognizerPreprocessor")
      .def(pybind11::init<>())
      .def_property("static_shape_infer",
                    &vision::ocr::RecognizerPreprocessor::GetStaticShapeInfer,
                    &vision::ocr::RecognizerPreprocessor::SetStaticShapeInfer)
      .def_property("rec_image_shape",
                    &vision::ocr::RecognizerPreprocessor::GetRecImageShape,
                    &vision::ocr::RecognizerPreprocessor::SetRecImageShape)
      .def("set_normalize",
           [](vision::ocr::RecognizerPreprocessor &self,
              const std::vector<float> &mean, const std::vector<float> &std,
              bool is_scale) { self.SetNormalize(mean, std, is_scale); })
      .def("run",
           [](vision::ocr::RecognizerPreprocessor &self,
              std::vector<pybind11::array> &im_list) {
             std::vector<vision::FDMat> images;
             for (size_t i = 0; i < im_list.size(); ++i) {
               images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
             }
             std::vector<FDTensor> outputs;
             if (!self.Run(&images, &outputs)) {
               throw std::runtime_error(
                   "Failed to preprocess the input data in "
                   "RecognizerPreprocessor.");
             }
             for (size_t i = 0; i < outputs.size(); ++i) {
               outputs[i].StopSharing();
             }
             return outputs;
           })
      .def("disable_normalize",
           [](vision::ocr::RecognizerPreprocessor &self) {
             self.DisableNormalize();
           })
      .def("disable_permute", [](vision::ocr::RecognizerPreprocessor &self) {
        self.DisablePermute();
      });

  pybind11::class_<vision::ocr::RecognizerPostprocessor>(
      m, "RecognizerPostprocessor")
      .def(pybind11::init<std::string>())
      .def("run",
           [](vision::ocr::RecognizerPostprocessor &self,
              std::vector<FDTensor> &inputs) {
             std::vector<std::string> texts;
             std::vector<float> rec_scores;
             if (!self.Run(inputs, &texts, &rec_scores)) {
               throw std::runtime_error(
                   "Failed to preprocess the input data in "
                   "RecognizerPostprocessor.");
             }
             return std::make_pair(texts, rec_scores);
           })
      .def("run", [](vision::ocr::RecognizerPostprocessor &self,
                     std::vector<pybind11::array> &input_array) {
        std::vector<FDTensor> inputs;
        PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
        std::vector<std::string> texts;
        std::vector<float> rec_scores;
        if (!self.Run(inputs, &texts, &rec_scores)) {
          throw std::runtime_error("Failed to preprocess the input data in "
                                   "RecognizerPostprocessor.");
        }
        return std::make_pair(texts, rec_scores);
      });

  pybind11::class_<vision::ocr::Recognizer, UltraInferModel>(m, "Recognizer")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def(pybind11::init<>())
      .def_property_readonly("preprocessor",
                             &vision::ocr::Recognizer::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::ocr::Recognizer::GetPostprocessor)
      .def("clone", [](vision::ocr::Recognizer &self) { return self.Clone(); })
      .def("predict",
           [](vision::ocr::Recognizer &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::OCRResult ocr_result;
             self.Predict(mat, &ocr_result);
             return ocr_result;
           })
      .def("batch_predict", [](vision::ocr::Recognizer &self,
                               std::vector<pybind11::array> &data) {
        std::vector<cv::Mat> images;
        for (size_t i = 0; i < data.size(); ++i) {
          images.push_back(PyArrayToCvMat(data[i]));
        }
        vision::OCRResult ocr_result;
        self.BatchPredict(images, &ocr_result);
        return ocr_result;
      });

  // Table
  pybind11::class_<vision::ocr::StructureV2TablePreprocessor,
                   vision::ProcessorManager>(m, "StructureV2TablePreprocessor")
      .def(pybind11::init<>())
      .def("run", [](vision::ocr::StructureV2TablePreprocessor &self,
                     std::vector<pybind11::array> &im_list) {
        std::vector<vision::FDMat> images;
        for (size_t i = 0; i < im_list.size(); ++i) {
          images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
        }
        std::vector<FDTensor> outputs;
        if (!self.Run(&images, &outputs)) {
          throw std::runtime_error("Failed to preprocess the input data in "
                                   "StructureV2TablePreprocessor.");
        }

        auto batch_det_img_info = self.GetBatchImgInfo();
        for (size_t i = 0; i < outputs.size(); ++i) {
          outputs[i].StopSharing();
        }

        return std::make_pair(outputs, *batch_det_img_info);
      });

  pybind11::class_<vision::ocr::StructureV2TablePostprocessor>(
      m, "StructureV2TablePostprocessor")
      .def(pybind11::init<std::string, std::string>())
      .def("run",
           [](vision::ocr::StructureV2TablePostprocessor &self,
              std::vector<FDTensor> &inputs,
              const std::vector<std::array<float, 6>> &batch_det_img_info) {
             std::vector<std::vector<std::array<int, 8>>> boxes;
             std::vector<std::vector<std::string>> structure_list;

             if (!self.Run(inputs, &boxes, &structure_list,
                           batch_det_img_info)) {
               throw std::runtime_error(
                   "Failed to postprocess the input data in "
                   "StructureV2TablePostprocessor.");
             }
             return std::make_pair(boxes, structure_list);
           })
      .def("run",
           [](vision::ocr::StructureV2TablePostprocessor &self,
              std::vector<pybind11::array> &input_array,
              const std::vector<std::array<float, 6>> &batch_det_img_info) {
             std::vector<FDTensor> inputs;
             PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
             std::vector<std::vector<std::array<int, 8>>> boxes;
             std::vector<std::vector<std::string>> structure_list;

             if (!self.Run(inputs, &boxes, &structure_list,
                           batch_det_img_info)) {
               throw std::runtime_error(
                   "Failed to postprocess the input data in "
                   "StructureV2TablePostprocessor.");
             }
             return std::make_pair(boxes, structure_list);
           });

  pybind11::class_<vision::ocr::StructureV2Table, UltraInferModel>(
      m, "StructureV2Table")
      .def(pybind11::init<std::string, std::string, std::string, std::string,
                          RuntimeOption, ModelFormat>())
      .def(pybind11::init<>())
      .def_property_readonly("preprocessor",
                             &vision::ocr::StructureV2Table::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::ocr::StructureV2Table::GetPostprocessor)
      .def("clone",
           [](vision::ocr::StructureV2Table &self) { return self.Clone(); })
      .def("predict",
           [](vision::ocr::StructureV2Table &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::OCRResult ocr_result;
             self.Predict(mat, &ocr_result);
             return ocr_result;
           })
      .def("batch_predict", [](vision::ocr::StructureV2Table &self,
                               std::vector<pybind11::array> &data) {
        std::vector<cv::Mat> images;
        for (size_t i = 0; i < data.size(); ++i) {
          images.push_back(PyArrayToCvMat(data[i]));
        }

        std::vector<vision::OCRResult> ocr_results;
        self.BatchPredict(images, &ocr_results);
        return ocr_results;
      });

  // Layout
  pybind11::class_<vision::ocr::StructureV2LayoutPreprocessor,
                   vision::ProcessorManager>(m, "StructureV2LayoutPreprocessor")
      .def(pybind11::init<>())
      .def_property(
          "static_shape_infer",
          &vision::ocr::StructureV2LayoutPreprocessor::GetStaticShapeInfer,
          &vision::ocr::StructureV2LayoutPreprocessor::SetStaticShapeInfer)
      .def_property(
          "layout_image_shape",
          &vision::ocr::StructureV2LayoutPreprocessor::GetLayoutImageShape,
          &vision::ocr::StructureV2LayoutPreprocessor::SetLayoutImageShape)
      .def("set_normalize",
           [](vision::ocr::StructureV2LayoutPreprocessor &self,
              const std::vector<float> &mean, const std::vector<float> &std,
              bool is_scale) { self.SetNormalize(mean, std, is_scale); })
      .def("run",
           [](vision::ocr::StructureV2LayoutPreprocessor &self,
              std::vector<pybind11::array> &im_list) {
             std::vector<vision::FDMat> images;
             for (size_t i = 0; i < im_list.size(); ++i) {
               images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
             }
             std::vector<FDTensor> outputs;
             if (!self.Run(&images, &outputs)) {
               throw std::runtime_error(
                   "Failed to preprocess the input data in "
                   "StructureV2LayoutPreprocessor.");
             }

             auto batch_layout_img_info = self.GetBatchLayoutImgInfo();
             for (size_t i = 0; i < outputs.size(); ++i) {
               outputs[i].StopSharing();
             }

             return std::make_pair(outputs, *batch_layout_img_info);
           })
      .def("disable_normalize",
           [](vision::ocr::StructureV2LayoutPreprocessor &self) {
             self.DisableNormalize();
           })
      .def("disable_permute",
           [](vision::ocr::StructureV2LayoutPreprocessor &self) {
             self.DisablePermute();
           });

  pybind11::class_<vision::ocr::StructureV2LayoutPostprocessor>(
      m, "StructureV2LayoutPostprocessor")
      .def(pybind11::init<>())
      .def_property(
          "score_threshold",
          &vision::ocr::StructureV2LayoutPostprocessor::GetScoreThreshold,
          &vision::ocr::StructureV2LayoutPostprocessor::SetScoreThreshold)
      .def_property(
          "nms_threshold",
          &vision::ocr::StructureV2LayoutPostprocessor::GetNMSThreshold,
          &vision::ocr::StructureV2LayoutPostprocessor::SetNMSThreshold)
      .def_property("num_class",
                    &vision::ocr::StructureV2LayoutPostprocessor::GetNumClass,
                    &vision::ocr::StructureV2LayoutPostprocessor::SetNumClass)
      .def_property("fpn_stride",
                    &vision::ocr::StructureV2LayoutPostprocessor::GetFPNStride,
                    &vision::ocr::StructureV2LayoutPostprocessor::SetFPNStride)
      .def_property("reg_max",
                    &vision::ocr::StructureV2LayoutPostprocessor::GetRegMax,
                    &vision::ocr::StructureV2LayoutPostprocessor::SetRegMax)
      .def("run",
           [](vision::ocr::StructureV2LayoutPostprocessor &self,
              std::vector<FDTensor> &inputs,
              const std::vector<std::array<int, 4>> &batch_layout_img_info) {
             std::vector<vision::DetectionResult> results;

             if (!self.Run(inputs, &results, batch_layout_img_info)) {
               throw std::runtime_error(
                   "Failed to postprocess the input data in "
                   "StructureV2LayoutPostprocessor.");
             }
             return results;
           });

  pybind11::class_<vision::ocr::StructureV2Layout, UltraInferModel>(
      m, "StructureV2Layout")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def(pybind11::init<>())
      .def_property_readonly("preprocessor",
                             &vision::ocr::StructureV2Layout::GetPreprocessor)
      .def_property_readonly("postprocessor",
                             &vision::ocr::StructureV2Layout::GetPostprocessor)
      .def("clone",
           [](vision::ocr::StructureV2Layout &self) { return self.Clone(); })
      .def("predict",
           [](vision::ocr::StructureV2Layout &self, pybind11::array &data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult result;
             self.Predict(mat, &result);
             return result;
           })
      .def("batch_predict", [](vision::ocr::StructureV2Layout &self,
                               std::vector<pybind11::array> &data) {
        std::vector<cv::Mat> images;
        for (size_t i = 0; i < data.size(); ++i) {
          images.push_back(PyArrayToCvMat(data[i]));
        }
        std::vector<vision::DetectionResult> results;
        self.BatchPredict(images, &results);
        return results;
      });

  pybind11::class_<vision::ocr::StructureV2SERViLayoutXLMModel,
                   UltraInferModel>(m, "StructureV2SERViLayoutXLMModel")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("clone",
           [](vision::ocr::StructureV2SERViLayoutXLMModel &self) {
             return self.Clone();
           })
      .def("predict",
           [](vision::ocr::StructureV2SERViLayoutXLMModel &self,
              pybind11::array &data) {
             throw std::runtime_error(
                 "StructureV2SERViLayoutXLMModel do not support predict.");
           })
      .def(
          "batch_predict",
          [](vision::ocr::StructureV2SERViLayoutXLMModel &self,
             std::vector<pybind11::array> &data) {
            throw std::runtime_error(
                "StructureV2SERViLayoutXLMModel do not support batch_predict.");
          })
      .def("infer",
           [](vision::ocr::StructureV2SERViLayoutXLMModel &self,
              std::map<std::string, pybind11::array> &data) {
             std::vector<FDTensor> inputs(data.size());
             int index = 0;
             for (auto iter = data.begin(); iter != data.end(); ++iter) {
               std::vector<int64_t> data_shape;
               data_shape.insert(data_shape.begin(), iter->second.shape(),
                                 iter->second.shape() + iter->second.ndim());
               auto dtype = NumpyDataTypeToFDDataType(iter->second.dtype());

               inputs[index].Resize(data_shape, dtype);
               memcpy(inputs[index].MutableData(), iter->second.mutable_data(),
                      iter->second.nbytes());
               inputs[index].name = iter->first;
               index += 1;
             }

             std::vector<FDTensor> outputs(self.NumOutputsOfRuntime());
             self.Infer(inputs, &outputs);

             std::vector<pybind11::array> results;
             results.reserve(outputs.size());
             for (size_t i = 0; i < outputs.size(); ++i) {
               auto numpy_dtype = FDDataTypeToNumpyDataType(outputs[i].dtype);
               results.emplace_back(
                   pybind11::array(numpy_dtype, outputs[i].shape));
               memcpy(results[i].mutable_data(), outputs[i].Data(),
                      outputs[i].Numel() * FDDataTypeSize(outputs[i].dtype));
             }
             return results;
           })
      .def("get_input_info",
           [](vision::ocr::StructureV2SERViLayoutXLMModel &self, int &index) {
             return self.InputInfoOfRuntime(index);
           });
}
} // namespace ultra_infer
