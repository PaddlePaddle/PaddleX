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

namespace py = pybind11;
using namespace py::literals;

namespace ultra_infer {

void BindUIE(py::module &m);

py::dict ConvertUIEResultToDict(const text::UIEResult &self) {
  py::dict d;
  d["start"] = self.start_;
  d["end"] = self.end_;
  d["probability"] = self.probability_;
  d["text"] = self.text_;

  if (!self.relation_.empty()) {
    d["relation"] = py::dict();
    for (auto iter = self.relation_.begin(); iter != self.relation_.end();
         ++iter) {
      py::list l;
      for (auto result_iter = iter->second.begin();
           result_iter != iter->second.end(); ++result_iter) {
        l.append(ConvertUIEResultToDict(*result_iter));
      }
      d["relation"][iter->first.c_str()] = l;
    }
  }
  return d;
}

void BindText(py::module &m) {
  py::class_<text::UIEResult>(m, "UIEResult", py::dynamic_attr())
      .def(py::init())
      .def_readwrite("start", &text::UIEResult::start_)
      .def_readwrite("end", &text::UIEResult::end_)
      .def_readwrite("probability", &text::UIEResult::probability_)
      .def_readwrite("text", &text::UIEResult::text_)
      .def_readwrite("relation", &text::UIEResult::relation_)
      .def("get_dict",
           [](const text::UIEResult &self) {
             return ConvertUIEResultToDict(self);
           })
      .def("__repr__", &text::UIEResult::Str)
      .def("__str__", &text::UIEResult::Str);
  BindUIE(m);
}

} // namespace ultra_infer
