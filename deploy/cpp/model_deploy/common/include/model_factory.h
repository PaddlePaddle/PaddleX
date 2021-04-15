// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "model_deploy/common/include/base_model.h"
#include "model_deploy/ppdet/include/det_model.h"

namespace PaddleDeploy {

typedef std::shared_ptr<Model> (*NewInstance)();

class ModelFactory {
 private:
  static std::map<std::string, NewInstance> model_map_;

 public:
  static std::shared_ptr<Model> CreateObject(const std::string &name);

  static void Register(const std::string &name, NewInstance model);
};

class Register {
 public:
  Register(const std::string &name, NewInstance func) {
    ModelFactory::Register(name, func);
  }
};

#define REGISTER_CLASS(model_type, class_name)                    \
  class class_name##Register {                                    \
   public:                                                        \
    static std::shared_ptr<Model> newInstance() {                 \
      std::cout << "REGISTER_CLASS:" << #model_type << std::endl; \
      return std::make_shared<class_name>(#model_type);           \
    }                                                             \
                                                                  \
   private:                                                       \
    static Register reg_;                                         \
  };                                                              \
  Register class_name##Register::reg_(#model_type,                \
                class_name##Register::newInstance);

}  // namespace PaddleDeploy
