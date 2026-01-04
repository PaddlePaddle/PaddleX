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
#pragma once

#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>

#include "ultra_infer/utils/utils.h"

namespace ultra_infer {

class ULTRAINFER_DECL FDHostAllocator {
public:
  bool operator()(void **ptr, size_t size) const;
};

class ULTRAINFER_DECL FDHostFree {
public:
  void operator()(void *ptr) const;
};

#ifdef WITH_GPU

class ULTRAINFER_DECL FDDeviceAllocator {
public:
  bool operator()(void **ptr, size_t size) const;
};

class ULTRAINFER_DECL FDDeviceFree {
public:
  void operator()(void *ptr) const;
};

class ULTRAINFER_DECL FDDeviceHostAllocator {
public:
  bool operator()(void **ptr, size_t size) const;
};

class ULTRAINFER_DECL FDDeviceHostFree {
public:
  void operator()(void *ptr) const;
};

#endif

} // namespace ultra_infer
