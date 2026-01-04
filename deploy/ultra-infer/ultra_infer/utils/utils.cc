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

#include "ultra_infer/utils/utils.h"

#include <sstream>

namespace ultra_infer {

bool FDLogger::enable_info = true;
bool FDLogger::enable_warning = true;

void SetLogger(bool enable_info, bool enable_warning) {
  FDLogger::enable_info = enable_info;
  FDLogger::enable_warning = enable_warning;
}

FDLogger::FDLogger(bool verbose, const std::string &prefix) {
  verbose_ = verbose;
  line_ = "";
  prefix_ = prefix;
}

FDLogger &FDLogger::operator<<(std::ostream &(*os)(std::ostream &)) {
  if (!verbose_) {
    return *this;
  }
  std::cout << prefix_ << " " << line_ << std::endl;
  line_ = "";
  return *this;
}

bool ReadBinaryFromFile(const std::string &file, std::string *contents) {
  std::ifstream fin(file, std::ios::in | std::ios::binary);
  if (!fin.is_open()) {
    FDERROR << "Failed to open file: " << file << " to read." << std::endl;
    return false;
  }
  fin.seekg(0, std::ios::end);
  contents->clear();
  contents->resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
  return true;
}

std::vector<int64_t> GetStride(const std::vector<int64_t> &dims) {
  auto dims_size = dims.size();
  std::vector<int64_t> result(dims_size, 1);
  for (int i = dims_size - 2; i >= 0; --i) {
    result[i] = result[i + 1] * dims[i + 1];
  }
  return result;
}

} // namespace ultra_infer
