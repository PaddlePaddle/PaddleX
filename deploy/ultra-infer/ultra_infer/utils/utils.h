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

#include <cstdio>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#ifdef ULTRAINFER_LIB
#define ULTRAINFER_DECL __declspec(dllexport)
#else
#define ULTRAINFER_DECL __declspec(dllimport)
#endif // ULTRAINFER_LIB
#else
#define ULTRAINFER_DECL __attribute__((visibility("default")))
#endif // _WIN32

namespace ultra_infer {

class ULTRAINFER_DECL FDLogger {
public:
  static bool enable_info;
  static bool enable_warning;

  FDLogger() {
    line_ = "";
    prefix_ = "[UltraInfer]";
    verbose_ = true;
  }
  explicit FDLogger(bool verbose, const std::string &prefix = "[UltraInfer]");

  template <typename T> FDLogger &operator<<(const T &val) {
    if (!verbose_) {
      return *this;
    }
    std::stringstream ss;
    ss << val;
    line_ += ss.str();
    return *this;
  }

  FDLogger &operator<<(std::ostream &(*os)(std::ostream &));

  ~FDLogger() {
    if (verbose_ && line_ != "") {
      std::cout << line_ << std::endl;
    }
  }

private:
  std::string line_;
  std::string prefix_;
  bool verbose_ = true;
};

ULTRAINFER_DECL bool ReadBinaryFromFile(const std::string &file,
                                        std::string *contents);

#ifndef __REL_FILE__
#define __REL_FILE__ __FILE__
#endif

#define FDERROR                                                                \
  FDLogger(true, "[ERROR]")                                                    \
      << __REL_FILE__ << "(" << __LINE__ << ")::" << __FUNCTION__ << "\t"

#define FDWARNING                                                              \
  FDLogger(ultra_infer::FDLogger::enable_warning, "[WARNING]")                 \
      << __REL_FILE__ << "(" << __LINE__ << ")::" << __FUNCTION__ << "\t"

#define FDINFO                                                                 \
  FDLogger(ultra_infer::FDLogger::enable_info, "[INFO]")                       \
      << __REL_FILE__ << "(" << __LINE__ << ")::" << __FUNCTION__ << "\t"

#define FDASSERT(condition, format, ...)                                       \
  if (!(condition)) {                                                          \
    int n = std::snprintf(nullptr, 0, format, ##__VA_ARGS__);                  \
    std::vector<char> buffer(n + 1);                                           \
    std::snprintf(buffer.data(), n + 1, format, ##__VA_ARGS__);                \
    FDERROR << buffer.data() << std::endl;                                     \
    std::abort();                                                              \
  }

///////// Basic Marco ///////////

#define FD_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, HINT, ...)      \
  case enum_type: {                                                            \
    using HINT = type;                                                         \
    __VA_ARGS__();                                                             \
    break;                                                                     \
  }

#define FD_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...)                       \
  FD_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, data_t, __VA_ARGS__)

// Visit different data type to match the corresponding function of FDTensor
#define FD_VISIT_ALL_TYPES(TYPE, NAME, ...)                                    \
  [&] {                                                                        \
    const auto &__dtype__ = TYPE;                                              \
    switch (__dtype__) {                                                       \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::UINT8, uint8_t,    \
                           __VA_ARGS__)                                        \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::BOOL, bool,        \
                           __VA_ARGS__)                                        \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::INT32, int32_t,    \
                           __VA_ARGS__)                                        \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::INT64, int64_t,    \
                           __VA_ARGS__)                                        \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::FP32, float,       \
                           __VA_ARGS__)                                        \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::FP64, double,      \
                           __VA_ARGS__)                                        \
    default:                                                                   \
      FDASSERT(false,                                                          \
               "Invalid enum data type. Expect to accept data "                \
               "type BOOL, INT32, "                                            \
               "INT64, FP32, FP64, but receive type %s.",                      \
               Str(__dtype__).c_str());                                        \
    }                                                                          \
  }()

#define FD_VISIT_INT_FLOAT_TYPES(TYPE, NAME, ...)                              \
  [&] {                                                                        \
    const auto &__dtype__ = TYPE;                                              \
    switch (__dtype__) {                                                       \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::INT32, int32_t,    \
                           __VA_ARGS__)                                        \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::INT64, int64_t,    \
                           __VA_ARGS__)                                        \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::FP32, float,       \
                           __VA_ARGS__)                                        \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::FP64, double,      \
                           __VA_ARGS__)                                        \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::UINT8, uint8_t,    \
                           __VA_ARGS__)                                        \
    default:                                                                   \
      FDASSERT(false,                                                          \
               "Invalid enum data type. Expect to accept data type INT32, "    \
               "INT64, FP32, FP64, UINT8 but receive type %s.",                \
               Str(__dtype__).c_str());                                        \
    }                                                                          \
  }()

#define FD_VISIT_FLOAT_TYPES(TYPE, NAME, ...)                                  \
  [&] {                                                                        \
    const auto &__dtype__ = TYPE;                                              \
    switch (__dtype__) {                                                       \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::FP32, float,       \
                           __VA_ARGS__)                                        \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::FP64, double,      \
                           __VA_ARGS__)                                        \
    default:                                                                   \
      FDASSERT(false,                                                          \
               "Invalid enum data type. Expect to accept data type FP32, "     \
               "FP64, but receive type %s.",                                   \
               Str(__dtype__).c_str());                                        \
    }                                                                          \
  }()

#define FD_VISIT_INT_TYPES(TYPE, NAME, ...)                                    \
  [&] {                                                                        \
    const auto &__dtype__ = TYPE;                                              \
    switch (__dtype__) {                                                       \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::INT32, int32_t,    \
                           __VA_ARGS__)                                        \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::INT64, int64_t,    \
                           __VA_ARGS__)                                        \
      FD_PRIVATE_CASE_TYPE(NAME, ::ultra_infer::FDDataType::UINT8, uint8_t,    \
                           __VA_ARGS__)                                        \
    default:                                                                   \
      FDASSERT(false,                                                          \
               "Invalid enum data type. Expect to accept data type INT32, "    \
               "INT64, UINT8 but receive type %s.",                            \
               Str(__dtype__).c_str());                                        \
    }                                                                          \
  }()

ULTRAINFER_DECL std::vector<int64_t>
GetStride(const std::vector<int64_t> &dims);

template <typename T> std::string Str(const std::vector<T> &shape) {
  std::ostringstream oss;
  oss << "[ " << shape[0];
  for (size_t i = 1; i < shape.size(); ++i) {
    oss << " ," << shape[i];
  }
  oss << " ]";
  return oss.str();
}

/// Set behaviour of logging while using UltraInfer
ULTRAINFER_DECL void SetLogger(bool enable_info = true,
                               bool enable_warning = true);

template <typename T>
void CalculateStatisInfo(const void *src_ptr, int size, double *mean,
                         double *max, double *min) {
  const T *ptr = static_cast<const T *>(src_ptr);
  *mean = static_cast<double>(0);
  *max = static_cast<double>(-99999999);
  *min = static_cast<double>(99999999);
  for (int i = 0; i < size; ++i) {
    if (*(ptr + i) > *max) {
      *max = *(ptr + i);
    }
    if (*(ptr + i) < *min) {
      *min = *(ptr + i);
    }
    *mean += *(ptr + i);
  }
  *mean = *mean / size;
}

} // namespace ultra_infer
