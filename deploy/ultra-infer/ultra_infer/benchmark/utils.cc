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

#include <sys/types.h>
#ifdef __linux__
#include <sys/resource.h>
#endif
#include <cmath>

#include "ultra_infer/benchmark/utils.h"
#include "ultra_infer/utils/path.h"
#if defined(ENABLE_BENCHMARK) && defined(ENABLE_VISION)
#include "ultra_infer/vision/utils/utils.h"
#endif

namespace ultra_infer {
namespace benchmark {

#if defined(ENABLE_BENCHMARK)
std::string Strip(const std::string &str, char ch) {
  int i = 0;
  while (str[i] == ch) {
    i++;
  }
  int j = str.size() - 1;
  while (str[j] == ch) {
    j--;
  }
  return str.substr(i, j + 1 - i);
}

void Split(const std::string &s, std::vector<std::string> &tokens, char delim) {
  tokens.clear();
  size_t lastPos = s.find_first_not_of(delim, 0);
  size_t pos = s.find(delim, lastPos);
  while (lastPos != std::string::npos) {
    tokens.emplace_back(s.substr(lastPos, pos - lastPos));
    lastPos = s.find_first_not_of(delim, pos);
    pos = s.find(delim, lastPos);
  }
  return;
}

ResourceUsageMonitor::ResourceUsageMonitor(int sampling_interval_ms, int gpu_id)
    : is_supported_(false), sampling_interval_(sampling_interval_ms),
      gpu_id_(gpu_id) {
#ifdef __linux__
  is_supported_ = true;
#else
  is_supported_ = false;
#endif
  if (!is_supported_) {
    FDASSERT(false, "Currently ResourceUsageMonitor only supports Linux.")
    return;
  }
}

void ResourceUsageMonitor::Start() {
  if (!is_supported_) {
    return;
  }
  if (check_memory_thd_ != nullptr) {
    FDINFO << "Memory monitoring has already started!" << std::endl;
    return;
  }
  FDINFO << "Start monitoring memory!" << std::endl;
  stop_signal_ = false;
  check_memory_thd_.reset(new std::thread(([this]() {
    // Note we retrieve the memory usage at the very beginning of the thread.
    while (true) {
#ifdef __linux__
      rusage res;
      if (getrusage(RUSAGE_SELF, &res) == 0) {
        max_cpu_mem_ =
            std::max(max_cpu_mem_, static_cast<float>(res.ru_maxrss / 1024.0));
      }
#endif
#if defined(WITH_GPU)
      std::string gpu_mem_info = GetCurrentGpuMemoryInfo(gpu_id_);
      // get max_gpu_mem and max_gpu_util
      std::vector<std::string> gpu_tokens;
      Split(gpu_mem_info, gpu_tokens, ',');
      max_gpu_mem_ = std::max(max_gpu_mem_, stof(gpu_tokens[6]));
      max_gpu_util_ = std::max(max_gpu_util_, stof(gpu_tokens[7]));
#endif
      if (stop_signal_) {
        break;
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(sampling_interval_));
    }
  })));
}

void ResourceUsageMonitor::Stop() {
  if (!is_supported_) {
    return;
  }
  if (check_memory_thd_ == nullptr) {
    FDINFO << "Memory monitoring hasn't started yet or has stopped!"
           << std::endl;
    return;
  }
  FDINFO << "Stop monitoring memory!" << std::endl;
  StopInternal();
}

void ResourceUsageMonitor::StopInternal() {
  stop_signal_ = true;
  if (check_memory_thd_ == nullptr) {
    return;
  }
  if (check_memory_thd_ != nullptr) {
    check_memory_thd_->join();
  }
  check_memory_thd_.reset(nullptr);
}

std::string ResourceUsageMonitor::GetCurrentGpuMemoryInfo(int device_id) {
  std::string result = "";
#if defined(__linux__) && defined(WITH_GPU)
  std::string command = "nvidia-smi --id=" + std::to_string(device_id) +
                        " --query-gpu=index,uuid,name,timestamp,memory.total,"
                        "memory.free,memory.used,utilization.gpu,utilization."
                        "memory --format=csv,noheader,nounits";
  FILE *pp = popen(command.data(), "r");
  if (!pp)
    return "";
  char tmp[1024];

  while (fgets(tmp, sizeof(tmp), pp) != NULL) {
    result += tmp;
  }
  pclose(pp);
#else
  FDASSERT(false,
           "Currently collect gpu memory info only supports Linux in GPU.")
#endif
  return result;
}
#endif // ENABLE_BENCHMARK

/// Utils for precision evaluation
#if defined(ENABLE_BENCHMARK)
static const char KEY_VALUE_SEP = '#';
static const char VALUE_SEP = ',';

std::vector<std::string> ReadLines(const std::string &path) {
  std::ifstream fin(path);
  std::vector<std::string> lines;
  std::string line;
  if (fin.is_open()) {
    while (getline(fin, line)) {
      lines.push_back(line);
    }
  } else {
    FDERROR << "Failed to open file " << path << std::endl;
    std::abort();
  }
  fin.close();
  return lines;
}

std::map<std::string, std::vector<std::string>>
SplitDataLine(const std::string &data_line) {
  std::map<std::string, std::vector<std::string>> dict;
  std::vector<std::string> tokens, value_tokens;
  Split(data_line, tokens, KEY_VALUE_SEP);
  std::string key = tokens[0];
  std::string value = tokens[1];
  Split(value, value_tokens, VALUE_SEP);
  dict[key] = value_tokens;
  return dict;
}

bool ResultManager::SaveFDTensor(const FDTensor &tensor,
                                 const std::string &path) {
  if (tensor.CpuData() == nullptr || tensor.Numel() <= 0) {
    FDERROR << "Input tensor is empty!" << std::endl;
    return false;
  }
  std::ofstream fs(path, std::ios::out);
  if (!fs.is_open()) {
    FDERROR << "Fail to open file:" << path << std::endl;
    return false;
  }
  fs.precision(20);
  if (tensor.Dtype() != FDDataType::FP32 &&
      tensor.Dtype() != FDDataType::INT32 &&
      tensor.Dtype() != FDDataType::INT64) {
    FDERROR << "Only support FP32/INT32/INT64 now, but got "
            << Str(tensor.dtype) << std::endl;
    return false;
  }
  // name
  fs << "name" << KEY_VALUE_SEP << tensor.name << "\n";
  // shape
  fs << "shape" << KEY_VALUE_SEP;
  for (int i = 0; i < tensor.shape.size(); ++i) {
    if (i < tensor.shape.size() - 1) {
      fs << tensor.shape[i] << VALUE_SEP;
    } else {
      fs << tensor.shape[i];
    }
  }
  fs << "\n";
  // dtype
  fs << "dtype" << KEY_VALUE_SEP << Str(tensor.dtype) << "\n";
  // data
  fs << "data" << KEY_VALUE_SEP;
  const void *data_ptr = tensor.CpuData();
  for (int i = 0; i < tensor.Numel(); ++i) {
    if (tensor.Dtype() == FDDataType::INT64) {
      if (i < tensor.Numel() - 1) {
        fs << (static_cast<const int64_t *>(data_ptr))[i] << VALUE_SEP;
      } else {
        fs << (static_cast<const int64_t *>(data_ptr))[i];
      }
    } else if (tensor.Dtype() == FDDataType::INT32) {
      if (i < tensor.Numel() - 1) {
        fs << (static_cast<const int32_t *>(data_ptr))[i] << VALUE_SEP;
      } else {
        fs << (static_cast<const int32_t *>(data_ptr))[i];
      }
    } else { // FP32
      if (i < tensor.Numel() - 1) {
        fs << (static_cast<const float *>(data_ptr))[i] << VALUE_SEP;
      } else {
        fs << (static_cast<const float *>(data_ptr))[i];
      }
    }
  }
  fs << "\n";
  fs.close();
  return true;
}

bool ResultManager::LoadFDTensor(FDTensor *tensor, const std::string &path) {
  if (!CheckFileExists(path)) {
    FDERROR << "Can't found file from " << path << std::endl;
    return false;
  }
  auto lines = ReadLines(path);
  std::map<std::string, std::vector<std::string>> data;
  // name
  data = SplitDataLine(lines[0]);
  tensor->name = data.begin()->first;
  // shape
  data = SplitDataLine(lines[1]);
  tensor->shape.clear();
  for (const auto &s : data.begin()->second) {
    tensor->shape.push_back(std::stol(s));
  }
  // dtype
  data = SplitDataLine(lines[2]);
  if (data.begin()->second.at(0) == Str(FDDataType::INT64)) {
    tensor->dtype = FDDataType::INT64;
  } else if (data.begin()->second.at(0) == Str(FDDataType::INT32)) {
    tensor->dtype = FDDataType::INT32;
  } else if (data.begin()->second.at(0) == Str(FDDataType::FP32)) {
    tensor->dtype = FDDataType::FP32;
  } else {
    FDERROR << "Only support FP32/INT64/INT32 now, but got "
            << data.begin()->second.at(0) << std::endl;
    return false;
  }
  // data
  data = SplitDataLine(lines[3]);
  tensor->Allocate(tensor->shape, tensor->dtype, tensor->name);
  if (tensor->dtype == FDDataType::INT64) {
    int64_t *mutable_data_ptr = static_cast<int64_t *>(tensor->MutableData());
    for (int i = 0; i < data.begin()->second.size(); ++i) {
      mutable_data_ptr[i] = std::stol(data.begin()->second[i]);
    }
  } else if (tensor->dtype == FDDataType::INT32) {
    int32_t *mutable_data_ptr = static_cast<int32_t *>(tensor->MutableData());
    for (int i = 0; i < data.begin()->second.size(); ++i) {
      mutable_data_ptr[i] = std::stoi(data.begin()->second[i]);
    }
  } else { // FP32
    float *mutable_data_ptr = static_cast<float *>(tensor->MutableData());
    for (int i = 0; i < data.begin()->second.size(); ++i) {
      mutable_data_ptr[i] = std::stof(data.begin()->second[i]);
    }
  }
  return true;
}

TensorDiff ResultManager::CalculateDiffStatis(const FDTensor &lhs,
                                              const FDTensor &rhs) {
  if (lhs.Numel() != rhs.Numel() || lhs.Dtype() != rhs.Dtype()) {
    FDASSERT(false,
             "The size and dtype of input FDTensor must be equal!"
             " But got size %d, %d, dtype %s, %s",
             lhs.Numel(), rhs.Numel(), Str(lhs.Dtype()).c_str(),
             Str(rhs.Dtype()).c_str())
  }
  FDDataType dtype = lhs.Dtype();
  int numel = lhs.Numel();
  if (dtype != FDDataType::FP32 && dtype != FDDataType::INT64 &&
      dtype != FDDataType::INT32) {
    FDASSERT(false, "Only support FP32/INT64/INT32 now, but got %s",
             Str(dtype).c_str())
  }
  if (dtype == FDDataType::INT64) {
    std::vector<int64_t> tensor_diff(numel);
    const int64_t *lhs_data_ptr = static_cast<const int64_t *>(lhs.CpuData());
    const int64_t *rhs_data_ptr = static_cast<const int64_t *>(rhs.CpuData());
    for (int i = 0; i < numel; ++i) {
      tensor_diff[i] = lhs_data_ptr[i] - rhs_data_ptr[i];
    }
    TensorDiff diff;
    CalculateStatisInfo<int64_t>(tensor_diff.data(), numel, &(diff.data.mean),
                                 &(diff.data.max), &(diff.data.min));
    return diff;
  } else if (dtype == FDDataType::INT32) {
    std::vector<int32_t> tensor_diff(numel);
    const int32_t *lhs_data_ptr = static_cast<const int32_t *>(lhs.CpuData());
    const int32_t *rhs_data_ptr = static_cast<const int32_t *>(rhs.CpuData());
    for (int i = 0; i < numel; ++i) {
      tensor_diff[i] = lhs_data_ptr[i] - rhs_data_ptr[i];
    }
    TensorDiff diff;
    CalculateStatisInfo<float>(tensor_diff.data(), numel, &(diff.data.mean),
                               &(diff.data.max), &(diff.data.min));
    return diff;
  } else { // FP32
    std::vector<float> tensor_diff(numel);
    const float *lhs_data_ptr = static_cast<const float *>(lhs.CpuData());
    const float *rhs_data_ptr = static_cast<const float *>(rhs.CpuData());
    for (int i = 0; i < numel; ++i) {
      tensor_diff[i] = lhs_data_ptr[i] - rhs_data_ptr[i];
    }
    TensorDiff diff;
    CalculateStatisInfo<float>(tensor_diff.data(), numel, &(diff.data.mean),
                               &(diff.data.max), &(diff.data.min));
    return diff;
  }
}

void ResultManager::SaveBenchmarkResult(const std::string &res,
                                        const std::string &path) {
  if (path.empty()) {
    FDERROR << "Benchmark data path can not be empty!" << std::endl;
    return;
  }
  auto openmode = std::ios::app;
  std::ofstream fs(path, openmode);
  if (!fs.is_open()) {
    FDERROR << "Fail to open result file: " << path << std::endl;
  }
  fs << res;
  fs.close();
}

bool ResultManager::LoadBenchmarkConfig(
    const std::string &path,
    std::unordered_map<std::string, std::string> *config_info) {
  if (!CheckFileExists(path)) {
    FDERROR << "Can't found file from " << path << std::endl;
    return false;
  }
  auto lines = ReadLines(path);
  for (auto line : lines) {
    std::vector<std::string> tokens;
    Split(line, tokens, ':');
    (*config_info)[tokens[0]] = Strip(tokens[1], ' ');
  }
  return true;
}

std::vector<std::vector<int32_t>>
ResultManager::GetInputShapes(const std::string &raw_shapes) {
  std::vector<std::vector<int32_t>> shapes;
  std::vector<std::string> shape_tokens;
  Split(raw_shapes, shape_tokens, ':');
  for (auto str_shape : shape_tokens) {
    std::vector<int32_t> shape;
    std::string tmp_str = str_shape;
    while (!tmp_str.empty()) {
      int dim = atoi(tmp_str.data());
      shape.push_back(dim);
      size_t next_offset = tmp_str.find(",");
      if (next_offset == std::string::npos) {
        break;
      } else {
        tmp_str = tmp_str.substr(next_offset + 1);
      }
    }
    shapes.push_back(shape);
  }
  return shapes;
}

std::vector<std::string>
ResultManager::GetInputNames(const std::string &raw_names) {
  std::vector<std::string> names_tokens;
  Split(raw_names, names_tokens, ':');
  return names_tokens;
}

std::vector<std::string> ResultManager::SplitStr(const std::string &raw_str,
                                                 char delim) {
  std::vector<std::string> str_tokens;
  Split(raw_str, str_tokens, delim);
  return str_tokens;
}

std::vector<FDDataType>
ResultManager::GetInputDtypes(const std::string &raw_dtypes) {
  std::vector<FDDataType> dtypes;
  std::vector<std::string> dtypes_tokens;
  Split(raw_dtypes, dtypes_tokens, ':');
  for (auto dtype : dtypes_tokens) {
    if (dtype == "FP32") {
      dtypes.push_back(FDDataType::FP32);
    } else if (dtype == "INT32") {
      dtypes.push_back(FDDataType::INT32);
    } else if (dtype == "INT64") {
      dtypes.push_back(FDDataType::INT64);
    } else if (dtype == "INT8") {
      dtypes.push_back(FDDataType::INT8);
    } else if (dtype == "UINT8") {
      dtypes.push_back(FDDataType::UINT8);
    } else if (dtype == "FP16") {
      dtypes.push_back(FDDataType::FP16);
    } else if (dtype == "FP64") {
      dtypes.push_back(FDDataType::FP64);
    } else {
      dtypes.push_back(FDDataType::FP32); // default
    }
  }
  return dtypes;
}

#if defined(ENABLE_VISION)
bool ResultManager::SaveDetectionResult(const vision::DetectionResult &res,
                                        const std::string &path) {
  if (res.boxes.empty()) {
    FDERROR << "DetectionResult can not be empty!" << std::endl;
    return false;
  }
  std::ofstream fs(path, std::ios::out);
  if (!fs.is_open()) {
    FDERROR << "Fail to open file:" << path << std::endl;
    return false;
  }
  fs.precision(20);
  // boxes
  fs << "boxes" << KEY_VALUE_SEP;
  for (int i = 0; i < res.boxes.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      if ((i == res.boxes.size() - 1) && (j == 3)) {
        fs << res.boxes[i][j];
      } else {
        fs << res.boxes[i][j] << VALUE_SEP;
      }
    }
  }
  fs << "\n";
  // scores
  fs << "scores" << KEY_VALUE_SEP;
  for (int i = 0; i < res.scores.size(); ++i) {
    if (i < res.scores.size() - 1) {
      fs << res.scores[i] << VALUE_SEP;
    } else {
      fs << res.scores[i];
    }
  }
  fs << "\n";
  // label_ids
  fs << "label_ids" << KEY_VALUE_SEP;
  for (int i = 0; i < res.label_ids.size(); ++i) {
    if (i < res.label_ids.size() - 1) {
      fs << res.label_ids[i] << VALUE_SEP;
    } else {
      fs << res.label_ids[i];
    }
  }
  fs << "\n";
  // TODO(qiuyanjun): dump masks
  fs.close();
  return true;
}

bool ResultManager::SaveClassifyResult(const vision::ClassifyResult &res,
                                       const std::string &path) {
  if (res.label_ids.empty()) {
    FDERROR << "ClassifyResult can not be empty!" << std::endl;
    return false;
  }
  std::ofstream fs(path, std::ios::out);
  if (!fs.is_open()) {
    FDERROR << "Fail to open file:" << path << std::endl;
    return false;
  }
  fs.precision(20);
  // label_ids
  fs << "label_ids" << KEY_VALUE_SEP;
  for (int i = 0; i < res.label_ids.size(); ++i) {
    if (i < res.label_ids.size() - 1) {
      fs << res.label_ids[i] << VALUE_SEP;
    } else {
      fs << res.label_ids[i];
    }
  }
  fs << "\n";
  // scores
  fs << "scores" << KEY_VALUE_SEP;
  for (int i = 0; i < res.scores.size(); ++i) {
    if (i < res.scores.size() - 1) {
      fs << res.scores[i] << VALUE_SEP;
    } else {
      fs << res.scores[i];
    }
  }
  fs << "\n";
  fs.close();
  return true;
}

bool ResultManager::SaveSegmentationResult(
    const vision::SegmentationResult &res, const std::string &path) {
  if (res.label_map.empty()) {
    FDERROR << "SegmentationResult can not be empty!" << std::endl;
    return false;
  }
  std::ofstream fs(path, std::ios::out);
  if (!fs.is_open()) {
    FDERROR << "Fail to open file:" << path << std::endl;
    return false;
  }
  fs.precision(20);
  // label_map
  fs << "label_map" << KEY_VALUE_SEP;
  for (int i = 0; i < res.label_map.size(); ++i) {
    if (i < res.label_map.size() - 1) {
      fs << static_cast<int32_t>(res.label_map[i]) << VALUE_SEP;
    } else {
      fs << static_cast<int32_t>(res.label_map[i]);
    }
  }
  fs << "\n";
  // score_map
  if (res.contain_score_map) {
    fs << "score_map" << KEY_VALUE_SEP;
    for (int i = 0; i < res.score_map.size(); ++i) {
      if (i < res.score_map.size() - 1) {
        fs << res.score_map[i] << VALUE_SEP;
      } else {
        fs << res.score_map[i];
      }
    }
    fs << "\n";
  }
  fs.close();
  return true;
}

bool ResultManager::SaveOCRDetResult(const std::vector<std::array<int, 8>> &res,
                                     const std::string &path) {
  if (res.empty()) {
    FDERROR << "OCRDetResult can not be empty!" << std::endl;
    return false;
  }
  std::ofstream fs(path, std::ios::out);
  if (!fs.is_open()) {
    FDERROR << "Fail to open file:" << path << std::endl;
    return false;
  }
  fs.precision(20);
  // boxes
  fs << "boxes" << KEY_VALUE_SEP;
  for (int i = 0; i < res.size(); ++i) {
    for (int j = 0; j < 8; ++j) {
      if ((i == res.size() - 1) && (j == 7)) {
        fs << res[i][j];
      } else {
        fs << res[i][j] << VALUE_SEP;
      }
    }
  }
  fs << "\n";
  fs.close();
  return true;
}

bool ResultManager::SaveMattingResult(const vision::MattingResult &res,
                                      const std::string &path) {
  if (res.alpha.empty()) {
    FDERROR << "MattingResult can not be empty!" << std::endl;
    return false;
  }
  std::ofstream fs(path, std::ios::out);
  if (!fs.is_open()) {
    FDERROR << "Fail to open file:" << path << std::endl;
    return false;
  }
  fs.precision(20);
  // alpha
  fs << "alpha" << KEY_VALUE_SEP;
  for (int i = 0; i < res.alpha.size(); ++i) {
    if (i < res.alpha.size() - 1) {
      fs << res.alpha[i] << VALUE_SEP;
    } else {
      fs << res.alpha[i];
    }
  }
  fs << "\n";
  // foreground
  if (res.contain_foreground) {
    fs << "foreground" << KEY_VALUE_SEP;
    for (int i = 0; i < res.foreground.size(); ++i) {
      if (i < res.foreground.size() - 1) {
        fs << res.foreground[i] << VALUE_SEP;
      } else {
        fs << res.foreground[i];
      }
    }
    fs << "\n";
  }
  fs.close();
  return true;
}

bool ResultManager::LoadDetectionResult(vision::DetectionResult *res,
                                        const std::string &path) {
  if (!CheckFileExists(path)) {
    FDERROR << "Can't found file from " << path << std::endl;
    return false;
  }
  auto lines = ReadLines(path);
  std::map<std::string, std::vector<std::string>> data;

  // boxes
  data = SplitDataLine(lines[0]);
  int boxes_num = data.begin()->second.size() / 4;
  res->Resize(boxes_num);
  for (int i = 0; i < boxes_num; ++i) {
    res->boxes[i][0] = std::stof(data.begin()->second[i * 4 + 0]);
    res->boxes[i][1] = std::stof(data.begin()->second[i * 4 + 1]);
    res->boxes[i][2] = std::stof(data.begin()->second[i * 4 + 2]);
    res->boxes[i][3] = std::stof(data.begin()->second[i * 4 + 3]);
  }
  // scores
  data = SplitDataLine(lines[1]);
  for (int i = 0; i < data.begin()->second.size(); ++i) {
    res->scores[i] = std::stof(data.begin()->second[i]);
  }
  // label_ids
  data = SplitDataLine(lines[2]);
  for (int i = 0; i < data.begin()->second.size(); ++i) {
    res->label_ids[i] = std::stoi(data.begin()->second[i]);
  }
  // TODO(qiuyanjun): load masks
  return true;
}

bool ResultManager::LoadClassifyResult(vision::ClassifyResult *res,
                                       const std::string &path) {
  if (!CheckFileExists(path)) {
    FDERROR << "Can't found file from " << path << std::endl;
    return false;
  }
  auto lines = ReadLines(path);
  std::map<std::string, std::vector<std::string>> data;
  // label_ids
  data = SplitDataLine(lines[0]);
  res->Resize(data.begin()->second.size());
  for (int i = 0; i < data.begin()->second.size(); ++i) {
    res->label_ids[i] = std::stoi(data.begin()->second[i]);
  }
  // scores
  data = SplitDataLine(lines[1]);
  for (int i = 0; i < data.begin()->second.size(); ++i) {
    res->scores[i] = std::stof(data.begin()->second[i]);
  }
  return true;
}

bool ResultManager::LoadSegmentationResult(vision::SegmentationResult *res,
                                           const std::string &path) {
  if (!CheckFileExists(path)) {
    FDERROR << "Can't found file from " << path << std::endl;
    return false;
  }
  auto lines = ReadLines(path);
  if (lines.size() > 1) {
    res->contain_score_map = true;
  }
  std::map<std::string, std::vector<std::string>> data;
  // label_map
  data = SplitDataLine(lines[0]);
  res->Resize(data.begin()->second.size());
  for (int i = 0; i < data.begin()->second.size(); ++i) {
    res->label_map[i] = std::stoi(data.begin()->second[i]);
  }
  // score_map
  if (lines.size() > 1) {
    data = SplitDataLine(lines[1]);
    for (int i = 0; i < data.begin()->second.size(); ++i) {
      res->score_map[i] = std::stof(data.begin()->second[i]);
    }
  }
  return true;
}

bool ResultManager::LoadOCRDetResult(std::vector<std::array<int, 8>> *res,
                                     const std::string &path) {
  if (!CheckFileExists(path)) {
    FDERROR << "Can't found file from " << path << std::endl;
    return false;
  }
  auto lines = ReadLines(path);
  std::map<std::string, std::vector<std::string>> data;
  // boxes
  data = SplitDataLine(lines[0]);
  int boxes_num = data.begin()->second.size() / 8;
  res->resize(boxes_num);
  for (int i = 0; i < boxes_num; ++i) {
    for (int j = 0; j < 8; ++j) {
      (*res)[i][j] = std::stoi(data.begin()->second[i * 8 + j]);
    }
  }
  return true;
}

bool ResultManager::LoadMattingResult(vision::MattingResult *res,
                                      const std::string &path) {
  if (!CheckFileExists(path)) {
    FDERROR << "Can't found file from " << path << std::endl;
    return false;
  }
  auto lines = ReadLines(path);
  if (lines.size() > 1) {
    res->contain_foreground = true;
  }
  std::map<std::string, std::vector<std::string>> data;
  // alpha
  data = SplitDataLine(lines[0]);
  res->Resize(data.begin()->second.size());
  for (int i = 0; i < data.begin()->second.size(); ++i) {
    res->alpha[i] = std::stof(data.begin()->second[i]);
  }
  // foreground
  if (lines.size() > 1) {
    data = SplitDataLine(lines[1]);
    for (int i = 0; i < data.begin()->second.size(); ++i) {
      res->foreground[i] = std::stof(data.begin()->second[i]);
    }
  }
  return true;
}

DetectionDiff
ResultManager::CalculateDiffStatis(const vision::DetectionResult &lhs,
                                   const vision::DetectionResult &rhs,
                                   const float &score_threshold) {
  vision::DetectionResult lhs_sort = lhs;
  vision::DetectionResult rhs_sort = rhs;
  // lex sort by x(w) & y(h)
  vision::utils::LexSortDetectionResultByXY(&lhs_sort);
  vision::utils::LexSortDetectionResultByXY(&rhs_sort);
  // get value diff & trunc it by score_threshold
  const int boxes_num = std::min(lhs_sort.boxes.size(), rhs_sort.boxes.size());
  std::vector<float> boxes_diff;
  std::vector<float> scores_diff;
  std::vector<int32_t> labels_diff;
  // TODO(qiuyanjun): process the diff of masks.
  for (int i = 0; i < boxes_num; ++i) {
    if (lhs_sort.scores[i] > score_threshold &&
        rhs_sort.scores[i] > score_threshold) {
      scores_diff.push_back(lhs_sort.scores[i] - rhs_sort.scores[i]);
      labels_diff.push_back(lhs_sort.label_ids[i] - rhs_sort.label_ids[i]);
      boxes_diff.push_back(lhs_sort.boxes[i][0] - rhs_sort.boxes[i][0]);
      boxes_diff.push_back(lhs_sort.boxes[i][1] - rhs_sort.boxes[i][1]);
      boxes_diff.push_back(lhs_sort.boxes[i][2] - rhs_sort.boxes[i][2]);
      boxes_diff.push_back(lhs_sort.boxes[i][3] - rhs_sort.boxes[i][3]);
    }
  }
  FDASSERT(boxes_diff.size() > 0,
           "Can't get any valid boxes while score_threshold is %f, "
           "The boxes.size of lhs is %d, the boxes.size of rhs is %d",
           score_threshold, lhs_sort.boxes.size(), rhs_sort.boxes.size())

  DetectionDiff diff;
  CalculateStatisInfo<float>(boxes_diff.data(), boxes_diff.size(),
                             &(diff.boxes.mean), &(diff.boxes.max),
                             &(diff.boxes.min));
  CalculateStatisInfo<float>(scores_diff.data(), scores_diff.size(),
                             &(diff.scores.mean), &(diff.scores.max),
                             &(diff.scores.min));
  CalculateStatisInfo<int32_t>(labels_diff.data(), labels_diff.size(),
                               &(diff.labels.mean), &(diff.labels.max),
                               &(diff.labels.min));
  return diff;
}

ClassifyDiff
ResultManager::CalculateDiffStatis(const vision::ClassifyResult &lhs,
                                   const vision::ClassifyResult &rhs) {
  const int class_nums = std::min(lhs.label_ids.size(), rhs.label_ids.size());
  std::vector<float> scores_diff;
  std::vector<int32_t> labels_diff;
  for (int i = 0; i < class_nums; ++i) {
    scores_diff.push_back(lhs.scores[i] - rhs.scores[i]);
    labels_diff.push_back(lhs.label_ids[i] - rhs.label_ids[i]);
  }

  ClassifyDiff diff;
  CalculateStatisInfo<float>(scores_diff.data(), scores_diff.size(),
                             &(diff.scores.mean), &(diff.scores.max),
                             &(diff.scores.min));
  CalculateStatisInfo<int32_t>(labels_diff.data(), labels_diff.size(),
                               &(diff.labels.mean), &(diff.labels.max),
                               &(diff.labels.min));
  return diff;
}

SegmentationDiff
ResultManager::CalculateDiffStatis(const vision::SegmentationResult &lhs,
                                   const vision::SegmentationResult &rhs) {
  const int pixel_nums = std::min(lhs.label_map.size(), rhs.label_map.size());
  std::vector<int32_t> labels_diff;
  std::vector<float> scores_diff;
  for (int i = 0; i < pixel_nums; ++i) {
    labels_diff.push_back(lhs.label_map[i] - rhs.label_map[i]);
    if (lhs.contain_score_map && rhs.contain_score_map) {
      scores_diff.push_back(lhs.score_map[i] - rhs.score_map[i]);
    }
  }
  SegmentationDiff diff;
  CalculateStatisInfo<int32_t>(labels_diff.data(), labels_diff.size(),
                               &(diff.labels.mean), &(diff.labels.max),
                               &(diff.labels.min));
  if (lhs.contain_score_map && rhs.contain_score_map) {
    CalculateStatisInfo<float>(scores_diff.data(), scores_diff.size(),
                               &(diff.scores.mean), &(diff.scores.max),
                               &(diff.scores.min));
  }
  return diff;
}

OCRDetDiff
ResultManager::CalculateDiffStatis(const std::vector<std::array<int, 8>> &lhs,
                                   const std::vector<std::array<int, 8>> &rhs) {
  const int boxes_nums = std::min(lhs.size(), rhs.size());
  std::vector<std::array<int, 8>> lhs_sort = lhs;
  std::vector<std::array<int, 8>> rhs_sort = rhs;
  // lex sort by x(w) & y(h)
  vision::utils::LexSortOCRDetResultByXY(&lhs_sort);
  vision::utils::LexSortOCRDetResultByXY(&rhs_sort);
  // get value diff
  const int boxes_num = std::min(lhs_sort.size(), rhs_sort.size());
  std::vector<float> boxes_diff;
  for (int i = 0; i < boxes_num; ++i) {
    for (int j = 0; j < 8; ++j) {
      boxes_diff.push_back(lhs_sort[i][j] - rhs_sort[i][j]);
    }
  }

  OCRDetDiff diff;
  CalculateStatisInfo<float>(boxes_diff.data(), boxes_diff.size(),
                             &(diff.boxes.mean), &(diff.boxes.max),
                             &(diff.boxes.min));
  return diff;
}

MattingDiff
ResultManager::CalculateDiffStatis(const vision::MattingResult &lhs,
                                   const vision::MattingResult &rhs) {
  const int pixel_nums = std::min(lhs.alpha.size(), rhs.alpha.size());
  std::vector<float> alpha_diff;
  std::vector<float> foreground_diff;
  for (int i = 0; i < pixel_nums; ++i) {
    alpha_diff.push_back(lhs.alpha[i] - rhs.alpha[i]);
    if (lhs.contain_foreground && rhs.contain_foreground) {
      foreground_diff.push_back(lhs.foreground[i] - rhs.foreground[i]);
    }
  }
  MattingDiff diff;
  CalculateStatisInfo<float>(alpha_diff.data(), alpha_diff.size(),
                             &(diff.alpha.mean), &(diff.alpha.max),
                             &(diff.alpha.min));
  if (lhs.contain_foreground && rhs.contain_foreground) {
    CalculateStatisInfo<float>(foreground_diff.data(), foreground_diff.size(),
                               &(diff.foreground.mean), &(diff.foreground.max),
                               &(diff.foreground.min));
  }
  return diff;
}

#endif // ENABLE_VISION
#endif // ENABLE_BENCHMARK

} // namespace benchmark
} // namespace ultra_infer
