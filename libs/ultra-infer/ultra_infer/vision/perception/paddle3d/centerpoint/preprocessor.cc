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
#include "ultra_infer/vision/perception/paddle3d/centerpoint/preprocessor.h"

namespace ultra_infer {
namespace vision {
namespace perception {

CenterpointPreprocessor::CenterpointPreprocessor(
    const std::string &config_file) {
  initialized_ = true;
}

bool CenterpointPreprocessor::ReadPoint(const std::string &file_path,
                                        const int64_t num_point_dim,
                                        std::vector<float> &data,
                                        int64_t *num_points) {
  std::ifstream file_in(file_path, std::ios::in | std::ios::binary);
  if (num_point_dim < 4) {
    FDERROR << "Point dimension must not be less than 4, but received "
            << "num_point_dim is " << num_point_dim << std::endl;
  }

  if (!file_in) {
    FDERROR << "Failed to read file: " << file_path << std::endl;
    return false;
  }

  std::streampos file_size;
  file_in.seekg(0, std::ios::end);
  file_size = file_in.tellg();
  file_in.seekg(0, std::ios::beg);

  data.resize(file_size / sizeof(float));
  file_in.read(reinterpret_cast<char *>(data.data()), file_size);
  file_in.close();

  if (file_size / sizeof(float) % num_point_dim != 0) {
    FDERROR << "Loaded file size (" << file_size
            << ") is not evenly divisible by num_point_dim (" << num_point_dim
            << ")\n";
    return false;
  }
  *num_points = file_size / sizeof(float) / num_point_dim;
  return true;
}

bool CenterpointPreprocessor::InsertTimeToPoints(const int64_t num_points,
                                                 const int64_t num_point_dim,
                                                 float *points) {
  for (int64_t i = 0; i < num_points; ++i) {
    *(points + i * num_point_dim + 4) = 0.;
  }
  return true;
}

bool CenterpointPreprocessor::Apply(std::vector<std::string> &points_dir,
                                    const int64_t num_point_dim,
                                    const int with_timelag,
                                    std::vector<FDTensor> &outputs) {
  for (int index = 0; index < points_dir.size(); ++index) {
    std::string file_path = points_dir[index];
    std::vector<int64_t> points_shape;
    std::vector<float> data;
    int64_t num_points;
    if (!ReadPoint(file_path, num_point_dim, data, &num_points)) {
      return false;
    }
    float *points = data.data();

    if (!with_timelag && num_point_dim == 5 || num_point_dim > 5) {
      InsertTimeToPoints(num_points, num_point_dim, points);
    }
    points_shape.push_back(num_points);
    points_shape.push_back(num_point_dim);

    FDTensor tensor;
    tensor.SetData(points_shape, FDDataType::FP32, points, true);
    outputs.push_back(tensor);
  }
  return true;
}

bool CenterpointPreprocessor::Run(std::vector<std::string> &points_dir,
                                  const int64_t num_point_dim,
                                  const int with_timelag,
                                  std::vector<FDTensor> &outputs) {
  bool ret = Apply(points_dir, num_point_dim, with_timelag, outputs);
  return ret;
}

} // namespace perception
} // namespace vision
} // namespace ultra_infer
