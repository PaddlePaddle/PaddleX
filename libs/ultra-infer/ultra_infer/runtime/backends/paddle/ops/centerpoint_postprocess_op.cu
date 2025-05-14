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

#if defined(PADDLEINFERENCE_API_COMPAT_2_4_x)
#include "paddle/include/experimental/ext_all.h"
#elif defined(PADDLEINFERENCE_API_COMPAT_2_5_x)
#include "paddle/include/paddle/extension.h"
#else
#include "paddle/extension.h"
#endif

namespace ultra_infer {
namespace paddle_custom_ops {

#define CHECK_INPUT_CUDA(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

#define CHECK_INPUT_BATCHSIZE(x)                                               \
  PD_CHECK(x.shape()[0] == 1, #x " batch size must be 1.")

// #define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
__host__ __device__ static inline int DIVUP(const int m, const int n) {
  return ((m) / (n) + ((m) % (n) > 0));
}

static const int THREADS_PER_BLOCK_NMS = sizeof(int64_t) * 8;

void NmsLauncher(const cudaStream_t &stream, const float *bboxes,
                 const int *index, const int64_t *sorted_index,
                 const int num_bboxes, const int num_bboxes_for_nms,
                 const float nms_overlap_thresh, const int decode_bboxes_dims,
                 int64_t *mask);

__global__ void decode_kernel(
    const float *score, const float *reg, const float *height, const float *dim,
    const float *vel, const float *rot, const float score_threshold,
    const int feat_w, const float down_ratio, const float voxel_size_x,
    const float voxel_size_y, const float point_cloud_range_x_min,
    const float point_cloud_range_y_min, const float post_center_range_x_min,
    const float post_center_range_y_min, const float post_center_range_z_min,
    const float post_center_range_x_max, const float post_center_range_y_max,
    const float post_center_range_z_max, const int num_bboxes,
    const bool with_velocity, const int decode_bboxes_dims, float *bboxes,
    bool *mask, int *score_idx) {
  int box_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (box_idx == num_bboxes || box_idx > num_bboxes) {
    return;
  }
  const int xs = box_idx % feat_w;
  const int ys = box_idx / feat_w;

  float x = reg[box_idx];
  float y = reg[box_idx + num_bboxes];
  float z = height[box_idx];

  bboxes[box_idx * decode_bboxes_dims] =
      (x + xs) * down_ratio * voxel_size_x + point_cloud_range_x_min;
  bboxes[box_idx * decode_bboxes_dims + 1] =
      (y + ys) * down_ratio * voxel_size_y + point_cloud_range_y_min;
  bboxes[box_idx * decode_bboxes_dims + 2] = z;
  bboxes[box_idx * decode_bboxes_dims + 3] = dim[box_idx];
  bboxes[box_idx * decode_bboxes_dims + 4] = dim[box_idx + num_bboxes];
  bboxes[box_idx * decode_bboxes_dims + 5] = dim[box_idx + 2 * num_bboxes];
  if (with_velocity) {
    bboxes[box_idx * decode_bboxes_dims + 6] = vel[box_idx];
    bboxes[box_idx * decode_bboxes_dims + 7] = vel[box_idx + num_bboxes];
    bboxes[box_idx * decode_bboxes_dims + 8] =
        atan2f(rot[box_idx], rot[box_idx + num_bboxes]);
  } else {
    bboxes[box_idx * decode_bboxes_dims + 6] =
        atan2f(rot[box_idx], rot[box_idx + num_bboxes]);
  }

  if (score[box_idx] > score_threshold && x <= post_center_range_x_max &&
      y <= post_center_range_y_max && z <= post_center_range_z_max &&
      x >= post_center_range_x_min && y >= post_center_range_y_min &&
      z >= post_center_range_z_min) {
    mask[box_idx] = true;
  }

  score_idx[box_idx] = box_idx;
}

void DecodeLauncher(
    const cudaStream_t &stream, const float *score, const float *reg,
    const float *height, const float *dim, const float *vel, const float *rot,
    const float score_threshold, const int feat_w, const float down_ratio,
    const float voxel_size_x, const float voxel_size_y,
    const float point_cloud_range_x_min, const float point_cloud_range_y_min,
    const float post_center_range_x_min, const float post_center_range_y_min,
    const float post_center_range_z_min, const float post_center_range_x_max,
    const float post_center_range_y_max, const float post_center_range_z_max,
    const int num_bboxes, const bool with_velocity,
    const int decode_bboxes_dims, float *bboxes, bool *mask, int *score_idx) {
  dim3 blocks(DIVUP(num_bboxes, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);
  decode_kernel<<<blocks, threads, 0, stream>>>(
      score, reg, height, dim, vel, rot, score_threshold, feat_w, down_ratio,
      voxel_size_x, voxel_size_y, point_cloud_range_x_min,
      point_cloud_range_y_min, post_center_range_x_min, post_center_range_y_min,
      post_center_range_z_min, post_center_range_x_max, post_center_range_y_max,
      post_center_range_z_max, num_bboxes, with_velocity, decode_bboxes_dims,
      bboxes, mask, score_idx);
}

std::vector<paddle::Tensor>
postprocess_gpu(const std::vector<paddle::Tensor> &hm,
                const std::vector<paddle::Tensor> &reg,
                const std::vector<paddle::Tensor> &height,
                const std::vector<paddle::Tensor> &dim,
                const std::vector<paddle::Tensor> &vel,
                const std::vector<paddle::Tensor> &rot,
                const std::vector<float> &voxel_size,
                const std::vector<float> &point_cloud_range,
                const std::vector<float> &post_center_range,
                const std::vector<int> &num_classes, const int down_ratio,
                const float score_threshold, const float nms_iou_threshold,
                const int nms_pre_max_size, const int nms_post_max_size,
                const bool with_velocity) {
  int num_tasks = hm.size();
  int decode_bboxes_dims = 9;
  if (!with_velocity) {
    decode_bboxes_dims = 7;
  }
  float voxel_size_x = voxel_size[0];
  float voxel_size_y = voxel_size[1];
  float point_cloud_range_x_min = point_cloud_range[0];
  float point_cloud_range_y_min = point_cloud_range[1];

  float post_center_range_x_min = post_center_range[0];
  float post_center_range_y_min = post_center_range[1];
  float post_center_range_z_min = post_center_range[2];
  float post_center_range_x_max = post_center_range[3];
  float post_center_range_y_max = post_center_range[4];
  float post_center_range_z_max = post_center_range[5];
  std::vector<paddle::Tensor> scores;
  std::vector<paddle::Tensor> labels;
  std::vector<paddle::Tensor> bboxes;
  for (int task_id = 0; task_id < num_tasks; ++task_id) {
    CHECK_INPUT_BATCHSIZE(hm[0]);

    int feat_h = hm[0].shape()[2];
    int feat_w = hm[0].shape()[3];
    int num_bboxes = feat_h * feat_w;

    // score and label
    auto sigmoid_hm_per_task = paddle::experimental::sigmoid(hm[task_id]);
    auto label_per_task =
        paddle::experimental::argmax(sigmoid_hm_per_task, 1, true, false);
    auto score_per_task =
        paddle::experimental::max(sigmoid_hm_per_task, {1}, true);
    // dim
    auto exp_dim_per_task = paddle::experimental::exp(dim[task_id]);

    // decode bboxed and get mask of bboxes for nms
    const float *score_ptr = score_per_task.data<float>();
    const float *reg_ptr = reg[task_id].data<float>();
    const float *height_ptr = height[task_id].data<float>();
    // const float* dim_ptr = dim[task_id].data<float>();
    const float *exp_dim_per_task_ptr = exp_dim_per_task.data<float>();
    const float *vel_ptr = vel[task_id].data<float>();
    const float *rot_ptr = rot[task_id].data<float>();
    auto decode_bboxes =
        paddle::empty({num_bboxes, decode_bboxes_dims},
                      paddle::DataType::FLOAT32, paddle::GPUPlace());
    float *decode_bboxes_ptr = decode_bboxes.data<float>();
    auto thresh_mask = paddle::full({num_bboxes}, 0, paddle::DataType::BOOL,
                                    paddle::GPUPlace());
    bool *thresh_mask_ptr = thresh_mask.data<bool>();
    auto score_idx = paddle::empty({num_bboxes}, paddle::DataType::INT32,
                                   paddle::GPUPlace());
    int *score_idx_ptr = score_idx.data<int32_t>();

    DecodeLauncher(score_per_task.stream(), score_ptr, reg_ptr, height_ptr,
                   exp_dim_per_task_ptr, vel_ptr, rot_ptr, score_threshold,
                   feat_w, down_ratio, voxel_size_x, voxel_size_y,
                   point_cloud_range_x_min, point_cloud_range_y_min,
                   post_center_range_x_min, post_center_range_y_min,
                   post_center_range_z_min, post_center_range_x_max,
                   post_center_range_y_max, post_center_range_z_max, num_bboxes,
                   with_velocity, decode_bboxes_dims, decode_bboxes_ptr,
                   thresh_mask_ptr, score_idx_ptr);

    // select score by mask
    auto selected_score_idx =
        paddle::experimental::masked_select(score_idx, thresh_mask);
    auto flattened_selected_score =
        paddle::experimental::reshape(score_per_task, {num_bboxes});
    auto selected_score = paddle::experimental::masked_select(
        flattened_selected_score, thresh_mask);
    int num_selected = selected_score.numel();
    if (num_selected == 0 || num_selected < 0) {
      auto fake_out_boxes =
          paddle::full({1, decode_bboxes_dims}, 0., paddle::DataType::FLOAT32,
                       paddle::GPUPlace());
      auto fake_out_score =
          paddle::full({1}, -1., paddle::DataType::FLOAT32, paddle::GPUPlace());
      auto fake_out_label =
          paddle::full({1}, 0, paddle::DataType::INT64, paddle::GPUPlace());
      scores.push_back(fake_out_score);
      labels.push_back(fake_out_label);
      bboxes.push_back(fake_out_boxes);
      continue;
    }

    // sort score by descending
    auto sort_out = paddle::experimental::argsort(selected_score, 0, true);
    auto sorted_index = std::get<1>(sort_out);
    int num_bboxes_for_nms =
        num_selected > nms_pre_max_size ? nms_pre_max_size : num_selected;

    // nms
    // in NmsLauncher, rot = - theta - pi / 2
    int col_blocks = DIVUP(num_bboxes_for_nms, THREADS_PER_BLOCK_NMS);
    auto nms_mask = paddle::empty({num_bboxes_for_nms * col_blocks},
                                  paddle::DataType::INT64, paddle::GPUPlace());
    int64_t *nms_mask_data = nms_mask.data<int64_t>();

    NmsLauncher(score_per_task.stream(), decode_bboxes.data<float>(),
                selected_score_idx.data<int>(), sorted_index.data<int64_t>(),
                num_selected, num_bboxes_for_nms, nms_iou_threshold,
                decode_bboxes_dims, nms_mask_data);

    const paddle::Tensor nms_mask_cpu_tensor =
        nms_mask.copy_to(paddle::CPUPlace(), true);
    const int64_t *nms_mask_cpu = nms_mask_cpu_tensor.data<int64_t>();

    auto remv_cpu = paddle::full({col_blocks}, 0, paddle::DataType::INT64,
                                 paddle::CPUPlace());
    int64_t *remv_cpu_data = remv_cpu.data<int64_t>();
    int num_to_keep = 0;
    auto keep = paddle::empty({num_bboxes_for_nms}, paddle::DataType::INT32,
                              paddle::CPUPlace());
    int *keep_data = keep.data<int>();

    for (int i = 0; i < num_bboxes_for_nms; i++) {
      int nblock = i / THREADS_PER_BLOCK_NMS;
      int inblock = i % THREADS_PER_BLOCK_NMS;

      if (!(remv_cpu_data[nblock] & (1ULL << inblock))) {
        keep_data[num_to_keep++] = i;
        const int64_t *p = &nms_mask_cpu[0] + i * col_blocks;
        for (int j = nblock; j < col_blocks; j++) {
          remv_cpu_data[j] |= p[j];
        }
      }
    }

    int num_for_gather =
        num_to_keep > nms_post_max_size ? nms_post_max_size : num_to_keep;
    auto keep_gpu = paddle::empty({num_for_gather}, paddle::DataType::INT32,
                                  paddle::GPUPlace());
    int *keep_gpu_ptr = keep_gpu.data<int>();
    cudaMemcpy(keep_gpu_ptr, keep_data, num_for_gather * sizeof(int),
               cudaMemcpyHostToDevice);

    auto gather_sorted_index =
        paddle::experimental::gather(sorted_index, keep_gpu, 0);
    auto gather_index = paddle::experimental::gather(selected_score_idx,
                                                     gather_sorted_index, 0);

    auto gather_score =
        paddle::experimental::gather(selected_score, gather_sorted_index, 0);
    auto flattened_label =
        paddle::experimental::reshape(label_per_task, {num_bboxes});
    auto gather_label =
        paddle::experimental::gather(flattened_label, gather_index, 0);
    auto gather_bbox =
        paddle::experimental::gather(decode_bboxes, gather_index, 0);
    auto start_label = paddle::full(
        {1}, num_classes[task_id], paddle::DataType::INT64, paddle::GPUPlace());
    auto added_label = paddle::experimental::add(gather_label, start_label);
    scores.push_back(gather_score);
    labels.push_back(added_label);
    bboxes.push_back(gather_bbox);
  }

  auto out_scores = paddle::experimental::concat(scores, 0);
  auto out_labels = paddle::experimental::concat(labels, 0);
  auto out_bboxes = paddle::experimental::concat(bboxes, 0);
  return {out_bboxes, out_scores, out_labels};
}

} // namespace paddle_custom_ops
} // namespace ultra_infer
