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

#include "ultra_infer/function/slice.h"
#include "ultra_infer/function/eigen.h"

#include <algorithm>

namespace ultra_infer {
namespace function {

std::vector<int64_t> GetSliceDims(const std::vector<int64_t> &in_dims,
                                  const std::vector<int64_t> &axes,
                                  const std::vector<int64_t> &starts,
                                  const std::vector<int64_t> &ends,
                                  std::vector<int64_t> *steps = nullptr) {
  std::vector<int64_t> slice_dims(in_dims);

  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    if (in_dims[axis] == -1) {
      continue;
    }

    int64_t start = starts[i];
    int64_t end = ends[i];
    int64_t step = steps == nullptr ? 1 : (*steps)[i];

    if (step > 0) {
      slice_dims[axis] = (end - start + step - 1) / step;
    } else {
      slice_dims[axis] = (end - start + step + 1) / step;
    }
  }
  return slice_dims;
}

void CheckAndUpdateSliceAttrs(const std::vector<int64_t> &in_dims,
                              const std::vector<int64_t> &axes,
                              std::vector<int64_t> *starts,
                              std::vector<int64_t> *ends,
                              std::vector<int64_t> *steps = nullptr) {
  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    FDASSERT(axis < in_dims.size(),
             "The axis value should be less than the rank of input, "
             "but received axes[%d] = %d, rank of input is %d.",
             i, axis, in_dims.size());
    int64_t dim_value = in_dims[axis];

    if (dim_value > 0) {
      int64_t step = steps == nullptr ? 1 : (*steps)[i];
      FDASSERT(step != 0, "Step should not be 0, but received step = %d.",
               step);
      int64_t start =
          (*starts)[i] < 0 ? ((*starts)[i] + dim_value) : (*starts)[i];
      start = (std::max)(start, static_cast<int64_t>(0));

      int64_t end =
          0 < step && (*ends)[i] < 0 ? ((*ends)[i] + dim_value) : (*ends)[i];
      end = (std::min)(end, dim_value);

      if (step > 0) {
        start = (std::min)(start, dim_value);
        end = (std::max)(end, static_cast<int64_t>(0));
        FDASSERT(end > start,
                 "When step > 0, end should be greater than start, but "
                 "received end = %d, start = %d.",
                 end, start)
      } else {
        start = (std::min)(start, dim_value - 1);
        if (end < -1) {
          end += dim_value;
        }
        end = (std::max)(end, static_cast<int64_t>(-1));
        FDASSERT(start >= end,
                 "When step < 0, start should be greater than end, but "
                 "received start = %d, end = %d.",
                 start, end);
      }

      (*starts)[i] = start;
      (*ends)[i] = end;
    } else if (dim_value == 0) {
      (*starts)[i] = 0;
      (*ends)[i] = 0;
    }
  }
}

template <typename T, size_t D>
void SliceKernel(const FDTensor &x, const std::vector<int64_t> &axes,
                 const std::vector<int64_t> &starts,
                 const std::vector<int64_t> &ends, FDTensor *out) {
  FDASSERT(starts.size() == axes.size(),
           "The size of starts must be equal to the size of axes.");
  FDASSERT(ends.size() == axes.size(),
           "The size of ends must be equal to the size of axes.");
  auto starts_idx = starts;
  auto end_idx = ends;
  auto in_dims = x.Shape();
  CheckAndUpdateSliceAttrs(in_dims, axes, &starts_idx, &end_idx);
  auto slice_dims = GetSliceDims(in_dims, axes, starts, ends);

  auto offsets = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto extents = Eigen::DSizes<Eigen::DenseIndex, D>();
  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = slice_dims[i];
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    offsets[axes[i]] = starts[i];
  }

  out->Allocate(slice_dims, x.Dtype());
  auto in_t = EigenTensor<T, D>::From(x, in_dims);
  auto out_t = EigenTensor<T, D>::From(*out, slice_dims);
  const auto &dev = *EigenDeviceWrapper::GetInstance()->GetDevice();
  out_t.device(dev) = in_t.slice(offsets, extents);
}

void Slice(const FDTensor &x, const std::vector<int64_t> &axes,
           const std::vector<int64_t> &starts, const std::vector<int64_t> &ends,
           FDTensor *out) {
  FD_VISIT_ALL_TYPES(
      x.dtype, "SliceKernel", ([&] {
        int rank = x.Shape().size();
        switch (rank) {
        case 1:
          SliceKernel<data_t, 1>(x, axes, starts, ends, out);
          break;
        case 2:
          SliceKernel<data_t, 2>(x, axes, starts, ends, out);
          break;
        case 3:
          SliceKernel<data_t, 3>(x, axes, starts, ends, out);
          break;
        case 4:
          SliceKernel<data_t, 4>(x, axes, starts, ends, out);
          break;
        case 5:
          SliceKernel<data_t, 5>(x, axes, starts, ends, out);
          break;
        case 6:
          SliceKernel<data_t, 6>(x, axes, starts, ends, out);
          break;
        default:
          FDASSERT(false,
                   "The rank of input should be less than 7, but received %d.",
                   rank);
        }
      }));
}

void Slice(const FDTensor &x, const std::vector<int64_t> &axes,
           const std::vector<int64_t> &index, FDTensor *out) {
  std::vector<int64_t> ends = index;
  for (int i = 0; i < ends.size(); ++i) {
    ends[i] += 1;
  }
  Slice(x, axes, index, ends, out);
  for (int i = 0; i < axes.size(); ++i) {
    if (out->Shape().size() <= 1) {
      break;
    }
    out->Squeeze(axes[i]);
  }
}

} // namespace function
} // namespace ultra_infer
