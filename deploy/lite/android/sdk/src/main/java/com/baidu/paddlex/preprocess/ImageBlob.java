// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

package com.baidu.paddlex.preprocess;

import java.util.LinkedHashMap;

public class ImageBlob {
    // Original image height and width
    public long[] ori_im_size_ = new long[]{1, 3, -1, -1};
    // Newest image height and width after process

    public long[] new_im_size_ = new long[]{1, 3, -1, -1};

    // Reshape order, Image height and width before resize
    public LinkedHashMap<String, int[]> reshape_info_ = new LinkedHashMap<String, int[]>();

    // Resize scale
    public float scale = 1;

    // Buffer for image data after preprocessing
    public float[] im_data_;

    public void clear() {
        ori_im_size_ = null;
        new_im_size_ = null;
        reshape_info_.clear();
        im_data_ = null;
    }
};