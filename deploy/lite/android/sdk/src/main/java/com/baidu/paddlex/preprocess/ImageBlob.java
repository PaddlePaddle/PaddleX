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
    private long[] oriImageSize = new long[]{1, 3, -1, -1};
    // Newest image height and width after process
    private long[] newImageSize = new long[]{1, 3, -1, -1};
    // Reshape order, Image height and width before resize
    private LinkedHashMap<String, int[]> reshapeInfo = new LinkedHashMap<String, int[]>();
    // Resize scale
    private float scale = 1;
    // Buffer for image data after preprocessing
    private float[] imageData;

    public void clear() {
        oriImageSize = new long[]{1, 3, -1, -1};
        newImageSize = new long[]{1, 3, -1, -1};
        reshapeInfo.clear();
        imageData = null;
    }

    public long[] getOriImageSize() {
        return oriImageSize;
    }

    public void setOriImageSize(long[] oriImageSize) {
        this.oriImageSize = oriImageSize;
    }

    public void setOriImageSize(long dim, int idx) {
        this.oriImageSize[idx] = dim;
    }

    public long[] getNewImageSize() {
        return newImageSize;
    }

    public void setNewImageSize(long[] newImageSize) {
        this.newImageSize = newImageSize;
    }

    public void setNewImageSize(long dim, int idx) {
        this.newImageSize[idx] = dim;
    }


    public LinkedHashMap<String, int[]> getReshapeInfo() {
        return reshapeInfo;
    }

    public void setReshapeInfo(LinkedHashMap<String, int[]> reshapeInfo) {
        this.reshapeInfo = reshapeInfo;
    }

    public float getScale() {
        return scale;
    }

    public void setScale(float scale) {
        this.scale = scale;
    }

    public float[] getImageData() {
        return imageData;
    }

    public void setImageData(float[] imageData) {
        this.imageData = imageData;
    }
}