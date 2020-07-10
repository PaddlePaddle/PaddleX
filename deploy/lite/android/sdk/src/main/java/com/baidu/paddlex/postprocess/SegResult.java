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

package com.baidu.paddlex.postprocess;

public class SegResult extends Result {
    static String type = "seg";
    protected Mask mask = new Mask();

    public Mask getMask() {
        return mask;
    }

    public void setMask(Mask mask) {
        this.mask = mask;
    }

    @Override
    public String getType() {
        return type;
    }

    public class Mask {
        protected float[] scoreData;
        protected long[] labelData;
        protected long[] labelShape = new long[4];
        protected long[] scoreShape = new long[4];

        public float[] getScoreData() {
            return scoreData;
        }

        public void setScoreData(float[] score_data) {
            this.scoreData = score_data;
        }

        public long[] getLabelData() {
            return labelData;
        }

        public void setLabelData(long[] label_data) {
            this.labelData = label_data;
        }

        public long[] getLabelShape() {
            return labelShape;
        }

        public void setLabelShape(long[] labelShape) {
            this.labelShape = labelShape;
        }

        public long[] getScoreShape() {
            return scoreShape;
        }

        public void setScoreShape(long[] scoreShape) {
            this.scoreShape = scoreShape;
        }
    }
}
