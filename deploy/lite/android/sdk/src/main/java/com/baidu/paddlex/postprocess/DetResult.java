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

import java.util.ArrayList;
import java.util.List;

public class DetResult extends Result {
    static String type = "det";
    protected List<Box> boxes = new ArrayList<Box>();

    public List<Box> getBoxes() {
        return boxes;
    }

    public void setBoxes(List<Box> boxes) {
        this.boxes = boxes;
    }

    @Override
    public String getType() {
        return type;
    }

    public class Box {
        protected int categoryId;
        protected String category;
        protected float score;
        protected float[] coordinate = new float[4];

        public int getCategoryId() {
            return categoryId;
        }

        public void setCategoryId(int category_id) {
            this.categoryId = category_id;
        }

        public String getCategory() {
            return category;
        }

        public void setCategory(String category) {
            this.category = category;
        }

        public float getScore() {
            return score;
        }

        public void setScore(float score) {
            this.score = score;
        }

        public float[] getCoordinate() {
            return coordinate;
        }

        public void setCoordinate(float[] coordinate) {
            this.coordinate = coordinate;
        }
    }

}
