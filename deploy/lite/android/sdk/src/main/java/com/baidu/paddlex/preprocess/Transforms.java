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

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.util.Log;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

public class Transforms {
    private static final String TAG = Transforms.class.getSimpleName();
    private List<transform_op> transform_ops = new ArrayList<transform_op>();

    public void load_config(List transforms_list, String transformsMode) {
        for (int i = 0; i < transforms_list.size(); i++) {
            HashMap transform_op = (HashMap) (transforms_list.get(i));
            if (transform_op.containsKey("ResizeByShort")) {
                HashMap<String, Integer> info = (HashMap<String, Integer>) transform_op.get("ResizeByShort");
                ResizeByShort resizeByShort = new ResizeByShort();
                resizeByShort.max_size = info.get("max_size");
                resizeByShort.short_size = info.get("short_size");
                transform_ops.add(resizeByShort);
            } else if (transform_op.containsKey("ResizeByLong")) {
                HashMap<String, Integer> info = (HashMap<String, Integer>) transform_op.get("ResizeByLong");
                ResizeByLong resizeByLong = new ResizeByLong();
                resizeByLong.long_size = info.get("long_size");
                transform_ops.add(resizeByLong);
            } else if (transform_op.containsKey("CenterCrop")) {
                HashMap info = (HashMap) transform_op.get("CenterCrop");
                CenterCrop centerCrop = new CenterCrop();
                if (info.get("crop_size") instanceof Integer) {
                    centerCrop.cropHeight = (int) info.get("crop_size");
                    centerCrop.cropWidth = (int) info.get("crop_size");
                } else {
                    centerCrop.cropWidth = ((List<Integer>) info.get("crop_size")).get(0);
                    centerCrop.cropHeight = ((List<Integer>) info.get("crop_size")).get(1);
                }
                transform_ops.add(centerCrop);
            } else if (transform_op.containsKey("Normalize")) {
                HashMap<String, List<Float>> info = (HashMap<String, List<Float>>) transform_op.get("Normalize");
                Normalize normalize = new Normalize();
                normalize.transformsMode = transformsMode;
                normalize.mean = info.get("mean").toArray(new Double[info.get("mean").size()]);
                normalize.std = info.get("std").toArray(new Double[info.get("std").size()]);
                ;
                transform_ops.add(normalize);
            } else if (transform_op.containsKey("Resize")) {
                HashMap info = (HashMap) transform_op.get("Resize");
                Resize resize = new Resize();
                if (info.get("target_size") instanceof Integer) {
                    resize.width = (int) info.get("target_size");
                    resize.height = (int) info.get("target_size");
                } else {
                    resize.width = ((List<Integer>) info.get("target_size")).get(0);
                    resize.height = ((List<Integer>) info.get("target_size")).get(1);
                }
                transform_ops.add(resize);
            } else if (transform_op.containsKey("Padding")) {
                HashMap info = (HashMap) transform_op.get("Padding");
                Padding padding = new Padding();
                if (info.containsKey("coarsest_stride")) {
                    padding.coarsest_stride = (int) info.get("coarsest_stride");
                }
                if (info.containsKey("target_size")) {
                    if (info.get("target_size") instanceof Integer) {
                        padding.width = (int) info.get("target_size");
                        padding.height = (int) info.get("target_size");
                    } else {
                        padding.width = ((List<Integer>) info.get("target_size")).get(0);
                        padding.height = ((List<Integer>) info.get("target_size")).get(1);
                    }
                }
                transform_ops.add(padding);
            }
        }
    }

    public ImageBlob run(Bitmap inputImage, ImageBlob imageBlob) {
        imageBlob.ori_im_size_[2] = inputImage.getHeight();
        imageBlob.ori_im_size_[3] = inputImage.getWidth();
        imageBlob.new_im_size_[2] = inputImage.getHeight();
        imageBlob.new_im_size_[3] = inputImage.getWidth();
        for (transform_op op : transform_ops) {
            inputImage = op.run(inputImage, imageBlob);
        }
        float avg = 0;
        for (int i = 0; i < imageBlob.im_data_.length; i = i + 1) {
            avg += imageBlob.im_data_[i];

        }
        return imageBlob;
    }

    private class transform_op {
        public Bitmap run(Bitmap inputImage, ImageBlob data) {
            return inputImage;
        }

        ;
    }

    private class Resize extends transform_op {
        public int height;
        public int width;

        public Bitmap run(Bitmap inputImage, ImageBlob data) {
            int origin_w = inputImage.getWidth();
            int origin_h = inputImage.getHeight();
            data.reshape_info_.put("resize", new int[]{origin_w, origin_h});
            inputImage = Bitmap.createScaledBitmap(inputImage, width, height, true);
            data.new_im_size_[2] = inputImage.getHeight();
            data.new_im_size_[3] = inputImage.getWidth();
            return inputImage;
        }
    }

    private class ResizeByShort extends transform_op {
        public int max_size;
        public int short_size;

        public Bitmap run(Bitmap inputImage, ImageBlob data) {
            int origin_w = inputImage.getWidth();
            int origin_h = inputImage.getHeight();
            data.reshape_info_.put("resize", new int[]{origin_w, origin_h});
            int im_size_max = Math.max(origin_w, origin_h);
            int im_size_min = Math.min(origin_w, origin_h);
            float scale = (float) (short_size) / (float) (im_size_min);
            if (max_size > 0) {
                if (Math.round(scale * im_size_max) > max_size) {
                    scale = (float) (max_size) / (float) (im_size_max);
                }
            }
            int width = Math.round(scale * origin_w);
            int height = Math.round(scale * origin_h);
            inputImage = Bitmap.createScaledBitmap(inputImage, width, height, true);

            data.new_im_size_[2] = inputImage.getHeight();
            data.new_im_size_[3] = inputImage.getWidth();
            data.scale = scale;
            return inputImage;
        }
    }

    private class ResizeByLong extends transform_op {
        public int long_size;

        public Bitmap run(Bitmap inputImage, ImageBlob data) {
            int origin_w = inputImage.getWidth();
            int origin_h = inputImage.getHeight();
            data.reshape_info_.put("resize", new int[]{origin_w, origin_h});

            int im_size_max = Math.max(origin_w, origin_h);
            float scale = (float) (long_size) / (float) (im_size_max);
            int width = Math.round(scale * origin_w);
            int height = Math.round(scale * origin_h);
            inputImage = Bitmap.createScaledBitmap(inputImage, width, height, true);

            data.new_im_size_[2] = inputImage.getHeight();
            data.new_im_size_[3] = inputImage.getWidth();
            data.scale = scale;
            return inputImage;
        }
    }

    private class CenterCrop extends transform_op {
        public int cropHeight;
        public int cropWidth;

        public Bitmap run(Bitmap inputImage, ImageBlob data) {
            int origin_w = inputImage.getWidth();
            int origin_h = inputImage.getHeight();
            if (origin_h < cropHeight || origin_w < cropWidth) {
                Log.e(TAG, "[CenterCrop] Image size less than crop size");
            }
            final Matrix m = new Matrix();
            final float scale = Math.max(
                    (float) cropWidth / origin_h,
                    (float) cropHeight / origin_h);
            m.setScale(scale, scale);
            int srcX, srcY;
            srcX = (int) ((origin_w - cropWidth) / 2);
            srcY = (int) ((origin_h - cropHeight) / 2);
            srcX = Math.max(Math.min(srcX, origin_w - cropWidth), 0);
            srcY = Math.max(Math.min(srcY, origin_h - cropHeight), 0);
            inputImage = Bitmap.createBitmap(inputImage, srcX, srcY, cropWidth, cropHeight);

            data.new_im_size_[2] = inputImage.getHeight();
            data.new_im_size_[3] = inputImage.getWidth();

            return inputImage;
        }
    }

    private class Normalize extends transform_op {
        String transformsMode = "RGB";
        private Double[] mean = new Double[3];
        private Double[] std = new Double[3];

        public Bitmap run(Bitmap inputImage, ImageBlob data) {
            int w = inputImage.getWidth();
            int h = inputImage.getHeight();
            data.im_data_ = new float[w * h * 3];
            int[] channelIdx = null;
            if (transformsMode.equalsIgnoreCase("RGB")) {
                channelIdx = new int[]{0, 1, 2};
                Log.i(TAG, " color format " + transformsMode + "!!");
            } else if (transformsMode.equalsIgnoreCase("BGR")) {
                Log.i(TAG, " color format " + transformsMode + "!!");
                channelIdx = new int[]{2, 1, 0};
            } else {
                Log.e(TAG, "unknown color format " + transformsMode + ", only RGB and BGR color format is " + "supported!");
            }

            int[] channelStride = new int[]{w * h, w * h * 2};
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int color = inputImage.getPixel(x, y);
                    float[] rgb = new float[]{(float) red(color), (float) green(color), (float) blue(color)};
                    data.im_data_[y * w + x] = (rgb[channelIdx[0]] / 255 - mean[0].floatValue()) / (std[0].floatValue());
                    data.im_data_[y * w + x + channelStride[0]] = (rgb[channelIdx[1]] / 255 - mean[1].floatValue()) / std[1].floatValue();
                    data.im_data_[y * w + x + channelStride[1]] = (rgb[channelIdx[2]] / 255 - mean[2].floatValue()) / std[2].floatValue();
                }
            }
            return inputImage;
        }
    }

    private class Padding extends transform_op {
        public double width;
        public double height;
        public double coarsest_stride;

        public Bitmap run(Bitmap inputImage, ImageBlob data) {
            int origin_w = inputImage.getWidth();
            int origin_h = inputImage.getHeight();
            data.reshape_info_.put("padding", new int[]{origin_w, origin_h});
            double padding_w = 0;
            double padding_h = 0;

            if (width > 1 & height > 1) {
                padding_w = width;
                padding_h = height;
            } else if (coarsest_stride > 1) {
                padding_h = Math.ceil(origin_h / coarsest_stride) * coarsest_stride;
                padding_w = Math.ceil(origin_w / coarsest_stride) * coarsest_stride;
            }

            Bitmap outputImage = Bitmap.createBitmap((int) padding_w, (int) padding_h, inputImage.getConfig());
            Canvas canvas = new Canvas(outputImage);
            canvas.drawColor(Color.rgb(127, 127, 127));
            canvas.drawBitmap(inputImage, 0, 0, null);
            data.new_im_size_[2] = outputImage.getHeight();
            data.new_im_size_[3] = outputImage.getWidth();
            return outputImage;
        }
    }
}
