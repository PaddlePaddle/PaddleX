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
import android.util.Log;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

public class Transforms {
    private static final String TAG = Transforms.class.getSimpleName();
    private List<transformOp> transformOps = new ArrayList<transformOp>();
    private String transformsMode = "RGB";
	private HashMap<String, Integer> interpMap = new HashMap<String, Integer>(){{
        put("LINEAR", Imgproc.INTER_LINEAR);
        put("NEAREST", Imgproc.INTER_NEAREST);
        put("AREA", Imgproc.INTER_AREA);
        put("CUBIC", Imgproc.INTER_CUBIC);
        put("LANCZOS4", Imgproc.INTER_LANCZOS4);
        }
    };

    public void loadConfig(List transforms_list, String transformsMode) {
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG,"OpenCV Loadding failed.");
        }
        this.transformsMode = transformsMode;
        for (int i = 0; i < transforms_list.size(); i++) {
            HashMap transform_op = (HashMap) (transforms_list.get(i));
            if (transform_op.containsKey("ResizeByShort")) {
                HashMap info = (HashMap) transform_op.get("ResizeByShort");
                ResizeByShort resizeByShort = new ResizeByShort();
                resizeByShort.max_size = (int)info.get("max_size");
                resizeByShort.short_size = (int)info.get("short_size");
                if (info.containsKey("interp")) {
                    resizeByShort.interp = (String) info.get("interp");
                }
                transformOps.add(resizeByShort);
            } else if (transform_op.containsKey("ResizeByLong")) {
                HashMap info = (HashMap) transform_op.get("ResizeByLong");
                ResizeByLong resizeByLong = new ResizeByLong();
                resizeByLong.long_size = (int)info.get("long_size");
                if (info.containsKey("interp")) {
                    resizeByLong.interp = (String) info.get("interp");
                }
                transformOps.add(resizeByLong);

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
                transformOps.add(centerCrop);
            } else if (transform_op.containsKey("Normalize")) {
                HashMap<String, List<Float>> info = (HashMap<String, List<Float>>) transform_op.get("Normalize");
                Normalize normalize = new Normalize();
                normalize.mean = info.get("mean").toArray(new Double[info.get("mean").size()]);
                normalize.std = info.get("std").toArray(new Double[info.get("std").size()]);
                transformOps.add(normalize);
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
                if (info.containsKey("interp")) {
                    resize.interp = (String) info.get("interp");
                }
                transformOps.add(resize);
            } else if (transform_op.containsKey("Padding")) {
                HashMap info = (HashMap) transform_op.get("Padding");
                Padding padding = new Padding();
                if (info.containsKey("coarsest_stride")) {
                    padding.coarsest_stride = (int) info.get("coarsest_stride");
                }
                if (info.containsKey("im_padding_value")) {
                    List<Double> im_padding_value = (List<Double>) info.get("im_padding_value");
                    if (im_padding_value.size()!=3){
                        Log.e(TAG, "len of im_padding_value in padding must == 3.");
                    }
                    for (int k =0; i<im_padding_value.size(); i++){
                        padding.paddding_value[k] = im_padding_value.get(k);
                    }
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
                transformOps.add(padding);
            }
        }
    }

    public ImageBlob run(Mat inputMat, ImageBlob imageBlob) {
        imageBlob.setOriImageSize(inputMat.height(),2);
        imageBlob.setOriImageSize(inputMat.width(),3);
        imageBlob.setNewImageSize(inputMat.height(),2);
        imageBlob.setNewImageSize(inputMat.width(),3);

        if(transformsMode.equalsIgnoreCase("RGB")){
            Imgproc.cvtColor(inputMat, inputMat, Imgproc.COLOR_BGR2RGB);
        }else if(!transformsMode.equalsIgnoreCase("BGR")){
            Log.e(TAG, "transformsMode only support RGB or BGR.");
        }
        inputMat.convertTo(inputMat, CvType.CV_32FC(3));

        for (transformOp op : transformOps) {
            inputMat = op.run(inputMat, imageBlob);
        }

        int w = inputMat.width();
        int h = inputMat.height();
        int c = inputMat.channels();
        imageBlob.setImageData(new float[w * h * c]);

        Mat singleChannelMat = new Mat(h, w, CvType.CV_32FC(1));
        float[] singleChannelImageData = new float[w * h];
        for (int i = 0; i < c; i++) {
            Core.extractChannel(inputMat, singleChannelMat, i);
            singleChannelMat.get(0, 0, singleChannelImageData);
            System.arraycopy(singleChannelImageData ,0, imageBlob.getImageData(),i*w*h, w*h);
        }

        return imageBlob;
    }

    private class transformOp {
        public Mat run(Mat inputMat, ImageBlob data) {
            return inputMat;
        }
    }

    private class ResizeByShort extends transformOp {
        private int max_size;
        private int short_size;
        private String interp = "LINEAR";

        public Mat run(Mat inputMat, ImageBlob imageBlob) {
            int origin_w = inputMat.width();
            int origin_h = inputMat.height();
            imageBlob.getReshapeInfo().put("resize", new int[]{origin_w, origin_h});
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
            Size sz = new Size(width, height);
            Imgproc.resize(inputMat, inputMat, sz,0,0, interpMap.get(interp));
            imageBlob.setNewImageSize(inputMat.height(),2);
            imageBlob.setNewImageSize(inputMat.width(),3);
            imageBlob.setScale(scale);
            return inputMat;
        }
    }

    private class ResizeByLong extends transformOp {
        private int long_size;
        private String interp = "LINEAR";

        public Mat run(Mat inputMat, ImageBlob imageBlob) {
            int origin_w = inputMat.width();
            int origin_h = inputMat.height();
            imageBlob.getReshapeInfo().put("resize", new int[]{origin_w, origin_h});
            int im_size_max = Math.max(origin_w, origin_h);
            float scale = (float) (long_size) / (float) (im_size_max);
            int width = Math.round(scale * origin_w);
            int height = Math.round(scale * origin_h);
            Size sz = new Size(width, height);
            Imgproc.resize(inputMat, inputMat, sz,0,0, interpMap.get(interp));
            imageBlob.setNewImageSize(inputMat.height(),2);
            imageBlob.setNewImageSize(inputMat.width(),3);
            imageBlob.setScale(scale);
            return inputMat;
        }
    }

    private class CenterCrop extends transformOp {
        private int cropHeight;
        private int cropWidth;

        public Mat run(Mat inputMat, ImageBlob imageBlob) {
            int origin_w = inputMat.width();
            int origin_h = inputMat.height();
            if (origin_h < cropHeight || origin_w < cropWidth) {
                Log.e(TAG, "[CenterCrop] Image size less than crop size");
            }
            int offset_x, offset_y;
            offset_x = (origin_w - cropWidth) / 2;
            offset_y = (origin_h - cropHeight) / 2;
            offset_x = Math.max(Math.min(offset_x, origin_w - cropWidth), 0);
            offset_y = Math.max(Math.min(offset_y, origin_h - cropHeight), 0);
            Rect crop_roi = new Rect(offset_x, offset_y, cropHeight, cropWidth);
            inputMat = inputMat.submat(crop_roi);
            imageBlob.setNewImageSize(inputMat.height(),2);
            imageBlob.setNewImageSize(inputMat.width(),3);
            return inputMat;
        }
    }

    private class Resize extends transformOp {
        private int height;
        private int width;
        private String interp = "LINEAR";

        public Mat run(Mat inputMat, ImageBlob imageBlob) {
            int origin_w = inputMat.width();
            int origin_h = inputMat.height();
            imageBlob.getReshapeInfo().put("resize", new int[]{origin_w, origin_h});
            Size sz = new Size(width, height);
            Imgproc.resize(inputMat, inputMat, sz,0,0,  interpMap.get(interp));
            imageBlob.setNewImageSize(inputMat.height(),2);
            imageBlob.setNewImageSize(inputMat.width(),3);
            return inputMat;
        }
    }

    private class Padding extends transformOp {
        private double width;
        private double height;
        private double coarsest_stride;
        private double[] paddding_value = {0.0, 0.0, 0.0};

        public Mat run(Mat inputMat, ImageBlob imageBlob) {
            int origin_w = inputMat.width();
            int origin_h = inputMat.height();
            imageBlob.getReshapeInfo().put("padding", new int[]{origin_w, origin_h});
            double padding_w = 0;
            double padding_h = 0;
            if (width > 1 & height > 1) {
                padding_w = width;
                padding_h = height;
            } else if (coarsest_stride > 1) {
                padding_h = Math.ceil(origin_h / coarsest_stride) * coarsest_stride;
                padding_w = Math.ceil(origin_w / coarsest_stride) * coarsest_stride;
            }
            imageBlob.setNewImageSize(inputMat.height(),2);
            imageBlob.setNewImageSize(inputMat.width(),3);
            Core.copyMakeBorder(inputMat, inputMat, 0, (int)padding_h, 0, (int)padding_w, Core.BORDER_CONSTANT, new Scalar(paddding_value));
            return inputMat;
        }
    }

    private class Normalize extends transformOp {
        private Double[] mean = new Double[3];
        private Double[] std = new Double[3];

        public Mat run(Mat inputMat, ImageBlob imageBlob) {
            inputMat.convertTo(inputMat, CvType.CV_32FC(3), 1/255.0);
            Scalar meanScalar = new Scalar(mean[0], mean[1], mean[2]);
            Scalar stdScalar = new Scalar(std[0], std[1], std[2]);
            Core.subtract(inputMat, meanScalar, inputMat);
            Core.divide(inputMat, stdScalar, inputMat);
            return inputMat;
        }
    }
}

