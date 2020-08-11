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

package com.baidu.paddlex.visual;

import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.Log;

import com.baidu.paddlex.postprocess.DetResult;
import com.baidu.paddlex.postprocess.SegResult;
import com.baidu.paddlex.preprocess.ImageBlob;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

public class Visualize {
    protected static final String TAG = Visualize.class.getSimpleName();
    protected float detectConfidenceThreshold = (float) 0.5;
    protected Scalar[] colormap = new Scalar[]{};

    protected void generateColorMap(int num_class) {
        this.colormap = new Scalar[num_class];
        this.colormap[0] = new Scalar(0, 0, 0);
        for (int i = 0; i < num_class; i++) {
            int j = 0;
            int lab = i;
            while (lab > 0) {
                int r = (((lab >> 0) & 1) << (7 - j));
                int g = (((lab >> 1) & 1) << (7 - j));
                int b = (((lab >> 2) & 1) << (7 - j));
                this.colormap[i] = new Scalar(r, g, b);
                ++j;
                lab >>= 3;
            }
        }
    }

    public float getDetectConfidenceThreshold() {
        return detectConfidenceThreshold;
    }

    public void setDetectConfidenceThreshold(float detectConfidenceThreshold) {
        this.detectConfidenceThreshold = detectConfidenceThreshold;
    }

    public Scalar[] getColormap() {
        return colormap;
    }

    public void setColormap(Scalar[] colormap) {
        this.colormap = colormap;
    }

    public void init(int num_class) {
        generateColorMap(num_class);
    }

    public Mat draw(DetResult result, Mat visualizeMat) {
        Paint rectPaint = new Paint();
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(2);
        Paint txtPaint = new Paint();
        txtPaint.setTextSize(15);
        txtPaint.setAntiAlias(true);
        for (DetResult.Box box : result.getBoxes()) {
            if (box.getScore() < detectConfidenceThreshold) {
                continue;
            }

            String text = box.getCategory() + ":" + String.valueOf(box.getScore()).substring(0, 4);
            Scalar roiColor = colormap[box.getCategoryId()];
            double font_scale = 0.5;
            int thickness = 1;
            int font_face = Core.FONT_HERSHEY_SIMPLEX;

            Point roiXyMin = new Point(box.getCoordinate()[0],box.getCoordinate()[1]);
            Point roiXyMax = new Point(box.getCoordinate()[2],box.getCoordinate()[3]);
            Size text_size = Imgproc.getTextSize(text, font_face,font_scale, thickness,null);
            Imgproc.rectangle(visualizeMat, roiXyMin, roiXyMax, roiColor,2);

            Point textXyMin =  new Point(box.getCoordinate()[0],box.getCoordinate()[1]-text_size.height);
            Point textXyMax = new Point(box.getCoordinate()[0]+text_size.width,box.getCoordinate()[1]);
            Imgproc.rectangle(visualizeMat,textXyMin, textXyMax, roiColor,-1);
            Imgproc.putText(visualizeMat,
                    text,
                    roiXyMin,
                    font_face,
                    font_scale,
                    new Scalar(255, 255, 255));
        }
        return visualizeMat;
    }

    public Mat draw(SegResult result, Mat visualizeMat, ImageBlob imageBlob, int cutoutClass) {
        int new_h = (int)imageBlob.getNewImageSize()[2];
        int new_w = (int)imageBlob.getNewImageSize()[3];
        Mat mask = new Mat(new_h, new_w, CvType.CV_32FC(1));
        float[] scoreData = new float[new_h*new_w];
        for  (int h = 0; h < new_h; h++) {
            for  (int w = 0; w < new_w; w++){
                scoreData[new_h * h + w] =  (1-result.getMask().getScoreData()[cutoutClass + h * new_h + w]) * 255;
            }
        }
        mask.put(0,0, scoreData);
        mask.convertTo(mask,CvType.CV_8UC(1));
        ListIterator<Map.Entry<String, int[]>> reverseReshapeInfo = new ArrayList<Map.Entry<String, int[]>>(imageBlob.getReshapeInfo().entrySet()).listIterator(imageBlob.getReshapeInfo().size());
        while (reverseReshapeInfo.hasPrevious()) {
            Map.Entry<String, int[]> entry = reverseReshapeInfo.previous();
            if (entry.getKey().equalsIgnoreCase("padding")) {
                Rect crop_roi = new Rect(0, 0, entry.getValue()[0], entry.getValue()[1]);
                mask = mask.submat(crop_roi);
            } else if (entry.getKey().equalsIgnoreCase("resize")) {
                Size sz = new Size(entry.getValue()[0], entry.getValue()[1]);
                Imgproc.resize(mask, mask, sz,0,0,Imgproc.INTER_LINEAR);
            }
        }
        Mat dst  = new Mat();
        List<Mat> listMat = Arrays.asList(visualizeMat, mask);
        Core.merge(listMat, dst);

        return dst;
    }
}
