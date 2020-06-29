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

import com.baidu.paddlex.postprocess.DetResult;
import com.baidu.paddlex.postprocess.SegResult;
import com.baidu.paddlex.preprocess.ImageBlob;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

import java.util.ArrayList;
import java.util.ListIterator;
import java.util.Map;


public class Visualize {
    protected static final String TAG = Visualize.class.getSimpleName();
    protected float detectConfidenceThreshold = (float) 0.5;
    protected int[] colormap;

    protected int[] generateColorMap(int num_class) {
        colormap = new int[num_class];
        for (int i = 0; i < num_class; i++) {
            int j = 0;
            int lab = i;
            while (lab > 0) {
                int r = (((lab >> 0) & 1) << (7 - j));
                int g = (((lab >> 1) & 1) << (7 - j));
                int b = (((lab >> 2) & 1) << (7 - j));
                colormap[i] = Color.rgb(r, g, b);
                lab >>= 3;
            }
        }
        return colormap;
    }

    public float getDetectConfidenceThreshold() {
        return detectConfidenceThreshold;
    }

    public void setDetectConfidenceThreshold(float detectConfidenceThreshold) {
        this.detectConfidenceThreshold = detectConfidenceThreshold;
    }

    public int[] getColormap() {
        return colormap;
    }

    public void setColormap(int[] colormap) {
        this.colormap = colormap;
    }

    public void init(int num_class) {
        colormap = generateColorMap(num_class);
    }

    public Bitmap draw(DetResult result, Bitmap visualizeImage) {
        Canvas canvas = new Canvas(visualizeImage);

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
            int color = colormap[box.getCategory_id()];
            rectPaint.setColor(color);
            txtPaint.setColor(color);
            canvas.drawRect(box.getCoordinate()[0], box.getCoordinate()[1], box.getCoordinate()[2], box.getCoordinate()[3], rectPaint);
            canvas.drawText(box.getCategory() + ":" + String.valueOf(box.getScore()).substring(0,4), (int) box.getCoordinate()[0] + 2, (int) box.getCoordinate()[1] + 2, txtPaint);
        }
        return visualizeImage;
    }

    public Bitmap draw(SegResult result, Bitmap visualizeImage, ImageBlob imageBlob) {
        int[] color_data = new int[result.getMask().getLabelData().length];
        for (int i = 0; i < result.getMask().getLabelData().length; i++) {
            color_data[i] = colormap[(int) result.getMask().getLabelData()[i]];
        }
        Bitmap maskImage = Bitmap.createBitmap(color_data, (int) result.getMask().getLabelShape()[2], (int) result.getMask().getLabelShape()[1], visualizeImage.getConfig());

        ListIterator<Map.Entry<String, int[]>> reverseReshapeInfo = new ArrayList<Map.Entry<String, int[]>>(imageBlob.reshape_info_.entrySet()).listIterator(imageBlob.reshape_info_.size());
        while (reverseReshapeInfo.hasPrevious()) {
            Map.Entry<String, int[]> entry = reverseReshapeInfo.previous();
            if (entry.getKey().equalsIgnoreCase("padding")) {
                maskImage = Bitmap.createBitmap(maskImage, 0, 0, entry.getValue()[0], entry.getValue()[1]);
            } else if (entry.getKey().equalsIgnoreCase("resize")) {
                maskImage = Bitmap.createScaledBitmap(maskImage, entry.getValue()[0], entry.getValue()[1], true);
            }
            Log.i(TAG, "postprocess operator: " + entry.getKey());
        }

        Bitmap bmOverlay = Bitmap.createBitmap(maskImage.getWidth(), maskImage.getHeight(), maskImage.getConfig());
        Canvas canvas = new Canvas(bmOverlay);
        Paint paint = new Paint();
        paint.setAlpha(0x80);
        canvas.drawBitmap(maskImage, 0, 0, null);
        canvas.drawBitmap(visualizeImage, 0, 0, paint);
        return bmOverlay;
    }
}
