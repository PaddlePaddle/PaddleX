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

package com.baidu.paddlex;

import android.content.Context;
import android.os.Environment;

import org.opencv.android.OpenCVLoader;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class Utils {
    private static final String TAG = Utils.class.getSimpleName();

    public static void copyFileFromAssets(Context appCtx, String srcPath, String dstPath) {
        if (srcPath.isEmpty() || dstPath.isEmpty()) {
            return;
        }
        InputStream is = null;
        OutputStream os = null;
        try {
            is = new BufferedInputStream(appCtx.getAssets().open(srcPath));
            os = new BufferedOutputStream(new FileOutputStream(new File(dstPath)));
            byte[] buffer = new byte[1024];
            int length = 0;
            while ((length = is.read(buffer)) != -1) {
                os.write(buffer, 0, length);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                os.close();
                is.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void copyDirectoryFromAssets(Context appCtx, String srcDir, String dstDir) {
        if (srcDir.isEmpty() || dstDir.isEmpty()) {
            return;
        }
        try {
            if (!new File(dstDir).exists()) {
                new File(dstDir).mkdirs();
            }
            for (String fileName : appCtx.getAssets().list(srcDir)) {
                String srcSubPath = srcDir + File.separator + fileName;
                String dstSubPath = dstDir + File.separator + fileName;
                copyFileFromAssets(appCtx, srcSubPath, dstSubPath);
                if (new File(srcSubPath).isDirectory()) {
                    copyDirectoryFromAssets(appCtx, srcSubPath, dstSubPath);
                } else {
                    copyFileFromAssets(appCtx, srcSubPath, dstSubPath);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static String getFileNameFromString(String srcDir) {
        if (srcDir.isEmpty()) {
            return null;
        }
        try {
            String fileName = srcDir.substring(srcDir.lastIndexOf("/") + 1);
            return fileName;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static float[] parseFloatsFromString(String string, String delimiter) {
        String[] pieces = string.trim().toLowerCase().split(delimiter);
        float[] floats = new float[pieces.length];
        for (int i = 0; i < pieces.length; i++) {
            floats[i] = Float.parseFloat(pieces[i].trim());
        }
        return floats;
    }

    public static long[] parseLongsFromString(String string, String delimiter) {
        String[] pieces = string.trim().toLowerCase().split(delimiter);
        long[] longs = new long[pieces.length];
        for (int i = 0; i < pieces.length; i++) {
            longs[i] = Long.parseLong(pieces[i].trim());
        }
        return longs;
    }

    public static String getSDCardDirectory() {
        return Environment.getExternalStorageDirectory().getAbsolutePath();
    }

    public static boolean isSupportedNPU() {
        String hardware = android.os.Build.HARDWARE;
        return hardware.equalsIgnoreCase("kirin810") || hardware.equalsIgnoreCase("kirin990");
    }

    public static boolean initialOpencv() {
        if (!OpenCVLoader.initDebug()) {
            return false;
        }
        return true;
    }

}
