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
import android.graphics.Bitmap;
import android.util.Log;
import java.io.File;
import java.util.Date;
import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.PowerMode;
import com.baidu.paddle.lite.Tensor;
import com.baidu.paddlex.config.ConfigParser;
import com.baidu.paddlex.postprocess.ClsResult;
import com.baidu.paddlex.postprocess.DetResult;
import com.baidu.paddlex.postprocess.Result;
import com.baidu.paddlex.postprocess.SegResult;
import com.baidu.paddlex.preprocess.ImageBlob;
import com.baidu.paddlex.preprocess.Transforms;


public class Predictor {
    private static final String TAG = Predictor.class.getSimpleName();

    protected boolean isLoaded = false;
    protected int warmupIterNum = 0;
    protected int inferIterNum = 1;
    protected Context appCtx = null;
    protected int cpuThreadNum = 1;
    protected String cpuPowerMode = "LITE_POWER_HIGH";
    protected ImageBlob imageBlob = new ImageBlob();
    protected String modelPath = "";
    protected String modelName = "";
    protected Result result;
    protected PaddlePredictor paddlePredictor = null;
    protected Bitmap inputImage;
    protected float inferenceTime = 0;
    protected String outputResult = "";
    protected float preprocessTime = 0;
    protected float postprocessTime = 0;
    protected ConfigParser configParser = new ConfigParser();
    protected Transforms transforms = new Transforms();
    public Predictor() {
        super();
    }

    public boolean init(Context appCtx, String modelPath, int cpuThreadNum, String cpuPowerMode) {
        this.appCtx = appCtx;
        if (configParser.getModelType().equalsIgnoreCase("classifier")) {
            result = new ClsResult();
        } else if (configParser.getModelType().equalsIgnoreCase("detector")) {
            result = new DetResult();
        } else if (configParser.getModelType().equalsIgnoreCase("segmenter")) {
            result = new SegResult();
        } else {
            Log.i(TAG, "model_type: " + configParser.getModelType() + " is not support! Only support: 'classifier' or 'detector' or 'segmenter'");
        }
        isLoaded = loadModel(modelPath, cpuThreadNum, cpuPowerMode);
        return isLoaded;
    }

    public boolean init(Context appCtx, ConfigParser configParser) {
        this.configParser = configParser;
        init(appCtx, configParser.getModelPath(), configParser.getCpuThreadNum(), configParser.getCpuPowerMode());
        transforms.load_config(configParser.getTransformsList(), configParser.getTransformsMode());

        if (!isLoaded()) {
            return false;
        }
        Log.i(TAG, configParser.toString());
        return isLoaded;
    }

    public boolean predict() {
        imageBlob = transforms.run(inputImage, imageBlob);
        if (configParser.getModelType().equalsIgnoreCase("classifier")) {
            runModel((ClsResult) result);
        } else if (configParser.getModelType().equalsIgnoreCase("detector")) {
            runModel((DetResult) result);
        } else if (configParser.getModelType().equalsIgnoreCase("segmenter")) {
            runModel((SegResult) result);
        } else {
            Log.i(TAG, "model_type: " + configParser.getModelType() + " is not support! Only support: 'classifier' or 'detector' or 'segmenter'");
        }
        return true;
    }

    private boolean runModel(DetResult detReult) {
        Tensor imTensor = getInput(0);
        imTensor.resize(imageBlob.new_im_size_);
        imTensor.setData(imageBlob.im_data_);
        if (configParser.getModel().equalsIgnoreCase("YOLOv3")) {
            Tensor imSizeTensor = getInput(1);
            long[] imSize = {1, 2};
            imSizeTensor.resize(imSize);
            imSizeTensor.setData(new int[]{(int) imageBlob.ori_im_size_[2], (int) imageBlob.ori_im_size_[3]});
        } else if (configParser.getModel().equalsIgnoreCase("FasterRCNN")) {
            Tensor imInfoTensor = getInput(1);
            long[] imInfo = {1, 3};
            imInfoTensor.resize(imInfo);
            imInfoTensor.setData(new float[]{imageBlob.new_im_size_[2], imageBlob.new_im_size_[3], imageBlob.scale});

            Tensor imShapeTensor = getInput(2);
            long[] imShape = {1, 3};
            imShapeTensor.resize(imShape);
            imShapeTensor.setData(new float[]{imageBlob.ori_im_size_[2], imageBlob.ori_im_size_[3], 1});
        }

        runModel();

        Tensor outputTensor = getOutput(0);

        float[] output = outputTensor.getFloatData();
        long outputShape[] = outputTensor.shape();
        long outputSize = 1;

        for (long s : outputShape) {
            outputSize *= s;
            Log.i(TAG, "****" + String.valueOf(s));
        }

        int num_boxes = (int) (outputSize / 6);
        for (int i = 0; i < num_boxes; i++) {
            DetResult.Box box = detReult.new Box();
            box.setCategory_id((int) output[i * 6]);
            box.setCategory(configParser.getLabeList().get(box.getCategory_id()));
            box.setScore(output[i * 6 + 1]);
            float xmin = output[i * 6 + 2];
            float ymin = output[i * 6 + 3];
            float xmax = output[i * 6 + 4];
            float ymax = output[i * 6 + 5];
            box.setCoordinate(new float[]{xmin, ymin, xmax, ymax});

            detReult.getBoxes().add(box);
        }
        return true;

    }

    private boolean runModel(SegResult segReult) {
        Tensor imTensor = getInput(0);
        imTensor.resize(imageBlob.new_im_size_);
        imTensor.setData(imageBlob.im_data_);
        for (long shape : imageBlob.new_im_size_) {
            Log.i(TAG, "input shape:" + shape);

        }
        runModel();

        Tensor labelTensor = getOutput(0);

        long[] labelData = labelTensor.getLongData();
        segReult.getMask().setLabelShape(labelTensor.shape());
        long labelSize = 1;
        for (long s : segReult.getMask().getLabelShape()) {
            labelSize *= s;
        }
        segReult.getMask().setLabelData( new long[(int) labelSize]);

        for (int i = 0; i < labelData.length; i++) {
            segReult.getMask().getLabelData()[i] = labelData[i];
        }
        Tensor scoreTensor = getOutput(1);
        long[] scoreData = scoreTensor.getLongData();
        segReult.getMask().setScoreShape(scoreTensor.shape());
        long scoreSize = 1;
        for (long s : segReult.getMask().getScoreShape()) {
            scoreSize *= s;
        }
        segReult.getMask().setScoreData(new float[(int) scoreSize]);
        for (int i = 0; i < scoreData.length; i++) {
            segReult.getMask().getScoreData()[i] = scoreData[i];
        }
        return true;
    }

    private boolean runModel(ClsResult clsReult) {
        // set input shape
        Tensor imTensor = getInput(0);
        imTensor.resize(imageBlob.new_im_size_);
        imTensor.setData(imageBlob.im_data_);

        runModel();

        // Fetch output tensor
        Tensor outputTensor = getOutput(0);
        long outputShape[] = outputTensor.shape();
        long outputSize = 1;
        for (long s : outputShape) {
            outputSize *= s;
        }

        int[] max_index = new int[3]; // Top3 indices
        float[] max_num = new float[3]; // Top3 scores
        for (int i = 0; i < outputSize; i++) {
            float tmp = outputTensor.getFloatData()[i];
            int tmp_index = i;
            for (int j = 0; j < 3; j++) {
                if (tmp > max_num[j]) {
                    tmp_index += max_index[j];
                    max_index[j] = tmp_index - max_index[j];
                    tmp_index -= max_index[j];
                    tmp += max_num[j];
                    max_num[j] = tmp - max_num[j];
                    tmp -= max_num[j];
                }
            }
        }

        clsReult.setCategoryId(max_index[0]);
        clsReult.setCategory(configParser.getLabeList().get(max_index[0]));
        clsReult.setScore(max_num[0]);
        Log.i(TAG, outputResult);
        if (configParser.getLabeList().size() > 0) {
            outputResult = "Top1: " + configParser.getLabeList().get(max_index[0]) + " - " + String.format("%.3f", max_num[0]);
        }
        return true;
    }

    private boolean loadModel(String modelPath, int cpuThreadNum, String cpuPowerMode) {
        // release model if exists
        releaseModel();

        // load model
        if (modelPath.isEmpty()) {
            return false;
        }

        String realPath = modelPath;
        if (!modelPath.substring(0, 1).equals("/")) {
            // read model files from custom path if the first character of mode path is '/'
            // otherwise copy model to cache from assets
            String modelFileName = Utils.getFileNameFromString(modelPath);
            realPath = appCtx.getCacheDir() + File.separator + modelFileName;
            Utils.copyFileFromAssets(appCtx, modelPath, realPath);
        }

        if (realPath.isEmpty()) {
            return false;
        }

        MobileConfig config = new MobileConfig();
        config.setModelFromFile(realPath);
        config.setThreads(cpuThreadNum);

        if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_LOW);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_FULL")) {
            config.setPowerMode(PowerMode.LITE_POWER_FULL);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_NO_BIND")) {
            config.setPowerMode(PowerMode.LITE_POWER_NO_BIND);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_LOW);
        } else {
            Log.e(TAG, "unknown cpu power mode!");
            return false;
        }
        paddlePredictor = PaddlePredictor.createPaddlePredictor(config);
        this.cpuThreadNum = cpuThreadNum;
        this.cpuPowerMode = cpuPowerMode;
        this.modelPath = realPath;
        this.modelName = configParser.getModel();
        return true;
    }

    private boolean runModel() {
        if (!isLoaded()) {
            return false;
        }

        // warm up
        for (int i = 0; i < warmupIterNum; i++) {
            paddlePredictor.run();
        }

        Date start = new Date();
        // inference
        for (int i = 0; i < inferIterNum; i++) {
            paddlePredictor.run();
        }
        Date end = new Date();
        inferenceTime = (end.getTime() - start.getTime()) / (float) inferIterNum;

        return true;

    }

    public void releaseModel() {
        paddlePredictor = null;
        isLoaded = false;
        cpuThreadNum = 1;
        cpuPowerMode = "LITE_POWER_HIGH";
        modelPath = "";
        modelName = "";
    }


    public boolean isLoaded() {
        return paddlePredictor != null && isLoaded;
    }

    public void setLoaded(boolean loaded) {
        isLoaded = loaded;
    }

    public int getWarmupIterNum() {
        return warmupIterNum;
    }

    public void setWarmupIterNum(int warmupIterNum) {
        this.warmupIterNum = warmupIterNum;
    }

    public int getInferIterNum() {
        return inferIterNum;
    }

    public void setInferIterNum(int inferIterNum) {
        this.inferIterNum = inferIterNum;
    }

    public float getInferenceTime() {
        return inferenceTime;
    }

    public void setInferenceTime(float inferenceTime) {
        this.inferenceTime = inferenceTime;
    }

    public int getCpuThreadNum() {
        return cpuThreadNum;
    }

    public void setCpuThreadNum(int cpuThreadNum) {
        this.cpuThreadNum = cpuThreadNum;
    }

    public String getCpuPowerMode() {
        return cpuPowerMode;
    }

    public void setCpuPowerMode(String cpuPowerMode) {
        this.cpuPowerMode = cpuPowerMode;
    }

    public String getModelPath() {
        return modelPath;
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public Result getResult() {
        return result;
    }

    public void setResult(Result result) {
        this.result = result;
    }

    public PaddlePredictor getPaddlePredictor() {
        return paddlePredictor;
    }

    public void setPaddlePredictor(PaddlePredictor paddlePredictor) {
        this.paddlePredictor = paddlePredictor;
    }

    public String getOutputResult() {
        return outputResult;
    }

    public void setOutputResult(String outputResult) {
        this.outputResult = outputResult;
    }

    public float getPreprocessTime() {
        return preprocessTime;
    }

    public void setPreprocessTime(float preprocessTime) {
        this.preprocessTime = preprocessTime;
    }

    public float getPostprocessTime() {
        return postprocessTime;
    }

    public void setPostprocessTime(float postprocessTime) {
        this.postprocessTime = postprocessTime;
    }

    public void setConfigParser(ConfigParser configParser) {
        this.configParser = configParser;
    }

    public Transforms getTransforms() {
        return transforms;
    }

    public void setTransforms(Transforms transforms) {
        this.transforms = transforms;
    }

    public Bitmap getInputImage() {
        return inputImage;
    }

    public void setInputImage(Bitmap inputImage) {
        this.inputImage = inputImage.copy(inputImage.getConfig(), true);
    }

    public DetResult getDetResult() {
        if (result.getType() != "det") {
            Log.e(TAG, "this model_type is not detector");
            return null;
        }
        return (DetResult) result;
    }

    public SegResult getSegResult() {
        if (result.getType() != "seg") {
            Log.e(TAG, "this model_type is not segmeter");
            return null;
        }
        return (SegResult) result;
    }

    public ClsResult getClsResult() {
        if (result.getType() != "cls") {
            Log.e(TAG, "this model_type is not classifier");
            return null;
        }
        return (ClsResult) result;
    }

    public ImageBlob getImageBlob() {
        return imageBlob;
    }

    public void setImageBlob(ImageBlob imageBlob) {
        this.imageBlob = imageBlob;
    }

    public Tensor getInput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return paddlePredictor.getInput(idx);
    }

    public Tensor getOutput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return paddlePredictor.getOutput(idx);
    }

}
