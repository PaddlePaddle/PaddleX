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

package com.baidu.paddlex.lite.demo;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import com.baidu.paddlex.Predictor;
import com.baidu.paddlex.Utils;
import com.baidu.paddlex.config.ConfigParser;
import com.baidu.paddlex.postprocess.ClsResult;
import com.baidu.paddlex.postprocess.DetResult;
import com.baidu.paddlex.postprocess.SegResult;
import com.baidu.paddlex.visual.Visualize;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
    public static final int OPEN_GALLERY_REQUEST_CODE = 0;
    public static final int TAKE_PHOTO_REQUEST_CODE = 1;
    public static final int REQUEST_LOAD_MODEL = 0;
    public static final int REQUEST_RUN_MODEL = 1;
    public static final int RESPONSE_LOAD_MODEL_SUCCESSED = 0;
    public static final int RESPONSE_LOAD_MODEL_FAILED = 1;
    public static final int RESPONSE_RUN_MODEL_SUCCESSED = 2;
    public static final int RESPONSE_RUN_MODEL_FAILED = 3;
    private static final String TAG = MainActivity.class.getSimpleName();
    protected ProgressDialog pbLoadModel = null;
    protected ProgressDialog pbRunModel = null;

    protected Handler receiver = null; // receive messages from worker thread
    protected Handler sender = null; // send command to worker thread
    protected HandlerThread worker = null; // worker thread to load&run model

    protected TextView tvInputSetting;
    protected ImageView ivInputImage;
    protected TextView tvOutputResult;
    protected TextView tvInferenceTime;
    private Button predictButton;
    protected String testImagePathFromAsset;
    protected String testYamlPathFromAsset;
    protected String testModelPathFromAsset;

    // Predictor
    protected Predictor predictor = new Predictor();
    // model config
    protected ConfigParser configParser = new ConfigParser();
    // Visualize
    protected Visualize visualize = new Visualize();
    // Predict Mat of Opencv
    protected Mat predictMat;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        receiver = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case RESPONSE_LOAD_MODEL_SUCCESSED:
                        pbLoadModel.dismiss();
                        Toast.makeText(MainActivity.this, "Load model successfully!", Toast.LENGTH_SHORT).show();
                        break;
                    case RESPONSE_LOAD_MODEL_FAILED:
                        pbLoadModel.dismiss();
                        Toast.makeText(MainActivity.this, "Load model failed!", Toast.LENGTH_SHORT).show();
                        break;
                    case RESPONSE_RUN_MODEL_SUCCESSED:
                        pbRunModel.dismiss();
                        onRunModelSuccessed();
                        break;
                    case RESPONSE_RUN_MODEL_FAILED:
                        pbRunModel.dismiss();
                        Toast.makeText(MainActivity.this, "Run model failed!", Toast.LENGTH_SHORT).show();
                        onRunModelFailed();
                        break;
                    default:
                        break;
                }
            }
        };
        worker = new HandlerThread("Predictor Worker");
        worker.start();
        sender = new Handler(worker.getLooper()) {
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case REQUEST_LOAD_MODEL:
                        // load model and reload test image
                        if (onLoadModel()) {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_FAILED);
                        }
                        break;
                    case REQUEST_RUN_MODEL:
                        // run model if model is loaded
                        if (onRunModel()) {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_FAILED);
                        }
                        break;
                    default:
                        break;
                }
            }
        };

        tvInputSetting = findViewById(R.id.tv_input_setting);
        ivInputImage = findViewById(R.id.iv_input_image);
        predictButton = findViewById(R.id.iv_predict_button);
        tvInferenceTime = findViewById(R.id.tv_inference_time);
        tvOutputResult = findViewById(R.id.tv_output_result);
        tvInputSetting.setMovementMethod(ScrollingMovementMethod.getInstance());
        tvOutputResult.setMovementMethod(ScrollingMovementMethod.getInstance());
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        String image_path = sharedPreferences.getString(getString(R.string.IMAGE_PATH_KEY),
                getString(R.string.IMAGE_PATH_DEFAULT));
        Utils.initialOpencv();
        loadTestImageFromAsset(image_path);
        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(predictor.isLoaded()){
                    onLoadModelSuccessed();
                }
            }
        });

    }

    public boolean onLoadModel() {
        return predictor.init(configParser);
    }

    public boolean onRunModel() {
        return predictor.isLoaded() && predictor.predict();
    }

    public void onRunModelFailed() {
    }

    public void loadModel() {
        pbLoadModel = ProgressDialog.show(this, "", "Loading model...", false, false);
        sender.sendEmptyMessage(REQUEST_LOAD_MODEL);
    }

    public void runModel() {
        pbRunModel = ProgressDialog.show(this, "", "Running model...", false, false);
        sender.sendEmptyMessage(REQUEST_RUN_MODEL);
    }

    public void onLoadModelSuccessed() {
        if (predictMat != null && predictor.isLoaded()) {
            int w = predictMat.width();
            int h = predictMat.height();
            int c = predictMat.channels();
            predictor.setInputMat(predictMat);
            runModel();
        }
    }

    public void onRunModelSuccessed() {
        // obtain results and update UI
        tvInferenceTime.setText("Inference time: " + predictor.getInferenceTime() + " ms");

        if (configParser.getModelType().equalsIgnoreCase("segmenter")) {
            SegResult segResult = predictor.getSegResult();
            Mat maskMat = visualize.draw(segResult, predictMat.clone(), predictor.getImageBlob(), 1);
            Imgproc.cvtColor(maskMat, maskMat, Imgproc.COLOR_BGRA2RGBA);
            Bitmap outputImage = Bitmap.createBitmap(maskMat.width(), maskMat.height(), Bitmap.Config.ARGB_8888);
            org.opencv.android.Utils.matToBitmap(maskMat, outputImage);
            if (outputImage != null) {
                ivInputImage.setImageBitmap(outputImage);
            }
        } else if (configParser.getModelType().equalsIgnoreCase("detector")) {
            DetResult detResult = predictor.getDetResult();
            Mat roiMat  = visualize.draw(detResult,  predictMat.clone());
            Imgproc.cvtColor(roiMat, roiMat, Imgproc.COLOR_BGR2RGB);
            Bitmap outputImage = Bitmap.createBitmap(roiMat.width(),roiMat.height(), Bitmap.Config.ARGB_8888);
            org.opencv.android.Utils.matToBitmap(roiMat,outputImage);
            if (outputImage != null) {
                ivInputImage.setImageBitmap(outputImage);
            }
        } else if (configParser.getModelType().equalsIgnoreCase("classifier")) {
            ClsResult clsResult = predictor.getClsResult();
            if (configParser.getLabeList().size() > 0) {
                String outputResult = "Top1: " + clsResult.getCategory() + " - " + String.format("%.3f", clsResult.getScore());
                tvOutputResult.setText(outputResult);
                tvOutputResult.scrollTo(0, 0);
            }
        }
    }

    public void onMatChanged(Mat mat) {
        this.predictMat = mat.clone();
    }

    public void onImageChanged(Bitmap image) {
        ivInputImage.setImageBitmap(image);
        tvOutputResult.setText("");
        tvInferenceTime.setText("Inference time: -- ms");
    }

    public void onSettingsClicked() {
        startActivity(new Intent(MainActivity.this, SettingsActivity.class));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_action_options, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                finish();
                break;
            case R.id.open_gallery:
                if (requestAllPermissions()) {
                    openGallery();
                }
                break;
            case R.id.take_photo:
                if (requestAllPermissions()) {
                    takePhoto();
                }
                break;
            case R.id.settings:
                if (requestAllPermissions()) {
                    // make sure we have SDCard r&w permissions to load model from SDCard
                    onSettingsClicked();
                }
                break;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            switch (requestCode) {
                case OPEN_GALLERY_REQUEST_CODE:
                    try {
                        ContentResolver resolver = getContentResolver();
                        Uri uri = data.getData();
                        Bitmap image = MediaStore.Images.Media.getBitmap(resolver, uri);
                        String[] proj = {MediaStore.Images.Media.DATA};
                        Cursor cursor = managedQuery(uri, proj, null, null, null);
                        cursor.moveToFirst();
                        int columnIndex = cursor.getColumnIndex(proj[0]);
                        String imgDecodableString = cursor.getString(columnIndex);
                        File file = new File(imgDecodableString);
                        Mat mat = Imgcodecs.imread(file.getAbsolutePath(),Imgcodecs.IMREAD_COLOR);
                        onImageChanged(image);
                        onMatChanged(mat);
                    } catch (IOException e) {
                        Log.e(TAG, e.toString());
                    }
                    break;
                case TAKE_PHOTO_REQUEST_CODE:
                    Bitmap image = (Bitmap) data.getParcelableExtra("data");
                    Mat mat = new Mat();
                    org.opencv.android.Utils.bitmapToMat(image, mat);
                    Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2BGR);
                    onImageChanged(image);
                    onMatChanged(mat);
                    break;
                default:
                    break;
            }
        }
    }

    private boolean requestAllPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED || ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.CAMERA},
                    0);
            return false;
        }
        return true;
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, null);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, OPEN_GALLERY_REQUEST_CODE);
    }

    private void takePhoto() {
        Intent takePhotoIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePhotoIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePhotoIntent, TAKE_PHOTO_REQUEST_CODE);
        }
    }

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        boolean isLoaded = predictor.isLoaded();
        menu.findItem(R.id.open_gallery).setEnabled(isLoaded);
        menu.findItem(R.id.take_photo).setEnabled(isLoaded);
        return super.onPrepareOptionsMenu(menu);
    }

    @Override
    protected void onResume() {
        Log.i(TAG, "begin onResume");
        super.onResume();
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);

        boolean settingsChanged = false;
        boolean testImageChanged = false;
        String modelPath = sharedPreferences.getString(getString(R.string.MODEL_PATH_KEY),
                getString(R.string.MODEL_PATH_DEFAULT));
        settingsChanged |= !modelPath.equalsIgnoreCase(testModelPathFromAsset);
        String yamlPath = sharedPreferences.getString(getString(R.string.YAML_PATH_KEY),
                getString(R.string.YAML_PATH_DEFAULT));
        settingsChanged |= !yamlPath.equalsIgnoreCase(testYamlPathFromAsset);
        int cpuThreadNum = Integer.parseInt(sharedPreferences.getString(getString(R.string.CPU_THREAD_NUM_KEY),
                getString(R.string.CPU_THREAD_NUM_DEFAULT)));
        settingsChanged |= cpuThreadNum != configParser.getCpuThreadNum();
        String cpuPowerMode = sharedPreferences.getString(getString(R.string.CPU_POWER_MODE_KEY),
                getString(R.string.CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !cpuPowerMode.equalsIgnoreCase(configParser.getCpuPowerMode());
        String imagePath = sharedPreferences.getString(getString(R.string.IMAGE_PATH_KEY),
                getString(R.string.IMAGE_PATH_DEFAULT));
        testImageChanged |= !imagePath.equalsIgnoreCase(testImagePathFromAsset);

        testYamlPathFromAsset = yamlPath;
        testModelPathFromAsset = modelPath;
        if (settingsChanged) {
            try {
                String realModelPath = modelPath;
                if (!modelPath.substring(0, 1).equals("/")) {
                    String modelFileName = Utils.getFileNameFromString(modelPath);
                    realModelPath = this.getCacheDir() + File.separator + modelFileName;
                    Utils.copyFileFromAssets(this, modelPath, realModelPath);
                }
                String realYamlPath = yamlPath;
                if (!yamlPath.substring(0, 1).equals("/")) {
                    String yamlFileName = Utils.getFileNameFromString(yamlPath);
                    realYamlPath = this.getCacheDir() + File.separator + yamlFileName;
                    Utils.copyFileFromAssets(this, yamlPath, realYamlPath);
                }
                configParser.init(realModelPath, realYamlPath, cpuThreadNum, cpuPowerMode);
                visualize.init(configParser.getNumClasses());
            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(MainActivity.this, "Load config failed!", Toast.LENGTH_SHORT).show();
            }
            // update UI
            tvInputSetting.setText("Model: " + configParser.getModel()+ "\n" + "CPU" +
                    " Thread Num: " + Integer.toString(configParser.getCpuThreadNum()) + "\n" + "CPU Power Mode: " + configParser.getCpuPowerMode());
            tvInputSetting.scrollTo(0, 0);
            // reload model if configure has been changed
            loadModel();
        }

        if (testImageChanged){
            loadTestImageFromAsset(imagePath);
        }
    }

    public void loadTestImageFromAsset(String imagePath){
        if (imagePath.isEmpty()) {
            return;
        }
        // read test image file from custom file_paths if the first character of mode file_paths is '/', otherwise read test
        // image file from assets
        testImagePathFromAsset = imagePath;
        if (!imagePath.substring(0, 1).equals("/")) {
            InputStream imageStream = null;
            try {
                imageStream = getAssets().open(imagePath);
            } catch (IOException e) {
                e.printStackTrace();
            }
            onImageChanged(BitmapFactory.decodeStream(imageStream));
            String realPath;
            String imageFileName = Utils.getFileNameFromString(imagePath);
            realPath = this.getCacheDir() + File.separator + imageFileName;
            Utils.copyFileFromAssets(this, imagePath, realPath);
            onMatChanged(Imgcodecs.imread(realPath, Imgcodecs.IMREAD_COLOR));
        } else {
            if (!new File(imagePath).exists()) {
                return;
            }
            onMatChanged(Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_COLOR));
            onImageChanged( BitmapFactory.decodeFile(imagePath));
        }
    }

    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.releaseModel();
        }
        worker.quit();
        super.onDestroy();
    }
}