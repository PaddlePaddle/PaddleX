package com.example.paddlex;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;


import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;


import com.baidu.paddlex.Predictor;
import com.baidu.paddlex.config.ConfigParser;
import com.baidu.paddlex.preprocess.ImageBlob;
import com.baidu.paddlex.preprocess.Transforms;
import org.json.JSONException;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.IOException;
import java.io.InputStream;


import static org.junit.Assert.*;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class ExampleInstrumentedTest {
    @Test
    public void useAppContext() throws IOException, JSONException {
        // Context of the app under test.
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        AssetManager ass =  appContext.getAssets();

        InputStream imageStream = ass.open("face_detection/images/face.jpg");

        Bitmap inputImage = BitmapFactory.decodeStream(imageStream);
        String  modelPath = "face_detection/models/facedetection_for_cpu";
        String  imagePath = "face_detection/images/face.jpg";
        String  ymlPath = "face_detection/yaml/model.yml";
        ConfigParser configParser= new ConfigParser();
        configParser.init(appContext, modelPath, ymlPath, imagePath,1,"LITE_POWER_HIGH");

        ImageBlob imageBlob = new ImageBlob();

        Transforms transforms= new Transforms();
        transforms.load_config(configParser);

        imageBlob = transforms.run(inputImage, imageBlob);

        Predictor predictor = new Predictor();
        predictor.init(appContext,"face_detection/models/facedetection_for_cpu",1,"LITE_POWER_HIGH");
        predictor.runModel();

        assertEquals(configParser.transforms_config.centerCrop.cropHeight, 224);
        assertEquals(imageBlob.new_im_size_[2], 224);
        assertEquals(imageBlob.new_im_size_[3], 224);
        assertEquals(configParser.label_list.get(0), "bocai");
        assertEquals(configParser.model_type, "classifier");
    }
}
