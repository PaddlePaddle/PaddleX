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

package com.baidu.paddlex.config;

import org.yaml.snakeyaml.Yaml;

import android.content.Context;
import android.content.res.AssetManager;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class ConfigParser {
    protected String model = "";
    protected List<String> labeList = new ArrayList<>();
    protected int numClasses = 0;
    protected String modelType = "";

    protected String transformsMode = "RGB";
    protected List transformsList = new ArrayList();
    protected String modelPath = "";
    protected int cpuThreadNum = 1;
    protected String cpuPowerMode = "";
    protected String yamlPath = "";

    public void init(Context appCtx, String modelPath, String yamlPath, int cpuThreadNum,
                     String cpuPowerMode) throws IOException {

        this.modelPath = modelPath;
        this.cpuThreadNum = cpuThreadNum;
        this.cpuPowerMode = cpuPowerMode;
        this.yamlPath = yamlPath;
        AssetManager ass = appCtx.getAssets();
        InputStream ymlStream = ass.open(yamlPath);
        Yaml yml = new Yaml();

        HashMap yml_map = (HashMap) yml.load(ymlStream);
        model = (String) yml_map.get("Model");
        if (yml_map.containsKey("TransformsMode")) {
            transformsMode = (String) yml_map.get("TransformsMode");
        }

        HashMap _Attributes = (HashMap) yml_map.get("_Attributes");
        // parser label_list
        labeList = (List<String>) _Attributes.get("labels");
        numClasses = (int) _Attributes.get("num_classes");

        // parser model_type(classifier, segmenter, detector)
        modelType = (String) _Attributes.get("model_type");
        // parser Transforms
        transformsList = (List) yml_map.get("Transforms");

    }

    @Override
    public String toString() {
        return "ConfigParser{" +
                "model='" + model + '\'' +
                ", labeList=" + labeList +
                ", numClasses=" + numClasses +
                ", modelType='" + modelType + '\'' +
                ", transformsMode='" + transformsMode + '\'' +
                ", transformsList=" + transformsList +
                ", modelPath='" + modelPath + '\'' +
                ", cpuThreadNum=" + cpuThreadNum +
                ", cpuPowerMode='" + cpuPowerMode + '\'' +
                ", yamlPath='" + yamlPath + '\'' +
                '}';
    }
    public int getNumClasses() {
        return numClasses;
    }

    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    public List<String> getLabeList() {
        return labeList;
    }

    public void setLabeList(List<String> labeList) {
        this.labeList = labeList;
    }

    public String getModelType() {
        return modelType;
    }

    public void setModelType(String modelType) {
        this.modelType = modelType;
    }

    public List getTransformsList() {
        return transformsList;
    }

    public void setTransformsList(List transformsList) {
        this.transformsList = transformsList;
    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public String getTransformsMode() {
        return transformsMode;
    }

    public void setTransformsMode(String transformsMode) {
        this.transformsMode = transformsMode;
    }

    public String getModelPath() {
        return modelPath;
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
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

    public String getYamlPath() {
        return yamlPath;
    }

    public void setYamlPath(String yamlPath) {
        this.yamlPath = yamlPath;
    }
}
