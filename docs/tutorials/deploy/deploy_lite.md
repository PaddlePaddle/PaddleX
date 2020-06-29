# 移动端部署

PaddleX的移动端部署由PaddleLite实现，部署的流程如下，首先将训练好的模型导出为inference model，然后使用PaddleLite的python接口对模型进行优化，最后使用PaddleLite的预测库进行部署，
PaddleLite的详细介绍和使用可参考：[PaddleLite文档](https://paddle-lite.readthedocs.io/zh/latest/)

> PaddleX --> Inference Model --> PaddleLite Opt --> PaddleLite Inference

以下介绍如何将PaddleX导出为inference model，然后使用PaddleLite的OPT模块对模型进行优化：

## step 1: 安装PaddleLite

```
pip install paddlelite
```

## step 2: 将PaddleX模型导出为inference模型

参考[导出inference模型](deploy_server/deploy_python.html#inference)将模型导出为inference格式模型。
**注意：由于PaddleX代码的持续更新，版本低于1.0.0的模型暂时无法直接用于预测部署，参考[模型版本升级](./upgrade_version.md)对模型版本进行升级。**

step 3: 将inference模型转换成PaddleLite模型

```
python /path/to/PaddleX/deploy/lite/export_lite.py --model_dir /path/to/inference_model --save_file /path/to/lite_model --place place/to/run

```

|  参数   | 说明  |
|  ----  | ----  |
| model_dir  | 预测模型所在路径，包含"__model__", "__params__", "model.yml"文件 |
| save_file  | 模型输出的名称，默认为"paddlex.nb" |
| place | 运行的平台，可选：arm|opencl|x86|npu|xpu|rknpu|apu |

## step 4: 移动端（Android）预测

### 4.1 要求
Android Studio 3.4
Android手机或开发版，NPU的功能暂时只在nova5、mate30和mate30 5G上进行了测试，用户可自行尝试其它搭载了麒麟810和990芯片的华为手机（如nova5i pro、mate30 pro、荣耀v30，mate40或p40，且需要将系统更新到最新版）；

### 4.2 Demo
- 打开Android Studio，在"Welcome to Android Studio"窗口点击"Open an existing Android Studio project"，在弹出的路径选择窗口中进入""目录，然后点击右下角的"Open"按钮即可导入工程
- 通过USB连接Android手机或开发板；
- 载入工程后，点击菜单栏的Run->Run 'App'按钮，在弹出的"Select Deployment Target"窗口选择已经连接的Android设备，然后点击"OK"按钮；

### 4.3 PaddleX Android SDK介绍

PaddleX Android SDK是基于Paddle-Lite的安卓端AI推理工具，可实现在加载PaddleX导出的Lite模型和Yaml配置文件，方便开发者集成到业务中。
该SDK自底向上主要包括：Paddle-Lite推理引擎层，Paddle-Lite接口层以及PaddleX业务层。

- Paddle-Lite推理引擎层，是在Android上编译好的二进制包，只涉及到Kernel 的执行，且可以单独部署，以支持极致的轻量级部署。
- Paddle-Lite接口层，Android以Java接口封装了底层c++接口的上层API。
- PaddleX业务层，封装了PaddleX导出模型的预处理，推理和后处理，以及可视化，支持检测、分割、分类模型。

#### 4.3.1 安装

首先下载paddlex.aar，并拷贝到android工程目录app/libs/下面，然后为app的build.gradle添加依赖：

```
repositories {
    flatDir {
        dirs 'libs' 
   }
}

dependencies {
    implementation(name:'paddlex', ext:'aar')
}
```

#### 4.3.2 SDK API用例

```
import com.baidu.paddlex.Predictor;
import com.baidu.paddlex.config.ConfigParser;
import com.baidu.paddlex.postprocess.DetResult;
import com.baidu.paddlex.postprocess.SegResult;
import com.baidu.paddlex.postprocess.ClsResult;
import com.baidu.paddlex.visual.Visualize;

// Predictor
Predictor predictor = new Predictor();
// model config
ConfigParser configParser = new ConfigParser();
// Visualize
Visualize visualize = new Visualize();
// image to predict
Bitmap predictImage;

// initialize
configParser.init(context, model_path, yaml_path, cpu_thread_num, cpu_power_mode);
visualize.init(configParser.getNumClasses());
predictor.init(context, configParser)

if (predictImage != null && predictor.isLoaded()) {
    predictor.setInputImage(predictImage);
    runModel();
}

if (configParser.getModelType().equalsIgnoreCase("segmenter")) {
    SegResult segResult = predictor.getSegResult();
    outputImage = visualize.draw(segResult, predictor.getInputImage(), predictor.getImageBlob());
} else if (configParser.getModelType().equalsIgnoreCase("detector")) {
    DetResult detResult = predictor.getDetResult();
    outputImage = visualize.draw(detResult, predictor.getInputImage());
} else if (configParser.getModelType().equalsIgnoreCase("classifier")) {
    ClsResult clsResult = predictor.getClsResult();
}
```
