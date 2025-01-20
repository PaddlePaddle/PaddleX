---
comments: true
---

# Tutorial for Multilingual Speech Recognition Module

## I. Overview
Speech recognition is an advanced tool that can automatically convert human-spoken multiple languages into corresponding texts. This technology also plays an important role in various fields such as intelligent customer service, voice assistants, and meeting minutes. Multilingual speech recognition can support automatic language retrieval and recognize speech in multiple different languages.

## II. Supported Model List

### Whisper Model
Demo Link | Training Data | Size | Descriptions | CER | Model
:-----------: | :-----:| :-------: | :-----: | :-----: |:---------:|
Whisper | 680kh from internet | large: 5.8G,</br>medium: 2.9G,</br>small: 923M,</br>base: 277M,</br>tiny: 145M | Encoder:Transformer,</br> Decoder:Transformer, </br>Decoding method: </br>Greedy search | 0.027 </br>(large, Librispeech) | [whisper-large](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-large-model.tar.gz) </br>[whisper-medium](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-medium-model.tar.gz) </br>[whisper-small](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-small-model.tar.gz) </br>[whisper-base](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-base-model.tar.gz) </br>[whisper-tiny](https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-tiny-model.tar.gz) </br>

## III. Quick Integration
Before quick integration, you need to install the PaddleX wheel package. For the installation method, please refer to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md). After installing the wheel package, a few lines of code can complete the inference of the text recognition module. You can switch models under this module freely, and you can also integrate the model inference of the text recognition module into your project.

Before running the following code, please download the [demo audio](https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav) to your local machine.

```python
from paddlex import create_model
model = create_model("whisper_large")
output = model.predict("./zh.wav", batch_size=1)
for res in output:
    res.print(json_format=False)
```
For more information on using PaddleX's single-model inference APIs, please refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
Currently, this model only supports inference.

### 4.1 Data Preparation
#### 4.1.1 Demo Data Download
You can use the following commands to download the Demo dataset to a specified folder:

```bash
wget https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
```

### 4.2 Model Training
Not support for now.

### **4.3 Model Evaluation**
Not support for now.

### **4.4 Model Inference and Model Integration**

#### 4.4.1 Model Inference
To perform inference prediction via the command line, simply use the following command:

Before running the following code, please download the [demo audio](https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav) to your local machine.

```bash
python main.py -c paddlex/configs/modules/multilingual_speech_recognition/whisper_large.yaml \
    -o Global.mode=predict \
    -o Predict.input="./zh.wav"
```
the following steps are required for model inference:

* Specify the `.yaml` configuration file path for the model (here it is `whisper_large.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
Models can be directly integrated into the PaddleX pipelines or into your own projects.

1.<b>Pipeline Integration</b>

No example for now.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the text recognition module. Refer to the [Quick Integration](#iii-quick-integration) Python example code. Simply replace the model with the path to your trained model.
