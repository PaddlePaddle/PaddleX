<p align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/logo.png" width="735" height ="200" alt="PaddleX" align="middle" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202-red.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python-3.8%2C%203.9%2C%203.10-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Windows-orange.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/hardware-CPU%2C%20GPU%2C%20XPU%2C%20NPU%2C%20MLU%2C%20DCU-yellow.svg"></a>
</p>

<h4 align="center">
  <a href=#-why-paddlex->🌟 Features</a> | <a href=https://aistudio.baidu.com/pipeline/mine>🌐  Online Experience</a>｜<a href=#️-quick-start>🚀  Quick Start</a> | <a href=#-documentation> 📖 Documentation</a> | <a href=#-what-can-paddlex-do> 🔥Capabilities</a> | <a href=./docs/support_list/models_list_en.md> 📋 Models</a>
</h4>

<h5 align="center">
  <a href="README.md">🇨🇳 Simplified Chinese</a> | <a href="README_en.md">🇬🇧 English</a></a>
</h5>

## 🔍 Introduction

PaddleX 3.0 is a low-code development tool for AI models built on the PaddlePaddle framework. It integrates numerous **ready-to-use pre-trained models**, enabling **full-process development** from model training to inference, supporting **a variety of mainstream hardware** both domestic and international, and aiding AI developers in industrial practice.
 

|                                                            [**Image Classification**](./docs/pipeline_usage/tutorials/cv_pipelines/image_classification_en.md)                                                            |                                                            [**Multi-label Image Classification**](./docs/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification_en.md)                                                            |                                                            [**Object Detection**](./docs/pipeline_usage/tutorials/cv_pipelines/object_detection_en.md)                                                            |                                                            [**Instance Segmentation**](./docs/pipeline_usage/tutorials/cv_pipelines/instance_segmentation_en.md)                                                            |
|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/b302cd7e-e027-4ea6-86d0-8a4dd6d61f39" height="126px" width="180px"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/multilabel_cls.png" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/099e2b00-0bbe-4b20-9c5a-96b69e473bd2" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/09f683b4-27df-4c24-b8a7-84da20fdd182" height="126px" width="180px"> |
|                                                              [**Semantic Segmentation**](./docs/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation_en.md)                                                               |                                                            [**Image Anomaly Detection**](./docs/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection_en.md)                                                            |                                                          [**OCR**](./docs/pipeline_usage/tutorials/ocr_pipelines/OCR_en.md)                                                          |                                                          [**Table Recognition**](./docs/pipeline_usage/tutorials/ocr_pipelines/table_recognition_en.md)                                                          |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/02637f8c-f248-415b-89ab-1276505f198c" height="126px" width="180px"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/image_anomaly_detection.png" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1ef48536-48d4-484b-a6fb-0d6631ba2386" height="126px" width="180px"> |  <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1e798e05-dee7-4b41-9cc4-6708b6014efa" height="126px" width="180px"> |
|                                                              [**PP-ChatOCRv3-doc**](./docs/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction_en.md)                                                              |                                                            [**Time Series Forecasting**](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting_en.md)                                                            |                                                              [**Time Series Anomaly Detection**](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection_en.md)                                                              |                                                         [**Time Series Classification**](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_classification_en.md)                                                         |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/e3d97f4e-ab46-411c-8155-494c61492b0a" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/6e897bf6-35fe-45e6-a040-e9a1a20cfdf2" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/c54c66cc-da4f-4631-877b-43b0fbb192a6" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0ce925b2-3776-4dde-8ce0-5156d5a2476e" height="126px" width="180px"> |

## 🌟 Why PaddleX ?

  🎨 **Rich Models One-click Call**: Integrate over **200 PaddlePaddle models** covering multiple key areas such as OCR, object detection, and time series forecasting into **19 pipelines**. Experience the model effects quickly through easy Python API calls. Also supports **more than 20 modules** for easy model combination use by developers.

  🚀 **High Efficiency and Low barrier of entry**: Achieve model **full-process development** based on graphical interfaces and unified commands, creating **8 featured model pipelines** that combine large and small models, semi-supervised learning of large models, and multi-model fusion, greatly reducing the cost of iterating models.

  🌐 **Flexible Deployment in Various Scenarios**: Support various deployment methods such as **high-performance deployment**, **service-oriented deployment**, and **edge deployment** to ensure efficient operation and rapid response of models in different application scenarios.

  🔧 **Efficient Support for Mainstream Hardware**: Support seamless switching of various mainstream hardware such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU to ensure efficient operation.

## 📣 Recent Updates

🔥🔥 **"PaddleX Document Information Personalized Extraction Upgrade"**, PP-ChatOCRv3 innovatively provides custom development functions for OCR models based on data fusion technology, offering stronger model fine-tuning capabilities. Millions of high-quality general OCR text recognition data are automatically integrated into vertical model training data at a specific ratio, solving the problem of weakened general text recognition capabilities caused by vertical model training in the industry. Suitable for practical scenarios in industries such as automated office, financial risk control, healthcare, education and publishing, and legal and government sectors. **October 17th (Thursday) at 19:00** live broadcast will provide a detailed interpretation of data fusion technology and how to use prompt engineering to achieve better information extraction results. [Registration Link](https://www.wjx.top/vm/mFhGfwx.aspx?udsid=772552)

🔥🔥 **9.30, 2024**, PaddleX 3.0 Beta1 open source version is officially released, providing **more than 200 models** that can be called with a simple Python API; achieve model full-process development based on unified commands, and open source the basic capabilities of the **PP-ChatOCRv3** pipeline; support **more than 100 models for high-performance inference and service-oriented deployment** (iterating continuously), **more than 7 key visual models for edge-deployment**; **more than 70 models have been adapted for the full development process of Ascend 910B**, **more than 15 models have been adapted for the full development process of Kunlun chips and Cambricon**

🔥 **6.27, 2024**, PaddleX 3.0 Beta open source version is officially released, supporting the use of various mainstream hardware for pipeline and model development in a low-code manner on the local side.

🔥 **3.25, 2024**, PaddleX 3.0 cloud release, supporting the creation of pipelines in the AI Studio Galaxy Community in a zero-code manner.

## 🔠 Explanation of Pipeline
PaddleX is dedicated to achieving pipeline-level model training, inference, and deployment. A pipeline refers to a series of predefined development processes for specific AI tasks, which includes a combination of single models (single-function modules) capable of independently completing a certain type of task.

## 📊 What can PaddleX do？

All pipelines of PaddleX support **online experience** and local **inference**. You can quickly experience the effects of each pre-trained pipeline. If you are satisfied with the effects of the pre-trained pipeline, you can directly perform [high-performance inference](./docs/pipeline_deploy/high_performance_deploy_en.md) / [Service-Oriented Deployment](./docs/pipeline_deploy/service_deploy_en.md) / [edge deployment](./docs/pipeline_deploy/lite_deploy_en.md) on the pipeline. If not satisfied, you can also **Custom Development** to improve the pipeline effect. For the complete pipeline development process, please refer to the [PaddleX pipeline Development Tool Local Use Tutorial](./docs/pipeline_usage/pipeline_develop_guide_en.md).

In addition, PaddleX provides developers with a full-process efficient model training and deployment tool based on a [cloud-based GUI](https://aistudio.baidu.com/pipeline/mine). Developers **do not need code development**, just need to prepare a dataset that meets the pipeline requirements to **quickly start model training**. For details, please refer to the tutorial ["Developing Industrial-level AI Models with Zero Barrier"](https://aistudio.baidu.com/practical/introduce/546656605663301).

<table>
    <tr>
        <th>Pipeline</th>
        <th>Online Experience</th>
        <th>Local Inference</th>
        <th>High-Performance Inference</th>
        <th>Service-Oriented Deployment</th>
        <th>Edge Deployment</th>
        <th>Custom Development</th>
        <th><a href="https://aistudio.baidu.com/pipeline/mine">No-Code Development On AI Studio</a></td> 
    </tr>
    <tr>
        <td>OCR</td>
        <td><a href="https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>PP-ChatOCRv3</td>
        <td><a href="https://aistudio.baidu.com/community/app/182491/webUI?source=appCenter">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Table Recognition</td>
        <td><a href="https://aistudio.baidu.com/community/app/91661?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Object Detection</td>
        <td><a href="https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Instance Segmentation</td>
        <td><a href="https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Classification</td>
        <td><a href="https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Semantic Segmentation</td>
        <td><a href="https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Forecasting</td>
        <td><a href="https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Anomaly Detection</td>
        <td><a href="https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Classification</td>
        <td><a href="https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
        <tr>
        <td>Small Object Detection</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
        <tr>
        <td>Multi-label Image Classification</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>Image Anomaly Detection</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>Formula Recognition</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>Seal Recognition</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>Image Recognition</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>Pedestrian Attribute Recognition</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>Vehicle Attribute Recognition</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>Face Recognition</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>Layout Parsing</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
</table>

> ❗Note: The above capabilities are implemented based on GPU/CPU. PaddleX can also perform local inference and custom development on mainstream hardware such as Kunlunxin, Ascend, Cambricon, and Haiguang. The table below details the support status of the pipelines. For specific supported model lists, please refer to the [Model List (Kunlunxin XPU)](./docs/support_list/model_list_xpu_en.md)/[Model List (Ascend NPU)](./docs/support_list/model_list_npu_en.md)/[Model List (Cambricon MLU)](./docs/support_list/model_list_mlu_en.md)/[Model List (Haiguang DCU)](./docs/support_list/model_list_dcu_en.md). We are continuously adapting more models and promoting the implementation of high-performance and service-oriented deployment on mainstream hardware.

🔥🔥 **Support for Domestic Hardware Capabilities**

<table>
  <tr>
    <th>Pipeline</th>
    <th>Ascend 910B</th>
    <th>Kunlunxin R200/R300</th>
    <th>Cambricon MLU370X8</th>
    <th>Haiguang Z100</th>
  </tr>
  <tr>
    <td>OCR</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>Table Recognition</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>Object Detection</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>Instance Segmentation</td>
    <td>✅</td>
    <td>🚧</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>Image Classification</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>Semantic Segmentation</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>Time Series Forecasting</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>Time Series Anomaly Detection</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>Time Series Classification</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
</table>


## ⏭️ Quick Start

### 🛠️ Installation

> ❗Before installing PaddleX, please ensure you have a basic **Python environment** (Note: Currently supports Python 3.8 to Python 3.10, with more Python versions being adapted).

* **Installing PaddlePaddle**
```bash
# cpu
python -m pip install paddlepaddle==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# gpu，该命令仅适用于 CUDA 版本为 11.8 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，该命令仅适用于 CUDA 版本为 12.3 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```
> ❗For more PaddlePaddle versions, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation./docs/zh/install/pip/linux-pip.html). 

* **Installing PaddleX**

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0b1-py3-none-any.whl
```

> ❗For more installation methods, refer to the [PaddleX Installation Guide](./docs/installation/installation_en.md).


### 💻 CLI Usage

One command can quickly experience the pipeline effect, the unified CLI format is:

```bash
paddlex --pipeline [Pipeline Name] --input [Input Image] --device [Running Device]
```

You only need to specify three parameters:
* `pipeline`: The name of the pipeline
* `input`: The local path or URL of the input image to be processed
* `device`: The GPU number used (for example, `gpu:0` means using the 0th GPU), you can also choose to use the CPU (`cpu`)

For example, using the  OCR pipeline:
```bash
paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png  --device gpu:0
```
<details>
  <summary><b>👉 Click to view the running result</b></summary>

```bash
{
'input_path': '/root/.paddlex/predict_input/general_ocr_002.png', 
'dt_polys': [array([[161,  27],
       [353,  22],
       [354,  69],
       [162,  74]], dtype=int16), array([[426,  26],
       [657,  21],
       [657,  58],
       [426,  62]], dtype=int16), array([[702,  18],
       [822,  13],
       [824,  57],
       [704,  62]], dtype=int16), array([[341, 106],
       [405, 106],
       [405, 128],
       [341, 128]], dtype=int16)
       ...], 
'dt_scores': [0.758478200014338, 0.7021546472698513, 0.8536622648391111, 0.8619181462164781, 0.8321051217096188, 0.8868756173427551, 0.7982964727675609, 0.8289939036796322, 0.8289428877522524, 0.8587063317632897, 0.7786755892491615, 0.8502032769081344, 0.8703346500042997, 0.834490931790065, 0.908291103353393, 0.7614978661708064, 0.8325774055997542, 0.7843421347676149, 0.8680889482955594, 0.8788859304537682, 0.8963341277518075, 0.9364654810069546, 0.8092413027028257, 0.8503743089091863, 0.7920740420391101, 0.7592224394793805, 0.7920547400069311, 0.6641757962457888, 0.8650289477605955, 0.8079483304467047, 0.8532207681055275, 0.8913377034754717], 
'rec_text': ['登机牌', 'BOARDING', 'PASS', '舱位', 'CLASS', '序号 SERIALNO.', '座位号', '日期 DATE', 'SEAT NO', '航班 FLIGHW', '035', 'MU2379', '始发地', 'FROM', '登机口', 'GATE', '登机时间BDT', '目的地TO', '福州', 'TAIYUAN', 'G11', 'FUZHOU', '身份识别IDNO', '姓名NAME', 'ZHANGQIWEI', 票号TKTNO', '张祺伟', '票价FARE', 'ETKT7813699238489/1', '登机口于起飞前10分钟关闭GATESCLOSE10MINUTESBEFOREDEPARTURETIME'], 
'rec_score': [0.9985831379890442, 0.999696917533874512, 0.9985735416412354, 0.9842517971992493, 0.9383274912834167, 0.9943678975105286, 0.9419361352920532, 0.9221674799919128, 0.9555020928382874, 0.9870321154594421, 0.9664073586463928, 0.9988052248954773, 0.9979352355003357, 0.9985110759735107, 0.9943482875823975, 0.9991195797920227, 0.9936401844024658, 0.9974591135978699, 0.9743705987930298, 0.9980487823486328, 0.9874696135520935, 0.9900962710380554, 0.9952947497367859, 0.9950481653213501, 0.989926815032959, 0.9915552139282227, 0.9938777685165405, 0.997239887714386, 0.9963340759277344, 0.9936134815216064, 0.97223961353302]}
```

The visualization result is as follows:

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/boardingpass.png)

</details>

To use the command line for other pipelines, simply adjust the `pipeline` parameter to the name of the corresponding pipeline. Below are the commands for each pipeline:

<details>
  <summary><b>👉 More CLI usage for pipelines</b></summary>

| Pipeline Name                | Command                                                                                                                                                                                    |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Image Classification | `paddlex --pipeline image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0`                    |
| Object Detection     | `paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device gpu:0`                            |
| Instance Segmentation| `paddlex --pipeline instance_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png --device gpu:0`                  |
| Semantic Segmentation| `paddlex --pipeline semantic_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/application/semantic_segmentation/makassaridn-road_demo.png --device gpu:0` |
| Image Multi-label Classification | `paddlex --pipeline multi_label_image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0`        |
| Small Object Detection       | `paddlex --pipeline small_object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg --device gpu:0`                            |
| Image Anomaly Detection      | `paddlex --pipeline anomaly_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png --device gpu:0 `                                              |
| OCR                  | `paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0`                                                      |
| Table Recognition    | `paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --device gpu:0`                                      |
| Time Series Forecasting | `paddlex --pipeline ts_fc --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv --device gpu:0`                                                                   |
| Time Series Anomaly Detection | `paddlex --pipeline ts_ad --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv --device gpu:0`                                                                    |
| Time Series Classification | `paddlex --pipeline ts_cls --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv --device gpu:0`                                                                 |

</details>

### 📝 Python Script Usage

A few lines of code can complete the quick inference of the pipeline, the unified Python script format is as follows:
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline=[Pipeline Name])
output = pipeline.predict([Input Image Name])
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```
The following steps are executed:

* `create_pipeline()` instantiates the pipeline object
* Passes the image and calls the `predict` method of the pipeline object for inference prediction
* Processes the prediction results

For other pipelines in Python scripts, just adjust the `pipeline` parameter of the `create_pipeline()` method to the corresponding name of the pipeline. Below is a list of each pipeline's corresponding parameter name and detailed usage explanation:

<details>
  <summary>👉 More Python script usage for pipelines</summary>

| pipeline Name           | Corresponding Parameter               | Detailed Explanation                                                                                                      |
|-------------------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------|
| PP-ChatOCRv3-doc   | `PP-ChatOCRv3-doc` | [PP-ChatOCRv3-doc Pipeline Python Script Usage Instructions](./docs/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction_en.md) |
|  Image Classification       | `image_classification` | [ Image Classification Pipeline Python Script Usage Instructions](./docs/pipeline_usage/tutorials/cv_pipelines/image_classification_en.md) |
|  Object Detection       | `object_detection` | [ Object Detection Pipeline Python Script Usage Instructions](./docs/pipeline_usage/tutorials/cv_pipelines/object_detection_en.md) |
|  Instance Segmentation       | `instance_segmentation` | [ Instance Segmentation Pipeline Python Script Usage Instructions](./docs/pipeline_usage/tutorials/cv_pipelines/instance_segmentation_en.md) |
|  Semantic Segmentation       | `semantic_segmentation` | [ Semantic Segmentation Pipeline Python Script Usage Instructions](./docs/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation_en.md) |
|  Image Multi-Label Classification | `multilabel_classification` | [ Image Multi-Label Classification Pipeline Python Script Usage Instructions](./docs/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification_en.md) |
| Small Object Detection         | `small_object_detection` | [Small Object Detection Pipeline Python Script Usage Instructions](./docs/pipeline_usage/tutorials/cv_pipelines/small_object_detection_en.md) |
| Image Anomaly Detection       | `image_classification` | [Image Anomaly Detection Pipeline Python Script Usage Instructions](./docs/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection_en.md) |
|  OCR            | `OCR` | [ OCR Pipeline Python Script Usage Instructions](./docs/pipeline_usage/tutorials/ocr_pipelines/OCR_en.md) |
|  Table Recognition       | `table_recognition` | [ Form Recognition Pipeline Python Script Usage Instructions](./docs/pipeline_usage/tutorials/ocr_pipelines/table_recognition_en.md) |
|  Time Series Forecast       | `ts_forecast` | [ Time Series Forecast Pipeline Python Script Usage Instructions](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting_en.md) |
|  Time Series Anomaly Detection   | `ts_anomaly_detection` | [ Time Series Anomaly Detection Pipeline Python Script Usage Instructions](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection_en.md) |
|  Time Series Classification       | `ts_cls` | [ Time Series Classification Pipeline Python Script Usage Instructions](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_classification_en.md) |
</details>

## 📖 Documentation
<details>
  <summary> <b> ⬇️ Installation </b></summary>
  
  * [📦 PaddlePaddle Installation](./docs/installation/paddlepaddle_install_en.md)
  * [📦 PaddleX Installation](./docs/installation/installation_en.md) 

</details>

<details open>
<summary> <b> 🔥 Pipeline Usage </b></summary>

* [📑 PaddleX pipeline Usage Overview](./docs/pipeline_usage/pipeline_develop_guide_en.md)

* <details open>
    <summary> <b> 📝 Information Extracion</b></summary>

   * [📄 PP-ChatOCRv3 pipeline Tutorial](./docs/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction_en.md)
  </details>

* <details open>
    <summary> <b> 🔍 OCR </b></summary>

    * [📜 OCR pipeline Tutorial](./docs/pipeline_usage/tutorials/ocr_pipelines/OCR_en.md)
    * [📊 Table Recognition pipeline Tutorial](./docs/pipeline_usage/tutorials/ocr_pipelines/table_recognition_en.md)
  </details>

* <details open>
    <summary> <b> 🎥 Computer Vision </b></summary>

   * [🖼️ Image Classification pipeline Tutorial](./docs/pipeline_usage/tutorials/cv_pipelines/image_classification_en.md)
   * [🎯 Object Detection pipeline Tutorial](./docs/pipeline_usage/tutorials/cv_pipelines/object_detection_en.md)
   * [📋 Instance Segmentation pipeline Tutorial](./docs/pipeline_usage/tutorials/cv_pipelines/instance_segmentation_en.md)
   * [🗣️ Semantic Segmentation pipeline Tutorial](./docs/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation_en.md)
   * [🏷️ Multi-label Image Classification pipeline Tutorial](./docs/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification_en.md)
   * [🔍 Small Object Detection pipeline Tutorial](./docs/pipeline_usage/tutorials/cv_pipelines/small_object_detection_en.md)
   * [🖼️ Image Anomaly Detection pipeline Tutorial](./docs/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection_en.md)
  </details>
  
* <details open>
    <summary> <b> ⏱️ Time Series Analysis</b> </summary>

   * [📈 Time Series Forecasting pipeline Tutorial](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting_en.md)
   * [📉 Time Series Anomaly Detection pipeline Tutorial](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection_en.md)
   * [🕒 Time Series Classification pipeline Tutorial](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_classification_en.md)
  </details>

* <details open>
    <summary> <b>🔧 Related Instructions</b> </summary>

   * [🖥️ PaddleX pipeline Command Line Instruction](./docs/pipeline_usage/instructions/pipeline_CLI_usage_en.md)
   * [📝 PaddleX pipeline Python Script Instruction](./docs/pipeline_usage/instructions/pipeline_python_API_en.md)
  </details>
  
</details>

<details open>
<summary> <b> ⚙️ Module Usage </b></summary>

* <details open>
  <summary> <b> 🔍 OCR </b></summary>

  * [📝 Text Detection Module Tutorial](./docs/module_usage/tutorials/ocr_modules/text_detection_en.md)
  * [🔖 Seal Text Detection Module Tutorial](./docs/module_usage/tutorials/ocr_modules/seal_text_detection_en.md)
  * [🔠 Text Recognition Module Tutorial](./docs/module_usage/tutorials/ocr_modules/text_recognition_en.md)
  * [🗺️ Layout Parsing Module Tutorial](./docs/module_usage/tutorials/ocr_modules/layout_detection_en.md)
  * [📊 Table Structure Recognition Module Tutorial](./docs/module_usage/tutorials/ocr_modules/table_structure_recognition_en.md)
  * [📄 Document Image Orientation Classification Tutorial](./docs/module_usage/tutorials/ocr_modules/doc_img_orientation_classification_en.md)
  * [🔧 Document Image Unwarp Module Tutorial](./docs/module_usage/tutorials/ocr_modules/text_image_unwarping_en.md)
  * [📐 Formula Recognition Module Tutorial](./docs/module_usage/tutorials/ocr_modules/formula_recognition_en.md)
  </details>

* <details open>
  <summary> <b> 🖼️ Image Classification </b></summary>

  * [📂 Image Classification Module Tutorial](./docs/module_usage/tutorials/cv_modules/image_classification_en.md)
  * [🏷️ Multi-label Image Classification Module Tutorial](./docs/module_usage/tutorials/cv_modules/ml_classification_en.md)

  * [👤 Pedestrian Attribute Recognition Module Tutorial](./docs/module_usage/tutorials/cv_modules/pedestrian_attribute_recognition_en.md)
  * [🚗 Vehicle Attribute Recognition Module Tutorial](./docs/module_usage/tutorials/cv_modules/vehicle_attribute_recognition_en.md)

  </details>

* <details open>
  <summary> <b> 🏞️ Image Features </b></summary>

    * [🔗 Image Feature Module Tutorial](./docs/module_usage/tutorials/cv_modules//image_feature_en.md)
  </details>

* <details open>
  <summary> <b> 🎯 Object Detection </b></summary>

  * [🎯 Object Detection Module Tutorial](./docs/module_usage/tutorials/cv_modules/object_detection_en.md)
  * [📏 Small Object Detection Module Tutorial](./docs/module_usage/tutorials/cv_modules/small_object_detection_en.md)
  * [🧑‍🤝‍🧑 Face Detection Module Tutorial](./docs/module_usage/tutorials/cv_modules/face_detection_en.md)
  * [🔍 Mainbody Detection Module Tutorial](./docs/module_usage/tutorials/cv_modules/mainbody_detection_en.md)
  * [🚶 Pedestrian Detection Module Tutorial](./docs/module_usage/tutorials/cv_modules/human_detection_en.md)
  * [🚗 Vehicle Detection Module Tutorial](./docs/module_usage/tutorials/cv_modules/vehicle_detection_en.md)

  </details>

* <details open>
  <summary> <b> 🖼️ Image Segmentation </b></summary>

  * [🗺️ Semantic Segmentation Module Tutorial](./docs/module_usage/tutorials/cv_modules/semantic_segmentation_en.md)
  * [🔍 Instance Segmentation Module Tutorial](./docs/module_usage/tutorials/cv_modules/instance_segmentation_en.md)
  * [🚨 Image Anomaly Detection Module Tutorial](./docs/module_usage/tutorials/cv_modules/anomaly_detection_en.md)
  </details>

* <details open>
  <summary> <b> ⏱️ Time Series Analysis </b></summary>

  * [📈 Time Series Forecasting Module Tutorial](./docs/module_usage/tutorials/ts_modules/time_series_forecast_en.md)
  * [🚨 Time Series Anomaly Detection Module Tutorial](./docs/module_usage/tutorials/time_series_modules/time_series_anomaly_detection.md)
  * [🕒 Time Series Classification Module Tutorial](./docs/module_usage/tutorials/ts_modules/time_series_classification_en.md)
  </details>
    
* <details open>
  <summary> <b> 📄 Related Instructions </b></summary>

  * [📝 PaddleX Single Model Python Script Instruction](./docs/module_usage/instructions/model_python_API_en.md)
  * [📝 PaddleX General Model Configuration File Parameter Instruction](./docs/module_usage/instructions/config_parameters_common_en.md)
  * [📝 PaddleX Time Series Task Model Configuration File Parameter Instruction](./docs/module_usage/instructions/config_parameters_time_series_en.md)
  </details>

</details>

<details open>
  <summary> <b> 🏗️ Pipeline Deployment </b></summary>

  * [🚀 PaddleX High-Performance Inference Tutorial](./docs/pipeline_deploy/high_performance_deploy_en.md)
  * [🖥️ PaddleX Service-Oriented Deployment Tutorial](./docs/pipeline_deploy/service_deploy_en.md)
  * [📱 PaddleX Edge Deployment Tutorial](./docs/pipeline_deploy/lite_deploy_en.md)

</details>
<details open>
  <summary> <b> 🖥️ Multi-Hardware Usage </b></summary>

  * [⚙️ Multi-Hardware Usage Guide](./docs/other_devices_support/multi_devices_use_guide_en.md)
  * [⚙️ DCU Paddle Installation](./docs/other_devices_support/paddlepaddle_install_DCU_en.md)
  * [⚙️ MLU Paddle Installation](./docs/other_devices_support/paddlepaddle_install_MLU_en.md)
  * [⚙️ NPU Paddle Installation](./docs/other_devices_support/paddlepaddle_install_NPU_en.md)
  * [⚙️ XPU Paddle Installation](./docs/other_devices_support/paddlepaddle_install_XPU_en.md)

</details>

<details>
  <summary> <b> 📝 Tutorials & Examples </b></summary>

* [🖼️ General Image Classification Model Line —— Garbage Classification Tutorial](./docs/practical_tutorials/image_classification_garbage_tutorial_en.md)
* [🧩 General Instance Segmentation Model Line —— Remote Sensing Image Instance Segmentation Tutorial](./docs/practical_tutorials/instance_segmentation_remote_sensing_tutorial_en.md)
* [👥 General Object Detection Model Line —— Pedestrian Fall Detection Tutorial](./docs/practical_tutorials/object_detection_fall_tutorial_en.md)
* [👗 General Object Detection Model Line —— Fashion Element Detection Tutorial](./docs/practical_tutorials/object_detection_fashion_pedia_tutorial_en.md)
* [🚗 General OCR Model Line —— License Plate Recognition Tutorial](./docs/practical_tutorials/ocr_det_license_tutorial_en.md)
* [✍️ General OCR Model Line —— Handwritten Chinese Character Recognition Tutorial](./docs/practical_tutorials/ocr_rec_chinese_tutorial_en.md)
* [🗣️ General Semantic Segmentation Model Line —— Road Line Segmentation Tutorial](./docs/practical_tutorials/semantic_segmentation_road_tutorial_en.md)
* [🛠️ Time Series Anomaly Detection Model Line —— Equipment Anomaly Detection Application Tutorial](./docs/practical_tutorials/ts_anomaly_detection_en.md)
* [🎢 Time Series Classification Model Line —— Heartbeat Monitoring Time Series Data Classification Application Tutorial](./docs/practical_tutorials/ts_classification_en.md)
* [🔋 Time Series Forecasting Model Line —— Long-term Electricity Consumption Forecasting Application Tutorial](./docs/practical_tutorials/ts_forecast_en.md)

  </details>




## 🤔 FAQ

For answers to some common questions about our project, please refer to the [FAQ](./docs/FAQ_en.md). If your question has not been answered, please feel free to raise it in [Issues](https://github.com/PaddlePaddle/PaddleX/issues).

## 💬 Discussion

We warmly welcome and encourage community members to raise questions, share ideas, and feedback in the [Discussions](https://github.com/PaddlePaddle/PaddleX/discussions) section. Whether you want to report a bug, discuss a feature request, seek help, or just want to keep up with the latest project news, this is a great platform.

## 📄 License

The release of this project is licensed under the [Apache 2.0 license](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta/LICENSE).






