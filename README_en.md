<p align="center">
  <img src="/tmp/logo.png" width="735" height ="200" alt="PaddleX" align="middle" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-red.svg"></a> 
    <a href=""><img src="https://img.shields.io/badge/python-3.8%2C%203.9%2C%203.10-blue.svg"></a> 
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20windows-orange.svg"></a> 
    <a href=""><img src="https://img.shields.io/badge/hardware-intel  cpu%2C%20gpu%2C%20xpu%2C%20npu%2C%20mlu-yellow.svg"></a>
</p>

<h4 align="center">
  <a href=##-why-paddlex->🌟 Features</a> | <a href=https://aistudio.baidu.com/pipeline/mine>🌐  Online Experience</a>｜<a href=#️-quick-start>🚀  Quick Start</a> | <a href=#-documentation> 📖 Documentation</a> | <a href=/docs_new_en/support_list/pipelines_list_en.md> 🔥Pipelines List</a>
</h4>

[](/docs_new_en/support_list/pipelines_list_en.md)
<h5 align="center">
  <a href="README.md">🇨🇳 Simplified Chinese</a> | <a href="README_en.md">🇬🇧 English</a></a>
</h5>

## 🔍 Introduction

PaddleX 3.0 is a low-code development tool for AI models built on the PaddlePaddle framework. It integrates numerous **ready-to-use pre-trained models**, enabling **full-process development** from model training to inference, supporting **a variety of mainstream hardware** both domestic and international, and aiding AI developers in industrial practice.

|                                                            **Image Classification**                                                            |                                                            **Object Detection**                                                            |                                                            **Semantic Segmentation**                                                            |                                                            **Instance Segmentation**                                                            |
|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/b302cd7e-e027-4ea6-86d0-8a4dd6d61f39" height="126px" width="180px"> | <img src="/tmp/images/multilabel_cls.png" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/099e2b00-0bbe-4b20-9c5a-96b69e473bd2" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/09f683b4-27df-4c24-b8a7-84da20fdd182" height="126px" width="180px"> |
|                                                              **Multi-label Image Classification**                                                               |                                                            **OCR**                                                            |                                                          **Table Recognition**                                                          |                                                          **PP-ChatOCR v3**                                                          |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/02637f8c-f248-415b-89ab-1276505f198c" height="126px" width="180px"> | <img src="https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=31aabc8473cb49d982126b48a864eaa2&docGuid=1NT96A2Q0Ln0o-" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1ef48536-48d4-484b-a6fb-0d6631ba2386" height="126px" width="180px"> |  <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1e798e05-dee7-4b41-9cc4-6708b6014efa" height="126px" width="180px"> |
|                                                              **Time Series Forecasting**                                                              |                                                            **Time Series Anomaly Detection**                                                            |                                                              **Time Series Classification**                                                              |                                                         **Image Anomaly Detection**                                                         |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/e3d97f4e-ab46-411c-8155-494c61492b0a" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/6e897bf6-35fe-45e6-a040-e9a1a20cfdf2" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/c54c66cc-da4f-4631-877b-43b0fbb192a6" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0ce925b2-3776-4dde-8ce0-5156d5a2476e" height="126px" width="180px"> |

## 🌟 Why PaddleX ?
🔥🔥《PaddleX Document Information Personalized Extraction Upgrade》，PP-ChatOCRv3 innovatively provides OCR model secondary development capabilities based on data fusion technology, with stronger model fine-tuning capabilities. Millions of high-quality general OCR text recognition data are automatically integrated into the vertical model training data at specific ratios, solving the problem of weakening general text recognition capabilities caused by industry-specific model training. Suitable for actual scenarios in industries such as automated office, financial risk control, healthcare, education and publishing, and legal party and government. October 10th (Thursday) 19:00 live broadcast to detail the data fusion technology and how to use prompt engineering to achieve better information extraction effects.
  🎨 **Rich Models One-click Call**: Integrate over **200 PaddlePaddle models** covering multiple key areas such as OCR, object detection, and time series forecasting into **13 model pipelines**. Experience the model effects quickly through minimalist Python API calls. Also supports **more than 20 modules** for easy model combination use by developers.

  🚀 **High Efficiency and Low barrier of entry**: Achieve model **full-process development** based on graphical interfaces and unified commands, creating **8 featured model pipelines** that combine large and small models, semi-supervised learning of large models, and multi-model fusion, greatly reducing the cost of iterating models.

  🌐 **Flexible Deployment in Various Scenarios**: Support various deployment methods such as **high-performance deployment**, **service deployment**, and **lite deployment** to ensure efficient operation and rapid response of models in different application scenarios.

  🔧 **Efficient Support for Mainstream Hardware**: Support seamless switching of various mainstream hardware such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU to ensure efficient operation.

## 📣 Recent Updates

🔥🔥 **9.30, 2024**, PaddleX 3.0 Beta1 open source version is officially released, providing **more than 200 models** that can be called with a minimalist Python API; achieve model full-process development based on unified commands, and open source the basic capabilities of the **PP-ChatOCRv3** featured model production line; support **more than 100 models for high-performance inference and service-oriented deployment** (iterating continuously), **more than 7 key visual models for edge-side deployment**; **more than 70 models have been adapted for the full development process of Ascend 910B**, **more than 15 models have been adapted for the full development process of Kunlun chips and Cambricon**

🔥 **6.27, 2024**, PaddleX 3.0 Beta open source version is officially released, supporting the use of various mainstream hardware for production line and model development in a low-code manner on the local side.

🔥 **3.25, 2024**, PaddleX 3.0 cloud release, supporting the creation of pipelines in the AI Studio Galaxy Community in a zero-code manner.

## 📊 What can PaddleX do？

All pipelines of PaddleX support **online experience** and local **fast inference**. You can quickly experience the pre-trained effects of each production line. If you are satisfied with the pre-trained effects of the production line, you can directly perform [high-performance deployment](/docs_new_en/pipeline_deploy/high_performance_deploy_en.md) / [service deployment](/docs_new_en/pipeline_deploy/service_deploy_en.md) / [lite deployment](/docs_new_en/pipeline_deploy/lite_deploy_en.md) on the production line. If not satisfied, you can also **second development** to improve the production line effect. For the complete production line development process, please refer to the [PaddleX Production Line Development Tool Local Use Tutorial](/docs_new_en/pipeline_usage/pipeline_develop_guide_en.md).

In addition, PaddleX provides developers with a full-process efficient model training and deployment tool based on a [cloud-based graphical development interface](https://aistudio.baidu.com/pipeline/mine). Developers **do not need code development**, just need to prepare a dataset that meets the production line requirements to **quickly start model training**. For details, please refer to the tutorial ["Developing Industrial-level AI Models with Zero Threshold"](https://aistudio.baidu.com/practical/introduce/546656605663301).

<table>
    <tr>
        <th>Model Production Line</th>
        <th>Online Experience</th>
        <th>Quick Inference</th>
        <th>High-Performance Deployment</th>
        <th>Service Deployment</th>
        <th>Edge Deployment</th>
        <th>Secondary Development</th>
        <th><a href="https://aistudio.baidu.com/pipeline/mine">Galaxy Zero-Code Production Line</a></td> 
    </tr>
    <tr>
        <td>General OCR</td>
        <td><a href="https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Document Scene Information Extraction v3</td>
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
        <td>General Object Detection</td>
        <td><a href="https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>General Instance Segmentation</td>
        <td><a href="https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>General Image Classification</td>
        <td><a href="https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>General Semantic Segmentation</td>
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
        <td>Image Multi-Label Classification</td>
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
        <td>General Image Recognition</td>
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
</table>

> ❗Note: All the above features are implemented based on GPU/CPU. PaddleX can also perform fast inference and secondary development on mainstream hardware such as Kunlun, Ascend, Cambricon, and Hygon. The following table details the support status of the model production line, and for the specific list of supported models, please refer to [Model List (MLU)](./docs_new_en/support_list/model_list_mlu_en.md) / [Model List (NPU)](./docs_new_en/support_list/model_list_npu_en.md) / [Model List (XPU)](./docs_new_en/support_list/model_list_xpu_en.md) / [Model List DCU](./docs_new_en/support_list/model_list_dcu_en.md). We are adapting more models and promoting the implementation of high-performance and service-oriented deployment on mainstream hardware.


<details>
  <summary>👉 Support for Domestic Hardware Capabilities</summary>

<table>
  <tr>
    <th>Production Line Name</th>
    <th>NPU 910B</th>
    <th>XPU R200/R300</th>
    <th>MLU 370X8</th>
    <th>DCU Z100</th>
  </tr>
  <tr>
    <td>General OCR</td>
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
    <td>General Object Detection</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>General Instance Segmentation</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>General Image Classification</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>General Semantic Segmentation</td>
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
</details>



## ⏭️ Quick Start

### 🛠️ Installation

> ❗Please ensure you have a basic Python runtime environment before installing PaddleX.

* **Installing PaddlePaddle**
```bash
# cpu
python -m pip install paddlepaddle

# gpu, this command is only applicable to machines with CUDA version 11.8
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu, this command is only applicable to machines with CUDA version 12.3
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```

* **Installing PaddleX**

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git 
cd PaddleX
pip install -e .
```

For more installation methods, please refer to the [PaddleX Installation Guide](/docs_new_en/installation/installation_en.md)


### 💻 CLI Usage

One command can quickly experience the production line effect, the unified CLI format is:

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
{'img_path': '/root/.paddlex/predict_input/general_ocr_002.png', 'dt_polys': [[[5, 12], [88, 10], [88, 29], [5, 31]], [[208, 14], [249, 14], [249, 22], [208, 22]], [[695, 15], [824, 15], [824, 60], [695, 60]], [[158, 27], [355, 23], [356, 70], [159, 73]], [[421, 25], [659, 19], [660, 59], [422, 64]], [[337, 104], [460, 102], [460, 127], [337, 129]], [[486, 103], [650, 100], [650, 125], [486, 128]], [[675, 98], [835, 94], [835, 119], [675, 124]], [[64, 114], [192, 110], [192, 131], [64, 134]], [[210, 108], [318, 106], [318, 128], [210, 130]], [[82, 140], [214, 138], [214, 163], [82, 165]], [[226, 136], [328, 136], [328, 161], [226, 161]], [[404, 134], [432, 134], [432, 161], [404, 161]], [[509, 131], [570, 131], [570, 158], [509, 158]], [[730, 138], [771, 138], [771, 154], [730, 154]], [[806, 136], [817, 136], [817, 146], [806, 146]], [[342, 175], [470, 173], [470, 197], [342, 199]], [[486, 173], [616, 171], [616, 196], [486, 198]], [[677, 169], [813, 166], [813, 191], [677, 194]], [[65, 181], [170, 177], [171, 202], [66, 205]], [[96, 208], [171, 205], [172, 230], [97, 232]], [[336, 220], [476, 215], [476, 237], [336, 242]], [[507, 217], [554, 217], [554, 236], [507, 236]], [[87, 229], [204, 227], [204, 251], [87, 254]], [[344, 240], [483, 236], [483, 258], [344, 262]], [[66, 252], [174, 249], [174, 271], [66, 273]], [[75, 279], [264, 272], [265, 297], [76, 303]], [[459, 297], [581, 295], [581, 320], [459, 322]], [[101, 314], [210, 311], [210, 337], [101, 339]], [[68, 344], [165, 340], [166, 365], [69, 368]], [[345, 350], [662, 346], [662, 368], [345, 371]], [[100, 459], [832, 444], [832, 465], [100, 480]]], 'dt_scores': [0.8183103704439653, 0.7609575621092027, 0.8662357274035412, 0.8619508290334809, 0.8495855993183273, 0.8676840017933314, 0.8807986687956436, 0.822308525056085, 0.8686617037621976, 0.8279022169854463, 0.952332847006758, 0.8742692553015098, 0.8477013022907575, 0.8528771493227294, 0.7622965906848765, 0.8492388224448705, 0.8344203789965632, 0.8078477124353284, 0.6300434587457232, 0.8359967356998494, 0.7618617265751318, 0.9481573079350023, 0.8712182945408912, 0.837416955846334, 0.8292475059403851, 0.7860382856406026, 0.7350527486717117, 0.8701022267947695, 0.87172526903969, 0.8779847108088126, 0.7020437651809734, 0.6611684983372949], 'rec_text': ['www.997', '151', 'PASS', '登机牌', 'BOARDING', '舱位 CLASS', '序号SERIALNO.', '座位号SEATNO', '航班 FLIGHT', '日期DATE', 'MU 2379', '03DEC', 'W', '035', 'F', '1', '始发地FROM', '登机口 GATE', '登机时间BDT', '目的地TO', '福州', 'TAIYUAN', 'G11', 'FUZHOU', '身份识别IDNO.', '姓名NAME', 'ZHANGQIWEI', '票号TKTNO.', '张祺伟', '票价FARE', 'ETKT7813699238489/1', '登机口于起飞前10分钟关闭GATESCLOSE1OMINUTESBEFOREDEPARTURETIME'], 'rec_score': [0.9617719054222107, 0.4199012815952301, 0.9652514457702637, 0.9978302121162415, 0.9853208661079407, 0.9445787072181702, 0.9714463949203491, 0.9841841459274292, 0.9564052224159241, 0.9959094524383545, 0.9386572241783142, 0.9825271368026733, 0.9356589317321777, 0.9985442161560059, 0.3965512812137604, 0.15236201882362366, 0.9976775050163269, 0.9547433257102966, 0.9974752068519592, 0.9646636843681335, 0.9907559156417847, 0.9895358681678772, 0.9374122023582458, 0.9909093379974365, 0.9796401262283325, 0.9899340271949768, 0.992210865020752, 0.9478569626808167, 0.9982215762138367, 0.9924325942993164, 0.9941263794898987, 0.96443772315979]}
......
```

The visualization result is as follows:

![alt text](tmp/images/boardingpass.png)

</details>

For other pipelines, just adjust the `pipeline` parameter to the corresponding name of the production line. Below is a list of each production line's corresponding parameter name and detailed usage explanation:

<details>
  <summary>👉 More CLI usage and explanations for pipelines</summary>

| Production Line Name           | Corresponding Parameter               | Detailed Explanation                                                                                                      |
|-------------------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Document Scene Information Extraction |                                                                                                                                                                                                                              |
| General Image Classification         | `paddlex --pipeline image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg  --device gpu:0`                           |
| General Object Detection            | `paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png  --device gpu:0`                                   |
| General Instance Segmentation       | `paddlex --pipeline instance_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png  --device gpu:0`                         |
| General Semantic Segmentation       | `paddlex --pipeline semantic_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png  --device gpu:0`                              |
| General Image Multilabel Classification | `paddlex --pipeline multilabel_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/garbage_demo.png  --device gpu:0`                                          |
| Small Object Detection              | `paddlex --pipeline smallobject_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/garbage_demo.png  --device gpu:0`                                              |
| Image Anomaly Detection            |                                                                                                                                                                                                                              |
| General OCR                        | `paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png  --device gpu:0`                                                             |
| General Table Recognition          | `paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg  --device gpu:0`                                             |
| General Time Series Forecasting    | `paddlex --pipeline ts_forecast --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv  --device` gpu:0                                                                    |
| General Time Series Anomaly Detection | `paddlex --pipeline ts_anomaly_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_anomaly_detection/test.csv  --device gpu:0` |
| General Time Series Classification  | `paddlex --pipeline ts_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_classification/test.csv  --device gpu:0`       |



</details>

### 📝 Python Script Usage

A few lines of code can complete the quick inference of the production line, the unified Python script format is as follows:
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline=[Pipeline Name])
output = pipeline.predict([Input Image Name])
for batch in output:
    for item in batch:
        res = item['result']
        res.print()
        res.save_to_img("./output/")
        res.save_to_json("./output/")
```
The following steps are executed:

* `create_pipeline()` instantiates the production line object
* Passes the image and calls the `predict` method of the production line object for inference prediction
* Processes the prediction results

For other pipelines in Python scripts, just adjust the `pipeline` parameter of the `create_pipeline()` method to the corresponding name of the production line. Below is a list of each production line's corresponding parameter name and detailed usage explanation:
<details>
  <summary>👉 More Python script usage for pipelines</summary>

| Production Line Name           | Corresponding Parameter               | Detailed Explanation                                                                                                      |
|-------------------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------|
| PP-ChatOCRv3   | `pp_chatocrv3` | [PP-ChatOCRv3 Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction_en.md) |
|  Image Classification       | `image_classification` | [ Image Classification Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_classification_en.md) |
|  Object Detection       | `object_detection` | [ Object Detection Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_classification_en.md) |
|  Instance Segmentation       | `instance_segmentation` | [ Instance Segmentation Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/instance_segmentation_en.md) |
|  Semantic Segmentation       | `semantic_segmentation` | [ Semantic Segmentation Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation_en.md) |
|  Image Multi-Label Classification | `multilabel_classification` | [ Image Multi-Label Classification Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_multi_label_lassification_en.md) |
| Small Object Detection         | `smallobject_detection` | [Small Object Detection Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/small_object_detection_en.md) |
| Image Anomaly Detection       | `image_classification` | [Image Anomaly Detection Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection_en.md) |
|  OCR            | `OCR` | [ OCR Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/ocr_pipelies/OCR_en.md) |
|  Form Recognition       | `table_recognition` | [ Form Recognition Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/ocr_pipelies/table_recognition_en.md) |
|  Time Series Forecast       | `ts_forecast` | [ Time Series Forecast Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting_en.md) |
|  Time Series Anomaly Detection   | `ts_anomaly_detection` | [ Time Series Anomaly Detection Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection_en.md) |
|  Time Series Classification       | `ts_classification` | [ Time Series Classification Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_classification_en.md) |
</details>

## 📖 Documentation
<details>
  <summary> <b> ⬇️ Installation </b></summary>
  
  * [📦 PaddlePaddle Installation Guide](/docs_new_en/installation/paddlepaddle_install_en.md)
  * [📦 PaddleX Installation Guide](/docs_new_en/installation/installation_en.md) 

</details>

<details open>
<summary> <b> 🔥 Production Line Usage </b></summary>

* [📑 PaddleX Production Line Usage Overview](/docs_new_en/pipeline_usage/pipeline_develop_guide_en.md)

* <details>
    <summary> <b> 📝 Text and Image Intelligent Analysis </b></summary>

   * [📄 Document Scene Information Extraction v3 Production Line Usage Guide](/docs_new_en/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction_en.md)
  </details>

* <details>
    <summary> <b> 🔍 OCR </b></summary>

    * [📜 General OCR Production Line Usage Guide](/docs_new_en/pipeline_usage/tutorials/ocr_pipelies/OCR_en.md)
    * [📊 Form Recognition Production Line Usage Guide](/docs_new_en/pipeline_usage/tutorials/ocr_pipelies/table_recognition_en.md)
  </details>

* <details>
    <summary> <b> 🎥 Computer Vision </b></summary>

   * [🖼️ General Image Classification Production Line Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_classification_en.md)
   * [🎯 General Object Detection Production Line Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/object_detection_en.md)
   * [📋 General Instance Segmentation Production Line Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/instance_segmentation_en.md)
   * [🗣️ General Semantic Segmentation Production Line Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation_en.md)
   * [🏷️ Image Multi-Label Classification Production Line Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification_en.md)
   * [🔍 Small Object Detection Production Line Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/small_object_detection_en.md)
   * [🖼️ Image Anomaly Detection Production Line Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection_en.md)
  </details>
  
* <details>
    <summary> <b> ⏱️ Time Series Analysis</b> </summary>

   * [📈 General Time Series Forecasting Production Line Usage Guide](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting_en.md)
   * [📉 General Time Series Anomaly Detection Production Line Usage Guide](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection_en.md)
   * [🕒 General Time Series Classification Production Line Usage Guide](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_classification_en.md)
  </details>

* <details>
    <summary> <b>🔧 Related Documentation</b> </summary>

   * [🖥️ PaddleX Production Line Command Line Usage Guide](/docs_new_en/pipeline_usage/instructions/pipeline_CLI_usage_en.md)
   * [📝 PaddleX Production Line Python Script Usage Guide](/docs_new_en/pipeline_usage/instructions/pipeline_python_API_en.md)
  </details>
  
</details>

<details open>
<summary> <b> ⚙️ Single Function Module Usage </b></summary>

* <details>
  <summary> <b> 🔍 OCR </b></summary>

  * [📝 Text Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/ocr_modules/text_detection_en.md)
  * [🔖 Seal Text Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/ocr_modules/seal_text_detection_en.md)
  * [🔠 Text Recognition Module Usage Guide](/docs_new_en/module_usage/tutorials/ocr_modules/text_recognition_en.md)
  * [🗺️ Layout Area Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/ocr_modules/layout_detection_en.md)
  * [📊 Table Structure Recognition Module Usage Guide](/docs_new_en/module_usage/tutorials/ocr_modules/table_structure_recognition_en.md)
  * [📄 Document Image Orientation Classification Usage Guide](/docs_new_en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification_en.md)
  * [🔧 Document Image Correction Module Usage Guide](/docs_new_en/module_usage/tutorials/ocr_modules/text_image_unwarping_en.md)
  </details>

* <details>
  <summary> <b> 🖼️ Image Classification </b></summary>

  * [📂 Image Classification Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/image_classification_en.md)
  * [🏷️ Image Multi-Label Classification Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/ml_classification_en.md)

  * [👤 Pedestrian Attribute Recognition Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/pedestrian_attribute_recognition_en.md)
  * [🚗 Vehicle Attribute Recognition Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/vehicle_attribute_recognition_en.md)

  </details>

* <details>
  <summary> <b> 🏞️ Image Features </b></summary>

    * [🔗 General Image Feature Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules//image_feature_en.md)
  </details>

* <details>
  <summary> <b> 🎯 Object Detection </b></summary>

  * [🎯 Object Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/object_detection_en.md)
  * [📏 Small Object Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/small_object_detection_en.md)
  * [🧑‍🤝‍🧑 Face Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/face_detection_en.md)
  * [🔍 Main Body Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/mainbody_detection_en.md)
  * [🚶 Pedestrian Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/human_detection_en.md)
  * [🚗 Vehicle Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/vehicle_detection_en.md)

  </details>

* <details>
  <summary> <b> 🖼️ Image Segmentation </b></summary>

  * [🗺️ Semantic Segmentation Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/semantic_segmentation_en.md)
  * [🔍 Instance Segmentation Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/instance_segmentation_en.md)
  * [🚨 Image Anomaly Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/anomaly_detection_en.md)
  </details>

* <details>
  <summary> <b> ⏱️ Time Series Analysis </b></summary>

  * [📈 Time Series Forecasting Module Usage Guide](/docs_new_en/module_usage/tutorials/time_series_modules/time_series_forecasting_en.md)
  * [🚨 Time Series Anomaly Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/time_series_modules/time_series_anomaly_detection_en.md)
  * [🕒 Time Series Classification Module Usage Guide](/docs_new_en/module_usage/tutorials/time_series_modules/time_series_classification_en.md)
  </details>
    
* <details>
  <summary> <b> 📄 Related Documentation </b></summary>

  * [📝 PaddleX Single Model Python Script Usage Guide](/docs_new_en/module_usage/instructions/model_python_API_en.md)
  * [📝 PaddleX General Model Configuration File Parameter Guide](/docs_new_en/module_usage/instructions/config_parameters_common_en.md)
  * [📝 PaddleX Time Series Task Model Configuration File Parameter Guide](/docs_new_en/module_usage/instructions/config_parameters_time_series_en.md)
  </details>

</details>

<details>
  <summary> <b> 🔗 Multi-Module Combination Usage </b></summary>

  * [[🧩 Multi-Function Module Combination Usage Guide]()
</details>
<details>
  <summary> <b> 🏗️ Model Production Line Deployment </b></summary>

  * [🚀 PaddleX High-Performance Deployment Guide](/docs_new_en/pipeline_deploy/high_performance_deploy_en.md)
  * [🖥️ PaddleX Service Deployment Guide](/docs_new_en/pipeline_deploy/service_deploy_en.md)
  * [📱 PaddleX Edge Deployment Guide](/docs_new_en/pipeline_deploy/lite_deploy_en.md)

</details>
<details>
  <summary> <b> 🖥️ Multi-Hardware Usage </b></summary>

  * [⚙️ DCU Paddle Installation Guide](/docs_new_en/other_devices_support/installation_other_devices_en.md)
  * [⚙️ MLU Paddle Installation Guide](/docs_new_en/other_devices_support/installation_other_devices_en.md)
  * [⚙️ NPU Paddle Installation Guide](/docs_new_en/other_devices_support/installation_other_devices_en.md)
  * [⚙️ XPU Paddle Installation Guide](/docs_new_en/other_devices_support/installation_other_devices_en.md)

</details>

<details>
  <summary> <b> 📝 Tutorials & Examples </b></summary>

* [🖼️ General Image Classification Model Line —— Garbage Classification Tutorial](/docs_new_en/practical_tutorials/image_classification_garbage_tutorial_en.md)
* [🧩 General Instance Segmentation Model Line —— Remote Sensing Image Instance Segmentation Tutorial](/docs_new_en/practical_tutorials/image_classification_garbage_tutorial_en.md)
* [👥 General Object Detection Model Line —— Pedestrian Fall Detection Tutorial](/docs_new_en/practical_tutorials/object_detection_fall_tutorial_en.md)
* [👗 General Object Detection Model Line —— Fashion Element Detection Tutorial](/docs_new_en/practical_tutorials/object_detection_fashion_pedia_tutorial_en.md)
* [🚗 General OCR Model Line —— License Plate Recognition Tutorial](/docs_new_en/practical_tutorials/ocr_det_license_tutorial_en.md)
* [✍️ General OCR Model Line —— Handwritten Chinese Character Recognition Tutorial](/docs_new_en/practical_tutorials/ocr_rec_chinese_tutorial_en.md)
* [🗣️ General Semantic Segmentation Model Line —— Road Line Segmentation Tutorial](/docs_new_en/practical_tutorials/semantic_segmentation_road_tutorial_en.md)
* [🛠️ Time Series Anomaly Detection Model Line —— Equipment Anomaly Detection Application Tutorial](/docs_new_en/practical_tutorials/ts_anomaly_detection_en.md)
* [🎢 Time Series Classification Model Line —— Heartbeat Monitoring Time Series Data Classification Application Tutorial](/docs_new_en/practical_tutorials/ts_classification_en.md)
* [🔋 Time Series Forecasting Model Line —— Long-term Electricity Consumption Forecasting Application Tutorial](/docs_new_en/practical_tutorials/ts_forecast_en.md)

  </details>




## 🤔 FAQ

For answers to some common questions about our project, please refer to the [FAQ](/docs_new_en/FAQ_en.md). If your question has not been answered, please feel free to raise it in [Issues](https://github.com/PaddlePaddle/PaddleX/issues).

## 💬 Discussion

We warmly welcome and encourage community members to raise questions, share ideas, and feedback in the [Discussions](https://github.com/PaddlePaddle/PaddleX/discussions) section. Whether you want to report a bug, discuss a feature request, seek help, or just want to keep up with the latest project news, this is a great platform.

## 📄 License

The release of this project is licensed under the [Apache 2.0 license](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta/LICENSE).






