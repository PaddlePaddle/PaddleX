<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleX/assets/45199522/63c6d059-234f-4a27-955e-ac89d81409ee"  width="360" height ="55" alt="PaddleX" align="middle" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-red.svg"></a> 
    <a href=""><img src="https://img.shields.io/badge/python-3.8%2C%203.9%2C%203.10-blue.svg"></a> 
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20windows-orange.svg"></a> 
    <a href=""><img src="https://img.shields.io/badge/hardware-intel  cpu%2C%20gpu%2C%20xpu%2C%20npu%2C%20mlu-yellow.svg"></a>
</p>

<h4 align="center">
  <a href=#-features>🌟 Features</a> | <a href=https://aistudio.baidu.com/pipeline/mine>🌐  Online Experience</a>｜<a href=#️-quick-start>🚀  Quick Start</a> | <a href=#-documentation> 📖 Documentation</a> | <a href=#-model-pipeline-list List> 🔥Model Production Line List</a>
</h4>


<h5 align="center">
  <a href="README_en.md">🇨🇳 Simplified Chinese</a> | <a href="README_en.md">🇬🇧 English</a></a>
</h5>

## 🔍 Introduction

PaddleX 3.0 is a low-code development tool for AI models built on the PaddlePaddle framework. It integrates numerous **ready-to-use pre-trained models**, enabling **full-process development** from model training to inference, supporting **a variety of mainstream hardware** both domestic and international, and aiding AI developers in industrial practice.

|                                                            **Image Classification**                                                            |                                                            **Object Detection**                                                            |                                                            **Semantic Segmentation**                                                            |                                                            **Instance Segmentation**                                                            |
|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/b302cd7e-e027-4ea6-86d0-8a4dd6d61f39"  height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/099e2b00-0bbe-4b20-9c5a-96b69e473bd2"  height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/02637f8c-f248-415b-89ab-1276505f198c"  height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/09f683b4-27df-4c24-b8a7-84da20fdd182"  height="126px" width="180px"> |
|                                                              **Multi-label Image Classification**                                                               |                                                            **OCR**                                                            |                                                          **Table Recognition**                                                          |                                                          **PP-ChatOCR v3**                                                          |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1ef48536-48d4-484b-a6fb-0d6631ba2386"  height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1ef48536-48d4-484b-a6fb-0d6631ba2386"  height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1e798e05-dee7-4b41-9cc4-6708b6014efa"  height="126px" width="180px"> |  <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/e3d97f4e-ab46-411c-8155-494c61492b0a"  height="126px" width="180px"> |
|                                                              **Time Series Forecasting**                                                              |                                                            **Time Series Anomaly Detection**                                                            |                                                              **Time Series Classification**                                                              |                                                         **Image Anomaly Detection**                                                         |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/6e897bf6-35fe-45e6-a040-e9a1a20cfdf2"  height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/c54c66cc-da4f-4631-877b-43b0fbb192a6"  height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0ce925b2-3776-4dde-8ce0-5156d5a2476e"  height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0ce925b2-3776-4dde-8ce0-5156d5a2476e"  height="126px" width="180px"> |

## 🌟 Why PaddleX ?
  🎨 **Rich Models One-click Call**: Integrate over **200 PaddlePaddle models** covering multiple key areas such as OCR, object detection, and time series forecasting into **13 model pipelines**. Experience the model effects quickly through minimalist Python API calls. Also supports **more than 20 modules** for easy model combination use by developers.

  🚀 **High Efficiency and Low barrier of entry**: Achieve model **full-process development** based on graphical interfaces and unified commands, creating **8 featured model pipelines** that combine large and small models, semi-supervised learning of large models, and multi-model fusion, greatly reducing the cost of iterating models.

  🌐 **Flexible Deployment in Various Scenarios**: Support various deployment methods such as **high-performance deployment**, **service deployment**, and **lite deployment** to ensure efficient operation and rapid response of models in different application scenarios.

  🔧 **Efficient Support for Mainstream Hardware**: Support seamless switching of various mainstream hardware such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU to ensure efficient operation.

## 📣 Recent Updates

🔥🔥 **9.30, 2024**, PaddleX 3.0 Beta1 open source version is officially released, providing **more than 200 models** that can be called with a minimalist Python API; achieve model full-process development based on unified commands, and open source the basic capabilities of the **PP-ChatOCRv3** featured model production line; support **more than 100 models for high-performance inference and service-oriented deployment** (iterating continuously), **more than 7 key visual models for edge-side deployment**; **more than 70 models have been adapted for the full development process of Ascend 910B**, **more than 15 models have been adapted for the full development process of Kunlun chips and Cambricon**

🔥 **6.27, 2024**, PaddleX 3.0 Beta open source version is officially released, supporting the use of various mainstream hardware for production line and model development in a low-code manner on the local side.

🔥 **3.25, 2024**, PaddleX 3.0 cloud release, supporting the creation of pipelines in the AI Studio Galaxy Community in a zero-code manner.

## 📊 Capability Support

All pipelines of PaddleX support **online experience** and local **fast inference**. You can quickly experience the pre-trained effects of each production line. If you are satisfied with the pre-trained effects of the production line, you can directly perform [high-performance deployment](/docs_new_en/pipeline_deploy/high_performance_deploy_en.md) / [service deployment](/docs_new_en/pipeline_deploy/service_deploy_en.md) / [edge deployment](/docs_new_en/pipeline_deploy/lite_deploy_en.md) on the production line. If not satisfied, you can also **second development** to improve the production line effect. For the complete production line development process, please refer to the [PaddleX Production Line Development Tool Local Use Tutorial](/docs_new_en/pipeline_usage/pipeline_develop_guide_en.md).

In addition, PaddleX provides developers with a full-process efficient model training and deployment tool based on a [cloud-based graphical development interface](https://aistudio.baidu.com/pipeline/mine). Developers **do not need code development**, just need to prepare a dataset that meets the production line requirements to **quickly start model training**. For details, please refer to the tutorial ["Developing Industrial-level AI Models with Zero Threshold"](https://aistudio.baidu.com/practical/introduce/546656605663301).

<table >
    <tr>
        <td></td>
        <td>Online Experience</td>
        <td>Fast Inference</td>
        <td>High-Performance Deployment</td>
        <td>Service Deployment</td>
        <td>Lite Deployment</td>
        <td>Secondary Development</td>
        <td><a href = "https://aistudio.baidu.com/pipeline/mine">Galaxy Zero-Code Pipeline</a></td> 
    </tr>
    <tr>
        <td>OCR</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Classification</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Object Detection</td>
        <td><a href = "https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Semantic Segmentation</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Instance Segmentation</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Table Recognition</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91661?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Forecasting</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Anomaly Detection</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Classification</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Multi-label Classification</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Small Object Detection</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Anomaly Detection</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Document Scene Information Extraction</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
</table>

> ❗**Note: The above capabilities are implemented by PaddleX based on GPU/CPU. PaddleX also supports seamless switching among various mainstream hardware such as NVIDIA GPUs, Kunlun chips, Ascend, and Cambricon, but the functions supported by different chips vary. The following lists the capabilities supported by the other three types of hardware:**

<details>
  <summary>👉 NPU Capability Support</summary>

<table >
    <tr>
        <td></td>
        <td>Online Experience</td>
        <td>Fast Inference</td>
        <td>High-Performance Deployment</td>
        <td>Service Deployment</td>
        <td>Lite Deployment</td>
        <td>Secondary Development</td>
        <td><a href = "https://aistudio.baidu.com/pipeline/mine">Galaxy Zero-Code Pipeline</a></td> 
    </tr>
    <tr>
        <td>OCR</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Classification</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Object Detection</td>
        <td><a href = "https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Semantic Segmentation</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Instance Segmentation</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Table Recognition</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91661?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Forecasting</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Anomaly Detection</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Classification</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Multi-label Classification</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Small Object Detection</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Anomaly Detection</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Document Scene Information Extraction</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
</table>

</details>

<details>
  <summary>👉 MLU Capability Support</summary>

<table >
    <tr>
        <td></td>
        <td>Online Experience</td>
        <td>Fast Inference</td>
        <td>High-Performance Deployment</td>
        <td>Service Deployment</td>
        <td>Lite Deployment</td>
        <td>Secondary Development</td>
        <td><a href = "https://aistudio.baidu.com/pipeline/mine">Galaxy Zero-Code Pipeline</a></td> 
    </tr>
    <tr>
        <td>OCR</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Classification</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Object Detection</td>
        <td><a href = "https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Semantic Segmentation</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Instance Segmentation</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Table Recognition</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91661?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Forecasting</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Anomaly Detection</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Classification</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Multi-label Classification</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Small Object Detection</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Anomaly Detection</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Document Scene Information Extraction</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
</table>

</details>

<details>
  <summary>👉 DCU Capability Support</summary>

<table >
    <tr>
        <td></td>
        <td>Online Experience</td>
        <td>Fast Inference</td>
        <td>High-Performance Deployment</td>
        <td>Service Deployment</td>
        <td>Lite Deployment</td>
        <td>Secondary Development</td>
        <td><a href = "https://aistudio.baidu.com/pipeline/mine">Galaxy Zero-Code Pipeline</a></td> 
    </tr>
    <tr>
        <td>OCR</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Classification</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Object Detection</td>
        <td><a href = "https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Semantic Segmentation</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Instance Segmentation</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Table Recognition</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91661?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Forecasting</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Anomaly Detection</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Classification</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Multi-label Classification</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Small Object Detection</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Anomaly Detection</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Document Scene Information Extraction</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
</table>

</details>

<details>
  <summary>👉 XPU Capability Support</summary>

<table >
    <tr>
        <td></td>
        <td>Online Experience</td>
        <td>Fast Inference</td>
        <td>High-Performance Deployment</td>
        <td>Service Deployment</td>
        <td>Lite Deployment</td>
        <td>Secondary Development</td>
        <td><a href = "https://aistudio.baidu.com/pipeline/mine">Galaxy Zero-Code Pipeline</a></td> 
    </tr>
    <tr>
        <td>OCR</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Classification</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Object Detection</td>
        <td><a href = "https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Semantic Segmentation</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Instance Segmentation</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Table Recognition</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91661?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Forecasting</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Anomaly Detection</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Time Series Classification</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent">Link</a></td> 
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Multi-label Classification</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Small Object Detection</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Image Anomaly Detection</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>Document Scene Information Extraction</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
</table>

</details>

## ⏭️ Quick Start

### 🛠️ Installation

> ❗Before installing PaddleX, please ensure you have a basic Python runtime environment. If you have not yet installed the Python environment, you can refer to [Runtime Environment Preparation](/docs_new_en/installation/installation_en.md#1-Runtime-Environment-Preparation) for installation.

```bash
# Install PaddlePaddle
python -m pip install paddlepaddle # cpu
# gpu, this command is only applicable to machines with CUDA version 11.8
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ 
# gpu, this command is only applicable to machines with CUDA version 12.3
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/ 

# Install PaddleX
git clone https://github.com/PaddlePaddle/PaddleX.git 
cd PaddleX
pip install -e .
```

For more installation methods, refer to the [PaddleX Installation Guide](/docs_new_en/installation/installation_en.md)

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
  <summary>👉 Click to view the running result</summary>

```bash
The prediction result is:
['Boarding gate closes 10 minutes before departure']
The prediction result is:
['GATES CLOSE 1O MINUTES BEFORE DEPARTURE TIME']
The prediction result is:
['ETKT7813699238489/1']
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
| PP-ChatOCRv3   | `pp_chatocrv3` | [PP-ChatOCRv3 Pipeline CLI Usage Instructions](/docs_new_en/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction_en.md) |
|  Image Classification       | `image_classification` | [ Image Classification Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_classification_en.md) |
|  Object Detection       | `object_detection` | [ Object Detection Pipeline CLI Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_classification_en.md) |
|  Instance Segmentation       | `instance_segmentation` | [ Instance Segmentation Pipeline CLI Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/instance_segmentation_en.md) |
|  Semantic Segmentation       | `semantic_segmentation` | [ Semantic Segmentation Pipeline CLI Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation_en.md) |
|  Image Multi-Label Classification | `multilabel_classification` | [ Image Multi-Label Classification Pipeline CLI Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_multi_label_lassification_en.md) |
| Small Object Detection         | `smallobject_detection` | [Small Object Detection Pipeline CLI Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/small_object_detection_en.md) |
| Image Anomaly Detection       | `image_classification` | [Image Anomaly Detection Pipeline CLI Usage Instructions](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection_en.md) |
|  OCR            | `OCR` | [ OCR Pipeline CLI Usage Instructions](/docs_new_en/pipeline_usage/tutorials/ocr_pipelies/OCR_en.md) |
|  Form Recognition       | `table_recognition` | [ Form Recognition Pipeline CLI Usage Instructions](/docs_new_en/pipeline_usage/tutorials/ocr_pipelies/table_recognition_en.md) |
|  Time Series Forecast       | `ts_forecast` | [ Time Series Forecast Pipeline CLI Usage Instructions](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting_en.md) |
|  Time Series Anomaly Detection   | `ts_anomaly_detection` | [ Time Series Anomaly Detection Pipeline CLI Usage Instructions](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection_en.md) |
|  Time Series Classification       | `ts_classification` | [ Time Series Classification Pipeline CLI Usage Instructions](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_classification_en.md) |

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
  <summary> <b> Installation </b></summary>

  * [PaddleX Installation Guide](/docs_new_en/installation/installation_en.md) 
  * [PaddlePaddle Installation Guide](/docs_new_en/installation/paddlepaddle_install_en.md)

</details>

<details open>
<summary> <b> Pipeline Usage </b></summary>

* [PaddleX Model Pipeline Usage Overview](/docs_new_en/pipeline_usage/pipeline_develop_guide_en.md)

* <details open>
    <summary> <b> Text Image Intelligent Analysis </b></summary>

   * [Document Scene Information Extraction Pipeline Usage Guide](/docs_new_en/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction_en.md)
  </details>

* <details open>
    <summary> <b> Computer Vision </b></summary>

   * [ Image Classification Pipeline Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_classification_en.md)
   * [ Object Detection Pipeline Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/object_detection_en.md)
   * [ Instance Segmentation Pipeline Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/instance_segmentation_en.md)
   * [ Semantic Segmentation Pipeline Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation_en.md)
   * [ Image Multi-Label Classification Pipeline Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_multi_label_lassification_en.md)
   * [Small Object Detection Pipeline Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/small_object_detection_en.md)
   * [Image Anomaly Detection Pipeline Usage Guide](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection_en.md)
  </details>
  
* <details open>
    <summary> <b> OCR </b></summary>

    * [ OCR Pipeline Usage Guide](/docs_new_en/pipeline_usage/tutorials/ocr_pipelies/OCR_en.md)
    * [ Form Recognition Pipeline Usage Guide](/docs_new_en/pipeline_usage/tutorials/ocr_pipelies/table_recognition_en.md)
  </details>

* <details open>
    <summary> <b> Time Series Analysis</b> </summary>

   * [ Time Series Forecasting Pipeline Usage Guide](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting_en.md)
   * [ Time Series Anomaly Detection Pipeline Usage Guide](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection_en.md)
   * [ Time Series Classification Pipeline Usage Guide](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_classification_en.md)
  </details>

* <details>
    <summary> <b> Related Documentation </b></summary>

   * [PaddleX Pipeline CLI Usage Instructions](/docs_new_en/pipeline_usage/instructions/pipeline_CLI_usage_en.md)
   * [PaddleX Pipeline Python Script Usage Instructions](/docs_new_en/pipeline_usage/instructions/pipeline_python_API_en.md)
  </details>
   
</details>

<details open>
<summary> <b> Single Function Module Usage </b></summary>

* [PaddleX Single Function Module Usage Overview](/docs_new_en/pipeline_usage/pipeline_develop_guide_en.md)

* <details>
  <summary> <b> Computer Vision </b></summary>
  
  * [Image Classification Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/image_classification_en.md)
  * [Image Recognition Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/image_recognition_en.md)
  * [Object Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/object_detection_en.md)
  * [Small Object Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/small_object_detection_en.md)
  * [Face Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/face_detection_en.md)
  * [Mainbody Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/mainbody_detection_en.md)
  * [Pedestrian Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/human_detection_en.md)
  * [Vehicle Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/vehicle_detection_en.md)
  * [Semantic Segmentation Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/semantic_segmentation_en.md)
  * [Instance Segmentation Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/instance_segmentation_en.md)
  * [Document Image Orientation Classification Usage Guide](/docs_new_en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification_en.md)
  * [Image Multi-Label Classification Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/ml_classification_en.md)
  * [Pedestrian Attribute Recognition Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/pedestrian_attribute_recognition_en.md)
  * [Vehicle Attribute Recognition Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/vehicle_attribute_recognition_en.md)
  * [Image Correction Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/image_correction_en.md)
  * [Unsupervised Anomaly Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/cv_modules/unsupervised_anomaly_detection_en.md)
  </details> 
  
* <details>
  <summary> <b> OCR </b></summary>

  * [Text Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/ocr_modules/text_detection_en.md)
  * [Seal Text Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/ocr_modules/curved_text_detection_en.md)
  * [Text Recognition Module Usage Guide](/docs_new_en/module_usage/tutorials/ocr_modules/text_recognition_en.md)
  * [Layout Area Localization Module Usage Guide](/docs_new_en/module_usage/tutorials/ocr_modules/structure_analysis_en.md)
  * [Table Structure Recognition Module Usage Guide](/docs_new_en/module_usage/tutorials/ocr_modules/table_structure_recognition_en.md)
  </details>

* <details>
  <summary> <b> Time Series Analysis </b></summary>

  * [Time Series Forecasting Module Usage Guide](/docs_new_en/module_usage/tutorials/time_series_modules/time_series_forecasting_en.md)
  * [Time Series Anomaly Detection Module Usage Guide](/docs_new_en/module_usage/tutorials/time_series_modules/time_series_anomaly_detection_en.md)
  * [Time Series Classification Module Usage Guide](/docs_new_en/module_usage/tutorials/time_series_modules/time_series_classification_en.md)
  </details>
    
* <details>
  <summary> <b> Related Documentation </b></summary>

  * [PaddleX Single Model Python Script Usage Instructions](/docs_new_en/module_usage/instructions/model_python_API_en.md)
  * [PaddleX General Model Configuration File Parameter Explanation](/docs_new_en/module_usage/instructions/config_parameters_common_en.md)
  * [PaddleX Time Series Task Model Configuration File Parameter Explanation](/docs_new_en/module_usage/instructions/config_parameters_time_series_en.md)
  </details>

</details>

<details>
  <summary> <b> Multi-Module Combination Usage </b></summary>

  * [Multi-Function Module Combination Usage Guide]()
</details>
<details>
  <summary> <b> Model Pipeline Deployment </b></summary>

  * [PaddleX High-Performance Deployment Guide](/docs_new_en/pipeline_deploy/high_performance_deploy_en.md)
  * [PaddleX Service Deployment Guide](/docs_new_en/pipeline_deploy/service_deploy_en.md)
  * [PaddleX Edge Deployment Guide](/docs_new_en/pipeline_deploy/lite_deploy_en.md)

</details>
<details>
  <summary> <b> Multi-Hardware Usage </b></summary>

  * [Multi-Hardware Usage Guide](/docs_new_en/other_devices_support/installation_other_devices_en.md)
</details>


## 🤔 FAQ

For answers to some common questions about our project, please refer to the [FAQ](/docs_new_en/FAQ_en.md). If your question has not been answered, please feel free to raise it in [Issues](https://github.com/PaddlePaddle/PaddleX/issues).

## 💬 Discussion

We warmly welcome and encourage community members to raise questions, share ideas, and feedback in the [Discussions](https://github.com/PaddlePaddle/PaddleX/discussions) section. Whether you want to report a bug, discuss a feature request, seek help, or just want to keep up with the latest project news, this is a great platform.

## 🔥 Model Pipeline List

<details>
  <summary><b> Document Scene Information Extraction Pipeline </b></summary>
</details>

<details>
  <summary> <b>  OCR Pipeline </b></summary>

| Task Module | Model            | Accuracy | GPU Inference Time (ms) | CPU Inference Time (ms) | Model Storage Size (MB) |
|-------------|---------------------|---------|------------------------|-----------------------|------------------------|
| Text Detection | PP-OCRv4_mobile_det | 77.79 | 2.719474               | 79.1097                | 15                     |
|              | PP-OCRv4_server_det  | 82.69  | 22.20346               | 2662.158               | 198                    |
| Text Recognition | PP-OCRv4_mobile_rec | 78.20 | 2.719474               | 79.1097                | 15                     |
|               | PP-OCRv4_server_rec  | 79.20  | 22.20346               | 2662.158               | 198                    |

**Note: The accuracy metric for text detection models is Hmean(%), and for text recognition models, it is Accuracy(%).**

</details>

<details>
  <summary><b>  Form Recognition Pipeline </b> </summary>
</details>

<details>
  <summary> <b>  Image Classification Pipeline </b></summary>

| Task Module | Model            | Accuracy | GPU Inference Time (ms) | CPU Inference Time (ms) | Model Storage Size (MB) |
|-------------|---------------------|---------|------------------------|-----------------------|------------------------|
| Text Detection | PP-OCRv4_mobile_det | 77.79 | 2.719474               | 79.1097                | 15                     |
|              | PP-OCRv4_server_det  | 82.69  | 22.20346               | 2662.158               | 198                    |
| Text Recognition | PP-OCRv4_mobile_rec | 78.20 | 2.719474               | 79.1097                | 15                     |
|               | PP-OCRv4_server_rec  | 79.20  | 22.20346               | 2662.158               | 198                    |

**Note: The accuracy metric for text detection models is Hmean(%), and for text recognition models, it is Accuracy(%).**

</details>

<details>
  <summary> <b>  Object Detection Pipeline </b></summary>
</details>

<details>
  <summary><b>  Instance Segmentation Pipeline </b></summary>
</details>

<details>
  <summary> <b>  Semantic Segmentation Pipeline </b> </summary>
</details>

<details>
  <summary> <b>  Image Multi-Label Classification Pipeline </b> </summary>
</details>

<details>
  <summary><b> Small Object Detection Classification Pipeline </b> </summary>
</details>

<details>
  <summary> <b> Image Anomaly Detection Pipeline </b> </summary>
</details>

<details>
  <summary><b>  Time Series Forecasting Pipeline </b> </summary>
</details>

<details>
  <summary><b>  Time Series Anomaly Detection Pipeline </b> </summary>
</details>

<details>
  <summary><b>  Time Series Classification Pipeline </b> </summary>
</details>

## 📄 License

The release of this project is licensed under the [Apache 2.0 license](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta/LICENSE).






