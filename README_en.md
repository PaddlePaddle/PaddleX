<p align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/logo.png" width="735" height ="200" alt="PaddleX" align="middle" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202-red.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python-3.8%2C%203.9%2C%203.10-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Windows%2C%20Mac-orange.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/hardware-CPU%2C%20GPU%2C%20XPU%2C%20NPU%2C%20MLU%2C%20DCU-yellow.svg"></a>
</p>

<h4 align="center">
  <a href=#-why-paddlex->🌟 Features</a> | <a href=https://aistudio.baidu.com/pipeline/mine>🌐  Online Experience</a>｜<a href=#️-quick-start>🚀  Quick Start</a> | <a href=https://addlepaddle.github.io/PaddleX/latest/en/index.html> 📖 Documentation</a> | <a href=#-what-can-paddlex-do> 🔥Capabilities</a> | <a href=https://paddlepaddle.github.io/PaddleX/latest/en/support_list/models_list.html> 📋 Models</a>
</h4>

<h5 align="center">
  <a href="README.md">🇨🇳 Simplified Chinese</a> | <a href="README_en.md">🇬🇧 English</a></a>
</h5>

## 🔍 Introduction

PaddleX 3.0 is a low-code development tool for AI models built on the PaddlePaddle framework. It integrates numerous **ready-to-use pre-trained models**, enabling **full-process development** from model training to inference, supporting **a variety of mainstream hardware** both domestic and international, and aiding AI developers in industrial practice.


|                                                            [**Image Classification**](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_classification.html)                                                            |                                                            [**Multi-label Image Classification**](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.html)                                                            |                                                            [**Object Detection**](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/object_detection.html)                                                            |                                                            [**Instance Segmentation**](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.html)                                                            |
|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/b302cd7e-e027-4ea6-86d0-8a4dd6d61f39" height="126px" width="180px"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/multilabel_cls.png" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/099e2b00-0bbe-4b20-9c5a-96b69e473bd2" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/09f683b4-27df-4c24-b8a7-84da20fdd182" height="126px" width="180px"> |
|                                                              [**Semantic Segmentation**](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.html)                                                               |                                                            [**Image Anomaly Detection**](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.html)                                                            |                                                          [**OCR**](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html)                                                          |                                                          [**Table Recognition**](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html)                                                          |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/02637f8c-f248-415b-89ab-1276505f198c" height="126px" width="180px"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/image_anomaly_detection.png" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1ef48536-48d4-484b-a6fb-0d6631ba2386" height="126px" width="180px"> |  <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1e798e05-dee7-4b41-9cc4-6708b6014efa" height="126px" width="180px"> |
|                                                              [**PP-ChatOCRv3-doc**](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.html)                                                              |                                                            [**Time Series Forecasting**](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.html)                                                            |                                                              [**Time Series Anomaly Detection**](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.html)                                                              |                                                         [**Time Series Classification**](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.html)                                                         |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/e3d97f4e-ab46-411c-8155-494c61492b0a" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/6e897bf6-35fe-45e6-a040-e9a1a20cfdf2" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/c54c66cc-da4f-4631-877b-43b0fbb192a6" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0ce925b2-3776-4dde-8ce0-5156d5a2476e" height="126px" width="180px"> |



## 🌟 Why PaddleX ?

  🎨 **Rich Models One-click Call**: Integrate over **200 PaddlePaddle models** covering multiple key areas such as OCR, object detection, and time series forecasting into **19 pipelines**. Experience the model effects quickly through easy Python API calls. Also supports **more than 20 modules** for easy model combination use by developers.

  🚀 **High Efficiency and Low barrier of entry**: Achieve model **full-process development** based on graphical interfaces and unified commands, creating **8 featured model pipelines** that combine large and small models, semi-supervised learning of large models, and multi-model fusion, greatly reducing the cost of iterating models.

  🌐 **Flexible Deployment in Various Scenarios**: Support various deployment methods such as **high-performance inference**, **serving**, and **lite deployment** to ensure efficient operation and rapid response of models in different application scenarios.

  🔧 **Efficient Support for Mainstream Hardware**: Support seamless switching of various mainstream hardware such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU to ensure efficient operation.

## 📣 Recent Updates

🔥🔥 **"PaddleX Document Information Personalized Extraction Upgrade"**, PP-ChatOCRv3 innovatively provides custom development functions for OCR models based on data fusion technology, offering stronger model fine-tuning capabilities. Millions of high-quality general OCR text recognition data are automatically integrated into vertical model training data at a specific ratio, solving the problem of weakened general text recognition capabilities caused by vertical model training in the industry. Suitable for practical scenarios in industries such as automated office, financial risk control, healthcare, education and publishing, and legal and government sectors. **October 24th (Thursday) 19:00** Join our live session for an in-depth analysis of the open-source version of PP-ChatOCRv3 and the outstanding advantages of PaddleX 3.0 Beta1 in terms of accuracy and speed. [Registration Link](https://www.wjx.top/vm/wpPu8HL.aspx?udsid=994465)

> [❗ Get more courses for free](https://aistudio.baidu.com/education/group/info/32160)

🔥🔥 **11.15, 2024**, PaddleX 3.0 Beta2 open source version is officially released, PaddleX 3.0 Beta2 is fully compatible with the PaddlePaddle 3.0b2 version. <b>This update introduces new pipelines for general image recognition, face recognition, vehicle attribute recognition, and pedestrian attribute recognition. We have also developed 42 new models to fully support the Ascend 910B, with extensive documentation available on [GitHub Pages](https://paddlepaddle.github.io/PaddleX/latest/en/index.html).</b>

🔥🔥 **9.30, 2024**, PaddleX 3.0 Beta1 open source version is officially released, providing **more than 200 models** that can be called with a simple Python API; achieve model full-process development based on unified commands, and open source the basic capabilities of the **PP-ChatOCRv3** pipeline; support **more than 100 models for high-performance inference and serving** (iterating continuously), **more than 7 key visual models for edge-deployment**; **more than 70 models have been adapted for the full development process of Ascend 910B**, **more than 15 models have been adapted for the full development process of Kunlun chips and Cambricon**

🔥 **6.27, 2024**, PaddleX 3.0 Beta open source version is officially released, supporting the use of various mainstream hardware for pipeline and model development in a low-code manner on the local side.

🔥 **3.25, 2024**, PaddleX 3.0 cloud release, supporting the creation of pipelines in the AI Studio Galaxy Community in a zero-code manner.

## 🔠 Explanation of Pipeline
PaddleX is dedicated to achieving pipeline-level model training, inference, and deployment. A pipeline refers to a series of predefined development processes for specific AI tasks, which includes a combination of single models (single-function modules) capable of independently completing a certain type of task.

## 📊 What can PaddleX do？


All pipelines of PaddleX support **online experience** on [AI Studio]((https://aistudio.baidu.com/overview)) and local **fast inference**. You can quickly experience the effects of each pre-trained pipeline. If you are satisfied with the effects of the pre-trained pipeline, you can directly perform [high-performance inference](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/high_performance_inference.html) / [serving](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/serving.html) / [edge deployment](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/edge_deploy.html) on the pipeline. If not satisfied, you can also **Custom Development** to improve the pipeline effect. For the complete pipeline development process, please refer to the [PaddleX pipeline Development Tool Local Use Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/pipeline_develop_guide.html).

In addition, PaddleX provides developers with a full-process efficient model training and deployment tool based on a [cloud-based GUI](https://aistudio.baidu.com/pipeline/mine). Developers **do not need code development**, just need to prepare a dataset that meets the pipeline requirements to **quickly start model training**. For details, please refer to the tutorial ["Developing Industrial-level AI Models with Zero Barrier"](https://aistudio.baidu.com/practical/introduce/546656605663301).

<table>
    <tr>
        <th>Pipeline</th>
        <th>Online Experience</th>
        <th>Local Inference</th>
        <th>High-Performance Inference</th>
        <th>Serving</th>
        <th>Edge Deployment</th>
        <th>Custom Development</th>
        <th><a href="https://aistudio.baidu.com/pipeline/mine">Zero-Code Development On AI Studio</a></td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html">OCR</a></td>
        <td><a href="https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">Link</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.html">PP-ChatOCRv3</a></td>
        <td><a href="https://aistudio.baidu.com/community/app/182491/webUI?source=appCenter">Link</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html">Table Recognition</a></td>
        <td><a href="https://aistudio.baidu.com/community/app/91661?source=appMineRecent">Link</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/object_detection.html">Object Detection</a></td>
        <td><a href="https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">Link</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.html">Instance Segmentation</a></td>
        <td><a href="https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">Link</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_classification.html">Image Classification</a></td>
        <td><a href="https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">Link</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.html">Semantic Segmentation</a></td>
        <td><a href="https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">Link</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.html">Time Series Forecasting</a></td>
        <td><a href="https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent">Link</a></td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.html">Time Series Anomaly Detection</a></td>
        <td><a href="https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent">Link</a></td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.html">Time Series Classification</a></td>
        <td><a href="https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent">Link</a></td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
        <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/small_object_detection.html">Small Object Detection</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
        <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.html">Multi-label Image Classification</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.html">Image Anomaly Detection</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/human_keypoint_detection.html">Human Keypoint Detection</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/open_vocabulary_detection.html">Open Vocabulary Detection</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/open_vocabulary_segmentation.html">Open Vocabulary Segmentation</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/rotated_object_detection.html">Rotated Object Detection</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/3d_bev_detection.html">3D Bev Detection</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.html">Table Recognition v2</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/layout_parsing.html">Layout Parsing</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/layout_parsing_v2.html">Layout Parsing v2</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/formula_recognition.html">Formula Recognition</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/seal_recognition.html">Seal Recognition</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/doc_preprocessor.html">Document Image Preprocessing</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/general_image_recognition.html>Image Recognition</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/pedestrian_attribute.html">Pedestrian Attribute Recognition</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/vehicle_attribute.html">Vehicle Attribute Recognition</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/face_recognition.html">Face Recognition</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/speech_pipelines/multilingual_speech_recognition.html">Multilingual Speech Recognition</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/video_pipelines/video_classification.html">Video Classification</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/video_pipelines/video_detection.html">Video Detection</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>

</table>

> ❗Note: The above capabilities are implemented based on GPU/CPU. PaddleX can also perform local inference and custom development on mainstream hardware such as Kunlunxin, Ascend, Cambricon, and Haiguang. The table below details the support status of the pipelines. For specific supported model lists, please refer to the [Model List (Kunlunxin XPU)](https://paddlepaddle.github.io/PaddleX/latest/en/support_list/model_list_xpu.html)/[Model List (Ascend NPU)](https://paddlepaddle.github.io/PaddleX/latest/en/support_list/model_list_npu.html)/[Model List (Cambricon MLU)](https://paddlepaddle.github.io/PaddleX/latest/en/support_list/model_list_mlu.html)/[Model List (Haiguang DCU)](https://paddlepaddle.github.io/PaddleX/latest/en/support_list/model_list_dcu.html). We are continuously adapting more models and promoting the implementation of high-performance and serving on mainstream hardware.

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

> ❗Before installing PaddleX, please ensure you have a basic **Python environment** (Note: Currently supports Python 3.8 to Python 3.10, with more Python versions being adapted). The PaddleX 3.0-beta2 version depends on PaddlePaddle version 3.0.0b2.

* **Installing PaddlePaddle**

```bash
# cpu
python -m pip install paddlepaddle==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# gpu，该命令仅适用于 CUDA 版本为 11.8 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，该命令仅适用于 CUDA 版本为 12.3 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```
> ❗For more PaddlePaddle versions, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation./docs/zh/install/pip/linux-pip.html).

* **Installing PaddleX**

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0rc0-py3-none-any.whl
```

> ❗For more installation methods, refer to the [PaddleX Installation Guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/installation.html).


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
| Image Anomaly Detection       | `paddlex --pipeline anomaly_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png --device gpu:0`                                              |
| Pedestrian Attribute Recognition       | `paddlex --pipeline pedestrian_attribute --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_002.jpg --device gpu:0`                                              |
| Vehicle Attribute Recognition       | `paddlex --pipeline vehicle_attribute --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_attribute_002.jpg --device gpu:0`                                              |
| OCR                  | `paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0`                                                      |
| Table Recognition    | `paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --device gpu:0`                                      |
| Layout Parsing       | `paddlex --pipeline layout_parsing --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png --device gpu:0`                                      |
| Formula Recognition       | `paddlex --pipeline formula_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png --device gpu:0`                                      |
| Seal Recognition       | `paddlex --pipeline seal_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png --device gpu:0`                                      |
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
| PP-ChatOCRv3-doc   | `PP-ChatOCRv3-doc` | [PP-ChatOCRv3-doc Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.html) |
|  Image Classification       | `image_classification` | [ Image Classification Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_classification.html) |
|  Object Detection       | `object_detection` | [ Object Detection Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/object_detection.html) |
|  Instance Segmentation       | `instance_segmentation` | [ Instance Segmentation Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.html) |
|  Semantic Segmentation       | `semantic_segmentation` | [ Semantic Segmentation Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.html) |
|  Image Multi-Label Classification | `multilabel_classification` | [ Image Multi-Label Classification Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.html) |
| Small Object Detection         | `small_object_detection` | [Small Object Detection Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/small_object_detection.html) |
| Image Anomaly Detection       | `image_classification` | [Image Anomaly Detection Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.html) |
| Image Recognition       | `PP-ShiTuV2`                | [Image Recognition Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/general_image_recognition.html)                              |
| Face Recognition       | `face_recognition`                | [Face Recognition Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/face_recognition.html)                              |
| Pedestrian Attribute Recognition       | `pedestrian_attribute`                | [Pedestrian Attribute Recognition Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/pedestrian_attribute.html)                              |
|Vehicle Attribute Recognition       | `vehicle_attribute`                | [Vehicle Attribute Recognition Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/vehicle_attribute.html)                              |
|  OCR            | `OCR` | [ OCR Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html) |
|  Table Recognition       | `table_recognition` | [Table Recognition Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html) |
| Layout Parsing       | `layout_parsing`                | [Layout Parsing Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/layout_parsing.html)                                   |
| Formula Recognition       | `formula_recognition`                | [Formula Recognition Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/formula_recognition.html)                                   |
| Seal Recognition       | `seal_recognition`                | [Seal Recognition Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/seal_recognition.html)                 |
|  Time Series Forecast       | `ts_forecast` | [ Time Series Forecast Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.html) |
|  Time Series Anomaly Detection   | `ts_anomaly_detection` | [ Time Series Anomaly Detection Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.html) |
|  Time Series Classification       | `ts_cls` | [ Time Series Classification Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.html) |
</details>

## 📖 Documentation
<details>
  <summary> <b> ⬇️ Installation </b></summary>

  * [📦 PaddlePaddle Installation](https://paddlepaddle.github.io/PaddleX/latest/en/installation/paddlepaddle_install.html)
  * [📦 PaddleX Installation](https://paddlepaddle.github.io/PaddleX/latest/en/installation/installation.html)

</details>

<details open>
<summary> <b> 🔥 Pipeline Usage </b></summary>

* [📑 PaddleX Pipeline Usage Overview](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/pipeline_develop_guide.html)

* <details open>
    <summary> <b> 📝 Information Extracion</b></summary>

   * [📄 PP-ChatOCRv3 Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.html)
  </details>

* <details open>
    <summary> <b> 🔍 OCR </b></summary>

    * [📜 OCR Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html)
    * [📊 Table Recognition Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html)
    * [🗂️ Table Recognition v2 Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.html)
    * [📄 Layout Parsing Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/layout_parsing.html)
    * [🗞️ Layout Parsing v2 Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/layout_parsing_v2.html)
    * [📐 Formula Recognition Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/formula_recognition.html)
    * [📝 Seal Recognition Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/seal_recognition.html)
    * [🖌️ Document Image Preprocessing](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/doc_preprocessor.html)
  </details>

* <details open>
    <summary> <b> 🎥 Computer Vision </b></summary>

   * [🖼️ Image Classification Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_classification.html)
   * [🎯 Object Detection Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/object_detection.html)
   * [📋 Instance Segmentation Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.html)
   * [🗣️ Semantic Segmentation Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.html)
   * [🏷️ Multi-label Image Classification Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.html)
   * [🔍 Small Object Detection Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/small_object_detection.html)
   * [🖼️ Image Anomaly Detection Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.html)
   * [🔍 Human Keypoint Detection Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/human_keypoint_detection.html)
   * [📚 Open Vocabulary Detection Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/open_vocabulary_detection.html)
   * [🎨 Open Vocabulary Segmentation Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/open_vocabulary_segmentation.html)
   * [🔄 Rotated Object Detection Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/rotated_object_detection.html)
   * [🌐 3D Bev Detection Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/3d_bev_detection.html)
   * [🖼️ Image Recognition Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/general_image_recognition.html)
   * [🆔 Face Recognition Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/face_recognition.html)
   * [🚗 Vehicle Attribute Recognition Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/vehicle_attribute.html)
   * [🚶‍♀️ Pedestrian Attribute Recognition Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/cv_pipelines/pedestrian_attribute.html)
  </details>

* <details open>
    <summary> <b> ⏱️ Time Series Analysis</b> </summary>

   * [📈 Time Series Forecasting Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.html)
   * [📉 Time Series Anomaly Detection Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.html)
   * [🕒 Time Series Classification Pipeline Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.html)
  </details>

* <details open>
    <summary> <b> 🎤 Speech Analysis</b> </summary>

    * [🌐 Multilingual Speech Recognition Pipeline Usage Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/speech_pipelines/multilingual_speech_recognition.html)

* <details open>
    <summary> <b> 🎥 Video Processing</b> </summary>

    * [📈 General Video Classification Pipeline Usage Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/video_pipelines/video_classification.html)
    * [🔍 General Video Detection Pipeline Usage Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/video_pipelines/video_detection.html)

* <details open>
    <summary> <b>🔧 Related Instructions</b> </summary>

   * [🖥️ PaddleX pipeline Command Line Instruction](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/instructions/pipeline_CLI_usage.html)
   * [📝 PaddleX pipeline Python Script Instruction](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/instructions/pipeline_python_API.html)
  </details>

</details>

<details open>
<summary> <b> ⚙️ Module Usage </b></summary>

* <details open>
  <summary> <b> 🔍 OCR </b></summary>

  * [📝 Text Detection Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_detection.html)
  * [🔖 Seal Text Detection Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/seal_text_detection.html)
  * [🔠 Text Recognition Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_recognition.html)
  * [🗺️ Layout Parsing Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html)
  * [📊 Table Structure Recognition Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_structure_recognition.html)
  * [📄 Document Image Orientation Classification Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html)
  * [🔧 Document Image Unwarp Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_image_unwarping.html)
  * [📐 Formula Recognition Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/formula_recognition.html)
  * [📊 Table Cell Detection Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_cells_detection.html)
  * [📈 Table Classification Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_classification.html)
  * [📝 Text Line Orientation Classification Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html)
  </details>

* <details open>
  <summary> <b> 🖼️ Image Classification </b></summary>

  * [📂 Image Classification Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/image_classification.html)
  * [🏷️ Multi-label Image Classification Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/image_multilabel_classification.html)

  * [👤 Pedestrian Attribute Recognition Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.html)
  * [🚗 Vehicle Attribute Recognition Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/vehicle_attribute_recognition.html)

  </details>

* <details open>
  <summary> <b> 🏞️ Image Features </b></summary>

    * [🔗 Image Feature Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/image_feature.html)
    * [😁 Face_Feature Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/face_feature.html)
  </details>

* <details open>
  <summary> <b> 🎯 Object Detection </b></summary>

  * [🎯 Object Detection Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/object_detection.html)
  * [📏 Small Object Detection Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/small_object_detection.html)
  * [🧑‍🤝‍🧑 Face Detection Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/face_detection.html)
  * [🔍 Mainbody Detection Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/mainbody_detection.html)
  * [🚶 Pedestrian Detection Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/human_detection.html)
  * [🚗 Vehicle Detection Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/vehicle_detection.html)
  * [🚶‍♂️ Human Keypoint Detection Module Usage Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/human_keypoint_detection.html)
  * [🌐 Open-Vocabulary Object Detection Module Usage Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/open_vocabulary_detection.html)
  * [🔄 Rotated Object Detection Module Usage Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/rotated_object_detection.html)

  </details>

* <details open>
  <summary> <b> 🖼️ Image Segmentation </b></summary>

  * [🗺️ Semantic Segmentation Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/semantic_segmentation.html)
  * [🔍 Instance Segmentation Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/instance_segmentation.html)
  * [🚨 Image Anomaly Detection Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/anomaly_detection.html)
  * [🌐 Open-Vocabulary Segmentation Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/open_vocabulary_segmentation.html)
  </details>

* <details open>
  <summary> <b> ⏱️ Time Series Analysis </b></summary>

  * [📈 Time Series Forecasting Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/time_series_modules/time_series_forecasting.html)
  * [🚨 Time Series Anomaly Detection Module Tutorial](./docs/module_usage/tutorials/time_series_modules/time_series_anomaly_detection.md)
  * [🕒 Time Series Classification Module Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/time_series_modules/time_series_classification.html)
  </details>

* <details open>
  <summary> <b> 🌐 3D </b></summary>
  
  * [🚗 3D Multimodal Fusion Detection Module Usage Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/cv_modules/3d_bev_detection.html)

* <details open>
  <summary> <b> 🎤 Speech </b></summary>

  * [🌐 Multilingual Speech Recognition Module Usage Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/speech_modules/multilingual_speech_recognition.html)

* <details open>
  <summary> <b> 🎥 Video </b></summary>

  * [📈 Video Classification Module Usage Tutorial](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/video_modules/video_classification.html)
  * [🔍 Video Detection Module Usage Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/video_modules/video_detection.html)

* <details open>
  <summary> <b> 📄 Related Instructions </b></summary>

  * [📝 PaddleX Single Model Python Script Instruction](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/instructions/model_python_API.html)
  * [📝 PaddleX General Model Configuration File Parameter Instruction](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/instructions/config_parameters_common.html)
  * [📝 PaddleX Time Series Task Model Configuration File Parameter Instruction](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/instructions/config_parameters_time_series.html)
  </details>

</details>

<details open>
  <summary> <b> 🏗️ Pipeline Deployment </b></summary>

  * [🚀 PaddleX High-Performance Inference Guide](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/high_performance_inference.html)
  * [🖥️ PaddleX Serving Guide](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/serving.html)
  * [📱 PaddleX Edge Deployment Guide](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/edge_deploy.html)

</details>
<details open>
  <summary> <b> 🖥️ Multi-Hardware Usage </b></summary>

  * [⚙️ Multi-Hardware Usage Guide](https://paddlepaddle.github.io/PaddleX/latest/en/other_devices_support/multi_devices_use_guide.html)
  * [⚙️ DCU Paddle Installation](https://paddlepaddle.github.io/PaddleX/latest/en/other_devices_support/paddlepaddle_install_DCU.html)
  * [⚙️ MLU Paddle Installation](https://paddlepaddle.github.io/PaddleX/latest/en/other_devices_support/paddlepaddle_install_MLU.html)
  * [⚙️ NPU Paddle Installation](https://paddlepaddle.github.io/PaddleX/latest/en/other_devices_support/paddlepaddle_install_NPU.html)
  * [⚙️ XPU Paddle Installation](https://paddlepaddle.github.io/PaddleX/latest/en/other_devices_support/paddlepaddle_install_XPU.html)

</details>

<details>
  <summary> <b> 📝 Tutorials & Examples </b></summary>

* [📑 PP-ChatOCRv3 Model Line —— Paper Document Information Extract Tutorial](./docs/practical_tutorials/document_scene_information_extraction(layout_detection)_tutorial_en.md)
* [📑 PP-ChatOCRv3 Model Line —— Seal Information Extract Tutorial](./docs/practical_tutorials/document_scene_information_extraction(seal_recognition)_tutorial_en.md)
* [🖼️ General Image Classification Model Line —— Garbage Classification Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/practical_tutorials/image_classification_garbage_tutorial.html)
* [🧩 General Instance Segmentation Model Line —— Remote Sensing Image Instance Segmentation Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/practical_tutorials/instance_segmentation_remote_sensing_tutorial.html)
* [👥 General Object Detection Model Line —— Pedestrian Fall Detection Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/practical_tutorials/object_detection_fall_tutorial.html)
* [👗 General Object Detection Model Line —— Fashion Element Detection Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/practical_tutorials/object_detection_fashion_pedia_tutorial.html)
* [🚗 General OCR Model Line —— License Plate Recognition Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/practical_tutorials/ocr_det_license_tutorial.html)
* [✍️ General OCR Model Line —— Handwritten Chinese Character Recognition Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/practical_tutorials/ocr_rec_chinese_tutorial.html)
* [🗣️ General Semantic Segmentation Model Line —— Road Line Segmentation Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/practical_tutorials/semantic_segmentation_road_tutorial.html)
* [🛠️ Time Series Anomaly Detection Model Line —— Equipment Anomaly Detection Application Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/practical_tutorials/ts_anomaly_detection.html)
* [🎢 Time Series Classification Model Line —— Heartbeat Monitoring Time Series Data Classification Application Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/practical_tutorials/ts_classification.html)
* [🔋 Time Series Forecasting Model Line —— Long-term Electricity Consumption Forecasting Application Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/practical_tutorials/ts_forecast.html)

  </details>




## 🤔 FAQ

For answers to some common questions about our project, please refer to the [FAQ](https://paddlepaddle.github.io/PaddleX/latest/en/FAQ.html). If your question has not been answered, please feel free to raise it in [Issues](https://github.com/PaddlePaddle/PaddleX/issues).

## 💬 Discussion

We warmly welcome and encourage community members to raise questions, share ideas, and feedback in the [Discussions](https://github.com/PaddlePaddle/PaddleX/discussions) section. Whether you want to report a bug, discuss a feature request, seek help, or just want to keep up with the latest project news, this is a great platform.

## 📄 License

The release of this project is licensed under the [Apache 2.0 license](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta/LICENSE).
