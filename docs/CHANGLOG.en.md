---
comments: true
---

# Version Update Information

## Latest Version Information
### PaddleX v3.0.0rc0(2.14/2025)

PaddleX 3.0 rc0 is fully compatible with PaddlePaddle 3.0rc0 version, adding 10+ pipelines, 40+ models, optimizing model and pipeline APIs, and adapting more models to multiple hardware. The high-performance inference and serving capabilities have been comprehensively upgraded. The specific new features are as follows:

- <b>New pipelines</b>:
  - <b>[Document Image Preprocessing Pipeline](pipeline_usage/tutorials/ocr_pipelines/doc_preprocessor.en.md)</b>, supporting the correction of rotated and distorted document images.
  - <b>[PP-ChatOCRv4-doc Pipeline](pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v4.en.md)</b>, which integrates multimodal capabilities on the basis of the Document PP-ChatOCRv3-doc pipeline, enhances OCR recognition capabilities, optimizes Prompts, and ultimately improves the accuracy of document information extraction by 15 percentage points. Supports local large model OpenAI interface calls.
  - <b>[Layout Parsing v2 Pipeline](pipeline_usage/tutorials/ocr_pipelines/layout_parsing_v2.en.md)</b>, the core solution of PP-StructureV3. Based on the General Layout Parsing v1 pipeline, it optimizes layout area detection, table recognition, formula recognition, and reading order recovery capabilities, supports converting different types of document images and document PDF files into standard Markdown files, and performs strongly in document recovery capabilities in most scenarios.
  - <b>[Table Recognition v2 Pipeline](pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.en.md)</b>, adopting a multi-model series networking solution of "table classification + table structure recognition + cell detection" to achieve higher precision end-to-end table recognition.
  - <b>[Rotated Object Detection Pipeline](pipeline_usage/tutorials/cv_pipelines/rotated_object_detection.en.md)</b>, supporting the detection of rotated objects.
  - <b>[Human Keypoint Detection Pipeline](pipeline_usage/tutorials/cv_pipelines/human_keypoint_detection.en.md)</b>, supporting precise acquisition of human keypoint positions such as shoulders, elbows, knees, etc., for pose estimation and behavior recognition.
  - <b>[Open Vocabulary Object Detection Pipeline](pipeline_usage/tutorials/cv_pipelines/open_vocabulary_detection.en.md)</b>, supporting the detection of open-domain objects and predicting categories.
  - <b>[Open Vocabulary Segmentation Pipeline](pipeline_usage/tutorials/cv_pipelines/open_vocabulary_segmentation.en.md)</b>, supporting image segmentation of open-domain objects.
  - <b>[Video Detection Pipeline](pipeline_usage/tutorials/video_pipelines/video_detection.en.md)</b>, supporting efficient extraction of spatial and temporal features from videos, achieving accurate recognition and localization of objects in videos.
  - <b>[Video Classification Pipeline](pipeline_usage/tutorials/video_pipelines/video_classification.en.md)</b>, supporting the extraction of spatiotemporal features from videos and accurate classification.
  - <b>[Multilingual Speech Recognition Pipeline](pipeline_usage/tutorials/speech_pipelines/multilingual_speech_recognition.en.md)</b>, supporting automatic conversion of human-spoken multiple languages into corresponding text or commands.
  - <b>[3D Bev Detection Pipeline](pipeline_usage/tutorials/cv_pipelines/3d_bev_detection.en.md)</b>, supporting input from multiple sensors (LiDAR, surround-view RGB cameras, etc.), processing data through deep learning methods, and outputting information such as position, shape, orientation, and category of objects in three-dimensional space.

- <b>New models:</b>
  - Added 28 OCR models, including the self-developed PP-DocLayout series for high precision and efficiency in layout area detection, the self-developed PP-FormulaNet series for high precision and efficiency in formula recognition, the self-developed SLANeXt series for table structure recognition, and the PP-OCRv4_server_rec_doc model for higher recognition accuracy in text recognition.
  - Added 11 CV models, including 3D multimodal fusion detection models, open vocabulary object detection and segmentation models, rotated object detection models, and human keypoint detection models.
  - Added 5 Speech models, including 5 models from the Whisper series.
  - Added 4 Video models, including 1 video detection model and 3 video classification models.

- <b>Model and pipeline capability upgrades:</b>
  - Models and pipelines support more parameters, such as category thresholds for object detection models, expansion coefficients for text detection models, etc. CV and OCR models support PDF file input.
  - OCR pipelines support document preprocessing operations such as document orientation classification and document correction, with built-in text line orientation classification models.
  - Document Scene Information Extraction v3 pipeline supports standard OpenAI interface calls to large language models, supporting more large language model calls.
  - Optimized user experience, with changes to some model and pipeline interfaces. For details, refer to the [API Upgrade Document](API_change_log/v3.0.0rc.md).

- <b>Multi-hardware support:</b>
  - Added model training and inference capabilities for the Suiyuan GCU hardware, supporting 90+ models, [GCU Model List](support_list/model_list_gcu.md)
  - Added 50+ models for Ascend NPU, [NPU Model List](support_list/model_list_npu.en.md)
  - Added 10+ models for Kunlunxin XPU, [XPU Model List](support_list/model_list_xpu.en.md)
  - Added 10+ models for Cambricon MLU, [MLU Model List](support_list/model_list_mlu.en.md)
  - Added 30+ models for Hygon DCU, [DCU Model List](support_list/model_list_dcu.en.md)

- <b>Multi-environment adaptation:</b>
  - Adapted to Windows systems, supporting the use of PaddleX for model training and inference on Windows. Fixed some installation failures on Windows systems.
  - Training and inference fully adapted to Python3.11 and Python3.12.

- <b>Comprehensive upgrade of deployment capabilities:</b>
  - <b>High-performance inference:</b>
    - Simplified installation and usage: Supports one-click installation of high-performance inference plugins using PaddleX CLI; no authentication is required to use high-performance inference plugins.
    - Cross-platform support: Added support for Windows systems.
    - Expanded model support: Increased the number of supported models, currently supporting 220+ models.
    - Open-sourced core code: Open-sourced the core inference library ultra-infer, facilitating secondary development and customization by developers.
  - <b>Serving:</b>
    - Upgraded basic serving solutions: Upgraded basic serving solutions, supporting new pipelines and adapting to new features of existing pipelines.
    - Added high-stability serving solutions: Added high-stability serving solutions, supporting flexible adjustment of service configurations to optimize service performance, with multiple solutions to meet different user needs.

### PaddleX v3.0.0beta2 (11.15/2024)

PaddleX 3.0 Beta2 is fully compatible with the PaddlePaddle 3.0b2 version. <b>This update introduces new pipelines for general image recognition, face recognition, vehicle attribute recognition, and pedestrian attribute recognition. We have also developed 42 new models to fully support the Ascend 910B, with extensive documentation available on [GitHub Pages](https://paddlepaddle.github.io/PaddleX/latest/en/index.html).</b> Here’s a detailed look at the new features and improvements:

- <b>New Pipelines:</b>
  - <b>[General Image Recognition](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/pipeline_usage/tutorials/cv_pipelines/general_image_recognition.en.md):</b> Introducing a powerful pipeline with enhanced feature extraction models. This allows for user-defined image database recognition of unknown categories, providing more customizable recognition options compared to existing open-domain object detection. Supports high-performance inference and service deployment.
  - <b>[Face Recognition](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/pipeline_usage/tutorials/cv_pipelines/face_recognition.en.md):</b> This new pipeline enables the addition and removal of entries in the face database, with robust support for high-performance inference and service deployment.
  - <b>[Vehicle Attribute Recognition](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/pipeline_usage/tutorials/cv_pipelines/vehicle_attribute_recognition.md):</b> Detect and recognize vehicle attributes such as color and model in images. Supports high-performance inference and service deployment.
  - <b>[Pedestrian Attribute Recognition](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/pipeline_usage/tutorials/cv_pipelines/pedestrian_attribute_recognition.en.md):</b> Detect and recognize pedestrian attributes such as age, gender, and clothing in images. Supports high-performance inference and service deployment.

- <b>New Capabilities:</b>
  - <b>Documentation Support:</b> Comprehensive [GitHub Pages](https://paddlepaddle.github.io/PaddleX/latest/en/index.html) is available, featuring search functionality and user commentary.
  - <b>Benchmarking:</b> Print model inference benchmark information with related [documentation](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/module_usage/instructions/benchmark.en.md).
  - <b>Model Development:</b> We have introduced 42 new model development processes specifically adapted for Ascend 910B. Check out the [model list](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/support_list/model_list_npu.en.md).

- <b>Optimizations:</b>
  - <b>Formula Recognition:</b> Now supports PDF input and visualization of results.
  - <b>Seal Recognition:</b> Supports PDF format input.
  - <b>Layout Parsing:</b> Improved the naming convention for saved images.
  - <b>Pretrained Models:</b> Unified management of pretrained models, integrated into the default configuration file.
  - <b>Model Saving:</b> Upgraded format to ensure high-performance inference.
  - <b>Parameter Tuning:</b> Optimized default parameters for certain models to enable higher accuracy training.

- <b>Bug Fixes:</b>
  - Addressed inaccuracies and inappropriate expressions in documentation, and fixed some invalid URLs.
  - Resolved issues with the PP-LCNet_x1_0_doc_ori inference model.
  - Fixed bugs related to high-performance inference and service deployment.
  - Corrected training configuration issues in SLANet and SLANet_plus that led to low accuracy.


### PaddleX v3.0.0beta1 (9.30/2024)
PaddleX 3.0 Beta1 offers over 200 models accessible through a streamlined Python API for one-click deployment; realizes full-process model development based on unified commands, and opens source the foundational capabilities of the PP-ChatOCRv3-doc special model pipeline; supports high-performance inference and service-oriented deployment for over 100 models, as well as edge deployment for 7 key vision models; and fully adapts the development process of over 70 models to Huawei Ascend 910B, and over 15 models to XPU and MLU.

- <b>Rich Models with One-click Deployment</b>: Integrates over 200 PaddlePaddle models across key domains such as document image intelligent analysis, OCR, object detection, and time series prediction into 13 model pipelines, enabling rapid model experience through a streamlined Python API. Additionally, supports over 20 individual functional modules for convenient model combination.
- <b>Enhanced Efficiency and Lowered Thresholds</b>: Implements full-process model development based on a graphical interface and unified commands, creating 8 special model pipelines that combine large and small models, leverage large model semi-supervised learning, and multi-model fusion, significantly reducing iteration costs.
- <b>Flexible Deployment Across Scenarios</b>: Supports various deployment methods including high-performance, service-oriented, and edge deployment, ensuring efficient model operation and rapid response across different application scenarios.
- <b>Efficient Support for Mainstream Hardware</b>: Seamlessly switches between NVIDIA GPUs, XPU, Ascend, and MLU, ensuring efficient operation.

### PaddleX v3.0.0beta (6.27/2024)
PaddleX 3.0beta integrates the advantages of the PaddlePaddle ecosystem, covering 7 major scenario tasks, constructs 16 model pipelines, and provides a low-code development mode to assist developers in realizing full-process model development on various mainstream hardware.

- <b>Basic Model Pipelines (Rich Models, Comprehensive Scenarios)</b>: Selects 68 high-quality PaddlePaddle models, covering tasks such as image classification, object detection, image segmentation, OCR, text image layout analysis, and time series prediction.
- <b>Special Model Pipelines (Significant Efficiency Improvement)</b>: Provides efficient solutions combining large and small models, large model semi-supervised learning, and multi-model fusion.
- <b>Low-code Development Mode (Convenient Development and Deployment)</b>: Offers both zero-code and low-code development methods.
  - Zero-code Development: Users can interactively submit background training tasks through a graphical user interface (GUI), bridging online and offline deployment, and supporting API-based online service invocation.
  - Low-code Development: Achieves full-process development across 16 model pipelines through unified API interfaces, while supporting user-defined model process serialization.
- <b>Multi-hardware Local Support (Strong Compatibility)</b>: Supports NVIDIA GPUs, XPU, Ascend, and MLU, enabling pure offline usage.

### PaddleX v2.1.0 (12.10/2021)

Added the ultra-lightweight classification model PPLCNet, achieving approximately 5ms prediction speed for a single image on Intel CPUs, with a Top-1 Accuracy of 80.82% on the ImageNet-1K dataset, surpassing ResNet152's performance. Experience it now!
Added the lightweight detection model PP-PicoDet, the first to surpass 30+ mAP(0.5:0.95) within 1M parameters (at 416px input), achieving up to 150FPS prediction on ARM CPUs. Experience it now!
Upgraded PaddleX Restful API to support PaddlePaddle's dynamic graph development mode. Experience it now!
Added negative sample training strategies for detection models. Experience it now!
Added lightweight Python-based service deployment. Experience it now!

### PaddleX v2.0.0 (9.10/2021)
* PaddleX API
  - Added visualization of prediction results for detection and instance segmentation tasks, as well as analysis of prediction errors to assist in model effect analysis
  - Introduced negative sample optimization for detection tasks to suppress false detections in background regions
  - Improved prediction results for semantic segmentation tasks, supporting the return of predicted categories and normalized prediction confidence
  - Enhanced prediction results for image classification tasks, supporting the return of normalized prediction confidence
* Prediction Deployment
  - Completed PaddleX Python prediction deployment, enabling rapid deployment with just 2 APIs
  - Comprehensively upgraded PaddleX C++ deployment, supporting end-to-end unified deployment capabilities for PaddlePaddle vision suites including PaddleDetection, PaddleClas, PaddleSeg, and PaddleX
  - Newly released Manufacture SDK, providing a pre-compiled PaddlePaddle deployment development kit (SDK) for industrial-grade multi-end and multi-platform deployment acceleration, enabling rapid inference deployment through configuring business logic flow files in a low-code manner
* PaddleX GUI
  - Upgraded PaddleX GUI to support 30-series graphics cards
  - Added PP-YOLO V2 model for object detection tasks, achieving 49.5% accuracy on the COCO test dataset and 68.9 FPS prediction speed on V100
  - Introduced a 4.2MB ultra-lightweight model, PP-YOLO tiny, for object detection tasks
  - Added real-time segmentation model BiSeNetV2 for semantic segmentation tasks
  - Newly added the ability to export API training scripts for seamless switching to PaddleX API training
* Industrial Practice Cases
  - Added tutorial cases for steel bar counting and defect detection, focusing on object detection tasks
  - Added tutorial cases for robotic arm grasping, focusing on instance segmentation tasks
  - Added tutorial cases for training and deployment of industrial meter readings, which combines object detection, semantic segmentation, and traditional vision algorithms
  - Added a deployment case tutorial using C# language under Windows system

### PaddleX v2.0.0rc0 (5.19/2021)
* Fully supports PaddlePaddle 2.0 dynamic graphs for an easier development mode
* Added [PP-YOLOv2](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/tutorials/train/object_detection/ppyolov2.py) for object detection tasks, achieving 49.5% accuracy on the COCO test dataset and 68.9 FPS prediction speed on V100
* Introduced a 4.2MB ultra-lightweight model, [PP-YOLO tiny](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/tutorials/train/object_detection/ppyolotiny.py), for object detection tasks
* Added real-time segmentation model [BiSeNetV2](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/tutorials/train/semantic_segmentation/bisenetv2.py) for semantic segmentation tasks
* Comprehensive upgrade of C++ deployment module
    * PaddleInference deployment adapted to 2.0 prediction library [(Usage Documentation)](https://github.com/PaddlePaddle/PaddleX/tree/release/2.0-rc/deploy/cpp)
    * Supports deployment of models from PaddleDetection, PaddleSeg, PaddleClas, and PaddleX
    * Added multi-GPU prediction based on PaddleInference [(Usage Documentation)](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/demo/multi_gpu_model_infer.md)
    * GPU deployment added TensorRT high-performance acceleration engine deployment method based on ONNX
    * GPU deployment added Triton service-oriented deployment method based on ONNX [(Docker Usage Documentation)](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/compile/triton/docker.md)


### PaddleX v1.3.0 (12.19/2020)

- Model Updates
  > - Image Classification model ResNet50_vd adds a pre-trained model with 100,000 categories.
  > - Object Detection model FasterRCNN adds model pruning support.
  > - Object Detection models now support multi-channel image training.

- Model Deployment Updates
  > - Fixed bugs in OpenVINO deployment C++ code.
  > - Raspberry Pi deployment adds Arm V8 support.

- Industry Case Updates
 > - Added an industrial quality inspection case, providing GPU and CPU deployment scenarios for industrial quality inspection, along with optimization strategies related to quality inspection.

- <b>New RESTful API Module</b>
A new RESTful API module is added, enabling developers to quickly develop training platforms based on PaddleX.
 > - Added an HTML Demo based on RESTful API.
 > - Added a Remote version of the visualization client based on RESTful API.
Added deployment solutions for models through OpenVINO.

### PaddleX v1.2.0 (9.9/2020)
- Model Updates
  > - Added the object detection model PPYOLO.
  > - FasterRCNN, MaskRCNN, YOLOv3, DeepLabv3p, and other models now have pre-trained models on the COCO dataset.
  > - Object Detection models FasterRCNN and MaskRCNN add the backbone HRNet_W18.
  > - Semantic Segmentation model DeepLabv3p adds the backbone MobileNetV3_large_ssld.

- Model Deployment Updates
  > - Added deployment solutions for models through OpenVINO.
  > - Added deployment solutions for models on Raspberry Pi.
  > - Optimized data preprocessing and postprocessing code performance for PaddleLite Android deployment.
  > - Optimized Paddle Server-side C++ deployment code, added parameters such as use_mkl, significantly improving model prediction performance on CPUs through mkldnn.

- Industry Case Updates
  > - Added an RGB image remote sensing segmentation case.
  > - Added a multi-channel remote sensing segmentation case.

- Others
  > - Added a dataset splitting function, supporting command-line splitting of ImageNet, PascalVOC, MSCOCO, and semantic segmentation datasets.

  ### PaddleX v1.1.0 (7.13/2020)
- Model Updates
> - Added new semantic segmentation models: HRNet, FastSCNN
> - Added HRNet backbone for object detection (FasterRCNN) and instance segmentation (MaskRCNN)
> - Pre-trained models on COCO dataset for object detection and instance segmentation
> - Integrated X2Paddle, enabling all PaddleX classification and semantic segmentation models to export to ONNX protocol
- Model Deployment Updates
> - Added support for model encryption on Windows platform
> - New deployment and prediction solutions for Jetson and PaddleLite
> - C++ deployment code now supports batch prediction and utilizes OpenMP for parallel acceleration of preprocessing
- Added 2 PaddleX Industrial Cases
> - Portrait segmentation case
> - Industrial meter reading case
- New data format conversion feature, converting data annotated by LabelMe, Jingling Annotation Assistant, and EasyData platform to formats supported by PaddleX
- Updated PaddleX documentation, optimizing the document structure


### PaddleX v1.0.0 (5.21/2020)

- <b>End-to-End Pipeline</b>
  - <b>Data Preparation</b>: Supports the [EasyData Intelligent Data Service Platform](https://ai.baidu.com/easydata/) data protocol, facilitating intelligent annotation and low-quality data cleaning through the platform. It is also compatible with mainstream annotation tool protocols, helping developers complete data preparation faster.
  - <b>Model Training</b>: Integrates [PaddleClas](https://github.com/PaddlePaddle/PaddleClas), [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) vision development kits, providing a rich selection of high-quality pre-trained models for faster achievement of industrial-grade model performance.
  - <b>Model Tuning</b>: Built-in model interpretability modules and [VisualDL](https://github.com/PaddlePaddle/VisualDL) visualization analysis components, providing abundant information for better understanding and optimizing models.
  - <b>Secure Multi-platform Deployment</b>: Integrated with [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) model compression tools and <b>model encryption deployment modules</b>, facilitating high-performance and secure multi-platform deployment in conjunction with Paddle Inference or [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite).

- <b>Integrated Industrial Practices</b>
  - Selects mature model architectures from PaddlePaddle's industrial practices, opening up case study tutorials to accelerate developers' industrial implementation.

- <b>Easy-to-Use and Easy-to-Integrate</b>
  - Unified and user-friendly end-to-end APIs, enabling model training in 5 steps and high-performance Python/C++ deployment with just 10 lines of code.
  - Provides PaddleX-GUI, a cross-platform visualization tool centered on PaddleX, for a quick experience of the full PaddlePaddle deep learning pipeline.
