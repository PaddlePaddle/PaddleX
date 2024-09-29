# Version Update Information

## Latest Version Information
### PaddleX v3.0.0beta1 (9.30/2024)
PaddleX 3.0 Beta1 offers over 200 models accessible through a streamlined Python API for one-click deployment; realizes full-process model development based on unified commands, and opens source the foundational capabilities of the PP-ChatOCRv3 special model pipeline; supports high-performance inference and service-oriented deployment for over 100 models, as well as edge deployment for 7 key vision models; and fully adapts the development process of over 70 models to Huawei Ascend 910B, and over 15 models to XPU and MLU.

- **Rich Models with One-click Deployment**: Integrates over 200 PaddlePaddle models across key domains such as document image intelligent analysis, OCR, object detection, and time series prediction into 13 model pipelines, enabling rapid model experience through a streamlined Python API. Additionally, supports over 20 individual functional modules for convenient model combination.
- **Enhanced Efficiency and Lowered Thresholds**: Implements full-process model development based on a graphical interface and unified commands, creating 8 special model pipelines that combine large and small models, leverage large model semi-supervised learning, and multi-model fusion, significantly reducing iteration costs.
- **Flexible Deployment Across Scenarios**: Supports various deployment methods including high-performance, service-oriented, and edge deployment, ensuring efficient model operation and rapid response across different application scenarios.
- **Efficient Support for Mainstream Hardware**: Seamlessly switches between NVIDIA GPUs, XPU, Ascend, and MLU, ensuring efficient operation.

### PaddleX v3.0.0beta (6.27/2024)
PaddleX 3.0beta integrates the advantages of the PaddlePaddle ecosystem, covering 7 major scenario tasks, constructs 16 model pipelines, and provides a low-code development mode to assist developers in realizing full-process model development on various mainstream hardware.

- **Basic Model Pipelines (Rich Models, Comprehensive Scenarios)**: Selects 68 high-quality PaddlePaddle models, covering tasks such as image classification, object detection, image segmentation, OCR, text image layout analysis, and time series prediction.
- **Special Model Pipelines (Significant Efficiency Improvement)**: Provides efficient solutions combining large and small models, large model semi-supervised learning, and multi-model fusion.
- **Low-code Development Mode (Convenient Development and Deployment)**: Offers both zero-code and low-code development methods.
  - Zero-code Development: Users can interactively submit background training tasks through a graphical user interface (GUI), bridging online and offline deployment, and supporting API-based online service invocation.
  - Low-code Development: Achieves full-process development across 16 model pipelines through unified API interfaces, while supporting user-defined model process serialization.
- **Multi-hardware Local Support (Strong Compatibility)**: Supports NVIDIA GPUs, XPU, Ascend, and MLU, enabling pure offline usage.

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

- **New RESTful API Module**
A new RESTful API module is added, enabling developers to quickly develop training platforms based on PaddleX.
 > - Added an HTML Demo based on RESTful API.
 > - Added a Remote version of the visualization client based on RESTful API.
Added deployment solutions for models through OpenVINO [Detailed Link](https://paddlex.readthedocs.io/en/develop/deploy/openvino/index.html)

### PaddleX v1.2.0 (9.9/2020)
- Model Updates
  > - Added the object detection model PPYOLO [Detailed Link](https://paddlex.readthedocs.io/en/develop/apis/models/detection.html#paddlex-det-ppyolo)
  > - FasterRCNN, MaskRCNN, YOLOv3, DeepLabv3p, and other models now have pre-trained models on the COCO dataset.
  > - Object Detection models FasterRCNN and MaskRCNN add the backbone HRNet_W18 [Detailed Link](https://paddlex.readthedocs.io/en/develop/apis/models/detection.html#paddlex-det-fasterrcnn)
  > - Semantic Segmentation model DeepLabv3p adds the backbone MobileNetV3_large_ssld [Detailed Link](https://paddlex.readthedocs.io/en/develop/apis/models/semantic_segmentation.html#paddlex-seg-deeplabv3p)

- Model Deployment Updates
  > - Added deployment solutions for models through OpenVINO [Detailed Link](https://paddlex.readthedocs.io/en/develop/deploy/openvino/index.html)
  > - Added deployment solutions for models on Raspberry Pi [Detailed Link](https://paddlex.readthedocs.io/en/develop/deploy/raspberry/index.html)
  > - Optimized data preprocessing and postprocessing code performance for PaddleLite Android deployment.
  > - Optimized Paddle Server-side C++ deployment code, added parameters such as use_mkl, significantly improving model prediction performance on CPUs through mkldnn.

- Industry Case Updates
  > - Added an RGB image remote sensing segmentation case [Detailed Link](https://paddlex.readthedocs.io/en/develop/examples/remote_sensing.html)
  > - Added a multi-channel remote sensing segmentation case [Detailed Link](https://paddlex.readthedocs.io/en/develop/examples/multi-channel_remote_sensing/README.html)

- Others
  > - Added a dataset splitting function, supporting command-line splitting of ImageNet, PascalVOC, MSCOCO, and semantic segmentation datasets [Detailed Link](https://paddlex.readthedocs.io/en/develop/data/format/classification.html#id2)

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

- **End-to-End Pipeline**
  - **Data Preparation**: Supports the [EasyData Intelligent Data Service Platform](https://ai.baidu.com/easydata/) data protocol, facilitating intelligent annotation and low-quality data cleaning through the platform. It is also compatible with mainstream annotation tool protocols, helping developers complete data preparation faster.
  - **Model Training**: Integrates [PaddleClas](https://github.com/PaddlePaddle/PaddleClas), [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) vision development kits, providing a rich selection of high-quality pre-trained models for faster achievement of industrial-grade model performance.
  - **Model Tuning**: Built-in model interpretability modules and [VisualDL](https://github.com/PaddlePaddle/VisualDL) visualization analysis components, providing abundant information for better understanding and optimizing models.
  - **Secure Multi-platform Deployment**: Integrated with [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) model compression tools and **model encryption deployment modules**, facilitating high-performance and secure multi-platform deployment in conjunction with Paddle Inference or [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite).

- **Integrated Industrial Practices**
  - Selects mature model architectures from PaddlePaddle's industrial practices, opening up case study tutorials to accelerate developers' industrial implementation.

- **Easy-to-Use and Easy-to-Integrate**
  - Unified and user-friendly end-to-end APIs, enabling model training in 5 steps and high-performance Python/C++ deployment with just 10 lines of code.
  - Provides PaddleX-GUI, a cross-platform visualization tool centered on PaddleX, for a quick experience of the full PaddlePaddle deep learning pipeline.