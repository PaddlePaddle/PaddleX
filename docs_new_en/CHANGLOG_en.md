# Version Update Information

## Latest Version Information

### PaddleX v3.0.0beta1 (9.30/2024)
PaddleX 3.0 Beta1 offers over 200 models accessible through a streamlined Python API for one-click deployment; realizes full model development based on unified commands, and opens source the foundational capabilities of the PP-ChatOCRv3 feature model pipeline; supports high-performance inference and service deployment for over 100 models, as well as edge deployment for 7 key vision models; and fully adapts the development process of 70+ models to Huawei Ascend 910B, and 15+ models to XPU and MLU.

- **Rich Models with One-click Deployment**: Integrates over 200 PaddlePaddle models across key domains such as text and image intelligent analysis, OCR, object detection, and time series prediction into 13 pipelines, enabling rapid model experience through a simplified Python API. Additionally, supports over 20 standalone functional modules for easy model combination.
- **Enhanced Efficiency and Lowered Thresholds**: Implements full model development based on a graphical interface and unified commands, creating 8 feature pipelines combining large and small models, large model semi-supervised learning, and multi-model fusion, significantly reducing iteration costs.
- **Flexible Deployment Across Scenarios**: Supports various deployment methods including high-performance, service-oriented, and edge deployment, ensuring efficient model operation and rapid response in different application scenarios.
- **Efficient Support for Mainstream Hardware**: Seamlessly switches between NVIDIA GPUs, XPU, NPU, and MLU, ensuring efficient operation.

### PaddleX v3.0.0beta (6.27/2024)
PaddleX 3.0beta integrates the strengths of the PaddlePaddle ecosystem, covering 7 major scenarios and constructing 16 model pipelines, providing a low-code development mode to assist developers in achieving full model development on multiple mainstream hardware.

- **Basic Model Pipelines (Rich Models, Comprehensive Scenarios)**: Selects 68 high-quality PaddlePaddle models, covering tasks such as image classification, object detection, image segmentation, OCR, text and image layout analysis, and time series prediction.
- **Feature Model Pipelines (Significant Efficiency Improvements)**: Provides efficient solutions combining large and small models, large model semi-supervised learning, and multi-model fusion.
- **Low-code Development Mode (Convenient Development and Deployment)**: Offers both zero-code and low-code development methods.
  - Zero-code Development: Users submit background training tasks interactively through a graphical user interface (GUI), bridging online and offline deployment, and supporting API-based online service invocation.
  - Low-code Development: Achieves full-process development of 16 model pipelines through unified API interfaces, while supporting user-defined model process chaining.
- **Multi-hardware Local Support (Strong Compatibility)**: Supports NVIDIA GPUs, XPU, NPU, and MLU, enabling purely offline usage.

### PaddleX v2.1.0 (12.10/2021)

Added the ultra-lightweight classification model PPLCNet, achieving approximately 5ms prediction speed for a single image on Intel CPUs, with a Top-1 Accuracy of 80.82% on the ImageNet-1K dataset, surpassing ResNet152. Experience it now!
Introduced the lightweight detection feature model PP-PicoDet, the first to surpass 30+ mAP(0.5:0.95) within 1M parameters (at 416px input), achieving up to 150FPS prediction on ARM CPUs. Experience it now!
Upgraded PaddleX Restful API to support PaddlePaddle's dynamic graph development mode. Experience it now!
Added negative sample training strategies for detection models. Experience it now!
Introduced lightweight Python service deployment. Experience it now!

### PaddleX v2.0.0 (9.10/2021)
* PaddleX API
  - Added visualization of prediction results and error analysis for detection and instance segmentation tasks to assist in model effect analysis
  - Enhanced negative sample optimization for detection tasks to suppress false detections in background regions
  - Improved prediction results for semantic segmentation tasks, supporting the return of predicted classes and normalized prediction confidence
  - Improved prediction results for image classification tasks, supporting the return of normalized prediction confidence
* Prediction Deployment
  - Completed PaddleX Python prediction deployment, enabling rapid deployment with just 2 APIs
  - Comprehensively upgraded PaddleX C++ deployment, supporting end-to-end unified deployment capabilities```markdown
### PaddleX v1.3.0 (12.19/2020)

- Model Updates
  > - Image Classification model ResNet50_vd adds a pre-trained model with 100,000 categories.
  > - Object Detection model FasterRCNN adds model pruning support.
  > - Object Detection models now support multi-channel image training.

- Model Deployment Updates
  > - Fixes bugs in OpenVINO deployment C++ code.
  > - Raspberry Pi deployment adds Arm V8 support.

- Industrial Case Updates
 > - Adds an industrial quality inspection case, providing GPU and CPU deployment scenarios for industrial quality inspection, along with optimization strategies related to quality inspection [Details Link](https://paddlex.readthedocs.io/en/develop/examples/industrial_quality_inspection)

- **New RESTful API Module**
Adds a RESTful API module, enabling developers to quickly develop training platforms based on PaddleX
 > - Adds an HTML Demo based on RESTful API [Details Link](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/gui/introduction.md#paddlex-web-demo)
 > - Adds a Remote version of the visual client based on RESTful API [Details Link](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/gui/introduction.md#paddlex-remote-gui)
Adds a deployment solution for models through OpenVINO [Details Link](https://paddlex.readthedocs.io/en/develop/deploy/openvino/index.html)

### PaddleX v1.2.0 (9.9/2020)
- Model Updates
  > - Adds the object detection model PPYOLO [Details Link](https://paddlex.readthedocs.io/en/develop/apis/models/detection.html#paddlex-det-ppyolo)
  > - FasterRCNN, MaskRCNN, YOLOv3, DeepLabv3p, and other models add pre-trained models with the built-in COCO dataset.
  > - Object Detection models FasterRCNN and MaskRCNN add the backbone HRNet_W18 [Details Link](https://paddlex.readthedocs.io/en/develop/apis/models/detection.html#paddlex-det-fasterrcnn)
  > - Semantic Segmentation model DeepLabv3p adds the backbone MobileNetV3_large_ssld [Details Link](https://paddlex.readthedocs.io/en/develop/apis/models/semantic_segmentation.html#paddlex-seg-deeplabv3p)

- Model Deployment Updates
  > - Adds a deployment solution for models through OpenVINO [Details Link](https://paddlex.readthedocs.io/en/develop/deploy/openvino/index.html)
  > - Adds a deployment solution for models on Raspberry Pi [Details Link](https://paddlex.readthedocs.io/en/develop/deploy/raspberry/index.html)
  > - Optimizes the data preprocessing and postprocessing code performance for PaddleLite Android deployment.
  > - Optimizes Paddle Server-side C++ deployment code, adds parameters such as use_mkl, significantly improving model prediction performance on CPUs through mkldnn.

- Industrial Case Updates
  > - Adds an RGB image remote sensing segmentation case [Details Link](https://paddlex.readthedocs.io/en/develop/examples/remote_sensing.html)
  > - Adds a multi-channel remote sensing segmentation case [Details Link](https://paddlex.readthedocs.io/en/develop/examples/multi-channel_remote_sensing/README.html)

- Others
  > - Adds a dataset splitting function, supporting command-line splitting of ImageNet, PascalVOC, MSCOCO, and semantic segmentation datasets [Details Link](https://paddlex.readthedocs.io/en/develop/data/format/classification.html#id2)

### PaddleX v1.1.0 (7.13/2020)
- Model Updates
> - Adds semantic segmentation models HRNet, FastSCNN
> - Object Detection FasterRCNN, Instance Segmentation MaskRCNN add backbone HRNet
> - Object Detection/Instance Segmentation models add pre-trained models with the COCO dataset
> - Integrates X2Paddle, enabling all classification and semantic segmentation models in PaddleX to be exported to the ONNX protocol
- Model Deployment Updates
> - Model encryption adds support for the Windows platform
> - Adds Jetson, PaddleLite model deployment prediction solutions
> - C++ deployment code adds batch```markdown
- **Industry-oriented Practices**
  - Carefully selected mature model architectures from PaddlePaddle's industrial practices, with open case study tutorials to accelerate developers' industrial deployment.

- **Easy-to-use and Integrate**
  - Unified and user-friendly full-process APIs, enabling model training in 5 steps and high-performance deployment in Python/C++ with just 10 lines of code.
  - Provides PaddleX-GUI, a cross-platform visualization tool integrated with PaddleX, for a quick experience of the full deep learning process with PaddlePaddle.