# Introduction to OpenVINO deployment
PaddleX supports the prediction acceleration of the trained Paddle model through [OpenVINO](https://docs.openvinotoolkit.org/latest/index.html). For details and installation process, refer to OpenVINO document. This document is based on OpenVINO 2020.4 and 2021.1.
**Note**: Resize-11 is supported starting from OpenVINO 2021.1 because the PaddleX segmentation model uses ReSize-11 Op. Make sure to download OpenVINO 2021.1+.


## Deployment support
The following table lists the support status of using OpenVINO for acceleration by PaddleX in different environments

| Hardware platform | Linux | Windows | Raspbian OS | C++ | Python | Classification | Detection | Segmentation |
| ----|  ---- | ---- | ----|  ---- | ---- |---- | ---- |---- |
| CPU | Supported | Supported | Not supported | Supported | Supported | Supported | Supported | Supported |
| VPU | Supported | Supported | Supported | Supported | Supported | Supported | Not supported | Not supported |


**Note**: Raspbian OS is the Raspberry OS. The detection model supports only YOLOv3

## Deployment process
**The PaddleX to OpenVINO deployment process has the following two steps**:

* **Model conversion**: Convert Paddle's model to OpenVINO's Inference Engine.
* **Prediction Deployment**: Prediction with using Inference Engine**

## Model conversion
**For model conversion, refer to the [Model Conversion](./export_openvino_model.md) document.**
**Note**: Since the methods of converting OpenVINO model are the same under different hardware and software platforms, details on how to convert the model are omitted in subsequent documents.

## Inference deployment
The methods of deploying OpenVINO to implement predictions are not completely identical in different hardware and software. For details, refer to:

**[Linux](./linux.md)**: introduces the prediction acceleration by using OpenVINO when PaddleX operates on Linux or Raspbian OS with C++ programming language and hardware platform is CPU or VPU.

**[Windows](./windows.md)**: introduces the prediction acceleration by using OpenVINO when PaddleX operates on Windows OS with C++ programming language and hardware platform is CPU or VPU

**[Python](./python.md)**: introduces the prediction acceleration by using OpenVINO when PaddleX operates in Python
