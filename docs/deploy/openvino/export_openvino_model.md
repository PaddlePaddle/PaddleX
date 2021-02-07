# OpenVINO model conversion
The document describes how to convert Paddle models to Inference Engine of the OpenVINO.

## Environment dependence

* Paddle2ONNX 0.4
* ONNX 1.6.0+
* PaddleX 1.3+
* OpenVINO 2020.4+

**Note**: For PaddleX installation, refer to [PaddleX](https://paddlex.readthedocs.io/zh_CN/develop/install.html) document. For OpenVINO installation, refer to [OpenVINO](https://docs.openvinotoolkit.org/latest/index.html) document,Please set init env after install OpenVINO;otherwise,the error 'No module named mo' may ocuur. For ONNX, install V1.6.0 or later; otherwise, the conversion error may occur. For Paddle2ONNX, make sure install V0.4.

Make sure that the above basic software is installed on your system. **All the following examples are in the working directory `/root/projects/`**.

## Export an inference model
Before converting paddle model to openvino, you need to export the paddle model to inference format first. The exported model includes __model__, __params__, and model.yml. The export command is as follows:
```
paddlex --export_inference --model_dir=/path/to/paddle_model --save_dir=./inference_model --fixed_input_shape=[w,h]
```

**Note**: If you need to convert the OpenVINO model to export the inference model, make sure to specify the `--fixed_input_shape` parameter to fix the input size of the model, and the input size of the model should be the same as the OpenVINO model during training

## Export an OpenVINOmodel

```
mkdir -p /root/projects
cd /root/projects
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX/deploy/openvino/python

python converter.py --model_dir /path/to/inference_model --save_dir /path/to/openvino_model --fixed_input_shape [w,h]
```
**After the conversion is successful, three files with suffixes .xml, .bin and .mapping appear under save_dir.**
The conversion parameters are described as follows:

| Parameters | Description |
|  ----  | ----  |
| --model_dir | Paddle model path: make sure that __model__ and \_\_params__model.yml are in the same directory. |
| --save_dir | OpenVINO model storage path |
| --fixed_input_shape | W,H[ for model input] |
| --data type(option) | (Optional) FP32 and FP16. The default value is FP32. IR under VPU needs to be FP16. |

**Note**:
- Because OpenVINO supports the ONNX resize-11 OP from version 2021.1, make sure to download OpenVINO 2021.1+ when use CPU. 
- Because OpenVINO not supports Range Layer,make sure to download OpenVINO 2020.4 when use VPU. 
- Please init OpenVINO env first;otherwise,the error 'No module named mo' may ocuur.See [FAQ](./faq.md)
- In the deployment of YOLOv3 through OpenVINO, due to the OpenVINO’s limitation support for ONNX OPs, the special processing is performed to the last layer of multiclass_nms to export the ONNX model when the Paddle model of YOLOv3 is exported. The final output Box results include the background category (the Paddle model does not include it). Here, in the deployment codes of OpenVINO, the background category is filtered through post-processing.
