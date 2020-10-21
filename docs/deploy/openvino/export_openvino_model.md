# OpenVINO模型转换
将Paddle模型转换为OpenVINO的Inference Engine  

## 环境依赖

* ONNX 1.5.0+
* PaddleX 1.0+
* OpenVINO 2020.4

**说明**：PaddleX安装请参考[PaddleX](https://paddlex.readthedocs.io/zh_CN/develop/install.html) ， OpenVINO安装请参考[OpenVINO](https://docs.openvinotoolkit.org/latest/index.html)，ONNX请安装1.5.0以上版本否则会出现转模型错误。

请确保系统已经安装好上述基本软件，**下面所有示例以工作目录 `/root/projects/`演示**。

## 导出inference模型
paddle模型转openvino之前需要先把paddle模型导出为inference格式模型，导出的模型将包括__model__、__params__和model.yml三个文件名，导出命令如下
```
paddlex --export_inference --model_dir=/path/to/paddle_model --save_dir=./inference_model --fixed_input_shape=[w,h]
```

## 导出OpenVINO模型

```
mkdir -p /root/projects
cd /root/projects
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX/deploy/openvino/python

python converter.py --model_dir /path/to/inference_model --save_dir /path/to/openvino_model --fixed_input_shape [w,h]
```
**转换成功后会在save_dir下出现后缀名为.xml、.bin、.mapping三个文件**  
转换参数说明如下：

|  参数   | 说明  |
|  ----  | ----  |
| --model_dir  | Paddle模型路径，请确保__model__, \_\_params__model.yml在同一个目录|
| --save_dir  | OpenVINO模型保存路径 |
| --fixed_input_shape  | 模型输入的[W,H] |
| --data type(option)  | FP32、FP16，默认为FP32，VPU下的IR需要为FP16 |  

**注意**：
- 由于OpenVINO不支持ONNX的resize-11 OP的原因，目前还不支持Paddle的分割模型
- YOLOv3在通过OpenVINO部署时，由于OpenVINO对ONNX OP的支持限制，我们在将YOLOv3的Paddle模型导出时，对最后一层multiclass_nms进行了特殊处理，导出的ONNX模型，最终输出的Box结果包括背景类别（而Paddle模型不包含），此处在OpenVINO的部署代码中，我们通过后处理过滤了背景类别。
