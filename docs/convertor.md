# 模型转换

## 转ONNX模型
PaddleX基于[Paddle2ONNX工具](https://github.com/PaddlePaddle/paddle2onnx)，提供了便捷的API，支持用户将PaddleX训练保存的模型导出为ONNX模型。
通过如下示例代码，用户即可将PaddleX训练好的MobileNetV2模型导出
```
import paddlex as pdx
pdx.convertor.to_onnx(model_dir='paddle_mobilenet', save_dir='onnx_mobilenet')
```

## 转PaddleLite模型
PaddleX可支持导出为[PaddleLite](https://github.com/PaddlePaddle/Paddle-Lite)支持的模型格式，用于支持用户将模型部署更多硬件设备。
通过如下示例代码，用户即可将PaddleX训练好的MobileNetV2模型导出
```
import paddlex as pdx
pdx.convertor.to_lite(model_dir='paddle_mobilenet', save_dir='lite_mobilnet', terminal='arm')
```
