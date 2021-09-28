# PaddleX模型部署

当前对PaddleX静态图和动态图版本导出的模型都支持


## 步骤一 部署模型导出

请参考[PaddlX模型导出文档](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/export_model.md)


## 步骤二 编译

参考编译文档

- [Linux系统上编译指南](../compile/paddle/linux.md)
- [Windows系统上编译指南](../compile/paddle/windows.md)


## 步骤三 模型预测

编译后即可获取可执行的二进制demo程序`model_infer`和`multi_gpu_model_infer`，分别用于在单卡/多卡上加载模型进行预测，对于分类模型，调用如下命令即可进行预测

```sh
# 使用gpu加 --use_gpu=1 参数
./build/demo/model_infer --model_filename=model.pdmodel \
                         --params_filename=model.pdiparams \
                         --cfg_file=model.yml \
                         --image=test.jpg \
                         --model_type=paddlex
```

检测模型的输出结果如下(分别为类别id， 类别标签，置信度，xmin, ymin, width, height)

```
Box(0   person  0.295455    424.517 163.213 38.1692 114.158)
Box(0   person  0.13875 381.174 172.267 22.2411 44.209)
Box(0   person  0.0255658   443.665 165.08  35.4124 129.128)
Box(39  bottle  0.356306    551.603 288.384 34.9819 112.599)
```

分割模型输出结果如下(由于分割结果的score_map和label_map不便于直接输出，因此在demo程序中仅输出这两个mask的均值和方差)

```
ScoreMask(mean: 12.4814 std:    10.4955)    LabelMask(mean: 1.98847 std:    10.3141)
```

分类模型输出结果如下(分别为类别id， 类别标签，置信度)

```
Classify(809    sunscreen   0.939211)
```

关于demo程序的详细使用方法可分别参考以下文档

- [单卡加载模型预测示例](../demo/model_infer.md)
- [多卡加载模型预测示例](../demo/multi_gpu_model_infer.md)
- [PaddleInference集成TensorRT加载模型预测示例](../demo/tensorrt_infer.md)
- [Windows系统下使用C#语言部署](../../../../examples/C%23_deploy/)
