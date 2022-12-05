# 🆕 🆕 🆕 全新更新！

**强烈推荐!** 我们升级了PaddleX对PaddleDetection部署支持的代码，现在部署PaddleDetection模型，可使用FastDeploy快速部署（支持Python/C++/Android，以及Serving服务化部署)
- [FastDeploy部署PaddleDetection模型](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection)

# PaddleDetection模型部署

当前支持PaddleDetection release/0.5和release/2.1分支导出的模型进行部署（仅支持FasterRCNN/MaskRCNN/PPYOLO/PPYOLOv2/YOLOv3)。PaddleDetection相关详细文档可以查看[官网文档](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1)。

下面主要以YoloV3为例，讲解从模型导出到部署的整个流程。

## 步骤一 部署模型导出

### 1.获取PaddleDetection源码

```sh
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
```

### 2. 导出基于COCO数据的预训练模型

在进入`PaddleDetection`目录后，执行如下命令导出预训练模型

```python
# 导出YOLOv3模型
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml \
                             -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams \
                             --output_dir=./inference_model
```

**如果你需要使用TensorRT进行部署预测**，则需要在导出模型时固定输入shape，命令如下

```python
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml \
                              -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams \
                              TestReader.inputs_def.image_shape=[3,640,640] \
                              --output_dir=./inference_model
```

导出的部署模型会保存在`inference_model/yolov3_darknet53_270e_coco`目录，其结构如下

```
yolov3_darknet
  ├── infer_cfg.yml          # 模型配置文件信息
  ├── model.pdiparams        # 静态图模型参数
  ├── model.pdiparams.info   # 参数额外信息，一般无需关注
  └── model.pdmodel          # 静态图模型文件
```



## 步骤二 编译

参考编译文档

- [Linux系统上编译指南](../compile/paddle/linux.md)
- [Windows系统上编译指南(生成exe)](../compile/paddle/windows.md)
- [Windows系统上编译指南(生成dll供C#调用)](../csharp_deploy/)



## 步骤三 模型预测

编译后即可获取可执行的二进制demo程序`model_infer`和`multi_gpu_model_infer`，分别用于在单卡/多卡上加载模型进行预测，对于分类模型，调用如下命令即可进行预测

```sh
# 使用gpu加 --use_gpu=1 参数
./build/demo/model_infer --model_filename=inference_model/yolov3_darknet53_270e_coco/model.pdmodel \
                         --params_filename=inference_model/yolov3_darknet53_270e_coco/model.pdiparams \
                         --cfg_file=inference_model/yolov3_darknet53_270e_coco/infer_cfg.yml \
                         --image=test.jpg \
                         --model_type=det
```

输出结果如下(分别为类别id， 类别标签，置信度，xmin, ymin, width, height)

```
Box(0   person  0.295455    424.517 163.213 38.1692 114.158)
Box(0   person  0.13875 381.174 172.267 22.2411 44.209)
Box(0   person  0.0255658   443.665 165.08  35.4124 129.128)
Box(39  bottle  0.356306    551.603 288.384 34.9819 112.599)
```

关于demo程序的详细使用方法可分别参考以下文档

- [单卡加载模型预测示例](../demo/model_infer.md)
- [多卡加载模型预测示例](../demo/multi_gpu_model_infer.md)
- [PaddleInference集成TensorRT加载模型预测示例](../../demo/tensorrt_infer.md)
