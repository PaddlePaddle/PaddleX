# PaddleDetection部署模型导出

当前支持PaddleDetection release/0.5和release/2.1分支导出的模型进行部署（仅支持FasterRCNN/MaskRCNN/PPYOLO/PPYOLOv2/YOLOv3)。PaddleDetection相关详细文档可以查看[官网文档](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1)。

下面主要以YoloV3为例，讲解模型导出。

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
