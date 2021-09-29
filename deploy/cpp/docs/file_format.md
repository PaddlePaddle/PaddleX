# 各套件模型导出前后文件夹状态

| 套件名称 | 导出前文件 | 导出后文件 |
| :-- | :-- | :-- |
| PaddleX | model.pdparams、model.pdopt、model.yml | model.pdmodel、model.pdiparams、model.pdiparams.info、model.yml、pipeline.yml |
| PaddleDetection | configs/yolov3/yolov3_darknet53_270e_coco.yml、weights/yolov3_darknet53_270e_coco.pdparams |  infer_cfg.yml、model.pdiparams、model.pdiparams.info、model.pdmodel |
| PaddleSeg | configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml、bisenet/model.pdparams | deploy.yaml、model.pdiparams、model.pdiparams.info、model.pdmodel |
| PaddleClas | MobileNetV3_large_x1_0、output/MobileNetV3_large_x1_0/best_model/ppcls | model.pdiparams、model.pdiparams.info、model.pdmodel |
