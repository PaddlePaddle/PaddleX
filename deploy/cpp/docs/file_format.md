# 各套件模型导出前后文件夹状态

| 套件名称 | 导出前文件 | 导出后文件 |
| :-- | :-- | :-- |
| PaddleX静态图 | model.pdparams<br>model.pdmodel<br>model.yml | __ model__<br>__ params__<br>model.yml |
| PaddleX动态图 | model.pdparams<br>model.pdopt<br>model.yml | model.pdmodel<br>model.pdiparams<br>model.pdiparams.info<br>model.yml<br>pipeline.yml |
| PaddleDetection | weights/yolov3_darknet53_270e_coco.pdparams |  infer_cfg.yml<br>model.pdiparams<br>model.pdiparams.info<br>model.pdmodel |
| PaddleSeg | bisenet/model.pdparams | deploy.yaml<br>model.pdiparams<br>model.pdiparams.info<br>model.pdmodel |
| PaddleClas | MobileNetV3_large_x1_0<br>output/MobileNetV3_large_x1_0/best_model/ppcls | model.pdiparams<br>model.pdiparams.info<br>model.pdmodel |


