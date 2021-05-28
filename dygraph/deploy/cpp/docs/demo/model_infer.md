# 模型加载预测示例

本文档说明`PaddleX/deploy/cpp/demo/model_infer.cpp`编译后的使用方法，仅供用户参考进行使用，开发者可基于此demo示例进行二次开发，满足集成的需求。

## 步骤一、编译
参考编译文档
- [Linux系统上编译指南](../compile/paddle/linux.md)
- [Windows系统上编译指南](../compile/paddle/windows.md)

## 步骤二、准备PaddlePaddle部署模型
开发者可从以下套件获取部署模型，需要注意，部署时需要准备的是导出来的部署模型，一般包含`model.pdmodel`、`model.pdiparams`和`deploy.yml`三个文件，分别表示模型结构、模型权重和各套件自行定义的配置信息。
- [PaddleDetection导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/deploy/EXPORT_MODEL.md)
- [PaddleSeg导出模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0/docs/model_export.md)
- [PaddleClas导出模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/tutorials/getting_started.md#4-%E4%BD%BF%E7%94%A8inference%E6%A8%A1%E5%9E%8B%E8%BF%9B%E8%A1%8C%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)
- [PaddleX导出模型](https://paddlex.readthedocs.io/zh_CN/develop/deploy/export_model.html)


用户也可直接下载本教程中从PaddleDetection中导出的YOLOv3模型进行测试，[点击下载](https://bj.bcebos.com/paddlex/deploy2/models/yolov3_mbv1.tar.gz)。

## 步骤三、使用编译好的可执行文件预测
以步骤二中下载的YOLOv3模型为例，执行如下命令即可进行模型加载和预测

```sh
# 使用GPU 加参数 --use_gpu=1
build/demo/model_infer --model_filename=yolov3_mbv1/model/model.pdmodel \
                       --params_filename=yolov3_mbv1/model/model.pdiparams \
                       --cfg_file=yolov3_mbv1/model/infer_cfg.yml \
                       --image=yolov3_mbv1/images/000000010583.jpg \
                       --model_type=det
```
输出结果如下(分别为类别id、标签、置信度、xmin、ymin、w, h)
```
Box(0	person	0.0386442	2.11425	53.4415	36.2138	197.833)
Box(39	bottle	0.0134608	2.11425	53.4415	36.2138	197.833)
Box(41	cup	0.0255145	2.11425	53.4415	36.2138	197.833)
Box(43	knife	0.0824398	509.703	189.959	100.65	93.9368)
Box(43	knife	0.0211949	448.076	167.649	162.924	143.557)
Box(44	spoon	0.0234474	509.703	189.959	100.65	93.9368)
Box(45	bowl	0.0461333	0	0	223.386	83.5562)
Box(45	bowl	0.0191819	3.91156	1.276	225.888	214.273)
```
### 参数说明

| 参数            | 说明                                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------ |
| model_filename  | **[必填]** 模型结构文件路径，如`yolov3_darknet/model.pdmodel`                                                |
| params_filename | **[必填]** 模型权重文件路径，如`yolov3_darknet/model.pdiparams`                                              |
| cfg_file        | **[必填]** 模型配置文件路径，如`yolov3_darknet/infer_cfg.yml`                                                |
| model_type      | **[必填]** 模型来源，det/seg/clas/paddlex，分别表示模型来源于PaddleDetection、PaddleSeg、PaddleClas和PaddleX |
| image           | 待预测的图片文件路径                                                                                         |
| use_gpu         | 是否使用GPU，0或者1，默认为0                                                                                 |
| gpu_id          | 使用GPU预测时的GUI设备ID，默认为0                                                                            |



## 相关文档

- [部署接口和数据结构文档](../apis/model.md)
