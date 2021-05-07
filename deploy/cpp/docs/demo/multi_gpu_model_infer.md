# 多GPU卡模型加载预测示例

本文档说明`PaddleX/deploy/cpp/demo/multi_gpu_model_infer.cpp`编译后的使用方法，仅供用户参考进行使用，开发者可基于此demo示例进行二次开发，满足集成的需求。

在多卡上实现机制如下

- 模型初始化，针对每张GPU卡分别加一个独立的模型
- 模型预测时，根据传入的图像vector，将输入均分至每张GPU卡进行多线程并行预测
- 预测结束后，将各GPU卡上预测结果汇总返回

## 步骤一、编译

参考编译文档

- [Linux系统上编译指南](../compile/paddle/linux.md)
- [Windows系统上编译指南](../compile/paddle/windows.md)

## 步骤二、准备PaddlePaddle部署模型

开发者可从以下套件获取部署模型，需要注意，部署时需要准备的是导出来的部署模型，一般包含`model.pdmodel`、`model.pdiparams`和`deploy.yml`三个文件，分别表示模型结构、模型权重和各套件自行定义的配置信息。

- [PaddleDetection导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/deploy/EXPORT_MODEL.md)
- [PaddleSeg导出模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0/docs/model_export.md)
- [PaddleClas导出模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/tutorials/getting_started.md#4-%E4%BD%BF%E7%94%A8inference%E6%A8%A1%E5%9E%8B%E8%BF%9B%E8%A1%8C%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)
- [PaddleX导出模型](https://paddlex.readthedocs.io/zh_CN/develop/deploy/server/python.html#python)


 用户也可直接下载本教程中从PaddleDetection中导出的YOLOv3模型进行测试，[点击下载](https://bj.bcebos.com/paddlex/deploy2/models/yolov3_mbv1.tar.gz)。

## 步骤三、使用编译好的可执行文件预测

以步骤二中下载的YOLOv3模型为例，执行如下命令即可进行模型加载和预测

```
build/demo/model_infer --model_filename=yolov3_mbv1/model/model.pdmodel \
                       --params_filename=yolov3_mbv1/model/model.pdiparams \
                       --cfg_file=yolov3_mbv1/model/infer_cfg.yml \
                       --image=yolov3_mbv1/file_list.txt \
                       --gpu_id=0,1 \
                       --batch_size=4 \
                       --model_type=det
```

输出结果如下(分别为类别id、标签、置信度、xmin、ymin、w, h)

```
Box(0   person  0.0180757   0   386.488 52.8673 38.5124)
Box(14  bird    0.0226735   7.03722 9.77164 491.656 360.871)
Box(25  umbrella    0.0198202   7.03722 9.77164 491.656 360.871)
Box(26  handbag 0.0108408   0   386.488 52.8673 38.5124)
Box(39  bottle  0.12783 183.808 187.242 8.61859 34.643)
Box(56  chair   0.136626    546.628 283.611 62.4004 138.243)
```

### 参数说明

| 参数            | 说明                                                         |
| --------------- | ------------------------------------------------------------ |
| model_filename  | **[必填]** 模型结构文件路径，如`yolov3_darknet/model.pdmodel` |
| params_filename | **[必填]** 模型权重文件路径，如`yolov3_darknet/model.pdiparams` |
| cfg_file        | **[必填]** 模型配置文件路径，如`yolov3_darknet/infer_cfg.yml` |
| model_type      | **[必填]** 模型来源，det/seg/clas/paddlex，分别表示模型来源于PaddleDetection、PaddleSeg、PaddleClas和PaddleX |
| image_list      | 待预测的图片路径列表文件路径，如步骤三中的`yolov3_darknet/file_list.txt` |
| gpu_id          | 使用GPU预测时的GUI设备ID，默认为0                            |
| batch_size      | 设定每次预测时的batch大小(最终会均分至各张卡上)，默认为1(多卡时可填为，如0,1) |
| thread_num      | 每张GPU卡上模型在图像预处理时的并行线程数，默认为1           |



## 相关文档

- [部署API和数据结构文档](../apis/model.md)

