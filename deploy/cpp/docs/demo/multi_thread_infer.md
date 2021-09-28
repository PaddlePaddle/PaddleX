# 多线程预测示例

本文档说明`PaddleX/deploy/cpp/demo/multi_thread_infer.cpp`和`multi_thread_infer2.cpp`编译后的使用方法，仅供用户参考进行使用，开发者可基于此demo示例进行二次开发，满足集成的需求。


demo适用场景:
- 多GPU卡
- 单GPU卡多个模型（注意：此场景下，如果太多模型同时在一个GPU推理可能会导致每个模型的推理性能下降）
- 多线程CPU推理

**注意：**
- 多线程时不能频繁创建、销毁线程，否则会造成推理引擎的内存问题。可以使用线程池(`multi_thread_infer2.cpp`demo)或其它方式(`multi_thread_infer.cpp`demo)复用线程。


`multi_thread_infer.cpp`说明：
- 根据参数gpu_ids的个数，创建对应数量的实例
- 初始化各个实例(model.Init和model.PaddleEngineInit)
- 推理分 AddPredictTask 和 Predict 两种接口，根据需求选择其中一种即可
  - AddPredictTask接口：只需要提交任务(传入输入、输出), 各空闲线程会自动按队列顺序依次获取任务进行推理计算。通过返回的future.get()接口，确定结果计算完毕。
  - Predict 接口：传入batch大小图片，根据线程数将输入均摊到各线程进行计算， 最终等待所有线程计算完毕后合并各线程的计算错误。


`multi_thread_infer2.cpp`说明：
- 初始化n个model实例(modelx.Init和modelx.PaddleEngineInit)
- 初始化线程池(ThreadPool pool(n);pool.init();)
- 提交任务(pool.submit)， **注意：** 一个实例不能被两个线程同时使用
- futurex.get() 确认结果计算完毕后，处理结果resultx


## 步骤一、编译

参考编译文档

- [Linux系统上编译指南](../compile/paddle/linux.md)
- [Windows系统上编译指南](../compile/paddle/windows.md)

## 步骤二、准备PaddlePaddle部署模型

开发者可从以下套件获取部署模型，需要注意，部署时需要准备的是导出来的部署模型，一般包含`model.pdmodel`、`model.pdiparams`和`deploy.yml`三个文件，分别表示模型结构、模型权重和各套件自行定义的配置信息。

- [PaddleDetection导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/deploy/EXPORT_MODEL.md)
- [PaddleSeg导出模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0/docs/model_export.md)
- [PaddleClas导出模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/tutorials/getting_started.md#4-%E4%BD%BF%E7%94%A8inference%E6%A8%A1%E5%9E%8B%E8%BF%9B%E8%A1%8C%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)
- [PaddleX导出模型](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/export_model.md)



用户也可直接下载本教程中从PaddleDetection中导出的YOLOv3模型进行测试，[点击下载](https://bj.bcebos.com/paddlex/deploy2/models/yolov3_mbv1.tar.gz)。

## 步骤三、使用编译好的可执行文件预测

以步骤二中下载的YOLOv3模型为例，执行如下命令即可进行模型加载和预测

```
build/demo/multi_thread_infer --model_filename=yolov3_mbv1/model/model.pdmodel \
                       --params_filename=yolov3_mbv1/model/model.pdiparams \
                       --cfg_file=yolov3_mbv1/model/infer_cfg.yml \
                       --image=yolov3_mbv1/images/000000010583.jpg \
                       --gpu_id=0,1 \
                       --use_gpu=1 \
                       --model_type=det
```

### 参数说明

| 参数            | 说明                                                         |
| --------------- | ------------------------------------------------------------ |
| model_filename  | **[必填]** 模型结构文件路径，如`yolov3_darknet/model.pdmodel` |
| params_filename | **[必填]** 模型权重文件路径，如`yolov3_darknet/model.pdiparams` |
| cfg_file        | **[必填]** 模型配置文件路径，如`yolov3_darknet/infer_cfg.yml` |
| model_type      | **[必填]** 模型来源，det/seg/clas/paddlex，分别表示模型来源于PaddleDetection、PaddleSeg、PaddleClas和PaddleX |
| image      | 待预测的图片文件路径 |
| gpu_id          | 使用GPU预测时的GUI设备ID，默认为0                            |
| use_gpu         | 是否使用GPU，1或者0。默认为0，不使用GPU |
| thread_num      | 每张GPU卡上模型在图像预处理时的并行线程数，默认为1           |



## 相关文档

- [部署API和数据结构文档](../apis/model.md)
