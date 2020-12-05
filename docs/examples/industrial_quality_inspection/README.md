# 工业质检

本案例面向工业质检场景里的铝材表面缺陷检测，提供了针对GPU端和CPU端两种部署场景下基于PaddleX的解决方案，希望通过梳理优化模型精度和性能的思路能帮助用户更高效地解决实际质检应用中的问题。

## 1. GPU端解决方案

### 1.1 数据集介绍

本案例使用[天池铝材表面缺陷检测初赛](https://tianchi.aliyun.com/competition/entrance/231682/introduction)数据集，共有3005张图片，分别检测擦花、杂色、漏底、不导电、桔皮、喷流、漆泡、起坑、脏点和角位漏底10种缺陷，这10种缺陷的定义和示例可点击文档[天池铝材表面缺陷检测初赛数据集示例](./dataset.md)查看。

将这3005张图片按9:1随机切分成2713张图片的训练集和292张图片的验证集。

### 1.2 精度优化

本小节侧重展示在模型迭代过程中优化精度的思路，在本案例中，有些优化策略获得了精度收益，而有些没有。在其他质检场景中，可根据实际情况尝试这些优化策略。点击文档[精度优化](./accuracy_improvement.md)查看。

### 1.3 性能优化

在完成模型精度优化之后，从以下两个方面对模型进行加速：

#### (1) 减少FPN部分的通道数量

将FPN部分的通道数量由原本的256减少至64，使用方式在定义模型[FasterRCNN](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-fasterrcnn)类时设置参数`fpn_num_channels`为64即可，需要重新对模型进行训练。

#### (2) 减少测试阶段的候选框数量

将测试阶段RPN部分做非极大值抑制计算的候选框数量由原本的6000减少至500，将RPN部分做完非极大值抑制后保留的候选框数量由原本的1000减少至300。使用方式在定义模型[FasterRCNN](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-fasterrcnn)类时设置参数`test_pre_nms_top_n`为500，`test_post_nms_top_n`为300。

采用Fluid C++预测引擎在Tesla P40上测试模型的推理时间（输入数据拷贝至GPU的时间、计算时间、数据拷贝至CPU的时间），输入大小设置为800x1333，加速前后推理时间如下表所示：

| 模型 | 推理时间 （ms/image）| VOC mAP (%) |
| -- | -- | -- |
| baseline | 66.51 | 88.87 |
| + fpn channel=64 + test proposal=pre/post topk 500/300 | 46.08 | 87.72 |

### 1.4 最终方案

本案例面向GPU端的最终方案是选择二阶段检测模型FasterRCNN，其骨干网络选择加入了可变形卷积（DCN）的ResNet50_vd，训练时使用SSLD蒸馏方案训练得到的ResNet50_vd预训练模型，FPN部分的通道数量设置为64。使用复核过的数据集，训练阶段数据增强策略采用RandomHorizontalFlip、RandomDistort、RandomCrop，并加入背景图片。测试阶段的RPN部分做非极大值抑制计算的候选框数量由原本的6000减少至500、做完非极大值抑制后保留的候选框数量由原本的1000减少至300。模型在验证集上的VOC mAP为87.72%。

在Tesla P40的Linux系统下，对于输入大小是800 x 1333的模型，图像预处理时长为30ms/image，模型的推理时间为46.08ms/image，包括输入数据拷贝至GPU的时间、计算时间、数据拷贝至CPU的时间。

| 模型 | VOC mAP (%) | 推理时间 (ms/image)
| -- | -- | -- |
| FasterRCNN-ResNet50_vd_ssld | 81.05 | 48.62 |
| + dcn | 88.09 | 66.51 |
| + RandomHorizontalFlip/RandomDistort/RandomCrop | 90.23| 66.51 |
| + background images | 88.87 | 66.51 |
| + fpn channel=64 | 87.79 | 48.65 |
| + test proposal=pre/post topk 500/300 | 87.72 | 46.08 |

具体的训练和部署流程点击文档[GPU端最终解决方案](./gpu_solution.md)进行查看。

## 2. CPU端解决方案

为了实现高效的模型推理，面向CPU端的模型选择精度和效率皆优的单阶段检测模型YOLOv3，骨干网络选择基于PaddleClas中SSLD蒸馏方案训练得到的MobileNetv3_large。训练完成后，对模型做剪裁操作，以提升模型的性能。模型在验证集上的VOC mAP为79.02%。

部署阶段，借助OpenVINO预测引擎完成在Intel(R) Core(TM) i9-9820X CPU @ 3.30GHz Windows系统下高效推理。对于输入大小是608 x 608的模型，图像预处理时长为38.69 ms/image，模型的推理时间为34.50ms/image，

| 模型 | VOC mAP (%) | Inference Speed (ms/image)
| -- | -- | -- |
| YOLOv3-MobileNetv3_ssld | 78.52 | 56.71 |
| pruned YOLOv3-MobileNetv3_ssld | 79.02 | 34.50 |

### 模型训练

[环境前置依赖](./gpu_solution.md#%E5%89%8D%E7%BD%AE%E4%BE%9D%E8%B5%96)、[下载PaddleX源码](./gpu_solution.md#1-%E4%B8%8B%E8%BD%BDpaddlex%E6%BA%90%E7%A0%81)、[下载数据集](./gpu_solution.md#2-%E4%B8%8B%E8%BD%BD%E6%95%B0%E6%8D%AE%E9%9B%86)与GPU端是一样的，可点击文档[GPU端最终解决方案](./gpu_solution.md)查看，在此不做赘述。

如果不想再次训练模型，可以直接下载已经训练好的模型完成后面的模型测试和部署推理：

```
wget https://bj.bcebos.com/paddlex/examples/industrial_quality_inspection/models/yolov3_mobilenetv3_large_pruned.tar.gz
tar xvf yolov3_mobilenetv3_large_pruned.tar.gz
```

运行以下代码进行模型训练，代码会自动下载数据集，如若事先下载了数据集，需将下载和解压铝材缺陷检测数据集的相关行注释掉。代码中默认使用0,1,2,3,4,5,6,7号GPU训练，可根据实际情况设置卡号并调整`batch_size`和`learning_rate`。

```
python train_yolov3.py
```

### 模型剪裁

运行以下代码，分析在不同的精度损失下模型各层的剪裁比例：

```
python params_analysis.py
```

设置可允许的精度损失为0.05，对模型进行剪裁，剪裁后需要重新训练模型：

```
python train_pruned_yolov3.py
```

[分析预测错误的原因](./gpu_solution.md#4-%E5%88%86%E6%9E%90%E9%A2%84%E6%B5%8B%E9%94%99%E8%AF%AF%E7%9A%84%E5%8E%9F%E5%9B%A0)、[统计图片级召回率和误检率](./gpu_solution.md#5-%E7%BB%9F%E8%AE%A1%E5%9B%BE%E7%89%87%E7%BA%A7%E5%8F%AC%E5%9B%9E%E7%8E%87%E5%92%8C%E8%AF%AF%E6%A3%80%E7%8E%87)、[模型测试](./gpu_solution.md#6-%E6%A8%A1%E5%9E%8B%E6%B5%8B%E8%AF%95)这些步骤与GPU端是一样的，可点击文档[GPU端最终解决方案](./gpu_solution.md)查看，在此不做赘述。

### 推理部署

本案例采用C++部署方式，通过OpenVINO将模型部署在Intel(R) Core(TM) i9-9820X CPU @ 3.30GHz的Windows系统下，具体的部署流程请参考文档[PaddleX模型多端安全部署/OpenVINO部署](https://paddlex.readthedocs.io/zh_CN/develop/deploy/openvino/index.html)。
