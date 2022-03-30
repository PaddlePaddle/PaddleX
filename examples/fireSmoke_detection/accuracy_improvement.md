# 精度优化思路分析

本小节侧重展示在模型迭代过程中优化精度的思路，在本案例中，有些优化策略获得了精度收益，而有些没有。在其他场景中，可根据实际情况尝试这些优化策略。

## (1) 基线模型选择

相较于二阶段检测模型，单阶段检测模型的精度略低但是速度更快。考虑到是部署到GPU端，本案例选择单阶段检测模型PP-YOLOv2作为基线模型，其骨干网络选择ResNet50_vd_dcn。训练完成后，模型在验证集上图片级别的召回率为95.1%，图片级别的误检率为23.22%。

【名词解释】

* 图片级别的召回率：只要在有目标的图片上检测出目标（不论框的个数），该图片被认为召回。批量有目标图片中被召回图片所占的比例，即为图片级别的召回率。
* 图片级别的误检率：只要在无目标的图片上检测出目标（不论框的个数），该图片被认为误检。批量无目标图片中被误检图片所占的比例，即为图片级别的误检率。

## (2) 基线模型效果分析与优化

| 模型               | Recall（图片级别的召回率） | Error Rate（图片级别的误检率） |
| ------------------ | -------------------------- | ------------------------------ |
| PP-YOLOv2+ResNet50 | 95.1                       | 23.22                          |

由于烟雾和火灾的特殊性，无法精准的计算mAP，因此本案例采用图片级别的召回率和图片级别的误检率作为最终指标。

从分析表格中可以看出，对于没有烟雾和火灾的负样本，存在较严重的误检情况，同时对于正样本也存在一定的漏检情况。因此，将验证集上的预测结果进行了可视化，然后发现当前数据存在以下问题：

- 负样本中存在非常多的目标与烟雾、火灾极其相似，比如：云朵、绒毛与烟雾十分相似；红色的花海、黄昏时的风景与火灾十分相似。这些难区分目标都会对最终的检测结果产生非常大的影响。
- 自然图像中的火焰和烟雾形状非常不规则，同时数据集中火灾和烟雾图片背景丰富、光线、拍摄距离也各不相同，造成最终漏检较多。

## (3) 数据增强选择

| 训练预处理                                                   | 验证预处理                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| MixupImage(mixup_epoch=-1)                                   | Resize(target_size=640, interp='CUBIC')                      |
| RandomDistort()                                              | Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| RandomExpand(im_padding_value=[123.675, 116.28, 103.53])     |                                                              |
| RandomCrop()                                                 |                                                              |
| RandomHorizontalFlip()                                       |                                                              |
| BatchRandomResize(target_sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768],interp='RANDOM') |                                                              |
| Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |                                                              |

在加入了[RandomHorizontalFlip](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#randomhorizontalflip)、[RandomDistort](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#randomdistort)、[RandomCrop](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#randomcrop)、[RandomExpand](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#randomexpand)、[BatchRandomResize](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#batchrandomresize)、[MixupImage](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#mixupimage)这几种数据增强方法后，对模型的优化器到了一定的积极作用，模型在验证集上图片级别的召回率为94.1%，图片级别的误检率为14.9%。。

**PS**：建议在训练初期都加上这些预处理方法，到后期模型超参数以及相关结构确定最优之后，再进行数据方面的再优化: 比如数据清洗，数据预处理方法筛选等。

## (4) 背景图片加入

本案例将数据集中提供的背景图片按9:1切分成了1116张、135张两部分，并使用(3)中训练好的模型在135张背景图片上进行测试，发现图片级误检率高达21.5%。为了降低模型的误检率，使用[paddlex.datasets.VOCDetection.add_negative_samples](https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#add-negative-samples)接口将1116张背景图片加入到原本的训练集中，重新训练后图片级误检率降低至4%。为了不让训练被背景图片主导，本案例通过将`train_list.txt`中的文件路径多写了一遍，从而增加有目标图片的占比。

| 模型                                          | 图片级召回率 | 图片级误检率 |
| --------------------------------------------- | ------------ | ------------ |
| PP-YOLOv2+ResNet50(Baseline)                  | 95.1         | 23.22        |
| PP-YOLOv2+ResNet50+aug+COCO预训练+SPP+背景图  | 93.9         | 1.1          |
| PP-YOLOv2+ResNet101+aug+COCO预训练+SPP+背景图 | **96**       | **2.2**      |

