# Precision Optimization

This section focuses on the idea of optimizing the accuracy in the process of model iteration. In this case, some optimization strategies have achieved accuracy gains, while others have not. In other quality inspection scenarios, these optimization strategies can be tried according to the actual situation.

## (1) Baseline model selection

Compared with the single-stage detection model, the two-stage detection model has higher accuracy but slower speed.Considering that it is deployed to the GPU, this case selects the two-stage detection model FasterRCNN as the baseline model, and its backbone network selects ResNet50_vd, and uses the ResNet50_vd pre training model trained based on SSLD distillation scheme in PaddleClas (The Top1 Acc on ImageNet1k verification set is 82.39%). After training, the accuracy of the model on the verification set VOC mAP is 73.36%.

## (2) Model effect analysis

The [paddlex.det.coco_error_analysis](https://paddlex.readthedocs.io/zh_CN/develop/apis/visualize.html#paddlex-det-coco-error-analysis) interface provided by PaddleX is used to analyze the causes of the prediction error of the model on the verification set. The analysis results are shown in the form of chart as follows:

| All classes | Abrasion mark | Variegated | Leakage | Non-conductive | Orange peel | Jet | Lacquer bubble | Crater formation | Dirty spot | Corner leakage |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| ![](https://bj.bcebos.com/paddlex/examples/industrial_quality_inspection/datasets/allclasses_analysis_example.png) | ![](https://bj.bcebos.com/paddlex/examples/industrial_quality_inspection/datasets/cahua_analysis_example.png) | ![](https://bj.bcebos.com/paddlex/examples/industrial_quality_inspection/datasets/zase_analysis_example.png) | ![](https://bj.bcebos.com/paddlex/examples/industrial_quality_inspection/datasets/loudi_analysis_example.png) | ![](https://bj.bcebos.com/paddlex/examples/industrial_quality_inspection/datasets/budaodian_analysis_example.png) | ![](https://bj.bcebos.com/paddlex/examples/industrial_quality_inspection/datasets/jupi_analysis_example.png) | ![](https://bj.bcebos.com/paddlex/examples/industrial_quality_inspection/datasets/penliu_analysis_example.png) | ![](https://bj.bcebos.com/paddlex/examples/industrial_quality_inspection/datasets/qipao_analysis_example.png) | ![](https://bj.bcebos.com/paddlex/examples/industrial_quality_inspection/datasets/qikeng_analysis_example.png) | ![](https://bj.bcebos.com/paddlex/examples/industrial_quality_inspection/datasets/zangdian_analysis_example.png) | ![](https://bj.bcebos.com/paddlex/examples/industrial_quality_inspection/datasets/jiaoweiloudi_analysis_example.png) |

The analysis chart shows seven Precision Recall (PR) curves. Each curve represents a higher Average Precision (AP) than the one on the left because of the gradual relaxation of evaluation requirements. For example, the evaluation requirements of each PR curve are explained as follows:

* C75: When IoU is set to 0.75, the AP is 0.001.
* C50: When IoU is set to 0.5, the AP is 0.622. The white area between C50 and C75 represents the AP gain brought about by relaxing IoU from 0.75 to 0.5.
* Loc: When IoU is set to 0.1, the PR curve is 0.740. The blue area between Loc and C50 represents the AP gain brought about by relaxing IOU from 0.5 to 0.1. The larger the area of the blue area, the more detection frames are not accurate enough.
* Sim: On the basis of Loc, if the detection frame and the truth value box are not the same in category, but both belong to the same subclass, then the detection frame is not considered to be wrong. Under this evaluation requirement, the PR curve, AP is 0.742. The larger the red area between Sim and Loc, the higher the degree of confusion between subclasses. All categories of VOC data set belong to the same subclass.
* Oth: On the basis of Sim, if the subclass of the detection frame and the truth value box are not the same, then the detection frame is not considered to be wrong. Under this evaluation requirement, the PR curve, AP is 0.742. The larger the green area between Oth and Sim, the higher the degree of confusion between subclasses. All the categories in the VOC format dataset belong to the same subclass, so there is no confusion between subclasses.
* BG: On the basis of Oth, the detection frame on the background area is not considered to be wrong. Under this evaluation requirement, the PR curve, AP is 92.1. The larger the area of purple area between BG and Oth, the more false detection of background area is.
* FN: On the basis of BG, the missing truth box is not considered to be wrong. Under this evaluation requirement, the PR curve, AP is 1.00. The larger the area of orange area between FN and BG, the more true value boxes are missed.

From the analysis chart, it can be seen that the detection effect of three kinds of variegated, orange peel and pit is better. There are a few detection frames that fail to reach IoU 0.5 in the corner leakage bottom, and the problems are mainly scratching, non-conductive, spouting, paint bubble and dirty spots. The most serious problems of scratch type are false detection, inaccurate position and missed inspection. The most serious problems of non-conductive class are missed inspection and inaccurate position. The most serious problems of jet flow and paint bubble are inaccurate location and false detection. The most serious problems of dirty spots are false detection and missed inspection. In order to further understand the causes of these problems, the prediction results on the validation set are visualized, and the following problems are found in the annotation of datasets:

* Minor defects are not regarded as defects, but the definition of minor defects is not clear. Some minor defects are marked, resulting in more false inspection
* The appearance of non-conductive, bottom leakage and corner leakage are very similar, which is difficult to distinguish by naked eyes, which leads to the confusion of these three types, which leads to the occurrence of false inspection or missed inspection in the evaluation
* Some slight scratches and dirty spots have been marked, while some obvious ones have not been marked, resulting in serious false and missed detection of these two types
* Spray and paint bubble are mostly continuous defects. There will be other jets and a row of bubbles on the horizontal line of one jet. However, sometimes these continuous defects are labeled as a target, and sometimes different parts are labeled separately. Sometimes the model detects individual parts, sometimes it detects the whole, resulting in the inaccuracy of the two types of positions and more false detection.

## (3) Data review

In order to reduce the impact of the original data annotation problems on the model optimization, it is necessary to review the data. Examples of review criteria are as follows:

* Abrasion：The obscure abrasions are not marked, and the surface ones are represented by the same frame, while the strip ones are represented by a frame
* Bottom leakage, corner leakage and non-conductive: they are too similar and belong to the same category
* Orange peel: ignore surfaces that are not large particles
* Jet: if it is obviously a jet, use one box. If not, use multiple boxes
* Paint bubble: don't mark a single point. Mark a box with a series of dots
* Dirty spots: ignore slight stains

After rechecking and relabeling the dataset, FasterRCNN-ResNet50_vd_ssld was retrained on the training set. The VOC mAP of the model on the verification set was 81.05%.

## (4) Deformable convolution join

Due to the irregular shape of spray and paint bubble, many prediction frames of these two types are not accurate. In order to solve this problem, we choose to use deformable convolution (DCN) in the backbone network ResNet50_vd. After retraining, the VOC mAP of the model on the verification set was 88.09%, the VOC AP of jet flow was increased from 57.3% to 78.7%, and the VOC AP of paint bubble was increased from 74.7% to 96.7%.

## (5) Data enhancement options

On the basis of (4), we choose to add some data enhancement strategies to further improve the accuracy of the model. In this case, we choose to use [RandomHorizontalFlip](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#randomhorizontalflip)、[RandomDistort](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#randomdistort)、[RandomCrop](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#randomcrop) at the same time, and the VOC mAP of the retrained model on the verification set is 90.23%.

In addition, data enhancement methods that can be tried include [MultiScaleTraining](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#resizebyshort)、[RandomExpand](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#randomexpand). In the case of aluminum surface defect detection data set, the scale of the same category does not change much. Using MultiScaleTraining or RandomExpand will change the original data distribution. The VOC mAP of the model trained by RandomHorizontalFlip + RandomDistort + RandomCrop + MultiScaleTraining is 87.15% on the verification set, and 88.56% on the verification set by using the RandomHorizontalFlip + RandomDistort + RandomCrop + RandomExpand.

## (6) Background image added

In this case, the background images provided in the data set are divided into 1116 and 135 images according to the ratio of 9:1. The model trained in (5) is used to test 135 background images. It is found that the false detection rate of picture level is as high as 21.5%. In order to reduce the false detection rate of the model, 1116 background images are added to the original training set using [paddlex.datasets.VOCDetection.add_negative_samples](https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#add-negative-samples) interface. After retraining, the image level false detection rate is reduced to 4%. In order not to let the training be dominated by the background image, this case has written the file path in `train_list.txt` more once, so as to increase the proportion of target pictures.

| Models | VOC mAP (%) | Defective picture level recall rate | Background image level false detection rate |
| -- | -- | -- | -- |
| FasterRCNN-ResNet50_vd_ssld + DCN + RandomHorizontalFlip + RandomDistort + RandomCrop | 90.23 | 95.5 | 21.5 |
| FasterRCNN-ResNet50_vd_ssld + DCN + RandomHorizontalFlip + RandomDistort + RandomCrop + Background image | 88.87 | 95.2 | 4 |

【DEFINITION】

* Picture level recall rate: As long as a target is detected on a picture with a target (regardless of the number of frames), the picture is considered to be recalled. The proportion of recalled pictures in batch of target pictures is the recall rate of picture level.
* Image level false detection rate: As long as the target is detected on the image without target (regardless of the number of frames), the image is considered as false detection. The proportion of mistakenly detected pictures in the batch of no target pictures is the false detection rate of picture level.

## (7) Selection of classification loss function

In addition to the background image mentioned in (6), the classification loss function of RPN can be selected as`SigmoidFocalLoss`. More anchors can be added to the training to increase the proportion of hard to distinguish samples in the loss function, thus reducing the false detection rate. When defining the model [FasterRCNN](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-fasterrcnn) class, set the parameter`rpn_cls_loss`to' SigmoidFocalLoss'. At the same time, the settings of parameters `rpn_focal_loss_alpha`、`rpn_focal_loss_gamma`、`rpn_batch_size_per_im`、`rpn_fg_fraction` need to be adjusted.

## (8) Selection of loss function in position regression

In addition to 'SmoothL1Loss', the location regression loss function of RCNN can also choose' CIoULoss', which can be used by setting the parameter `rcnn_bbox_loss` when defining the model [FasterRCNN](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-fasterrcnn) class. In this case, choosing 'CIoULoss' does not bring precision benefits, so we still choose' SmoothL1Loss'. 其他质检场景下，也可尝试使用'CIoULoss'。

## (9) Selection of positive and negative sampling methods

当目标物体的区域只占图像的一小部分时，可以考虑采用[LibraRCNN](https://arxiv.org/abs/1904.02701)中提出的IoU-balanced Sampling采样方式来获取更多的难分负样本。使用方式在定义模型[FasterRCNN](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-fasterrcnn)类时将参数`bbox_assigner`设置为'LibraBBoxAssigner'即可。

## (10) Contrast enhancement in pretreatment

工业界常用灰度相机采集图片，会存在目标与周围背景对比度不明显而无法被检测出的情况。在这种情况下，可以在定义预处理的时候使用[paddlex.det.transforms.CLAHE](https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html#clahe)对灰度图像的对比度进行增强。

灰度图：

![](../../../examples/industrial_quality_inspection/image/before_clahe.png)

对比度增加后的灰度图:

![](../../../examples/industrial_quality_inspection/image/after_clahe.png) |

## (11) Sample generation

对于数量较少的类别或者小目标，可以通过把这些目标物体粘贴在背景图片上来生成新的图片和标注文件，并把这些新的样本加入到训练中从而提升模型精度。目前PaddleX提供了实现该功能的接口，详细见[paddlex.det.paste_objects](https://paddlex.readthedocs.io/zh_CN/develop/apis/tools.html#paddlex-det-paste-objects)，需要注意的是，前景目标颜色与背景颜色差异较大时生成的新图片才会比较逼真。
