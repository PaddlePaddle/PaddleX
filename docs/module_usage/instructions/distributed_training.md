---
comments: true
---

# 分布式训练

## 简介

分布式训练指的是将训练任务按照一定方法拆分到多个计算节点进行计算，再按照一定的方法对拆分后计算得到的梯度等信息进行聚合与更新。飞桨分布式训练技术源自百度的业务实践，在自然语言处理、计算机视觉、搜索和推荐等领域经过超大规模业务检验。分布式训练的高性能，是飞桨的核心优势技术之一，例如在图像分类等任务上，分布式训练可以达到几乎线性的加速比，以ImageNet为例，ImageNet22k数据集中包含1400W张图像，如果使用单卡训练，会非常耗时。因此PaddleX中支持使用分布式训练接口完成训练任务，同时支持单机训练与多机训练。更多关于分布式训练的方法与文档可以参考：[分布式训练快速开始教程](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/ps_quick_start.html)。


## 使用方法

* 以[图像分类模型训练](../tutorials/cv_modules/image_classification.md)为例，相比单机训练，多机训练时，只需要添加 `Train.dist_ips` 的参数，该参数表示需要参与分布式训练的机器的ip列表，不同机器的ip用逗号隔开。下面为运行代码示例。

```
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
    -o Train.dist_ips="xx.xx.xx.xx,xx.xx.xx.xx"
```
**注**：

- 不同机器的ip信息需要用逗号隔开，可以通过 `ifconfig` 或者 `ipconfig` 查看。
- 不同机器之间需要做免密设置，且可以直接ping通，否则无法完成通信。
- 不同机器之间的代码、数据与运行命令或脚本需要保持一致，且所有的机器上都需要运行设置好的训练命令或者脚本。最终 `Train.dist_ips` 中的第一台机器的第一块设备是trainer0，以此类推。