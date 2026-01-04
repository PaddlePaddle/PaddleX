---
comments: true
---

# Distributed Training

## Introduction

Distributed training refers to splitting a training task across multiple computing nodes according to certain methods, and then aggregating and updating the gradients and other information obtained from the split computations. PaddlePaddle’s distributed training technology originates from Baidu’s business practices and has been validated in ultra-large-scale business scenarios in fields such as natural language processing, computer vision, search, and recommendation. High-performance distributed training is one of PaddlePaddle’s core technical advantages. For example, in tasks such as image classification, distributed training can achieve nearly linear speedup. Take ImageNet as an example, the ImageNet22k dataset contains 14 million images, and training on a single GPU would be extremely time-consuming. Therefore, PaddleX supports distributed training interfaces to complete training tasks, supporting both single-machine and multi-machine training. For more methods and documentation on distributed training, please refer to:[Distributed Training Quick Start Tutorial](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/ps_quick_start.html)。


## Usage

* Taking [Image Classification Model Training](../tutorials/cv_modules/image_classification.en.md)as an example, compared to single-machine training, for multi-machine training you only need to add the `Train.dist_ips` parameter, which indicates the list of IP addresses of machines participating in distributed training, separated by commas. Below is a sample code to run.

```
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cls_flowers_examples
    -o Train.dist_ips="xx.xx.xx.xx,xx.xx.xx.xx"
```
**Note**：

- The IP addresses of different machines should be separated by commas and can be checked using `ifconfig` or `ipconfig`.
- Passwordless SSH should be set up between different machines, and they should be able to ping each other directly; otherwise, communication cannot be completed.
- The code, data, and execution commands or scripts must be consistent across all machines, and the training command or script must be run on all machines. Finally, the first device of the first machine in the `Train.dist_ips` list will be trainer0, and so on.
