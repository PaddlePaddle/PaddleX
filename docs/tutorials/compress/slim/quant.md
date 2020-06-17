# 模型量化

## 原理介绍
为了满足低内存带宽、低功耗、低计算资源占用以及低模型存储等需求，定点量化被提出。为此我们提供了训练后量化，该量化使用KL散度确定量化比例因子，将FP32模型转成INT8模型，且不需要重新训练，可以快速得到量化模型。


## 使用PaddleX量化模型
PaddleX提供了`export_quant_model`接口，让用户以接口的形式完成模型以post_quantization方式量化并导出。点击查看[量化接口使用文档](../apis/slim.md)。

## 量化性能对比
模型量化后的性能对比指标请查阅[PaddleSlim模型库](https://paddlepaddle.github.io/PaddleSlim/model_zoo.html)
