# 1. 贡献模型

飞桨生态的繁荣离不开开发者和用户的贡献，我们非常欢迎你为飞桨的多硬件适配贡献更多的模型，也十分感谢你的反馈。

当前PaddleX中对于各硬件适配模型的列表如下，您可以确认相关模型是否已经在对应硬件中进行过适配：

* [昇腾模型列表](../models/support_npu_model_list.md)

* [昆仑模型列表](../models/support_xpu_model_list.md)

* [海光模型列表](../models/support_dcu_model_list.md)

* [寒武纪模型列表](../models/support_mlu_model_list.md)

## 1.1 模型选择

当前PaddleX相关模型的源码放置在各个套件中，部分套件和模型并未接入PaddleX中，因此对模型进行适配前，请务必保证您的模型在PaddleX中已经接入，当前PaddleX模型列表详见 [PaddleX模型库](../models/support_model_list.md)。如果您有特殊的模型需求，请提交 [issue](https://github.com/PaddlePaddle/PaddleX/issues/new?assignees=&labels=&projects=&template=5_other.md&title=) 告知我们。

如果您适配的模型在相关硬件上涉及到模型组网代码的修改，请先提交代码到对应的套件中，参考各套件贡献指南：

1. https://github.com/PaddlePaddle/PaddleClas/tree/develop

2. https://github.com/PaddlePaddle/PaddleDetection/tree/develop

3. https://github.com/PaddlePaddle/PaddleSeg/tree/develop

4. https://github.com/PaddlePaddle/PaddleOCR/tree/develop

5. https://github.com/PaddlePaddle/PaddleTS

## 1.2 数据集选择

请确保您用于精度对齐的数据集是相应任务的标准数据集，如分类的ImageNet，检测的COCO/VOC等，务必确保相关模型在对应数据集上的训练精度可以对齐GPU模型。

# 2. 提交说明issue

当您完成特定硬件上某款模型的适配工作后，请给PaddleX提交一个 [issue](https://github.com/PaddlePaddle/PaddleX/issues/new?assignees=&labels=&projects=&template=6_hardware_contribute.md&title=) 说明相关信息，我们将会对模型进行验证，确认无问题后将合入相关代码并在文档中对模型列表进行更新

相关issue需要提供复现模型精度的信息，至少包含以下内容：

* 验证模型精度所用到的软件版本，包括但不限于：

  * Paddle版本

  * PaddleCustomDevice版本（如果有）

  * PaddleX或者对应套件的分支

* 验证模型精度所用到的机器环境，包括但不限于：

  * 芯片型号

  * 系统版本

  * 硬件驱动版本

  * 算子库版本等

* 用于说明训练精度的模型训练日志

* 详细的模型复现脚本，包含环境准备、模型训练、精度验证等

# 3. 更多文档

更多关于飞桨多硬件适配和使用的相关文档，可以参考

* [飞桨使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/index_cn.html)

* [飞桨硬件支持](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/hardware_support/index_cn.html)

* [PaddleX多硬件使用指南](../base/devices_use_guidance.md)

* [PaddleCustomDevice仓库](https://github.com/PaddlePaddle/PaddleCustomDevice)
