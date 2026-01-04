# 1. Contributing Models

The prosperity of the PaddlePaddle ecosystem is inseparable from the contributions of developers and users. We warmly welcome you to contribute more models for multi-hardware compatibility to PaddlePaddle and greatly appreciate your feedback.

The current list of hardware-compatible models in PaddleX is as follows. You can check whether the relevant models have been adapted for the corresponding hardware:

* [NPU Model List](../support_list/model_list_npu.en.md)

* [XPU Model List](../support_list/model_list_xpu.en.md)

* [DCU Model List](../support_list/model_list_dcu.en.md)

* [MLU Model List](../support_list/model_list_mlu.en.md)

The source code for current PaddleX-related models is placed in various kits, and some kits and models have not been integrated into PaddleX. Therefore, before adapting a model, please ensure that your model is already integrated into PaddleX. For the current list of PaddleX models, see [PaddleX Model Library](../support_list/models_list.en.md). If you have specific model requirements, please submit an [issue](https://github.com/PaddlePaddle/PaddleX/issues/new?assignees=&labels=&projects=&template=6_hardware_contribute.en.md&title=) to inform us.

If the model you are adapting involves modifications to the model networking code on the relevant hardware, please submit the code to the corresponding kit first, referring to the contribution guides for each kit:

1. [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop)

2. [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/develop)

3. [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop)

4. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/tree/develop)

5. [PaddleTS](https://github.com/PaddlePaddle/PaddleTS/tree/main)

# 2. Submitting an Issue for Explanation

When you have completed the adaptation of a model on a specific hardware, please submit an [issue](https://github.com/PaddlePaddle/PaddleX/issues/new?assignees=&labels=&projects=&template=6_hardware_contribute.md&title=) to PaddleX explaining the relevant information. We will verify the model and, upon confirmation of no issues, merge the relevant code and update the model list in the documentation.

The relevant issue needs to provide information to reproduce the model's accuracy, including at least the following:

* The software versions used to verify the model's accuracy, including but not limited to:

  * Paddle version

  * PaddleCustomDevice version (if applicable)

  * Branch of PaddleX or the corresponding kit

* The machine environment used to verify the model's accuracy, including but not limited to:

  * Chip model

  * Operating system version

  * Hardware driver version

  * Operator library version, etc.

# 3. More Documentation

For more documentation related to multi-hardware compatibility and usage in PaddlePaddle, please refer to:

* [PaddlePaddle User Guide](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/guides/index_en.html)

* [PaddlePaddle Hardware Support](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/hardware_support/index_en.html)

* [PaddleX Multi-device Usage Guide](./multi_devices_use_guide.en.md)

* [PaddleCustomDevice Repository](https://github.com/PaddlePaddle/PaddleCustomDevice)
