# Model quantification

To meet the requirements of low memory, low bandwidth, low power consumption, low computing resource usage, and low model storage in end-side deployment scenarios, PaddleX is integrated with PaddleSlim to implement model quantification functions, and to further improve the Paddle Lite end-side deployment performance.

## Introduction to the principles
In the fixed-point quantification, fewer bits (such as 8-bit, 3-bit, 2-bit, etc) are used to represent the weight and activation value of the neural network, to accelerate the speed of model inference. PaddleX provides post-training quantification technology. For the principles, see the [post-training quantification principle](https://paddlepaddle.github.io/PaddleSlim/algo/algo.html#id14). The quantification uses KL divergence to determine the quantification scale factor, converts the FP32 model to the INT8 model. This does not need re-training. The quantification model can be obtained quickly.

## Use the PaddleX quantification model
PaddleX provides the `export_quant_model` interface, allowing users to quantify the trained model in the form of an interface. Click to view the [quantification interface document](../../../apis/slim.md).

## Quantification performance comparison
For the performance comparison indicators after model quantification, refer to the [PaddleSlim model library] (https://paddlepaddle.github.io/PaddleSlim/model_zoo.html)
