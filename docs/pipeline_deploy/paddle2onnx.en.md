# Installation and Usage of the Paddle2ONNX Plugin

The Paddle2ONNX plugin for PaddleX provides the ability to convert PaddlePaddle static models to ONNX format models, leveraging the underlying [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX).

## 1. Installation

```bash
paddlex --install paddle2onnx
```

## 2. Usage

### 2.1 Parameter Introduction

<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Type</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>paddle_model_dir</td>
            <td>str</td>
            <td>Directory containing the Paddle model.</td>
        </tr>
        <tr>
            <td>onnx_model_dir</td>
            <td>str</td>
            <td>Output directory for the ONNX model, which can be the same as the Paddle model directory. Defaults to <code>onnx</code>.</td>
        </tr>
        <tr>
            <td>opset_version</td>
            <td>int</td>
            <td>The ONNX opset version to use. Defaults to <code>7</code>.</td>
        </tr>
    </tbody>
</table>

### 2.2 Usage Method

Usage:

```bash
paddlex \
    --paddle2onnx \  # Use the paddle2onnx function
    --paddle_model_dir /your/paddle_model/dir \  # Specify the directory where the Paddle model is located
    --onnx_model_dir /your/onnx_model/output/dir \  # Specify the output directory for the converted ONNX model
    --opset_version 7  # Specify the ONNX opset version to use
```

Taking the ResNet18 model from the image_classification module as an example:

```bash
paddlex \
    --paddle2onnx \
    --paddle_model_dir ./ResNet18 \
    --onnx_model_dir ./ResNet18 \
```
