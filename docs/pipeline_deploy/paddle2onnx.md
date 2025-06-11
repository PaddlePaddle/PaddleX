
# Paddle2ONNX 插件的安装与使用

PaddleX 的 Paddle2ONNX 插件提供了将飞桨静态图模型转化到 ONNX 格式模型的能力，底层使用 [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)。

## 1. 安装

```bash
# Windows 用户需使用以下命令安装 paddlepaddle dev版本
# python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# 安装 Paddle2ONNX 插件
paddlex --install paddle2onnx
```

## 2. 使用

### 2.1 参数介绍

<table>
    <thead>
        <tr>
            <th>参数</th>
            <th>类型</th>
            <th>描述</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>paddle_model_dir</td>
            <td>str</td>
            <td>包含 Paddle 模型的目录。</td>
        </tr>
        <tr>
            <td>onnx_model_dir</td>
            <td>str</td>
            <td>ONNX 模型的输出目录，可以与 Paddle 模型目录相同。默认为 <code>onnx</code>。</td>
        </tr>
        <tr>
            <td>opset_version</td>
            <td>int</td>
            <td>使用的 ONNX opset 版本。当使用低版本 opset 无法完成转换时，将自动选择更高版本的 opset 进行转换。默认为 <code>7</code>。</td>
        </tr>
    </tbody>
</table>

### 2.2 使用方式

使用方式：

```bash
paddlex \
    --paddle2onnx \  # 使用paddle2onnx功能
    --paddle_model_dir /your/paddle_model/dir \  # 指定 Paddle 模型所在的目录
    --onnx_model_dir /your/onnx_model/output/dir \  # 指定转换后 ONNX 模型的输出目录
    --opset_version 7  # 指定要使用的 ONNX opset 版本
```

以 image_classification 模块中的 ResNet18 模型为例：

```bash
paddlex \
    --paddle2onnx \
    --paddle_model_dir ./ResNet18 \
    --onnx_model_dir ./ResNet18 \
```
