# 使用 PaddleX wheel 进行推理预测

## 1. 安装

### 1.1 安装 PaddleX whl

1. 安装官方版本

```bash
pip install paddlex
```

2. 从源码编译安装

```bash
cd PaddleX
pip install .
```

### 1.2 安装 PaddleX 相关依赖

```bash
paddlex --install
```

## 2. 推理预测

### 2.1 使用 CLI 进行推理预测

以图像分类模型 `PP-LCNet_x1_0` 为例，使用 PaddleX 预置的官方模型对图像（`/paddle/dataset/paddlex/cls/cls_flowers_examples/images/image_00002.jpg`）进行预测，命令如下：

```bash
paddlex --pipeline image_classification --model PP-LCNet_x1_0 --input /paddle/dataset/paddlex/cls/cls_flowers_examples/images/image_00006.jpg
```

可以得到预测结果：

```
[{'class_ids': [309], 'scores': [0.19514], 'label_names': ['bee']}]
```

以 OCR 为例，使用PaddleX 预置的 `PP-OCRv4_mobile_det` 和 `PP-OCRv4_mobile_rec` 官方模型，对图像（`/paddle/dataset/paddlex/ocr_det/ocr_det_dataset_examples/images/train_img_100.jpg`）进行预测，命令如下：

```bash
paddlex --pipeline ocr --model PP-OCRv4_mobile_det PP-OCRv4_mobile_rec --input /paddle/dataset/paddlex/ocr_det/ocr_det_dataset_examples/images/train_img_100.jpg  --output ./
```

可以在当前目录下得到预测结果示例图 `ocr_result.jpg`。


### 2.2 使用 Python 进行推理预测

```python
import paddlex

model_name = "PP-LCNet_x1_0"

kernel_option = paddlex.PaddleInferenceOption()
kernel_option.set_device("gpu")

model = paddlex.create_model(model_name, kernel_option=kernel_option)
model.predict("/paddle/dataset/paddlex/cls/cls_flowers_examples/images/image_00002.jpg")
```
