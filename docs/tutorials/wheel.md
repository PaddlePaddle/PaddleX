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
python setup.py install
```

### 1.2 安装 PaddleX 相关依赖

```bash
paddlex --install
```

## 2. 推理预测

### 2.1 使用 CLI 进行推理预测

以图像分类模型 `PP-LCNet_x1_0` 为例，使用inference模型文件（`output/best_model`）对图像（`/paddle/dataset/paddlex/cls/cls_flowers_examples/images/image_00002.jpg`）进行预测，命令如下：

```bash
paddlex --model_name PP-LCNet_x1_0 --model output/best_model --device gpu:0 --input_path /paddle/dataset/paddlex/cls/cls_flowers_examples/images/image_00002.jpg --output
```

可以得到预测结果：

```
image_00002.jpg:        class id(s): [76, 24, 70, 18, 14], score(s): [0.66, 0.05, 0.02, 0.02, 0.01], label_name(s): ['tarantula', 'great grey owl, great gray owl, Strix nebulosa', 'harvestman, daddy longlegs, Phalangium opilio', 'magpie', 'indigo bunting, indigo finch, indigo bird, Passerina cyanea']
```

### 2.2 使用 Python 进行推理预测

```python
import paddlex
paddlex.predict("PP-LCNet_x1_0", "output/best_model", "/paddle/dataset/paddlex/cls/cls_flowers_examples/images/image_00002.jpg", "gpu", "./output")
```
