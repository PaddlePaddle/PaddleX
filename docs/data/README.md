# 数据准备

PaddleX提供了全套数据标注流程，用户可先安装、启动标注软件并标注数据，再将数据集标注格式转换成PaddleX要求的格式，最后进行数据划分。

因PaddleX对数据集的格式做了规定，使用LabelMe标注完数据后，需要将LabelMe的标注协议转换成PaddleX要求的格式。此外，PaddleX也支持精灵标注助手的协议转换。

在完成标注协议转换后，还需要将数据集划分出训练集和验证集用于模型的训练和精度验证。

通过执行以下步骤完成数据的准备：

### [1 数据格式说明](format/README.md)

点击文档[数据格式说明](format/README.md)查看PaddleX对数据集格式的要求。

### [2 标注工具LabelMe的安装和启动](./annotation/labelme.md)

参考文档[标注工具LabelMe的安装和启动](./annotation/labelme.md)安装标注工具。

### [3 数据标注](./annotation/README.md)

文档[数据标注](./annotation/README.md)示例了使用标注工具完成标注的步骤，并将各步骤以图片展示。

### [4 数据格式转换](./convert.md)

参考文档[数据格式转换](./convert.md)将标注工具的标注协议转换成PaddleX要求的格式。

### [5 数据划分](./split.md)

点击文档[数据划分](./split.md)将数据集划分成训练集、验证集和测试集（可选），用于模型的训练和精度验证。
