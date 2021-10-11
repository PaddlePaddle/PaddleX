# 数据准备

PaddleX提供了全套数据标注流程，用户可先安装、启动标注软件并标注数据，再将数据集标注格式转换成PaddleX要求的格式，最后进行数据划分。
<p align="center">
  <img src="https://user-images.githubusercontent.com/53808988/135017525-4b59ee12-65b1-4d9e-b135-8e3cec1651e6.png" width="800"  />
</p>
通过执行以下步骤完成数据的准备：

**1 标注工具安装和启动**：参考文档[标注工具LabelMe的安装和启动](./annotation/labelme.md)安装标注工具。

**2 数据标注**：文档[数据标注](./annotation/README.md)示例了使用标注工具完成标注的步骤，并将各步骤以图片展示。

**3 数据格式转换**：因PaddleX对数据集的格式做了规定，使用LabelMe标注完数据后，需要将LabelMe的标注协议转换成PaddleX要求的格式。此外，PaddleX也支持精灵标注助手的协议转换。
- 参考文档[数据格式转换](./convert.md)将标注工具的标注协议转换成PaddleX要求的格式。
- 点击文档[数据格式说明](format/README.md)查看PaddleX对数据集格式的要求。

**4 数据划分**：点击文档[数据划分](./split.md)将数据集划分成训练集、验证集和测试集（可选），用于模型的训练和精度验证。
