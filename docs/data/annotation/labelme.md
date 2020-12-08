# Labelme Installation and Startup

LabelMe可用于标注目标检测、实例分割、语义分割数据集，是一款开源的标注工具。

## 1. Install Anaconda

推荐使用Anaconda安装python依赖，有经验的开发者可以跳过此步骤。安装Anaconda的方式可以参考[文档](../../appendix/anaconda_install.md)。

在安装Anaconda，并创建环境之后，再进行接下来的步骤

## 2. Install Labelme

进入Python环境后，执行如下命令即可
```
conda activate my_paddlex
conda install pyqt
pip install labelme
```

## 3. Start Labelme

进入安装了LabelMe的Python环境，执行如下命令即可启动LabelMe
```
conda activate my_paddlex
labelme
```
