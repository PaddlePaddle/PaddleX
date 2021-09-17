# PaddleX API开发模式
通过简洁易懂的Python API，在兼顾功能全面性、开发灵活性、集成方便性的基础上，给开发者最流畅的深度学习开发体验。

## 快速安装
以下安装过程默认用户已安装好**paddlepaddle-gpu或paddlepaddle(版本大于或等于2.1.2)**，paddlepaddle安装方式参照[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/release/2.0.0/install/pip/windows-pip.html)

### PaddleX 2.0.0安装
**我们推荐大家先安装Anacaonda，而后在新建的conoda环境中使用上述pip安装方式**。Anaconda是一个开源的Python发行版本，其包含了conda、Python等180多个科学包及其依赖项。使用Anaconda可以通过创建多个独立的Python环境，避免用户的Python环境安装太多不同版本依赖导致冲突。参考[Anaconda安装PaddleX文档](./appendix/anaconda_install.md)

- Linux / macOS 操作系统

使用pip安装方式安装2.0.0版本：

```commandline
pip install paddlex==2.0.0 -i https://mirror.baidu.com/pypi/simple
```

paddlepaddle已集成pycocotools包，但也有pycocotools无法随paddlepaddle成功安装的情况。因PaddleX依赖pycocotools包，如遇到pycocotools安装失败，可参照如下方式安装pycocotools：

```commandline
pip install cython  
pip install pycocotools
```


- Windows 操作系统


使用pip安装方式安装2.0.0版本：

```commandline
pip install paddlex==2.0.0 -i https://mirror.baidu.com/pypi/simple
```

因PaddleX依赖pycocotools包，Windows安装时可能会提示`Microsoft Visual C++ 14.0 is required`，从而导致安装出错，[点击下载VC build tools](https://go.microsoft.com/fwlink/?LinkId=691126)安装再执行如下pip命令
> 注意：安装完后，需要重新打开新的终端命令窗口

```commandline
pip install cython
pip install git+https://gitee.com/jiangjiajun/philferriere-cocoapi.git#subdirectory=PythonAPI
```

### PaddleX develop安装

github代码会跟随开发进度不断更新，可以安装release/2.0.0分支的代码使用最新的功能，安装方式如下：

```commandline
git clone https://github.com/PaddlePaddle/PaddleX.git
git checkout develop
cd PaddleX
pip install -r requirements.txt
python setup.py install
```

如遇到pycocotools安装失败，参考[PaddleX 2.0.0安装](./install.md#paddlex-200安装)中介绍的解决方法。
