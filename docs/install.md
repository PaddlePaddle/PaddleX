# 快速安装PaddleX

## 目录

* [1. PaddleX API开发模式安装](#1)
* [2. PadldeX GUI开发模式安装](#2)
* [3. PaddleX Restful开发模式安装](#3)


**PaddleX提供三种开发模式，满足用户的不同需求：**

## <h2 id="1">1. PaddleX API开发模式安装</h2>

通过简洁易懂的Python API，在兼顾功能全面性、开发灵活性、集成方便性的基础上，给开发者最流畅的深度学习开发体验。<br>

以下安装过程默认用户已安装好**paddlepaddle-gpu或paddlepaddle(版本大于或等于2.1.0)**，paddlepaddle安装方式参照[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html)


### PaddleX 2.0.0安装

#### * Linux / macOS 操作系统

使用pip安装方式安装2.0.0版本：

```commandline
pip install paddlex==2.0.0rc4 -i https://mirror.baidu.com/pypi/simple
```

因PaddleX依赖pycocotools包，如遇到pycocotools安装失败，可参照如下方式安装pycocotools：

```commandline
pip install cython  
pip install pycocotools
```

**我们推荐大家先安装Anacaonda，而后在新建的conoda环境中使用上述pip安装方式**。Anaconda是一个开源的Python发行版本，其包含了conda、Python等180多个科学包及其依赖项。使用Anaconda可以通过创建多个独立的Python环境，避免用户的Python环境安装太多不同版本依赖导致冲突。参考[Anaconda安装PaddleX文档](./appendix/anaconda_install.md)

#### * Windows 操作系统


使用pip安装方式安装2.0.0版本：

```commandline
pip install paddlex==2.0.0rc4 -i https://mirror.baidu.com/pypi/simple
```

因PaddleX依赖pycocotools包，Windows安装时可能会提示`Microsoft Visual C++ 14.0 is required`，从而导致安装出错，[点击下载VC build tools](https://go.microsoft.com/fwlink/?LinkId=691126)安装再执行如下pip命令
> 注意：安装完后，需要重新打开新的终端命令窗口

```commandline
pip install cython
pip install git+https://gitee.com/jiangjiajun/philferriere-cocoapi.git#subdirectory=PythonAPI
```

### PaddleX develop安装

github代码会跟随开发进度不断更新，可以安装develop分支的代码使用最新的功能，安装方式如下：

```commandline
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
pip install -r requirements.txt
python setup.py install
```

如遇到pycocotools安装失败，参考[PaddleX 2.0.0安装](./install.md#paddlex-200安装)中介绍的解决方法。

## <h2 id="2">2. PadldeX GUI开发模式安装</h2>


   无代码开发的可视化客户端，应用PaddleX API实现，使开发者快速进行产业项目验证，并为用户开发自有深度学习软件/应用提供参照。

- 前往[PaddleX官网](https://www.paddlepaddle.org.cn/paddle/paddlex)，申请下载PaddleX GUI一键绿色安装包。

- 前往[PaddleX GUI使用教程](./gui/how_to_use.md)了解PaddleX GUI使用详情。

- [PaddleX GUI安装环境说明](./gui/download.md)


## <h2 id="3">3. PaddleX Restful开发模式安装</h2>

使用基于RESTful API开发的GUI与Web Demo实现远程的深度学习全流程开发；同时开发者也可以基于RESTful API开发个性化的可视化界面
- 前往[PaddleX RESTful API使用教程](./Resful_API/docs/readme.md)  
