# Quick installation

By default, the following installation process supposes that you have installed paddlepaddle-gpu or paddlepaddle (version greater than or equal to 1.8.1)**. For the paddlepaddle installation method, refer to the official website of [PaddlePaddle] (https://www.paddlepaddle.org.cn/install/quick). 

## pip installation

Note that the pycocotools installation in Windows is special. Refer to the following installation command in Windows

```
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```

## Anaconda installation
Anaconda is an open source Python released version which contains more than 180 science packages and their dependencies such as conda and Python. By creating multiple independent Python environments, the use of Anaconda can avoid conflict due to too many different version dependencies installed in your Python environment.
- Refer to the [PaddleX document on Anaconda installation](appendix/anaconda_install.md)

## Code installation

The github codes will be constantly updated with the development progress

```
git clone https://github.com/PaddlePaddle/PaddleX.git 
cd PaddleX
git checkout develop
python setup.py install
```


## pycocotools installation problems

For the PaddleX dependency pycocotools package. If the pycocotools installation fails, install pycocotools by referring to the following method

### Windows system
* During installation in Windows, the message `Microsoft Visual C++ 14.0 is required` may be displayed, resulting in installation error. [Click to download and install VC build] tools and then execute the following pip command (https://go.microsoft.com/fwlink/?LinkId=691126)
> Note: After the installation is complete, you must reopen a new terminal command window

```
pip install cython
pip install git+https://gitee.com/jiangjiajun/philferriere-cocoapi.git#subdirectory=PythonAPI
```

### Linux/Mac system
* Directly use pip to install the following two dependencies in the Linux/Mac system

```
pip install cython
pip install pycocotools
```
