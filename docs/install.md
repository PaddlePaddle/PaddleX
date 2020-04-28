# 安装

> 以下安装过程默认用户已安装好Anaconda和CUDA 10.1（有GPU卡的情况下）， Anaconda的安装可参考其官网https://www.anaconda.com/

## Linux/Mac安装
```
# 使用conda创建虚拟环境
conda create -n paddlex python=3.7
conda activate paddlex

# 安装paddlepaddle
# cpu版: pip install paddlepaddle
pip install paddlepaddle-gpu

# 安装cython
pip install cython

# 安装PaddleX
pip install paddlex
```

## Windows安装
```
# 使用conda创建虚拟环境
conda create -n paddlex python=3.7
conda activate paddlex

# 安装paddlepaddle
# cpu版: pip install paddlepaddle
pip install paddlepaddle-gpu

# 安装pycocotools
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

# 安装PaddleX
pip install paddlex
```
