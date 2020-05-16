# Anaconda安装使用
Anaconda是一个开源的Python发行版本，其包含了conda、Python等180多个科学包及其依赖项。使用Anaconda可以通过创建多个独立的Python环境，避免用户的Python环境安装太多不同版本依赖导致冲突。

## Windows安装Anaconda
### 第一步 下载
在Anaconda官网[(https://www.anaconda.com/products/individual)](https://www.anaconda.com/products/individual)选择下载Windows Python3.7 64-Bit版本

### 第二步 安装
运行下载的安装包(以.exe为后辍)，根据引导完成安装, 用户可自行修改安装目录（如下图）
![](../images/anaconda_windows.png)

### 第三步 使用
- 点击Windows系统左下角的Windows图标，打开：所有程序->Anaconda3/2（64-bit）->Anaconda Prompt  
- 在命令行中执行下述命令
```cmd
# 创建名为my_paddlex的环境，指定Python版本为3.7
conda create -n my_paddlex python=3.7
# 进入my_paddlex环境
conda activate my_paddlex
# 安装git
conda install git
# 安装pycocotools
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
# 安装paddlepaddle-gpu
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
# 安装paddlex
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```  
按如上方式配置后，即可在环境中使用PaddleX了，命令行输入`python`回车后，`import paddlex`试试吧，之后再次使用都可以通过打开'所有程序->Anaconda3/2（64-bit）->Anaconda Prompt'，再执行`conda activate my_paddlex`进入环境后，即可再次使用paddlex

## Linux/Mac安装

### 第一步 下载
在Anaconda官网[(https://www.anaconda.com/products/individual)](https://www.anaconda.com/products/individual)选择下载对应系统 Python3.7版本下载（Mac下载Command Line Installer版本即可)

### 第二步 安装
打开终端，在终端安装Anaconda
```
# ~/Downloads/Anaconda3-2019.07-Linux-x86_64.sh即下载的文件
bash ~/Downloads/Anaconda3-2019.07-Linux-x86_64.sh
```
安装过程中一直回车即可，如提示设置安装路径，可根据需求修改，一般默认即可。

### 第三步 使用
```
# 创建名为my_paddlex的环境，指定Python版本为3.7
conda create -n my_paddlex python=3.7
# 进入paddlex环境
conda activate my_paddlex
# 安装pycocotools
pip install cython
pip install pycocotools
# 安装paddlepaddle-gpu
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
# 安装paddlex
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```
按如上方式配置后，即可在环境中使用PaddleX了，终端输入`python`回车后，`import paddlex`试试吧，之后再次使用只需再打开终端，再执行`conda activate my_paddlex`进入环境后，即可使用paddlex
