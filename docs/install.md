# 快速安装

以下安装过程默认用户已安装好**paddlepaddle-gpu或paddlepaddle(版本大于或等于1.7.1)**，paddlepaddle安装方式参照[飞桨官网](https://www.paddlepaddle.org.cn/install/quick)

> 推荐使用Anaconda Python环境，Anaconda下安装PaddleX参考文档[Anaconda安装使用](../appendix/anaconda_install.md)

## Github代码安装
github代码会跟随开发进度不断更新

> 注意其中pycocotools在Windows安装较为特殊，可参考下面的Windows安装命令  

```
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
git checkout develop
python setup.py install
```

## pip安装
```
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```

## 安装问题
### 1. pycocotools安装问题  
> PaddleX依赖pycocotools包，如安装pycocotools失败，可参照如下方式安装pycocotools

**Windows**  
> Windows安装时可能会提示缺少`Microsoft Visual C++ 2015 build tools`，[点击下载](https://go.microsoft.com/fwlink/?LinkId=691126)安装再执行如下pip命令
```
pip install cython
pip install git+https://gitee.com/jiangjiajun/philferriere-cocoapi.git#subdirectory=PythonAPI
```

**Linux/Mac安装**
```
pip install cython  
pip install pycocotools
```
