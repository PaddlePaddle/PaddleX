# 快速安装

以下安装过程默认用户已安装好**paddlepaddle-gpu或paddlepaddle(版本大于或等于1.8.1)**，paddlepaddle安装方式参照[飞桨官网](https://www.paddlepaddle.org.cn/install/quick)

- <a href="#pip安装">pip安装PaddleX</a>  
- <a href="#github代码安装">github代码安装PaddleX</a>
- <a href="#pycocotools安装问题">pycocotools安装问题</a>

<a name="pip安装"></a>
**安装方式一 pip安装**

> 注意其中pycocotools在Windows安装较为特殊，可参考下面的Windows安装命令  

```
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```

<a name="github代码安装"></a>
**安装方式二 Github代码安装**  

github代码会跟随开发进度不断更新

```
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
git checkout develop
python setup.py install
```


<a name="pycocotools安装问题"></a>
**pycocotools安装问题**  
> PaddleX依赖pycocotools包，如安装pycocotools失败，可参照如下方式安装pycocotools

> Windows安装时可能会提示缺少`Microsoft Visual C++ 2015 build tools`，[点击下载VC build tools](https://go.microsoft.com/fwlink/?LinkId=691126)安装再执行如下pip命令
```
pip install cython
pip install git+https://gitee.com/jiangjiajun/philferriere-cocoapi.git#subdirectory=PythonAPI
```

> Linux/Mac系统下，直接使用pip安装如下两个依赖即可
```
pip install cython  
pip install pycocotools
```
