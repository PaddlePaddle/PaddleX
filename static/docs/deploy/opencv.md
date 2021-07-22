# 部署编译说明

目前PaddleX所有的模型部署测试环境为
- Ubuntu 16.04/18.04  / Windows 10
- gcc 4.8.5 / Microsoft Visual Studio 2019

如果切换至其它Linux系统环境（gcc版本不变)，预期会遇到opencv的问题。

在Linux编译脚本中，例如`deploy/cpp/script/build.sh`中，依赖`deploy/cpp/script/bootstrap.sh`去自动下载预先已经编译好的依赖的opencv库和加密库。而目前`bootstrap.sh`只提供了OpenCV在Ubuntu16.04/18.04两个系统环境下的预编译包，如果你的系统与此不同，尝试按照如下方式解决。


## Linux下自编译OpenCV

### 1. 下载OpenCV Source Code  
前往OpenCV官方网站下载OpenCV 3.4.6 Source Code，或者直接[点击这里](https://bj.bcebos.com/paddlex/deploy/opencv-3.4.6.zip)下载我们已经上传至服务器的源码压缩包。

### 2. 编译OpenCV
确认自己的gcc/g++版本为4.8.5版本，编译过程参考如下代码  

当前opencv-3.4.6.zip存放路径为`/home/paddlex/opencv-3.4.6.zip`
```
unzip opencv-3.4.6.zip
cd opencv-3.4.6
mkdir build && cd build
mkdir opencv3.4.6gcc4.8ffmpeg
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/home/paddlex/opencv-3.4.6/build/opencv3.4.6gcc4.8ffmpeg -D WITH_FFMPEG=ON ..
make -j5
make install
```
编译好的opencv会存放在设定的`/home/paddlex/opencv-3.4.6/build/opencv3.4.6gcc4.8ffmpeg`下

### 3. 编译PaddleX预测代码依赖自己的opencv
修改`deploy/cpp/script/build.sh`

1. 注释或删除掉如下代码

```
{
    bash $(pwd)/scripts/bootstrap.sh # 下载预编译版本的加密工具和opencv依赖库
} || {
    echo "Fail to execute script/bootstrap.sh"
    exit -1
}
```

2. 模型加密开关设置
如果您不需要用到PaddleX的模型加密功能，则将如下开关修改为OFF即可
```
WITH_ENCRYPTION=OFF
```
如果需要用到加密，则请手动下载加密库后解压，[点击下载](https://bj.bcebos.com/paddlex/tools/paddlex-encryption.zip)

3. 设置依赖库路径
将`OPENCV_DIR`设置为自己编译好的路径，如
```
OPENCV_DIR=/home/paddlex/opencv-3.4.6/build/opencv3.4.6gcc4.8ffmpeg
```
如果您还需要用到模型加密，已经将`WITH_ENCRYPTION`设为`ON`的前提下，也同时将`ENCRYPTION_DIR`设置为自己下载解压后的路径，如
```
ENCRYPTION_DIR=/home/paddlex/paddlex-encryption
```

4. 执行`sh script/build.sh`编译即可

## 反馈

如在使用中仍然存在问题，请前往PaddleX的Github提ISSUE反馈给我们。

- [PaddleX Issue](https://github.com/PaddlePaddle/PaddleX/issues)
