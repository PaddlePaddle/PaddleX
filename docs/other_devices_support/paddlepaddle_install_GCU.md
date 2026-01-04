---
comments: true
---

# 燧原 GCU 飞桨安装教程

当前 PaddleX 支持燧原 S60 芯片。考虑到环境差异性，我们推荐使用<b>飞桨官方提供的燧原 GCU 开发镜像</b>完成环境准备。

## 1、docker环境准备
* 拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包，镜像中已经默认安装了燧原软件栈 TopsRider。
```bash
# 适用于 X86 架构
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-gcu:topsrider3.2.109-ubuntu20-x86_64-gcc84
```
* 参考如下命令启动容器
```bash
docker run --name paddle-gcu-dev -v /home:/home \
    --network=host --ipc=host -it --privileged \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-gcu:topsrider3.2.109-ubuntu20-x86_64-gcc84 /bin/bash
```
* **容器外**安装驱动程序。可以参考[飞桨自定义接入硬件后端(GCU)](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/gcu/README_cn.md)环境准备章节。
```bash
bash TopsRider_i3x_*_deb_amd64.run --driver --no-auto-load
```
## 2、安装paddle包
在启动的 docker 容器中，下载并安装飞桨官网发布的 wheel 包。当前提供 Python3.10 的 wheel 安装包。如有其他 Python 版本需求，可以参考[飞桨官方文档](https://www.paddlepaddle.org.cn/install/quick)自行编译安装。

* 下载并安装 wheel 包。
```bash
# 注意需要先安装飞桨 cpu 版本
python -m pip install paddlepaddle==3.0.0.dev20241202 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -m pip install paddle_custom_gcu==3.0.0.dev20241203 -i https://www.paddlepaddle.org.cn/packages/nightly/gcu/
```
* 验证安装包：安装完成之后，运行如下命令：
```bash
python -c "import paddle; paddle.utils.run_check()"
```
预期得到类似如下输出结果：

```bash
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 gcu.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
