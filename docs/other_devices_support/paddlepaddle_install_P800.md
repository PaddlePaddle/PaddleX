---
comments: true
---

# 昆仑芯 P800 飞桨安装教程

当前 PaddleX 支持昆仑 P800 等芯片。考虑到环境差异性，我们推荐使用<b>飞桨官方发布的昆仑芯 XPU 开发镜像</b>，该镜像预装有昆仑基础运行环境库（XRE）。

## 1、docker环境准备
拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包

```
# 拉取镜像
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleqa:xpu-ubuntu2204-x86_64-gcc123-py310
```
参考如下命令启动容器

```
# 参考如下命令，启动容器
docker run -it --name paddle-xpu-dev -v $(pwd):/work \
  -v /usr/local/bin/xpu-smi:/usr/local/bin/xpu-smi \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleqa:xpu-ubuntu2204-x86_64-gcc123-py310 /bin/bash
```
## 2、安装paddle包
当前提供 Python3.10 的 wheel 安装包。如有其他 Python 版本需求，可以参考[飞桨官方文档](https://www.paddlepaddle.org.cn/install/quick)自行编译安装。

安装 Python3.10 的 wheel 安装包

```
# 下载并安装 wheel 包
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
```
验证安装包 安装完成之后，运行如下命令

```
python -c "import paddle; paddle.utils.run_check()"
```
预期得到如下输出结果

```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
