# 多硬件安装飞桨
本文档主要针对昇腾 NPU、寒武纪 MLU、昆仑 XPU 硬件平台，介绍如何安装飞桨。
## 1. 昇腾 NPU 飞桨安装
### 1.1 环境准备
当前 PaddleX 支持昇腾 910B 芯片（更多型号还在支持中，如果您有其他型号的相关需求，请提交issue告知我们），昇腾驱动版本为 23.0.3。考虑到环境差异性，我们推荐使用飞桨官方提供的标准镜像完成环境准备。
- 1. 拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包，镜像中已经默认安装了昇腾算子库 CANN-8.0.T13。

```
# 适用于 X86 架构
docker pull registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-x86_64-gcc84-py39
# 适用于 Aarch64 架构
docker pull registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-aarch64-gcc84-py39
```

- 2. 参考如下命令启动容器，ASCEND_RT_VISIBLE_DEVICES 指定可见的 NPU 卡号
```
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-$(uname -m)-gcc84-py39 /bin/bash
```
### 1.2 安装 paddle 包
当前提供 Python3.9 的 wheel 安装包。如有其他 Python 版本需求，可以参考[飞桨官方文档](https://www.paddlepaddle.org.cn/install/quick)自行编译安装。

- 1. 下载安装 Python3.9 的 wheel 安装包

```
# 注意需要先安装飞桨 cpu 版本
python3.9 -m pip install paddlepaddle==3.0.0.dev20240520 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python3.9 -m pip install paddle_custom_npu==3.0.0.dev20240719 -i https://www.paddlepaddle.org.cn/packages/nightly/npu/
```
- 2. 验证安装包
安装完成之后，运行如下命令。
```
python -c "import paddle; paddle.utils.run_check()"
```
预期得到如下输出结果
```
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 npu.
PaddlePaddle works well on 8 npus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## 2. 寒武纪 MLU 飞桨安装
### 2.1 环境准备
当前 PaddleX 支持寒武纪 MLU370X8 芯片。考虑到环境差异性，我们推荐使用飞桨官方提供的标准镜像完成环境准备。
- 1. 拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
```
# 适用于 X86 架构，暂时不提供 Arch64 架构镜像
docker pull registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-gcc84-py310
```
- 2. 参考如下命令启动容器
```
docker run -it --name paddle-mlu-dev -v $(pwd):/work \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/bin/cnmon:/usr/bin/cnmon \
  registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-gcc84-py310 /bin/bash
```
### 2.2 安装 paddle 包
当前提供 Python3.10 的 wheel 安装包。有其他 Python 版本需求，可以参考[飞桨官方文档](https://www.paddlepaddle.org.cn/install/quick)自行编译安装。

- 1. 下载安装 Python3.10 的wheel 安装包。
```
# 注意需要先安装飞桨 cpu 版本
python -m pip install paddlepaddle==3.0.0.dev20240624 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -m pip install paddle-custom-mlu==3.0.0.dev20240806 -i https://www.paddlepaddle.org.cn/packages/nightly/mlu/
```
- 2. 验证安装包
安装完成之后，运行如下命令。
```
python -c "import paddle; paddle.utils.run_check()"
```
预期得到如下输出结果
```
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 mlu.
PaddlePaddle works well on 16 mlus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## 3.昆仑 XPU 飞桨安装
### 3.1 环境准备
当前 PaddleX 支持昆仑 R200/R300 等芯片。考虑到环境差异性，我们推荐使用飞桨官方发布的昆仑 XPU 开发镜像，该镜像预装有昆仑基础运行环境库（XRE）。
- 1. 拉取镜像，此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
```
docker pull registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 # X86 架构
docker pull registry.baidubce.com/device/paddle-xpu:kylinv10-aarch64-gcc82-py310 # ARM 架构
```
- 2. 参考如下命令启动容器
```
docker run -it --name=xxx -m 81920M --memory-swap=81920M \
    --shm-size=128G --privileged --net=host \
    -v $(pwd):/workspace -w /workspace \
    registry.baidubce.com/device/paddle-xpu:$(uname -m)-py310 bash
```

### 3.2 安装 paddle 包
当前提供 Python3.10 的 wheel 安装包。有其他 Python 版本需求，可以参考[飞桨官方文档](https://www.paddlepaddle.org.cn/install/quick)自行编译安装。

- 1. 安装 Python3.10 的 wheel 安装包
```
pip install https://paddle-whl.bj.bcebos.com/paddlex/xpu/paddlepaddle_xpu-2.6.1-cp310-cp310-linux_x86_64.whl # X86 架构
pip install https://paddle-whl.bj.bcebos.com/paddlex/xpu/paddlepaddle_xpu-2.6.1-cp310-cp310-linux_aarch64.whl # ARM 架构
```
- 2. 验证安装包
安装完成之后，运行如下命令
```
python -c "import paddle; paddle.utils.run_check()"
```
预期得到如下输出结果
```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
