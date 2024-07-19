# 环境准备与安装

使用 PaddleX 前，需要进行环境准备，安装依赖项，推荐使用 PaddleX 官方镜像安装，也可使用其他自定义方式安装。

## 【推荐】 使用 PaddleX 官方镜像安装

PaddleX 官方镜像中已经内置了 PaddlePaddle、PaddleX，无需单独安装，获取 Docker 镜像并启动容器即可使用。

参考下述命令，使用 PaddleX 官方 Docker 镜像，创建一个名为 `paddlex` 的容器，并将当前工作目录映射到容器内的 `/paddle` 目录。

```bash
# 对于 CUDA11.8 用户
sudo nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it registry.baidubce.com/paddlex/paddlex:3.0.0b1-gpu-cuda11.8-cudnn8.9-trt8.5 /bin/bash

# 对于 CUDA12.3 用户
sudo docker run --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it registry.baidubce.com/paddlex/paddlex:3.0.0b1-gpu-cuda12.3-cudnn9.0-trt8.6 /bin/bash
```

## 其他自定义方式安装

自定义安装流程主要包括安装飞桨 PaddlePaddle 框架、获取 PaddleX 源码并安装依赖，如果已经通过PaddleX 官方镜像安装，则无需进行该步骤。

### 1. 安装飞桨 PaddlePaddle

#### 1.1 安装

##### 【推荐】使用 Docker 安装

参考下述命令，使用飞桨官方 Docker 镜像，创建一个名为 `paddlex` 的容器，并将当前工作目录映射到容器内的 `/paddle` 目录。

```bash
# 对于 gpu 用户
# CUDA11.8 用户
sudo nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it registry.baidubce.com/paddlepaddle/paddle:3.0.0b1-gpu-cuda11.8-cudnn8.6-trt8.5 /bin/bash

# CUDA12.3 用户
sudo nvidia-docker run --name paddlex -v $PWD:/paddle  --shm-size=8G --network=host -it registry.baidubce.com/paddlepaddle/paddle:3.0.0b1-gpu-cuda12.3-cudnn9.0-trt8.6 /bin/bash
```

更多飞桨官方 docker 镜像请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html)。

**注意**：
* 首次使用该镜像时，下述命令会自动下载该镜像文件，下载需要一定的时间，请耐心等待；
* 上述命令会创建一个名为 paddlex 的 Docker 容器，之后再次使用该容器时无需再次运行该命令；
* 参数 `--shm-size=8G` 将设置容器的共享内存为 8G，如机器环境允许，建议将该参数设置较大，如 `64G`；
* 上述镜像中默认的 Python 版本为 Python3.10，默认已经安装 PaddlePaddle 3.0beta1，如果您需要创建新的 Python 环境使用 PaddlePaddle，请参考下述 pip 安装方式。

##### 使用 pip 安装

请参考下述命令，用 pip 在当前环境中安装飞桨 PaddlePaddle。

```bash
# gpu，该命令仅适用于 CUDA 版本为 11.8 的机器环境
 python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，该命令仅适用于 CUDA 版本为 12.3 的机器环境
 python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

```
更多飞桨 Wheel 版本请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

##### 更多安装方式
关于其他硬件安装飞桨，请参考[多硬件安装飞桨](./INSTALL_OTHER_DEVICES.md)。

#### 1.2 验证

使用以下命令可以验证 PaddlePaddle 是否安装成功。

```bash
python -c "import paddle; paddle.utils.run_check()"
```

查看 PaddlePaddle 版本的命令如下：

```bash
python -c "import paddle; print(paddle.__version__)"
```

如果安装成功，将输出如下内容：
```bash
3.0.0-beta1
```

### 2. 安装 PaddleX

#### 2.1 获取源码

##### 【推荐】从 GitHub 下载

使用下述命令从 GitHub 获取 PaddleX 最新源码。

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git
```

##### 从 Gitee 下载

如果访问 GitHub 网速较慢，可以从 Gitee 下载，命令如下：

```bash
git clone https://gitee.com/paddlepaddle/PaddleX.git
```

#### 2.2 安装配置及依赖

参考下述命令，按提示操作，完成 PaddleX 依赖的安装。

```bash
cd PaddleX

# 安装 PaddleX whl
# -e：以可编辑模式安装，当前项目的代码更改，都会直接作用到已经安装的 PaddleX Wheel
pip install -e .

# 安装 PaddleX 相关依赖
paddlex --install

# 完成安装后会有如下提示：
# All packages are installed.
```

**注 :**
1. 在安装过程中，需要克隆 Paddle 官方模型套件，`--platform` 可以指定克隆源，可选 `github.com`，`gitee.com`，分别代表这些套件从 github 上和 gitee 上克隆，默认为 `github.com`；
2. 如仅需要部分 Paddle 官方模型套件，可在命令中指定，如仅克隆 PaddleDetection 套件可使用命令：paddlex --install PaddleDetection。默认获取全部 Paddle 官方模型套件，为方便后续开发使用，建议采用默认安装。
