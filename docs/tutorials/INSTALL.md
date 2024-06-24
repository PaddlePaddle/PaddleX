# 环境准备与安装

使用 PaddleX 前，需要进行环境准备，安装依赖项，主要包括安装飞桨 PaddlePaddle 框架、获取 PaddleX 源码并安装依赖。

## 1. 安装飞桨 PaddlePaddle

### 1.1 安装

#### 【推荐】使用 Docker 安装

参考下述命令，使用飞桨官方 Docker 镜像，创建一个名为 `paddlex` 的容器，并将当前工作目录映射到容器内的 `/paddle` 目录。

```shell
# 对于 GPU 用户
sudo nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it registry.baidubce.com/paddlepaddle/paddle:2.6.1-gpu-cuda12.0-cudnn8.9-trt8.6 /bin/bash

# 对于 CPU 用户
sudo docker run --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it registry.baidubce.com/paddlepaddle/paddle:2.6.1 /bin/bash
```

更多飞桨官方 docker 镜像请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html)。

**注意**：
* 首次使用该镜像时，下述命令会自动下载该镜像文件，下载需要一定的时间，请耐心等待；
* 请使用 **3.0** 版本的 PaddlePaddle；
* 上述命令会创建一个名为 paddlex 的 Docker 容器，之后再次使用该容器时无需再次运行该命令；
* 参数 `--shm-size=8G` 将设置容器的共享内存为 8G，如机器环境允许，建议将该参数设置较大，如 `64G`；

#### 使用 pip 安装

参考下述命令，用 pip 在当前环境中安装飞桨 PaddlePaddle。

<!-- 这里需要指定 paddle3.0 版本 -->
```bash
# GPU，该命令仅适用于 CUDA 版本为 12 的机器环境，对于其他 CUDA 版本的支持请参考飞桨官网
python -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# CPU
python -m pip install paddlepaddle==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

更多飞桨 Wheel 版本请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

#### 更多安装方式
关于其他硬件安装飞桨，请参考[多硬件安装飞桨](./INSTALL_OTHER_DEVICES.md)。

### 1.2 验证

使用以下命令可以验证 PaddlePaddle 是否安装成功。

```bash
python -c "import paddle; paddle.utils.run_check()"
```

查看 PaddlePaddle 版本的命令如下：

```bash
python -c "import paddle; print(paddle.__version__)"
```

<!-- 这里需要指明输出什么内容则表示正确 -->


## 2. 安装 PaddleX

### 2.1 获取源码

#### 【推荐】从 GitHub 下载

使用下述命令从 GitHub 获取 PaddleX 最新源码。

```shell
git clone https://github.com/PaddlePaddle/PaddleX.git
```

#### 从 Gitee 下载

如果访问 GitHub 网速较慢，可以从 Gitee 下载，命令如下：

```shell
git clone https://gitee.com/paddlepaddle/PaddleX.git
```

### 2.2 安装配置及依赖

参考下述命令，按提示操作，完成 PaddleX 依赖的安装。

<!-- 这里需要指明安装成功的状态， 廷权 -->
```bash
cd PaddleX

# 安装 PaddleX whl
# -e：以可编辑模式安装，当前项目的代码更改，都会直接作用到已经安装的 PaddleX Wheel
pip install -e .

# 安装 PaddleX 相关依赖
paddlex --install
```

**注 :** 在安装过程中，需要克隆 Paddle 官方模型套件，`--platform` 可以指定克隆源，可选 `github.com`，`gitee.com`，分别代表这些套件从 github 上和 gitee 上克隆，默认为 `github.com`。
