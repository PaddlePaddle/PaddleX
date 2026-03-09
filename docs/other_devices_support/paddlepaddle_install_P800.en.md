---
comments: true
---

# Kunlunxin XPU PaddlePaddle Installation Tutorial

Currently, PaddleX supports Kunlunxin P800. Considering environmental differences, we recommend using the <b>Kunlunxin P800 development image officially released by PaddlePaddle</b>, which is pre-installed with the Kunlunxin basic runtime environment library (XRE).

## 1. Docker Environment Preparation
Pull the image. This image is only for the development environment and does not include a pre-compiled PaddlePaddle installation package.

```bash
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleqa:xpu-ubuntu2204-x86_64-gcc123-py310
```
Refer to the following command to start the container:

```bash
docker run -it --name paddle-xpu-dev -v $(pwd):/work \
  -v /usr/local/bin/xpu-smi:/usr/local/bin/xpu-smi \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleqa:xpu-ubuntu2204-x86_64-gcc123-py310 /bin/bash
```

## 2. Install Paddle Package
Currently, Python3.10 wheel installation packages are provided. If you have a need for other Python versions, you can refer to the [PaddlePaddle official documentation](https://www.paddlepaddle.org.cn/en/install/quick) to compile and install them yourself.

Install the Python3.10 wheel installation package:

```bash
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/   # For X86 architecture
```

Verify the installation package. After installation, run the following command:

```bash
python -c "import paddle; paddle.utils.run_check()"
```

The expected output is:

```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
