---
comments: true
---

# Cambricon MLU Installation Tutorial for PaddlePaddle

Currently, PaddleX supports the Cambricon MLU370X8 chip. Considering environmental differences, we recommend using the <b>Cambricon MLU development image provided by PaddlePaddle</b> to prepare your environment.

## 1. Docker Environment Preparation
Pull the image. This image is for development only and does not include a pre-compiled PaddlePaddle installation package.

```bash
# Applicable to X86 architecture, Arch64 architecture image is not provided for now
docker pull registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-gcc84-py310
```

Start the container with the following command as a reference:

```bash
docker run -it --name paddle-mlu-dev -v $(pwd):/work \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/bin/cnmon:/usr/bin/cnmon \
  registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-gcc84-py310 /bin/bash
```

## 2. Install Paddle Package
Within the started docker container, download and install the wheel package released by PaddlePaddle. Currently, Python 3.10 wheel packages are provided. If you require other Python versions, refer to the [PaddlePaddle official documentation](https://www.paddlepaddle.org.cn/en/install/quick) for compilation and installation instructions.

```bash
# Download and install the wheel package
# Note: You need to install the CPU version of PaddlePaddle first
python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu
python -m pip install paddle-custom-mlu -i https://www.paddlepaddle.org.cn/packages/nightly/mlu
```

Verify the installation. After installation, run the following command:

```bash
python -c "import paddle; paddle.utils.run_check()"
```

The expected output is:

```
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 mlu.
PaddlePaddle works well on 16 mlus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
