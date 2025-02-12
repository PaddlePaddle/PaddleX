---
comments: true
---

# Hygon DCU PaddlePaddle Installation Tutorial

Currently, PaddleX supports Haiguang Z100 series chips. Considering environmental differences, we recommend using the <b>officially released Haiguang DCU development image by PaddlePaddle</b>, which is pre-installed with the Haiguang DCU basic runtime library (DTK).

## 1. Docker Environment Preparation
Pull the image. Note that this image is only for development environments and does not include pre-compiled PaddlePaddle installation packages.

```bash
docker pull registry.baidubce.com/device/paddle-dcu:dtk23.10.1-kylinv10-gcc73-py310
```

Start the container with the following command as a reference:

```bash
docker run -it --name paddle-dcu-dev -v `pwd`:/work \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  registry.baidubce.com/device/paddle-dcu:dtk23.10.1-kylinv10-gcc73-py310 /bin/bash
```

## 2. Install PaddlePaddle Package
Within the started docker container, download and install the wheel package released by PaddlePaddle's official website. <b>Note</b>: The DCU version of PaddlePaddle framework only supports Hygon C86 architecture.

```bash
# Download and install the wheel package
pip install paddlepaddle-dcu -i https://www.paddlepaddle.org.cn/packages/nightly/dcu
```

After the installation package is installed, run the following command to verify it:

```bash
python -c "import paddle; paddle.utils.run_check()"
```

The expected output is as follows:

```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
