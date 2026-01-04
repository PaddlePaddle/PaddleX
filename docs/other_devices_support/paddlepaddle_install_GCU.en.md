---
comments: true
---

# Enflame GCU PaddlePaddle Installation Tutorial

Currently, PaddleX supports the Enflame S60 chip. Considering environmental differences, we recommend using the <b>Enflame development image provided by PaddlePaddle</b> to complete the environment preparation.

## 1. Docker Environment Preparation
* Pull the image. This image is only for the development environment and does not contain a pre-compiled PaddlePaddle installation package. The image has TopsRider, the Enflame basic runtime environment library, installed by default.
```bash
# For X86 architecture
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-gcu:topsrider3.2.109-ubuntu20-x86_64-gcc84
```
* Start the container with the following command.
```bash
docker run --name paddle-gcu-dev -v /home:/home \
    --network=host --ipc=host -it --privileged \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-gcu:topsrider3.2.109-ubuntu20-x86_64-gcc84 /bin/bash
```
* Install the driver **outside of docker**. Please refer to the environment preparation section of [PaddlePaddle Custom Device Implementation for Enflame GCU](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/gcu/README.md).
```bash
bash TopsRider_i3x_*_deb_amd64.run --driver --no-auto-load
```
## 2. Install Paddle Package
Download and install the wheel package released by PaddlePaddle within the docker container. Currently, Python 3.10 wheel installation packages are provided. If you require other Python versions, refer to the [PaddlePaddle official documentation](https://www.paddlepaddle.org.cn/en/install/quick) for compilation and installation.

* Download and install the wheel package.
```bash
# Note: You need to install the CPU version of PaddlePaddle first
python -m pip install paddlepaddle==3.0.0.dev20241202 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -m pip install paddle_custom_gcu==3.0.0.dev20241203 -i https://www.paddlepaddle.org.cn/packages/nightly/gcu/
```
* Verify the installation package. After installation, run the following command:
```bash
python -c "import paddle; paddle.utils.run_check()"
```
* Expect to get output like this:
```bash
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 gcu.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
