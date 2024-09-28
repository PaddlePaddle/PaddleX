# PaddlePaddle Local Installation Tutorial

When installing PaddlePaddle, you can choose to install it via Docker or pip.

## Installing PaddlePaddle via Docker
**If you choose to install via Docker**, please refer to the following commands to use the official PaddlePaddle Docker image to create a container named `paddlex` and map the current working directory to the `/paddle` directory inside the container:

```bash
# For GPU users
# CUDA 11.8 users
nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it registry.baidubce.com/paddlepaddle/paddle:3.0.0b1-gpu-cuda11.8-cudnn8.6-trt8.5 /bin/bash

# CUDA 12.3 users
nvidia-docker run --name paddlex -v $PWD:/paddle  --shm-size=8G --network=host -it registry.baidubce.com/paddlepaddle/paddle:3.0.0b1-gpu-cuda12.3-cudnn9.0-trt8.6 /bin/bash
```
Note: For more official PaddlePaddle Docker images, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/docker/linux-docker.html). If you are a CUDA 11.8 user, please ensure your Docker version is >= 19.03; if you are a CUDA 12.3 user, please ensure your Docker version is >= 20.10.

## Installing PaddlePaddle via pip
**If you choose to install via pip**, please refer to the following commands to install PaddlePaddle in your current environment using pip:

```bash
# CPU
python -m pip install paddlepaddle

# GPU, this command is only suitable for machines with CUDA version 11.8
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# GPU, this command is only suitable for machines with CUDA version 12.3
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```
Note: For more PaddlePaddle Wheel versions, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip.html).

**For installing PaddlePaddle on other hardware, please refer to** [Installing PaddlePaddle on Other Devices](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/tutorials/INSTALL_OTHER_DEVICES.md).

After installation, you can verify if PaddlePaddle is successfully installed using the following command:

```bash
python -c "import paddle; print(paddle.__version__)"
```
If the installation is successful, the following content will be output:

```bash
3.0.0-beta1
```