---
comments: true
---

# PaddlePaddle Local Installation Tutorial

When installing PaddlePaddle, you can choose to install it via Docker or pip.

## Installing PaddlePaddle via Docker
<b>If you choose to install via Docker</b>, please refer to the following commands to use the official Docker image of the PaddlePaddle framework to create a container named `paddlex` and map the current working directory to the `/paddle` directory inside the container:

If your Docker version >= 19.03, please use:

```bash
# For CPU users:
docker run --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0 /bin/bash

# For GPU users:
# gpu，requires GPU driver version ≥450.80.02 (Linux) or ≥452.39 (Windows)
docker run --gpus all --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash

# gpu，requires GPU driver version ≥550.54.14 (Linux) or ≥550.54.14 (Windows)
docker run --gpus all --name paddlex -v $PWD:/paddle  --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5 /bin/bash
```

* If your Docker version <= 19.03 and >= 17.06, please use:

<details><summary> Click Here</summary>

<pre><code class="language-bash"># For CPU users:
docker run --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0 /bin/bash

# For GPU users:
# CUDA 11.8 users
nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash

# CUDA 12.3 users
nvidia-docker run --name paddlex -v $PWD:/paddle  --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5 /bin/bash
</code></pre></details>

* If your Docker version <= 17.06, please update your Docker.


* Note: For more official PaddlePaddle Docker images, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/docker/linux-docker.html)

To use [Paddle Inference TensorRT Subgraph Engine](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html#gpu), install TensorRT by executing the following instructions in the 'paddlex' container that has just been started

```bash
python -m pip install /usr/local/TensorRT-*/python/tensorrt-*-cp310-none-linux_x86_64.whl
```

## Installing PaddlePaddle via pip
<b>If you choose to install via pip</b>, please refer to the following commands to install PaddlePaddle in your current environment using pip:

```bash
# CPU
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# gpu，requires GPU driver version ≥450.80.02 (Linux) or ≥452.39 (Windows)
 python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，requires GPU driver version ≥550.54.14 (Linux) or ≥550.54.14 (Windows)
 python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```


Note: For more PaddlePaddle Wheel versions, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip.html).

<b>For installing PaddlePaddle on other hardware, please refer to</b> [PaddleX Multi-hardware Usage Guide](../other_devices_support/multi_devices_use_guide.en.md).

After installation, you can verify if PaddlePaddle is successfully installed using the following command:

```bash
python -c "import paddle; print(paddle.__version__)"
```
If the installation is successful, the following content will be output:

```bash
3.0.0-rc0
```

If you want to use the [Paddle Inference TensorRT Subgraph Engine](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/paddle_v3_features/paddle_trt_en.html), after installing Paddle, you need to refer to the [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html) to install the corresponding version of TensorRT:

- For PaddlePaddle with CUDA 11.8, the compatible TensorRT version is 8.x (where x >= 6). PaddleX has completed compatibility tests of Paddle-TensorRT on TensorRT 8.6.1.6, so it is **strongly recommended to install TensorRT 8.6.1.6**.
- For PaddlePaddle with CUDA 12.6, the compatible TensorRT version is 10.x (where x >= 5), and it is recommended to install TensorRT 10.5.0.18.

Below is an example of installing TensorRT 8.6.1.6 using the "Tar File Installation" method in a CUDA 11.8 environment:

```bash
# Download TensorRT tar file
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
# Extract TensorRT tar file
tar xvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
# Install TensorRT wheel package
python -m pip install TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl
# Add the absolute path of TensorRT's `lib` directory to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:TensorRT-8.6.1.6/lib"
```

> ❗ <b>Note</b>: If you encounter any issues during the installation process, feel free to [submit an issue](https://github.com/PaddlePaddle/Paddle/issues) in the Paddle repository.
