---
comments: true
---

# PaddlePaddle Local Installation Tutorial

When installing PaddlePaddle, you can choose to install it via Docker or pip.

## Installing PaddlePaddle via Docker
<b>If you choose to install via Docker</b>, please refer to the following commands to use the official PaddlePaddle Docker image to create a container named `paddlex` and map the current working directory to the `/paddle` directory inside the container:

If your Docker version >= 19.03, please use:

```bash
# For CPU users:
docker run --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0rc0 /bin/bash

# For GPU users:
# CUDA 11.8 users
docker run --gpus all --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0rc0-gpu-cuda11.8-cudnn8.6-trt8.5 /bin/bash

# CUDA 12.3 users
docker run --gpus all --name paddlex -v $PWD:/paddle  --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0rc0-gpu-cuda12.3-cudnn9.0-trt8.6 /bin/bash
```

* If your Docker version <= 19.03 and >= 17.06, please use:

<details><summary> Click Here</summary>

<pre><code class="language-bash"># For CPU users:
docker run --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0rc0 /bin/bash

# For GPU users:
# CUDA 11.8 users
nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0rc0-gpu-cuda11.8-cudnn8.6-trt8.5 /bin/bash

# CUDA 12.3 users
nvidia-docker run --name paddlex -v $PWD:/paddle  --shm-size=8G --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0rc0-gpu-cuda12.3-cudnn9.0-trt8.6 /bin/bash
</code></pre></details>

* If your Docker version <= 17.06, please update your Docker.


* Note: For more official PaddlePaddle Docker images, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/docker/linux-docker.html)

## Installing PaddlePaddle via pip
<b>If you choose to install via pip</b>, please refer to the following commands to install PaddlePaddle in your current environment using pip:

```bash
# CPU
python -m pip install paddlepaddle==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# GPU, this command is only suitable for machines with CUDA version 11.8
python -m pip install paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# GPU, this command is only suitable for machines with CUDA version 12.3
python -m pip install paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
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

> ❗ <b>Note</b>: If you encounter any issues during the installation process, feel free to [submit an issue](https://github.com/PaddlePaddle/Paddle/issues) in the Paddle repository.
