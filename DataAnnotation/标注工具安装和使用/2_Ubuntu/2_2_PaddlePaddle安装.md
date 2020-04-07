## 2.2.2.1 安装PaddlePaddle
PaddlePaddle可以在64-bit的Ubuntu14.04（支持CUDA8、CUDA10）、Ubuntu16.04（支持CUDA8、CUDA9、CUDA10）、Ubuntu18.04(支持CUDA10)上运行，同时支持python2（>=2.7.15）和python3（>= 3.5.1），但pip版本必须高于9.0.1。Windows版本同时支持CPU版和GPU版的PaddlePaddle，若使用GPU版，对于CUDA和CUDNN的安装，可参考NVIDIA官方文档[(https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)和[(https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/install/Tables.html/#ciwhls-release)](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/install/Tables.html/#ciwhls-release)了解。


- 在命令行中执行下述命令
```cmd
# 进入创建好的Anaconda环境
source activate mypaddle
# （选择1）安装CPU版本PaddlePaddle
pip install -U paddlepaddle
# （选择2）安装GPU版本PaddlePaddle
pip install -U paddlepaddle-gpu
# （选择3）安装指定版本PaddlePaddle
pip install -U paddlepaddle-gpu==[版本号]
pip install -U paddlepaddle==[版本号]
```
【注意】版本号可参考PyPi官网[(https://pypi.org/project/paddlepaddle-gpu/#history)](https://pypi.org/project/paddlepaddle-gpu/#history)       

- 安装成功后，打开python命令行，使用以下代码进行测试：
```python
import paddle.fluid as fluid
fluid.install_check.run_check()
# 若出现Your Paddle Fluid is installed successfully!字样则表示安装成功
```
