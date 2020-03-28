## 1. 安装PaddlePaddle
PaddlePaddle可以在64-bit的Windows7、Windows8、Windows10企业版、Windows10专业版上运行，同时支持python2（>=2.7.15）和python3（>= 3.5.1），但pip版本必须高于9.0.1。Windows版本同时支持CPU版和GPU版的PaddlePaddle，若使用GPU版，对于CUDA和CUDNN的安装，可参考NVIDIA官方文档[(https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)和[(https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/install/Tables.html/#ciwhls-release)](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/install/Tables.html/#ciwhls-release)了解。目前，Windows环境暂不支持NCCL，分布式等相关功能。                     
- 在命令行中执行下述命令
```cmd
# 进入创建好的Anaconda环境
conda activate mypaddle
# （选择1）安装CPU版本PaddlePaddle
pip install -U paddlepaddle
# （选择2）安装GPU版本PaddlePaddle
pip install -U paddlepaddle-gpu
```
【注意】默认提供的安装包需要计算机支持AVX指令集和MKL，若环境不支持，可以在PaddlePaddle官网[(https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/install/Tables.html/#ciwhls-release)](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/install/Tables.html/#ciwhls-release)下载openblas版本的安装包                 
- 安装成功后，打开python命令行，使用以下代码进行测试：
```python
import paddle.fluid as fluid
fluid.install_check.run_check()
# 若出现Your Paddle Fluid is installed successfully!字样则表示安装成功
```
