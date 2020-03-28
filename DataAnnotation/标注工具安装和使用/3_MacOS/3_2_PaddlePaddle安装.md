## 2.3.2.1 安装PaddlePaddle
PaddlePaddle可以在64-bit的MacOS10.11、MacOS10.12、MacOS10.13、MacOS10.14上运行，同时支持python2（>=2.7.15）和python3（>= 3.5.1），但pip版本必须高于9.0.1。目前，MacOS环境仅支持CPU版PaddlePaddle。

- 在命令行中执行下述命令
```cmd
# 进入创建好的Anaconda环境
source activate mypaddle
# （选择1）安装CPU版本PaddlePaddle
pip install -U paddlepaddle
# （选择2）安装指定版本PaddlePaddle
pip install -U paddlepaddle==[版本号]
```

- 安装成功后，打开python命令行，使用以下代码进行测试：
```python
import paddle.fluid as fluid
fluid.install_check.run_check()
# 若出现Your Paddle Fluid is installed successfully!字样则表示安装成功
```
