# 多卡GPU/CPU训练

## GPU卡数配置
PaddleX在训练过程中会优先选择**当前所有可用的GPU卡进行训练**，在评估时**分类和分割任务仍使用多张卡**而**检测任务只使用1张卡**进行计算，在预测时各任务**则只会使用1张卡进行计算**。

用户如想配置PaddleX在运行时使用的卡的数量，可在命令行终端（Shell）或Python代码中按如下方式配置：

命令行终端：
```
# 使用1号GPU卡
export CUDA_VISIBLE_DEVICES='1'
# 使用0, 1, 3号GPU卡
export CUDA_VISIBLE_DEVICES='0,1,3'
# 不使用GPU,仅使用CPU
export CUDA_VISIBLE_DEVICES=''
```

python代码：
```
# 注意：须要在第一次import paddlex或paddle前执行如下语句
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'
import paddlex as pdx
```

## 使用多个GPU卡训练

目前PaddlePaddle支持在Linux下使用多卡训练，Windows只支持单卡，在命令行终端输入`nvidia-smi`可以查看自己机器的GPU卡信息，如若提示命令未找到，则用户需要自行安装CUDA驱动。  

PaddleX在多卡GPU下训练时，无需额外的配置，用户按照上文的方式，通过`CUDA_VISIBLE_DEVICES`环境变量配置所需要使用的卡的数量即可。  

需要注意的是，在训练代码中，可根据卡的数量，调高`batch_size`和`learning_rate`等参数，GPU卡数量越多，则可以支持更高的`batch_size`(注意batch_size需能被卡的数量整除), 同时更高的`batch_size`也意味着学习率`learning_rate`也要对应上调。同理，在训练过程中，如若因显存或内存不够导致训练失败，用户也需自行调低`batch_size`，并且按比例调低学习率。

## CPU配置
PaddleX在训练过程中可以选择使用CPU进行训练、评估和预测。通过以下方式进行配置：

命令行终端：
```
export CUDA_VISIBLE_DEVICES=""
```

python代码：
```
# 注意：须要在第一次import paddlex或paddle前执行如下语句
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import paddlex as pdx
```
此时使用的CPU个数为1。

## 使用多个CPU训练
通过设置环境变量`CPU_NUM`可以改变CPU个数，如果未设置，则CPU数目默认设为1，即`CPU_NUM`=1。 在物理核心数量范围内，该参数的配置可以加速模型。

PaddleX在训练过程中会选择`CPU_NUM`个CPU进行训练，在评估时分类和分割任务仍使用`CPU_NUM`个CPU，而检测任务只使用1个CPU进行计算，在预测时各任务则只会使用1个CPU进行计算。
通过以下方式可设置CPU的个数：

命令行终端：
```
export CUDA_VISIBLE_DEVICES=""
export CPU_NUM=2
```

python代码：
```
# 注意：须要在第一次import paddlex或paddle前执行如下语句
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CPU_NUM'] = '2'
import paddlex as pdx
```
