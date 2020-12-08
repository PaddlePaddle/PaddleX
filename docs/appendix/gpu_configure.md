# Multi-card GPU/CPU training

## GPU card count configuration
During training, PaddleX gives priority to all available GPU cards**.** For the classification and segmentation tasks during evaluation, multiple cards are used. For the detection task, only one card is used for calculation. In the prediction of each task, only one card is used for calculation.** **********

To configure the number of cards that PaddleX uses during runtime, perform the configuration in the command line terminal (shell) or in Python codes as follows:

Command line terminal:
```
# Use GPU card 1 export CUDA_VISIBLE_DEVICES='1' # Use GPU cards 0, 1, and 3 export CUDA_VISIBLE_DEVICES='0,1,3' # No GPU, CPU only export CUDA_VISIBLE_DEVICES=''
```

python codes:
```
# Note: The following statement must be executed before the first running of import of paddlex or paddle. import os os. environ['CUDA_VISIBLE_DEVICES'] = '0,1,3' import paddlex as pdx
```

## Training with multiple GPU cards

PaddlePaddle currently supports multi-card training in the Linux, and single-card training in the Windows. You can view the computer GPU card information by typing `nvidia-smi` in the command line terminal. If the system prompts that the card information is not found, you can install CUDA driver.

When PaddleX is trained in the multi-card GPU, no additional configuration is required. You can configure the number of cards as required by running the `CUDA_VISIBLE_DEVICES` environment variable.

It should be noted that the number of cards in the training code can be adjusted, that is, set `batch_size` and `learning_rate` to higher values. The larger the number of GPU cards, the higher the batch_size supported (keep in mind that the batch_size should be divisible by the number of cards). A higher batch_size also means a higher learning rate, that is, the learning_rate should be set to a larger value accordingly.`Similarly, during the training process, if the training fails due to lack of video memory or memory, you need to set `batch_size` and learning_rate to smaller values proportionally.``` ``

## CPU Configuration
PaddleX has the option of using the CPU for training, evaluation, and prediction during the training process. Perform configurations in the following method:

Command line terminal:
```
export CUDA_VISIBLE_DEVICES=""
```

python codes:
```
# Note: The following statement must be executed before the first running of import of paddlex or paddle. import os os.environ['CUDA_VISIBLE_DEVICES'] = '' import paddlex as pdx
```
The number of CPUs used is 1.

## Training with multiple CPUs
The number of CPUs can be changed by setting the environment variable `CPU_NUM`. If it is not set, the number of CPUs is set to 1 by default, that is, CPU_NUM`=1.`Within the range of physical quantity, the configuration of this parameter can accelerate the model.

PaddleX selects the number of CPUs (specified by the `CPU_NUM`) for training, uses the number of CPUs (specified by the CPU_NUM) for classification and segmentation tasks during evaluation, uses only one CPU for computation for the detection task, and uses only one CPU for computation for each task in prediction.`Set the number of CPUs in the following methods:`

Command line terminal:
```
export CUDA_VISIBLE_DEVICES="" export CPU_NUM=2
```

python codes:
```
# Note: The following statement must be executed before the first running of import of paddlex or paddle. import os os.environ['CUDA_VISIBLE_DEVICES'] = '' os.environ['CPU_NUM'] = '2' import paddlex as pdx
```
