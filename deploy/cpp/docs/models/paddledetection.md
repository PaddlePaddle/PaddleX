# PaddleDetection模型部署

当前支持PaddleDetection release-0.5分支导出的模型进行部署（仅支持FasterRCNN/MaskRCNN/PPYOLO/YOLOv3)。PaddleDetection相关详细文档可以查看官网文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/README_cn.md)https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.5。

下面主要以YoloV3为例，讲解从模型导出到部署的整个流程。

## 环境依赖

- OS 64位操作系统
- Python2 >= 2.7.15 or Python 3(3.5.1+/3.6/3.7)，64位版本
- pip/pip3(9.0.1+)，64位版本操作系统是
- CUDA >= 9.0
- cuDNN >= 7.6

## PaddleDetection部署环境安装

1.PaddleDetection 依赖 PaddlePaddle ,我们需要先安装paddlepaddle。且需要安装2.0以上版本：

```python
# 如果您的机器安装的是CUDA9，请运行以下命令安装
python -m pip install paddlepaddle-gpu==2.0.2.post90 -f https://paddlepaddle.org.cn/whl/mkl/stable.html

# 如果您的机器安装的是CUDA10.0，请运行以下命令安装
python -m pip install paddlepaddle-gpu==2.0.2.post100 -f https://paddlepaddle.org.cn/whl/mkl/stable.html

# 如果您的机器是CPU，请运行以下命令安装
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

更多的安装方式如conda, docker安装，请参考[安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作

安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle` ，再输入 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

2.下载PaddleDetection源码，并切换到`release/0.5`分支

```shell
git clone https://github.com/PaddlePaddle/PaddleDetection.git
git checkout release/0.5
cd PaddleDetection
```

3.安装PaddleDetection依赖

```
pip install -r requirements.txt
```

## PaddleDetection模型导出

导出可以参考 [PaddleDetection模型导出文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.5/docs/advanced_tutorials/deploy/EXPORT_MODEL.md)，下面以yolov3_darknet导出为例：

1.导出yolov3_darknet模型

```python
python tools/export_model.py -c configs/yolov3_darknet.yml \
        --output_dir=./inference_model \
        -o weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar 
```

导出完成后，可以看到 inference_model/yolov3_darknet目录下有三个文件`__model__`、`__params__` 、`infer_cfg.yml`  

## PaddleX cpp部署

需要先参照下面的文档进行编译：

\- [Linux编译指南](../compile/paddle/linux.md)

\- [Windows编译指南](../compile/paddle/windows.md)

编译完成后，可以用编译后的可执行文件进行预测.

### 样例一：(对单张图像做预测)

不使用`GPU`,测试图片为  `images/xiaoduxiong.jpeg`  

```shell
# windows为.\paddlex_inference\model_infer.exe
./build/demo/model_infer --model_filename=PaddleDetection/inference_model/yolov3_darknet/__model__ --params_filename=PaddleDetection/inference_model/yolov3_darknet/__params__ --cfg_file=PaddleDetection/inference_model/yolov3_darknet/infer_cfg.yml --model_type=det --image=images/xiaoduxiong.jpeg --use_gpu=0

```

图片的结果会打印出来，如果要获取结果的值，可以参照demo/model_infer.cpp里的代码拿到model->results_


### 样例二：(对图像列表做预测)

使用`GPU`预测多个图片，batch_size为2。假设有个`images/image_list.txt`文件，image_list.txt内容的格式如下：

```
images/xiaoduxiong1.jpeg
images/xiaoduxiong2.jpeg
...
images/xiaoduxiongn.jpeg
```

```sh
# windows为.\paddlex_inference\model_infer.exe
./build/demo/model_infer --model_filename=PaddleDetection/inference_model/yolov3_darknet/__model__ --params_filename=PaddleDetection/inference_model/yolov3_darknet/__params__ --cfg_file=PaddleDetection/inference_model/yolov3_darknet/infer_cfg.yml --model_type=det --image=images/xiaoduxiong.jpeg --use_gpu=1 --batch_size=2 --thread_num=2
```

