# PaddleClas模型部署

当前支持PaddleClas release/2.1分支导出的模型进行部署。本文档以ResNet50模型为例，讲述从release-2.1分支导出模型并用PaddleX 进行cpp部署整个流程。 PaddleClas相关详细文档可以查看[官网文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/README_cn.md)

## 环境依赖

- PaddlePaddle >= 2.0
- Python3
- CUDA >= 9.0
- cuDNN >= 7.6.4

## PaddleClas部署环境安装

1.PaddleClas 依赖 PaddlePaddle ,我们需要先安装paddlepaddle。且需要安装2.0以上版本：

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

2.下载PaddleClas源码，并切换到release/2.1分支(当前默认是release/2.1分支就不用切)

```shell
git clone https://github.com/PaddlePaddle/PaddleClas.git
cd PaddleClas
```

3.安装依赖

```python
pip install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

## PaddleClas模型导出

下面以ResNet50模型导出为例：

1. 需要有一个训练好的模型。

可以在[官网模型库](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/models/models_intro.md)下载预训练好的模型,下面例子是从官网模型库下载ResNet50

```shell
#创建文件夹，并下载预训练模型参数(windows手动下载并放入文件夹中即可)
mkdir models
cd models
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_pretrained.pdparams
cd ../
```

也可以用模型训练/微调代码，训练一个自己的模型(可参考[文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/tutorials/getting_started.md))。

2. 导出部署模型

```python
# 基于下载的预训练模型
python tools/export_model.py --model ResNet50 --output_path ./inference/ResNet50 --pretrained_model ./models/ResNet50_pretrained
# 如果是自己的模型，只需将pretrained_model参数替换成自己的模型
python tools/export_model.py --model ResNet50 --output_path ./inference/ResNet50 --pretrained_model ./mymodel/mymodel_name
```

模型会默认导出到所填写的目录，上面例子会导出到inference/ResNet50下。

结果文件如下:

```
ResNet50
  ├── model.pdiparams        # 静态图模型参数
  ├── model.pdiparams.info   # 参数额外信息，一般无需关注
  └── model.pdmodel          # 静态图模型文件
```

3. PaddleClas导出是没有yaml文件的，使用PaddleX部署需要手动补上这个配置文件。可以参照我们写好的[resnet50_imagenet.yml](./../../../resources/resnet50_imagenet.yml)例子写自己的配置文件。

​       在这个例子中我们直接将这个`resnet50_imagenet.yml`文件拷贝到ResNet50目录下，即可进行后续的推理部署。

​       yaml配置文件详细介绍可看:  [配置文件讲解](../compile/apis/yaml.md)

## PaddleX cpp部署

需要先参照下面的文档进行编译：

\- [Linux编译指南](../compile/paddle/linux.md)

\- [Windows编译指南](../compile/paddle/windows.md)

编译完成后，可以用编译后的可执行文件进行预测.

### 样例一：(对单张图像做预测)

不使用`GPU`,测试图片为  `images/xiaoduxiong.jpeg`  

```shell
# windows为.\paddlex_inference\model_infer.exe
./build/demo/model_infer --model_filename=PaddleClas/inference/ResNet50/model.pdmodel --params_filename=PaddleClas/inference/ResNet50/model.pdiparams --cfg_file=PaddleClas/inference/ResNet50/resnet50_imagenet.yml --model_type=clas --image=images/xiaoduxiong.jpeg --use_gpu=0

```

图片的结果会打印出来，如果要获取结果的值，可以参照demo/model_infer.cpp里的代码拿到model->results_


### 样例二：(对图像列表做预测)

使用`GPU`预测多个图片，batch_size为2。假设有个`images/image_list.txt`文件，image_list.txt内容的格式如下：

```
images/image1.jpeg
images/image2.jpeg
...
images/imagen.jpeg
```

```sh
# windows为.\paddlex_inference\model_infer.exe
./build/demo/model_infer --model_filename=PaddleClas/inference/ResNet50/model.pdmodel --params_filename=PaddleClas/inference/ResNet50/model.pdiparams --cfg_file=PaddleClas/inference/ResNet50/resnet50_imagenet.yml --model_type=clas --image=images/xiaoduxiong.jpeg --use_gpu=1 --batch_size=2 --thread_num=2
```

### 样例三：(使用多卡对图像列表做预测)

使用`GPU`的第0,1两张卡预测多个图片，batch_size为4。假设有个`images/image_list.txt`文件，image_list.txt内容的格式如下：

```
images/image1.jpeg
images/image2.jpeg
...
images/imagen.jpeg
```

```sh
# windows为.\paddlex_inference\model_infer.exe
./build/demo/multi_gpu_model_infer --model_filename=PaddleClas/inference/ResNet50/model.pdmodel --params_filename=PaddleClas/inference/ResNet50/model.pdiparams --cfg_file=PaddleClas/inference/ResNet50/resnet50_imagenet.yml --model_type=clas --image=images/xiaoduxiong.jpeg --use_gpu=1 --batch_size=4 --thread_num=2 --gpu_id=0,1
```

