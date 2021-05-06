# PaddleSeg模型部署

PaddlX的cpp部署，目前支持PaddleSeg release-2.0版本导出的模型。本文档以[Deeplabv3P](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0/configs/deeplabv3p)模型为例，讲述从release-2.0版本导出模型并用PaddleX 进行cpp部署整个流程。 PaddleSeg相关详细文档查看[官网文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0/README_CN.md)

## 环境依赖

- PaddlePaddle >= 2.0.0
- Python >= 3.6+
- CUDA >= 9.0
- cuDNN >= 7.6

## PaddleSeg部署环境安装

1.PaddleSeg 依赖 PaddlePaddle ,我们需要先安装paddlepaddle。且需要安装2.0以上版本：

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

2.下载PaddleSeg源码，并切换到release-2.0版本(当前默认是release-2.0分支就不用切)

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
cd PaddleSeg
```

3.安装paddleseg包/依赖

```python
# 通过pip形式安装paddleseg库，不仅安装了代码运行的环境依赖，也安装了PaddleSeg的API
pip install paddleseg
# 只是部署也可只安装依赖
pip install -r requirements.txt
```

## PaddleSeg模型导出

导出可以参考 [PaddleSeg模型导出文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0/docs/model_export.md)，下面以Deeplabv3P模型导出为例：

```python
# linux 会主动下载预训练好的模型参数
python export.py --config configs/deeplabv3p/deeplabv3p_resnet50_os8_voc12aug_512x512_40k.yml
# windows 需要手动下载参数，下载地址在yml文件中。deeplabv3p_resnet50_os8_voc12aug_512x512_40k.yml中的预训练模型地址为：  https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz
# 下载后解压，解压后为resnet50_vd_ssld_v2_imagenet/model.pdparams ，将其传入model_path参数
python export.py --config configs/deeplabv3p/deeplabv3p_resnet50_os8_voc12aug_512x512_40k.yml \
                --model_path resnet50_vd_ssld_v2_imagenet/model.pdparams
```

模型会默认导出到output目录

结果文件如下:

```
output
  ├── deploy.yaml            # 部署相关的配置文件
  ├── model.pdiparams        # 静态图模型参数
  ├── model.pdiparams.info   # 参数额外信息，一般无需关注
  └── model.pdmodel          # 静态图模型文件
```

## PaddleX cpp部署

需要先参照下面的文档进行编译：

\- [Linux编译指南](../compile/paddle/linux.md)

\- [Windows编译指南](../compile/paddle/windows.md)

编译完成后，可以用编译后的可执行文件进行预测.

### 样例一：(对单张图像做预测)

不使用`GPU`,测试图片为  `images/image1.jpeg`  

```shell
# windows为.\paddlex_inference\model_infer.exe
./build/demo/model_infer --model_filename=PaddleSeg/output/model.pdmodel --params_filename=PaddleSeg/output/model.pdiparams --cfg_file=PaddleSeg/output/deploy.yaml --model_type=seg --image=images/image1.jpeg --use_gpu=0

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
./build/demo/model_infer --model_filename=PaddleSeg/output/model.pdmodel --params_filename=PaddleSeg/output/model.pdiparams --cfg_file=PaddleSeg/output/deploy.yaml --model_type=seg --image=images/image_list.txt --use_gpu=1 --batch_size=2 --thread_num=2
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
./build/demo/multi_gpu_model_infer --model_filename=PaddleSeg/output/model.pdmodel --params_filename=PaddleSeg/output/model.pdiparams --cfg_file=PaddleSeg/output/deploy.yaml --model_type=seg --image=images/image_list.txt --use_gpu=1 --batch_size=4 --thread_num=2 --gpu_id=0,1
```

