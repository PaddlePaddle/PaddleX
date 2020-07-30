# 轻量级服务化部署
## 简介
借助`PaddleHub-Serving`，可以将`PaddleX`的`Inference Model`进行快速部署，以提供在线预测的能力。

关于`PaddleHub-Serving`的更多信息，可参照[PaddleHub-Serving](https://github.com/PaddlePaddle/PaddleHub/blob/develop/docs/tutorial/serving.md)。


下面，我们按照步骤，实现将一个图像分类模型[MobileNetV3_small_ssld](https://bj.bcebos.com/paddlex/models/mobilenetv3_small_ssld_imagenet.tar.gz)转换成`PaddleHub`的预训练模型，并利用`PaddleHub-Serving`实现一键部署。


# 模型部署
## 1 模型转换
首先，我们将`PaddleX`的`Inference Model`转换成`PaddleHub`的预训练模型，使用命令`hub convert`即可一键转换，对此命令的说明如下：

```shell
$ hub convert --model_dir XXXX \
              --module_name XXXX \
              --module_version XXXX \
              --output_dir XXXX
```
**参数**：

|参数|用途|
|-|-|
|--model_dir/-m|`PaddleX Inference Model`所在的目录|
|--module_name/-n|生成预训练模型的名称|
|--module_version/-v|生成预训练模型的版本，默认为`1.0.0`|
|--output_dir/-o|生成预训练模型的存放位置，默认为`{module_name}_{timestamp}`|

因此，我们仅需要一行命令即可完成预训练模型的转换。

```shell
 hub convert --model_dir mobilenetv3_small_ssld_imagenet_hub --module_name mobilenetv3_small_ssld_imagenet_hub
```

转换成功后会打印提示信息，如下：
```shell
$ The converted module is stored in `MobileNetV3_small_ssld_hub_1596077881.868501`.
```
等待生成成功的提示后，我们就在输出目录中得到了一个`PaddleHub`的一个预训练模型。

## 2 模型安装
在模型转换一步中，我们得到了一个`.tar.gz`格式的预训练模型压缩包，在进行部署之前需要先安装到本机，使用命令`hub install`即可一键安装，对此命令的说明如下：
```shell
$ hub install ${MODULE}
```
其中${MODULE}为要安装的预训练模型文件路径。

因此，我们使用`hub install`命令安装：
```shell
hub install MobileNetV3_small_ssld_hub_1596077881.868501/mobilenetv3_small_ssld_imagenet_hub.tar.gz
```
安装成功后会打印提示信息，如下：
```shell
$ Successfully installed mobilenetv3_small_ssld_imagenet_hub
```

## 3 模型部署
下面，我们只需要使用`hub serving`命令即可完成模型的一键部署，对此命令的说明如下：
```shell
$ hub serving start --modules/-m [Module1==Version1, Module2==Version2, ...] \
                    --port/-p XXXX
                    --config/-c XXXX
```

**参数**：

|参数|用途|
|-|-|
|--modules/-m|PaddleHub Serving预安装模型，以多个Module==Version键值对的形式列出<br>*`当不指定Version时，默认选择最新版本`*|
|--port/-p|服务端口，默认为8866|
|--config/-c|使用配置文件配置模型|

因此，我们仅需要一行代码即可完成模型的部署，如下：

```shell
$ hub serving start -m mobilenetv3_small_ssld_imagenet_hub
```
等待模型加载后，此预训练模型就已经部署在机器上了。

我们还可以使用配置文件对部署的模型进行更多配置，配置文件格式如下：
```json
{
  "modules_info": {
    "mobilenetv3_small_ssld_imagenet_hub": {
      "init_args": {
        "version": "1.0.0"
      },
      "predict_args": {
        "batch_size": 1,
        "use_gpu": false
      }
    }
  },
  "port": 8866
}

```
|参数|用途|
|-|-|
|modules_info|PaddleHub Serving预安装模型，以字典列表形式列出，key为模型名称。其中:<br>`init_args`为模型加载时输入的参数，等同于`paddlehub.Module(**init_args)`<br>`predict_args`为模型预测时输入的参数，以`mobilenetv3_small_ssld_imagenet_hub`为例，等同于`mobilenetv3_small_ssld_imagenet_hub.batch_predict(**predict_args)`
|port|服务端口，默认为8866|

## 4 测试
在第二步模型安装的同时，会生成一个客户端请求示例，存放在模型安装目录，默认为`${HUB_HOME}/.paddlehub/modules`，对于此例，我们可以在`~/.paddlehub/modules/mobilenetv3_small_ssld_imagenet_hub`找到此客户端示例`serving_client_demo.py`，代码如下：

```python
# coding: utf8
import requests
import json
import cv2
import base64


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


if __name__ == '__main__':
    # 获取图片的base64编码格式
    img1 = cv2_to_base64(cv2.imread("IMAGE_PATH1"))
    img2 = cv2_to_base64(cv2.imread("IMAGE_PATH2"))
    data = {'images': [img1, img2]}
    # 指定content-type
    headers = {"Content-type": "application/json"}
    # 发送HTTP请求
    url = "http://127.0.0.1:8866/predict/mobilenetv3_small_ssld_imagenet_hub"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
```
使用的测试图片如下：

![](../train/images/test.jpg)

将代码中的`IMAGE_PATH1`改成想要进行预测的图片路径后，在命令行执行：
```python
python ~/.paddlehub/module/MobileNetV3_small_ssld_hub/serving_client_demo.py
```
即可收到预测结果，如下：
```shell
[[{'category': 'envelope', 'category_id': 549, 'score': 0.2141510397195816}]]
````

到此，我们就完成了`PaddleX`模型的一键部署。
