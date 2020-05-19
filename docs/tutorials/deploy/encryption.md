# Paddle模型加密方案

飞桨团队推出模型加密方案，使用业内主流的AES加密技术对最终模型进行加密。飞桨用户可以通过PaddleX导出模型后，使用该方案对模型进行加密，预测时使用解密SDK进行模型解密并完成推理，大大提升AI应用安全性和开发效率。

**注意：目前加密方案仅支持Linux系统**

## 1. 方案简介

### 1.1 加密工具

[PaddleX模型加密工具](https://bj.bcebos.com/paddlex/tools/paddlex-encryption.zip)。在编译部署代码时，编译脚本会自动下载加密工具，您也可以选择手动下载。

加密工具包含内容为：
```
paddlex-encryption
├── include # 头文件：paddle_model_decrypt.h（解密）和paddle_model_encrypt.h（加密）
|
├── lib # libpmodel-encrypt.so和libpmodel-decrypt.so动态库
|
└── tool # paddlex_encrypt_tool
```

### 1.2 加密PaddleX模型

对模型完成加密后，加密工具会产生随机密钥信息(用于AES加解密使用），需要在后续加密部署时传入该密钥来用于解密。
> 密钥由32字节key + 16字节iv组成， 注意这里产生的key是经过base64编码后的，这样可以扩充key的选取范围

```
./paddlex-encryption/tool/paddlex_encrypt_tool -model_dir /path/to/paddlex_inference_model -save_dir /path/to/paddlex_encrypted_model
```

`-model_dir`用于指定inference模型路径，可使用[导出小度熊识别模型](deploy.md#导出inference模型)中导出的`inference_model`。加密完成后，加密过的模型会保存至指定的`-save_dir`下，包含`__model__.encrypted`、`__params__.encrypted`和`model.yml`三个文件，同时生成密钥信息，命令输出如下图所示，密钥为`kLAl1qOs5uRbFt0/RrIDTZW2+tOf5bzvUIaHGF8lJ1c=`
![](images/encryt.png)

## 2. PaddleX C++加密部署

参考[Linux平台编译指南](deploy_cpp_linux.md)编译C++部署代码。编译成功后，预测demo的可执行程序分别为`build/demo/detector`，`build/demo/classifer`，`build/demo/segmenter`，用户可根据自己的模型类型选择，其主要命令参数说明如下：

|  参数   | 说明  |
|  ----  | ----  |
| model_dir  | 导出的预测模型所在路径 |
| image  | 要预测的图片文件路径 |
| image_list  | 按行存储图片路径的.txt文件 |
| use_gpu  | 是否使用 GPU 预测, 支持值为0或1(默认值为0) |
| use_trt  | 是否使用 TensorTr 预测, 支持值为0或1(默认值为0) |
| gpu_id  | GPU 设备ID, 默认值为0 |
| save_dir | 保存可视化结果的路径, 默认值为"output"，classfier无该参数 |
| key | 加密过程中产生的密钥信息，默认值为""表示加载的是未加密的模型 |


## 样例

可使用[导出小度熊识别模型](deploy.md#导出inference模型)中的测试图片进行预测。

`样例一`：

不使用`GPU`测试图片 `/path/to/xiaoduxiong.jpeg`  

```shell
./build/demo/detector --model_dir=/path/to/inference_model --image=/path/to/xiaoduxiong.jpeg --save_dir=output --key=kLAl1qOs5uRbFt0/RrIDTZW2+tOf5bzvUIaHGF8lJ1c=
```
`--key`传入加密工具输出的密钥，例如`kLAl1qOs5uRbFt0/RrIDTZW2+tOf5bzvUIaHGF8lJ1c=`, 图片文件`可视化预测结果`会保存在`save_dir`参数设置的目录下。


`样例二`:

使用`GPU`预测多个图片`/path/to/image_list.txt`，image_list.txt内容的格式如下：
```
/path/to/images/xiaoduxiong1.jpeg
/path/to/images/xiaoduxiong2.jpeg
...
/path/to/images/xiaoduxiongn.jpeg
```
```shell
./build/demo/detector --model_dir=/path/to/models/inference_model --image_list=/root/projects/images_list.txt --use_gpu=1 --save_dir=output --key=kLAl1qOs5uRbFt0/RrIDTZW2+tOf5bzvUIaHGF8lJ1c=
```
`--key`传入加密工具输出的密钥，例如`kLAl1qOs5uRbFt0/RrIDTZW2+tOf5bzvUIaHGF8lJ1c=`, 图片文件`可视化预测结果`会保存在`save_dir`参数设置的目录下。
