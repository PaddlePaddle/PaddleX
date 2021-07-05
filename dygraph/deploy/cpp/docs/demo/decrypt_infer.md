# 模型加密预测示例

本文档说明`PaddleX/deploy/cpp/demo/decrypt_infer.cpp`编译后的使用方法，仅供用户参考进行使用，开发者可基于此demo示例进行二次开发，满足集成的需求。

## 步骤一、编译
参考编译文档

- [Linux系统上编译指南](../compile/paddle/linux.md)
- [Windows系统上编译指南](../compile/paddle/windows.md)

**注意**:编译时打开加密开关WITH_ENCRYPTION， 并填写OpenSSL路径

## 步骤二、准备PaddlePaddle部署模型
开发者可从以下套件获取部署模型，需要注意，部署时需要准备的是导出来的部署模型，一般包含`model.pdmodel`、`model.pdiparams`和`deploy.yml`三个文件，分别表示模型结构、模型权重和各套件自行定义的配置信息。
- [PaddleDetection导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0/deploy/EXPORT_MODEL.md)
- [PaddleSeg导出模型](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0/docs/model_export.md)
- [PaddleClas导出模型](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/zh_CN/tutorials/getting_started.md#4-%E4%BD%BF%E7%94%A8inference%E6%A8%A1%E5%9E%8B%E8%BF%9B%E8%A1%8C%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)
- [PaddleX导出模型](https://github.com/PaddlePaddle/PaddleX/blob/develop/dygraph/docs/apis/export_model.md)


用户也可直接下载本教程中从PaddleDetection中导出的YOLOv3模型进行测试，[点击下载](https://bj.bcebos.com/paddlex/deploy2/models/yolov3_mbv1.tar.gz)。

## 步骤三、对模型文件进行加密

加密工具我们已经制作好，直接下载并执行指令，即可将模型加密保存到指定目录：
[windows加密工具](https://bj.bcebos.com/paddlex/deploy/windows_paddle_encrypt_tool)
[linux加密工具](https://bj.bcebos.com/paddlex/deploy/linx_paddle_encrypt_tool)

以YOLOV3模型为例，对其进行加密.
```sh
#windows为paddle_encrypt_tool.exe
./paddle_encrypt_tool --model_filename=yolov3_mbv1/model/model.pdmodel \
                      --params_filename=yolov3_mbv1/model/model.pdiparams \
                      --cfg_file=yolov3_mbv1/model/infer_cfg.yml \
                      --save_dir=yolov3_encryption

# 可加 --key 参数指定使用自己的密钥对模型进行加密， 如果不加默认为随机生产key
# 注意 --key 参数必须为原长为32位的密钥，并经过base64编码
# 例如 12345678123456781234567812345678密钥，经过base64编码后为MTIzNDU2NzgxMjM0NTY3ODEyMzQ1Njc4MTIzNDU2Nzg=。输入参数为： --key=MTIzNDU2NzgxMjM0NTY3ODEyMzQ1Njc4MTIzNDU2Nzg=
```

执行加密指令成功后会出现如下打印， 注意一定要保存好key， 部署时需要传入正确的密钥key才能部署:
```
key is 2DTPfe+K+I/hkHlDMDAoXdVzotbC8UCF9Ti0rwWd+KU=
save to yolov3_encryption
```

执行加密指令后，会在指定目录(如上指令中的yolov3_encryption)生成三个加密文件用于部署:
```
yolov3_encryption
├── encrypted.pdmodel       #模型文件，对应部署的model_filename参数
├── encrypted.pdparams      #模型参数，对应部署的params_filename参数
└── encrypted.yml           #配置文件，对应部署的cfg_file参数
```

## 步骤四、对加密模型进行预测
以步骤三中加密后的YOLOv3模型为例，执行如下命令即可进行模型加密预测

```sh
# 使用GPU 加参数 --use_gpu=1
build/demo/model_infer --model_filename=yolov3_encryption/encrypted.pdmodel \
                       --params_filename=yolov3_encryption/encrypted.pdparams \
                       --cfg_file=yolov3_encryption/model/encrypted.yml \
                       --image=yolov3_mbv1/images/000000010583.jpg \
                       --model_type=det \
                       --key=2DTPfe+K+I/hkHlDMDAoXdVzotbC8UCF9Ti0rwWd+KU=
```
**注意**：密钥key要是步骤三中使用加密工具对模型进行加密得到的key， 如果不传入key默认加载普通未加密模型。

输出结果如下(分别为类别id、标签、置信度、xmin、ymin、w, h)
```
Box(0	person	0.0386442	2.11425	53.4415	36.2138	197.833)
Box(39	bottle	0.0134608	2.11425	53.4415	36.2138	197.833)
Box(41	cup	0.0255145	2.11425	53.4415	36.2138	197.833)
Box(43	knife	0.0824398	509.703	189.959	100.65	93.9368)
Box(43	knife	0.0211949	448.076	167.649	162.924	143.557)
Box(44	spoon	0.0234474	509.703	189.959	100.65	93.9368)
Box(45	bowl	0.0461333	0	0	223.386	83.5562)
Box(45	bowl	0.0191819	3.91156	1.276	225.888	214.273)
```
### 参数说明

| 参数            | 说明                                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------ |
| model_filename  | **[必填]** 模型结构文件路径，如`yolov3_darknet/model.pdmodel`                                                |
| params_filename | **[必填]** 模型权重文件路径，如`yolov3_darknet/model.pdiparams`                                              |
| cfg_file        | **[必填]** 模型配置文件路径，如`yolov3_darknet/infer_cfg.yml`                        |
| model_type      | **[必填]** 模型来源，det/seg/clas/paddlex，分别表示模型来源于PaddleDetection、PaddleSeg、PaddleClas和PaddleX |
| image           | 待预测的图片文件路径                                                                |
| use_gpu         | 是否使用GPU，0或者1，默认为0                                                        |
| gpu_id          | 使用GPU预测时的GUI设备ID，默认为0                                                    |
| gpu_id          | 使用GPU预测时的GUI设备ID，默认为0                                                    |
| key             | 对模型加密使用的key，默认为空，只能加载未加密模型                                         |


## 相关文档

- [部署接口和数据结构文档](../apis/model.md)
