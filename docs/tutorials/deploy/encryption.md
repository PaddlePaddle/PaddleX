# Paddle模型加密方案

飞桨团队推出模型加密方案，使用业内主流的AES加密技术对最终模型进行加密。飞桨用户可以通过PaddleX导出模型后，使用该方案对模型进行加密，预测时使用解密SDK进行模型解密并完成推理，大大提升AI应用安全性和开发效率。
** 注意：目前加密方案仅支持Linux系统**

## 1. 方案介绍

### 1.1 工具组成

[PaddleX模型加密SDK下载](https://bj.bcebos.com/paddlex/tools/paddlex-encryption.zip)

下载并解压后，目录包含内容为：
```
paddlex-encryption
├── include # 头文件：paddle_model_decrypt.h(解密)和paddle_model_encrypt.h(加密)
|
├── lib # libpmodel-encrypt.so和libpmodel-decrypt.so动态库
|
└── tool # paddlex_encrypt_tool
```

### 1.2 加密PaddleX模型

模型加密后，会产生随机密钥信息（用于AES加解密使用），该key值需要在模型加载时传入作为解密使用。
> 32字节key + 16字节iv， 注意这里产生的key是经过base64编码后的，这样可以扩充选取key的范围
```
./paddlex-encryption -model_dir paddlex_inference_model -save_dir paddlex_encrypted_model
```
模型在加密后，会保存至指定的`-save_dir`下，同时生成密钥信息，命令输出如下图所示，密钥为`33NRtxvpDN+rkoiECm/e1Qc7sDlODdac7wp1m+3hFSU=`
![](images/encryt.png)

## 2. PaddleX C++加密部署
