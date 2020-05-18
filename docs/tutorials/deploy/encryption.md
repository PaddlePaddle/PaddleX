# Paddle模型加密方案

飞桨团队推出模型加密方案，使用业内主流的AES加密技术对最终模型进行加密。飞桨用户可以通过PaddleX导出模型后，使用该方案对模型进行加密，预测时使用解密SDK进行模型解密并完成推理，大大提升AI应用安全性和开发效率。

## 1. 方案介绍

### 1.1 工具组成

[链接](http://wiki.baidu.com/pages/viewpage.action?pageId=1128566963)

下载并解压后，目录包含内容为：
```
paddle_model_encrypt
├── include # 头文件：paddle_model_decrypt.h(解密)和paddle_model_encrypt.h(加密)
|
├── lib # libpmodel-encrypt.so和libpmodel-decrypt.so动态库
|
└── tool # paddle_encrypt_tool
```

### 1.2 二进制工具

#### 1.2.1 生成密钥

产生随机密钥信息（用于AES加解密使用）（32字节key + 16字节iv， 注意这里产生的key是经过base64编码后的，这样可以扩充选取key的范围）

```
paddle_encrypt_tool    -g
```
#### 1.2.1 文件加密

```
 paddle_encrypt_tool    -e    -key    keydata     -infile    infile    -outfile    outfile
```

#### 1.3 SDK

```
// 加密API
int paddle_encrypt_model(const char* keydata, const char* infile, const char* outfile);
// 加载加密模型API：
int paddle_security_load_model(
        paddle::AnalysisConfig *config,
        const char *key,
        const char *model_file,
        const char *param_file);
```

## 2. PaddleX C++加密部署
