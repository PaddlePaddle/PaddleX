# Python预测部署
文档说明了在树莓派上使用python版本的Paddle-Lite进行PaddleX模型好的预测部署，Paddle-Lite python版本的预测库下载，用户也可以下载whl文件进行安装[Paddle-Lite_2.6.0_python](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.6.0/armlinux_python_installer.zip)，更多版本请参考[Paddle-Lite Release Note](https://github.com/PaddlePaddle/Paddle-Lite/releases)
```
python -m pip install paddlelite
```
部署前需要先将PaddleX模型转换为Paddle-Lite的nb模型，具体请参考[Paddle-Lite模型转换](./export_nb_model.md)



## 前置条件
* Python 3.6+
* Paddle-Lite_python 2.6.0+

请确保系统已经安装好上述基本软件，**下面所有示例以工作目录 `/root/projects/`演示**。

## 预测部署
运行/root/projects/PaddleX/deploy/raspberry/python目录下demo.py文件可以进行预测，其命令参数说明如下：

|  参数   | 说明  |
|  ----  | ----  |
| --model_dir  | 模型转换生成的.xml文件路径，请保证模型转换生成的三个文件在同一路径下|
| --img  | 要预测的图片文件路径 |
| --image_list  | 按行存储图片路径的.txt文件 |
| --cfg_dir | PaddleX model 的.yml配置文件 |
| --thread_num  | 预测的线程数, 默认值为1 |
| --input_shape  | 模型输入中图片输入的大小[N,C,H.W] |
  
### 样例
`样例一`：  
测试图片 `/path/to/test_img.jpeg`  

```
cd /root/projects/python  

python demo.py --model_dir /path/to/openvino_model --img /path/to/test_img.jpeg --cfg_dir /path/to/PadlleX_model.yml --thread_num 4 --input_shape [1,3,224,224]
```  

样例二`:

预测多个图片`/path/to/image_list.txt`，image_list.txt内容的格式如下：

```
/path/to/images/test_img1.jpeg
/path/to/images/test_img2.jpeg
...
/path/to/images/test_imgn.jpeg
```

```
cd /root/projects/python  

python demo.py --model_dir /path/to/models/openvino_model --image_list /root/projects/images_list.txt --cfg_dir=/path/to/PadlleX_model.yml --thread_num 4 --input_shape [1,3,224,224]
```