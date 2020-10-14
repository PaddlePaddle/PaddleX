# Python预测部署
文档说明了在树莓派上使用Python版本的Paddle-Lite进行PaddleX模型好的预测部署，根据下面的命令安装Python版本的Paddle-Lite预测库，若安装不成功用户也可以下载whl文件进行安装[Paddle-Lite_2.6.0_python](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.6.0/armlinux_python_installer.zip)，更多版本请参考[Paddle-Lite Release Note](https://github.com/PaddlePaddle/Paddle-Lite/releases)
```
python -m pip install paddlelite
```
部署前需要先将PaddleX模型转换为Paddle-Lite的nb模型，具体请参考[Paddle-Lite模型转换](./export_nb_model.md)
**注意**：若用户使用2.6.0的Python预测库，请下载2.6.0版本的opt转换工具转换模型



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
| --cfg_file | PaddleX model 的.yml配置文件 |
| --thread_num  | 预测的线程数, 默认值为1 |  

**注意**：由于Paddle-lite的python api尚不支持int64数据的输入，目前树莓派在python下不支持部署YoloV3，如需要请使用C++代码部署YoloV3模型

### 样例
`样例一`：  
测试图片 `/path/to/test_img.jpeg`  

```
cd /root/projects/python  

python demo.py --model_dir /path/to/openvino_model --img /path/to/test_img.jpeg --cfg_file /path/to/PadlleX_model.yml --thread_num 4 
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

python demo.py --model_dir /path/to/models/openvino_model --image_list /root/projects/images_list.txt --cfg_file=/path/to/PadlleX_model.yml --thread_num 4 
```
