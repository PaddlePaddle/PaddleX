# Python预测部署
文档说明了在python下基于OpenVINO的预测部署，部署前需要先将paddle模型转换为OpenVINO的Inference Engine，请参考[模型转换](./export_openvino_model.md)。目前CPU硬件上支持PadlleX的分类、检测、分割模型；VPU上支持PaddleX的分类模型。

## 前置条件
* Python 3.6+
* OpenVINO 2021.1

**说明**：OpenVINO安装请参考[OpenVINO](https://docs.openvinotoolkit.org/latest/index.html)  


请确保系统已经安装好上述基本软件，**下面所有示例以工作目录 `/root/projects/`演示**。

## 预测部署
运行/root/projects/PaddleX/deploy/openvino/python目录下demo.py文件可以进行预测，其命令参数说明如下：

|  参数   | 说明  |
|  ----  | ----  |
| --model_dir  | 模型转换生成的.xml文件路径，请保证模型转换生成的三个文件在同一路径下|
| --img  | 要预测的图片文件路径 |
| --image_list  | 按行存储图片路径的.txt文件 |
| --device  | 运行的平台, 默认值为"CPU" |
| --cfg_file | PaddleX model 的.yml配置文件 |

### 样例
`样例一`：  
测试图片 `/path/to/test_img.jpeg`  

```
cd /root/projects/python  

python demo.py --model_dir /path/to/openvino_model --img /path/to/test_img.jpeg --cfg_file /path/to/PadlleX_model.yml
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

python demo.py --model_dir /path/to/models/openvino_model --image_list /root/projects/images_list.txt --cfg_file=/path/to/PadlleX_model.yml
```
