# 图像分类训练示例

本目录下为图像分类示例代码，用户在安装完PaddlePaddle和PaddleX即可直接进行训练。

- [PaddlePaddle安装](https://www.paddlepaddle.org.cn/install/quick)
- [PaddleX安装](https://paddlex.readthedocs.io/zh_CN/develop/install.html)

## 模型训练
如下所示，直接下载代码后运行即可，代码会自动下载训练数据
```
python mobilenetv3_small_ssld.py
```

## VisualDL可视化训练指标
在模型训练过程，在`train`函数中，将`use_vdl`设为True，则训练过程会自动将训练日志以VisualDL的格式打点在`save_dir`（用户自己指定的路径）下的`vdl_log`目录，用户可以使用如下命令启动VisualDL服务，查看可视化指标
```
visualdl --logdir output/mobilenetv3_small_ssld/vdl_log --port 8001
```

服务启动后，使用浏览器打开 https://0.0.0.0:8001 或 https://localhost:8001 
