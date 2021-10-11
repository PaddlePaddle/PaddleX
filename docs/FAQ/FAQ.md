# FAQ(常见问题)

- [GUI相关问题](#GUI相关问题)
- [API训练相关问题](#API训练相关问题)
- [推理部署问题](#推理部署问题)

## GUI相关问题
**Q:**  GUI卡死后怎么解决？

**A:**  卡死后点击一下这个按钮即可恢复正常。
<p align="center">
  <img src="./images/FAQ1.png"  alt="QR" align="middle" />
</p>


**Q:**  GUI训练时报错怎么办？

**A:**  首先打开当前项目的日志文件，查看报错信息。

例如此前将PaddleX GUI的工作空间设置在`D:/work_space`下，则根据在GUI上的项目ID和任务ID找到当前任务的日志文件，例如`D:/work_space/projects/P0001/T0001/err.log/err.log`和`D:/work_space/projects/P0001/T0001/err.log/out.log`

如果无法定位出问题，可进一步查看PaddleX GUI的系统日志：例如在`C:/User/User_name/.paddlex/logs/paddlex.log`

查看上述三个日志文件，基本可以定位出是否是显存不足、或者是数据路径不对等问题。如果是显存不足，请调低batch_size（需同时按比例调低学习率等参数）。其他无法解决的问题，可以前往GitHub[提ISSUE](https://github.com/PaddlePaddle/PaddleX/issues)，描述清楚问题会有工程师及时回复。

## API训练相关问题
**Q:**  loss为nan时怎么办？

**A:**  loss为nan表示梯度爆炸，导致loss为无穷大。这时候，需要将学习率（learning rate）调小，或者增大批大小（batch_size）。


**Q:**  YOLO系列为什么要训练这么久？

**A:**  像yolo系列的数据增强比较多，所以训练的epoch要求要多一点，具体在不同的数据集上的时候，训练参数需要调整一下。比如我们先前示例给出ppyolo，ppyolov2的训练参数都是针对COCO数据集换算到单卡上的配置，但是在昆虫这份数据集上的效果并不好，后来我们进行了调整，您可以参考我们调整的参数相应调整自己的参数，具体调了哪些可以看我们之前的[pr](https://github.com/PaddlePaddle/PaddleX/pull/853/files)。


**Q:**  用命令行跑 `.\paddlex_inference\detector.exe` 这个指令没有什么提示，也没有输出，怎么回事？

**A:**  可能是缺少dll，双击执行一下out目录下的detector.exe或model_infer.exe，会有提示。


## 推理部署问题
**Q:**  如何在程序中手动释放inference model和占用的显存?

**A:**  在主进程中初始化predictor，然后在线程里完成图片的预测，这样使用是没有问题的。线程退出后显存不会释放，主进程退出才会释放显存。线程退出后，后续显存是可以复用的，不会一直增长。


**Q:**  提高预测速度的策略都有哪些？

**A:**  1. 可以考虑使用更加轻量的backbone；看看图像预处理和预测结果后处理有没有优化空间；相比于python推理预测，用C++会更快；同时对批量图片进行预测；可以尝试使用加速库，例如在CPU上部署时可以开启mkdldnn，或者使用用OpenVINO推理引擎加速性能，在Nvidia GPU上部署时可以使用TensorRT加速性能；
2. 在测试性能时，需要注意给模型进行预热，例如先让模型预测100轮之后，再开始进行性能测试和记录，这样得到的性能指标才准确。


**Q:**  预测结果如何可视化？

**A:**  检测结果可以用`pdx.det.visualize`，分割结果可以用`pdx.seg.visualize`，API说明见[文档](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/prediction.md)


**Q:**  如何用旧的部署代码部署新的模型？

**A:**  2.0版本的cpp部署支持新旧版本的paddlex/gui导出的模型进行部署， 但是python不兼容。 GUI新旧版本也不兼容， 新版本只能加载新版本训练的模型。
