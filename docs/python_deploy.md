# Python部署

PaddleX已经集成了基于Python的高性能预测接口，在安装PaddleX后，可参照如下代码示例，进行预测。

## 部署模型导出

在服务端部署模型时需要将训练过程中保存的模型导出为inference格式模型，具体的导出步骤请参考文档[部署模型导出](./apis/export_model.md)将模型导出为inference格式。

## 预测部署

接下来的预测部署将使用PaddleX python高性能预测接口，接口说明可参考[paddlex.deploy](./apis/deploy.md)


* 图片预测

```python
import paddlex as pdx
predictor = pdx.deploy.Predictor('./inference_model')
result = predictor.predict(img_file='test.jpg')
```

* 图片预测、并评估预测速度

**关于预测速度的说明**：加载模型后，前几张图片的预测速度会较慢，这是因为运行启动时涉及到内存显存初始化等步骤，通常在预测20-30张图片后模型的预测速度达到稳定。**如果需要评估预测速度，可通过指定预热轮数warmup_iters完成预热**。**为获得更加精准的预测速度，可指定repeats重复预测后取时间平均值**。

```python
import paddlex as pdx
predictor = pdx.deploy.Predictor('./inference_model')
result = predictor.predict(img_file='test.jpg',
                           warmup_iters=100,
                           repeats=100)
```

* 预测结果可视化
 
Python部署所得预测结果支持使用`paddlex.det.visualize`（适用于目标检测和实例分割模型）或`paddlex.seg.visualize`（适用于语义分割模型）进行可视化。
```python
# 目标检测和实例分割结果
pdx.det.visualize('test.jpg', result, save_dir='./')

# 语义分割结果
pdx.seg.visualize('test.jpg', result, save_dir='./')
```
