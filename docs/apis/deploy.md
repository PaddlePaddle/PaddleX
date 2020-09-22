# 预测部署-paddlex.deploy

使用Paddle Inference进行高性能的Python预测部署。更多关于Paddle Inference信息请参考[Paddle Inference文档](https://paddle-inference.readthedocs.io/en/latest/#)

## Predictor类

图像分类、目标检测、实例分割、语义分割统一的预测器，实现高性能预测。

```
paddlex.deploy.Predictor(model_dir, use_gpu=False, gpu_id=0, use_mkl=False, mkl_thread_num=4, use_trt=False, use_glog=False, memory_optimize=True)
```

**参数**

> * **model_dir** (str): 导出为inference格式的模型路径。
> * **use_gpu** (bool): 是否使用GPU进行预测。
> * **gpu_id** (int): 使用的GPU序列号。
> * **use_mkl** (bool): 是否使用mkldnn加速库。
> * **mkl_thread_num** (int): 使用mkldnn加速库时的线程数，默认为4
> * **use_trt** (boll): 是否使用TensorRT预测引擎。
> * **use_glog** (bool): 是否打印中间日志。
> * **memory_optimize** (bool): 是否优化内存使用。

> ### 示例
>
> ```
> import paddlex
>
> model = paddlex.deploy.Predictor(model_dir, use_gpu=True)
> result = model.predict(image_file)
> ```

### predict 接口

```
predict(image, topk=1)
```

单张图片预测接口。

> **参数**
>
> > * **image** (str|np.ndarray): 待预测的图片路径或numpy数组(HWC排列，BGR格式)。
> > * **topk** (int): 图像分类时使用的参数，表示预测前topk个可能的分类。

### batch_predict 接口
```
batch_predict(image_list, topk=1)
```
批量图片预测接口。

> **参数**
>
> > * **image_list** (list|tuple): 对列表（或元组）中的图像同时进行预测，列表中的元素可以是图像路径或numpy数组(HWC排列，BGR格式)。
> > * **topk** (int): 图像分类时使用的参数，表示预测前topk个可能的分类。
