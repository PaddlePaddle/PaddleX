# 预测部署-paddlex.deploy

使用Paddle Inference进行高性能的Python预测部署。更多关于Paddle Inference信息请参考[Paddle Inference文档](https://paddle-inference.readthedocs.io/en/latest/#)

## Predictor类

图像分类、目标检测、实例分割、语义分割统一的预测器，实现高性能预测。

```python
paddlex.deploy.Predictor(model_dir, use_gpu=False, gpu_id=0, cpu_thread_num=1, use_mkl=True, mkl_thread_num=4, use_trt=False, use_glog=False, memory_optimize=True, max_trt_batch_size=1, trt_precision_mode='float32', gpu_mem=200)
```

**参数**

> * **model_dir** (str): 模型路径（必须是导出的部署或量化模型）
> * **use_gpu** (bool): 是否使用gpu，默认False
> * **gpu_id** (int): 使用gpu的id，默认0
> * **cpu_thread_num** (int)：使用cpu进行预测时的线程数，默认为1
> * **use_mkl** (bool): 是否使用mkldnn计算库，CPU情况下使用，默认False
> * **mkl_thread_num** (int): mkldnn计算线程数，默认为4
> * **use_trt** (bool): 是否使用TensorRT，默认False
> * **use_glog** (bool): 是否启用glog日志, 默认False
> * **memory_optimize** (bool): 是否启动内存优化，默认True
> * **max_trt_batch_size** (int): 在使用TensorRT时配置的最大batch size，默认1
> * **trt_precision_mode** (str)：在使用TensorRT时采用的精度，可选值['float32', 'float16']。默认'float32'
> * **gpu_mem** (int): 使用的GPU显存大小，默认为 200

### predict 接口

图片预测

```python
predict(img_file, topk=1, transforms=None, warmup_iters=0, repeats=1)
```

> **参数**
>
> > * **img_file** (List[np.ndarray or str], str or np.ndarray):
                    图像路径；或者是解码后的排列格式为（H, W, C）且类型为float32且为BGR格式的数组。
> > * **topk** (int): 分类预测时使用，表示预测前topk的结果。默认值为1。
> > * **transforms** (paddlex.transforms): 数据预处理操作。默认值为None, 即使用`model.yml`中保存的数据预处理操作。
> > * **warmup_iters** (int): 预热轮数，用于评估模型推理以及前后处理速度。若大于1，会预先重复预测warmup_iters，而后才开始正式的预测及其速度评估。默认为0。
> > * **repeats** (int): 重复次数，用于评估模型推理以及前后处理速度。若大于1，会预测repeats次取时间平均值。默认值为1。


> **返回值**
>
> > * **图像分类**模型的返回值与[图像分类模型API中predict接口](./models/classification.md#predict)的返回值一致
> > * **目标检测** 模型的返回值与[目标检测模型API中predict接口](./models/detection.md#predict)的返回值一致
> > * **实例分割** 模型的返回值与[实例分割模型API中predict接口](./models/instance_segmentation.md#predict)的返回值一致
> > * **语义分割** 模型的返回值与[语义分割模型API中predict接口](./models/semantic_segmentation.md#predict)的返回值一致
