# Predictor部署-paddlex.deploy

使用AnalysisPredictor进行预测部署。

## Predictor类

```
paddlex.deploy.Predictor(model_dir, use_gpu=False, gpu_id=0, use_mkl=False, use_trt=False, use_glog=False, memory_optimize=True)
```

> **参数**

> > * **model_dir**: 训练过程中保存的模型路径, 注意需要使用导出的inference模型
> > * **use_gpu**: 是否使用GPU进行预测
> > * **gpu_id**: 使用的GPU序列号
> > * **use_mkl**: 是否使用mkldnn加速库
> > * **use_trt**: 是否使用TensorRT预测引擎
> > * **use_glog**: 是否打印中间日志
> > * **memory_optimize**: 是否优化内存使用

> > ### 示例
> >
> > ```
> > import paddlex
> > model = paddlex.deploy.Predictor(model_dir, use_gpu=True)
> > result = model.predict(image_file)
> > ```

### predict 接口
> ```
> predict(image, topk=1)
> ```

> **参数

* **image(str|np.ndarray)**: 待预测的图片路径或np.ndarray，若为后者需注意为BGR格式
* **topk(int)**: 图像分类时使用的参数，表示预测前topk个可能的分类
