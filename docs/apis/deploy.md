# Prediction deployment-paddlex.deploy

Paddle Inference is used to perform high-performance Python prediction deployment. For more information on Paddle Inference, refer to the [Paddle inference document] (https://paddle-inference.readthedocs.io/en/latest/#)

## Predictor class

Predictor for image classification, object detection, instance segmentation and segmentation to achieve high-performance prediction.

```
paddlex.deploy. Predictor(model_dir, use_gpu=False, gpu_id=0, use_mkl=False, mkl_thread_num=4, use_trt=False, use_glog=False, memory_optimize=True)
```

**Parameters**

> * **model_dir** (str): Path to a model that has been exported in prediction format.
> * **use_gpu** (bool): Whether to use a GPU to perform prediction.
> * **gpu_id** (int): Used GPU serial number.
> * **use_mkl** (bool): Whether to use an mkldnn acceleration library.
> * **mkl_thread_num** (int): Number of threads when an mkldnn library is used. It is 4 by default.
> * **use_trt** (boll): Whether to use a TensorRT prediction engine.
> * **use_glog** (bool): Whether to print an intermediate log.
> * **memory_optimize** (bool): Whether to optimize memory utilization.


> ### Example
> 
> ```
> import paddlex
>
> model = paddlex.deploy.Predictor(model_dir, use_gpu=True)
> result = model.predict(image_file)
> ```

### predict API

```
predict(image)
```

Single-image prediction API.

> **Parameters**
> * **image** (str|np.ndarray): Image path or numpy array to be predictred (HWC arrangement, BGR format).



### batch_predict API
```
batch_predict(image_list)
```
Batch image prediction API.

> **Parameters**
> * **image_list** (list|tuple): Simultaneously predicts images in the list (or tuple). Elements in the list may be image paths or numpy arrays (HWC arrangement, BGR format).
> > * **topk** (int): Parameter used during image classification, which indicates topk potential classifications before prediction.


