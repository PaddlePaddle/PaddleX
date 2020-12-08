# Complete model loading

PaddleX provides a unified model loading API which supports loading models saved by PaddleX and performs evaluations on the validation set or predicts test images

## paddlex.load_model
> **Load models saved by PaddleX**

```
paddlex.load_model(model_dir)
```

### Parameters

* **model_dir**: Model path saved in the training process

### Returned value
* **paddlex.cv.models**, model type. 

### Example
> 1. [Click to download](https://bj.bcebos.com/paddlex/models/xiaoduxiong_epoch_12.tar.gz) the MaskRCNN model trained by PaddleX on Xiaoduxiong sorting data
> 2. [Click to download](https://bj.bcebos.com/paddlex/datasets/xiaoduxiong_ins_det.tar.gz) the Xiaoduxiong sorting dataset


```
import paddlex as pdx

model_dir = './xiaoduxiong_epoch_12'
data_dir = './xiaoduxiong_ins_det/JPEGImages'
ann_file = './xiaoduxiong_ins_det/val.json'

# Load a waste sorting model
model = pdx.load_model(model_dir) 

# Predict 
pred_result = model.predict('./xiaoduxiong_ins_det/JPEGImages/WechatIMG114.jpeg')

# Evaluate on the validation set
eval_reader = pdx.cv.datasets.CocoDetection(data_dir=data_dir,
                                            ann_file=ann_file,
                                            transforms=model.eval_transforms)
eval_result = model.evaluate(eval_reader, batch_size=1)
```
```
