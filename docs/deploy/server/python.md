# Python deployment

PaddleX has been integrated with a high performance prediction interface based on Python. After installing PaddleX, you can refer to the following code example to make predictions.

## Export the prediction model

You can refer to Model Export to export the model in an inference format.[ ](../export_model.md)

## Inference deployment

For the prediction interface, refer to [paddlex.deploy .](../../apis/deploy.md)

Click to download the test image [xiaoduxiong_test_image.tar.gz ](https://bj.bcebos.com/paddlex/datasets/xiaoduxiong_test_image.tar.gz)

* Single image prediction

```
import paddlex as pdx
predictor = pdx.deploy.Predictor('./inference_model')
result = predictor.predict(image='xiaoduxiong_test_image/JPEGImages/WeChatIMG110.jpeg')
```

* Batch image prediction

```
import paddlex as pdx
predictor = pdx.deploy.Predictor('./inference_model')
image_list = ['xiaoduxiong_test_image/JPEGImages/WeChatIMG110.jpeg',
    'xiaoduxiong_test_image/JPEGImages/WeChatIMG111.jpeg']
result = predictor.batch_predict(image_list=image_list)
```

* Video Stream Prediction
```
import cv2
import paddlex as pdx
predictor = pdx.deploy.Predictor('./inference_model')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        result = predictor.predict(frame)
        vis_img = pdx.det.visualize(frame, result, threshold=0.6, save_dir=None)
        cv2.imshow('Xiaoduxiong', vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
```

> Description about the prediction speed: The prediction speed of the first few images after loading the model is slow, because the initialization of the video card and memory is involved in the start-up. Generally, the prediction speed after predicting 20-30 images is stable.

## Prediction performance comparison
### Test environment

- CUDA 9.0
- CUDNN 7.5
- PaddlePaddle 1.71
- GPU: Tesla P40
- AnalysisPredictor: a Python high-performance prediction method is used.
- Executor: A common Python prediction method of the PaddlePaddle is used.
- Batch Size: It is 1, time consumption unit is ms/image. Only model runtime is calculated, excluding data pre-processing and post-processing.

### Performance comparison


| Model | AnalysisPredictor time consumption | Executor time consumption | Input image size |
| :---- | :--------------------- | :------------ | :------------ |
| resnet50 | 4.84 | 7.57 | 224*224 |
| mobilenet_v2 | 3.27 | 5.76 | 224*224 |
| unet | 22.51 | 34.60 |513*513 |
| deeplab_mobile | 63.44 | 358.31 |1025*2049 |
| yolo_mobilenetv2 | 15.20 | 19.54 |  608*608 |
| faster_rcnn_r50_fpn_1x | 50.05 | 69.58 |800*1088 |
| faster_rcnn_r50_1x | 326.11 | 347.22 | 800*1067 |
| mask_rcnn_r50_fpn_1x | 67.49 | 91.02 | 800*1088 |
| mask_rcnn_r50_1x | 326.11 | 350.94 | 800*1067 |