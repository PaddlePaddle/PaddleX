# Quick start within 10 minutes

This document shows how to perform training on a small dataset through PaddleX. This example is synchronized to AIStudio. You can directly [experience this model training online] (https://aistudio.baidu.com/aistudio/projectdetail/450220).

The codes of this example are derived from Github [tutorials/train/classification/mobilenetv3_small_ssld.py] (https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/mobilenetv3_small_ssld.py). You can download and run them locally.

All model trainings in PaddleX follow the following three steps to quickly finish the development of training codes.

| Steps |  | Description |
| :--- | :--------------- | :-------------- |
| Step 1 | <a href=#Define a training/validation image processing flow transforms>Define transforms</a> | Used to define <br>input image preprocessing and data enhancement operations during model training, validation and inference|
| Step 2 | <a href="#Define a dataset and load an image classification dataset">Define datasets</a> | Used to define model training and validation datasets to be loaded|
| Step 3 | <a href="#Start training using the MobileNetV3_small_ssld model">Define models and start training</a> | Select any required models and perform training|

> **Note**: The transforms, datasets and training parameters of different models are quite different. For more model trainings, you can get more model training codes directly from the tutorial. [Model training tutorial] (train/index.html)

Other usages of PaddleX

- <a href="#View a change in training indexes using VisualDL during training">Use VisualDL to view an index change during training</a>
- <a href="#Load a model saved during training and perform inference">Load a model saved during training and perform inference</a>


<a name="Install PaddleX"></a>
**1 Install PaddleX**
> For the installation-related process and problems, refer to the PaddleX [installation document]. (./install.md)
```
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```

<a name="Prepare a vegetable classification dataset"></a>
**2 Prepare a vegetable classification dataset**
```
wget https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz
tar xzvf vegetables_cls.tar.gz
```

<a name="Define a training/validation image processing flow transforms"></a>
**3 Define a training/validation image processing flow transforms**

Model data processing flows must be respectively defined during training and validation because data enhancement operations are added during training. [RandomCrop](apis/transforms/cls_transforms.html#randomcrop) and [RandomHorizontalFlip](apis/transforms/cls_transforms.html#randomhorizontalflip) data enhancement methods are added in train_transforms`, as shown in the following codes. For more methods, refer to the [data enhancement document] (apis/transforms/augment.md).
```
from paddlex.cls import transforms
train_transforms = transforms. Compose([
    transforms. RandomCrop(crop_size=224),
    transforms. RandomHorizontalFlip(),
    transforms. Normalize() 
]) 
eval_transforms = transforms. Compose([
    transforms. ResizeByShort(short_size=256),
    transforms. CenterCrop(crop_size=224),
    transforms. Normalize()
])
```

<a name="Define a dataset and load an image classification dataset"></a>
**4 Define a `dataset` and load an image classification dataset**

Define a dataset. `pdx.datasets. ImageNet` inidicates reading a classification dataset in ImageNet format
- [paddlex.datasets. ImageNet API description](apis/datasets.md)
- [ImageNet data format description](data/format/classification.md)

```
train_dataset = pdx.datasets. ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/train_list.txt',
    label_list='vegetables_cls/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets. ImageNet(
     data_dir='vegetables_cls',
     file_list='vegetables_cls/val_list.txt',
     label_list='vegetables_cls/labels.txt',
     transforms=eval_transforms)
```

<a name="Start training using the MobileNetV3_small_ssld model"></a>
**5 Start training using the MobileNetV3_small_ssld model**

In this document, the MobileNetV3 pre-training model obtained by Baidu based on the distillation method is used. The model structure is the same as MobileNetV3, but the precision is higher. PaddleX has more than 20 built-in classification models. For the details of more classification models, refer to the [PaddleX model library] (appendix/model_zoo.md).
```
num_classes = len(train_dataset.labels)
model = pdx.cls. MobileNetV3_small_ssld(num_classes=num_classes)

model.train(num_epochs=20,
            train_dataset=train_dataset,
            train_batch_size=32,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[4, 6, 8],
            save_dir='output/mobilenetv3_small_ssld',
            use_vdl=True)
```

<a name="View a change in training indexes using VisualDL during training"></a>
**6 View a change in training indexes using VisualDL during training**

Model indexes on both the training and validation sets are outputted to a command terminal in the form of standard output stream during training When you set `use_vdl=True`, indexes are also sent to the `vdl_log` folder in the `save_dir` directory in VisualDL format. Run the following command in the terminal to start visualdl and view a visual index change.
```
visualdl --logdir output/mobilenetv3_small_ssld --port 8001
```
After the service is started, open https://0.0.0.0:8001 or https://localhost:8001 on the browser.

If you use the AIStudio platform for training, you cannot start visualdl using this method. Refer to the AIStudio VisualDL start tutorial

<a name="Load a model saved during training and perform inference"></a>
**7 Load a model saved during training and perform inference**

A model is saved every certain number of rounds during training. The round with the best evaluation on the validation set is saved in the `best_model` folder in the `save_dir` directory. The following method is used to load a model and perform inference.
- [load_model API description](apis/load_model.md)
- [predict API description for a classification model](apis/models/classification.html#predict)
```
import paddlex as pdx
model = pdx. load_model('output/mobilenetv3_small_ssld/best_model')
result = model.predict('vegetables_cls/bocai/100.jpg')
print("Predict Result:", result)
```
The inference results are outputted as follows:
```
Predict Result:Predict Result:[{'score':0.9999393, 'category':'bocai', 'category_id':0}]
```

<a name="More tutorials"></a>
**More tutorials**
- 1 [Object detection model training](train/object_detection.md)
- 2 [Semantic segmentation model training](train/semantic_segmentation.md)
- 3 [Instance segmentation model training](train/instance_segmentation.md)
- 4 [If a model is too large and you want to have a small model, try to prune it.](https://github.com/PaddlePaddle/PaddleX/tree/develop/tutorials/compress)
