# Instance Segmentation MSCOCO

## Dataset folder structure

In PaddleX, the instance segmentation supports the MSCOCO dataset format (MSCOCO format can also be used for the object detection). It is recommended to organize the datasets in the following way: original images are in the same directory (for example, JPEGImages), and the annotations files (for example, annotations.json) are in the same level directory as JPEGImages. The instance structure is as follows:
```
MyDataset/ # Instances segmentation dataset root directory |--JPEGImages/ # The directory where the original image files are located. |--1.jpg |--2.jpg |--. . . |--. . . | |--annotations.json # Directory of annotation files
```

## Divide the training set and validation sets

In PaddleX, to distinguish between the training set and the validation set, different json files are used to indicate the segmentation of data in the same level directory as `MyDataset`, for example, `train .`json` and val.json`. [Click to download the instance segmentation and instance dataset](https://bj.bcebos.com/paddlex/datasets/garbage_ins_det.tar.gz)

> Note: You can also use PaddleX's own tool to randomly divide the dataset. After the dataset is organized in the above format, run the following commands to quickly complete the random division of the dataset, where val_value indicates the ratio of the validation set, and test_value indicates the ratio of the test set (which can be 0), and the remaining ratio is used for the training set. ****
> ```
> paddlex --split_dataset --format COCO --dataset_dir MyDataset --val_value 0.2 --test_value 0.1
> ```

MSCOCO data annotation files are in the JSON format. Users can use the annotation tools such as Labelme, Wizard Annotation Assistant or EasyData to annotate. For details, see the data annotation tool.[](../annotation.md)

## PaddleX loads dataset
Example codes are as follows:
```
import paddlex as pdx from paddlex.det import transforms train_transforms = transforms. Compose([ transforms. RandomHorizontalFlip(), transforms. Normalize(), transforms. ResizeByShort(short_size=800, max_size=1333), transforms. Padding(coarsest_stride=32) ]) eval_transforms = transforms. Compose([ transforms. Normalize(), transforms. ResizeByShort(short_size=800, max_size=1333), transforms. Padding(coarsest_stride=32), ]) train_dataset = pdx.dataset. CocoDetection( data_dir = '. /MyDataset/JPEGImages', ann_file = '. /MyDataset/train.json', transforms=train_transforms) eval_dataset = pdx.dataset. CocoDetection( data_dir = '. /MyDataset/JPEGImages', ann_file = '. /MyDataset/val.json', transforms=eval_transforms)
```
