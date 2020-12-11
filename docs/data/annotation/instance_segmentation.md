# Instance Segmentation

Labelme annotation tool is recommended for instance segmentation data annotation. If you have not previously installed labelme, please refer to [Labelme installation and startup](labelme.md) for the installation of labelme

**Note: LabelMe is not friendly to Chinese support, so please do not appear Chinese characters in the following path and file name!**

## Preparation     

1. Store the collected images in the `JPEGImages` folder, for example, in `D:\MyDataset\JPEGImages`
2. Create a folder `Annotations` corresponding to the image folder to store annotated JSON files, such as `D:MyDataset\Annotations`
3. Open LabelMe, click the "Open Dir" button, select the folder where the image to be labeled is opened, and the "File List" dialog box will display the absolute path corresponding to all images, and then you can start to traverse each image and label      

## Target edge annotation  

1. Open polygon annotation tool (right-click menu > Create Polygon) to circle the outline of the target by dot, and write the corresponding label in the pop-up dialog box（Click when the label already exists. Please note that the label should not be used in Chinese），Specifically, as shown below, when the box is marked incorrectly, you can click "Edit Polygons" on the left, and then click the label box to modify it by dragging, or click "Delete Polygon" to delete it.
![](./pics/detection2.png)

2. Click "Save" on the right to save the annotation results to the Annotations directory created in

## Format conversion

LabelMe annotated data needs to be converted to MSCOCO format before it can be used for instance segmentation task training. Create the save directory`D:\dataset_seg`. After installing paddlex in Python environment, use the following command 
```
paddlex --data_conversion --source labelme --to MSCOCO \
        --pics D:\MyDataset\JPEGImages \
        --annotations D:\MyDataset\Annotations \
        --save_dir D:\dataset_coco
```

## Dataset partition

After data conversion, in order to train, the data needs to be divided into training set, verification set and test set. After installing paddlex, the data can be divided into 70% training set, 20% verification set and 10% test set by using the following command
```
paddlex --split_dataset --format COCO --dataset_dir D:\MyDataset --val_value 0.2 --test_value 0.1
```
If you execute the above command line, `train.json`, `val.json`, `test.json` will be generated under`D:\MyDataset`, which will store training sample information, verification sample information and test sample information respectively

> Note: if you use PaddleX visual client for model training, the data set partition function is integrated in the client, and there is no need to use command partition by yourself


- [Instance segmentation task training example code](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/instance_segmentation/mask_rcnn_r50_fpn.py)

