# Multi-channel remote sensing image segmentation
Remote sensing image segmentation is an important application scene in the field of image segmentation, and is widely used in land surveying and mapping, environmental monitoring, urban construction and other fields. The targets of remote sensing image segmentation are diversified, such as snow, crops, roads, buildings, water sources and other features, as well as air targets (for example, cloud).

This case implements multi-channel remote sensing image segmentation based on PaddleX, covering data analysis, model training, model prediction and other processes, and aims to help users solve the multi-channel remote sensing image segmentation problem by using the deep learning technology.

## Directory
* [Pre-dependence](#1)
* [Data preparation](#2)
* [Data analysis](#3)
* [Model training](#4)
* [Model predictions](#5)

## <h2 id="1">Pre-dependence</h2>

* PaddlePaddle >= 1.8.4
* Python >= 3.5
* PaddleX >= 1.1.4

For installation related issues, refer to [PaddleX Installation]. (../../docs/install.md)

In addition, you need to install gdal**. There may be an error in the installation of gdal by using pip. It is recommended to use conda to install gdal.

```
conda install gdal
```

Download PaddleX source code:

```
git clone https://github.com/PaddlePaddle/PaddleX
```

All scripts for the case are located in `PaddleX/examples/channel_remote_sensing/`. Access the directory.

```
cd PaddleX/examples/channel_remote_sensing/
```

## <h2 id="2">Data preparation</h2>

Remote sensing images are available in a variety of formats, and the formats of the data produced by different sensors may vary. PaddleX is now compatible with the following four formats for image reading:

- `tif`
- `png`, `jpeg`, `bmp`
- `img`
- `npy`

The annotation map must be a single-channel image in png format, the pixel value is the corresponding category, and the pixel annotation category needs to be incremented from 0. For example, 0, 1, 2, and 3 indicate that there are 4 categories, and 255 is used to specify pixels that are not involved in training and evaluation, and the maximum number of label categories is 256.

This case uses the [L8 SPARCS public dataset] (https://www.usgs.gov/land-resources/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation) for segmentation of cloud and snow. The dataset contains 80 satellite images, covering 10 bands. The original annotated images contain 7 classes, that is, `cloud`, `cloud shadow`, `shadow over water`, `snow/ice`, `water`, `land`, and `flooded`. Since `flooded` and `shadow over water` account for only `1.8%` and `0.24%`, `flooded` is merged into `land` and `shadow over water` is merged into `shadow`. After the merge, there are five classes in total.

The corresponding table of value, class, and color:

|Pixel value|Class|Color|
|---|---|---|
|0|cloud|white|
|1|shadow|black|
|2|snow/ice|cyan|
|3|water|blue|
|4|land|grey|

<p align="center">
 <img src="./docs/images/dataset.png" align="middle"
</p>

<p align='center'>
L8 SPARCS dataset example
</p>

Run the following command to download and decompress the class-merged dataset.
```shell script
mkdir dataset && cd dataset
wget https://paddleseg.bj.bcebos.com/dataset/remote_sensing_seg.zip
unzip remote_sensing_seg.zip
cd . .
```
The `data` directory stores remote sensing images, the `data_vis` directory stores color composite preview images, and the `mask` directory stores labeled images.

## <h2 id="2">Data analysis</h2>

Remote sensing images are often composed of many wavelengths, and the distribution of data in different wavelengths may vary greatly, for example, the visible and thermal infrared wavelengths are differently distributed. In order to better understand the distribution of the data to optimize the training effect of the model, the data needs to be analyzed.

[Statistical analysis] (./docs/analysis.md) is performed on the training set with reference to the document data, to determine the truncation range of image pixels, and to make statistics of the mean and variance of the truncated data.

## <h2 id="2">Model training</h2>

In this case, the `UNet` semantic segmentation model is selected to complete the segmentation of clouds and snow. Perform the following steps to complete the model training. The optimal precision of the model `miou` is `78.38%`.

* Set the GPU card number.
```shell script
export CUDA_VISIBLE_DEVICES=0
```

* Run the following script to start training:
```shell script
python train.py --data_dir dataset/remote_sensing_seg \
--train_file_list dataset/remote_sensing_seg/train.txt \
--eval_file_list dataset/remote_sensing_seg/val.txt \
--label_list dataset/remote_sensing_seg/labels.txt \
--save_dir saved_model/remote_sensing_unet \
--num_classes 5 \
--channel 10 \
--lr 0.01 \
--clip_min_value 7172 6561 5777 5103 4291 4000 4000 4232 6934 7199 \
--clip_max_value 50000 50000 50000 50000 50000 40000 30000 18000 40000 36000 \
--mean 0.15163569 0.15142828 0.15574491 0.1716084  0.2799778  0.27652043 0.28195933 0.07853807 0.56333154 0.5477584 \
--std  0.09301891 0.09818967 0.09831126 0.1057784  0.10842132 0.11062996 0.12791838 0.02637859 0.0675052  0.06168227 \
--num_epochs 500 \
--train_batch_size 3
```

It is also possible to skip the model training step and download pre-training models for direct model prediction.

```
wget https://bj.bcebos.com/paddlex/examples/multi-channel_remote_sensing/models/l8sparcs_remote_model.tar.gz
tar -xvf l8sparcs_remote_model.tar.gz
```

## <h2 id="2">Model predictions</h2>
Run the following script to predict the remote sensing image and visualize the prediction result, and also visualize the corresponding label file to compare the prediction effect.

```shell script
export CUDA_VISIBLE_DEVICES=0 python predict.py
```
The visualization effect is as follows:


<img src="./docs/images/prediction.jpg" alt="Prediction graph" align="center" />


The corresponding table of value, class, and color:

|Pixel value|Class|Color|
|---|---|---|
|0|cloud|white|
|1|shadow|black|
|2|snow/ice|cyan|
|3|water|blue|
|4|land|grey|
