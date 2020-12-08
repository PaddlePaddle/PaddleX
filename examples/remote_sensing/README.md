# RGB remote sensing image segmentation

This case is the implementation of remote sensing image segmentation based on PaddleX, and provides a prediction method in a sliding window to avoid the occurrence of insufficient display memory in the direct prediction of large-size images. In addition, the degree of overlapping between the sliding windows can be configured. This can eliminate cracks in the final prediction results aton the window splices.

## Directory
* [Data preparation](#1)
* [Model training](#2)
* [Model predictions](#3)
* [Model evaluation](#4)

#### Pre-dependence

* PaddlePaddle >= 1.8.4
* Python >= 3.5
* PaddleX >= 1.1.4

For installation related issues, refer to [PaddleX Installation]. (../install.md)

Download PaddleX source code:

```
git clone https://github.com/PaddlePaddle/PaddleX
```

All scripts for the case are located in `PaddleX/examples/remote_sensing/`. Access the directory:

```
cd PaddleX/examples/remote_sensing/
```

## <h2 id="1">Data preparation</h2>

In this case, the high-definition remote sensing images provided by the 2015 CCF Big Data competition is used, containing five RGB images with labels with a maximum image size of 7969 × 7939 and a minimum image size of 4011 × 2470. The dataset is labeled with a total of 5 categories of objects: background (labeled 0), vegetation (labeled 1), buildings (labeled 2), bodies of water (labeled 3), and roads (labeled 4).

In this case, the first four images are categorized into the training set and the fifth image is used as the validation set. In order to increase the batch size during training, the first four images are cut with sliding window (1024, 1024) and step (512, 512). In addition to the original four large-size images, there are 688 images in the training set in total. In order to avoid the occurrence of insufficient display memory in the validation of large-size images during training, the 5th image is cut in the sliding window (769, 769) and step (769, 769) for the validation set to obtain 40 sub-images.

Run the following script to download the original dataset and complete the cut of the dataset:

```
python prepare_data.py
```

## <h2 id="2">Model training</h2>

For the split model, the Deeplabv3 model whose Backbone is set to MobileNetv3_large_ssld and the model features high performance and high precision. Run the following script to carry out the model training:
```
python train.py
```

You can also skip the model training step and directly download the pre-trained model for subsequent model prediction and evaluation.
```
wget https://bj.bcebos.com/paddlex/examples/remote_sensing/models/ccf_remote_model.tar.gz
tar -xvf ccf_remote_model.tar.gz
```

## <h2 id="3">Model predictions</h2>

The direct prediction of large-size images can lead to insufficient video memory. In order to avoid such a problem, this case provides a sliding window prediction interface, which supports both overlapping and non-overlapping methods.

* Non-overlapping sliding window prediction

The images under each window are predicted separately by sliding the windows at a fixed size on input images. Finally, the prediction results of each window are stitched together into the prediction result of the input images. Since the prediction effect of the edge part of each window is worse than that of the middle part, there may be obvious cracks in each window splice.

The API for this prediction method is described in [overlap_tile_predict] (https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#overlap-tile-predict). **You need to set the parameter `pad_size` to `[0, 0]`**.

* Overlapping sliding window prediction

In the Unet paper, the author proposes an Overlap-tile prediction strategy with overlapping sliding window to eliminate the cracking sensation at the splice. In the prediction in each sliding window, a certain area is expanded around the expanded window, such as the blue part of the area in the figure below. Only the middle part of the window is predicted in the splice, for example, the yellow part area in the figure below. The pixels under the expanded area of the window located at the edge of the input image are obtained by mirroring the pixels at the edge.

The API for this prediction method is described in [overlap_tile_predict] (https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#overlap-tile-predict).

![](images/overlap_tile.png)

Compared to the non-overlapping sliding window prediction, the overlapping sliding window prediction strategy improves the model precision miou from 80.58% to 81.52%, and eliminates the cracking sensation in the prediction visualization. See the effect comparison of the two prediction methods.

![](images/visualize_compare.jpg)

Run the following script for prediction by using overlapping sliding windows:
```
python predict.py
```

## <h2 id="4">Model evaluation</h2>

During the training process, the precision of the model in the validation set is evaluated every 10 iteration rounds. Since the original large-size images are sliced into small-size blocks beforehand, this means that a non-overlapping sliding window prediction mode is used. The optimal model precision miou is 80.58%. Run the following script to re-evaluate the model precision of the original large-size image by using the overlapping sliding window prediction method. At this time, miou is 81.52%.
```
python eval.py
```
