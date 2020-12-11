# Dataset tools

## Dataset analysis

### paddlex.datasets.analysis.Seg
```python
paddlex.datasets.analysis.Seg(data_dir, file_list, label_list)
```

Construct the analyzer of statistical analysis semantic classification dataset.

> **Parameters**
>
> > * **data_dir** (str): The directory path where the dataset is located.  
> > * **file_list** (str): Describes the file path of the image file and category id of the dataset （the path of each line in the text is the relative path of the relative`data_dir`）。  
> > * **label_list** (str): Describes the path of the category information file contained in the dataset.  

#### analysis
```python
analysis(self)
```

The analysis interface of SEG analyzer completes the analysis and statistics of the following information:

> * Number of images
> * Maximum and minimum size of image
> * Number of image channels
> * The minimum and maximum values of each channel of the image
> * Pixel value distribution of each channel of the image
> * Mean and variance of normalized image channels
> * Mark the number and proportion of each category in the diagram

[Code Example](https://github.com/PaddlePaddle/PaddleX/blob/develop/examples/multi-channel_remote_sensing/tools/analysis.py)

[Sample Example](../../examples/multi-channel_remote_sensing/analysis.html#id2)

#### cal_clipped_mean_std
```python
cal_clipped_mean_std(self, clip_min_value, clip_max_value, data_info_file)
```

SEG analyzer is used to calculate the mean and variance of image after truncation.

> Parameters
>
> > * **clip_min_value** (list):  The lower limit of truncation and the values less than min_val are set as min_val.
> > * **clip_max_value** (list): The upper limit of truncation and the value greater than max_val is set as max_val.
> > * **data_info_file** (str): The path of the analysis result file (named `train_information.pkl`)saved in the analysis() interface.

[Code Example](https://github.com/PaddlePaddle/PaddleX/blob/develop/examples/multi-channel_remote_sensing/tools/cal_clipped_mean_std.py)

[Calculation Results Example](../examples/multi-channel_remote_sensing/analysis.html#id4)

## Dataset generation

### paddlex.det.paste_objects
```python
paddlex.det.paste_objects(templates, background, save_dir='dataset_clone')
```

Paste the target object on the background image to generate a new image and annotation file

> **Parameters**
>
> > * **templates** (list|tuple)：The target objects on multiple images can be pasted on the same background image at the same time, so templates is a list, in which each element is a dict, which represents the target object of a picture. The target object of an image has two keywords `image` and`annos`. The key value of `image` is the path of the image, or it is an array of decoded array format (H, W, C) of uint8 and BGR format. There can be multiple target objects on the image, so the key value of `annos` is a list. Each element in the list is a dict, which represents the information of a target object. The dict contains two keywords`polygon`and`category`, where`polygon`represents the edge coordinates of the target object, such as [[0, 0], [0, 1], [1, 1], [1, 0]], and`category`represents the category of the target object, such as' dog'.
> > * **background** (dict): Background images can have true values, so background is a dict, which contains the keywords`image`and`annos`. The key value of `image` is the path of the background image, or it is an array of decoded array format (H, W, C) with uint8 type and BGR format. If there is no true value on the background image, the key value of `annos` is an empty list [], if there is one, the key value of`annos` is a list composed of multiple dicts. Each dict represents the information of an object, including the keywords`bbox`and`category`. The key value of`bbox`is the coordinates of the upper left corner and the lower right corner of the object frame, i.e. [x1, Y1, X2, Y2], and`category`represents the category of the target object, such as' dog'.
> > * **save_dir** (str)：Storage directory for new pictures and their annotation files. The default value is`dataset_clone`.

> Code Example

```python
import paddlex as pdx
templates = [{'image': 'dataset/JPEGImages/budaodian-10.jpg',
              'annos': [{'polygon': [[146, 169], [909, 169], [909, 489], [146, 489]],
                        'category': 'lou_di'},
                        {'polygon': [[146, 169], [909, 169], [909, 489], [146, 489]],
                        'category': 'lou_di'}]}]
background = {'image': 'dataset/JPEGImages/budaodian-12.jpg', 'annos': []}
pdx.det.paste_objects(templates, background, save_dir='dataset_clone')
```
