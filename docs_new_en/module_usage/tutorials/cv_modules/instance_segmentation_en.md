# Instance Segmentation Module Development Tutorial

## I. Overview
The instance segmentation module is a crucial component in computer vision systems, responsible for identifying and marking pixels that contain specific object instances in images or videos. The performance of this module directly impacts the accuracy and efficiency of the entire computer vision system. The instance segmentation module typically outputs pixel-level masks (masks) for each target instance, which are then passed as input to the object recognition module for subsequent processing.

## II. Supported Model List

<details>
   <summary> 👉Model List Details</summary>

<table>
    <tr>
        <th>Model</th>
        <th>Mask AP</th>
        <th>GPU Inference Time (ms)</th>
        <th>CPU Inference Time</th>
        <th>Model Size (M)</th>
        <th>Description</th>
    </tr>
 <tr>
        <td>Cascade-MaskRCNN-ResNet50-FPN</td>
        <td>36.3</td>
        <td >-</td>
        <td >-</td>
        <td>254.8</td>
        <td rowspan="2">Cascade-MaskRCNN is an improved Mask RCNN instance segmentation model that utilizes multiple detectors in a cascade, optimizing segmentation results by leveraging different IOU thresholds to address the mismatch between detection and inference stages, thereby enhancing instance segmentation accuracy.</td>
    </tr>
    <tr>
        <td>Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN</td>
        <td>39.1</td>
        <td >-</td>
        <td >-</td>
        <td>254.7</td>
    </tr>
    <tr>
        <td>Mask-RT-DETR-H</td>
        <td>50.6</td>
        <td>132.693</td>
        <td>4896.17</td>
        <td>449.9</td>
        <td rowspan="5">Mask-RT-DETR is an instance segmentation model based on RT-DETR. By adopting the high-performance PP-HGNetV2 as the backbone network and constructing a MaskHybridEncoder encoder, along with introducing IOU-aware Query Selection technology, it achieves state-of-the-art (SOTA) instance segmentation accuracy with the same inference time.</td>
    </tr>
    <tr>
        <td>Mask-RT-DETR-L</td>
        <td>45.7</td>
        <td>46.5059</td>
        <td>2575.92</td>
        <td>113.6</td>
    </tr>
    <tr>
        <td>Mask-RT-DETR-M</td>
        <td>42.7</td>
        <td>36.8329</td>
        <td>-</td>
        <td>66.6 M</td>
    </tr>
    <tr>
        <td>Mask-RT-DETR-S</td>
        <td>41.0</td>
        <td>33.5007</td>
        <td>-</td>
        <td>51.8 M</td>
    </tr>
    <tr>
        <td>Mask-RT-DETR-X</td>
        <td>47.5</td>
        <td>75.755</td>
        <td>3358.04</td>
        <td>237.5 M</td>
    </tr>
    <tr>
        <td>MaskRCNN-ResNet50-FPN</td>
        <td>35.6</td>
        <td>-</td>
        <td>-</td>
        <td>157.5 M</td>
        <td rowspan="7">Mask R-CNN is a full-task deep learning model from Facebook AI Research (FAIR) that can perform object classification and localization in a single model, combined with image-level masks to complete segmentation tasks.</td>
    </tr>
    <tr>
        <td>MaskRCNN-ResNet50-vd-FPN</td>
        <td>36.4</td>
        <td>-</td>
        <td>-</td>
        <td>157.5 M</td>
    </tr>
    <tr>
        <td>MaskRCNN-ResNet50-vd-SSLDv2-FPN</td>
        <td>38.2</td>
        <td>-</td>
        <td>-</td>
        <td>127.2 M</td>
    </tr>
    <tr>
        <td>MaskRCNN-ResNet50</td>
        <td>32.8</td>
        <td>-</td>
        <td>-</td>
        <td>128.7 M</td>
    </tr>
    <tr>
        <td>MaskRCNN-ResNet101-FPN</td>
        <td>36.6</td>
        <td>-</td>
        <td>-</td>
        <td>225.4 M</td>
    </tr>
    <tr>
        <td>MaskRCNN-ResNet101-vd-FPN</td>
        <td>38.1</td>
        <td>-</td>
        <td>-</td>
        <td>225.4 M</td>
    </tr>
    <tr>
        <td>MaskRCNN-ResNeXt101-vd-FPN</td>
        <td>39.5</td>
        <td>-</td>
        <td>-</td>
        <td>370.0 M</td>
        <td></td>
    </tr>
    <tr>
        <td>PP-YOLOE_seg-S</td>
        <td>32.5</td>
        <td>-</td>
        <td>-</td>
        <td>31.5 M</td>
        <td>PP-YOLOE_seg is an instance segmentation model based on PP-YOLOE. This model inherits PP-YOLOE's backbone and head, significantly enhancing instance segmentation performance and inference speed through the design of a PP-YOLOE instance segmentation head.</td>
    </tr>
</table>


**Note: The above accuracy metrics are based on the Mask AP of the [COCO2017](https://cocodataset.org/#home) validation set. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

</details>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Tutorial](../../../installation/installation_en.md)

After installing the wheel package, a few lines of code can complete the inference of the instance segmentation module. You can switch models under this module freely, and you can also integrate the model inference of the instance segmentation module into your project.

```python
from paddlex import create_model
model = create_model("Mask-RT-DETR-L")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference APIs, please refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API_en.md).

## IV. Custom Development
If you are seeking higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better instance segmentation models. Before using PaddleX to develop instance segmentation models, please ensure that you have installed the relevant model training plugins for segmentation in PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Guide](../../../installation/installation_en.md).

### 4.1 Data Preparation
Before model training, it is necessary to prepare the corresponding dataset for each task module. PaddleX provides data verification functionality for each module, and **only data that passes the verification can be used for model training**. Additionally, PaddleX provides demo datasets for each module, allowing you to complete subsequent development based on the officially provided demo data. If you wish to use a private dataset for subsequent model training, you can refer to the [PaddleX Instance Segmentation Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/instance_segmentation_en.md).

#### 4.1.1 Download Demo Data

You can download the demo dataset to a specified folder using the following command:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/instance_seg_coco_examples.tar -P ./dataset
tar -xf ./dataset/instance_seg_coco_examples.tar -C ./dataset/
```

#### 4.1.2 Data Verification
Data verification can be completed with a single command:

```bash
python main.py -c paddlex/configs/instance_segmentation/Mask-RT-DETR-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/instance_seg_coco_examples
```

After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.
<details>
  <summary>👉 <b>Details of Validation Results (Click to Expand)</b></summary>

The specific content of the validation result file is:

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 2,
    "train_samples": 79,
    "train_sample_paths": [
      "check_dataset/demo_img/pexels-photo-634007.jpeg",
      "check_dataset/demo_img/pexels-photo-59576.png"
    ],
    "val_samples": 19,
    "val_sample_paths": [
      "check_dataset/demo_img/peasant-farmer-farmer-romania-botiza-47862.jpeg",
      "check_dataset/demo_img/pexels-photo-715546.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/instance_seg_coco_examples",
  "show_type": "image",
  "dataset_type": "COCOInstSegDataset"
}
```
In the above verification results, `check_pass` being `True` indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:

* `attributes.num_classes`: The number of classes in this dataset is 2;
* `attributes.train_samples`: The number of training samples in this dataset is 79;
* `attributes.val_samples`: The number of validation samples in this dataset is 19;
* `attributes.train_sample_paths`: A list of relative paths to the visualized training samples in this dataset;
* `attributes.val_sample_paths`: A list of relative paths to the visualized validation samples in this dataset;
Additionally, the dataset verification also analyzes the distribution of sample numbers across all categories in the dataset and generates a distribution histogram (`histogram.png`):

![](/tmp/images/modules/instanceseg/01.png)
</details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)
After completing data verification, you can convert the dataset format or re-split the training/validation ratio by **modifying the configuration file** or **appending hyperparameters**.

<details>
  <summary>👉 <b>Details of Format Conversion/Dataset Splitting (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

The instance segmentation task supports converting `LabelMe` format to `COCO` format. The parameters for dataset format conversion can be set by modifying the fields under `CheckDataset` in the configuration file. Below are some example explanations for some of the parameters in the configuration file:

* `CheckDataset`:
  * `convert`:
    * `enable`: Whether to perform dataset format conversion. Set to `True` to enable dataset format conversion, default is `False`;
    * `src_dataset_type`: If dataset format conversion is performed, the source dataset format needs to be set. The available source format is `LabelMe`;
For example, if you want to convert a `LabelMe` dataset to `COCO` format, you need to modify the configuration file as follows:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/instance_seg_labelme_examples.tar -P ./dataset
tar -xf ./dataset/instance_seg_labelme_examples.tar -C ./dataset/
```
```bash
......
CheckDataset:
  ......
  convert: 
    enable: True
    src_dataset_type: LabelMe
  ......
```
Then execute the command:

```bash
python main.py -c paddlex/configs/instance_segmentation/Mask-RT-DETR-L.yaml\
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/instance_seg_labelme_examples 
```
After the data conversion is executed, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters also support being set by appending command line arguments:

```bash
python main.py -c paddlex/configs/instance_segmentation/Mask-RT-DETR-L.yaml\
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/instance_seg_labelme_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
```


**(2) Dataset Splitting**

The parameters for dataset splitting can be set by modifying the fields under `CheckDataset` in the configuration file. Some example explanations for the parameters in the configuration file are as follows:

* `CheckDataset`:
  * `split`:
    * `enable`: Whether to re-split the dataset. When set to `True`, the dataset will be re-split. The default is `False`;
    * `train_percent`: If the dataset is to be re-split, the percentage of the training set needs to be set. The type is any integer between 0-100, and the sum with `val_percent` must be 100;

For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, you need to modify the configuration file as follows:

```bash
......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
```
Then execute the command:

```bash
python main.py -c paddlex/configs/instance_segmentation/Mask-RT-DETR-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/instance_seg_labelme_examples 
```
After data splitting, the original annotation files will be renamed as `xxx.bak` in the original path.

The above parameters can also be set by appending command line arguments:

```bash
python main.py -c paddlex/configs/instance_segmentation/Mask-RT-DETR-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/instance_seg_labelme_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### 4.2 Model Training
A single command can complete model training. Taking the training of the instance segmentation model Mask-RT-DETR-L as an example:

```bash
python main.py -c paddlex/configs/instance_segmentation/Mask-RT-DETR-L.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/instance_seg_coco_examples
```
The following steps are required:

* Specify the path to the `.yaml` configuration file of the model (here it is `Mask-RT-DETR-L.yaml`)
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`. 
Other related parameters can be set by modifying the fields under `Global` and `Train` in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify the first 2 GPUs for training: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration File Parameters Instructions](../../instructions/config_parameters_common_en.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

* During model training, PaddleX automatically saves the model weight files, with the default being `output`. If you need to specify a save path, you can set it through the `-o Global.output` field in the configuration file.
* PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.
* When training other models, you need to specify the corresponding configuration file. The correspondence between models and configuration files can be found in [PaddleX Model List (CPU/GPU)](../../../support_list/models_list_en.md). After completing the model training, all outputs are saved in the specified output directory (default is `./output/`), typically including:

* `train_result.json`: Training result record file, recording whether the training task was completed normally, as well as the output weight metrics, related file paths, etc.;
* `train.log`: Training log file, recording changes in model metrics and loss during training;
* `config.yaml`: Training configuration file, recording the hyperparameter configuration for this training session;
* `.pdparams`, `.pdema`, `.pdopt.pdstate`, `.pdiparams`, `.pdmodel`: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;
</details>

## **4.3 Model Evaluation**
After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:


```bash
python main.py -c paddlex/configs/instance_segmentation/Mask-RT-DETR-L.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/instance_seg_coco_examples
```
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `Mask-RT-DETR-L.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`. Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common_en.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as `-o Evaluate.weight_path=./output/best_model/best_model.pdparams`.

After completing the model evaluation, an `evaluate_result.json` file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully and the model's evaluation metrics, including AP.

</details>


### **4.4 Model Inference and Model Integration**
After completing model training and evaluation, you can use the trained model weights for inference prediction or Python integration.

#### 4.4.1 Model Inference
To perform inference prediction via the command line, simply use the following command:

```bash
python main.py -c paddlex/configs/instance_segmentation/Mask-RT-DETR-L.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it's `Mask-RT-DETR-L.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`. Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX Pipeline or into your own project.

1.**Pipeline Integration**

The instance segmentation module can be integrated into the [General Instance Segmentation Pipeline](../../../pipeline_usage/tutorials/cv_pipelines/instance_segmentation_en.md) of PaddleX. Simply replace the model path to update the instance segmentation module of the relevant pipeline.

2.**Module Integration**
The weights you produce can be directly integrated into the instance segmentation module. Refer to the Python example code in [Quick Integration](#三快速集成), and simply replace the model with the path to your trained model.