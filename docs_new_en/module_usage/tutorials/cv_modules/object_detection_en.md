# Tutorial on Developing Object Detection Modules

## I. Overview
The object detection module is a crucial component in computer vision systems, responsible for locating and marking regions containing specific objects in images or videos. The performance of this module directly impacts the accuracy and efficiency of the entire computer vision system. The object detection module typically outputs bounding boxes for the target regions, which are then passed as input to the object recognition module for further processing.

## II. List of Supported Models
<details>
   <summary> 👉Details of Model List</summary>

<table>
  <tr>
    <th>Model</th>
    <th>mAP(%)</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time</th>
    <th>Model Size (M)</th>
    <th>Description</th>
  </tr>
 <tr>
    <td>Cascade-FasterRCNN-ResNet50-FPN</td>
    <td>41.1</td>
    <td>-</td>
    <td>-</td>
    <td>245.4 M</td>
    <td rowspan="2">Cascade-FasterRCNN is an improved version of the Faster R-CNN object detection model. By coupling multiple detectors and optimizing detection results using different IoU thresholds, it addresses the mismatch problem between training and prediction stages, enhancing the accuracy of object detection.</td>
  </tr>
  <tr>
    <td>Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
    <td>45.0</td>
    <td>-</td>
    <td>-</td>
    <td>246.2 M</td>
    <td></td>
  </tr>
  <tr>
    <td>CenterNet-DLA-34</td>
    <td>37.6</td>
    <td>-</td>
    <td>-</td>
    <td>75.4 M</td>
    <td rowspan="2">CenterNet is an anchor-free object detection model that treats the keypoints of the object to be detected as a single point—the center point of its bounding box, and performs regression through these keypoints.</td>
  </tr>
  <tr>
    <td>CenterNet-ResNet50</td>
    <td>38.9</td>
    <td>-</td>
    <td>-</td>
    <td>319.7 M</td>
    <td></td>
  </tr>
  <tr>
    <td>DETR-R50</td>
    <td>42.3</td>
    <td>59.2132</td>
    <td>5334.52</td>
    <td>159.3 M</td>
    <td >DETR is a transformer-based object detection model proposed by Facebook. It achieves end-to-end object detection without the need for predefined anchor boxes or NMS post-processing strategies.</td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet34-FPN</td>
    <td>37.8</td>
    <td>-</td>
    <td>-</td>
    <td>137.5 M</td>
    <td rowspan="9">Faster R-CNN is a typical two-stage object detection model that first generates region proposals and then performs classification and regression on these proposals. Compared to its predecessors R-CNN and Fast R-CNN, Faster R-CNN's main improvement lies in the region proposal aspect, using a Region Proposal Network (RPN) to provide region proposals instead of traditional selective search. RPN is a Convolutional Neural Network (CNN) that shares convolutional features with the detection network, reducing the computational overhead of region proposals.</td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet50-FPN</td>
    <td>38.4</td>
    <td>-</td>
    <td>-</td>
    <td>148.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet50-vd-FPN</td>
    <td>39.5</td>
    <td>-</td>
    <td>-</td>
    <td>148.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
    <td>41.4</td>
    <td>-</td>
    <td>-</td>
    <td>148.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet50</td>
    <td>36.7</td>
    <td>-</td>
    <td>-</td>
    <td>120.2 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet101-FPN</td>
    <td>41.4</td>
    <td>-</td>
    <td>-</td>
    <td>216.3 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet101</td>
    <td>39.0</td>
    <td>-</td>
    <td>-</td>
    <td>188.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNeXt101-vd-FPN</td>
    <td>43.4</td>
    <td>-</td>
    <td>-</td>
    <td>360.6 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-Swin-Tiny-FPN</td>
    <td>42.6</td>
    <td>-</td>
    <td>-</td>
    <td>159.8 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FCOS-ResNet50</td>
    <td>39.6</td>
    <td>103.367</td>
    <td>3424.91</td>
    <td>124.2 M</td>
    <td>FCOS is an anchor-free object detection model that performs dense predictions. It uses the backbone of RetinaNet and directly regresses the width and height of the target object on the feature map, predicting the object's category and centerness (the degree of offset of pixels on the feature map from the object's center), which is eventually used as a weight to adjust the object score.</td>
  </tr>
  <tr>
    <td>PicoDet-L</td>
    <td>42.6</td>
    <td>16.6715</td>
    <td>169.904</td>
    <td>20.9 M</td>
    <td rowspan="4">PP-PicoDet is a lightweight object detection algorithm designed for full-size and wide-aspect-ratio targets, with a focus on mobile device computation. Compared to traditional object detection algorithms, PP-PicoDet boasts smaller model sizes and lower computational complexity, achieving higher speeds and lower latency while maintaining detection accuracy.</td>
  </tr>
  <tr>
    <td>PicoDet-M</td>
    <td>37.5</td>
    <td>16.2311</td>
    <td>71.7257</td>
    <td>16.8 M</td>
    <td></td>
  </tr>
  <tr>
    <td>PicoDet-S</td>
    <td>29.1</td>
    <td>14.097</td>
    <td>37.6563</td>
    <td>4.4 M</td>
    <td></td>
  </tr>
  <tr>
    <td>PicoDet-XS</td>
    <td>26.2</td>
    <td>13.8102</td>
    <td>48.3139</td>
    <td>5.7 M</td>
    <td></td>
  </tr>
    <tr>
    <td>PP-YOLOE_plus-L</td>
    <td>52.9</td>
    <td>33.5644</td>
    <td>814.825</td>
    <td>185.3 M</td>
    <td rowspan="4">PP-YOLOE_plus is an iteratively optimized and upgraded version of PP-YOLOE, a high-precision cloud-edge integrated model developed by Baidu PaddlePaddle's Vision Team. By leveraging the large-scale Objects365 dataset and optimizing preprocessing, it significantly enhances the end-to-end inference speed of the model.</td>
  </tr>
  <tr>
    <td>PP-YOLOE_plus-M</td>
    <td>49.8</td>
    <td>19.843</td>
    <td>449.261</td>
    <td>82.3 M</td>
    <td></td>
  </tr>
  <tr>
    <td>PP-YOLOE_plus-S</td>
    <td>43.7</td>
    <td>16.8884</td>
    <td>223.059</td>
    <td>28.3 M</td>
    <td></td>
  </tr>
  <tr>
    <td>PP-YOLOE_plus-X</td>
    <td>54.7</td>
    <td>57.8995</td>
    <td>1439.93</td>
    <td>349.4 M</td>
    <td></td>
  </tr>
  <tr>
    <td>RT-DETR-H</td>
    <td>56.3</td>
    <td>114.814</td>
    <td>3933.39</td>
    <td>435.8 M</td>
    <td rowspan="5">RT-DETR is the first real-time end-to-end object detector. It features an efficient hybrid encoder that balances model performance and throughput, efficiently processes multi-scale features, and introduces an accelerated and optimized query selection mechanism to dynamize decoder queries. RT-DETR supports flexible end-to-end inference speeds through the use of different decoders.</td>
  </tr>
  <tr>
    <td>RT-DETR-L</td>
    <td>53.0</td>
    <td>34.5252</td>
    <td>1454.27</td>
    <td>113.7 M</td>
    <td></td>
  </tr>
  <tr>
    <td>RT-DETR-R18</td>
    <td>46.5</td>
    <td>19.89</td>
    <td>784.824</td>
    <td>70.7 M</td>
    <td></td>
  </tr>
  <tr>
    <td>RT-DETR-R50</td>
    <td>53.1</td>
    <td>41.9327</td>
    <td>1625.95</td>
    <td>149.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>RT-DETR-X</td>
    <td>54.8</td>
    <td>61.8042</td>
    <td>2246.64</td>
    <td>232.9 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOv3-DarkNet53</td>
    <td>39.1</td>
    <td>40.1055</td>
    <td>883.041</td>
    <td>219.7 M</td>
    <td rowspan="3">YOLOv3 is a real-time end-to-end object detector that utilizes a unique single Convolutional Neural Network (CNN) to frame the object detection problem as a regression task, enabling real-time detection. The model employs multi-scale detection to enhance performance across different object sizes.</td>
  </tr>
  <tr>
    <td>YOLOv3-MobileNetV3</td>
    <td>31.4</td>
    <td>18.6692</td>
    <td>267.214</td>
    <td>83.8 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOv3-ResNet50_vd_DCN</td>
    <td>40.6</td>
    <td>31.6276</td>
    <td>856.047</td>
    <td>163.0 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-L</td>
    <td>50.1</td>
    <td>185.691</td>
    <td>1250.58</td>
    <td>192.5 M</td>
    <td rowspan="6">Building upon YOLOv3's framework, YOLOX significantly boosts detection performance in complex scenarios by incorporating Decoupled Head, Data Augmentation, Anchor Free, and SimOTA components.</td>
  </tr>
  <tr>
    <td>YOLOX-M</td>
    <td>46.9</td>
    <td>123.324</td>
    <td>688.071</td>
    <td>90.0 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-N</td>
    <td>26.1</td>
    <td>79.1665</td>
    <td>155.59</td>
    <td>3.4 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-S</td>
    <td>40.4</td>
    <td>184.828</td>
    <td>474.446</td>
    <td>32.0 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-T</td>
    <td>32.9</td>
    <td>102.748</td>
    <td>212.52</td>
    <td>18.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-X</td>
    <td>51.8</td>
    <td>227.361</td>
    <td>2067.84</td>
    <td>351.5 M</td>
    <td></td>
  </tr>
</table>

**Note: The precision metrics mentioned are based on the [COCO2017](https://cocodataset.org/#home) validation set mAP(0.5:0.95). All model GPU inference times are measured on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

</details>

## III. Quick Integration

> ❗ Before proceeding with quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.md).

After installing the wheel package, you can perform object detection inference with just a few lines of code. You can easily switch between models within the module and integrate the object detection inference into your projects.

```python
from paddlex import create_model
model = create_model("PicoDet-S")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

For more information on using PaddleX's single-model inference APIs, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.md).

## IV. Custom Development

If you seek higher precision from existing models, you can leverage PaddleX's custom development capabilities to develop better object detection models. Before developing object detection models with PaddleX, ensure you have installed the object detection related training plugins. For installation instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.md).

### 4.1 Data Preparation

Before model training, prepare the corresponding dataset for the task module. PaddleX provides a data validation feature for each module, and **only datasets that pass validation can be used for model training**. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use a private dataset for model training, refer to the [PaddleX Object Detection Task Module Data Annotation Guide](../../../data_annotations/cv_modules/object_detection.md).

#### 4.1.1 Download Demo Data

You can download the demo dataset to a specified folder using the following command:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_coco_examples.tar -P ./dataset
tar -xf ./dataset/det_coco_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation

Validate your dataset with a single command:

```bash
python main.py -c paddlex/configs/object_detection/PicoDet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples
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
    "num_classes": 4,
    "train_samples": 701,
    "train_sample_paths": [
      "check_dataset/demo_img/road839.png",
      "check_dataset/demo_img/road363.png",
      "check_dataset/demo_img/road148.png",
      "check_dataset/demo_img/road237.png",
      "check_dataset/demo_img/road733.png",
      "check_dataset/demo_img/road861.png",
      "check_dataset/demo_img/road762.png",
      "check_dataset/demo_img/road515.png",
      "check_dataset/demo_img/road754.png",
      "check_dataset/demo_img/road173.png"
    ],
    "val_samples": 176,
    "val_sample_paths": [
      "check_dataset/demo_img/road218.png",
      "check_dataset/demo_img/road681.png",
      "check_dataset/demo_img/road138.png",
      "check_dataset/demo_img/road544.png",
      "check_dataset/demo_img/road596.png",
      "check_dataset/demo_img/road857.png",
      "check_dataset/demo_img/road203.png",
      "check_dataset/demo_img/road589.png",
      "check_dataset/demo_img/road655.png",
      "check_dataset/demo_img/road245.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/det_coco_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}
```
In the above validation results, `check_pass` being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:

* `attributes.num_classes`: The number of classes in this dataset is 4;
* `attributes.train_samples`: The number of training samples in this dataset is 704;
* `attributes.val_samples`: The number of validation samples in this dataset is 176;
* `attributes.train_sample_paths`: A list of relative paths to the visualization images of training samples in this dataset;
* `attributes.val_sample_paths`: A list of relative paths to the visualization images of validation samples in this dataset;

Additionally, the dataset verification also analyzes the distribution of sample numbers across all classes in the dataset and generates a histogram (histogram.png) for visualization:

![](/tmp/images/modules/obj_det/01.png)
</details>


### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format and re-split the training/validation ratio by **modifying the configuration file** or **appending hyperparameters**.

<details>
  <summary>👉 <b>Details of Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

Object detection supports converting datasets in `VOC` and `LabelMe` formats to `COCO` format.

Parameters related to dataset validation can be set by modifying the fields under `CheckDataset` in the configuration file. Examples of some parameters in the configuration file are as follows:

* `CheckDataset`:
  * `convert`:
    * `enable`: Whether to perform dataset format conversion. Object detection supports converting `VOC` and `LabelMe` format datasets to `COCO` format. Default is `False`;
    * `src_dataset_type`: If dataset format conversion is performed, the source dataset format needs to be set. Default is `null`, with optional values `VOC`, `LabelMe`, `VOCWithUnlabeled`, `LabelMeWithUnlabeled`;
For example, if you want to convert a `LabelMe` format dataset to `COCO` format, taking the following `LabelMe` format dataset as an example, you need to modify the configuration as follows:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_labelme_examples.tar -P ./dataset
tar -xf ./dataset/det_labelme_examples.tar -C ./dataset/
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
python main.py -c paddlex/configs/object_detection/PicoDet-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_labelme_examples
```
Of course, the above parameters also support being set by appending command line arguments. Taking a `LabelMe` format dataset as an example:

```bash
python main.py -c paddlex/configs/object_detection/PicoDet-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_labelme_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
```

**(2) Dataset Splitting**

Parameters for dataset splitting can be set by modifying the fields under `CheckDataset` in the configuration file. Examples of some parameters in the configuration file are as follows:

* `CheckDataset`:
  * `split`:
    * `enable`: Whether to re-split the dataset. When `True`, dataset splitting is performed. Default is `False`;
    * `train_percent`: If the dataset is re-split, the percentage of the training set needs to be set. The type is any integer between 0-100, and it needs to ensure that the sum with `val_percent` is 100;
    * `val_percent`: If the dataset is re-split, the percentage of the validation set needs to be set. The type is any integer between 0-100, and it needs to ensure that the sum with `train_percent` is 100;
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
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples 
```
After dataset splitting is executed, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters also support being set by appending command line arguments:

```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>


### 4.2 Model Training
Model training can be completed with a single command, taking the training of the object detection model PicoDet-S as an example:

```bash
python main.py -c paddlex/configs/object_detection/PicoDet-S.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/det_coco_examples
```
The following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PicoDet-S.yaml`)
* Set the mode to model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`. 
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file instructions for the corresponding task module of the model [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

* During model training, PaddleX automatically saves the model weight files, with the default being `output`. If you need to specify a save path, you can set it through the `-o Global.output` field in the configuration file.
* PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.
* When training other models, you need to specify the corresponding configuration file. The correspondence between models and configuration files can be found in [PaddleX Model List (CPU/GPU)](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/GvMbk70MZz/0PKFjfhs0UN4Qs?t=mention&mt=doc&dt=doc). After completing the model training, all outputs are saved in the specified output directory (default is `./output/`), typically including:

* `train_result.json`: Training result record file, recording whether the training task was completed normally, as well as the output weight metrics, related file paths, etc.;
* `train.log`: Training log file, recording changes in model metrics and loss during training;
* `config.yaml`: Training configuration file, recording the hyperparameter configuration for this training session;
* `.pdparams`, `.pdema`, `.pdopt.pdstate`, `.pdiparams`, `.pdmodel`: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;
</details>

## **4.3 Model Evaluation**
After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/object_detection/PicoDet-S.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/det_coco_examples
```
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PicoDet-S.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`. Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as `-o Evaluate.weight_path=./output/best_model/best_model.pdparams`.

After completing the model evaluation, an `evaluate_result.json` file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully and the model's evaluation metrics, including AP.

</details>

### **4.4 Model Inference and Integration**
After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference

* To perform inference predictions through the command line, use the following command:
```bash
python main.py -c paddlex/configs/object_detection/PicoDet-S.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PicoDet-S.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipelines or directly into your own project.

1.**Pipeline Integration**

The object detection module can be integrated into the [General Object Detection Pipeline](../../../pipeline_usage/tutorials/cv_pipelines/object_detection_en.md) of PaddleX. Simply replace the model path to update the object detection module of the relevant pipeline. In pipeline integration, you can use high-performance deployment and service-oriented deployment to deploy your model.

2.**Module Integration**

The weights you produce can be directly integrated into the object detection module. Refer to the Python example code in [Quick Integration](#三快速集成), and simply replace the model with the path to your trained model.


