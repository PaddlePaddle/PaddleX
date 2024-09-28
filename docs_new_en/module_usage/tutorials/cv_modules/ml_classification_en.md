# Tutorial on Developing Image Multi-Label Classification Modules

## I. Overview
The image multi-label classification module is a crucial component in computer vision systems, responsible for assigning multiple labels to input images. Unlike traditional image classification tasks that assign a single category to an image, multi-label classification tasks require assigning multiple relevant categories to an image. The performance of this module directly impacts the accuracy and efficiency of the entire computer vision system. The image multi-label classification module typically takes an image as input and, through deep learning or other machine learning algorithms, classifies it into multiple predefined categories based on its characteristics and content. For example, an image containing both a cat and a dog might be labeled as both "cat" and "dog" by the image multi-label classification module. These classification labels are then output for subsequent processing and analysis by other modules or systems.

## II. Supported Model List

<details>
   <summary> 👉Model List Details</summary>

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
    <td>CLIP_vit_base_patch16_448_ML</td>
    <td>89.15</td>
    <td>-</td>
    <td>-</td>
    <td>325.6</td>
    <td>CLIP_ML is an image multi-label classification model based on CLIP, which significantly improves accuracy on multi-label classification tasks by incorporating an ML-Decoder.</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B0_ML</td>
    <td>80.98</td>
    <td>-</td>
    <td>-</td>
    <td>39.6</td>
    <td rowspan="3">PP-HGNetV2_ML is an image multi-label classification model based on PP-HGNetV2, which significantly improves accuracy on multi-label classification tasks by incorporating an ML-Decoder.</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B4_ML</td>
    <td>87.96</td>
    <td>-</td>
    <td>-</td>
    <td>88.5</td>
  </tr>
  <tr>
    <td>PP-HGNetV2-B6_ML</td>
    <td>91.25</td>
    <td>-</td>
    <td>-</td>
    <td>286.0</td>
  </tr>
  <tr>
    <td>PP-LCNet_x1_0_ML</td>
    <td>77.96</td>
    <td>-</td>
    <td>-</td>
    <td>29.4</td>
    <td>PP-LCNet_ML is an image multi-label classification model based on PP-LCNet, which significantly improves accuracy on multi-label classification tasks by incorporating an ML-Decoder.</td>
  </tr>
  <tr>
    <td>ResNet50_ML</td>
    <td>83.50</td>
    <td>-</td>
    <td>-</td>
    <td>108.9</td>
    <td>ResNet50_ML is an image multi-label classification model based on ResNet50, which significantly improves accuracy on multi-label classification tasks by incorporating an ML-Decoder.</td>
  </tr>
</table>

**Note: The above accuracy metrics are mAP for the multi-label classification task on [COCO2017](https://cocodataset.org/#home). GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**
</details>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide]((../../../installation/installation_en.md))

After installing the wheel package, you can complete multi-label classification module inference with just a few lines of code. You can switch between models in this module freely, and you can also integrate the model inference of the multi-label classification module into your project.
```bash
from paddlex import create_model
model = create_model("PP-LCNet_x1_0_ML")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/multilabel_classification_005.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference APIs, please refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.md).
## IV. Custom Development
If you are seeking higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better multi-label classification models. Before using PaddleX to develop multi-label classification models, please ensure that you have installed the relevant model training plugins for image classification in PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Guide](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/y0mmii50BW/dF1VvOPZmZXXzn?t=mention&mt=doc&dt=doc).

### 4.1 Data Preparation
Before model training, you need to prepare the dataset for the corresponding task module. PaddleX provides data validation functionality for each module, and **only data that passes data validation can be used for model training**. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use your own private dataset for subsequent model training, please refer to the [PaddleX Image Multi-Label Classification Task Module Data Annotation Guide](../../../data_annotations/cv_modules/ml_classification_en.md).

#### 4.1.1 Demo Data Download
You can use the following command to download the demo dataset to a specified folder:
```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/mlcls_nus_examples.tar -P ./dataset
tar -xf ./dataset/mlcls_nus_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/mlcls_nus_examples
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
    "label_file": "../../dataset/mlcls_nus_examples/label.txt",
    "num_classes": 33,
    "train_samples": 17463,
    "train_sample_paths": [
      "check_dataset/demo_img/0543_4338693.jpg",
      "check_dataset/demo_img/0272_347806939.jpg",
      "check_dataset/demo_img/0069_2291994812.jpg",
      "check_dataset/demo_img/0012_1222850604.jpg",
      "check_dataset/demo_img/0238_53773041.jpg",
      "check_dataset/demo_img/0373_541261977.jpg",
      "check_dataset/demo_img/0567_519506868.jpg",
      "check_dataset/demo_img/0023_289621557.jpg",
      "check_dataset/demo_img/0581_484524659.jpg",
      "check_dataset/demo_img/0325_120753036.jpg"
    ],
    "val_samples": 17463,
    "val_sample_paths": [
      "check_dataset/demo_img/0546_130758157.jpg",
      "check_dataset/demo_img/0284_2230710138.jpg",
      "check_dataset/demo_img/0090_1491261559.jpg",
      "check_dataset/demo_img/0013_392798436.jpg",
      "check_dataset/demo_img/0246_2248376356.jpg",
      "check_dataset/demo_img/0377_1349296474.jpg",
      "check_dataset/demo_img/0570_2457645006.jpg",
      "check_dataset/demo_img/0027_309333946.jpg",
      "check_dataset/demo_img/0584_132639537.jpg",
      "check_dataset/demo_img/0329_206031527.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/mlcls_nus_examples",
  "show_type": "image",
  "dataset_type": "MLClsDataset"
}
```
In the above validation results, `check_pass` being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:

* `attributes.num_classes`: The number of classes in this dataset is 33;
* `attributes.train_samples`: The number of training set samples in this dataset is 17463;
* `attributes.val_samples`: The number of validation set samples in this dataset is 17463;
* `attributes.train_sample_paths`: A list of relative paths to the visual samples in the training set of this dataset;
* `attributes.val_sample_paths`: A list of relative paths to the visual samples in the validation set of this dataset;

Additionally, the dataset validation analyzes the sample number distribution across all classes in the dataset and generates a distribution histogram (histogram.png):
![](/tmp/images/modules/ml_classification/01.png)
</details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)

After completing data validation, you can convert the dataset format or re-split the training/validation ratio of the dataset by **modifying the configuration file** or **appending hyperparameters**.

<details>
  <summary>👉 <b>Dataset Format Conversion/Dataset Splitting Details (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

The multi-label image classification supports the conversion of `COCO` format datasets to `MLClsDataset` format. The parameters for dataset format conversion can be set by modifying the fields under `CheckDataset` in the configuration file. Examples of some parameters in the configuration file are as follows:

* `CheckDataset`:
  * `convert`:
    * `enable`: Whether to perform dataset format conversion. Multi-label image classification supports converting `COCO` format datasets to `MLClsDataset` format. Default is `False`;
    * `src_dataset_type`: If dataset format conversion is performed, the source dataset format needs to be set. Default is `null`, with the optional value of `COCO`; 

For example, if you want to convert a `COCO` format dataset to `MLClsDataset` format, you need to modify the configuration file as follows:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_coco_examples.tar -P ./dataset
tar -xf ./dataset/det_coco_examples.tar -C ./dataset/
```
```bash
......
CheckDataset:
  ......
  convert: 
    enable: True
    src_dataset_type: COCO
  ......
```
Then execute the command:

```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples 
```
After the data conversion is executed, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters also support being set by appending command line arguments:

```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=COCO
```

#### (2) Dataset Splitting

The dataset splitting parameters can be set by modifying the fields under `CheckDataset` in the configuration file. An example of part of the configuration file is shown below:

* `CheckDataset`:
  * `split`:
    * `enable`: Whether to re-split the dataset. Set to `True` to perform dataset splitting, default is `False`;
    * `train_percent`: If re-splitting the dataset, set the percentage of the training set, an integer between 0-100, ensuring the sum with `val_percent` is 100;
    * `val_percent`: If re-splitting the dataset, set the percentage of the validation set, an integer between 0-100, ensuring the sum with `train_percent` is 100;

For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, modify the configuration file as follows:

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

After the data splitting is executed, the original annotation files will be renamed to `xxx.bak` in the original path.

These parameters can also be set by appending command-line arguments:

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
A single command can complete the model training. Taking the training of the image multi-label classification model PP-LCNet_x1_0_ML as an example:
```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/mlcls_nus_examples
```
the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0_ML.yaml`)
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path of the training dataset: `-o Global.dataset_dir`. Other related parameters can be set by modifying the fields under `Global` and `Train` in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first 2 GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file parameter instructions for the corresponding task module of the model [PaddleX Common Model Configuration File Parameters](../../instructions/config_parameters_common.md).


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
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/mlcls_nus_examples
```
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it's `PP-LCNet_x1_0_ML.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path. If you need to change it, simply append the command line parameter to set it, such as `-o Evaluate.weight_path=./output/best_model/best_model.pdparams`.

After completing the model evaluation, an `evaluate_result.json` file will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully and the model's evaluation metrics, including MultiLabelMAP;


</details>

### **4.4 Model Inference and Model Integration**
After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
* Inference predictions can be performed through the command line with just one command:

```bash
python main.py -c paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/multilabel_classification_005.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it's `PP-LCNet_x1_0_ML.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common_en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or directly into your own project.

1.**Pipeline Integration**

The image multi-label classification module can be integrated into the [General Image Multi-label Classification Pipeline](../../../pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification_en.md) of PaddleX. Simply replace the model path to update the image multi-label classification module of the relevant pipeline. In pipeline integration, you can use high-performance deployment and service-oriented deployment to deploy your model.

2.**Module Integration**

The weights you produce can be directly integrated into the image multi-label classification module. Refer to the Python example code in [Quick Integration](#三快速集成) and simply replace the model with the path to your trained model.
