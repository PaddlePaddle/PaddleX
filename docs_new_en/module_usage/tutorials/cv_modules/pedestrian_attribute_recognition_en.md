# Pedestrian Attribute Recognition Module Development Tutorial

## I. Overview
Pedestrian attribute recognition is a crucial component in computer vision systems, responsible for locating and labeling specific attributes of pedestrians in images or videos, such as gender, age, clothing color, and type. The performance of this module directly impacts the accuracy and efficiency of the entire computer vision system. The pedestrian attribute recognition module typically outputs attribute information for each pedestrian, which is then passed as input to other modules (e.g., pedestrian tracking, pedestrian re-identification) for subsequent processing.

## II. Supported Model List



<details>
   <summary> 👉 Model List Details</summary>

| Model | mA (%) | GPU Inference Time (ms) | CPU Inference Time | Model Size (M) | Description |
|-|-|-|-|-|-|
| PP-LCNet_x1_0_pedestrian_attribute | 92.2 |3.84845 | 9.23735| 6.7M | PP-LCNet_x1_0_pedestrian_attribute is a lightweight pedestrian attribute recognition model based on PP-LCNet, covering 26 categories |

**Note: The above accuracy metrics are mA on PaddleX's internal self-built dataset. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**
</details>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.md)

After installing the wheel package, a few lines of code can complete the inference of the pedestrian attribute recognition module. You can easily switch models under this module and integrate the model inference of pedestrian attribute recognition into your project.

```bash
from paddlex import create_model
model = create_model("PP-LCNet_x1_0_pedestrian_attribute")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_006.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better pedestrian attribute recognition models. Before developing pedestrian attribute recognition with PaddleX, ensure you have installed the classification-related model training plugins for PaddleX.  The installation process can be found in the custom development section of the [PaddleX Local Installation Guide](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/y0mmii50BW/dF1VvOPZmZXXzn?t=mention&mt=doc&dt=doc).


### 4.1 Data Preparation
Before model training, you need to prepare a dataset for the specific task module. PaddleX provides data validation functionality for each module, and **only data that passes validation can be used for model training**. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use a private dataset for model training, refer to the [PaddleX Multi-Label Classification Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/ml_classification.md).

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/pedestrian_attribute_examples.tar -P ./dataset
tar -xf ./dataset/pedestrian_attribute_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
Run a single command to complete data validation:

```bash
python main.py -c paddlex/configs/pedestrian_attribute/PP-LCNet_x1_0_pedestrian_attribute.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/pedestrian_attribute_examples
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
    "label_file": "../../dataset/pedestrian_attribute_examples/label.txt",
    "num_classes": 26,
    "train_samples": 1000,
    "train_sample_paths": [
      "check_dataset/demo_img/020907.jpg",
      "check_dataset/demo_img/004274.jpg",
      "check_dataset/demo_img/009412.jpg",
      "check_dataset/demo_img/026873.jpg",
      "check_dataset/demo_img/030560.jpg",
      "check_dataset/demo_img/022846.jpg",
      "check_dataset/demo_img/009055.jpg",
      "check_dataset/demo_img/015399.jpg",
      "check_dataset/demo_img/006435.jpg",
      "check_dataset/demo_img/055307.jpg"
    ],
    "val_samples": 500,
    "val_sample_paths": [
      "check_dataset/demo_img/080381.jpg",
      "check_dataset/demo_img/080469.jpg",
      "check_dataset/demo_img/080146.jpg",
      "check_dataset/demo_img/080003.jpg",
      "check_dataset/demo_img/080283.jpg",
      "check_dataset/demo_img/080104.jpg",
      "check_dataset/demo_img/080149.jpg",
      "check_dataset/demo_img/080313.jpg",
      "check_dataset/demo_img/080131.jpg",
      "check_dataset/demo_img/080412.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/pedestrian_attribute_examples",
  "show_type": "image",
  "dataset_type": "MLClsDataset"
}
```

In the above validation results, `check_pass` being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:

* `attributes.num_classes`: The number of classes in this dataset is 26;
* `attributes.train_samples`: The number of samples in the training set of this dataset is 1000;
* `attributes.val_samples`: The number of samples in the validation set of this dataset is 500;
* `attributes.train_sample_paths`: The list of relative paths to the visualization images of samples in the training set of this dataset;
* `attributes.val_sample_paths`: The list of relative paths to the visualization images of samples in the validation set of this dataset;

Additionally, the dataset verification also analyzes the distribution of the length and width of all images in the dataset and plots a histogram (histogram.png):

![](/tmp/images/modules/ped_attri/image.png)

</details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format or re-split the training/validation ratio of the dataset by **modifying the configuration file** or **appending hyperparameters**.

<details>
  <summary>👉 <b>Dataset Format Conversion/Dataset Splitting Details (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

Pedestrian attribute recognition does not support data format conversion.

**(2) Dataset Splitting**

The dataset splitting parameters can be set by modifying the fields under `CheckDataset` in the configuration file. An example of part of the configuration file is shown below:

* `CheckDataset`:
  * `split`:
    * `enable`: Whether to re-split the dataset. Set to `True` to enable dataset splitting, default is `False`;
    * `train_percent`: If re-splitting the dataset, set the percentage of the training set. The type is any integer between 0-100, ensuring the sum with `val_percent` is 100;

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
python main.py -c paddlex/configs/pedestrian_attribute/PP-LCNet_x1_0_pedestrian_attribute.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/pedestrian_attribute_examples
```
After the data splitting is executed, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters also support being set by appending command-line arguments:

```bash
python main.py -c paddlex/configs/pedestrian_attribute/PP-LCNet_x1_0_pedestrian_attribute.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/pedestrian_attribute_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>


### 4.2 Model Training
Model training can be completed with a single command. Taking the training of the PP-LCNet pedestrian attribute recognition model (PP-LCNet_x1_0_pedestrian_attribute) as an example:

```bash
python main.py -c paddlex/configs/pedestrian_attribute/PP-LCNet_x1_0_pedestrian_attribute.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/pedestrian_attribute_examples
```
the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0_pedestrian_attribute.yaml`)
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

### **4.3 Model Evaluation**
After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/pedestrian_attribute/PP-LCNet_x1_0_pedestrian_attribute.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/pedestrian_attribute_examples
```
Similar to model training, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0_pedestrian_attribute.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as `-o Evaluate.weight_path=./output/best_model/best_model.pdparams`.

After completing the model evaluation, an `evaluate_result.json` file will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully and the model's evaluation metrics, including MultiLabelMAP;

</details>

### **4.4 Model Inference and Integration**
After completing model training and evaluation, you can use the trained model weights for inference prediction or Python integration.

#### 4.4.1 Model Inference
To perform inference prediction through the command line, simply use the following command:

```bash
python main.py -c paddlex/configs/pedestrian_attribute/PP-LCNet_x1_0_pedestrian_attribute.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_006.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0_pedestrian_attribute.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
. Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or directly into your own project.

1.**Pipeline Integration**

The pedestrian attribute recognition module can be integrated into the [General Image Multi-label Classification Pipeline](../../../pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification_en.md) of PaddleX. Simply replace the model path to update the pedestrian attribute recognition module of the relevant pipeline. In pipeline integration, you can use high-performance deployment and service-oriented deployment to deploy your model.

2.**Module Integration**

The weights you produce can be directly integrated into the pedestrian attribute recognition module. Refer to the Python example code in [Quick Integration](#三快速集成) and simply replace the model with the path to your trained model.
