# Tutorial for Developing Mainbody detection Modules

## I. Overview
Mainbody detection is a fundamental task in object detection, aiming to identify and extract the location and size of specific target objects, people, or entities from images and videos. By constructing deep neural network models, mainbody detection learns the feature representations of image subjects to achieve efficient and accurate detection.

## II. Supported Model List

<details>
   <summary> 👉 Details of Model List</summary>

<table>
  <tr>
    <th>Model</th>
    <th>mAP(0.5:0.95)</th>
    <th>mAP(0.5)</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time</th>
    <th>Model Size (M)</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>PP-ShiTuV2_det</td>
    <td>41.5</td>
    <td>62.0</td>
    <td>33.7</td>
    <td>537.0</td>
    <td>27.54</td>
    <td>A mainbody detection model based on PicoDet_LCNet_x2_5, which may detect multiple common subjects simultaneously.</td>
  </tr>
</table>

**Note: The evaluation set for the above accuracy metrics is  PaddleClas mainbody detection dataset mAP(0.5:0.95). GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**
</details>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to [PaddleX Local Installation Guide](../../../installation/installation_en.md)

After installing the wheel package, you can perform mainbody detection inference with just a few lines of code. You can easily switch between models under this module, and integrate the mainbody detection model inference into your project.

```python
from paddlex.inference import create_model 

model_name = "PP-ShiTuV2_det"

model = create_model(model_name)
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png", batch_size=1)

for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

For more information on using PaddleX's single-model inference APIs, refer to [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better mainbody detection models. Before developing mainbody detection models with PaddleX, ensure you have installed the PaddleDetection plugin for PaddleX. The installation process can be found in [PaddleX Local Installation Guide](../../../installation/installation_en.md).

### 4.1 Data Preparation
Before model training, you need to prepare a dataset for the specific task module. PaddleX provides a data validation function for each module, and **only data that passes validation can be used for model training**. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use a private dataset for model training, refer to [PaddleX Object Detection Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/object_detection_en.md).

#### 4.1.1 Demo Data Download
You can download the demo dataset to a specified folder using the following commands:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/mainbody_det_examples.tar -P ./dataset
tar -xf ./dataset/mainbody_det_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
You can complete data validation with a single command:

```bash
python main.py -c paddlex/configs/mainbody_detection/PP-ShiTuV2_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/mainbody_det_examples
```
After executing the above command, PaddleX will validate the dataset and collect its basic information. Upon successful execution, the log will print the message `Check dataset passed !`. The validation result file will be saved in `./output/check_dataset_result.json`, and related outputs will be saved in the `./output/check_dataset` directory of the current directory. The output directory includes visualized example images and histograms of sample distributions.

<details>
  <summary>👉 <b>Details of validation results (click to expand)</b></summary>


The specific content of the validation result file is:

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 1,
    "train_samples": 701,
    "train_sample_paths": [
      "check_dataset/demo_img/road839.png",
      "check_dataset/demo_img/road363.png",
      "check_dataset/demo_img/road148.png"
    ],
    "val_samples": 176,
    "val_sample_paths": [
      "check_dataset/demo_img/road218.png",
      "check_dataset/demo_img/road681.png",
      "check_dataset/demo_img/road138.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/example_data/mainbody_det_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}
```
In the above validation results, `check_pass` being `True` indicates that the dataset format meets the requirements. The explanations for other indicators are as follows:

* `attributes.num_classes`：The number of classes in this dataset is 1.
* `attributes.train_samples`：The number of samples in the training set of this dataset is 701.
* `attributes.val_samples`：The number of samples in the validation set of this dataset is 176.
* `attributes.train_sample_paths`：A list of relative paths to the visualized images of samples in the training set of this dataset.
* `attributes.val_sample_paths`： A list of relative paths to the visualized images of samples in the validation set of this dataset.


The dataset validation also analyzes the distribution of sample counts across all classes in the dataset and generates a histogram (histogram.png) to visualize this distribution. 

![](/tmp/images/modules/subj_det/01.png)
</details>

#### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the dataset verification, you can convert the dataset format or re-split the training/validation ratio by **modifying the configuration file** or **appending hyperparameters**.

<details>
  <summary>👉 <b>Details on Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

Mainbody detection does not support data format conversion.

**(2) Dataset Splitting**

Dataset splitting parameters can be set by modifying the `CheckDataset` section in the configuration file. Some example parameters in the configuration file are explained below:

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
python main.py -c paddlex/configs/mainbody_detection/PP-ShiTuV2_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/mainbody_det_examples
```
After dataset splitting, the original annotation files will be renamed to `xxx.bak` in their original paths.

The above parameters can also be set by appending command-line arguments:

```bash
python main.py -c paddlex/configs/mainbody_detection/PP-ShiTuV2_det.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/mainbody_det_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### 4.2 Model Training
Model training can be completed with a single command, taking the training of `PP-ShiTuV2_det.yaml` as an example:

```bash
python main.py -c paddlex/configs/mainbody_detection/PP-ShiTuV2_det.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/mainbody_det_examples
```
The steps required are:

* Specify the `.yaml` configuration file path for the model (here it is `PP-ShiTuV2_det.yaml`)
* Specify the mode as model training: `-o Global.mode=train`
* Specify the training dataset path: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters for Model Tasks](../../instructions/config_parameters_common.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

* During model training, PaddleX automatically saves model weight files, defaulting to `output`. To specify a save path, use the `-o Global.output` field in the configuration file.
* PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.
* When training other models, specify the corresponding configuration file. The correspondence between models and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.md).
After completing model training, all outputs are saved in the specified output directory (default is `./output/`), typically```markdown
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it is `PP-ShiTuV2_det.yaml`)
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path of the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.md).
</details>

### **4.3 Model Evaluation**
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation, you can complete the evaluation with a single command:

```bash
python main.py -c paddlex/configs/mainbody_detection/PP-ShiTuV2_det.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/mainbody_det_examples
```
Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model（here it's `PP-ShiTuV2_det.yaml`）
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to [PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common.md)。

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>


When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as `-o Evaluate.weight_path=./output/best_model/best_model/model.pdparams`.

After completing the model evaluation, an `evaluate_result.json` file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully, and the model's evaluation metrics, including AP.

</details>

### **4.4 Model Inference**
After completing model training and evaluation, you can use the trained model weights for inference predictions. In PaddleX, model inference predictions can be achieved through two methods: command line and wheel package.

#### 4.4.1 Model Inference
* To perform inference predictions through the command line, simply use the following command:
```bash
python main.py -c paddlex/configs/mainbody_detection/PP-ShiTuV2_det.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it is `PP-ShiTuV2_det.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or directly into your own project.

1. **Pipeline Integration**

The main body detection module can be integrated into PaddleX pipelines such as **General Object Detection** (comming soon). Simply replace the model path to update the main body detection module of the relevant pipeline. In pipeline integration, you can use high-performance deployment and service-oriented deployment to deploy your trained model.

2. **Module Integration**

The weights you produce can be directly integrated into the main body detection module. You can refer to the Python example code in [Quick Integration](#三快速集成), simply replace the model with the path to your trained model.
