# Image Recognition Module Development Tutorial

## I. Overview
Image recognition is a crucial task in computer vision, primarily involving the automatic extraction of useful features from image data through deep learning methods to facilitate subsequent image retrieval tasks. The performance of this module directly impacts the accuracy and efficiency of downstream tasks. In practical applications, image recognition typically outputs a set of feature vectors that effectively represent the content, structure, texture, and other information of the image, which are then passed as input to the subsequent retrieval module for processing.

## II. Supported Model List

<details>
   <summary> 👉Model List Details</summary>

<table>
  <tr>
    <th>Model</th>
    <th>Recall@1 (%)</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time</th>
    <th>Model Size (M)</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>PP-ShiTuV2_rec</td>
    <td>84.2</td>
    <td>5.23428</td>
    <td>19.6005</td>
    <td>16.3 M</td>
    <td rowspan="3">PP-ShiTuV2 is a general image recognition system consisting of three modules: object detection, feature extraction, and vector retrieval. These models are part of the feature extraction module and can be selected based on system requirements.</td>
  </tr>
  <tr>
    <td>PP-ShiTuV2_rec_CLIP_vit_base</td>
    <td>88.69</td>
    <td>13.1957</td>
    <td>285.493</td>
    <td>306.6 M</td>
  </tr>
  <tr>
    <td>PP-ShiTuV2_rec_CLIP_vit_large</td>
    <td>91.03</td>
    <td>51.1284</td>
    <td>1131.28</td>
    <td>1.05 G</td>
  </tr>
</table>

**Note: The above accuracy metrics are Recall@1 from [AliProducts](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/en/training/PP-ShiTu/feature_extraction.md). All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**
</details>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.md)

After installing the wheel package, a few lines of code can complete the inference of the image recognition module. You can switch between models under this module freely, and you can also integrate the model inference of the image recognition module into your project.

```python
from paddlex import create_model
model = create_model("PP-ShiTuV2_rec")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_recognition_001.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference APIs, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better image recognition models. Before developing image recognition models with PaddleX, ensure you have installed the classification-related model training plugins for PaddleX. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation.md)

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides data validation functionality for each module, and **only data that passes validation can be used for model training**.  Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to [PaddleX Multi-Label Classification Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/ml_classification.md).


#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:
```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Inshop_examples.tar -P ./dataset
tar -xf ./dataset/Inshop_examples.tar -C ./dataset/
```
#### 4.1.2 Data Validation
A single command can complete data validation:
```bash
python main.py -c paddlex/configs/general_recognition/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.

<details>
  <summary>👉 <b>Details of Validation Results (Click to Expand)</b></summary>

The specific content of the validation result file is:

```bash

  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 1000,
    "train_sample_paths": [
      "check_dataset/demo_img/05_1_front.jpg",
      "check_dataset/demo_img/02_1_front.jpg",
      "check_dataset/demo_img/02_3_back.jpg",
      "check_dataset/demo_img/04_3_back.jpg",
      "check_dataset/demo_img/04_2_side.jpg",
      "check_dataset/demo_img/12_1_front.jpg",
      "check_dataset/demo_img/07_2_side.jpg",
      "check_dataset/demo_img/04_7_additional.jpg",
      "check_dataset/demo_img/04_4_full.jpg",
      "check_dataset/demo_img/01_1_front.jpg"
    ],
    "gallery_samples": 110,
    "gallery_sample_paths": [
      "check_dataset/demo_img/06_2_side.jpg",
      "check_dataset/demo_img/01_4_full.jpg",
      "check_dataset/demo_img/04_7_additional.jpg",
      "check_dataset/demo_img/02_1_front.jpg",
      "check_dataset/demo_img/02_3_back.jpg",
      "check_dataset/demo_img/02_3_back.jpg",
      "check_dataset/demo_img/02_4_full.jpg",
      "check_dataset/demo_img/03_4_full.jpg",
      "check_dataset/demo_img/02_2_side.jpg",
      "check_dataset/demo_img/03_2_side.jpg"
    ],
    "query_samples": 125,
    "query_sample_paths": [
      "check_dataset/demo_img/08_7_additional.jpg",
      "check_dataset/demo_img/01_7_additional.jpg",
      "check_dataset/demo_img/02_4_full.jpg",
      "check_dataset/demo_img/04_4_full.jpg",
      "check_dataset/demo_img/09_7_additional.jpg",
      "check_dataset/demo_img/04_3_back.jpg",
      "check_dataset/demo_img/02_1_front.jpg",
      "check_dataset/demo_img/06_2_side.jpg",
      "check_dataset/demo_img/02_7_additional.jpg",
      "check_dataset/demo_img/02_2_side.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/Inshop_examples",
  "show_type": "image",
  "dataset_type": "ShiTuRecDataset"
}
```
In the above validation results, `check_pass` being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:
* `attributes.train_samples`: The number of training samples in this dataset is 1000;
* `attributes.gallery_samples`: The number of gallery (or reference) samples in this dataset is 110;
* `attributes.query_samples`: The number of query samples in this dataset is 125;
* `attributes.train_sample_paths`: A list of relative paths to the visual images of training samples in this dataset;
* `attributes.gallery_sample_paths`: A list of relative paths to the visual images of gallery (or reference) samples in this dataset;
* `attributes.query_sample_paths`: A list of relative paths to the visual images of query samples in this dataset;

Additionally, the dataset verification also analyzes the number of images and image categories within the dataset, and generates a distribution histogram (histogram.png):

![](/tmp/images/modules/img_recognition/01.png)
</details>

### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the data verification, you can convert the dataset format and re-split the training/validation ratio by **modifying the configuration file** or **appending hyperparameters**.

<details>
  <summary>👉 <b>Details of Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

The image feature task supports converting `LabelMe` format datasets to `ShiTuRecDataset` format. The parameters for dataset format conversion can be set by modifying the fields under `CheckDataset` in the configuration file. Some example parameter descriptions in the configuration file are as follows:

* `CheckDataset`:
  * `convert`:
    * `enable`: Whether to perform dataset format conversion. The image feature task supports converting `LabelMe` format datasets to `ShiTuRecDataset` format, default is `False`;
    * `src_dataset_type`: If dataset format conversion is performed, the source dataset format needs to be set, default is `null`, optional value is `LabelMe`;

For example, if you want to convert a `LabelMe` format dataset to `ShiTuRecDataset` format, you need to modify the configuration file as follows:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/image_classification_labelme_examples.tar -P ./dataset
tar -xf ./dataset/image_classification_labelme_examples.tar -C ./dataset/
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
python main.py -c paddlex/configs/general_recognition/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/image_classification_labelme_examples
```

After the data conversion is executed, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters also support being set by appending command line arguments:

```bash
python main.py -c paddlex/configs/general_recognition/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/image_classification_labelme_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe 
```

**(2) Dataset Splitting**

The parameters for dataset splitting can be set by modifying the fields under `CheckDataset` in the configuration file. Some example parameter descriptions in the configuration file are as follows:

* `CheckDataset`:
  * `split`:
    * `enable`: Whether to re-split the dataset. When `True`, the dataset will be re-split, default is `False`;
    * `train_percent`: If the dataset is re-split, the percentage of the training set needs to be set, the type is any integer between 0-100, and it needs to ensure that the sum of `gallery_percent` and `query_percent` values is 100;

For example, if you want to re-split the dataset with 70% training set, 20% gallery set, and 10% query set, you need to modify the configuration file as follows:

```bash
......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 70
    gallery_percent: 20
    query_percent: 10
  ......
```

Then execute the command:

```bash
python main.py -c paddlex/configs/general_recognition/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples
```

After the data splitting is executed, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters also support being set by appending command line arguments:

```bash
python main.py -c paddlex/configs/general_recognition/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=70 \
    -o CheckDataset.split.gallery_percent=20 \
    -o CheckDataset.split.query_percent=10 
```
> ❗Note: Due to the specificity of image recognition model evaluation, data partitioning is meaningful only when the train, query, and gallery sets belong to the same category system. During the evaluation of recognition models, it is imperative that the gallery and query sets belong to the same category system, which may or may not be the same as the train set. If the gallery and query sets do not belong to the same category system as the train set, the evaluation after data partitioning becomes meaningless. It is recommended to proceed with caution.

</details>

### 4.2 Model Training
Model training can be completed with a single command, taking the training of the image feature model PP-ShiTuV2_rec as an example:

```bash
python main.py -c paddlex/configs/general_recognition/PP-ShiTuV2_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
The following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-ShiTuV2_rec.yaml`)
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
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/general_recognition/PP-ShiTuV2_rec.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-ShiTuV2_rec.yaml`)
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`. 
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file, detailed instructions can be found in [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>


When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as `-o Evaluate.weight_path=./output/best_model/best_model.pdparams`.

After completing the model evaluation, an `evaluate_result.json` file will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully and the model's evaluation metrics, including recall1、recall5、mAP；

</details>

### **4.4 Model Inference and Integration**
After completing model training and evaluation, you can use the trained model weights for inference prediction or Python integration.


#### 4.4.1 Model Inference
To perform inference prediction through the command line, simply use the following command:

```bash
python main.py -c paddlex/configs/general_recognition/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_recognition_001.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-ShiTuV2_rec.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`. 
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.md).

> ❗ Note: The inference result of the recognition model is a set of vectors, which requires a retrieval module to complete image recognition.

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or directly into your own project.

1.**Pipeline Integration**

The image recognition module can be integrated into the **General Image Recognition Pipeline** (comming soon) of PaddleX. Simply replace the model path to update the image recognition module of the relevant pipeline. In pipeline integration, you can use service-oriented deployment to deploy your trained model.

2.**Module Integration**

The weights you produce can be directly integrated into the image recognition module. Refer to the Python example code in [Quick Integration](#三Quick-Integration), and simply replace the model with the path to your trained model.
