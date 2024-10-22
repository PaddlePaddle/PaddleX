[简体中文](formula_recognition.md) | English

# Formula Recognition Module Development Tutorial

## I. Overview
The formula recognition module is a crucial component of OCR (Optical Character Recognition) systems, responsible for converting mathematical formulas in images into editable text or computer-readable formats. The performance of this module directly impacts the accuracy and efficiency of the entire OCR system. The module typically outputs LaTeX or MathML codes of mathematical formulas, which are then passed on to the text understanding module for further processing.

## II. Supported Model List


<table>
  <tr>
    <th>Model</th>
    <th>Normed Edit Distance</th>
    <th>BLEU Score</th>
    <th>ExpRate (%)</th>
    <th>Model Size (M)</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>LaTeX_OCR_rec</td>
    <td>0.8821</td>
    <td>0.0823</td>
    <td>40.01</td>
    <td>89.7 M</td>
    <td>LaTeX-OCR is a formula recognition algorithm based on an autoregressive large model. By adopting Hybrid ViT as the backbone network and transformer as the decoder, it significantly improves the accuracy of formula recognition.</td>
  </tr>
  
</table>

**Note: The above accuracy metrics are measured on the LaTeX-OCR formula recognition test set.**

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation_en.md).

After installing the wheel package, a few lines of code can complete the inference of the formula recognition module. You can switch models under this module freely, and you can also integrate the model inference of the formula recognition module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png) to your local machine.

```bash
from paddlex import create_model
model = create_model("LaTeX_OCR_rec")
output = model.predict("general_formula_rec_001.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API_en.md).

## IV. Custom Development
If you aim for higher accuracy with existing models, you can leverage PaddleX's custom development capabilities to develop better formula recognition models. Before developing formula recognition models with PaddleX, ensure you have installed the PaddleOCR-related model training plugins for PaddleX. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation_en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides a data validation function for each module, and **only data that passes the validation can be used for model training**. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to the [LaTeX-OCR Formula Recognition Project](https://github.com/lukas-blecher/LaTeX-OCR).

#### 4.1.1 Demo Data Download
You can download the demo dataset to a specified folder using the following command:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_latexocr_dataset_example.tar -P ./dataset
tar -xf ./dataset/ocr_rec_latexocr_dataset_example.tar -C ./dataset/
```

#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/formula_recognition/LaTeX_OCR_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example
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
    "train_samples": 9452,
    "train_sample_paths": [
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0109284.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0217434.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0166758.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0022294.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/val_0071799.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0017043.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0026204.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0209202.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/val_0157332.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0232582.png"
    ],
    "val_samples": 1050,
    "val_sample_paths": [
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0070221.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0157901.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0085392.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0196480.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0096180.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0136149.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0143310.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0004560.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0115191.png",
      "../dataset/ocr_rec_latexocr_dataset_example/images/train_0015323.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/ocr_rec_latexocr_dataset_example",
  "show_type": "image",
  "dataset_type": "LaTeXOCRDataset"
}
```
In the above validation results, `check_pass` being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:
* `attributes.train_samples`: The number of training samples in this dataset is 9452;
* `attributes.val_samples`: The number of validation samples in this dataset is 1050;
* `attributes.train_sample_paths`: A list of relative paths to the visualized training samples in this dataset;
* `attributes.val_sample_paths`: A list of relative paths to the visualized validation samples in this dataset;

Additionally, the dataset verification also analyzes the distribution of sample numbers across all categories in the dataset and generates a distribution histogram (`histogram.png`):
![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/formula_recognition/01.jpg)
</details>


### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the data verification, you can convert the dataset format and re-split the training/validation ratio by **modifying the configuration file** or **appending hyperparameters**.

<details>
  <summary>👉 <b>Details of Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

The formula recognition supports converting `PKL` format datasets to `LaTeXOCRDataset` format. The parameters for dataset format conversion can be set by modifying the fields under `CheckDataset` in the configuration file. Examples of some parameters in the configuration file are as follows:

* `CheckDataset`:
  * `convert`:
    * `enable`: Whether to perform dataset format conversion. Formula recognition supports converting `PKL` format datasets to `LaTeXOCRDataset` format, default is `True`;
    * `src_dataset_type`: If dataset format conversion is performed, the source dataset format needs to be set, default is `PKL`, optional value is `PKL`;

For example, if you want to convert a `PKL` format dataset to `LaTeXOCRDataset` format, you need to modify the configuration file as follows:

```bash
......
CheckDataset:
  ......
  convert: 
    enable: True
    src_dataset_type: PKL
  ......
```
Then execute the command:

```bash
python main.py -c paddlex/configs/formula_recognition/LaTeX_OCR_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example
```
After the data conversion is executed, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters also support being set by appending command line arguments:

```bash
python main.py -c  paddlex/configs/formula_recognition/LaTeX_OCR_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=PKL
```
**(2) Dataset Splitting**

The parameters for dataset splitting can be set by modifying the fields under `CheckDataset` in the configuration file. Examples of some parameters in the configuration file are as follows:

* `CheckDataset`:
  * `split`:
    * `enable`: Whether to re-split the dataset. When set to `True`, dataset splitting is performed, default is `False`;
    * `train_percent`: If the dataset is re-split, the percentage of the training set needs to be set, which is an integer between 0 and 100, and the sum with `val_percent` should be 100;

For example, if you want to re-split the dataset with 90% for the training set and 10% for the validation set, you need to modify the configuration file as follows:

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
python main.py -c paddlex/configs/formula_recognition/LaTeX_OCR_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example
```
After the data splitting is executed, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters also support being set by appending command line arguments:

```bash
python main.py -c  paddlex/configs/formula_recognition/LaTeX_OCR_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### 4.2 Model Training
Model training can be completed with a single command, taking the training of the formula recognition model LaTeX_OCR_rec as an example:

```bash
python main.py -c paddlex/configs/formula_recognition/LaTeX_OCR_rec.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example
```
The following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `LaTeX_OCR_rec.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list_en.md))
* Set the mode to model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`. 
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file instructions for the corresponding task module of the model [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common_en.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

* During model training, PaddleX automatically saves the model weight files, with the default being `output`. If you need to specify a save path, you can set it through the `-o Global.output` field in the configuration file.
* PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.
* After completing the model training, all outputs are saved in the specified output directory (default is `./output/`), typically including:

* `train_result.json`: Training result record file, recording whether the training task was completed normally, as well as the output weight metrics, related file paths, etc.;
* `train.log`: Training log file, recording changes in model metrics and loss during training;
* `config.yaml`: Training configuration file, recording the hyperparameter configuration for this training session;
* `.pdparams`, `.pdema`, `.pdopt.pdstate`, `.pdiparams`, `.pdmodel`: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;
</details>


## **4.3 Model Evaluation**
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/formula_recognition/LaTeX_OCR_rec.yaml  \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example
```
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `LaTeX_OCR_rec.yaml`)
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`. 
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file, detailed instructions can be found in [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common_en.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>


When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as `-o Evaluate.weight_path=./output/best_accuracy/best_accuracy.pdparams`.

After completing the model evaluation, an `evaluate_result.json` file will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully and the model's evaluation metrics, including recall1、recall5、mAP；

</details>


### **4.4 Model Inference and Integration**
After completing model training and evaluation, you can use the trained model weights for inference prediction or Python integration.


#### 4.4.1 Model Inference
To perform inference prediction through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png) to your local machine.
```bash
python main.py -c paddlex/configs/formula_recognition/LaTeX_OCR_rec.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_accuracy/inference" \
    -o Predict.input="general_formula_rec_001.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `LaTeX_OCR_rec.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_accuracy/inference"`
* Specify the input data path: `-o Predict.input="..."`. 
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common_en.md).


#### 4.4.2 Model Integration

The weights you produce can be directly integrated into the formula recognition module. Refer to the Python example code in [Quick Integration](#iii-quick-integration), and simply replace the model with the path to your trained model.
