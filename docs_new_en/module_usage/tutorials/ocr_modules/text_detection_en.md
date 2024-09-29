# Text Detection Module Development Tutorial

## I. Overview
The text detection module is a crucial component in OCR (Optical Character Recognition) systems, responsible for locating and marking regions containing text within images. The performance of this module directly impacts the accuracy and efficiency of the entire OCR system. The text detection module typically outputs bounding boxes (Bounding Boxes) for text regions, which are then passed on to the text recognition module for further processing.

## II. Supported Models
| Model | Detection Hmean (%) | GPU Inference Time (ms) | CPU Inference Time | Model Size (M) | Description |
|-|-|-|-|-|-|
| PP-OCRv4_server_det | 82.69 | 83.3501 | 2434.01 | 109 | The server-side text detection model of PP-OCRv4, featuring higher accuracy and suitable for deployment on high-performance servers |
| PP-OCRv4_mobile_det | 77.79 | 10.6923 | 120.177 | 4.7 | The mobile text detection model of PP-OCRv4, optimized for efficiency and suitable for deployment on edge devices |

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation_en.md).

Just a few lines of code can complete the inference of the text detection module, allowing you to easily switch between models under this module. You can also integrate the model inference of the text detection module into your project.

```python
from paddlex import create_model
model = create_model("PP-OCRv4_mobile_det")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference APIs, refer to the [PaddleX Single Model Python Script Usage Instructions](../../../module_usage/instructions/model_python_API_en.MD).

## IV. Custom Development
If you seek even higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better text detection models. Before developing text detection models with PaddleX, ensure you have installed the PaddleOCR plugin for PaddleX. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation_en.md).

### 4.1 Data Preparation
Before model training, you need to prepare a dataset for the specific task module. PaddleX provides data validation functionality for each module, and **only data that passes validation can be used for model training**.
Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to the [PaddleX Text Detection/Text Recognition Task Module Data Annotation Tutorial](../../../data_annotations/ocr_modules/text_detection_recognition_en.md).

#### 4.1.1 Demo Data Download

You can use the following commands to download the demo dataset to a specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_det_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ocr_det_dataset_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation

A single command can complete data validation:

```bash
python main.py -c paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_det_dataset_examples
```
After executing the above command, PaddleX will validate the dataset, summarize its basic information, and print `Check dataset passed !` in the log upon successful completion. 

<details>
<summary>👉 <b>Validation Results Details (Click to Expand)</b></summary>

The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the current directory's `./output/check_dataset` directory, including visualized sample images and sample distribution histograms.

</details>

### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)

After completing data validation, you can convert the dataset format and re-split the training/validation ratio by **modifying the configuration file** or **appending hyperparameters**.

<details>
  <summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

Text detection does not support data format conversion.

**(2) Dataset Splitting**

The parameters for dataset splitting can be set by modifying the `CheckDataset` section in the configuration file. Below are some example explanations for the parameters in the configuration file:

* `CheckDataset`:
  * `split`:
    * `enable`: Whether to re-split the dataset. Set to `True` to enable dataset splitting, default is `False`;
    * `train_percent`: If re-splitting the dataset, set the percentage of the training set. The type is any integer between 0-100, and the sum with `val_percent` must be 100;

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
python main.py -c paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_det_dataset_examples
```
After dataset splitting, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters can also be set by appending command-line arguments:

```bash
python main.py -c paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_det_dataset_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### 4.2 Model Training
Model training can be completed with a single command. Here's an example of training the PP-OCRv4 mobile text detection model (`PP-OCRv4_mobile_det`):

```bash
python main.py -c paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ocr_det_dataset_examples
```
The steps required are:

* Specify the path to the model's `.yaml` configuration file (here it's `PP-OCRv4_mobile_det.yaml`)
* Set the mode to model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters Documentation](../../../module_usage/instructions/config_parameters_common_en.md).

<details>
  <summary>👉 <b>More Information (Click to Expand)</b></summary>

* During model training, PaddleX automatically saves model weight files, with the default path being `output`. To specify a different save path, use the `-o Global.output` field in the configuration file.
* PaddleX abstracts away the concepts of dynamic graph weights and static graph weights from you. During model training, both dynamic and static graph weights are produced, and static graph weights are used by default for model inference.
* When training other models, specify the corresponding configuration file. The correspondence between models and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list_en.md)
</details>

### **4.4 Model Inference and Model Integration**
After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
To perform inference predictions via the command line, simply use the following command:

```bash
python main.py -c paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_accuracy/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it's `PP-OCRv4_mobile_det.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_accuracy/inference"`
* Specify the input data path: `-o Predict.inputh="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../../module_usage/instructions/config_parameters_common_en.md).

* Alternatively, you can use the PaddleX wheel package for inference, easily integrating the model into your own projects.

#### 4.4.2 Model Integration
Models can be directly integrated into PaddleX pipelines or into your own projects.

1.**Pipeline Integration**

The text detection module can be integrated into PaddleX pipelines such as the [General OCR Pipeline](../../../pipeline_usage/tutorials/ocr_pipelines/OCR_en.md), [Table Recognition Pipeline](../../../pipeline_usage/tutorials/ocr_pipelines/table_recognition_en.md), and [Document Scene Information Extraction Pipeline v3 (PP-ChatOCRv3)](../../../pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_en.md). Simply replace the model path to update the text detection module of the relevant pipeline.

2.**Module Integration**

The model weights you produce can be directly integrated into the text detection module. Refer to the Python example code in [Quick Integration](#iii-quick-integration), and simply replace the model with the path to your trained model.