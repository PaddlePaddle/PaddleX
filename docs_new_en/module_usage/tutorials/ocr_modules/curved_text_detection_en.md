# Seal Text Detection Module Development Tutorial

## I. Overview
The seal text detection module typically outputs multi-point bounding boxes around text regions, which are then passed as inputs to the distortion correction and text recognition modules for subsequent processing to identify the textual content of the seal. Recognizing seal text is an integral part of document processing and finds applications in various scenarios such as contract comparison, inventory access auditing, and invoice reimbursement verification. The seal text detection module serves as a subtask within OCR (Optical Character Recognition), responsible for locating and marking the regions containing seal text within an image. The performance of this module directly impacts the accuracy and efficiency of the entire seal text OCR system.

## II. Supported Model List

<details>
   <summary> 👉 Model List Details</summary>

|Model Name| Hmean（%）|GPU Inference Time (ms)|CPU Inference Time|Model Size (M)| Introduce |
|-|-|-|-|-|-|
|PP-OCRv4_server_seal_det |98.21|||109 M|The server-side seal text detection model of PP-OCRv4 boasts higher accuracy and is suitable for deployment on better-equipped servers.|
|PP-OCRv4_mobile_seal_det|96.47|||4.6 M| The mobile-side seal text detection model of PP-OCRv4, on the other hand, offers greater efficiency and is suitable for deployment on end devices.|


**Note: The evaluation set for the above accuracy metrics is a self-built dataset containing 500 circular seal images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

</details>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.md)


Just a few lines of code can complete the inference of the Seal Text Detection module, allowing you to easily switch between models under this module. You can also integrate the model inference of the the Seal Text Detection module into your project.

```bash
from paddlex import create_model
model = create_model("PP-OCRv4_server_seal_det")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.md).

## IV. Custom Development

If you seek higher accuracy, you can leverage PaddleX's custom development capabilities to develop better Seal Text Detection models. Before developing a Seal Text Detection model with PaddleX, ensure you have installed PaddleOCR plugin for PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Tutorial](https://github.com/AmberC0209/PaddleX/blob/docs_change/docs_new/installation/installation.md).

### 4.1 Dataset Preparation

Before model training, you need to prepare a dataset for the task. PaddleX provides data validation functionality for each module. **Only data that passes validation can be used for model training.** Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to [PaddleX Seal Text Detection Task Module Data Preparation Tutorial](/docs_new_en/data_annotations/cv_modules/text_detection_seal_en.md).

#### 4.1.1 Demo Data Download

You can download the demo dataset to a specified folder using the following commands:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_curve_det_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ocr_curve_det_dataset_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation

Data validation can be completed with a single command:

```bash
python main.py -c paddlex/configs/text_detection_seal/PP-OCRv4_server_seal_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_curve_det_dataset_examples
```

After executing the above command, PaddleX will verify the dataset and collect basic information about it. Once the command runs successfully, a message saying `Check dataset passed !` will be printed in the log. The verification results will be saved in `./output/check_dataset_result.json`, and related outputs will be stored in the `./output/check_dataset` directory, including visual examples of sample images and a histogram of sample distribution.


<details>
  <summary>👉 <b>Verification Result Details (click to expand)</b></summary>

The specific content of the verification result file is:

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 606,
    "train_sample_paths": [
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug07834.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug09943.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug04079.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug05701.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug08324.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug07451.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug09562.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug08237.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug01788.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug06481.png"
    ],
    "val_samples": 152,
    "val_sample_paths": [
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug03724.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug06456.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug04029.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug03603.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug05454.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug06269.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug00624.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug02818.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug00538.png",
      "..\/ocr_curve_det_dataset_examples\/images\/circle_Aug04935.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": ".\/ocr_curve_det_dataset_examples",
  "show_type": "image",
  "dataset_type": "TextDetDataset"
}
```



The verification results above indicate that `check_pass` being `True` means the dataset format meets the requirements. Explanations for other indicators are as follows:

* `attributes.train_samples`: The number of training samples in this dataset is 606;
* `attributes.val_samples`: The number of validation samples in this dataset is 152;
* `attributes.train_sample_paths`: A list of relative paths to the visualization images of training samples in this dataset;
* `attributes.val_sample_paths`: A list of relative paths to the visualization images of validation samples in this dataset;

The dataset verification also analyzes the distribution of sample numbers across all classes and plots a histogram (histogram.png):

![](/tmp/images/modules/seal_text_det/01.png)

</details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional) 
<details>
  <summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>

After completing dataset verification, you can convert the dataset format or re-split the training/validation ratio by modifying the configuration file or appending hyperparameters.

**(1) Dataset Format Conversion**

Seal text detection does not support data format conversion.

**(2) Dataset Splitting**

Parameters for dataset splitting can be set by modifying the `CheckDataset` fields in the configuration file. Example explanations for some parameters in the configuration file are as follows:

* `CheckDataset`:
  * `split`:
    * `enable`: Whether to enable re-splitting the dataset, set to `True` to perform dataset splitting, default is `False`;
    * `train_percent`: If re-splitting the dataset, set the percentage of the training set, which should be an integer between 0 and 100, ensuring the sum with `val_percent` is 100;

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
python main.py -c paddlex/configs/text_detection_seal/PP-OCRv4_server_seal_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_curve_det_dataset_examples
```
After dataset splitting, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters also support setting through appending command line arguments:

```bash
python main.py -c paddlex/configs/text_detection_seal/PP-OCRv4_server_seal_det.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_curve_det_dataset_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### 4.2 Model Training

Model training can be completed with just one command. Here, we use the Seal Text Detection model (PP-OCRv4_server_seal_det) as an example:

```bash
python main.py -c paddlex/configs/text_detection_seal/PP-OCRv4_server_seal_det.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ocr_curve_det_dataset_examples
```

You need to follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `PP-OCRv4_server_seal_det.yaml`).
* Set the mode to model training: `-o Global.mode=train`
* Specify the training dataset path: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to train using the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters Documentation](../../instructions/config_parameters_common.md).
</details>

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

* During model training, PaddleX automatically saves model weight files, with the default path being `output`. To specify a different save path, use the `-o Global.output` field in the configuration file.
* PaddleX abstracts the concepts of dynamic graph weights and static graph weights from you. During model training, both dynamic and static graph weights are produced, and static graph weights are used by default for model inference.
* When training other models, specify the corresponding configuration file. The mapping between models and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.md).

After model training, all outputs are saved in the specified output directory (default is `./output/`), typically including:

* `train_result.json`: Training result record file, including whether the training task completed successfully, produced weight metrics, and related file paths.
* `train.log`: Training log file, recording model metric changes, loss changes, etc.
* `config.yaml`: Training configuration file, recording the hyperparameters used for this training session.
* `.pdparams`, `.pdema`, `.pdopt.pdstate`, `.pdiparams`, `.pdmodel`: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, and static graph network structure.
</details>

### 4.3 Model Evaluation
After model training, you can evaluate the specified model weights on the validation set to verify model accuracy. Using PaddleX for model evaluation requires just one command:

```bash
python main.py -c paddlex/configs/text_detection_seal/PP-OCRv4_server_seal_det.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ocr_curve_det_dataset_examples
```

Similar to model training, follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `PP-OCRv4_server_seal_det.yaml`).
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the validation dataset path: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For more details, refer to the [PaddleX Common Configuration Parameters Documentation](../../instructions/config_parameters_common.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

When evaluating the model, you need to specify the model weight file path. Each configuration file has a default weight save path. If you need to change it, simply append the command line parameter, e.g., `-o Evaluate.weight_path=./output/best_model/best_model.pdparams`.

After model evaluation, the following outputs are typically produced:

* `evaluate_result.json`: Records the evaluation results, specifically whether the evaluation task completed successfully and the model's evaluation metrics, including precision, recall and Hmean.

</details>

### 4.4 Model Inference and Integration
After model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
To perform inference predictions via the command line, use the following command:


```bash
python main.py -c paddlex/configs/text_detection_seal/PP-OCRv4_server_seal_det.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_accuracy/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png"
```

Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it's `PP-OCRv4_server_seal_det.yaml`)

* Set the mode to model inference prediction: `-o Global.mode=predict`

* Specify the model weights path: -o Predict.model_dir="./output/best_accuracy/inference"

Specify the input data path: `-o Predict.inputh="..."` Other related parameters can be set by modifying the fields under Global and Predict in the `.yaml` configuration file. For details, refer to PaddleX Common Model Configuration File Parameter Description.

Alternatively, you can use the PaddleX wheel package for inference, easily integrating the model into your own projects.

#### 4.4.2 Model Integration

The model can be directly integrated into the PaddleX pipeline or into your own projects.

1. **Pipeline Integration**

The document Seal Text Detection module can be integrated into PaddleX pipelines such as the [General OCR Pipeline](../../../pipeline_usage/tutorials/ocr_pipelines/OCR_en.md) and [Document Scene Information Extraction Pipeline v3 (PP-ChatOCRv3)](../../../pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_en.md). Simply replace the model path to update the text detection module of the relevant pipeline.

2. **Module Integration**

The weights you produce can be directly integrated into the Seal Text Detection module. You can refer to the Python sample code in [Quick Integration](#quick-integration) and just replace the model with the path to the model you trained.
    
