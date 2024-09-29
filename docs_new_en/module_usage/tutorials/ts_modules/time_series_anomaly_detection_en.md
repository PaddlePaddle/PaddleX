# Time Series Anomaly Detection Module Development Tutorial

## I. Overview
Time series anomaly detection focuses on identifying abnormal points or periods in time series data that do not conform to expected patterns, trends, or periodic regularities. These anomalies can be caused by system failures, external shocks, data entry errors, or rare events, and are of great significance for timely response, risk assessment, and business decision-making.

## II. Supported Model List

<details>
   <summary> 👉 Model List Details</summary>

| Model Name | Precision | Recall | F1-Score | Model Size (M) | Description |
|-|-|-|-|-|-|
| AutoEncoder_ad_ad | 0.9898 | 0.9396 | 0.9641 | 72.8K | AutoEncoder_ad_ad is a simple, efficient, and easy-to-use time series anomaly detection model |
| Nonstationary_ad | 0.9855 | 0.8895 | 0.9351 | 1.5MB | Based on the transformer structure, optimized for anomaly detection in non-stationary time series |
| AutoEncoder_ad | 0.9936 | 0.8436 | 0.9125 | 32K | AutoEncoder_ad is a classic autoencoder-based, efficient, and easy-to-use time series anomaly detection model |
| PatchTST_ad | 0.9878 | 0.9070 | 0.9457 | 164K | PatchTST is a high-precision time series anomaly detection model that balances local patterns and global dependencies |
| TimesNet_ad | 0.9837 | 0.9480 | 0.9656 | 732K | Through multi-period analysis, TimesNet is an adaptive and high-precision time series anomaly detection model |

**Note: The above accuracy metrics are measured on the PSM dataset with a time series length of 100.**

</details>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For details, refer to the [PaddleX Local Installation Guide](../../../installation/installation_en.md)

After installing the wheel package, a few lines of code can complete the inference of the time series anomaly detection module. You can switch models under this module freely, and you can also integrate the model inference of the time series anomaly detection module into your project.

```bash
from paddlex import create_model
model = create_model("AutoEncoder_ad")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_csv("./output/")
```
For more information on using PaddleX's single model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API_en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better time series anomaly detection models. Before developing time series anomaly models with PaddleX, please ensure that the PaddleTS plugin is installed. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation_en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the dataset for the corresponding task module. PaddleX provides data validation functionality for each module, and only data that passes validation can be used for model training. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for subsequent model training, refer to the [PaddleX Time Series Anomaly Detection Task Module Data Annotation Tutorial](../../../data_annotations/time_series_modules/time_series_anomaly_detection_en.md).

#### 4.1.1 Demo Data Download
You can use the following command to download the demo dataset to a specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar -P ./dataset
tar -xf ./dataset/ts_anomaly_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
You can complete data validation with a single command:
```bash
python main.py -c paddlex/configs/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples
```
After executing the above command, PaddleX will validate the dataset, summarize its basic information, and print `Check dataset passed !` in the log if the command runs successfully. The validation result file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the current directory's `./output/check_dataset` directory, including example time series data.

<details>
  <summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>

The specific content of the validation result file is:

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 22032,
    "train_table": [
      [
        "timestamp",
        "feature_0",
        "...",
        "feature_24",
        "label"
      ],
      [
        0.0,
        0.7326893750079723,
        "...",
        0.1382488479262673,
        0.0
      ]
    ],
    "val_samples": 198290,
    "val_table": [
      [
        "timestamp",
        "feature_0",
        "...",
        "feature_24",
        "label"
      ],
      [
        22032.0,
        0.8604795809835284,
        "...",
        0.1428571428571428,
        0.0
      ]
    ]
  },
  "analysis": {
    "histogram": ""
  },
  "dataset_path": "./dataset/ts_anomaly_examples",
  "show_type": "csv",
  "dataset_type": "TSADDataset"
}
```

The verification results above indicate that `check_pass` being `True` means the dataset format meets the requirements. Explanations for other indicators are as follows:

* `attributes.train_samples`: The number of training samples in this dataset is 22032;
* `attributes.val_samples`: The number of validation samples in this dataset is 198290;
* `attributes.train_sample_paths`: A list of relative paths to the top 10 rows of training samples in this dataset;
* `attributes.val_sample_paths`: A list of relative paths to the top 10 rows of validation samples in this dataset.
**Note**: Only data that has passed validation can be used for training and evaluation.
</details>


### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the data validation, you can convert the dataset format and re-split the training/validation ratio by **modifying the configuration file** or **appending hyperparameters**.


<details>
  <summary>👉 <b>Details of Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

Time series anomaly detection supports converting `xlsx` and `xlss` format datasets to `csv` format.

Parameters related to dataset validation can be set by modifying the fields under `CheckDataset` in the configuration file. Some example parameter descriptions in the configuration file are as follows:

* `CheckDataset`:
  * `convert`:
    * `enable`: Whether to convert the dataset format, supporting `xlsx` and `xlss` formats to `CSV` format, default is `False`;
    * `src_dataset_type`: If dataset format conversion is performed, the source dataset format does not need to be set, default is `null`;

To enable format conversion, modify the configuration as follows:

```bash
......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: null
  ......
```
Then execute the command:

```bash
python main.py -c paddlex/configs/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples
```
The above parameters also support setting through appending command line arguments:

```bash
python main.py -c paddlex/configs/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples \
    -o CheckDataset.convert.enable=True
```

**(2) Dataset Splitting**

Parameters related to dataset validation can be set by modifying the fields under `CheckDataset` in the configuration file. Some example parameter descriptions in the configuration file are as follows:

* `CheckDataset`:
  * `convert`:
    * `enable`: Whether to convert the dataset format, `True` to enable dataset format conversion, default is `False`;
    * `src_dataset_type`: If dataset format conversion is performed, time series anomaly detection only supports converting xlsx annotation files to csv, the source dataset format does not need to be set, default is `null`;
  * `split`:
    * `enable`: Whether to re-split the dataset, `True` to enable dataset splitting, default is `False`;
    * `train_percent`: If re-splitting the dataset, set the percentage of the training set, an integer between 0-100, ensuring the sum with `val_percent` is 100;
    * `val_percent`: If re-splitting the dataset, set the percentage of the validation set, an integer between 0-100, ensuring the sum with `train_percent` is 100;

For example, if you want to re-split the dataset with 90% training set and 10% validation set, modify the configuration file as follows:

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
python main.py -c paddlex/configs/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples
```
After dataset splitting, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters also support setting through appending command line arguments:

```bash
python main.py -c paddlex/configs/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### 4.2 Model Training
Model training can be completed with just one command. Here, we use the Time Series Forecasting model (AutoEncoder_ad) as an example:

```bash
python main.py -c paddlex/configs/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples
```

You need to follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `AutoEncoder_ad.yaml`).
* Set the mode to model training: `-o Global.mode=train`
* Specify the training dataset path: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to train using the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX TS Configuration Parameters Documentation](../../instructions/config_parameters_time_series_en.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

* During model training, PaddleX automatically saves model weight files, with the default path being `output`. To specify a different save path, use the `-o Global.output` field in the configuration file.
* PaddleX abstracts the concepts of dynamic graph weights and static graph weights from you. During model training, both dynamic and static graph weights are produced, and static graph weights are used by default for model inference.
* When training other models, specify the corresponding configuration file. The mapping between models and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list_en.md).

After model training, all outputs are saved in the specified output directory (default is `./output/`), typically including:

* `train_result.json`: Training result record file, including whether the training task completed successfully, produced weight metrics, and related file paths.
* `train.log`: Training log file, recording model metric changes, loss changes, etc.
* `config.yaml`: Training configuration file, recording the hyperparameters used for this training session.
* `best_accuracy.pdparams.tar`, `scaler.pkl`, `.checkpoints`, `.inference`: Model weight-related files, including Model weight-related files, including network parameters, optimizers, and network architecture.
</details>


### 4.3 Model Evaluation
After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ts_anomaly_examples
```
Similar to model training, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it's `AutoEncoder_ad.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Time Series Task Model Configuration File Parameter Description](../../instructions/config_parameters_time_series_en.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as `-o Evaluate.weight_path=./output/best_model/model.pdparams`.

After completing the model evaluation, the following outputs are typically generated:

Upon completion of model evaluation, an `evaluate_result.json` file will be produced, which records the evaluation results, specifically indicating whether the evaluation task was completed successfully and the model's evaluation metrics, including `f1`, `recall`, and `precision`.

</details>

### 4.4 Model Inference and Integration
After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
To perform inference predictions through the command line, simply use the following command:

```bash
python main.py -c paddlex/configs/ts_anomaly_detection/AutoEncoder_ad.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv"
```
Similar to model training and evaluation, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it's `AutoEncoder_ad.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Time Series Task Model Configuration File Parameter Description](../../instructions/config_parameters_time_series_en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or directly into your own project.

1. **Pipeline Integration**

The time series prediction module can be integrated into PaddleX pipelines such as [Time Series Anomaly Detection](../../../pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection_en.md). Simply replace the model path to update the time series prediction model. In pipeline integration, you can use service deployment to deploy your obtained model.

2. **Module Integration**

The weights you produce can be directly integrated into the time series anomaly detection module. Refer to the Python example code in [Quick Integration](#iii-quick-integration), simply replace the model with the path to your trained model.
