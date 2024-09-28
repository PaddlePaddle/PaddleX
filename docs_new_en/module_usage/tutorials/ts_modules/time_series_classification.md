# Time Series Classification Module Development Tutorial

## I. Overview
Time series classification involves identifying and categorizing different patterns in time series data by analyzing trends, periodicity, seasonality, and other factors that vary over time. This technique is widely used in medical diagnosis and other fields, effectively classifying key information in time series data to provide robust support for decision-making.

## II. Supported Model List

<details>
   <summary> 👉 Model List Details</summary>

|Model Name|Acc(%)|Model Size (M)|Description|
|-|-|-|-|
|TimesNet_cls|87.5|792K|TimesNet is an adaptive and high-accuracy time series classification model through multi-period analysis|

**Note: The evaluation set for the above accuracy metrics is UWaveGestureLibrary.**

</details>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to [PaddleX Local Installation Guide](../../../installation/installation.md)

After installing the wheel package, you can perform inference for the time series classification module with just a few lines of code. You can switch models under this module freely, and you can also integrate the model inference of the time series classification module into your project.

```bash
from paddlex import create_model
model = create_model("TimesNet_cls")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_csv("./output/")
```
For more information on using PaddleX's single-model inference APIs, refer to [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.md).

## IV. Custom Development
If you aim for higher accuracy with existing models, you can leverage PaddleX's custom development capabilities to develop better time series classification models. Before using PaddleX to develop time series classification models, ensure you have installed the PaddleTS plugin. Refer to the [PaddleX Local Installation Guide](../../../installation/installation.md) for the installation process.

### 4.1 Data Preparation
Before model training, you need to prepare the dataset for the corresponding task module. PaddleX provides data validation functionality for each module, and **only data that passes validation can be used for model training**. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for subsequent model training, refer to [PaddleX Time Series Classification Task Module Data Annotation Tutorial](../../../data_annotations/time_series_modules/time_series_classification.md).

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_classify_examples.tar -P ./dataset
tar -xf ./dataset/ts_classify_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
You can complete data validation with a single command:

```bash
python main.py -c paddlex/configs/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_classify_examples
```
After executing the above command, PaddleX will validate the dataset, summarize its basic information, and print `Check dataset passed !` in the log if the command runs successfully. The validation result file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the current directory's `./output/check_dataset` directory, including example time series data and class distribution histograms.



<details>
  <summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>

The specific content of the validation result file is:

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 82620,
    "train_table": [
      [
        "Unnamed: 0",
        "group_id",
        "dim_0",
        ...,
        "dim_60",
        "label",
        "time"
      ],
      [
        0.0,
        0.0,
        0.000949,
        ...,
        0.12107,
        1.0,
        0.0
      ]
    ],
    "val_samples": 83025,
    "val_table": [
      [
        "Unnamed: 0",
        "group_id",
        "dim_0",
        ...,
        "dim_60",
        "label",
        "time"
      ],
      [
        0.0,
        0.0,
        0.004578,
        ...,
        0.15728,
        1.0,
        0.0
      ]
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/ts_classify_examples",
  "show_type": "csv",
  "dataset_type": "TSCLSDataset"
}
```

The verification results above indicate that `check_pass` being `True` means the dataset format meets the requirements. Explanations for other indicators are as follows:


* `attributes.train_samples`: The number of training samples in this dataset is 12194;
* `attributes.val_samples`: The number of validation samples in this dataset is 3484;
* `attributes.train_sample_paths`: A list of relative paths to the top 10 rows of training samples in this dataset;
* `attributes.val_sample_paths`: A list of relative paths to the top 10 rows of validation samples in this dataset;

Furthermore, the dataset validation also involved an analysis of the distribution of sample numbers across all categories within the dataset, and a distribution histogram (histogram.png) was generated accordingly.

![](/tmp/images/modules/time_classification/01.png)


**Note**: Only data that has passed validation can be used for training and evaluation.
</details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format and re-split the training/validation ratio by **modifying the configuration file** or **appending hyperparameters**.

<details>
  <summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

Time-series classification supports converting `xlsx` and `xlss` format datasets to `csv` format.

Parameters related to dataset validation can be set by modifying the fields under `CheckDataset` in the configuration file. Examples of some parameters in the configuration file are as follows:

* `CheckDataset`:
  * `convert`:
    * `enable`: Whether to perform dataset format conversion, supporting conversion from `xlsx` and `xlss` formats to `CSV` format, default is `False`;
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
python main.py -c paddlex/configs/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_classify_examples
```
The above parameters can also be set by appending command line arguments:

```bash
python main.py -c paddlex/configs/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_classify_examples \
    -o CheckDataset.convert.enable=True
```

**(2) Dataset Splitting**

Parameters related to dataset validation can be set by modifying the fields under `CheckDataset` in the configuration file. Examples of some parameters in the configuration file are as follows:

* `CheckDataset`:
  * `convert`:
    * `enable`: Whether to perform dataset format conversion, `True` to enable, default is `False`;
    * `src_dataset_type`: If dataset format conversion is performed, time-series classification only supports converting xlsx annotation files to csv, the source dataset format does not need to be set, default is `null`;
  * `split`:
    * `enable`: Whether to re-split the dataset, `True` to enable, default is `False`;
    * `train_percent`: If the dataset is re-split, the percentage of the training set needs to be set, an integer between 0-100, ensuring the sum with `val_percent` is 100;
    * `val_percent`: If the dataset is re-split, the percentage of the validation set needs to be set, an integer between 0-100, ensuring the sum with `train_percent` is 100;

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
python main.py -c paddlex/configs/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_classify_examples
```
After dataset splitting, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters can also be set by appending command line arguments:

```bash
python main.py -c paddlex/configs/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_classify_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>


### 4.2 Model Training

Model training can be completed with just one command. Here, we use the Time Series Forecasting model (TimesNet_cls) as an example:

```bash
python main.py -c paddlex/configs/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ts_classify_examples
```

You need to follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `TimesNet_cls.yaml`).
* Set the mode to model training: `-o Global.mode=train`
* Specify the training dataset path: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to train using the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX TS Configuration Parameters Documentation](../../instructions/config_parameters_ts_en.md).

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
python main.py -c paddlex/configs/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ts_classify_examples
```
Similar to model training, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it's `TimesNet_cls.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other relevant parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Time Series Task Model Configuration File Parameter Description](../../instructions/config_parameters_time_series.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as `-o Evaluate.weight_path=./output/best_model/model.pdparams`.

After completing the model evaluation, typically, the following outputs are generated:

Upon completion of model evaluation, an `evaluate_result.json` file is produced, which records the evaluation results, specifically whether the evaluation task was completed successfully and the model's evaluation metrics, including Top-1 Accuracy.

</details>

### 4.4 Model Inference and Model Integration
After completing model training and evaluation, you can use the trained model weights for inference prediction or Python integration.

#### 4.4.1 Model Inference
To perform inference prediction via the command line, simply use the following command:

```bash
python main.py -c paddlex/configs/ts_classification/TimesNet_cls.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv"
```
Similar to model training and evaluation, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it's `TimesNet_cls.yaml` - **Note**: This should likely be `TimesNet_cls.yaml` for consistency)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other relevant parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Time Series Task Model Configuration File Parameter Description](../../../module_usage/instructions/config_parameters_time_series.md).

#### 4.4.2 Model Integration
Models can be directly integrated into the PaddleX pipeline or directly into your own projects.

1. **Pipeline Integration**

The time series prediction module can be integrated into PaddleX pipelines such as [Time Series Classification](../../../pipeline_usage/tutorials/time_series_pipelines/time_series_classification.md). Simply replace the model path to update the time series prediction model. In pipeline integration, you can use service deployment to deploy your trained model.

2. **Module Integration**

The weights you produce can be directly integrated into the time series classification module. Refer to the Python example code in [Quick Integration](#iii-quick-integration) (Note: This section header is in Chinese and should be translated or removed for consistency), simply replace the model with the path to your trained model.
