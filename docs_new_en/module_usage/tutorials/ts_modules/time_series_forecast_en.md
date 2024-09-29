# Time Series Forecasting Module Development Tutorial

## I. Overview
Time series forecasting aims to predict the possible values or states at a future point in time or within a future time period by analyzing patterns, trends, periodicity, and other characteristics in historical data. This helps enterprises and organizations make more accurate decisions, optimize resource allocation, reduce risks, and seize potential market opportunities. These time series data typically originate from various sensors, economic activities, social behaviors, and other real-world application scenarios. For example, stock prices, temperature changes, website traffic, sales data, and the like are all typical examples of time series data.

## II. Supported Model List

<details>
   <summary> 👉 Model List Details</summary>

|Model Name| mse | mae |Model Size (M)| Introduce |
|-|-|-|-|-|
|DLinear|0.382|0.394|76k|Simple structure, high efficiency and easy-to-use time series prediction model|
|Nonstationary|0.600|0.515|60.3M|Based on the transformer structure, targeted optimization of long-term time series prediction models for non-stationary time series|
|PatchTST|0.385|0.397|2.2M|High-precision long-term time series prediction model that takes into account both local patterns and global dependencies |
|TiDE|0.405|0.412|34.9M|High-precision model suitable for handling multivariate, long-term time series prediction problems|
|TimesNet|0.417|0.431|5.2M|Through multi-period analysis, TimesNet is a highly adaptable high-precision time series analysis model|


**Note: The above accuracy metrics are measured on the [ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar) test dataset, with an input sequence length of 96, and a prediction sequence length of 96 for all models except TiDE, which has a prediction sequence length of 720.**


</details>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation_en.md)


Just a few lines of code can complete the inference of the Time Series Forecasting module, allowing you to easily switch between models under this module. You can also integrate the model inference of the the Time Series Forecasting module into your project.

```bash
from paddlex import create_model
model = create_model("DLinear")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_csv("./output/")
```
For more information on using PaddleX's single-model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API_en.md).

## IV. Custom Development

If you seek higher accuracy, you can leverage PaddleX's custom development capabilities to develop better Time Series Forecasting models. Before developing a Time Series Forecasting model with PaddleX, ensure you have installed PaddleClas plugin for PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Tutorial](../../installation/installation_en.md).

### 4.1 Dataset Preparation

Before model training, you need to prepare a dataset for the task. PaddleX provides data validation functionality for each module. **Only data that passes validation can be used for model training.** Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to [PaddleX Time Series Forecasting Task Module Data Preparation Tutorial](/../../../data_annotations/time_series_modules/time_series_forecasting_en.md).

#### 4.1.1 Demo Data Download

You can download the demo dataset to a specified folder using the following commands:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ts_dataset_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation

Data validation can be completed with a single command:

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
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
    "train_samples": 12194,
    "train_table": [
      [
        "date",
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
        "OT"
      ],
      [
        "2016-07-01 00:00:00",
        5.827000141143799,
        2.009000062942505,
        1.5989999771118164,
        0.4620000123977661,
        4.203000068664552,
        1.3400000333786009,
        30.5310001373291
      ],
      [
        "2016-07-01 01:00:00",
        5.692999839782715,
        2.075999975204468,
        1.4919999837875366,
        0.4259999990463257,
        4.142000198364259,
        1.371000051498413,
        27.78700065612793
      ]
    ],
    "val_samples": 3484,
    "val_table": [
      [
        "date",
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
        "OT"
      ],
      [
        "2017-11-21 02:00:00",
        12.994000434875488,
        4.889999866485597,
        10.055999755859377,
        2.878000020980835,
        2.559000015258789,
        1.2489999532699585,
        4.7129998207092285
      ],
      [
        "2017-11-21 03:00:00",
        11.92199993133545,
        4.554999828338623,
        9.097000122070312,
        3.0920000076293945,
        2.559000015258789,
        1.2790000438690186,
        4.8540000915527335
      ]
    ]
  },
  "analysis": {
    "histogram": ""
  },
  "dataset_path": "./dataset/ts_dataset_examples",
  "show_type": "csv",
  "dataset_type": "TSDataset"
}
```

</details>

The verification results above indicate that `check_pass` being `True` means the dataset format meets the requirements. Explanations for other indicators are as follows:

* `attributes.train_samples`: The number of training samples in this dataset is 12194;
* `attributes.val_samples`: The number of validation samples in this dataset is 3484;
* `attributes.train_sample_paths`: A list of relative paths to the top 10 rows of training samples in this dataset;
* `attributes.val_sample_paths`: A list of relative paths to the top 10 rows of validation samples in this dataset;


</details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional) (Click to Expand)
<details>
  <summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>

After completing dataset verification, you can convert the dataset format or re-split the training/validation ratio by modifying the configuration file or appending hyperparameters.

**(1) Dataset Format Conversion**

Time Series Forecasting supports converting `xlsx` and `xlss` format datasets to the required format.

Parameters related to dataset verification can be set by modifying the `CheckDataset` fields in the configuration file. Example explanations for some parameters in the configuration file are as follows:

* `CheckDataset`:
  * `convert`:
    * `enable`: Whether to enable dataset format conversion, supporting `xlsx` and `xlss` format conversion, default is `False`;
    * `src_dataset_type`: If dataset format conversion is enabled, the source dataset format needs to be set, default is `null`.


Modify the `paddlex/configs/ts_forecast/DLinear.yaml` configuration as follows:

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
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_forecast_to_convert
```

Of course, the above parameters also support being set by appending command-line arguments. For a `LabelMe` format dataset, the command is:

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_forecast_to_convert \
    -o CheckDataset.convert.enable=True \
```

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
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
```
After dataset splitting, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters also support setting through appending command line arguments:

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ts_dataset_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```

</details>

### 4.2 Model Training

Model training can be completed with just one command. Here, we use the Time Series Forecasting model (DLinear) as an example:

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
```

You need to follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `DLinear.yaml`).
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
After model training, you can evaluate the specified model weights on the validation set to verify model accuracy. Using PaddleX for model evaluation requires just one command:

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ts_dataset_examples
```

Similar to model training, follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `DLinear.yaml`).
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the validation dataset path: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For more details, refer to the [PaddleX TS Configuration Parameters Documentation](../../instructions/config_parameters_time_series_en.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

When evaluating the model, you need to specify the model weight file path. Each configuration file has a default weight save path. If you need to change it, simply append the command line parameter, e.g., `-o Evaluate.weight_path=./output/best_model/best_model.pdparams`.

After model evaluation, the following outputs are typically produced:

* `evaluate_result.json`: Records the evaluation results, specifically whether the evaluation task completed successfully and the model's evaluation metrics, including `mse` and `mae`.

</details>

### 4.4 Model Inference and Integration
After model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
To perform inference predictions via the command line, use the following command:


```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv"
```

Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it's `DLinear.yaml`)

* Set the mode to model inference prediction: `-o Global.mode=predict`

* Specify the model weights path: `-o Predict.model_dir="./output/best_accuracy/inference"`

Specify the input data path: `-o Predict.inputh="..."` Other related parameters can be set by modifying the fields under Global and Predict in the `.yaml` configuration file. For details, refer to PaddleX Common Model Configuration File Parameter Description.

Alternatively, you can use the PaddleX wheel package for inference, easily integrating the model into your own projects.

#### 4.4.2 Model Integration

The model can be directly integrated into the PaddleX pipeline or into your own projects.

1. **Pipeline Integration**

The Time Series Forecasting module can be integrated into PaddleX pipelines such as the [Time Series Forecasting Pipeline (ts_fc)](../../../pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting_en.md). Simply replace the model path to update the Time Series Forecasting module's model.

2. **Module Integration**

The weights you produce can be directly integrated into the Time Series Forecasting module. You can refer to the Python sample code in [Quick Integration](#iii-quick-integration) and just replace the model with the path to the model you trained.
    
