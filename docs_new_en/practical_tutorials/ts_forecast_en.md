# PaddleX 3.0 Time Series Forecasting Pipeline — Long-term Electricity Consumption Forecasting Tutorial

PaddleX offers a rich set of pipelines, each consisting of one or more models tailored to solve specific scenario tasks. All PaddleX pipelines support quick trials, and if the results are not satisfactory, you can also fine-tune the models with private data. PaddleX provides Python APIs to easily integrate pipelines into personal projects. Before use, you need to install PaddleX. For installation instructions, refer to [PaddleX Local Installation Guide](../INSTALL.md) or [PaddleX Installation Guide](../../../installation/installation.md). This tutorial introduces the usage of the time series forecasting pipeline tool with an example of long-term electricity consumption forecasting.

## 1. Select a Pipeline
First, choose the corresponding PaddleX pipeline based on your task scenario. The goal of this task is to predict future electricity consumption based on historical data. Recognizing this as a time series forecasting task, we will use PaddleX's time series forecasting pipeline. If you're unsure about the correspondence between tasks and pipelines, you can refer to the [PaddleX Pipeline List (CPU/GPU)](../../../support_list/models_list.md) for an overview of pipeline capabilities.

## 2. Quick Experience
PaddleX offers two ways to experience its pipelines: locally on your machine or on the **Baidu AIStudio Community**.

* Local Experience:
```python
from paddlex import create_model
model = create_model("DLinear")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_csv("./output/")
```

* AIStudio Community Experience: Visit the [Official Time Series Forecasting Application](https://aistudio.baidu.com/community/app/105706/webUI?source=appCenter) to experience time series forecasting capabilities.
Note: Due to the tight correlation between time series data and scenarios, the official online experience models for time series tasks are tailored to specific scenarios and are not universal. Therefore, the online experience does not support using arbitrary files to test the official model solutions. However, after training your own model with scenario-specific data, you can select your trained model solution and use corresponding scenario data for online experience.

## 3. Choose a Model
PaddleX provides five end-to-end time series forecasting models:

| Model Name | MSE | MAE | Model Size (M) | Description |
|-|-|-|-|-|
| DLinear | 0.382 | 0.394 | 76k | A simple, efficient, and easy-to-use time series forecasting model |
| Nonstationary | 0.600 | 0.515 | 60.3M | Based on transformer architecture, optimized for long-term forecasting of non-stationary time series |
| PatchTST | 0.385 | 0.397 | 2.2M | A high-accuracy long-term forecasting model that balances local patterns and global dependencies |
| TiDE | 0.405 | 0.412 | 34.9M | A high-accuracy model suitable for handling multivariate, long-term time series forecasting problems |
| TimesNet | 0.417 | 0.431 | 5.2M | Through multi-period analysis, TimesNet is an adaptable and high-accuracy time series analysis model |

**Note: The above accuracy metrics are measured on the ETTH1 test dataset with an input sequence length of 96 and a prediction sequence length of 96 for all models except TiDE, which is 720.**

Based on your actual usage scenario, select an appropriate model for training. After training, evaluate the model weights within the pipeline and use them in practical scenarios.

## 4. Data Preparation and Validation
### 4.1 Data Preparation
To demonstrate the entire time series forecasting process, we will use the [Electricity](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) dataset for model training and validation. This dataset collects electricity consumption at a certain node from 2012 to 2014, with data collected every hour. Each data point consists of the current timestamp and corresponding electricity consumption. This dataset is commonly used to test and validate the performance of time series forecasting models.

In this tutorial, we will use this dataset to predict the electricity consumption for the next 96 hours. We have already converted this dataset into a standard data format, and you can obtain a sample dataset by running the following command. For an introduction to the data format, you can refer to the [Time Series Prediction Module Development Tutorial](docs_new/module_usage/tutorials/time_series_modules/time_series_forecasting.md).


You can use the following commands to download the demo dataset to a specified folder:

```
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_forecast/electricity.tar -P ./dataset
tar -xf ./dataset/electricity.tar -C ./dataset/
```

* **Data Considerations**
 * When annotating data for time series forecasting tasks, based on the collected real data, all data is arranged in chronological order. During training, the data is automatically divided into multiple time segments, where the historical time series data and the future sequences respectively represent the input data for training the model and its corresponding prediction targets, forming a set of training samples.
 * Handling Missing Values: To ensure data quality and integrity, missing values can be imputed based on expert knowledge or statistical methods.
 * Non-Repetitiveness: Ensure that data is collected in chronological order by row, with no duplication of timestamps.

### 4.2 Data Validation
Data Validation can be completed with just one command:

```
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/electricity
```

After executing the above command, PaddleX will validate the dataset, summarize its basic information, and print `Check dataset passed !` in the log if the command runs successfully. The validation result file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the current directory's `./output/check_dataset` directory, including example time series data.

```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 22880,
    "train_table": [
      [
        "date",
        "OT"
      ],
      [
        "2012-01-01 00:00:00",
        2162.0
      ],
      [
        "2012-01-01 01:00:00",
        2835.0
      ],
      [
        "2012-01-01 02:00:00",
        2764.0
      ],
      [
        "2012-01-01 03:00:00",
        2735.0
      ],
      [
        "2012-01-01 04:00:00",
        2721.0
      ],
      [
        "2012-01-01 05:00:00",
        2742.0
      ],
      [
        "2012-01-01 06:00:00",
        2716.0
      ],
      [
        "2012-01-01 07:00:00",
        2716.0
      ],
      [
        "2012-01-01 08:00:00",
        2680.0
      ],
      [
        "2012-01-01 09:00:00",
        2581.0
      ]
    ],
    "val_samples": 3424,
    "val_table": [
      [
        "date",
        "OT"
      ],
      [
        "2014-08-11 08:00:00",
        3528.0
      ],
      [
        "2014-08-11 09:00:00",
        3800.0
      ],
      [
        "2014-08-11 10:00:00",
        3889.0
      ],
      [
        "2014-08-11 11:00:00",
        4340.0
      ],
      [
        "2014-08-11 12:00:00",
        4321.0
      ],
      [
        "2014-08-11 13:00:00",
        4293.0
      ],
      [
        "2014-08-11 14:00:00",
        4393.0
      ],
      [
        "2014-08-11 15:00:00",
        4384.0
      ],
      [
        "2014-08-11 16:00:00",
        4495.0
      ],
      [
        "2014-08-11 17:00:00",
        4374.0
      ]
    ]
  },
  "analysis": {
    "histogram": ""
  },
  "dataset_path": "./dataset/electricity",
  "show_type": "csv",
  "dataset_type": "TSDataset"
} 
```

The above verification results have omitted some data parts. `check_pass` being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:

* `attributes.train_samples`: The number of samples in the training set of this dataset is 22880.
* `attributes.val_samples`: The number of samples in the validation set of this dataset is 3424.
* `attributes.train_table`: Sample data rows from the training set of this dataset.
* `attributes.val_table`: Sample data rows from the validation set of this dataset.


**Note**: Only data that passes the verification can be used for training and evaluation.

### 4.3 Dataset Format Conversion/Dataset Splitting (Optional)
If you need to convert the dataset format or re-split the dataset, you can modify the configuration file or append hyperparameters for settings. Refer to Section 4.1.3 in the [Time Series Prediction Module Development Tutorial](docs_new/module_usage/tutorials/time_series_modules/time_series_forecasting.md).

## 5. Model Training and Evaluation

### 5.1 Model Training
Before training, ensure that you have validated the dataset. To complete PaddleX model training, simply use the following command:

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
-o Global.mode=train \
-o Global.dataset_dir=./dataset/electricity \
-o Train.epochs_iters=5 \
-o Train.batch_size=16 \
-o Train.learning_rate=0.0001 \
-o Train.time_col=date \
-o Train.target_cols=OT \
-o Train.freq=1h \
-o Train.input_len=96 \
-o Train.predict_len=96
```

PaddleX supports modifying training hyperparameters and single-machine single-GPU training (time series models only support single-GPU training). Simply modify the configuration file or append command-line parameters.

Each model in PaddleX provides a configuration file for model development to set relevant parameters. Model training-related parameters can be set by modifying the `Train` fields in the configuration file. Some example parameter descriptions in the configuration file are as follows:

* `Global`:
  * `mode`: Mode, supporting dataset validation (`check_dataset`), model training (`train`), model evaluation (`evaluate`), and single instance testing (`predict`);
  * `device`: Training device, options include `cpu`, `gpu`, `xpu`, `npu`, `mlu`; check the [Model Support List](../../../support_list/models_list.md) for models supported on different devices.
* `Train`: Training hyperparameter settings;
  * `epochs_iters`: Number of training epochs;
  * `learning_rate`: Training learning rate;
  * `batch_size`: Training batch size per GPU;
  * `time_col`: Time column, set the column name of the time series dataset's time column based on your data;
  * `target_cols`: Target variable columns, set the column name(s) of the time series dataset's target variable(s) based on your data. Multiple columns can be separated by commas;
  * `freq`: Frequency, set the time frequency based on your data, e.g., 1min, 5min, 1h;
  * `input_len`: The length of historical time series input to the model; the input length should be considered comprehensively with the prediction length. Generally, the larger the setting, the more historical information can be referenced, and the higher the model accuracy.
  * `predict_len`: The length of the future sequence that the model is expected to predict; the prediction length should be considered comprehensively with the actual scenario. Generally, the larger the setting, the longer the future sequence you want to predict, and the lower the model accuracy.
  * `patience`: The parameter for the early stopping mechanism, indicating how many times the model's performance on the validation set can be continuously unimproved before stopping training; the larger the patience value, the longer the training time.
For more hyperparameter introductions, refer to [PaddleX Time Series Task Model Configuration File Parameter Description](docs_new/module_usage/instructions/config_parameters_time_series.md).

**Note**:

* The above parameters can be set by appending command-line parameters, e.g., specifying the mode as model training: `-o Global.mode=train`; specifying the first GPU for training: `-o Global.device=gpu:0`; setting the number of training epochs to 10: `-o Train.epochs_iters=10`.
* During model training, PaddleX automatically saves the model weight files, with the default being `output`. If you need to specify a save path, you can use the `-o Global.output` field in the configuration file.

### 5.2 Model Evaluation

After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation requires just one command:

```
    python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/electricity \
```

Similar to model training, model evaluation supports setting through modifying the configuration file or appending command-line parameters.

**Note**: When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command-line parameter, such as `-o Evaluate.weight_path=./output/best_model/model.pdparams`.

After completing the model evaluation, typically, the following outputs are generated:

Upon completion of model evaluation, `evaluate_result.json` will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully, and the model's evaluation metrics, including mse and mae.


### 5.3 Model Optimization

After learning about model training and evaluation, we can improve the model's accuracy by adjusting hyperparameters. By reasonably adjusting the number of training epochs, you can control the depth of model training, avoiding overfitting or underfitting. The setting of the learning rate is related to the speed and stability of model convergence. Therefore, when optimizing model performance, it is essential to carefully consider the values of these two parameters and adjust them flexibly based on actual conditions to achieve the best training effect.

Based on the method of controlled variables, we can adopt the following approach for hyperparameter tuning in time-series forecast:

It is recommended to follow the method of controlled variables when debugging parameters:
1. Initial Setup: Set the training epochs to 5, batch size to 16, and input length to 96.

2. Experiments with DLinear Model and Launch three experiments with learning rates: 0.0001, 0.001, 0.01.

3. Learning Rate Exploration: Experiment 2 yields the highest accuracy. Therefore, fix the learning rate at 0.001 and increase the training epochs to 30. Note: Due to the built-in early stopping mechanism for temporal tasks, training will automatically stop if the validation set accuracy does not improve after 10 patience epochs. To adjust the patience epochs, modify the `patience` hyperparameter in the advanced configuration.

4. Increasing Training Epochs and Input Length.

After increasing the training epochs, Experiment 4 achieves the highest accuracy. Next, increase the input length to 144 (using 144 hours of historical data to predict the next 96 hours), resulting in Experiment 5 with an accuracy of 0.188.

**Learning Rate Exploration Results**:

| Experiment ID | Epochs | Learning Rate | Batch Size | Input Length | Prediction Length | Training Environment | Validation MSE |
|---------------|--------|---------------|------------|--------------|-------------------|--------------------|----------------|
| Experiment 1  | 5      | 0.0001        | 16         | 96           | 96                | 1 GPU              | 0.314          |
| Experiment 2  | 5      | 0.001         | 16         | 96           | 96                | 1 GPU              | 0.302          |
| Experiment 3  | 5      | 0.01          | 16         | 96           | 96                | 1 GPU              | 0.320          |

**Increasing Training Epochs Results**:

| Experiment ID | Epochs | Learning Rate | Batch Size | Input Length | Prediction Length | Training Environment | Validation MSE |
|---------------|--------|---------------|------------|--------------|-------------------|--------------------|----------------|
| Experiment 2  | 5      | 0.001         | 16         | 96           | 96                | 1 GPU              | 0.302          |
| Experiment 4  | 30     | 0.001         | 16         | 96           | 96                | 1 GPU              | 0.301          |

**Increasing Input Length Results**:

| Experiment ID | Epochs | Learning Rate | Batch Size | Input Length | Prediction Length | Training Environment | Validation MSE |
|---------------|--------|---------------|------------|--------------|-------------------|--------------------|----------------|
| Experiment 4  | 30     | 0.001         | 16         | 96           | 96                | 1 GPU              | 0.301          |
| Experiment 5  | 30     | 0.001         | 16         | 144          | 96                | 1 GPU              | 0.188          |

## 6. Production Line Testing
Replace the model in the production line with the fine-tuned model and test using [this power test data](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_forecast/test.csv) for prediction:

```bash
python main.py -c paddlex/configs/ts_forecast/DLinear.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/inference" \
    -o Predict.input=https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/timeseries_forecast/test.csv
```
This will generate prediction results under `./output`, with the predictions for `test.csv` saved in `result.csv`.

Similar to model training and evaluation, follow these steps:
* Specify the path to the model's `.yaml` configuration file (here it's `PatchTST_ad.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Configuration](../../instructions/config_parameters_time_series_en.md).

## 7.Integration/Deployment

If the general-purpose time series forecast pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

1. If you need to apply the general-purpose time series forecast pipeline directly in your Python project, you can refer to the following sample code:
```
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="ts_forecast")
output = pipeline.predict("pre_ts.csv")
for res in output:
    res.print() 
    res.save_to_csv("./output/") 
```
For more parameters, please refer to the [Time Series forecast Pipeline Usage Tutorial](docs_new/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection_en.md)

2. Additionally, PaddleX's time series forecast pipeline also offers a service-oriented deployment method, detailed as follows:

Service-Oriented Deployment: This is a common deployment form in actual production environments. By encapsulating the inference functionality as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving service-oriented deployment of pipelines at low cost. For detailed instructions on service-oriented deployment, please refer to the [PPaddleX Service-Oriented Deployment Guide](docs_new/pipeline_deploy/service_deploy_en.md).
You can choose the appropriate method to deploy your model pipeline based on your needs, and proceed with subsequent AI application integration.