[简体中文](config_parameters_time_series.md) | English

# PaddleX Time Series Task Model Configuration File Parameters Explanation

# Global

| Parameter Name | Data Type | Description | Default Value |
|-|-|-|-|
| model | str | Specifies the model name | Model name specified in the YAML file |
| mode | str | Specifies the mode (check_dataset/train/evaluate/export/predict) | check_dataset |
| dataset_dir | str | Path to the dataset | Dataset path specified in the YAML file |
| device | str | Specifies the device to use | Device ID specified in the YAML file |
| output | str | Output path | "output" |

# CheckDataset

| Parameter Name | Data Type | Description | Default Value |
|-|-|-|-|
| convert.enable | bool | Whether to convert the dataset format; time series prediction, anomaly detection, and classification support data conversion from xlsx and xls formats | False |
| convert.src_dataset_type | str | The source dataset format to be converted | null |
| split.enable | bool | Whether to re-split the dataset | False |
| split.train_percent | int | Sets the percentage of the training set, an integer between 0-100, ensuring the sum with val_percent is 100; | null |
| split.val_percent | int | Sets the percentage of the validation set, an integer between 0-100, ensuring the sum with train_percent is 100; | null |


# Train
### Common Parameters for Time Series Tasks
| Parameter Name | Data Type | Description | Default Value |
|-|-|-|-|
| epochs_iters | int | The number of times the model repeats learning the training data | Number of iterations specified in the YAML file |
| batch_size | int | Batch size | Batch size specified in the YAML file |
| learning_rate | float | Initial learning rate | Initial learning rate specified in the YAML file |
| time_col | str | Time column, set the column name of the time series dataset's time column based on your data. | Time column specified in the YAML file |
| freq | str or int | Frequency, set the time frequency based on your data, e.g., 1min, 5min, 1h. | Frequency specified in the YAML file |
### Time Series Forecasting Parameters
| Parameter Name | Data Type | Description | Default Value |
|-|-|-|-|
| target_cols | str | Target variable column(s), set the column name(s) of the target variable(s) in the time series dataset, can be multiple, separated by commas | OT |
| input_len | int | For time series forecasting tasks, this parameter represents the length of historical time series input to the model; the input length should be considered in conjunction with the prediction length, generally, the larger the setting, the more historical information can be referenced, and the higher the model accuracy. | 96 |
| predict_len | int | The length of the future sequence that you want the model to predict; the prediction length should be considered in conjunction with the actual scenario, generally, the larger the setting, the longer the future sequence you want to predict, and the lower the model accuracy. | 96 |
| patience | int | Early stopping mechanism parameter, indicating how many times the model's performance on the validation set can be continuously unimproved before stopping training; a larger patience value generally results in longer training time. | 10 |
### Time Series Anomaly Detection
| Parameter Name | Data Type | Description | Default Value |
|-|-|-|-|
| input_len | int | For time series anomaly detection tasks, this parameter represents the length of the time series input to the model, which will slice the time series according to this length to predict whether there is an anomaly in this segment of the time series; the input length should be considered in conjunction with the actual scenario. For example, an input length of 96 indicates that you want to predict whether there are anomalies in 96 time points. | 96 |
| feature_cols | str | Feature variables indicating variables related to whether the device is abnormal, e.g., whether the device is abnormal may be related to the heat dissipation during its operation. Set the column name(s) of the feature variable(s) based on your data, can be multiple, separated by commas. | feature_0,feature_1 |
| label_col | str | Represents the number indicating whether a time series point is abnormal, with 1 for abnormal points and 0 for normal points. | label |

### Time Series Classification
| Parameter Name | Data Type | Description | Default Value |
|-|-|-|-|
| target_cols | str | Feature variable columns used for category discrimination. You need to set the column names of the target variables in the time series dataset based on your own data. It can be multiple, separated by commas. | dim_0,dim_1,dim_2 |
| freq | str or int | Frequency, which needs to be set based on your own data. Examples of time frequencies include: 1min, 5min, 1h. | 1 |
| group_id | str | A group ID represents a time series sample. Time series sequences with the same ID constitute a sample. Set the column name of the specified group ID based on your own data, e.g., group_id. | group_id |
| static_cov_cols | str | Represents the category number column of the time series. The labels of the same sample are the same. Set the column name of the category based on your own data, e.g., label. | label |

# Evaluate
| Parameter Name | Data Type | Description | Default Value |
|-|-|-|-|
| weight_path | str | Evaluation model path | Default local path from training output, when specified as None, indicates using official weights |

# Export
| Parameter Name | Data Type | Description | Default Value |
|-|-|-|-|
| weight_path | str | Dynamic graph weight path for exporting the model | Default local path from training output, when specified as None, indicates using official weights |

# Predict
| Parameter Name | Data Type | Description | Default Value |
|-|-|-|-|
| batch_size | int | Prediction batch size | The prediction batch size specified in the YAML file |
| model_dir | str | Path to the prediction model | The default local inference model path produced by training. When specified as None, it indicates the use of official weights |
| input | str | Path to the prediction input | The prediction input path specified in the YAML file |
