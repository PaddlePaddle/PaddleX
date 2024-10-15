[简体中文](config_parameters_common.md) | English

# PaddleX Common Model Configuration File Parameter Explanation

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
| convert.enable | bool | Whether to convert the dataset format; Image classification, pedestrian attribute recognition, vehicle attribute recognition, document orientation classification, object detection, pedestrian detection, vehicle detection, face detection, anomaly detection, text detection, seal text detection, text recognition, table recognition, image rectification, and layout area detection do not support data format conversion; Image multi-label classification supports COCO format conversion; Image feature, semantic segmentation, and instance segmentation support LabelMe format conversion; Object detection and small object detection support VOC and LabelMe format conversion; Formula recognition supports PKL format conversion; Time series prediction, time series anomaly detection, and time series classification support xlsx and xls format conversion | False |
| convert.src_dataset_type | str | The source dataset format to be converted | null |
| split.enable | bool | Whether to re-split the dataset | False |
| split.train_percent | int | Sets the percentage of the training set, an integer between 0-100, ensuring the sum with val_percent is 100; | null |
| split.val_percent | int | Sets the percentage of the validation set, an integer between 0-100, ensuring the sum with train_percent is 100; | null |
| split.gallery_percent | int | Sets the percentage of gallery samples in the validation set, an integer between 0-100, ensuring the sum with train_percent and query_percent is 100; This parameter is only used in the image feature module | null |
| split.query_percent | int | Sets the percentage of query samples in the validation set, an integer between 0-100, ensuring the sum with train_percent and gallery_percent is 100; This parameter is only used in the image feature module | null |

# Train
| Parameter Name | Data Type | Description | Default Value |
|-|-|-|-|
| num_classes | int | Number of classes in the dataset; If you need to train on a private dataset, you need to set this parameter; Image rectification, text detection, seal text detection, text recognition, formula recognition, table recognition, time series prediction, time series anomaly detection, and time series classification do not support this parameter | Number of classes specified in the YAML file |
| epochs_iters | int | Number of times the model repeats learning the training data | Number of iterations specified in the YAML file |
| batch_size | int | Training batch size | Training batch size specified in the YAML file |
| learning_rate | float | Initial learning rate | Initial learning rate specified in the YAML file |
| pretrain_weight_path | str | Pre-trained weight path | null |
| warmup_steps | int | Warm-up steps | Warm-up steps specified in the YAML file |
| resume_path | str | Model resume path after interruption | null |
| log_interval | int | Training log printing interval | Training log printing interval specified in the YAML file |
| eval_interval | int | Model evaluation interval | Model evaluation interval specified in the YAML file |
| save_interval | int | Model saving interval; not supported for anomaly detection, semantic segmentation, image rectification, time series forecasting, time series anomaly detection, and time series classification  | Model saving interval specified in the YAML file |

# Evaluate
| Parameter Name | Data Type | Description | Default Value |
|-|-|-|-|
| weight_path | str | Evaluation model path | Default local path from training output, when specified as None, indicates using official weights |
| log_interval | int | Evaluation log printing interval | Evaluation log printing interval specified in the YAML file |

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
