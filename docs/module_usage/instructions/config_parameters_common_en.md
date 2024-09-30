[简体中文](config_parameters_common.md) | English


# PaddleX General Model Configuration File Parameter Explanation

# Global
| Parameter Name | Data Type | Description | Default Value | Required/Optional |
|-|-|-|-|-|
| model | str | Specifies the model name | - | Required |
| mode | str | Specifies the mode (check_dataset/train/evaluate/export/predict) | - | Required |
| dataset_dir | str | Path to the dataset | - | Required |
| device | str | Specifies the device to use | - | Required |
| output | str | Output path | "output" | Optional |

# CheckDataset
| Parameter Name | Data Type | Description | Default Value | Required/Optional |
|-|-|-|-|-|
| convert.enable | bool | Whether to enable dataset format conversion | False | Optional |
| convert.src_dataset_type | str | Source dataset format to convert | null | Optional |
| split.enable | bool | Whether to re-split the dataset | False | Optional |
| split.train_percent | int | Sets the percentage of the training set, an integer between 0-100, which needs to sum up to 100 with val_percent | null | Optional |
| split.val_percent | int | Sets the percentage of the validation set, an integer between 0-100, which needs to sum up to 100 with train_percent | null | Optional |
| split.gallery_percent | int | Sets the percentage of gallery samples in the validation set, an integer between 0-100, which needs to sum up to 100 with train_percent and query_percent; this parameter is only used in the image feature module | null | Optional |
| split.query_percent | int | Sets the percentage of query samples in the validation set, an integer between 0-100, which needs to sum up to 100 with train_percent and gallery_percent; this parameter is only used in the image feature module | null | Optional |

# Train
| Parameter Name | Data Type | Description | Default Value | Required/Optional |
|-|-|-|-|-|
| num_classes | int | Number of classes in the dataset | - | Required |
| epochs_iters | int | Number of times the model learns from the training data | - | Required |
| batch_size | int | Training batch size | - | Required |
| learning_rate | float | Initial learning rate | - | Required |
| pretrain_weight_path | str | Pre-trained weight path | null | Optional |
| warmup_steps | int | Warmup steps | - | Required |
| resume_path | str | Path to resume the model after interruption | null | Optional |
| log_interval | int | Interval for printing training logs | - | Required |
| eval_interval | int | Interval for model evaluation | - | Required |
| save_interval | int | Interval for saving the model | - | Required |

# Evaluate
| Parameter Name | Data Type | Description | Default Value | Required/Optional |
|-|-|-|-|-|
| weight_path | str | Path to the model for evaluation | - | Required |
| log_interval | int | Interval for printing evaluation logs | - | Required |

# Export
| Parameter Name | Data Type | Description | Default Value | Required/Optional |
|-|-|-|-|-|
| weight_path | str | Path to the dynamic graph weights of the model to export | Official dynamic graph weights URL for each model | Required |

# Predict
| Parameter Name | Data Type | Description | Default Value | Required/Optional |
|-|-|-|-|-|
| batch_size | int | Prediction batch size | - | Required |
| model_dir | str | Path to the prediction model | Official PaddleX model weights | Optional |
| input | str | Path to the prediction input | - | Required |