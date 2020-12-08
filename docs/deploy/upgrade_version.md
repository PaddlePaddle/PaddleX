# Model version upgrade

Due to the continuous updating of PaddleX codes, models earlier than version 1.0.0 cannot be used directly for prediction deployment at the moment. Users need to follow the steps below to convert the model version. After the conversion, the model can be deployed in multiple terminals.

## Check the model version

There is a `model. yml` file in the folder where the model is stored. The `version` value in the last line of the file represents the version number of the model. If the version number is earlier than 1.0.0, you need to perform the version conversion. If the version number is later than or equal to 1.0.0, you do not need to perform version conversion.

## Version conversion

```
paddlex --export_inference --model_dir=/path/to/low_version_model --save_dir=/path/to/high_version_model
```
`--model_dir` is the path of the model with version number earlier than 1.0.0, which can be the model saved in the PaddleX training process, or the model exported as the inference format. `--save_dir` is a model converted to a later version, which can later be used for multi-terminal deployment.
