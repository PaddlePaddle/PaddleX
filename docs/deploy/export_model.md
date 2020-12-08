# Deployment model export

When deploying models on the server, you need to export the model saved during training to a model in the inference format. The exported inference format model includes three files: `__model__`, `__params__` and `model.yml`, which represent the network structure, model weights, and model configuration file (including data preprocessing parameters) respectively.

> **Check your model folder**. If it contains `model.pdparams`, `model.pdmodel` and `model.yml` files, you need to export the model in the following process.` ` `

After installing PaddleX, run the following command to export the model in a command line terminal. You can directly download the DUDU sorting model to test the process: [xiaoduxiong_epoch_12 . tar.gz](https://bj.bcebos.com/paddlex/models/xiaoduxiong_epoch_12.tar.gz).

```
paddlex --export_inference --model_dir=. /xiaoduxiong_epoch_12 --save_dir=. /inference_model
```

| Parameters | Description |
| ---- | ---- |
| --export_inference | Whether or not to export the model to the inference format for deployment ,it is specified as True. |
| --model_dir | The path of the model to be exported |
| --save_dir | Path for storing the exported model |
| --fixed_input_shape | Fixed the input size of the exported model. The default value is None. |


When using TensorRT for prediction, you need to fix the input size of the model. You can prepare the input size [w,h] by using `--fixed_input_shape`.

**Note**:
- Keep the fixed input size for the classification model to be the same as the input size for training;
- In the setting of [w,h], w and h are separated by a comma, and no other characters such as spaces are allowed.

```
paddlex --export_inference --model_dir=. /xiaoduxiong_epoch_12 --save_dir=. /inference_model --fixed_input_shape=[640,960]
```
