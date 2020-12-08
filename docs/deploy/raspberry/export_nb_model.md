# Paddle-Lite model conversion
The PaddleX model is converted to Paddle-Lite nb model. The model conversion mainly includes PaddleX to inference model and inference model to Paddle-Lite nb model.
### Step1: Export the inference model
Before converting PaddleX model to Paddle-Lite model, you need to export the PaddleX model to inference format first. The exported model includes three file names: __model__, __params__, and model.yml. For more details, refer to the [Inference Model Export](../export_model.md).
### Step2：Export the Paddle-Lite model
The Paddle-Lite model needs to be converted through the Paddle-Lite opt tool. Download and decompress the [model optimization tool opt (2.6.1-linux)](https://bj.bcebos.com/paddlex/deploy/Rasoberry/opt.zip) and run it on Linux:
```bash
./opt --model_file=<model_path> \
      --param_file=<param_path> \
      --valid_targets=arm \
      --optimize_out_type=naive_buffer \
      --optimize_out=model_output_name
```
| Parameters | Description |
|  ----  | ----  |
| --model_file | Export the network structure file contained in the inference model: the path where `__model__` is located. |
| --param_file | Export the parameter file contained in the inference model: the path where `__params__` is located. |
| --valid_targets | Specify the model executable backend. Here it is specified as `arm`. |
| --optimize_out_type | Output model type. Currently supports two types: protobuf and naive_buffer, where naive_buffer is a more lightweight serialization/deserialization. Here it is specified as `naive_buffer`. |


If the python Paddle-Lite is installed, it can also be converted in the following method:
```
./paddle_lite_opt --model_file=<model_path> \
      --param_file=<param_path> \
      --valid_targets=arm \
      --optimize_out_type=naive_buffer \
      --optimize_out=model_output_name
```

For more detailed instructions and parameter meanings, refer to [Using the Opt Conversion Model](https://paddle-lite.readthedocs.io/zh/latest/user_guides/opt/opt_bin.html). For more opt pre-compiling versions, see the [Paddle-Lite Release Note](https://github.com/PaddlePaddle/Paddle-Lite/releases).

**Note**: The opt version needs to be consistent with the prediction library version. If you want to use the 2.6.0 prediction library, download the opt conversion model of version 2.6.0 from the Release Note.