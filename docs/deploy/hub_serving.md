# Lightweight service-oriented deployments
## Introduction
With `PaddleHub-Serving`, `PaddleX`'s `Inference Model` can be rapidly deployed to provide the online prediction ability.

For more information on `PaddleHub-Serving`, refer to [PaddleHub-Serving]. (https://github.com/PaddlePaddle/PaddleHub/blob/develop/docs/tutorial/serving.md)

**Note: To deploy in this way, you need to make sure that the version of PaddleHub in the Python environment is later than 1.8.0. You can run `pip show paddlehub` to confirm the version information.**


Next, follow the steps to convert an image classification model [MobileNetV3_small_ssld](https://bj.bcebos.com/paddlex/models/mobilenetv3_small_ssld_imagenet.tar.gz) into a pre-training model for `PaddleHub`, and use `PaddleHub-Serving` to implement one-key deployment.


## Model Deployment

### 1 Preparation for deployment model
The format of the deployment model is three files `__model__`, `__params__` and `model.yml` contained in the directory. For the format of the files, refer to the [deployment model export file]. (./export_model.md)

### 2 Model conversion
First, convert `PaddleX`'s `Inference Model` to `PaddleHub`'s pre-training model. Run `hub convert` to implement the one-key conversion. The command is described as follows:

```shell
$ hub convert --model_dir XXXX \
              --module_name XXXX \
              --module_version XXXX \
              --output_dir XXXX
```
**Parameters**:

|Parameters|Use|
|-|-| |--model_dir/-m|`PaddleX Inference Model` directory|
|--module_name/-n| Generate the name of the pre-training model|
|--module_version/-v| Version of generating the pre-training model. The default value is `1.0.0`|
|`--output_dir/-o| Directory of storing the generated pre-trained model. The default name is `{module_name}_{timestamp}`|

Therefore, you only need to run a command to complete the conversion of the pre-training model.

```shell
hub convert --model_dir mobilenetv3_small_ssld_imagenet_hub --module_name mobilenetv3_small_ssld_imagenet_hub
```

After the conversion is complete the prompted information is as follows:
```shell
$ The converted module is stored in `MobileNetV3_small_ssld_hub_1596077881.868501`.
```
After the prompt, a pre-training model of `PaddleHub` in the output directory is obtained.

### 3 Model installation
In the model conversion step, a pre-trained model compression package in the `.tar. gz` format. Before deploying, you need to install it locally by running the command `hub install`. The description is as follows:
```shell
$ hub install ${MODULE}
```
${MODULE} is the path of the pre-training model file to be installed.

Run `hub install`.
```shell
hub install MobileNetV3_small_ssld_hub_1596077881.868501/mobilenetv3_small_ssld_imagenet_hub.tar.gz
```
After a successful installation, the following message is displayed:
```shell
$ Successfully installed mobilenetv3_small_ssld_imagenet_hub
```

### 4 Model deployment
You can run `hub serving` to deploy the model through one-key. The description is as follows:
```shell
$ hub serving start --modules/-m [Module1==Version1, Module2==Version2, ...] \
                    --port/-p XXXX
                    --config/-c XXXX
```

**Parameters**:

|Parameters|Use|
|-|-|
|--modules/-m|PaddleHub Serving pre-installed models, listed as multiple Module==Version key-value pairs<br>*. `When Version is not specified, the default selection is the latest version`*|
|--port/-p|Service port, default is 8866|
|--config/-c|Configure the model using configuration files|

Therefore, only one line of code is needed to deploy the model.

```shell
$ hub serving start -m mobilenetv3_small_ssld_imagenet_hub
```
After the model is loaded, this pre-training model is now deployed on the machine.

You can perform more configurations by using the configuration file. The format of the configuration file is as follows:
```json
{
  "modules_info": {
    "mobilenetv3_small_ssld_imagenet_hub": {
      "init_args": {
        "version": "1.0.0"
      },
      "predict_args": {
        "batch_size": 1,
        "use_gpu": false
      }
    }
  },
  "port": 8866
}

```
|Parameters|Use|
|-|-|
|modules_info| PaddleHub Serving pre-installed models, listed as a dictionary list, key is the model name. where: <br>`init_args` is the parameter to be entered when the model is loaded, equivalent to `paddlehub. Module(**init_args)`<br>`predict_args` is the parameter entered when the model is predicted. For example, in `mobilenetv3_small_ssld_imagenet_hub`, it is equivalent to `mobilenetv3_small_ssld_imagenet_hub.batch_predict(**predict_args)`
|port| service port, default is 8866|

### 5 Test
While the model is installed in the second step, a client request example is generated and stored in the model installation directory. By default, it is `${HUB_HOME}/.paddlehub/modules`. In this example, the client example `serving_client_demo.py` can be found in `~/.paddlehub/modules/mobilenetv3_small_ssld_imagenet_hub`. The codes are as follows: 

```python
# coding: utf8
import requests
import json
import cv2
import base64


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


if __name__ == '__main__':
      # Get the base64 encoding format of the image 
       img1 = cv2_to_base64(cv2.imread("IMAGE_PATH1"))
       img2 = cv2_to_base64(cv2.imread("IMAGE_PATH2"))
       data = {'images':[img1, img2]} 
       # Specify content-type
       headers = {"Content-type":"application/json"} 
       # Send an HTTP request url = "http://127.0.0.1:8866/predict/mobilenetv3_small_ssld_imagenet_hub"
       r = requests.post(url=url, headers=headers, data=json.dumps(data)) 

       # Print the prediction result
        print(r.json()["results"])
```
The following test images are used.

![](../train/images/test.jpg)

After changing `IMAGE_PATH1` in the code to the path of the image where you want to make the prediction, run the following command line:
```python
python ~/. paddlehub/module/MobileNetV3_small_ssld_hub/serving_client_demo.py
```
The following prediction results can be received:
```shell
[[{'category':'envelope', 'category_id':549, 'score':0.2141510397195816}]]
````

The one-key deployment of the `PaddleX` model is completed.
