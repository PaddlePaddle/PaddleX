# Inference Deployment
This document describes how to use the Python Paddle-Lite for PaddleX model prediction deployment on Raspberry. You can install the Python Paddle-Lite prediction library according to the following command. If the installation fails, download the whl file to install [Paddle-Lite_2.6.0_python](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.6.0/armlinux_python_installer.zip). For more versions, see the [Paddle-Lite Release Note.](https://github.com/PaddlePaddle/Paddle-Lite/releases)
```
python -m pip install paddlelite
```
Before deployment, you need to convert PaddleX model to Paddle-Lite nb model. For details, see [Paddle-Lite model conversion](./export_nb_model.md). 
**Note**: If the Python Prediction Library 2.6.0 is used, download version 2.6.0 of the opt conversion tool to convert the model.



## Pre-conditions
* Python 3.6+
* Paddle-Lite_python 2.6.0+

Make sure that the above basic software is installed on your system. **All the following examples are in the working directory `/root/projects/`**.

## Inference deployment
Run the demo.py file in the /root/projects/PaddleX/deploy/raspberry/python directory to perform the prediction. The command parameters are described as follows:

| Parameters | Description |
|  ----  | ----  |
| --model_dir | The path of the .xml file generated in the model conversion. Make sure that the three files generated in the model conversion are in the same path. |
| --img | The path of the image file to be predicted |
| --image_list | .txt file of storing image paths by line |
| --cfg_file | .yml configuration file of PaddleX model. |
| --thread_num | Number of predicted threads. The default value is 1. |

**Note**: The Python API of the Paddle-lite doesn't support the input of int64 data yet; therefore, Raspberry doesn't support the deployment of YoloV3 in python. If it is required, use C++ codes to deploy YoloV3.

### Example
`Example 1`: 
test images `/path/to/test_img.jpeg`

```
cd /root/projects/python

python demo. py --model_dir /path/to/openvino_model --img /path/to/test_img.jpeg --cfg_file /path/to/PadlleX_model.yml --thread_num 4
```

`Example 2`:

Prediction of multiple images: `/path/to/image_list.txt`. The format of the image_list.txt content is as follows:

```
/path/to/images/test_img1.jpeg /path/to/images/test_img2.jpeg . . . /path/to/images/test_imgn.jpeg
```

```
/path/to/images/test_img1.jpeg
/path/to/images/test_img2.jpeg
...
/path/to/images/test_imgn.jpeg
```

```
cd /root/projects/python  

python demo.py --model_dir /path/to/models/openvino_model --image_list /root/projects/images_list.txt --cfg_file=/path/to/PadlleX_model.yml --thread_num 4 
```
