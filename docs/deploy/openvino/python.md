# Inference deployment
The document describes OpenVINO-based prediction deployment in python. Before deployment, you need to convert the paddle model to OpenVINO's Inference Engine. For details, see [Model Conversion](docs/deploy/openvino/export_openvino_model.md). Currently, the classification, detection, and segmentation model of PadlleX is supported on CPU hardware; the classification model of PaddleX is supported on VPU.

## Pre-conditions
* Python 3.6+
* OpenVINO 2021.1

**Note**: For OpenVINO installation, refer to [OpenVINO](https://docs.openvinotoolkit.org/latest/index.html) description.


Make sure that the above basic software is installed on your system. **All the following examples are in the working directory `/root/projects/`**.

## Inference deployment
Running the demo.py file in the /root/projects/PaddleX/deploy/openvino/python directory can make predictions with the following command parameters.

| Parameters | Description |
|  ----  | ----  |
| --model_dir | The path of the .xml file generated in the model conversion. Make sure that the three files generated in the model conversion are in the same path. |
| --img | The path of the image file to be predicted |
| --image_list | .txt file of storing image paths by line |
| --device | Running platform. The default value is "CPU" |
| --cfg_file | .yml configuration file of PaddleX model. |

### Example
`Example 1`: test images: `/path/to/test_img.jpeg`

```
cd /root/projects/python  

python demo.py --model_dir /path/to/openvino_model --img /path/to/test_img.jpeg --cfg_file /path/to/PadlleX_model.yml
```

`Example 2`:

Prediction of multiple images: `/path/to/image_list.txt`. The format of the image_list.txt content is as follows:

```
/path/to/images/test_img1.jpeg
/path/to/images/test_img2.jpeg
...
/path/to/images/test_imgn.jpeg
```

```
cd /root/projects/python  

python demo.py --model_dir /path/to/models/openvino_model --image_list /root/projects/images_list.txt --cfg_file=/path/to/PadlleX_model.yml
```
