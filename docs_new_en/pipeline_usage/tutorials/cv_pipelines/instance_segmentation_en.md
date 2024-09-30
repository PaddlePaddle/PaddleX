# General Instance Segmentation Pipeline Tutorial

## 1. Introduction to the General Instance Segmentation Pipeline
Instance segmentation is a computer vision task that not only identifies the object categories in an image but also distinguishes the pixels of different instances within the same category, enabling precise segmentation of each object. Instance segmentation can separately label each car, person, or animal in an image, ensuring they are independently processed at the pixel level. For example, in a street scene image containing multiple cars and pedestrians, instance segmentation can clearly separate the contours of each car and person, forming multiple independent region labels. This technology is widely used in autonomous driving, video surveillance, and robotic vision, often relying on deep learning models (such as Mask R-CNN) to achieve efficient pixel classification and instance differentiation through Convolutional Neural Networks (CNNs), providing powerful support for understanding complex scenes.

![](/tmp/images/pipelines/instance_segmentation/01.png)

**The General Instance Segmentation Pipeline includes a** **Object Detection** **module. If you prioritize model precision, choose a model with higher precision. If you prioritize inference speed, choose a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.**

<details>
   <summary> 👉Model List Details</summary>

|Model Name|Mask AP|GPU Inference Time (ms)|CPU Inference Time|Model Size (M)|
|-|-|-|-|-|
|Mask-RT-DETR-H|50.6|132.693|4896.17|449.9|
|Mask-RT-DETR-L|45.7|46.5059|2575.92|113.6|
|Mask-RT-DETR-M|42.7|36.8329|-|66.6 M|
|Mask-RT-DETR-S|41.0|33.5007|-|51.8 M|
|Mask-RT-DETR-X|47.5|75.755|3358.04|237.5 M|
|Cascade-MaskRCNN-ResNet50-FPN|36.3|-|-|254.8|
|Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN|39.1|-|-|254.7|
|MaskRCNN-ResNet50-FPN|35.6|-|-|157.5 M|
|MaskRCNN-ResNet50-vd-FPN|36.4|-|-|157.5 M|
|MaskRCNN-ResNet50-vd-SSLDv2-FPN|38.2|-|-|157.2 M|
|MaskRCNN-ResNet50|32.8|-|-|127.8 M|
|MaskRCNN-ResNet101-FPN|36.6|-|-|225.4 M|
|MaskRCNN-ResNet101-vd-FPN|38.1|-|-|225.1 M|
|MaskRCNN-ResNeXt101-vd-FPN|39.5|-|-|370.0 M|
|PP-YOLOE_seg-S|32.5|-|-|31.5 M|

**Note: The above accuracy metrics are Mask AP(0.5:0.95) on the **[COCO2017](https://cocodataset.org/#home)** validation set. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

</details>

## 2. Quick Start
The pre-trained model pipelines provided by PaddleX allow for quick experience of the effects. You can experience the effects of the General Instance Segmentation Pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/100063/webUI) the effects of the General Instance Segmentation Pipeline using the demo images provided by the official. For example:

![](/tmp/images/pipelines/instance_segmentation/02.png)

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to **fine-tune the model within the pipeline**.

### 2.2 Local Experience
Before using the General Image Classification Pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation_en.md).

#### 2.2.1 Command Line Experience
A single command is all you need to quickly experience the image classification pipeline:

```bash
paddlex --pipeline instance_segmentation --input  https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png --device gpu:0
```

Parameter Description:

```
--pipeline: The name of the pipeline, here it refers to the object detection pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 indicates using the first GPU, gpu:1,2 indicates using the second and third GPUs), or you can choose to use CPU (--device cpu).
```

When executing the above Python script, the default instance segmentation pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details>
   <summary> 👉Click to expand</summary>

```
paddlex --get_pipeline_config instance_segmentation
```

After execution, the instance segmentation pipeline configuration file will be saved in the current path. If you wish to customize the save location, you can execute the following command (assuming the custom save location is `./my_path`):

```
paddlex --get_pipeline_config instance_segmentation --config_save_path ./my_path
```

After obtaining the pipeline configuration file, you can replace `--pipeline` with the configuration file save path to make the configuration file take effect. For example, if the configuration file save path is `./instance_segmentation.yaml`, simply execute:

```
paddlex --pipeline ./instance_segmentation.yaml --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png
```

Where `--model`, `--device`, and other parameters do not need to be specified, and the parameters in the configuration file will be used. If parameters are still specified, the specified parameters will take precedence.

</details>

After running, the result is:

```
{'img_path': '/root/.paddlex/predict_input/general_instance_segmentation_004.png', 'boxes': [{'cls_id': 0, 'label': 'person', 'score': 0.8698326945304871, 'coordinate': [339, 0, 639, 575]}, {'cls_id': 0, 'label': 'person', 'score': 0.8571141362190247, 'coordinate': [0, 0, 195, 575]}, {'cls_id': 0, 'label': 'person', 'score': 0.8202633857727051, 'coordinate': [88, 113, 401, 574]}, {'cls_id': 0, 'label': 'person', 'score': 0.7108577489852905, 'coordinate': [522, 21, 767, 574]}, {'cls_id': 27, 'label': 'tie', 'score': 0.554280698299408, 'coordinate': [247, 311, 355, 574]}]}
```

![](/tmp/images/pipelines/instance_segmentation/03.png)

The visualization image is saved in the `output` directory by default, and you can customize it through `--save_path`.

#### 2.2.2 Integration via Python Script
A few lines of code can complete the quick inference of the pipeline. Taking the general instance segmentation pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="instance_segmentation")

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png")
for res in output:
    res.print() # Print the structured output of the prediction
    res.save_to_img("./output/") # Save the visualization image of the result
    res.save_to_json("./output/") # Save the structured output of the prediction
```
The results obtained are the same as those obtained through the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: The specific parameter descriptions are as follows:

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
|`pipeline` | The name of the pipeline or the path to the pipeline configuration file. If it is the name of the pipeline, it must be a pipeline supported by PaddleX. | `str` | None |
|`device` | The device for pipeline model inference. Supports: "gpu", "cpu". | `str` | "gpu" |
|`enable_hpi` | Whether to enable high-performance inference, which is only available when the pipeline supports it. | `bool` | `False` |

(2) Call the `predict` method of the image classification pipeline object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Description |
|----------------|-------------|
| Python Var | Supports directly passing Python variables, such as numpy.ndarray representing image data. |
| `str` | Supports passing the path of the file to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| `str` | Supports passing the URL of the file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png). |
| `str` | Supports passing a local directory, which should contain files to be predicted, such as the local path: `/root/data/`. |
| `dict` | Supports passing a dictionary type, where the key needs to correspond to the specific task, such as "img" for the image classification task, and the value of the dictionary supports the above data types, e.g., `{"img": "/root/data1"}`. |
| `list` | Supports passing a list, where the list elements need to be the above data types, such as `[numpy.ndarray, numpy.ndarray]`, `["/root/data/img1.jpg", "/root/data/img2.jpg"]`, `["/root/data1", "/root/data2"]`, `[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]`. |

3）Obtain prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

| Method         | Description                     | Method Parameters |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | Prints results to the terminal  | `- format_json`: bool, whether to format the output content with json indentation, default is True;<br>`- indent`: int, json formatting setting, only valid when format_json is True, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, only valid when format_json is True, default is False; |
| save_to_json | Saves results as a json file   | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;<br>`- indent`: int, json formatting setting, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, default is False; |
| save_to_img  | Saves results as an image file | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type; |

If you have a configuration file, you can customize the configurations of the image anomaly detection pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/instance_segmentation.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/instance_segmentation.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png")
for res in output:
    res.print() # Print the structured output of prediction
    res.save_to_img("./output/") # Save the visualized image of the result
    res.save_to_json("./output/") # Save the structured output of prediction
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, you can refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Deployment**: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins that aim to deeply optimize model inference and pre/post-processing for significant speedups in the end-to-end process. For detailed high-performance deployment procedures, please refer to the [PaddleX High-Performance Deployment Guide](../../../pipeline_deploy/high_performance_deploy_en.md).

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service-oriented deployment of pipelines. For detailed service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy_en.md).

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing functions on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/lite_deploy_en.md).
You can choose the appropriate deployment method for your model pipeline based on your needs and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the general instance segmentation pipeline do not meet your requirements for accuracy or speed in your scenario, you can try to further **fine-tune** the existing model using **data specific to your domain or application scenario** to improve the recognition effect of the general instance segmentation pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the general instance segmentation pipeline includes an instance segmentation module, if the performance of the pipeline does not meet expectations, you need to refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/instance_segmentation_en.md#四二次开发) section in the [Instance Segmentation Module Development Tutorial](../../../module_usage/tutorials/cv_modules/instance_segmentation_en.md).

### 4.2 Model Application
After you complete fine-tuning training using your private dataset, you will obtain local model weight files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```python
......
Pipeline:
  model: Mask-RT-DETR-S  # Can be modified to the local path of the fine-tuned model
  device: "gpu"
  batch_size: 1
......
```

Then, refer to the command line method or Python script method in the local experience to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modify the `--device` parameter** to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for instance segmentation pipeline inference, the Python command is:

```bash
paddlex --pipeline instance_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png --device gpu:0
``````
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the Python command to `npu`:

```bash
paddlex --pipeline instance_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png --device npu:0
```

If you want to use the General Instance Segmentation Pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../installation/installation_other_devices_en.md).