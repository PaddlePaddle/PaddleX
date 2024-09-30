# General Image Multi-Label Classification Pipeline Tutorial

## 1. Introduction to the General Image Multi-Label Classification Pipeline
Image multi-label classification is a technique that assigns multiple relevant categories to a single image simultaneously, widely used in image annotation, content recommendation, and social media analysis. It can identify multiple objects or features present in an image, for example, an image containing both "dog" and "outdoor" labels. By leveraging deep learning models, image multi-label classification automatically extracts image features and performs accurate classification, providing users with more comprehensive information. This technology is of great significance in applications such as intelligent search engines and automatic content generation.

![](/tmp/images/pipelines/image_multi_label_classification/01.png)

**The General Image Multi-Label Classification Pipeline includes a module for image multi-label classification. If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, choose a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.**

<details>
   <summary> 👉Model List Details</summary>

|Model Name|mAP (%)|Model Storage Size (M)|
|-|-|-|
|CLIP_vit_base_patch16_448_ML|89.15|-|-|325.6|
|PP-HGNetV2-B0_ML|80.98|39.6|
|PP-HGNetV2-B4_ML|87.96|88.5|
|PP-HGNetV2-B6_ML|91.25|286.5|
|PP-LCNet_x1_0_ML|77.96|29.4|
|ResNet50_ML|83.50|108.9|

**Note: The above accuracy metrics are mAP for the multi-label classification task on **[COCO2017](https://cocodataset.org/#home)**. The GPU inference time for all models is based on an NVIDIA Tesla T4 machine with FP32 precision. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**
</details>

## 2. Quick Start
PaddleX supports experiencing the effects of the General Image Multi-Label Classification Pipeline locally using command line or Python.

Before using the General Image Multi-Label Classification Pipeline locally, please ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation_en.md).

### 2.1 Experience via Command Line
Experience the effects of the image multi-label classification pipeline with a single command:

```bash
paddlex --pipeline multi_label_image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it is the image multi-label classification pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the second and third GPUs). You can also choose to use CPU (--device cpu).
```

When executing the above command, the default configuration file for the image multi-label classification pipeline is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details>
   <summary> 👉Click to Expand</summary>

```bash
paddlex --get_pipeline_config multi_label_image_classification
```
After execution, the configuration file for the image multi-label classification pipeline will be saved in the current path. If you wish to customize the save location, you can execute the following command (assuming the custom save location is `./my_path`):

```bash
paddlex --get_pipeline_config multi_label_image_classification --config_save_path ./my_path
```

After obtaining the pipeline configuration file, replace `--pipeline` with the saved path of the configuration file to make it effective. For example, if the configuration file is saved at `./multi_label_image_classification.yaml`, simply execute:

```bash
paddlex --pipeline ./multi_label_image_classification.yaml --input https://paddle-model-ecology.bj
```

Where `--model`, `--device`, and other parameters are not specified, the parameters in the configuration file will be used. If parameters are specified, the specified parameters will take precedence.

</details>

After running, the result obtained is:

```
{'img_path': '/root/.paddlex/predict_input/general_image_classification_001.jpg', 'class_ids': [21, 0, 30, 24], 'scores': [0.99257, 0.70596, 0.63001, 0.57852], 'label_names': ['bear', 'person', 'skis', 'backpack']}
```
![](/tmp/images/pipelines/image_multi_label_classification/02.png)

The visualization image is saved in the `output` directory by default, and you can also customize it through `--save_path`.

### 2.2 Integration via Python Script
A few lines of code can complete the rapid inference of the pipeline. Taking the general image multi-label classification pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="multi_label_image_classification")

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_img("./output/")  # Save the result visualization image
    res.save_to_json("./output/")  # Save the structured output of the prediction
```
The result obtained is the same as that of the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: Specific parameter descriptions are as follows:

| Parameter | Description | Type | Default Value |
|-----------|-------------|------|---------------|
|`pipeline` | The name of the pipeline or the path of the pipeline configuration file. If it is the name of the pipeline, it must be a pipeline supported by PaddleX. | `str` | None |
|`device` | The device for pipeline model inference. Supports: "gpu", "cpu". | `str` | "gpu" |
|`enable_hpi` | Whether to enable high-performance inference, which is only available when the pipeline supports it. | `bool` | `False` |

(2) Call the `predict` method of the multi-label classification pipeline object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Description |
|----------------|-------------|
| Python Var | Supports directly passing in Python variables, such as numpy.ndarray representing image data. |
| str | Supports passing in the file path of the data file to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| str | Supports passing in the URL of the data file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg). |
| str | Supports passing in a local directory, which should contain the data files to be predicted, such as the local path: `/root/data/`. |
| dict | Supports passing in a dictionary type, where the key of the dictionary needs to correspond to the specific task, such as "img" for image classification tasks, and the value of the dictionary supports the above data types, for example: `{"img": "/root/data1"}`. |
| list | Supports passing in a list, where the list elements need to be the above data types, such as `[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]`. |

（3）Obtain prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

| Method         | Description                     | Method Parameters |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | Prints results to the terminal  | `- format_json`: bool, whether to format the output content with json indentation, default is True;<br>`- indent`: int, json formatting setting, only valid when format_json is True, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, only valid when format_json is True, default is False; |
| save_to_json | Saves results as a json file   | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;<br>`- indent`: int, json formatting setting, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, default is False; |
| save_to_img  | Saves results as an image file | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type; |

If you have a configuration file, you can customize the configurations of the image anomaly detection pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/multi_label_image_classification.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/multi_label_image_classification.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_img("./output/")  # Save the visualization image of the result
    res.save_to_json("./output/")  # Save the structured output of prediction
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Deployment**: In actual production environments, many applications have strict standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins that aim to deeply optimize model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance deployment procedures, refer to the [PaddleX High-Performance Deployment Guide](../../../pipeline_deploy/high_performance_deploy_en.md).

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service-oriented deployment of pipelines. For detailed service-oriented deployment procedures, refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy_en.md).

📱 **Edge Deployment**: Edge deployment is a way to place computing and data processing functions on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/lite_deploy_en.md).
You can choose the appropriate deployment method for your model pipeline based on your needs and proceed with subsequent AI application integration.

## 4. Customization and Fine-tuning
If the default model weights provided by the general image multi-label classification pipeline do not meet your requirements in terms of accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using **your own domain-specific or application-specific data** to improve the recognition performance of the general image multi-label classification pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the general image multi-label classification pipeline includes an image multi-label classification module, if the performance of the pipeline does not meet expectations, you need to refer to the [Customization](../../../module_usage/tutorials/cv_modules/ml_classification_en.md#Customization) section in the [Image Multi-Label Classification Module Development Tutorial](../../../module_usage/tutorials/cv_modules/ml_classification_en.md) to fine-tune the image multi-label classification model using your private dataset.

### 4.2 Model Application
After you have completed fine-tuning training using your private dataset, you will obtain local model weights files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```python
......
Pipeline:
  model: PP-LCNet_x1_0_ML   # Can be modified to the local path of the fine-tuned model
  batch_size: 1
  device: "gpu:0"
......
```
Then, refer to the command line method or Python script method in the local experience section to load the modified pipeline configuration file.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modify the `--device` parameter** to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for inference of the image multi-label classification pipeline, the Python command is:

```bash
paddlex --pipeline multi_label_image_classification --input https://paddle-model-ecology.bj.bcebos.com/padd
```

At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the Python command to `npu`:
```
paddlex --pipeline multi_label_image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device npu:0
```
If you want to use the General Image Multi-label Classification Pipeline on more diverse hardware, please refer to the [PaddleX Multi-device Usage Guide](../../../installation/installation_other_devices_en.md).