# General OCR Pipeline Usage Tutorial

## 1. Introduction to OCR Pipeline
OCR (Optical Character Recognition) is a technology that converts text in images into editable text. It is widely used in document digitization, information extraction, and data processing. OCR can recognize printed text, handwritten text, and even certain types of fonts and symbols.

The General OCR Pipeline is designed to solve text recognition tasks, extracting text information from images and outputting it in text form. PP-OCRv4 is an end-to-end OCR system that achieves millisecond-level text content prediction on CPUs, reaching state-of-the-art (SOTA) performance in open-source projects for general scenarios. Based on this project, developers from academia, industry, and research have rapidly deployed various OCR applications across fields such as general use, manufacturing, finance, transportation, and more.

![](/tmp/images/pipelines/ocr/01.png)

**The General OCR Pipeline comprises a text detection module and a text recognition module**, each containing several models. The specific models to use can be selected based on the benchmark data provided below. **If you prioritize model accuracy, choose models with higher accuracy. If you prioritize inference speed, choose models with faster inference. If you prioritize model size, choose models with smaller storage requirements.**

<details>
   <summary> 👉Model List Details</summary>

<table>
  <tr>
    <th>Pipeline Module</th>
    <th>Specific Model</th>
    <th>Accuracy</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time</th>
    <th>Model Size (M)</th>
  </tr>
  <tr>
    <td rowspan="2">Text Detection</td>
    <td>PP-OCRv4_mobile_det</td>
    <td>77.79%</td>
    <td>2.719474</td>
    <td>79.1097</td>
    <td>15</td>
  </tr>
  <tr>
    <td>PP-OCRv4_server_det</td>
    <td>82.69%</td>
    <td>22.20346</td>
    <td>2662.158</td>
    <td>198</td>
  </tr>
  <tr>
    <td rowspan="5">Text Recognition</td>
    <td>PP-OCRv4_mobile_rec</td>
    <td>78.20%</td>
    <td>2.719474</td>
    <td>79.1097</td>
    <td>15</td>
  </tr>
  <tr>
    <td>PP-OCRv4_server_rec</td>
    <td>79.20%</td>
    <td>22.20346</td>
    <td>2662.158</td>
    <td>198</td>
  </tr>
  <tr>
    <td>ch_RepSVTR_rec</td>
    <td>65.07%</td>
    <td>-</td>
    <td>-</td>
    <td>22.1 M</td>
  </tr>
  <tr>
    <td>ch_SVTRv2_rec</td>
    <td>68.81%</td>
    <td> - </td>
    <td> - </td>
    <td>73.9 M</td>
  </tr>
</table>

**Note: The accuracy metric for text detection models is Hmean(%), and for text recognition models, it is Accuracy(%). All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

</details>

## 2. Quick Start
PaddleX provides pre-trained models for the OCR Pipeline, allowing you to quickly experience its effects. You can try the General OCR Pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience the General OCR Pipeline online](https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent) using the official demo images for recognition, for example:

![](/tmp/images/pipelines/ocr/02.png)

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. You can download the deployment package from the cloud or use the [local experience method in Section 2.2](#3-Development-and-Deployment). If not satisfied, you can also use your private data to **fine-tune the models in the pipeline online**.

### 2.2 Local Experience
> ❗ Before using the General OCR Pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Installation Guide](../../../installation/installation_en.md).

#### 2.2.1 Command Line Experience
* Experience the OCR Pipeline with a single command:

Experience the image anomaly detection pipeline with a single command，Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png), and replace `--input` with the local path to perform prediction.

```bash
paddlex --pipeline OCR --input general_ocr_002.png --device gpu:0
```
Parameter explanations:

```
--pipeline: The name of the pipeline, here it is OCR.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the second and third GPUs). You can also choose to use CPU (--device cpu).
```

When executing the above command, the default OCR Pipeline configuration file is loaded. If you need to customize the configuration file, you can use the following command to obtain it:

<details>
   <summary> 👉 Click to expand</summary>

```bash
paddlex --get_pipeline_config OCR
```

After execution, the OCR Pipeline configuration file will be saved in the current directory. If you wish to customize the save location, you can execute the following command (assuming the custom save location is `./my_path`):

```bash
paddlex --get_pipeline_config OCR --config_save_path ./my_path
```

After obtaining the Pipeline configuration file, replace `--pipeline` with the configuration file's save path to make the configuration file effective. For example, if the configuration file is saved as `./ocr.yaml`, simply execute:

```bash
paddlex --pipeline ./ocr.yaml --input general_ocr_002.png
```

Here, parameters such as `--model` and `--device` do not need to be specified, as they will use the parameters in the configuration file. If parameters are still specified, the specified parameters will take precedence.

</details>

#### 2.2.2 Integration via Python Script
* Quickly perform inference on the production line with just a few lines of code, taking the general OCR production line as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="ocr")

output = pipeline.predict("pre_image.jpg")
for batch in output:
    for item in batch:
        res = item['result']
        res.print()
        res.save_to_img("./output/")
        res.save_to_json("./output/")
```
> ❗ The results obtained from running the Python script are the same as those from the command line.

The Python script above executes the following steps:

（1）Instantiate the OCR production line object using `create_pipeline`: Specific parameter descriptions are as follows:

| Parameter | Description | Type | Default |
|-|-|-|-|
|`pipeline`| The name of the production line or the path to the production line configuration file. If it is the name of the production line, it must be supported by PaddleX. |`str`|None|
|`device`| The device for production line model inference. Supports: "gpu", "cpu". |`str`|`gpu`|
|`enable_hpi`| Whether to enable high-performance inference, only available if the production line supports it. |`bool`|`False`|

（2）Invoke the `predict` method of the OCR production line object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Parameter Description |
|---------------|-----------------------------------------------------------------------------------------------------------|
| Python Var    | Supports directly passing in Python variables, such as numpy.ndarray representing image data. |
| str         | Supports passing in the path of the file to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| str           | Supports passing in the URL of the file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png). |
| str           | Supports passing in a local directory, which should contain files to be predicted, such as the local path: `/root/data/`. |
| dict          | Supports passing in a dictionary type, where the key needs to correspond to a specific task, such as "img" for image classification tasks. The value of the dictionary supports the above types of data, for example: `{"img": "/root/data1"}`. |
| list          | Supports passing in a list, where the list elements need to be of the above types of data, such as `[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]`. |

（3）Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

| Method         | Description                     | Method Parameters |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | Prints results to the terminal  | `- format_json`: bool, whether to format the output content with json indentation, default is True;<br>`- indent`: int, json formatting setting, only valid when format_json is True, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, only valid when format_json is True, default is False; |
| save_to_json | Saves results as a json file   | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;<br>`- indent`: int, json formatting setting, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, default is False; |
| save_to_img  | Saves results as an image file | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type; |

If you have a configuration file, you can customize the configurations of the image anomaly detection pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/ocr.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/ocr.yaml")
output = pipeline.predict("general_ocr_002.png")
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

## 3. Development Integration/Deployment
If the general OCR pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the general OCR pipeline directly in your Python project, refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Deployment**: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end speedups. For detailed high-performance deployment procedures, refer to the [PaddleX High-Performance Deployment Guide](../../../pipeline_deploy/high_performance_deploy_en.md).

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service-oriented deployment of pipelines. For detailed service-oriented deployment procedures, refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy_en.md).

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing capabilities on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/lite_deploy_en.md).
You can choose the appropriate deployment method based on your needs to proceed with subsequent AI application integration.


## 4. Customization and Fine-tuning
If the default model weights provided by the general OCR pipeline do not meet your requirements for accuracy or speed in your specific scenario, you can try to further fine-tune the existing models using **your own domain-specific or application-specific data** to improve the recognition performance of the general OCR pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the general OCR pipeline consists of two modules (text detection and text recognition), unsatisfactory performance may stem from either module.

You can analyze images with poor recognition results. If you find that many texts are undetected (i.e., text miss detection), it may indicate that the text detection model needs improvement. You should refer to the [Customization](../../../module_usage/tutorials/ocr_modules/text_detection_en.md#customization) section in the [Text Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_detection_en.md) and use your private dataset to fine-tune the text detection model. If many recognition errors occur in detected texts (i.e., the recognized text content does not match the actual text content), it suggests that the text recognition model requires further refinement. You should refer to the [Customization](../../../module_usage/tutorials/ocr_modules/text_recognition_en.md#customization) section in the [Text Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_recognition_en.md) and fine-tune the text recognition model.

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain local model weights files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local paths of the fine-tuned model weights to the corresponding positions in the pipeline configuration file:

```python
......
Pipeline:
  det_model: PP-OCRv4_server_det  # Can be replaced with the local path of the fine-tuned text detection model
  det_device: "gpu"
  rec_model: PP-OCRv4_server_rec  # Can be replaced with the local path of the fine-tuned text recognition model
  rec_batch_size: 1
  rec_device: "gpu"
......
```

Then, refer to the command line method or Python script method in [2.2 Local Experience](#22-local-experience) to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPU, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modifying the `--device` parameter** allows seamless switching between different hardware.

For example, if you are using an NVIDIA GPU for OCR pipeline inference, the Python command would be:

```bash
paddlex --pipeline OCR --input general_ocr_002.png --device gpu:0
```
Now, if you want to switch the hardware to Ascend NPU, you only need to modify the `--device` in the Python command:

```bash
paddlex --pipeline OCR --input general_ocr_002.png --device npu:0
```

If you want to use the General OCR pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../installation/installation_other_devices_en.md).