# General Table Recognition Pipeline Usage Tutorial

## 1. Introduction to the General Table Recognition Pipeline
Table recognition is a technology that automatically identifies and extracts table content and its structure from documents or images. It is widely used in data entry, information retrieval, and document analysis. By leveraging computer vision and machine learning algorithms, table recognition can convert complex table information into editable formats, facilitating further data processing and analysis for users.

![](/tmp/images/pipelines/table_recognition/01.png)

**The General Table Recognition Pipeline comprises modules for table structure recognition, layout analysis, text detection, and text recognition.**

**If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model size, choose a model with a smaller storage footprint.**

<details>
   <summary> 👉Model List Details</summary>

**Table Recognition Module Models**:

<table>
  <tr>
    <th>Model</th>
    <th>Accuracy (%)</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time (ms)</th>
    <th>Model Size (M)</th>
    <th>Description</th>
  </tr>
  <tr>
     <td>SLANet</td>
    <td>59.52</td>
    <td>522.536</td>
    <td>1845.37</td>
    <td>6.9 M</td>
    <td rowspan="1">SLANet is a table structure recognition model developed by Baidu PaddlePaddle Vision Team. The model significantly improves the accuracy and inference speed of table structure recognition by adopting a CPU-friendly lightweight backbone network PP-LCNet, a high-low-level feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structural and positional information.</td>
  </tr>
   </tr>
   <tr>
    <td>SLANet_plus</td>
    <td>63.69</td>
    <td>522.536</td>
    <td>1845.37</td>
    <td>6.9 M</td>
        <td rowspan="1">
SLANet_plus is an enhanced version of SLANet, a table structure recognition model developed by Baidu PaddlePaddle's Vision Team. Compared to SLANet, SLANet_plus significantly improves its recognition capabilities for wireless and complex tables, while reducing the model's sensitivity to the accuracy of table localization. Even when there are offsets in table localization, it can still perform relatively accurate recognition.
</td>
  </tr>
</table>

**Note: The above accuracy metrics are measured on PaddleX's internal self-built English table recognition dataset. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

**Layout Analysis Module Models**:

|Model Name|mAP (%)|GPU Inference Time (ms)|CPU Inference Time|Model Size (M)|
|-|-|-|-|-|
|PicoDet_layout_1x|86.8|13.036|91.2634|7.4M|
|PicoDet-L_layout_3cls|89.3|15.7425|159.771|22.6 M|
|RT-DETR-H_layout_3cls|95.9|114.644|3832.62|470.1M|
|RT-DETR-H_layout_17cls|92.6|115.126|3827.25|470.2M|

**Note: The above accuracy metrics are evaluated on PaddleX's self-built layout analysis dataset containing 10,000 images. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

**Text Detection Module Models**:

|Model Name|Detection Hmean (%)|GPU Inference Time (ms)|CPU Inference Time|Model Size (M)|
|-|-|-|-|-|
|PP-OCRv4_mobile_det|77.79|10.6923|120.177|4.2 M|
|PP-OCRv4_server_det|82.69|83.3501|2434.01|100.1M```markdown

</details>

## 2. Quick Start
PaddleX's pre-trained model pipelines allow for quick experience of their effects. You can experience the effects of the General Image Classification pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/91661/webUI) the effects of the General Table Recognition pipeline by using the demo images provided by the official. For example:

![](/tmp/images/pipelines/table_recognition/02.png)

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to **fine-tune the models in the pipeline online**.

### 2.2 Local Experience
Before using the General Table Recognition pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Guide](../../../installation/installation.md).

### 2.1 Command Line Experience
Experience the effects of the table recognition pipeline with a single command:

Experience the image anomaly detection pipeline with a single command，Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg), and replace `--input` with the local path to perform prediction.

```bash
paddlex --pipeline table_recognition --input table_recognition.jpg --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it's the table recognition pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the 1st and 2nd GPUs). CPU can also be selected (--device cpu).
```

When executing the above command, the default table recognition pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details>
   <summary> 👉Click to expand</summary>

```bash
paddlex --get_pipeline_config table_recognition
```

After execution, the table recognition pipeline configuration file will be saved in the current directory. If you wish to customize the save location, you can execute the following command (assuming the custom save location is `./my_path`):

```bash
paddlex --get_pipeline_config table_recognition --config_save_path ./my_path
```

After obtaining the pipeline configuration file, replace `--pipeline` with the configuration file save path to make the configuration file take effect. For example, if the configuration file save path is `./table_recognition.yaml`, simply execute:

```bash
paddlex --pipeline ./table_recognition.yaml --input table_recognition.jpg
```

Here, parameters like `--model` and `--device` do not need to be specified, as they will use the parameters in the configuration file. If they are still specified, the specified parameters will take precedence.

</details>

After running, the result is:

![](/tmp/images/pipelines/table_recognition/03.png)

The visualized image is saved in the `output` directory by default, and you can customize it with `--save_path`.

### 2.2 Python Script Integration
A few lines of code are all you need to quickly perform inference with the pipeline. Taking the General Table Recognition pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="table_recognition")

output = pipeline.predict("table_recognition.jpg")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_csv("./output/")  # Save the results in CSV format
    res.save_to_xlsx("./output/")  # Save the results in Excel format
```
The results are the same as those obtained through the command line.

In the above Python script, the following steps are executed:

（1）Instantiate the  production line object using `create_pipeline`: Specific parameter descriptions are as follows:

| Parameter | Description | Type | Default |
|-|-|-|-|
|`pipeline`| The name of the production line or the path to the production line configuration file. If it is the name of the production line, it must be supported by PaddleX. |`str`|None|
|`device`| The device for production line model inference. Supports: "gpu", "cpu". |`str`|`gpu`|
|`enable_hpi`| Whether to enable high-performance inference, only available if the production line supports it. |`bool`|`False`|

（2）Invoke the `predict` method of the  production line object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Parameter Description |
|---------------|-----------------------------------------------------------------------------------------------------------|
| Python Var    | Supports directly passing in Python variables, such as numpy.ndarray representing image data. |
| str         | Supports passing in the path of the file to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| str           | Supports passing in the URL of the file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg). |
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

For example, if your configuration file is saved at `./my_path/table_recognition.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/table_recognition.yaml")
output = pipeline.predict("table_recognition.jpg")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_csv("./output/")  # Save results in CSV format
    res.save_to_xlsx("./output/")  # Save results in Excel format
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy in production, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Deployment**: In actual production environments, many applications have stringent standards for deployment strategy performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins that aim to deeply optimize model inference and pre/post-processing for significant end-to-end process acceleration. For detailed high-performance deployment procedures, refer to the [PaddleX High-Performance Deployment Guide](../../../pipeline_deploy/high_performance_deploy.md).

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service-oriented deployment of pipelines. For detailed service-oriented deployment procedures, refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy.md).

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing capabilities directly on user devices, allowing devices to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/lite_deploy.md).
Choose the appropriate deployment method for your model pipeline based on your needs, and proceed with subsequent AI application integration.

## 4. Customization and Fine-tuning
If the default model weights provided by the general table recognition pipeline do not meet your requirements for accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using **your own domain-specific or application-specific data** to improve the recognition performance of the general table recognition pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the general table recognition pipeline consists of four modules, unsatisfactory performance may stem from any of these modules.

Analyze images with poor recognition results and follow the rules below for analysis and model fine-tuning:

* If the detected table structure is incorrect (e.g., row and column recognition errors, incorrect cell positions), the table structure recognition module may be insufficient. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md#customization) section in the [Table Structure Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md) and use your private dataset to fine-tune the table structure recognition model.
* If the table area is incorrectly located within the overall layout, the layout detection module may be insufficient. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/layout_detection.md#customization) section in the [Layout Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/layout_detection.md) and use your private dataset to fine-tune the layout detection model.
* If many texts are undetected (i.e., text miss detection), the text detection model may be insufficient. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/text_recognition.md#customization) section in the [Text Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_recognition.md) and use your private dataset to fine-tune the text detection model.
* If many detected texts contain recognition errors (i.e., the recognized text content does not match the actual text content), the text recognition model requires further improvement. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md#customization) section```markdown
### 4.2 Model Application
After fine-tuning your model with a private dataset, you will obtain local model weights files.

To use the fine-tuned model weights, simply modify the production line configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the configuration file:

```python
......
 Pipeline:
  layout_model: PicoDet_layout_1x  # Can be modified to the local path of the fine-tuned model
  table_model: SLANet  # Can be modified to the local path of the fine-tuned model
  text_det_model: PP-OCRv4_mobile_det  # Can be modified to the local path of the fine-tuned model
  text_rec_model: PP-OCRv4_mobile_rec  # Can be modified to the local path of the fine-tuned model
  layout_batch_size: 1
  text_rec_batch_size: 1
  table_batch_size: 1
  device: "gpu:0"
......
```
Then, refer to the command line or Python script method in the local experience to load the modified production line configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPU, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modify the `--device` parameter** to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for table recognition pipeline inference, the Python command is:

```bash
paddlex --pipeline table_recognition --input table_recognition.jpg --device gpu:0
```
At this time, if you want to switch the hardware to Ascend NPU, simply modify `--device` in the Python command to npu:

```bash
paddlex --pipeline table_recognition --input table_recognition.jpg --device npu:0
```
If you want to use the general table recognition pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../installation/installation_other_devices.md).