# Image Anomaly Detection Pipeline Tutorial

## 1. Introduction to Image Anomaly Detection Pipeline
Image anomaly detection is an image processing technique that identifies unusual or non-conforming patterns within images through analysis. It is widely applied in industrial quality inspection, medical image analysis, and security monitoring. By leveraging machine learning and deep learning algorithms, image anomaly detection can automatically recognize potential defects, anomalies, or abnormal behaviors in images, enabling us to promptly identify issues and take corresponding actions. The image anomaly detection system is designed to automatically detect and mark anomalies in images, enhancing work efficiency and accuracy.

![](/tmp/images/pipelines/image_anomaly_detection/01.png)

**The image anomaly detection pipeline includes an unsupervised anomaly detection module, with the following model benchmarks**:

| Model Name | Avg (%) | Model Size (M) |
|-|-|-|
| STFPM | 96.2 | 21.5 M |

**Note: The above accuracy metrics are the average anomaly scores on the **[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)** validation set. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

## 2. Quick Start
PaddleX provides pre-trained models for the anomaly detection pipeline, allowing for quick experience of its effects. You can use the command line or Python to experience the image anomaly detection pipeline locally.

Before using the image anomaly detection pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation_en.md).

### 2.1 Command Line Experience
Experience the image anomaly detection pipeline with a single command，Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png), and replace `--input` with the local path to perform prediction.

```bash
paddlex --pipeline anomaly_detection --input uad_grid.png --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it's the image anomaly detection pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the second and third GPUs). CPU can also be selected (--device cpu).
```

When executing the above command, the default image anomaly detection pipeline configuration file is loaded. If you need to customize the configuration file, you can run the following command to obtain it:

<details>
   <summary> 👉Click to expand</summary>

```bash
paddlex --get_pipeline_config anomaly_detection
```

After execution, the image anomaly detection pipeline configuration file will be saved in the current directory. If you wish to customize the save location, you can execute the following command (assuming the custom save location is `./my_path`):

```bash
paddlex --get_pipeline_config anomaly_detection --config_save_path ./my_path
```

After obtaining the pipeline configuration file, replace `--pipeline` with the configuration file save path to make the configuration file take effect. For example, if the configuration file save path is `./anomaly_detection.yaml`, simply execute:

```bash
paddlex --pipeline ./anomaly_detection.yaml --input uad_grid.png
```

Here, parameters such as `--model` and `--device` do not need to be specified, as they will use the parameters in the configuration file. If parameters are still specified, the specified parameters will take precedence.

</details>

After running, the result is:

```
{'img_path': '/root/.paddlex/predict_input/uad_grid.png'}
```
![](/tmp/images/pipelines/image_anomaly_detection/02.png)

The visualized image is saved in the `output` directory by default, which can be customized using `--save_path`.

### 2.2 Python Script Integration
A few lines of code are sufficient for quick inference using the pipeline. Taking the image anomaly detection pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="anomaly_detection")

output = pipeline.predict("uad_grid.png")
for res in output:
    res.print() 
    res.save_to_img("./output/") 
    res.save_to_json("./output/") 
```

The results obtained are the same as those from the command line approach.

In the above Python script, the following steps are executed:

（1）Instantiate the `create_pipeline` to create a pipeline object: Specific parameter descriptions are as follows:

| Parameter | Description | Type | Default Value |
|-|-|-|-|
|`pipeline`| The name of the pipeline or the path to the pipeline configuration file. If it's a pipeline name, it must be a pipeline supported by PaddleX. |`str`| None |
|`device`| The device for pipeline model inference. Supports: "gpu", "cpu". |`str`|`gpu`|
|`enable_hpi`| Whether to enable high-performance inference, only available if the pipeline supports it. |`bool`|`False`|

（2）Invoke the `predict` method of the pipeline object for inference prediction: The `predict` method takes `x` as its parameter, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Description |
|---------------|-----------------------------------------------------------------------------------------------------------|
| Python Var    | Supports directly passing Python variables, such as numpy.ndarray representing image data. |
| str         | Supports passing the path to the data file to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| str           | Supports passing the URL of the data file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png). |
| str           | Supports passing a local directory, which should contain the data files to be predicted, such as the local path: `/root/data/`. |
| dict          | Supports passing a dictionary type, where the key needs to correspond to the specific task, e.g., "img" for image classification tasks, and the value of the dictionary supports the above data types, for example: `{"img": "/root/data1"}`. |
| list          | Supports passing a list, where the list elements need to be of the above data types, such as `[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]`. |

（3）Obtain prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

| Method         | Description                     | Method Parameters |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | Prints results to the terminal  | `- format_json`: bool, whether to format the output content with json indentation, default is True;<br>`- indent`: int, json formatting setting, only valid when format_json is True, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, only valid when format_json is True, default is False; |
| save_to_json | Saves results as a json file   | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;<br>`- indent`: int, json formatting setting, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, default is False; |
| save_to_img  | Saves results as an image file | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type; |

If you have a configuration file, you can customize the configurations of the image anomaly detection pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/anomaly_detection.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/anomaly_detection.yaml")
output = pipeline.predict("uad_grid.png")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_img("./output/")  # Save the visualized image of the result
    res.save_to_json("./output/")  # Save the structured output of prediction
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy in production, you can proceed directly with development integration/deployment.

If you need to apply the pipeline directly in your Python project, refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Deployment**: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance deployment procedures, refer to the [PaddleX High-Performance Deployment Guide](../../../pipeline_deploy/high_performance_deploy_en.md).

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service-oriented deployment of pipelines. For detailed service-oriented deployment procedures, refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy_en.md).

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing functions on user devices themselves, enabling devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/lite_deploy_en.md).
You can choose the appropriate deployment method for your model pipeline based on your needs and proceed with subsequent AI application integration.

## 4. Customization and Fine-tuning
If the default model weights provided by the image anomaly detection pipeline do not meet your requirements for accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using **your own domain-specific or application-specific data** to improve the recognition performance of the image anomaly detection pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the image anomaly detection pipeline includes an unsupervised image anomaly detection module, if the performance of the pipeline does not meet expectations, you need to refer to the [Customization](../../../module_usage/tutorials/cv_modules/anomaly_detection_en.md) section in the [Unsupervised Anomaly Detection Module Tutorial](../../../module_usage/tutorials/cv_modules/anomaly_detection_en.md) and use your private dataset to fine-tune the image anomaly detection model.

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain local model weights files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding position in the pipeline configuration file:

```python
......
Pipeline:
  model: STFPM   # Can be modified to the local path of the fine-tuned model
  batch_size: 1
  device: "gpu:0"
......
```
Then, refer to the command line or Python script methods in the local experience section to load the modified pipeline configuration file.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modify the `--device` parameter** to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for inference with the image anomaly detection pipeline, the Python command is:

```bash
paddlex --pipeline anomaly_detection --input uad_grid.png --device gpu:0
``````
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the Python command to `npu`:

```bash
paddlex --pipeline anomaly_detection --input uad_grid.png --device npu:0
```
If you want to use the image anomaly detection pipeline on more types of hardware, please refer to the [PaddleX Multi-device Usage Guide](../../../installation/installation_other_devices_en.md).
