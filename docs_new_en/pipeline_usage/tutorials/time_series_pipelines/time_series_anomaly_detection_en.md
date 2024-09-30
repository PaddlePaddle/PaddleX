# Time Series Anomaly Detection Pipeline Tutorial

## 1. Introduction to the General Time Series Anomaly Detection Pipeline
Time series anomaly detection is a technique for identifying abnormal patterns or behaviors in time series data. It is widely applied in fields such as network security, equipment monitoring, and financial fraud detection. By analyzing normal trends and patterns in historical data, it discovers events that significantly deviate from expected behaviors, such as sudden spikes in network traffic or unusual transaction activities. Time series anomaly detection typically employs statistical methods or machine learning algorithms (e.g., Isolation Forest, LSTM), enabling automatic identification of anomalies in data. This technology provides real-time alerts for enterprises and organizations, helping them promptly address potential risks and issues. It plays a crucial role in ensuring system stability and security.

![](/tmp/images/pipelines/time_series/05.png)

**The General Time Series Anomaly Detection Pipeline includes a time series anomaly detection module. If you prioritize model accuracy, choose a model with higher precision. If you prioritize inference speed, select a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage footprint.**

<details>
   <summary> 👉Model List Details</summary>

| Model Name | Precision | Recall | F1-Score | Model Storage Size (M) |
|-|-|-|-|-|
| AutoEncoder_ad | 99.36 | 84.36 | 91.25 | 52K |
| DLinear_ad | 98.98 | 93.96 | 96.41 | 112K |
| Nonstationary_ad | 98.55 | 88.95 | 93.51 | 1.8M |
| PatchTST_ad | 98.78 | 90.70 | 94.57 | 320K |
| TimesNet_ad | 98.37 | 94.80 | 96.56 | 1.3M |

**Note: The above precision metrics are measured on the **[PSM](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar)** dataset. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

</details>

## 2. Quick Start
The pre-trained model pipelines provided by PaddleX allow for quick experience of their effects. You can experience the effects of the General Time Series Anomaly Detection Pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/105706/webUI?source=appCenter) the effects of the General Time Series Anomaly Detection Pipeline using the official demo for recognition, for example:

![](/tmp/images/pipelines/time_series/06.png)

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to **fine-tune the model within the pipeline online**.

**Note**: Due to the close relationship between time series data and scenarios, the official built-in models for online experience of time series tasks are only model solutions for a specific scenario and are not universal. They are not applicable to other scenarios. Therefore, the experience mode does not support using arbitrary files to experience the effects of the official model solutions. However, after training a model for your own scenario data, you can select your trained model solution and use data from the corresponding scenario for online experience.

### 2.2 Local Experience
Before using the General Time Series Anomaly Detection Pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation.md).

#### 2.2.1 Command Line Experience
A single command is all you need to quickly experience the effects of the time series anomaly detection pipeline:

Experience the image anomaly detection pipeline with a single command，Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv), and replace `--input` with the local path to perform prediction.

```bash
paddlex --pipeline ts_ad --input ts_ad.csv --device gpu:0
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
paddlex --get_pipeline_config ts_ad --config_save_path ./my_path
```

After obtaining the pipeline configuration file, you can replace `--pipeline` with the configuration file save path to make the configuration file take effect. For example, if the configuration file save path is `./ts_ad.yaml`, simply execute:

```bash
paddlex --pipeline ./ts_ad.yaml --input ts_ad.csv
```

Here, parameters such as `--model` and `--device` do not need to be specified, as they will use the parameters in the configuration file. If parameters are still specified, the specified parameters will take precedence.

</details>

After running, the result obtained is:

```json
{'ts_path': '/root/.paddlex/predict_input/ts_ad.csv', 'anomaly':            label
timestamp  
220226         0
220227         0
220228         0
220229         0
220230         0
...          ...
220317         1
220318         1
220319         1
220320         1
220321         0

[96 rows x 1 columns]}
```

#### 2.2.2 Python Script Integration
A few lines of code can complete the rapid inference of the pipeline. Taking the general time series anomaly detection pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="ts_ad")

output = pipeline.predict("ts_ad.csv")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_csv("./output/")  # Save the result in CSV format
    res.save_to_xlsx("./output/")  # Save the result in Excel format
```

The result obtained is the same as that of the command line method.

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
| str           | Supports passing in the URL of the file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv). |
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

For example, if your configuration file is saved at `./my_path/ts_ad.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/ts_ad.yaml")
output = pipeline.predict("ts_ad.csv")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_csv("./output/")  # Save results in CSV format
    res.save_to_xlsx("./output/")  # Save results in Excel format
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy in production, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Deployment**: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end speedups. For detailed high-performance deployment procedures, refer to the [PaddleX High-Performance Deployment Guide](../../../pipeline_deploy/high_performance_deploy.md).

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX enables users to achieve low-cost service-oriented deployment of pipelines. For detailed service-oriented deployment procedures, refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy.md).

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing capabilities on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/lite_deploy.md).
Choose the appropriate deployment method for your model pipeline based on your needs, and proceed with subsequent AI application integration.

## 4. Customization and Fine-tuning
If the default model weights provided by the General Time Series Anomaly Detection Pipeline do not meet your requirements for accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using **your own domain-specific or application-specific data** to improve the recognition performance of the pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the General Time Series Anomaly Detection Pipeline includes a time series anomaly detection module, if the performance of the pipeline does not meet expectations, you need to refer to the [Customization](../../../module_usage/tutorials/time_series_modules/time_series_anomaly_detection.md#Customization) section in the [Time Series Modules Development Tutorial](../../../module_usage/tutorials/time_series_modules/time_series_anomaly_detection.md) to fine-tune the time series anomaly detection model using your private dataset.

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain local model weights files.

To use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```python
......
Pipeline:
  model: DLinear_ad  # Can be modified to the local path of the fine-tuned model
  batch_size: 1
  device: "gpu:0"
......
```

Then, refer to the command line method or Python script method in the local experience section to load the modified pipeline configuration file.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modify the `--device` parameter** to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for inference of the time series anomaly detection pipeline, the Python command is:

```bash
paddlex --pipeline ts_ad --input ts_ad.csv --device gpu:0
``````
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the Python command to `npu`:

```bash
paddlex --pipeline ts_ad --input ts_ad.csv --device npu:0
```
If you want to use the General Time-Series Anomaly Detection Pipeline on more diverse hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../installation/installation_other_devices.md).