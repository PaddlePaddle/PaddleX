# Document Scene Information Extraction v3 Pipeline Usage Tutorial

## 1. Introduction to Document Scene Information Extraction v3 Pipeline
Document Scene Information Extraction v3 (PP-ChatOCRv3) is a unique intelligent analysis solution for documents and images developed by PaddlePaddle. It combines Large Language Models (LLM) and OCR technology to provide a one-stop solution for complex document information extraction challenges such as layout analysis, rare characters, multi-page PDFs, tables, and seal recognition. By integrating with ERNIE Bot, it fuses massive data and knowledge to achieve high accuracy and wide applicability.

![](https://github.com/user-attachments/assets/90cb740b-7741-4383-bc4c-663f9d042d02)

The Document Scene Information Extraction v3 pipeline includes modules for **Table Structure Recognition, Layout Region Detection, Text Detection, Text Recognition, Seal Text Detection, Text Image Rectification, and Document Image Orientation Classification**.

**If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, choose a model with faster inference speed. If you prioritize model storage size, choose a model with a smaller storage size.** Some benchmarks for these models are as follows:

<details>
   <summary> 👉Model List Details</summary>

**Table Structure Recognition Models**:

<table>
  <tr>
    <th>Model</th>
    <th>Accuracy (%)</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time</th>
    <th>Model Size (M)</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>SLANet</td>
    <td>59.52</td>
    <td>522.536</td>
    <td>1845.37</td>
    <td>6.9 M</td>
    <td>SLANet is a table structure recognition model developed by the PaddlePaddle Vision Team. It significantly improves the accuracy and inference speed of table structure recognition by adopting a CPU-friendly lightweight backbone network PP-LCNet, a high-low-level feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structural and positional information.</td>
  </tr>
  <tr>
    <td>SLANet_plus</td>
    <td>63.69</td>
    <td>522.536</td>
    <td>1845.37</td>
    <td>6.9 M</td>
    <td>SLANet_plus is an enhanced version of SLANet. Compared to SLANet, SLANet_plus significantly improves the recognition ability for wireless tables and complex tables, and reduces the model's sensitivity to the accuracy of table positioning. Even if the table positioning is offset, it can still perform accurate recognition.</td>
  </tr>
</table>

**Note: The above accuracy metrics are measured on PaddleX's internal self-built English table recognition dataset. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

**Layout Region Detection Models**:

|Model Name|mAP (%)|GPU Inference Time (ms)|CPU Inference Time|Model Size (M)|
|-|-|-|-|-|
|PicoDet_layout_1x|86.8|13.036|91.2634|7.4M|
|PicoDet-L_layout_3cls|89.3|15.7425|159.771|22.6 M|
|RT-DETR-H_layout_3cls|95.9|114.644|3832.62|470.1M|
|RT-DETR-H_layout_17cls|92.6|115.126|3827.25|470.2M|

**Note: The above accuracy metrics are evaluated on PaddleX's self-built layout region analysis dataset containing 10,000 images. All GPU inference times are based on an NVIDIA Tesla T4 machine with

</details>

## 2. Quick Start
PaddleX's pre-trained model pipelines can be quickly experienced. You can experience the effect of the Document Scene Information Extraction v3 pipeline online or locally using Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/182491/webUI) the effect of the Document Scene Information Extraction v3 pipeline, using the official demo images for recognition, for example:

![](https://github.com/user-attachments/assets/aa261b2b-b79c-4487-9323-dfcc43c3d581)

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to **fine-tune the models in the pipeline online**.

### 2.2 Local Experience
Before using the Document Scene Information Extraction v3 pipeline locally, please ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Guide](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/GvMbk70MZz/dF1VvOPZmZXXzn?t=mention&mt=doc&dt=doc).

A few lines of code are all you need to complete the quick inference of the pipeline. Using the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf), taking the General Document Scene Information Extraction v3 pipeline as an example:

```python
from paddlex import create_pipeline

predict = create_pipeline(pipeline="PP-ChatOCRv3-doc",
                          llm_name="ernie-3.5",
                          llm_params={"api_type":"qianfan","ak":"","sk":""})  ## Please fill in your ak and sk, or you cannot call the large model

visual_result, visual_inf = predict(["contract.pdf"])

for res in visual_result:
    res.save_to_img("./output")
    res.save_to_html('./output')
    res.save_to_xlsx('./output')

print(predict.chat("Party B, Phone Number"))
```
**Note**: Please first obtain your ak and sk from the [Baidu Qianfan Platform](https://qianfan.cloud.baidu.com/) and fill them in the designated places to properly call the large model.

After running, the output is as follows:

```
{'chat_res': {'Party B': 'Shareholding Test Co., Ltd.', 'Phone Number': '19331729920'}, 'prompt': ''}
```

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a Document Scene Information Extraction v3 pipeline object: Specific parameter descriptions are as follows:

| Parameter | Description | Default | Type |
|-|-|-|-|
| `pipeline` | Pipeline name or pipeline configuration file path. If it's a pipeline name, it must be supported by PaddleX. | None | str |
| `llm_name` | Large Language Model name | "ernie-3.5" | str |
| `llm_params` | API configuration | {} | dict |
| `device(kwargs)` | Running device (None for automatic adaptation) | None | str/None |

(2) Call the `predict` method of the Document Scene Information Extraction v3 pipeline object for inference prediction: The `predict` method parameter is `x`, used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Description |
|-|-|
| Python Var | Supports directly passing Python variables, such as numpy.ndarray representing image data; |
| str | Supports passing the path of the file to be predicted, such as the local path of an image file: /root/data/img.jpg; |
| str | Supports passing the URL of the file to be predicted, such as [example](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf); |
| str | Supports passing a local directory, which should contain files to be predicted, such as the local path: /root/data/; |
| dict | Supports passing a dictionary type, where the key needs to correspond to the specific pipeline, such as "img

When executing the above command, the default Pipeline configuration file is loaded. If you need to customize the configuration file, you can use the following command to obtain it:

```bash
paddlex --get_pipeline_config PP-ChatOCRv3-doc
```

After execution, the configuration file for the document scene information extraction v3 pipeline will be saved in the current path. If you wish to customize the save location, you can execute the following command (assuming the custom save location is `./my_path`):

```bash
paddlex --get_pipeline_config PP-ChatOCRv3-doc --config_save_path ./my_path
```
After obtaining the configuration file, you can customize the various configurations of the document scene information extraction v3 pipeline:

```yaml
Pipeline:
  layout_model: RT-DETR-H_layout_3cls
  table_model: SLANet_plus
  text_det_model: PP-OCRv4_server_det
  text_rec_model: PP-OCRv4_server_rec
  seal_text_det_model: PP-OCRv4_server_seal_det
  doc_image_ori_cls_model: null 
  doc_image_unwarp_model: null 
  llm_name: "ernie-3.5"
  llm_params: 
    api_type: qianfan
    ak: 
    sk: 
```

In the above configuration, you can modify the models loaded by each module of the pipeline, as well as the large language model used. Please refer to the module documentation for the list of supported models for each module, and the list of supported large language models includes: ernie-4.0, ernie-3.5, ernie-3.5-8k, ernie-lite, ernie-tiny-8k, ernie-speed, ernie-speed-128k, ernie-char-8k.

After making modifications, simply update the `pipeline` parameter value in the `create_pipeline` method to the path of your pipeline configuration file to apply the configuration.

For example, if your configuration file is saved at `./my_path/PP-ChatOCRv3-doc.yaml`, you would execute:

```python
from paddlex import create_pipeline

predict = create_pipeline(pipeline="./my_path/PP-ChatOCRv3-doc.yaml",
                          llm_name="ernie-3.5",
                          llm_params={"api_type":"qianfan","ak":"","sk":""} )  ## Please fill in your ak and sk, or you will not be able to call the large language model
                          
visual_result, visual_inf = predict(["contract.pdf"])

for res in visual_result:
    res.save_to_img("./output")
    res.save_to_html('./output')
    res.save_to_xlsx('./output')

print(predict.chat("Party B, phone number"))
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to directly apply the pipeline in your Python project, you can refer to the example code in [2.2 Local Experience](#22-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Deployment**: In actual production environments, many applications have stringent standards for the performance metrics (especially response speed) of deployment strategies to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance deployment procedures, please refer to the [PaddleX High-Performance Deployment Guide](../../../pipeline_deploy/high_performance_deploy.md).

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service-oriented deployment of pipelines. For detailed service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy.md).

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing functions on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_```markdown
## 4. Customization and Fine-tuning
If the default model weights provided by the General Document Scene Information Extraction v3 Pipeline do not meet your requirements in terms of accuracy or speed for your specific scenario, you can attempt to further **fine-tune** the existing models using **your own domain-specific or application-specific data** to enhance the recognition performance of the general table recognition pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the General Document Scene Information Extraction v3 Pipeline comprises four modules, unsatisfactory performance may stem from any of these modules (note that the text image rectification module does not support customization at this time).

You can analyze images with poor recognition results and follow the guidelines below for analysis and model fine-tuning:

* Incorrect table structure detection (e.g., row/column misidentification, cell position errors) may indicate deficiencies in the table structure recognition module. You need to refer to the **Customization** section in the [Table Structure Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md) and fine-tune the table structure recognition model using your private dataset.
* Misplaced layout elements (e.g., incorrect positioning of tables or seals) may suggest issues with the layout detection module. Consult the **Customization** section in the [Layout Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/layout_detection.md) and fine-tune the layout detection model with your private dataset.
* Frequent undetected text (i.e., text leakage) may indicate limitations in the text detection model. Refer to the **Customization** section in the [Text Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_detection.md) and fine-tune the text detection model using your private dataset.
* High text recognition errors (i.e., recognized text content does not match the actual text) suggest that the text recognition model requires improvement. Follow the **Customization** section in the [Text Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_recognition.md) to fine-tune the text recognition model.
* Frequent recognition errors in detected seal text indicate that the seal text detection model needs further refinement. Consult the **Customization** section in the [Seal Text Detection Module Development Tutorials](../../../module_usage/tutorials/ocr_modules/) to fine-tune the seal text detection model.
* Frequent misidentifications of document or certificate orientations with text regions suggest that the document image orientation classification model requires improvement. Refer to the **Customization** section in the [Document Image Orientation Classification Module Development Tutorial](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/y0mmii50BW/J5-rNhRB_xfhDZ?t=mention&mt=doc&dt=doc) to fine-tune the document image orientation classification model.

### 4.2 Model Deployment
After fine-tuning your models using your private dataset, you will obtain local model weights files.

To use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local paths of the default model weights with those of your fine-tuned models:

```yaml
......
Pipeline:
  layout_model: RT-DETR-H_layout_3cls  # Replace with the local path of your fine-tuned model
  table_model: SLANet_plus  # Replace with the local path of your fine-tuned model
  text_det_model: PP-OCRv4_server_det  # Replace with the local path of your fine-tuned model
  text_rec_model: PP-OCRv4_server_rec  # Replace with the local path of your fine-tuned model
  seal_text_det_model: PP-OCRv4_server_seal_det  # Replace with the local path of your fine-tuned model
  doc_image_ori_cls_model: null   # Replace with the local path of your fine-tuned model if applicable
  doc_image_unwarp_model: null   # Replace with the local path of your fine-tuned model if applicable
......
```

Subsequently, load the modified pipeline configuration file using the command-line interface or Python script as described in the local experience section.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Seamless switching between different hardware can be achieved by simply setting the `--device` parameter**.

For example, to perform inference using the Document Scene Information Extraction v3 Pipeline on an NVIDIA GPU```
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the script to `npu`:

```python
from paddlex import create_pipeline
predict = create_pipeline(pipeline="PP-ChatOCRv3-doc",
                            llm_name="ernie-3.5",
                            llm_params = {"api_type":"qianfan","ak":"","sk":""},  ## Please fill in your ak and sk, or you will not be able to call the large model
                            device = "npu:0") 
```
If you want to use the General Document Scene Information Extraction v3 Pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../installation/installation_other_devices.md).