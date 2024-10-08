[简体中文](document_scene_information_extraction.md) | English

# PP-ChatOCRv3-doc Pipeline Usage Tutorial

## 1. Introduction to PP-ChatOCRv3-doc Pipeline
PP-ChatOCRv3-doc is a unique intelligent analysis solution for documents and images developed by PaddlePaddle. It combines Large Language Models (LLM) and OCR technology to provide a one-stop solution for complex document information extraction challenges such as layout analysis, rare characters, multi-page PDFs, tables, and seal recognition. By integrating with ERNIE Bot, it fuses massive data and knowledge to achieve high accuracy and wide applicability.

![](https://github.com/user-attachments/assets/90cb740b-7741-4383-bc4c-663f9d042d02)

The **PP-ChatOCRv3-doc** pipeline includes modules for **Table Structure Recognition, Layout Region Detection, Text Detection, Text Recognition, Seal Text Detection, Text Image Rectification, and Document Image Orientation Classification**.

**If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, choose a model with faster inference speed. If you prioritize model storage size, choose a model with a smaller storage size.** Some benchmarks for these models are as follows:

<details>
   <summary> 👉Model List Details</summary>

**Table Structure Recognition Module Models:**

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
    <td>SLANet is a table structure recognition model independently developed by Baidu's PaddlePaddle vision team. The model significantly enhances the accuracy and inference speed for table structure recognition by utilizing the CPU-friendly lightweight backbone network PP-LCNet, the CSP-PAN high-low layer feature fusion module, and the SLA Head feature decoding module that aligns structural and positional information.</td>
  </tr>
  <tr>
    <td>SLANet_plus</td>
    <td>63.69</td>
    <td>522.536</td>
    <td>1845.37</td>
    <td>6.9 M</td>
    <td>SLANet_plus is an enhanced version of the SLANet model independently developed by Baidu's PaddlePaddle vision team. Compared to SLANet, SLANet_plus significantly improves the recognition capability for unbounded and complex tables, and reduces the model's sensitivity to table localization accuracy. Even if the table localization is offset, it can still accurately recognize the table structure.</td>
  </tr>
</table>

**Note: The above accuracy metrics are measured on PaddleX's internal English table recognition dataset. All models' GPU inference times are based on NVIDIA Tesla T4 machines, with accuracy type FP32. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads and accuracy type FP32.**

**Layout Area Detection Module Models:**

|Model Name|mAP (%)|GPU Inference Time (ms)|CPU Inference Time|Model Size (M)|
|-|-|-|-|-|
|PicoDet_layout_1x|86.8|13.036|91.2634|7.4 M|
|PicoDet-L_layout_3cls|89.3|15.7425|159.771|22.6 M|
|RT-DETR-H_layout_3cls|95.9|114.644|3832.62|470.1 M|
|RT-DETR-H_layout_17cls|92.6|115.126|3827.25|470.2 M|

**Note: The above accuracy metrics are evaluated on PaddleX's internal layout area analysis dataset, which includes 10,000 images. All models' GPU inference times are based on NVIDIA Tesla T4 machines, with accuracy type FP32. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads and accuracy type FP32.**

**Text Detection Module Models:**

|Model Name|Detection Hmean (%)|GPU Inference Time (ms)|CPU Inference Time|Model Size (M)|
|-|-|-|-|-|
|PP-OCRv4_mobile_det|77.79|10.6923|120.177|4.2 M|
|PP-OCRv4_server_det|82.69|83.3501|2434.01|100.1 M|

**Note: The above accuracy metrics are evaluated on PaddleOCR's internal Chinese dataset, covering multiple scenarios such as street views, web images, documents, and handwriting, with 500 images for detection. All models' GPU inference times are based on NVIDIA Tesla T4 machines, with accuracy type FP32. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads and accuracy type FP32.**

**Text Recognition Module Models:**

|Model Name|Avg Accuracy (%)|GPU Inference Time (ms)|CPU Inference Time|Model Size (M)|
|-|-|-|-|-|
|PP-OCRv4_mobile_rec|78.20|7.95018|46.7868|10.6 M|
|PP-OCRv4_server_rec|79.20|7.19439|140.179|71.2 M|

**Note: The above accuracy metrics are evaluated on PaddleOCR's internal Chinese dataset, covering various scenarios such as street views, web images, documents, and handwriting, with 11,000 images for text recognition. All models' GPU inference times are based on NVIDIA Tesla T4 machines, with accuracy type FP32. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads and accuracy type FP32.**

**Seal Text Detection Module Models:**

|Model|Detection Hmean (%)|GPU Inference Time (ms)|CPU Inference Time (ms)|Model Size (M)|Description|
|-|-|-|-|-|-|
|PP-OCRv4_server_seal_det|98.21|84.341|2425.06|109|The server-side seal text detection model of PP-OCRv4, with higher accuracy, suitable for deployment on high-performance servers.|
|PP-OCRv4_mobile_seal_det|96.47|10.5878|131.813|4.6|The mobile-side seal text detection model of PP-OCRv4, with higher efficiency, suitable for edge deployment.|

**Note: The above accuracy metrics are evaluated on an internal dataset containing 500 circular seal images. GPU inference times are based on NVIDIA Tesla T4 machines, with accuracy type FP32. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads and accuracy type FP32.**

**Text Image Correction Module Models:**

|Model|MS-SSIM (%)|Model Size (M)|Description|
|-|-|-|-|
|UVDoc|54.40|30.3 M|High-precision text image correction model|

**The accuracy metrics for the model are measured using the [DocUNet benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html).**

**Document Image Orientation Classification Module Models:**

|Model|Top-1 Acc (%)|GPU Inference Time (ms)|CPU Inference Time (ms)|Model Size (M)|Description|
|-|-|-|-|-|-|
|PP-LCNet_x1_0_doc_ori|99.06|3.84845|9.23735|7|Document image classification model based on PP-LCNet_x1_0, containing four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees.|

**Note: The above accuracy metrics are evaluated on an internal dataset covering various scenarios such as certificates and documents, with 1,000 images. GPU inference times are based on NVIDIA Tesla T4 machines, with accuracy type FP32. CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads and accuracy type FP32.**

</details>

## 2. Quick Start
PaddleX's pre-trained model pipelines can be quickly experienced. You can experience the effect of the Document Scene Information Extraction v3 pipeline online or locally using Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/182491/webUI) the effect of the Document Scene Information Extraction v3 pipeline, using the official demo images for recognition, for example:

![](https://github.com/user-attachments/assets/aa261b2b-b79c-4487-9323-dfcc43c3d581)

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to **fine-tune the models in the pipeline online**.

### 2.2 Local Experience
Before using the PP-ChatOCRv3 pipeline locally, please ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Guide](../../../installation/installation_en.md).

A few lines of code are all you need to complete the quick inference of the pipeline. Using the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf), taking the PP-ChatOCRv3-doc pipeline as an example:

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

(1) Instantiate the `create_pipeline` to create a PP-ChatOCRv3-doc pipeline object: Specific parameter descriptions are as follows:

| Parameter | Description | Default | Type |
|-|-|-|-|
| `pipeline` | Pipeline name or pipeline configuration file path. If it's a pipeline name, it must be supported by PaddleX. | None | str |
| `llm_name` | Large Language Model name | "ernie-3.5" | str |
| `llm_params` | API configuration | {} | dict |
| `device(kwargs)` | Running device (None for automatic adaptation) | None | str/None |

(2) Call the `predict` method of the PP-ChatOCRv3-doc pipeline object for inference prediction: The `predict` method parameter is `x`, used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

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

After execution, the configuration file for the PP-ChatOCRv3-doc pipeline will be saved in the current path. If you wish to customize the save location, you can execute the following command (assuming the custom save location is `./my_path`):

```bash
paddlex --get_pipeline_config PP-ChatOCRv3-doc --save_path ./my_path
```
After obtaining the configuration file, you can customize the various configurations of the PP-ChatOCRv3-doc pipeline:

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

🚀 **High-Performance Deployment**: In actual production environments, many applications have stringent standards for the performance metrics (especially response speed) of deployment strategies to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance deployment procedures, please refer to the [PaddleX High-Performance Deployment Guide](../../../pipeline_deploy/high_performance_deploy_en.md).

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service-oriented deployment of pipelines. For detailed service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy_en.md).

Below are the API references and multi-language service invocation examples:

<details>
<summary>API Reference</summary>

For all operations provided by the service:

- Both the response body and the request body for POST requests are JSON data (JSON objects).
- When the request is processed successfully, the response status code is `200`, and the response body attributes are as follows:

    | Name | Type | Description |
    |------|------|-------------|
    | `errorCode` | `integer` | Error code. Fixed as `0`. |
    | `errorMsg` | `string` | Error description. Fixed as `"Success"`. |

    The response body may also have a `result` attribute of type `object`, which stores the operation result information.

- When the request is not processed successfully, the response body attributes are as follows:

    | Name | Type | Description |
    |------|------|-------------|
    | `errorCode` | `integer` | Error code. Same as the response status code. |
    | `errorMsg` | `string` | Error description. |

Operations provided by the service are as follows:

- **`analyzeImage`**

    Analyzes images using computer vision models to obtain OCR, table recognition results, etc., and extracts key information from the images.

    `POST /chatocr-vision`

    - Request body attributes:

        | Name | Type | Description | Required |
        |------|------|-------------|----------|
        | `image` | `string` | The URL of an image file or PDF file accessible by the service, or the Base64 encoded result of the content of the above-mentioned file types. For PDF files with more than 10 pages, only the content of the first 10 pages will be used. | Yes |
        | `fileType` | `integer` | File type. `0` indicates a PDF file, `1` indicates an image file. If this attribute is not present in the request body, the service will attempt to automatically infer the file type based on the URL. | No |
        | `useOricls` | `boolean` | Whether to enable document image orientation classification. This feature is enabled by default. | No |
        | `useCurve` | `boolean` | Whether to enable seal text detection. This feature is enabled by default. | No |
        | `useUvdoc` | `boolean` | Whether to enable text image correction. This feature is enabled by default. | No |
        | `inferenceParams` | `object` | Inference parameters. | No |

        Properties of `inferenceParams`:

        | Name | Type | Description | Required |
        |------|------|-------------|----------|
        | `maxLongSide` | `integer` | During inference, if the length of the longer side of the input image for the text detection model is greater than `maxLongSide`, the image will be scaled so that the length of the longer side equals `maxLongSide`. | No |

    - When the request is processed successfully, the `result` of the response body has the following attributes:

        | Name | Type | Description |
        |------|------|-------------|
        | `visionResults` | `array` | Analysis results obtained using computer vision models. The array length is 1 (for image input) or the smaller of the number of document pages and 10 (for PDF input). For PDF input, each element in the array represents the processing result of each page in the PDF file. |
        | `visionInfo` | `object` | Key information in the image, which can be used as input for other operations. |

        Each element in `visionResults` is an `object` with the following attributes:

        | Name | Type | Description |
        |------|------|-------------|
        | `texts` | `array` | Text positions, contents, and scores. |
        | `tables` | `array` | Table positions and contents. |
        | `inputImage` | `string` | Input image. The image is in JPEG format and encoded using Base64. |
        | `ocrImage` | `string` | OCR result image. The image is in JPEG format and encoded using Base64. |
        | `layoutImage` | `string` | Layout area detection result image. The image is in JPEG format and encoded using Base64. |

        Each element in `texts` is an `object` with the following attributes:

        | Name | Type | Description |
        |------|------|-------------|
        | `poly` | `array` | Text position. The elements in the array are the vertex coordinates of the polygon enclosing the text in```markdown
### chat

Interact with large language models to extract key information.

`POST /chatocr-vision`

- Request body attributes:

    | Name | Type | Description | Required |
    |------|------|-------------|----------|
    |`keys`|`array`|List of keywords.|Yes|
    |`visionInfo`|`object`|Key information from the image. Provided by the `analyzeImage` operation.|Yes|
    |`taskDescription`|`string`|Task prompt.|No|
    |`rules`|`string`|Extraction rules. Used to customize the information extraction rules, e.g., to specify output formats.|No|
    |`fewShot`|`string`|Example prompts.|No|
    |`useVectorStore`|`boolean`|Whether to enable the vector database. Enabled by default.|No|
    |`vectorStore`|`object`|Serialized result of the vector database. Provided by the `buildVectorStore` operation.|No|
    |`retrievalResult`|`string`|Knowledge retrieval result. Provided by the `retrieveKnowledge` operation.|No|
    |`returnPrompts`|`boolean`|Whether to return the prompts used. Enabled by default.|No|
    |`llmName`|`string`|Name of the large language model.|No|
    |`llmParams`|`object`|API parameters for the large language model.|No|

    Currently, `llmParams` can take the following form:

    ```json
    {
      "apiType": "qianfan",
      "apiKey": "{Qianfan Platform API key}",
      "secretKey": "{Qianfan Platform secret key}"
    }
    ```

- On successful request processing, the `result` in the response body has the following attributes:

    | Name | Type | Description |
    |------|------|-------------|
    |`chatResult`|`string`|Extracted key information result.|
    |`prompts`|`object`|Prompts used.|

    Properties of `prompts`:

    | Name | Type | Description |
    |------|------|-------------|
    |`ocr`|`string`|OCR prompt.|
    |`table`|`string`|Table prompt.|
    |`html`|`string`|HTML prompt.|

</details>

<details>
<summary>Multilingual Service Invocation Examples</summary>

<details>
<summary>Python</summary>

```python
import base64
import pprint
import sys

import requests


API_BASE_URL = "http://0.0.0.0:8080"
API_KEY = "{Qianfan Platform API key}"
SECRET_KEY = "{Qianfan Platform secret key}"
LLM_NAME = "ernie-3.5"
LLM_PARAMS = {
    "apiType": "qianfan",
    "apiKey": API_KEY,
    "secretKey": SECRET_KEY,
}


if __name__ == "__main__":
    file_path = "./demo.jpg"
    keys = ["phone number"]

    with open(file_path, "rb") as file:
        file_bytes = file.read()
        file_data = base64.b64encode(file_bytes).decode("ascii")

    payload = {
        "file": file_data,
        "useOricls": True,
        "useCurve": True,
        "useUvdoc": True,
    }
    resp_vision = requests.post(url=f"{API_BASE_URL}/chatocr-vision", json=payload)
    if resp_vision.status_code != 200:
        print(
            f"Request to chatocr-vision failed with status code {resp_vision.status_code}."
        )
        pprint.pp(resp_vision.json())
        sys.exit(1)
    result_vision = resp_vision.json()["result"]

    for i, res in enumerate(result_vision["visionResults"]):
        print("Texts:")
        pprint.pp(res["texts"])
        print("Tables:")
        pprint.pp(res["tables"])
        ocr_img_path = f"ocr_{i}.jpg"
        with open(ocr_img_path, "wb") as f:
            f.write(base64.b64decode(res["ocrImage"]))
        layout_img_path = f"layout_{i}.jpg"
        with open(layout_img_path, "wb") as f:
            f.write(base64.b64decode(res["layoutImage"]))
        print(f"Output images saved at {ocr_img_path} and {layout_img_path}")
        print("")
    print("="*50 + "\n\n")

    payload = {
        "visionInfo": result_vision["visionInfo"],
        "minChars": 200,
        "llmRequestInterval": 1000,
        "llmName": LLM_NAME,
        "llmParams": LLM_PARAMS,
    }
    resp_vector = requests.post(url=f"{API_BASE_URL}/chatocr-vector", json=payload)
    if resp_vector.status_code != 200:
        print(
            f"Request to chatocr-vector failed with status code {resp_vector.status_code}."
        )
        pprint.pp(resp_vector.json())
        sys.exit(1)
    result_vector = resp_vector.json()["result"]
    print("="*50 + "\n\n")

    payload = {
        "keys": keys,
        "vectorStore": result_vector["vectorStore"],
        "visionInfo": result_vision["visionInfo"],
        "llmName": LLM_NAME,
        "llmParams": LLM_PARAMS,
    }
    resp_retrieval = requests.post(url=f"{API_BASE_URL}/chatocr-retrieval", json=payload)
    if resp_retrieval.status_code != 200:
        print(
            f"Request to chatocr-retrieval failed with status code {resp_retrieval.status_code}."
        )
        pprint.pp(resp_retrieval.json())
        sys.exit(1)
    result_retrieval = resp_retrieval.json()["result"]
    print("Knowledge retrieval result:")
    print(result_retrieval["retrievalResult"])
    print("="*50 + "\n\n")

    payload = {
        "keys": keys,
        "visionInfo": result_vision["visionInfo"],
        "taskDescription": "",
        "rules": "",
        "fewShot": "",
        "useVectorStore": True,
        "vectorStore": result_vector["vectorStore"],
        "retrievalResult": result_retrieval["retrievalResult"],
        "returnPrompts": True,
        "llmName": LLM_NAME,
        "llmParams": LLM_PARAMS,
    }
    resp_chat = requests.post(url=f"{API_BASE_URL}/chatocr-chat", json=payload)
    if resp_chat.status_code != 200:
        print(
            f"Request to chatocr-chat failed with status code {resp_chat.status_code}."
        )
        pprint.pp(resp_chat.json())
        sys.exit(1)
    result_chat = resp_chat.json()["result"]
    print("Prompts:")
    pprint.pp(result_chat["prompts"])
    print("Final result:")
    print(len(result_chat["chatResult"]))
```
</details>
</details>
<br/>

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing functions on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/lite_deploy_en.md).
## 4. Customization and Fine-tuning
If the default model weights provided by the PP-ChatOCRv3-doc Pipeline do not meet your requirements in terms of accuracy or speed for your specific scenario, you can attempt to further **fine-tune** the existing models using **your own domain-specific or application-specific data** to enhance the recognition performance of the general table recognition pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the PP-ChatOCRv3-doc Pipeline comprises four modules, unsatisfactory performance may stem from any of these modules (note that the text image rectification module does not support customization at this time).

You can analyze images with poor recognition results and follow the guidelines below for analysis and model fine-tuning:

* Incorrect table structure detection (e.g., row/column misidentification, cell position errors) may indicate deficiencies in the table structure recognition module. You need to refer to the **Customization** section in the [Table Structure Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/table_structure_recognition_en.md) and fine-tune the table structure recognition model using your private dataset.
* Misplaced layout elements (e.g., incorrect positioning of tables or seals) may suggest issues with the layout detection module. Consult the **Customization** section in the [Layout Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/layout_detection_en.md) and fine-tune the layout detection model with your private dataset.
* Frequent undetected text (i.e., text leakage) may indicate limitations in the text detection model. Refer to the **Customization** section in the [Text Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_detection_en.md) and fine-tune the text detection model using your private dataset.
* High text recognition errors (i.e., recognized text content does not match the actual text) suggest that the text recognition model requires improvement. Follow the **Customization** section in the [Text Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_recognition_en.md) to fine-tune the text recognition model.
* Frequent recognition errors in detected seal text indicate that the seal text detection model needs further refinement. Consult the **Customization** section in the [Seal Text Detection Module Development Tutorials](../../../module_usage/tutorials/ocr_modules/text_detection_en.md) to fine-tune the seal text detection model.
* Frequent misidentifications of document or certificate orientations with text regions suggest that the document image orientation classification model requires improvement. Refer to the **Customization** section in the [Document Image Orientation Classification Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification_en.md) to fine-tune the document image orientation classification model.

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

For example, to perform inference using the PP-ChatOCRv3-doc Pipeline on an NVIDIA GPU```
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the script to `npu`:

```python
from paddlex import create_pipeline
predict = create_pipeline(pipeline="PP-ChatOCRv3-doc",
                            llm_name="ernie-3.5",
                            llm_params = {"api_type":"qianfan","ak":"","sk":""},  ## Please fill in your ak and sk, or you will not be able to call the large model
                            device = "npu:0")
```
If you want to use the PP-ChatOCRv3-doc Pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../installation/installation_other_devices_en.md).
