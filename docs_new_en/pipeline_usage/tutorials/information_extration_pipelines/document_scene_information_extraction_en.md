# PP-ChatOCRv3 Pipeline Usage Tutorial

## 1. Introduction to PP-ChatOCRv3-doc Pipeline
Document Scene Information Extraction v3 (PP-ChatOCRv3) is a unique intelligent analysis solution for documents and images developed by PaddlePaddle. It combines Large Language Models (LLM) and OCR technology to provide a one-stop solution for complex document information extraction challenges such as layout analysis, rare characters, multi-page PDFs, tables, and seal recognition. By integrating with ERNIE Bot, it fuses massive data and knowledge to achieve high accuracy and wide applicability.

![](https://github.com/user-attachments/assets/90cb740b-7741-4383-bc4c-663f9d042d02)

The **PP-ChatOCRv3-doc** pipeline includes modules for **Table Structure Recognition, Layout Region Detection, Text Detection, Text Recognition, Seal Text Detection, Text Image Rectification, and Document Image Orientation Classification**.

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
paddlex --get_pipeline_config PP-ChatOCRv3-doc --config_save_path ./my_path
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
  
对于服务提供的所有操作：

- 响应体以及POST请求的请求体均为JSON数据（JSON对象）。
- 当请求处理成功时，响应状态码为`200`，响应体的属性如下：

    |名称|类型|含义|
    |-|-|-|
    |`errorCode`|`integer`|错误码。固定为`0`。|
    |`errorMsg`|`string`|错误说明。固定为`"Success"`。|

    响应体还可能有`result`属性，类型为`object`，其中存储操作结果信息。

- 当请求处理未成功时，响应体的属性如下：

    |名称|类型|含义|
    |-|-|-|
    |`errorCode`|`integer`|错误码。与响应状态码相同。|
    |`errorMsg`|`string`|错误说明。|

服务提供的操作如下：

- **`analyzeImage`**

    使用计算机视觉模型对图像进行分析，获得OCR、表格识别结果等，并提取图像中的关键信息。

    `POST /chatocr-vision`

    - 请求体的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`image`|`string`|服务可访问的图像文件或PDF文件的URL，或上述类型文件内容的Base64编码结果。对于超过10页的PDF文件，只有前10页的内容会被使用。|是|
        |`fileType`|`integer`|文件类型。`0`表示PDF文件，`1`表示图像文件。若请求体无此属性，则服务将尝试根据URL自动推断文件类型。|否|
        |`useOricls`|`boolean`|是否启用文档图像方向分类功能。默认启用该功能。|否|
        |`useCurve`|`boolean`|是否启用印章文本检测功能。默认启用该功能。|否|
        |`useUvdoc`|`boolean`|是否启用文本图像矫正功能。默认启用该功能。|否|
        |`inferenceParams`|`object`|推理参数。|否|

        `inferenceParams`的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`maxLongSide`|`integer`|推理时，若文本检测模型的输入图像较长边的长度大于`maxLongSide`，则将对图像进行缩放，使其较长边的长度等于`maxLongSide`。|否|

    - 请求处理成功时，响应体的`result`具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`visionResults`|`array`|使用计算机视觉模型得到的分析结果。数组长度为1（对于图像输入）或文档页数与10中的较小者（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中每一页的处理结果。|
        |`visionInfo`|`object`|图像中的关键信息，可用作其他操作的输入。|

        `visionResults`中的每个元素为一个`object`，具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`texts`|`array`|文本位置、内容和得分。|
        |`tables`|`array`|表格位置和内容。|
        |`inputImage`|`string`|输入图像。图像为JPEG格式，使用Base64编码。|
        |`ocrImage`|`string`|OCR结果图。图像为JPEG格式，使用Base64编码。|
        |`layoutImage`|`string`|版面区域检测结果图。图像为JPEG格式，使用Base64编码。|

        `texts`中的每个元素为一个`object`，具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`poly`|`array`|文本位置。数组中元素依次为包围文本的多边形的顶点坐标。|
        |`text`|`string`|文本内容。|
        |`score`|`number`|文本识别得分。|

        `tables`中的每个元素为一个`object`，具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`bbox`|`array`|表格位置。数组中元素依次为边界框左上角x坐标、左上角y坐标、右下角x坐标以及右下角y坐标。|
        |`html`|`string`|HTML格式的表格识别结果。|

- **`buildVectorStore`**

    构建向量数据库。

    `POST /chatocr-vector`

    - 请求体的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`visionInfo`|`object`|图像中的关键信息。由`analyzeImage`操作提供。|是|
        |`minChars`|`integer`|启用向量数据库的最小数据长度。|否|
        |`llmRequestInterval`|`number`|调用大语言模型API的间隔时间。|否|
        |`llmName`|`string`|大语言模型名称。|否|
        |`llmParams`|`object`|大语言模型API参数。|否|

        当前，`llmParams`可以采用如下两种形式之一：
        
        ```json
        {
          "apiType": "qianfan",
          "apiKey": "{千帆平台API key}",
          "secretKey": "{千帆平台secret key}"
        }
        ```

        ```json
        {
          "apiType": "{aistudio}",
          "accessToken": "{AI Studio访问令牌}"
        }
        ```

    - 请求处理成功时，响应体的`result`具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`vectorStore`|`object`|向量数据库序列化结果，可用作其他操作的输入。|

- **`retrieveKnowledge`**

    进行知识检索。

    `POST /chatocr-retrieval`

    - 请求体的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`keys`|`array`|关键词列表。|是|
        |`vectorStore`|`object`|向量数据库序列化结果。由`buildVectorStore`操作提供。|是|
        |`visionInfo`|`object`|图像中的关键信息。由`analyzeImage`操作提供。|是|
        |`llmName`|`string`|大语言模型名称。|否|
        |`llmParams`|`object`|大语言模型API参数。|否|

        当前，`llmParams`可以采用如下两种形式之一：
        
        ```json
        {
          "apiType": "qianfan",
          "apiKey": "{千帆平台API key}",
          "secretKey": "{千帆平台secret key}"
        }
        ```

        ```json
        {
          "apiType": "{aistudio}",
          "accessToken": "{AI Studio访问令牌}"
        }
        ```

    - 请求处理成功时，响应体的`result`具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`retrievalResult`|`string`|知识检索结果，可用作其他操作的输入。|

- **`chat`**

    与大语言模型交互，利用大语言模型提炼关键信息。

    `POST /chatocr-vision`

    - 请求体的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`keys`|`array`|关键词列表。|是|
        |`visionInfo`|`object`|图像中的关键信息。由`analyzeImage`操作提供。|是|
        |`taskDescription`|`string`|提示词任务。|否|
        |`rules`|`string`|提示词规则。用于自定义信息抽取规则，例如规范输出格式。|否|
        |`fewShot`|`string`|提示词示例。|否|
        |`useVectorStore`|`boolean`|是否启用向量数据库。默认启用。|否|
        |`vectorStore`|`object`|向量数据库序列化结果。由`buildVectorStore`操作提供。|否|
        |`retrievalResult`|`string`|知识检索结果。由`retrieveKnowledge`操作提供。|否|
        |`returnPrompts`|`boolean`|是否返回使用的提示词。默认启用。|否|
        |`llmName`|`string`|大语言模型名称。|否|
        |`llmParams`|`object`|大语言模型API参数。|否|

        当前，`llmParams`可以采用如下两种形式之一：
        
        ```json
        {
          "apiType": "qianfan",
          "apiKey": "{千帆平台API key}",
          "secretKey": "{千帆平台secret key}"
        }
        ```

        ```json
        {
          "apiType": "{aistudio}",
          "accessToken": "{AI Studio访问令牌}"
        }
        ```

    - 请求处理成功时，响应体的`result`具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`chatResult`|`string`|关键信息抽取结果。|
        |`prompts`|`object`|使用的提示词。|

        `prompts`的属性如下：

        |名称|类型|含义|
        |-|-|-|
        |`ocr`|`string`|OCR提示词。|
        |`table`|`string`|表格提示词。|
        |`html`|`string`|HTML提示词。|

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
API_KEY = "{千帆平台API key}"
SECRET_KEY = "{千帆平台secret key}"
LLM_NAME = "ernie-3.5"
LLM_PARAMS = {
    "apiType": "qianfan", 
    "apiKey": API_KEY, 
    "secretKey": SECRET_KEY,
}


if __name__ == "__main__":
    file_url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/serving/pipeline_data/ppchatocr/driving_license.jpg"
    keys = ["电话"]

    payload = {
        "file": file_url,
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
