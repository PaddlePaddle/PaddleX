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
|PP-OCRv4_server_det|82.69|83.3501|2434.01|100.1M|

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

- **`infer`**

    定位并识别图中的表格。

    `POST /table-recognition`

    - 请求体的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`image`|`string`|服务可访问的图像文件的URL或图像文件内容的Base64编码结果。|是|
        |`inferenceParams`|`object`|推理参数。|否|

        `inferenceParams`的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`maxLongSide`|`integer`|推理时，若文本检测模型的输入图像较长边的长度大于`maxLongSide`，则将对图像进行缩放，使其较长边的长度等于`maxLongSide`。|否|

    - 请求处理成功时，响应体的`result`具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`tables`|`array`|表格位置和内容。|
        |`layoutImage`|`string`|版面区域检测结果图。图像为JPEG格式，使用Base64编码。|
        |`ocrImage`|`string`|OCR结果图。图像为JPEG格式，使用Base64编码。|

        `tables`中的每个元素为一个`object`，具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`bbox`|`array`|表格位置。数组中元素依次为边界框左上角x坐标、左上角y坐标、右下角x坐标以及右下角y坐标。|
        |`html`|`string`|HTML格式的表格识别结果。|

</details>

<details>
<summary>Multilingual Service Invocation Examples</summary>  

<details>  
<summary>Python</summary>  
  
```python
import base64
import requests

API_URL = "http://localhost:8080/table-recognition" # 服务URL
image_path = "./demo.jpg"
ocr_image_path = "./ocr.jpg"
layout_image_path = "./table.jpg"

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64编码的文件内容或者图像URL

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()["result"]
with open(ocr_image_path, "wb") as file:
    file.write(base64.b64decode(result["ocrImage"]))
print(f"Output image saved at {ocr_image_path}")
with open(layout_image_path, "wb") as file:
    file.write(base64.b64decode(result["layoutImage"]))
print(f"Output image saved at {layout_image_path}")
print("\nDetected tables:")
print(result["tables"])
```
  
</details>

<details>  
<summary>C++</summary>  
  
```cpp
#include <iostream>
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client("localhost:8080");
    const std::string imagePath = "./demo.jpg";
    const std::string ocrImagePath = "./ocr.jpg";
    const std::string layoutImagePath = "./table.jpg";

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    // 对本地图像进行Base64编码
    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["image"] = encodedImage;
    std::string body = jsonObj.dump();

    // 调用API
    auto response = client.Post("/table-recognition", headers, body, "application/json");
    // 处理接口返回数据
    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        encodedImage = result["ocrImage"];
        std::string decoded_string = base64::from_base64(encodedImage);
        std::vector<unsigned char> decodedOcrImage(decoded_string.begin(), decoded_string.end());
        std::ofstream outputOcrFile(ocrImagePath, std::ios::binary | std::ios::out);
        if (outputOcrFile.is_open()) {
            outputOcrFile.write(reinterpret_cast<char*>(decodedOcrImage.data()), decodedOcrImage.size());
            outputOcrFile.close();
            std::cout << "Output image saved at " << ocrImagePath << std::endl;
        } else {
            std::cerr << "Unable to open file for writing: " << ocrImagePath << std::endl;
        }

        encodedImage = result["layoutImage"];
        decodedString = base64::from_base64(encodedImage);
        std::vector<unsigned char> decodedTableImage(decodedString.begin(), decodedString.end());
        std::ofstream outputTableFile(layoutImagePath, std::ios::binary | std::ios::out);
        if (outputTableFile.is_open()) {
            outputTableFile.write(reinterpret_cast<char*>(decodedTableImage.data()), decodedTableImage.size());
            outputTableFile.close();
            std::cout << "Output image saved at " << layoutImagePath << std::endl;
        } else {
            std::cerr << "Unable to open file for writing: " << layoutImagePath << std::endl;
        }

        auto tables = result["tables"];
        std::cout << "\nDetected tables:" << std::endl;
        for (const auto& category : tables) {
            std::cout << category << std::endl;
        }
    } else {
        std::cout << "Failed to send HTTP request." << std::endl;
        return 1;
    }

    return 0;
}
```
  
</details>

<details>  
<summary>Java</summary>  
  
```java
import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/table-recognition"; // 服务URL
        String imagePath = "./demo.jpg"; // 本地图像
        String ocrImagePath = "./ocr.jpg";
        String layoutImagePath = "./table.jpg";

        // 对本地图像进行Base64编码
        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("image", imageData); // Base64编码的文件内容或者图像URL

        // 创建 OkHttpClient 实例
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // 调用API并处理接口返回数据
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get("result");
                String ocrBase64Image = result.get("ocrImage").asText();
                String layoutBase64Image = result.get("layoutImage").asText();
                JsonNode tables = result.get("tables");

                byte[] imageBytes = Base64.getDecoder().decode(ocrBase64Image);
                try (FileOutputStream fos = new FileOutputStream(ocrImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println("Output image saved at " + ocrBase64Image);

                imageBytes = Base64.getDecoder().decode(layoutBase64Image);
                try (FileOutputStream fos = new FileOutputStream(layoutImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println("Output image saved at " + layoutImagePath);

                System.out.println("\nDetected tables: " + tables.toString());
            } else {
                System.err.println("Request failed with code: " + response.code());
            }
        }
    }
}
```
  
</details>

<details>  
<summary>Go</summary>  
  
```go
package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
)

func main() {
	API_URL := "http://localhost:8080/table-recognition"
	imagePath := "./demo.jpg"
	ocrImagePath := "./ocr.jpg"
	layoutImagePath := "./table.jpg"

	// 对本地图像进行Base64编码
	imageBytes, err := ioutil.ReadFile(imagePath)
	if err != nil {
		fmt.Println("Error reading image file:", err)
		return
	}
	imageData := base64.StdEncoding.EncodeToString(imageBytes)

	payload := map[string]string{"image": imageData} // Base64编码的文件内容或者图像URL
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		fmt.Println("Error marshaling payload:", err)
		return
	}

	// 调用API
	client := &http.Client{}
	req, err := http.NewRequest("POST", API_URL, bytes.NewBuffer(payloadBytes))
	if err != nil {
		fmt.Println("Error creating request:", err)
		return
	}

	res, err := client.Do(req)
	if err != nil {
		fmt.Println("Error sending request:", err)
		return
	}
	defer res.Body.Close()

    // 处理接口返回数据
	body, err := ioutil.ReadAll(res.Body)
	if err != nil {
		fmt.Println("Error reading response body:", err)
		return
	}
	type Response struct {
		Result struct {
			OcrImage      string   `json:"ocrImage"`
            TableImage      string   `json:"layoutImage"`
			Tables []map[string]interface{} `json:"tables"`
		} `json:"result"`
	}
	var respData Response
	err = json.Unmarshal([]byte(string(body)), &respData)
	if err != nil {
		fmt.Println("Error unmarshaling response body:", err)
		return
	}

	ocrImageData, err := base64.StdEncoding.DecodeString(respData.Result.OcrImage)
	if err != nil {
		fmt.Println("Error decoding base64 image data:", err)
		return
	}
	err = ioutil.WriteFile(ocrImagePath, ocrImageData, 0644)
	if err != nil {
		fmt.Println("Error writing image to file:", err)
		return
	}
    fmt.Printf("Image saved at %s.jpg\n", ocrImagePath)

    layoutImageData, err := base64.StdEncoding.DecodeString(respData.Result.TableImage)
	if err != nil {
		fmt.Println("Error decoding base64 image data:", err)
		return
	}
	err = ioutil.WriteFile(layoutImagePath, layoutImageData, 0644)
	if err != nil {
		fmt.Println("Error writing image to file:", err)
		return
	}
    fmt.Printf("Image saved at %s.jpg\n", layoutImagePath)

	fmt.Println("\nDetected tables:")
	for _, category := range respData.Result.Tables {
		fmt.Println(category)
	}
}
```
  
</details>

<details>  
<summary>C#</summary>  
  
```csharp
using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = "http://localhost:8080/table-recognition";
    static readonly string imagePath = "./demo.jpg";
    static readonly string ocrImagePath = "./ocr.jpg";
    static readonly string layoutImagePath = "./table.jpg";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // 对本地图像进行Base64编码
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { "image", image_data } }; // Base64编码的文件内容或者图像URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        // 调用API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // 处理接口返回数据
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string ocrBase64Image = jsonResponse["result"]["ocrImage"].ToString();
        byte[] ocrImageBytes = Convert.FromBase64String(ocrBase64Image);
        File.WriteAllBytes(ocrImagePath, ocrImageBytes);
        Console.WriteLine($"Output image saved at {ocrImagePath}");

        string layoutBase64Image = jsonResponse["result"]["layoutImage"].ToString();
        byte[] layoutImageBytes = Convert.FromBase64String(layoutBase64Image);
        File.WriteAllBytes(layoutImagePath, layoutImageBytes);
        Console.WriteLine($"Output image saved at {layoutImagePath}");

        Console.WriteLine("\nDetected tables:");
        Console.WriteLine(jsonResponse["result"]["tables"].ToString());
    }
}
```
  
</details>

<details>  
<summary>Node.js</summary>  
  
```js
const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/table-recognition'
const imagePath = './demo.jpg'
const ocrImagePath = "./ocr.jpg";
const layoutImagePath = "./table.jpg";

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)  // Base64编码的文件内容或者图像URL
  })
};

// 对本地图像进行Base64编码
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// 调用API
axios.request(config)
.then((response) => {
    // 处理接口返回数据
    const result = response.data["result"];

    const imageBuffer = Buffer.from(result["ocrImage"], 'base64');
    fs.writeFile(ocrImagePath, imageBuffer, (err) => {
      if (err) throw err;
      console.log(`Output image saved at ${ocrImagePath}`);
    });

    imageBuffer = Buffer.from(result["layoutImage"], 'base64');
    fs.writeFile(layoutImagePath, imageBuffer, (err) => {
      if (err) throw err;
      console.log(`Output image saved at ${layoutImagePath}`);
    });

    console.log("\nDetected tables:");
    console.log(result["tables"]);
})
.catch((error) => {
  console.log(error);
});
```
  
</details>

<details>  
<summary>PHP</summary>  
  
```php
<?php

$API_URL = "http://localhost:8080/table-recognition"; // 服务URL
$image_path = "./demo.jpg";
$ocr_image_path = "./ocr.jpg";
$layout_image_path = "./table.jpg";

// 对本地图像进行Base64编码
$image_data = base64_encode(file_get_contents($image_path));
$payload = array("image" => $image_data); // Base64编码的文件内容或者图像URL

// 调用API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// 处理接口返回数据
$result = json_decode($response, true)["result"];
file_put_contents($ocr_image_path, base64_decode($result["ocrImage"]));
echo "Output image saved at " . $ocr_image_path . "\n";

file_put_contents($layout_image_path, base64_decode($result["layoutImage"]));
echo "Output image saved at " . $layout_image_path . "\n";

echo "\nDetected tables:\n";
print_r($result["tables"]);

?>
```
  
</details>
</details>
<br/>

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
* If many detected texts contain recognition errors (i.e., the recognized text content does not match the actual text content), the text recognition model requires further improvement. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md#customization) section.
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
