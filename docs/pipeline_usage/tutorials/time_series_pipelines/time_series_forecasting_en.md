[简体中文](time_series_forecasting.md) | English

# Time Series Forecasting Pipeline Tutorial

## 1. Introduction to the General Time Series Forecasting Pipeline
Time series forecasting is a technique that utilizes historical data to predict future trends by analyzing the patterns of change in time series data. It is widely applied in fields such as financial markets, weather forecasting, and sales prediction. Time series forecasting often employs statistical methods or deep learning models (e.g., LSTM, ARIMA), capable of handling temporal dependencies in data to provide accurate predictions, assisting decision-makers in better planning and response. This technology plays a crucial role in various industries, including energy management, supply chain optimization, and market analysis.

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/time_series/03.png)

**The General Time Series Forecasting Pipeline includes a time series forecasting module. If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.**

<details>
   <summary> 👉Model List Details</summary>

|Model Name|MSE|MAE|Model Storage Size (M)|
|-|-|-|-|
|DLinear|0.382|0.394|72K|
|NLinear|0.386|0.392|40K |
|Nonstationary|0.600|0.515|55.5 M|
|PatchTST|0.385|0.397|2.0M |
|RLinear|0.384|0.392|40K|
|TiDE|0.405|0.412|31.7M|
|TimesNet|0.417|0.431|4.9M|

**Note: The above accuracy metrics are measured on [ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar). All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

</details>

## 2. Quick Start
The pre-trained model pipelines provided by PaddleX allow for quick experience of their effects. You can experience the effects of the General Time Series Forecasting Pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience the General Time Series Forecasting Pipeline online](https://aistudio.baidu.com/community/app/105706/webUI?source=appCenter) using the demo provided by the official team, for example:

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/time_series/04.png)

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to **fine-tune the model within the pipeline online**.

Note: Due to the close relationship between time series data and scenarios, the official built-in models for online time series tasks are scenario-specific and not universal. Therefore, the experience mode does not support using arbitrary files to experience the effects of the official model solutions. However, after training a model with your own scenario data, you can select your trained model solution and use data from the corresponding scenario for online experience.

### 2.2 Local Experience
Before using the General Time Series Forecasting Pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation_en.md).

#### 2.2.1 Command Line Experience
Experience the time series forecasting pipeline with a single command:

Experience the image anomaly detection pipeline with a single command，Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv), and replace `--input` with the local path to perform prediction.

```bash
paddlex --pipeline ts_fc --input ts_fc.csv --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it is the time series forecasting pipeline.
--input: The local path or URL of the input sequence to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the second and third GPUs). You can also choose to use CPU (--device cpu).
```

When executing the above command, the default image anomaly detection pipeline configuration file is loaded. If you need to customize the configuration file, you can run the following command to obtain it:

<details>
   <summary> 👉Click to expand</summary>

```bash
paddlex --get_pipeline_config ts_fc --save_path ./my_path
```

After obtaining the pipeline configuration file, you can replace `--pipeline` with the configuration file save path to make the configuration file take effect. For example, if the configuration file save path is `./ts_fc.yaml`, simply execute:

```bash
paddlex --pipeline ./ts_fc.yaml --input ts_fc.csv --device gpu:0
```

Here, parameters such as `--model` and `--device` do not need to be specified, as they will use the parameters in the configuration file. If parameters are still specified, the specified parameters will take precedence.

</details>

After running, the result is:

```bash
{'input_path': 'ts_fc.csv', 'forecast':                            OT
date
2018-06-26 20:00:00  9.586131
2018-06-26 21:00:00  9.379762
2018-06-26 22:00:00  9.252275
2018-06-26 23:00:00  9.249993
2018-06-27 00:00:00  9.164998
...                       ...
2018-06-30 15:00:00  8.830340
2018-06-30 16:00:00  9.291553
2018-06-30 17:00:00  9.097666
2018-06-30 18:00:00  8.905430
2018-06-30 19:00:00  8.993793

[96 rows x 1 columns]}
```

#### 2.2.2 Python Script Integration
A few lines of code can complete the quick inference of the production line. Taking the general time series prediction production line as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="ts_fc")

output = pipeline.predict("ts_fc.csv")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_csv("./output/")  # Save the results in CSV format
```
The result obtained is the same as that of the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the production line object using `create_pipeline`: Specific parameter descriptions are as follows:

| Parameter | Description | Type | Default Value |
|-----------|-------------|------|---------------|
| `pipeline` | The name of the production line or the path to the production line configuration file. If it is the name of the production line, it must be supported by PaddleX. | `str` | None |
| `device` | The device for production line model inference. Supports: "gpu", "cpu". | `str` | "gpu" |
| `use_hpip` | Whether to enable high-performance inference, only available when the production line supports high-performance inference. | `bool` | `False` |

（2）Invoke the `predict` method of the  production line object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Parameter Description |
|---------------|-----------------------------------------------------------------------------------------------------------|
| Python Var    | Supports directly passing in Python variables, such as numpy.ndarray representing image data. |
| str         | Supports passing in the path of the file to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| str           | Supports passing in the URL of the file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv). |
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

For example, if your configuration file is saved at `./my_path/ts_fc.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/ts_fc.yaml")
output = pipeline.predict("ts_fc.csv")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_csv("./output/")  # Save results in CSV format
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy in production, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Inference**: In actual production environments, many applications have stringent standards for deployment strategy performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end process acceleration. For detailed high-performance inference procedures, refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference_en.md).

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service-oriented deployment of pipelines. For detailed service-oriented deployment procedures, refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy_en.md).

Below are the API references and multi-language service invocation examples:

<details>
<summary>API Reference</summary>

For all operations provided by the service:

- Both the response body and the request body for POST requests are JSON data (JSON objects).
- When the request is processed successfully, the response status code is `200`, and the response body properties are as follows:

    | Name | Type | Description |
    |------|------|-------------|
    |`errorCode`|`integer`|Error code. Fixed as `0`.|
    |`errorMsg`|`string`|Error message. Fixed as `"Success"`.|

    The response body may also have a `result` property of type `object`, which stores the operation result information.

- When the request is not processed successfully, the response body properties are as follows:

    | Name | Type | Description |
    |------|------|-------------|
    |`errorCode`|`integer`|Error code. Same as the response status code.|
    |`errorMsg`|`string`|Error message.|

Operations provided by the service are as follows:

- **`infer`**

    Performs time-series forecasting.

    `POST /time-series-forecasting`

    - The request body properties are as follows:

        | Name | Type | Description | Required |
        |------|------|-------------|----------|
        |`csv`|`string`|The URL of a CSV file accessible by the service or the Base64 encoded result of the CSV file content. The CSV file must be encoded in UTF-8.|Yes|

    - When the request is processed successfully, the `result` of the response body has the following properties:

        | Name | Type | Description |
        |------|------|-------------|
        |`csv`|`string`|The time-series forecasting result in CSV format. Encoded in UTF-8+Base64.|

        An example of `result` is as follows:

        ```json
        {
          "csv": "xxxxxx"
        }
        ```

</details>

<details>
<summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>

```python
import base64
import requests

API_URL = "http://localhost:8080/time-series-forecasting"
csv_path = "./test.csv"
output_csv_path = "./out.csv"

with open(csv_path, "rb") as file:
    csv_bytes = file.read()
    csv_data = base64.b64encode(csv_bytes).decode("ascii")

payload = {"csv": csv_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
with open(output_csv_path, "wb") as f:
    f.write(base64.b64decode(result["csv"]))
print(f"Output time-series data saved at  {output_csv_path}")
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
    const std::string csvPath = "./test.csv";
    const std::string outputCsvPath = "./out.csv";

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    std::ifstream file(csvPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    std::string encodedCsv = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["csv"] = encodedCsv;
    std::string body = jsonObj.dump();

    auto response = client.Post("/time-series-forecasting", headers, body, "application/json");
    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        encodedCsv = result["csv"];
        decodedString = base64::from_base64(encodedCsv);
        std::vector<unsigned char> decodedCsv(decodedString.begin(), decodedString.end());
        std::ofstream outputCsv(outputCsvPath, std::ios::binary | std::ios::out);
        if (outputCsv.is_open()) {
            outputCsv.write(reinterpret_cast<char*>(decodedCsv.data()), decodedCsv.size());
            outputCsv.close();
            std::cout << "Output time-series data saved at " << outputCsvPath << std::endl;
        } else {
            std::cerr << "Unable to open file for writing: " << outputCsvPath << std::endl;
        }
    } else {
        std::cout << "Failed to send HTTP request." << std::endl;
        std::cout << response->body << std::endl;
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
        String API_URL = "http://localhost:8080/time-series-forecasting";
        String csvPath = "./test.csv";
        String outputCsvPath = "./out.csv";

        File file = new File(csvPath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String csvData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("csv", csvData);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get("result");

                String base64Csv = result.get("csv").asText();
                byte[] csvBytes = Base64.getDecoder().decode(base64Csv);
                try (FileOutputStream fos = new FileOutputStream(outputCsvPath)) {
                    fos.write(csvBytes);
                }
                System.out.println("Output time-series data saved at " + outputCsvPath);
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
    API_URL := "http://localhost:8080/time-series-forecasting"
    csvPath := "./test.csv";
    outputCsvPath := "./out.csv";

    csvBytes, err := ioutil.ReadFile(csvPath)
    if err != nil {
        fmt.Println("Error reading csv file:", err)
        return
    }
    csvData := base64.StdEncoding.EncodeToString(csvBytes)

    payload := map[string]string{"csv": csvData}
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println("Error marshaling payload:", err)
        return
    }

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

    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }
    type Response struct {
        Result struct {
            Csv string `json:"csv"`
        } `json:"result"`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &respData)
    if err != nil {
        fmt.Println("Error unmarshaling response body:", err)
        return
    }

    outputCsvData, err := base64.StdEncoding.DecodeString(respData.Result.Csv)
    if err != nil {
        fmt.Println("Error decoding base64 csv data:", err)
        return
    }
    err = ioutil.WriteFile(outputCsvPath, outputCsvData, 0644)
    if err != nil {
        fmt.Println("Error writing csv to file:", err)
        return
    }
    fmt.Printf("Output time-series data saved at %s.csv", outputCsvPath)
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
    static readonly string API_URL = "http://localhost:8080/time-series-forecasting";
    static readonly string csvPath = "./test.csv";
    static readonly string outputCsvPath = "./out.csv";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] csvBytes = File.ReadAllBytes(csvPath);
        string csvData = Convert.ToBase64String(csvBytes);

        var payload = new JObject{ { "csv", csvData } };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Csv = jsonResponse["result"]["csv"].ToString();
        byte[] outputCsvBytes = Convert.FromBase64String(base64Csv);
        File.WriteAllBytes(outputCsvPath, outputCsvBytes);
        Console.WriteLine($"Output time-series data saved at {outputCsvPath}");
    }
}
```

</details>

<details>
<summary>Node.js</summary>

```js
const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/time-series-forecasting'
const csvPath = "./test.csv";
const outputCsvPath = "./out.csv";

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'csv': encodeFileToBase64(csvPath)
  })
};

function encodeFileToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

axios.request(config)
.then((response) => {
    const result = response.data["result"];

    const csvBuffer = Buffer.from(result["csv"], 'base64');
    fs.writeFile(outputCsvPath, csvBuffer, (err) => {
      if (err) throw err;
      console.log(`Output time-series data saved at ${outputCsvPath}`);
    });
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

$API_URL = "http://localhost:8080/time-series-forecasting";
$csv_path = "./test.csv";
$output_csv_path = "./out.csv";

$csv_data = base64_encode(file_get_contents($csv_path));
$payload = array("csv" => $csv_data);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)["result"];

file_put_contents($output_csv_path, base64_decode($result["csv"]));
echo "Output time-series data saved at " . $output_csv_path . "\n";

?>
```

</details>
</details>
<br/>

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing functions on user devices themselves, enabling devices to directly process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.md).
Choose the appropriate deployment method for your model pipeline based on your needs, and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the General Time Series Forecasting Pipeline do not meet your requirements in terms of accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using **your own domain-specific or application-specific data** to improve the recognition performance of the pipeline in your scenario.

#### 4.1 Model Fine-tuning
Since the General Time Series Forecasting Pipeline includes a time series forecasting module, if the performance of the pipeline does not meet expectations, you need to refer to the [Customization](../../../module_usage/tutorials/time_series_modules/time_series_forecast_en.md#iv-custom-development) section in the [Time Series Forecasting Module Development Tutorial](../../../module_usage/tutorials/time_series_modules/time_series_forecasting_en.md) and use your private dataset to fine-tune the time series forecasting model.

#### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain local model weight files.

To use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```python
......
Pipeline:
  model: DLinear  # Replace with the local path of the fine-tuned model
  device: "gpu"
  batch_size: 0
......
```
Then, refer to the command line or Python script methods in the local experience section to load the modified pipeline configuration file.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modify the `--device` parameter** to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for inference with the time series forecasting pipeline, the Python command would be:

```bash
paddlex --pipeline ts_fc --input ts_fc.csv --device gpu:0
``````
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the Python command to `npu:0`:

```bash
paddlex --pipeline ts_fc --input ts_fc.csv --device npu:0
```
If you want to use the General Time Series Forecasting Pipeline on a wider range of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide_en.md).
