---
comments: true
---

# Time Series Classification Pipeline Tutorial

## 1. Introduction to General Time Series Classification Pipeline
Time series classification is a technique that categorizes time-series data into predefined classes, widely applied in fields such as behavior recognition and financial trend analysis. By analyzing features that vary over time, it identifies different patterns or events, for example, classifying a speech signal as "greeting" or "request," or categorizing stock price movements as "rising" or "falling." Time series classification typically employs machine learning and deep learning models, effectively capturing temporal dependencies and variation patterns to provide accurate classification labels for data. This technology plays a pivotal role in applications such as intelligent monitoring and market forecasting.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/time_series/01.png">

<b>The General Time Series Classification Pipeline includes a Time Series Classification module.</b>

<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>Acc(%)</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>TimesNet_cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/TimesNet_cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_cls_pretrained.pdparams">Training Model</a></td>
<td>87.5</td>
<td>792K</td>
</tr>
</tbody>
</table>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
              <li><strong>Test Dataset：</strong><a href="https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TEST.csv">UWaveGestureLibrary</a> dataset.</li>
              <li><strong>Hardware Configuration：</strong>
                  <ul>
                      <li>GPU: NVIDIA Tesla T4</li>
                      <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>Other Environments: Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
                  </ul>
              </li>
          </ul>
      </li>
      <li><b>Inference Mode Description</b></li>
  </ul>

<table border="1">
    <thead>
        <tr>
            <th>Mode</th>
            <th>GPU Configuration </th>
            <th>CPU Configuration </th>
            <th>Acceleration Technology Combination</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Normal Mode</td>
            <td>FP32 Precision / No TRT Acceleration</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>High-Performance Mode</td>
            <td>Optimal combination of pre-selected precision types and acceleration strategies</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>Pre-selected optimal backend (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>

## 2. Quick Start
PaddleX provides pre-trained model pipelines that can be quickly experienced. You can experience the effects of the General Time Series Classification Pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/105707/webUI?source=appCenter) the effects of the General Time Series Classification Pipeline using the official demo for recognition, for example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/time_series/02.png">

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to <b>fine-tune the model in the pipeline online</b>.

Note: Due to the close relationship between time series data and scenarios, the official built-in model for online experience of time series tasks is only a model solution for a specific scenario and is not a general solution applicable to other scenarios. Therefore, the experience method does not support using arbitrary files to experience the effect of the official model solution. However, after training a model for your own scenario data, you can select your trained model solution and use data from the corresponding scenario for online experience.

### 2.2 Local Experience
Before using the general time-series classification pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md). If you wish to selectively install dependencies, please refer to the relevant instructions in the installation guide. The dependency group corresponding to this pipeline is `ts`.

#### 2.2.1 Command Line Experience
You can quickly experience the time-series classification pipeline with a single command. Use [the test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv) and replace `--input` with your local path for prediction.

```bash
paddlex --pipeline ts_classification --input ts_cls.csv --device gpu:0 --save_path ./output
```

For the explanation of the parameters, you can refer to the parameter description in Section 2.2.2 Integration via Python Script.

After running the command, the results will be printed to the terminal, as follows:

```
{'input_path': 'ts_cls.csv', 'classification':         classid     score
sample
0             0  0.617688}
```

The explanation of the result parameters can be found in the result interpretation section of [2.2.2 Python Script Integration](#222-python脚本方式集成).

The time-series file results are saved under `save_path`.

#### 2.2.2 Python Script Integration
The above command line is for quickly experiencing and viewing the results. Generally, in a project, you often need to integrate through code. You can complete the pipeline's fast inference with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="ts_classification", device="gpu:0")
output = pipeline.predict("ts_cls.csv")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_csv(save_path="./output/") ## 保存csv格式结果
    res.save_to_json(save_path="./output/") ## 保存json格式结果
```

In the above Python script, the following steps are executed:

(1) Instantiate the pipeline object using `create_pipeline()`: The parameters are explained as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is a pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the pipeline (if set simultaneously with <code>pipeline</code>, it takes precedence over <code>pipeline</code>, and the pipeline name must be consistent with <code>pipeline</code>).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the pipeline (if set simultaneously with <code>pipeline</code>, it takes precedence over <code>pipeline</code>, and the pipeline name must be consistent with <code>pipeline</code>).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for pipeline inference. It supports specifying specific GPU card numbers, such as "gpu:0", specific card numbers for other hardware, such as "npu:0", and CPU, such as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable the high-performance inference plugin. If set to <code>None</code>, the setting from the configuration file or <code>config</code> will be used.</td>
<td><code>bool</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>High-performance inference configuration</td>
<td><code>dict</code> | <code>None</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the ts_classification pipeline object for inference. This method returns a `generator`. The parameters and their descriptions for the `predict()` method are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>The data to be predicted. It supports multiple input types and is required.</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
  <li><b>Python Var</b>: Time-series data represented by <code>pandas.DataFrame</code>.</li>
  <li><b>str</b>: Local path of the time-series file, such as <code>/root/data/ts.csv</code>; <b>URL link</b>, such as the network URL of the time-series file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv">Example</a>; <b>Local directory</b>, which should contain the time-series data to be predicted, such as <code>/root/data/</code>.</li>
  <li><b>List</b>: The elements of the list must be of the above types, such as <code>[pandas.DataFrame, pandas.DataFrame]</code>, <code>["/root/data/ts1.csv", "/root/data/ts2.csv"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

(3) Process the prediction results. The prediction result for each sample is of type `dict`, and supports operations such as printing, saving as a `csv` file, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When a directory is provided, the saved file name will match the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_csv()</code></td>
<td>Save the result as a CSV file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>None</vd>
</tr>
</table>

- Calling the `print()` method will print the result to the terminal, with the following explanations for the printed content:
    - `input_path`: `(str)` The input path of the time-series file to be predicted
    - `classification`: `(Pandas.DataFrame)` The time-series classification result, including sample IDs, corresponding classification categories, and confidence scores.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_ts_basename}_res.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving NumPy arrays, `numpy.array` types will be converted to list format.
- Calling the `save_to_csv()` method will save the visualization results to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_ts_basename}_res.csv`. If a file is specified, it will be saved directly to that file.

* Additionally, it also supports obtaining prediction results in different formats through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>csv</code></td>
<td rowspan="2">Get the result in <code>csv</code> format</td>
</tr>
</table>

- The prediction result obtained through the `json` attribute is of type `dict`, with content consistent with what is saved by calling the `save_to_json()` method.
- The `csv` attribute returns a `Pandas.DataFrame` type data, which contains the time-series classification results.

Additionally, you can obtain the ts_classification pipeline configuration file and load it for prediction. You can run the following command to save the results in `my_path`:

```
paddlex --get_pipeline_config ts_classification --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the time-series classification pipeline by simply modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/ts_cls.yaml`, you just need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/ts_classification.yaml")
output = pipeline.predict("ts_cls.csv")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_csv("./output/") ## 保存csv格式结果
    res.save_to_json("./output/") ## 保存json格式结果
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the pipeline directly into your Python project, you can refer to the example code in [Section 2.2.2 Integration via Python Script](#222-integration-via-python-script).

Additionally, PaddleX offers three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In practical production environments, many applications have stringent performance requirements for deployment strategies, especially in terms of response speed, to ensure efficient system operation and a smooth user experience. To address this, PaddleX provides a high-performance inference plugin aimed at deeply optimizing the performance of model inference and pre/post-processing, significantly speeding up the end-to-end process. For detailed instructions, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Serving Deployment</b>: Serving Deployment is a common form of deployment in practical production environments. By encapsulating inference functionality into a service, clients can access these services via network requests to obtain inference results. PaddleX supports various pipeline serving deployment solutions. For detailed instructions, please refer to the [PaddleX Serving Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references for basic serving deployment and multi-language service call examples:

<details><summary>API Reference</summary>

<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>The result of the operation.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Same as the response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Classify time-series data.</p>
<p><code>POST /time-series-classification</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>csv</code></td>
<td><code>string</code></td>
<td>The URL of a CSV file accessible by the server or the Base64-encoded content of a CSV file. The CSV file must be encoded in UTF-8.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>visualize</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>
Whether to return the final visualization image and intermediate images during the processing.<br/>
<ul style="margin: 0 0 0 1em; padding-left: 0em;">
<li>If <code>true</code> is provided: return images.</li>
<li>If <code>false</code> is provided: do not return any images.</li>
<li>If this parameter is omitted from the request body, or if <code>null</code> is explicitly passed, the behavior will follow the value of <code>Serving.visualize</code> in the pipeline configuration.</li>
</ul>
<br/>
For example, adding the following setting to the pipeline config file:<br/>
<pre><code>Serving:
  visualize: False
</code></pre>
will disable image return by default. This behavior can be overridden by explicitly setting the <code>visualize</code> parameter in the request.<br/>
If neither the request body nor the configuration file is set (If <code>visualize</code> is set to <code>null</code> in the request and  not defined in the configuration file), the image is returned by default.
</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> in the response body has the following attributes:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>image</code></td>
<td><code>string</code> | <code>null</code></td>
<td>The image of time series classification result. The image is in JPEG format and encoded in Base64.</td>
</tr>
<tr>
<td><code>label</code></td>
<td><code>string</code></td>
<td>The class label.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The class score.</td>
</tr>
</tbody>
</table>
<p>An example of <code>result</code> is as follows:</p>
<pre><code class="language-json">{
"label": "running",
"score": 0.97,
"image": "xxxxxx"
}
</code></pre></details>

<details><summary>Multi-language Service Invocation Examples</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/time-series-classification" # Service URL
csv_path = "./test.csv"
output_image_path = "./out.jpg"

# Encode the local CSV file in Base64
with open(csv_path, "rb") as file:
    csv_bytes = file.read()
    csv_data = base64.b64encode(csv_bytes).decode("ascii")

payload = {"csv": csv_data}

# Call the API
response = requests.post(API_URL, json=payload)

# Process the response data
assert response.status_code == 200
result = response.json()["result"]
result = response.json()["result"]
with open(output_image_path, "wb") as f:
    f.write(base64.b64decode(result["image"]))
print(f"label: {result['label']}, score: {result['score']}")
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client("localhost:8080");
    const std::string outputImagePath = "./out.jpg";
    const std::string csvPath = "./test.csv";

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    // Encode in Base64
    std::ifstream file(csvPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; "Error reading file." &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedCsv = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["csv"] = encodedCsv;
    std::string body = jsonObj.dump();

    // Call the API
    auto response = client.Post("/time-series-classification", headers, body, "application/json");
    // Process the response data
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse["result"];
        std::cout &lt;&lt; "label: " &lt;&lt; result["label"] &lt;&lt; ", score: " &lt;&lt; result["score"] &lt;&lt; std::endl;

        std::string encodedImage = result["image"];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector<unsigned char> decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outputImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast<char*>(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout << "Output image data saved at " << outputImagePath << std::endl;
        } else {
            std::cerr << "Unable to open file for writing: " << outputImagePath << std::endl;
        }
    } else {
        std::cout &lt;&lt; "Failed to send HTTP request." &lt;&lt; std::endl;
        std::cout &lt;&lt; response-&gt;body &lt;&lt; std::endl;
        return 1;
    }

    return 0;
}
</code></pre></details>

<details><summary>Java</summary>

<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/time-series-classification";
        String outputImagePath = "./out.jpg";
        String csvPath = "./test.csv";

        // Encode the local CSV file using Base64
        File file = new File(csvPath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String csvData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("csv", csvData);

        // Create an OkHttpClient instance
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // Call the API and process the returned data
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get("result");
                System.out.println("label: " + result.get("label").asText() + ", score: " + result.get("score").asText());

                String base64Image = result.get("image").asText();
                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println("Output image data saved at " + outputImagePath);
            } else {
                System.err.println("Request failed with code: " + response.code());
            }
        }
    }
}
</code></pre></details>

<details><summary>Go</summary>

<pre><code class="language-go">package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
)

func main() {
    API_URL := "http://localhost:8080/time-series-classification"
    outputImagePath := "./out.jpg";
    csvPath := "./test.csv";

    // Read the CSV file and encode it with Base64
    csvBytes, err := ioutil.ReadFile(csvPath)
    if err != nil {
        fmt.Println("Error reading csv file:", err)
        return
    }
    csvData := base64.StdEncoding.EncodeToString(csvBytes)

    payload := map[string]string{"csv": csvData} // Base64-encoded file content
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println("Error marshaling payload:", err)
        return
    }

    // Call the API
    client := &amp;http.Client{}
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

    // Process the response data
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }
    type Response struct {
        Result struct {
            Label string `json:"label"`
            Score string `json:"score"`
            Image string `json:"image"`
        } `json:"result"`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println("Error unmarshaling response body:", err)
        return
    }

    fmt.Printf("label: %s, score: %s\n", respData.Result.Label, respData.Result.Score)

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println("Error decoding Base64 image data:", err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println("Error writing image to file:", err)
        return
    }
    fmt.Printf("Output image data saved at %s.jpg", outputImagePath)
}
</code></pre></details>

<details><summary>C#</summary>

<pre><code class="language-csharp">using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = "http://localhost:8080/time-series-classification";
    static readonly string outputImagePath = "./out.jpg";
    static readonly string csvPath = "./test.csv";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // Encode the local CSV file in Base64
        byte[] csvBytes = File.ReadAllBytes(csvPath);
        string csvData = Convert.ToBase64String(csvBytes);

        var payload = new JObject{ { "csv", csvData } }; // Base64-encoded file content
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        // Call the API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // Process the response data
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string label = jsonResponse["result"]["label"].ToString();
        string score = jsonResponse["result"]["score"].ToString();
        Console.WriteLine($"label: {label}, score: {score}");

        string base64Image = jsonResponse["result"]["image"].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);
        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($"Output image data saved at {outputImagePath}");
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/time-series-classification';
const outputImagePath = "./out.jpg";
const csvPath = './test.csv';

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'csv': encodeFileToBase64(csvPath)  // Base64-encoded file content
  })
};

// Read the CSV file and convert it to Base64
function encodeFileToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

axios.request(config)
.then((response) => {
    const result = response.data['result'];
    console.log(`label: ${result['label']}, score: ${result['score']}`);

    const imageBuffer = Buffer.from(result["image"], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image data saved at ${outputImagePath}`);
    });
})
.catch((error) => {
  console.log(error);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/time-series-classification"; // Service URL
$output_image_path = "./out.jpg";
$csv_path = "./test.csv";

// Encode the local CSV file using Base64
$csv_data = base64_encode(file_get_contents($csv_path));
$payload = array("csv" =&gt; $csv_data); // Base64-encoded file content

// Call the API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// Process the response data
$result = json_decode($response, true)["result"];
echo "label: " . $result["label"] . ", score: " . $result["score"];

file_put_contents($output_image_path, base64_decode($result["image"]));
echo "Output image data saved at " . $output_image_path . "\n";

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>On-Device Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on user devices, allowing them to process data locally without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed procedures, please refer to the [PaddleX On-Device Deployment Guide](../../../pipeline_deploy/on_device_deployment.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model pipeline into subsequent AI applications.


## 4. Custom Development
If the default model weights provided by the time-series classification pipeline do not meet your requirements in terms of accuracy or speed, you can try to <b>fine-tune</b> the existing model using <b>your own domain-specific or application data</b> to improve the performance of the time-series classification pipeline in your scenario.


### 4.1 Model Fine-Tuning
Since the time-series classification pipeline includes a time-series classification module, if the pipeline's performance is not satisfactory, you need to refer to the <b>Custom Development</b> section in the [Time-Series Classification Module Development Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/time_series_modules/time_series_classification.html) and fine-tune the time-series classification model using your private dataset.

### 4.2 Model Application
After you have completed fine-tuning training with your private dataset, you will obtain a local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file and fill in the local path of the fine-tuned model weights to the `model_dir` in the pipeline configuration file:

```yaml
pipeline_name: ts_classification

SubModules:
  TSClassification:
    module_name: ts_classification
    model_name: TimesNet_cls
    model_dir: null  # Can be modified to the local path of the fine-tuned model
    batch_size: 1
```

Subsequently, you can load the modified pipeline configuration file using the command line or Python script methods described in the local experience section.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware.

For example, if you are using Ascend NPU for inference in the time-series classification pipeline, the Python command is as follows:

```bash
paddlex --pipeline ts_classification --input ts_cls.csv --device npu:0
```

If you want to use the general time-series classification pipeline on a wider range of hardware devices, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
