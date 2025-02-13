---
comments: true
---

# Time Series Anomaly Detection Pipeline Tutorial

## 1. Introduction to the General Time Series Anomaly Detection Pipeline
Time series anomaly detection is a technique for identifying abnormal patterns or behaviors in time series data. It is widely applied in fields such as network security, equipment monitoring, and financial fraud detection. By analyzing normal trends and patterns in historical data, it discovers events that significantly deviate from expected behaviors, such as sudden spikes in network traffic or unusual transaction activities. Time series anomaly detection enable automatic identification of anomalies in data. This technology provides real-time alerts for enterprises and organizations, helping them promptly address potential risks and issues. It plays a crucial role in ensuring system stability and security.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/time_series/05.png">

<b>The General Time Series Anomaly Detection Pipeline includes a time series anomaly detection module. If you prioritize model accuracy, choose a model with higher precision. If you prioritize inference speed, select a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage footprint.</b>

<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>Precision</th>
<th>Recall</th>
<th>F1-Score</th>
<th>Model Storage Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>AutoEncoder_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/AutoEncoder_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/AutoEncoder_ad_pretrained.pdparams">Trained Model</a></td>
<td>99.36</td>
<td>84.36</td>
<td>91.25</td>
<td>52K</td>
</tr>
<tr>
<td>DLinear_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DLinear_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DLinear_ad_pretrained.pdparams">Trained Model</a></td>
<td>98.98</td>
<td>93.96</td>
<td>96.41</td>
<td>112K</td>
</tr>
<tr>
<td>Nonstationary_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Nonstationary_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Nonstationary_ad_pretrained.pdparams">Trained Model</a></td>
<td>98.55</td>
<td>88.95</td>
<td>93.51</td>
<td>1.8M</td>
</tr>
<tr>
<td>PatchTST_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PatchTST_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PatchTST_ad_pretrained.pdparams">Trained Model</a></td>
<td>98.78</td>
<td>90.70</td>
<td>94.57</td>
<td>320K</td>
</tr>
<tr>
<td>TimesNet_ad</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_ad_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_ad_pretrained.pdparams">Trained Model</a></td>
<td>98.37</td>
<td>94.80</td>
<td>96.56</td>
<td>1.3M</td>
</tr>
</tbody>
</table>
<p><b>Note: The above precision metrics are measured on the </b><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar">PSM</a><b> dataset. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## 2. Quick Start
The pre-trained model pipelines provided by PaddleX allow for quick experience of their effects. You can experience the effects of the General Time Series Anomaly Detection Pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/105706/webUI?source=appCenter) the effects of the General Time Series Anomaly Detection Pipeline using the official demo for recognition, for example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/time_series/06.png">

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to <b>fine-tune the model within the pipeline online</b>.

<b>Note</b>: Due to the close relationship between time series data and scenarios, the official built-in models for online experience of time series tasks are only model solutions for a specific scenario and are not universal. They are not applicable to other scenarios. Therefore, the experience mode does not support using arbitrary files to experience the effects of the official model solutions. However, after training a model for your own scenario data, you can select your trained model solution and use data from the corresponding scenario for online experience.

### 2.2 Local Experience
Before using the general time-series anomaly detection production line locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience
You can quickly experience the time-series anomaly detection production line with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv) and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline ts_anomaly_detection --input ts_ad.csv --device gpu:0 --save_path ./output
```

For the explanation of the parameters, you can refer to the parameter description in [Section 2.2.2 Integration via Python Script](#222-integration-via-python-script).

After running the command, the results will be printed to the terminal as follows:

```
{'input_path': 'ts_ad.csv', 'anomaly':            label
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

The result of the time series file is saved under `save_path`.

#### 2.2.2 Integration via Python Script
The above command line is for a quick experience to view the results. Generally, in a project, it is often necessary to integrate through code. You can complete the fast inference of the production line with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="ts_anomaly_detection")
output = pipeline.predict("ts_ad.csv")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_csv(save_path="./output/") ## 保存csv格式结果
    res.save_to_json(save_path="./output/") ## 保存json格式结果
```

In the above Python script, the following steps are performed:

(1) Instantiate the production line object through `create_pipeline()`: The specific parameter descriptions are as follows:

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
<td>The name of the production line or the path to the production line configuration file. If it is the name of the production line, it must be a production line supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the production line (if set simultaneously with <code>pipeline</code>, it takes precedence over <code>pipeline</code>, and the production line name must be consistent with <code>pipeline</code>).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The inference device for the production line. It supports specifying the specific card number of the hardware, such as "gpu:0" for GPU, "npu:0" for NPU, or "cpu" for CPU.</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference. This is only available if the production line supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the `ts_anomaly_detection` production line object for inference prediction. This method returns a `generator`. The parameters and their descriptions for the `predict()` method are as follows:

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
  <li><b>Python Var</b>: Time series data represented by <code>pandas.DataFrame</code>.</li>
  <li><b>str</b>: Local path of the time series file, such as <code>/root/data/ts.csv</code>; <b>URL link</b>, such as the network URL of the time series file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv">Example</a>; <b>Local directory</b>, which must contain the time series to be predicted, such as the local path: <code>/root/data/</code>.</li>
  <li><b>List</b>: The elements of the list must be of the above types, such as <code>[pandas.DataFrame, pandas.DataFrame]</code>, <code>["/root/data/ts1.csv", "/root/data/ts2.csv"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The inference device for the production line.</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>CPU</b>: <code>cpu</code> indicates using the CPU for inference;</li>
  <li><b>GPU</b>: <code>gpu:0</code> indicates using the first GPU for inference;</li>
  <li><b>NPU</b>: <code>npu:0</code> indicates using the first NPU for inference;</li>
  <li><b>XPU</b>: <code>xpu:0</code> indicates using the first XPU for inference;</li>
  <li><b>MLU</b>: <code>mlu:0</code> indicates using the first MLU for inference;</li>
  <li><b>DCU</b>: <code>dcu:0</code> indicates using the first DCU for inference;</li>
  <li><b>None</b>: If set to <code>None</code>, the value initialized for the production line will be used by default. During initialization, the local GPU device 0 will be prioritized. If not available, the CPU device will be used.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

* In addition, it also supports obtaining prediction results in different formats through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Obtain the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan = "2"><code>csv</code></td>
<td rowspan = "2">Obtain the prediction result in <code>csv</code> format</td>
</tr>
</table>

- The prediction result obtained through the `json` attribute is of dict type, and its content is consistent with the content saved by calling the `save_to_json()` method.
- The `csv` attribute returns a `Pandas.DataFrame` type data, which contains the time series anomaly detection results.

In addition, you can obtain the configuration file of the ts_anomaly_detection production line and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config ts_anomaly_detection --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the time series anomaly detection pipeline by simply modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/ts_anomaly_detection.yaml`, you just need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/ts_anomaly_detection.yaml")
output = pipeline.predict("ts_ad.csv")
for res in output:
    res.print()
    res.save_to_csv("./output/")
    res.save_to_json("./output/")
```

## 3. Development Integration/Deployment
If the production line meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the production line directly in your Python project, you can refer to the example code in [2.2.2 Python Script Method](#222-python脚本方式集成).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have strict performance requirements for deployment strategies, especially in terms of response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin, which aims to deeply optimize the performance of model inference and pre/post-processing to significantly speed up the end-to-end process. For details on high-performance inference, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service-based Deployment</b>: Service-based deployment is a common form of deployment in actual production environments. By encapsulating the inference function as a service, clients can access these services through network requests to obtain inference results. PaddleX supports various service-based deployment solutions for production lines. For details on service-based deployment, please refer to the [PaddleX Service-based Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references for basic service-based deployment and examples of multi-language service calls:

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
<p>Perform time-series anomaly detection.</p>
<p><code>POST /time-series-anomaly-detection</code></p>
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
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> of the response body has the following attributes:</li>
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
<td><code>csv</code></td>
<td><code>string</code></td>
<td>The result of time-series anomaly detection in CSV format. Encoded in UTF-8+Base64.</td>
</tr>
</tbody>
</table>
<p>An example of <code>result</code> is as follows:</p>
<pre><code class="language-json">{
&quot;csv&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>Multi-language Service Invocation Example</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/time-series-anomaly-detection"  # Service URL
csv_path = "./test.csv"
output_csv_path = "./out.csv"

# Encode the local CSV file using Base64
with open(csv_path, "rb") as file:
    csv_bytes = file.read()
    csv_data = base64.b64encode(csv_bytes).decode("ascii")

payload = {"csv": csv_data}

# Call the API
response = requests.post(API_URL, json=payload)

# Process the response data
assert response.status_code == 200
result = response.json()["result"]
with open(output_csv_path, "wb") as f:
    f.write(base64.b64decode(result["csv"]))
print(f"Output time-series data saved at {output_csv_path}")
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include <iostream>
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

    // Perform Base64 encoding
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

    // Call the API
    auto response = client.Post("/time-series-anomaly-detection", headers, body, "application/json");
    // Process the response data
    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        // Save the data
        encodedCsv = result["csv"];
        std::string decodedString = base64::from_base64(encodedCsv);
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
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // <url id="cu9m7enftaebd5d06u9g" type="url" status="parsed" title="GitHub - Huiyicc/cpp-httplib: A C++ header-only HTTP/HTTPS server and client library" wc="15064">https://github.com/Huiyicc/cpp-httplib</url>
#include &quot;nlohmann/json.hpp&quot; // <url id="cu9m7enftaebd5d06ua0" type="url" status="parsed" title="GitHub - nlohmann/json: JSON for Modern C++" wc="80311">https://github.com/nlohmann/json</url>
#include &quot;base64.hpp&quot; // <url id="cu9m7enftaebd5d06uag" type="url" status="parsed" title="GitHub - tobiaslocker/base64: A modern C++ base64 encoder / decoder" wc="2293">https://github.com/tobiaslocker/base64</url>

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string csvPath = &quot;./test.csv&quot;;
    const std::string outputCsvPath = &quot;./out.csv&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

    // Perform Base64 encoding
    std::ifstream file(csvPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; &quot;Error reading file.&quot; &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedCsv = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj[&quot;csv&quot;] = encodedCsv;
    std::string body = jsonObj.dump();

    // Call the API
    auto response = client.Post(&quot;/time-series-anomaly-detection&quot;, headers, body, &quot;application/json&quot;);
    // Process the response data
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        // Save the data
        encodedCsv = result[&quot;csv&quot;];
        std::string decodedString = base64::from_base64(encodedCsv);
        std::vector&lt;unsigned char&gt; decodedCsv(decodedString.begin(), decodedString.end());
        std::ofstream outputCsv(outputCsvPath, std::ios::binary | std::ios::out);
        if (outputCsv.is_open()) {
            outputCsv.write(reinterpret_cast&lt;char*&gt;(decodedCsv.data()), decodedCsv.size());
            outputCsv.close();
            std::cout &lt;&lt; &quot;Output time-series data saved at &quot; &lt;&lt; outputCsvPath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; outputCsvPath &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; &quot;Failed to send HTTP request.&quot; &lt;&lt; std::endl;
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
        String API_URL = &quot;http://localhost:8080/time-series-anomaly-detection&quot;;
        String csvPath = &quot;./test.csv&quot;;
        String outputCsvPath = &quot;./out.csv&quot;;

        // Base64 encode the local CSV file
        File file = new File(csvPath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String csvData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;csv&quot;, csvData);

        // Create an instance of OkHttpClient
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get(&quot;application/json; charset=utf-8&quot;);
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // Call the API and handle the response data
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get(&quot;result&quot;);

                // Save the returned data
                String base64Csv = result.get(&quot;csv&quot;).asText();
                byte[] csvBytes = Base64.getDecoder().decode(base64Csv);
                try (FileOutputStream fos = new FileOutputStream(outputCsvPath)) {
                    fos.write(csvBytes);
                }
                System.out.println(&quot;Output time-series data saved at &quot; + outputCsvPath);
            } else {
                System.err.println(&quot;Request failed with code: &quot; + response.code());
            }
        }
    }
}
</code></pre></details>

<details><summary>Go</summary>

<pre><code class="language-go">package main

import (
    &quot;bytes&quot;
    &quot;encoding/base64&quot;
    &quot;encoding/json&quot;
    &quot;fmt&quot;
    &quot;io/ioutil&quot;
    &quot;net/http&quot;
)

func main() {
    API_URL := &quot;http://localhost:8080/time-series-anomaly-detection&quot;
    csvPath := &quot;./test.csv&quot;;
    outputCsvPath := &quot;./out.csv&quot;;

    // Read the CSV file and encode it with Base64
    csvBytes, err := ioutil.ReadFile(csvPath)
    if err != nil {
        fmt.Println(&quot;Error reading CSV file:&quot;, err)
        return
    }
    csvData := base64.StdEncoding.EncodeToString(csvBytes)

    payload := map[string]string{&quot;csv&quot;: csvData} // Base64-encoded file content
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println(&quot;Error marshaling payload:&quot;, err)
        return
    }

    // Call the API
    client := &amp;http.Client{}
    req, err := http.NewRequest(&quot;POST&quot;, API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println(&quot;Error creating request:&quot;, err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println(&quot;Error sending request:&quot;, err)
        return
    }
    defer res.Body.Close()

    // Process the response data
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println(&quot;Error reading response body:&quot;, err)
        return
    }
    type Response struct {
        Result struct {
            Csv string `json:&quot;csv&quot;`
        } `json:&quot;result&quot;`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println(&quot;Error unmarshaling response body:&quot;, err)
        return
    }

    // Decode the Base64-encoded CSV data and save it as a file
    outputCsvData, err := base64.StdEncoding.DecodeString(respData.Result.Csv)
    if err != nil {
        fmt.Println(&quot;Error decoding Base64 CSV data:&quot;, err)
        return
    }
    err = ioutil.WriteFile(outputCsvPath, outputCsvData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing CSV to file:&quot;, err)
        return
    }
    fmt.Printf(&quot;Output time-series data saved at %s.csv&quot;, outputCsvPath)
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
    static readonly string API_URL = "http://localhost:8080/time-series-anomaly-detection";
    static readonly string csvPath = "./test.csv";
    static readonly string outputCsvPath = "./out.csv";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // Encode the local CSV file using Base64
        byte[] csvBytes = File.ReadAllBytes(csvPath);
        string csvData = Convert.ToBase64String(csvBytes);

        var payload = new JObject{ { "csv", csvData } }; // Base64 encoded file content
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        // Call the API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // Process the response data
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        // Save the CSV file
        string base64Csv = jsonResponse["result"]["csv"].ToString();
        byte[] outputCsvBytes = Convert.FromBase64String(base64Csv);
        File.WriteAllBytes(outputCsvPath, outputCsvBytes);
        Console.WriteLine($"Output time-series data saved at {outputCsvPath}");
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/time-series-anomaly-detection';
const csvPath = &quot;./test.csv&quot;;
const outputCsvPath = &quot;./out.csv&quot;;

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'csv': encodeFileToBase64(csvPath)  // Base64 encoded file content
  })
};

// Read the csv file and convert it to Base64
function encodeFileToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

axios.request(config)
.then((response) =&gt; {
    const result = response.data[&quot;result&quot;];

    // Save the csv file
    const csvBuffer = Buffer.from(result[&quot;csv&quot;], 'base64');
    fs.writeFile(outputCsvPath, csvBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output time-series data saved at ${outputCsvPath}`);
    });
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/time-series-anomaly-detection&quot;; // Service URL
$csv_path = &quot;./test.csv&quot;;
$output_csv_path = &quot;./out.csv&quot;;

// Base64 encode the local CSV file
$csv_data = base64_encode(file_get_contents($csv_path));
$payload = array(&quot;csv&quot; =&gt; $csv_data); // Base64 encoded file content

// Call the API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// Handle the response data
$result = json_decode($response, true)[&quot;result&quot;];

file_put_contents($output_csv_path, base64_decode($result[&quot;csv&quot;]));
echo &quot;Output time-series data saved at &quot; . $output_csv_path . &quot;\n&quot;;

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate method to deploy the model production line according to your needs, and then proceed with subsequent AI application integration.

## 4. Secondary Development
If the default model weights provided by the general time-series anomaly detection production line do not meet your accuracy or speed requirements in your scenario, you can try to further <b>fine-tune</b> the existing model using <b>your own specific domain or application scenario data</b> to improve the recognition effect of the general time-series anomaly detection production line in your scenario.

### 4.1 Model Fine-Tuning
Since the general time-series anomaly detection production line includes a time-series anomaly detection module, if the effect of the model production line is not as expected, you need to refer to the [Secondary Development](../../../module_usage/tutorials/time_series_modules/time_series_anomaly_detection.en.md#四二次开发) section in the [Time-Series Prediction Module Development Tutorial](../../../module_usage/tutorials/time_series_modules/time_series_anomaly_detection.en.md) to fine-tune the time-series anomaly detection model using your private dataset.

### 4.2 Model Application
After you complete the fine-tuning training with your private dataset, you can obtain the local model weight file.

If you need to use the fine-tuned model weights, simply modify the production line configuration file by filling in the local path of the fine-tuned model weights in the `model_dir` of the production line configuration file:

```yaml
pipeline_name: ts_anomaly_detection

SubModules:
  TSAnomalyDetection:
    module_name: ts_anomaly_detection
    model_name: DLinear_ad
    model_dir: null  # Can be modified to the local path of the fine-tuned model
    batch_size: 1
```

Then, you can load the modified pipeline configuration file by referring to the command-line method or the Python script method in the local experience.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. You can seamlessly switch between different hardware devices by simply modifying the `--device` parameter.

For example, if you are using Ascend NPU for inference in the time-series anomaly detection pipeline, the Python command you would use is:

```bash
paddlex --pipeline ts_anomaly_detection --input ts_ad.csv --device npu:0
```

If you want to use the universal time-series anomaly detection pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
