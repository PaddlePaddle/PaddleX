---
comments: true
---

# Time Series Forecasting Pipeline Tutorial

## 1. Introduction to the General Time Series Forecasting Pipeline
Time series forecasting is a technique that utilizes historical data to predict future trends by analyzing the patterns of change in time series data. It is widely applied in fields such as financial markets, weather forecasting, and sales prediction. Time series forecasting often employs statistical methods or deep learning models (e.g., LSTM, ARIMA), capable of handling temporal dependencies in data to provide accurate predictions, assisting decision-makers in better planning and response. This technology plays a crucial role in various industries, including energy management, supply chain optimization, and market analysis.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/time_series/03.png">

<b>The General Time Series Forecasting Pipeline includes a time series forecasting module. If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.</b>

<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>MSE</th>
<th>MAE</th>
<th>Model Storage Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>DLinear</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/DLinear_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/DLinear_pretrained.pdparams">Trained Model</a></td>
<td>0.382</td>
<td>0.394</td>
<td>72K</td>
</tr>
<tr>
<td>NLinear</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/NLinear_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/NLinear_pretrained.pdparams">Trained Model</a></td>
<td>0.386</td>
<td>0.392</td>
<td>40K</td>
</tr>
<tr>
<td>Nonstationary</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Nonstationary_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Nonstationary_pretrained.pdparams">Trained Model</a></td>
<td>0.600</td>
<td>0.515</td>
<td>55.5 M</td>
</tr>
<tr>
<td>PatchTST</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PatchTST_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PatchTST_pretrained.pdparams">Trained Model</a></td>
<td>0.385</td>
<td>0.397</td>
<td>2.0M</td>
</tr>
<tr>
<td>RLinear</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RLinear_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RLinear_pretrained.pdparams">Trained Model</a></td>
<td>0.384</td>
<td>0.392</td>
<td>40K</td>
</tr>
<tr>
<td>TiDE</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TiDE_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TiDE_pretrained.pdparams">Trained Model</a></td>
<td>0.405</td>
<td>0.412</td>
<td>31.7M</td>
</tr>
<tr>
<td>TimesNet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_pretrained.pdparams">Trained Model</a></td>
<td>0.417</td>
<td>0.431</td>
<td>4.9M</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are measured on <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar">ETTH1</a>. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## 2. Quick Start
The pre-trained model pipelines provided by PaddleX allow for quick experience of their effects. You can experience the effects of the General Time Series Forecasting Pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience the General Time Series Forecasting Pipeline online](https://aistudio.baidu.com/community/app/105706/webUI?source=appCenter) using the demo provided by the official team, for example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/time_series/04.png">

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to <b>fine-tune the model within the pipeline online</b>.

Note: Due to the close relationship between time series data and scenarios, the official built-in models for online time series tasks are scenario-specific and not universal. Therefore, the experience mode does not support using arbitrary files to experience the effects of the official model solutions. However, after training a model with your own scenario data, you can select your trained model solution and use data from the corresponding scenario for online experience.

### 2.2 Local Experience

Before using the general time-series forecasting pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience
You can quickly experience the time-series forecasting pipeline with a single command. Use [the test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv) and replace `--input` with your local path for prediction.

```bash
paddlex --pipeline ts_forecast --input ts_fc.csv --device gpu:0 --save_path ./output
```

The relevant parameter descriptions can be found in the parameter explanation section of [2.2.2 Python Script Integration](#222-python脚本方式集成).

After running, the result will be printed to the terminal as follows:

<details><summary>👉 Click to Expand</summary>

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

</details>

For the explanation of the running result parameters, you can refer to the result interpretation in [2.2.2 Python Script Integration](#222-python-script-integration).

The time-series file results are saved under `save_path`.


#### 2.2.2 Python Script Integration
The above command line is for quickly experiencing and viewing the results. Generally, in a project, it is often necessary to integrate through code. You can complete the fast inference of the production line with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="ts_forecast")

output = pipeline.predict(input="ts_fc.csv")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_csv(save_path="./output/") ## 保存csv格式结果
    res.save_to_json(save_path="./output/") ## 保存json格式结果
```

In the above Python script, the following steps are executed:

(1) Instantiate the pipeline object using `create_pipeline()`. The specific parameters are described as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>Pipeline name or path to pipeline config file, if it's set as a pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the pipeline (if set simultaneously with the <code>pipeline</code>, it takes precedence over the <code>pipeline</code>, and the pipeline name must match the <code>pipeline</code>).
</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Pipeline inference device. Supports specifying the specific GPU card number, such as "gpu:0", other hardware specific card numbers, such as "npu:0", CPU such as "cpu".</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available when the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the `ts_forecast` pipeline object to perform inference and prediction. This method returns a `generator`. The parameters and their descriptions for the `predict()` method are as follows:

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
  <li><b>str</b>: Local path to the time-series file (e.g., <code>/root/data/ts.csv</code>); <b>URL link</b>, such as the network URL of the time-series file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv">Example</a>; <b>Local directory</b>, which should contain the time-series to be predicted (e.g., <code>/root/data/</code>).</li>
  <li><b>List</b>: Elements of the list must be of the above types, such as <code>[pandas.DataFrame, pandas.DataFrame]</code>, <code>["/root/data/ts1.csv", "/root/data/ts2.csv"]</code>, <code>["/root/data1", "/root/data2"]</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for pipeline inference.</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>CPU</b>: Use CPU for inference (e.g., <code>cpu</code>).</li>
  <li><b>GPU</b>: Use the specified GPU for inference (e.g., <code>gpu:0</code> for the first GPU).</li>
  <li><b>NPU</b>: Use the specified NPU for inference (e.g., <code>npu:0</code> for the first NPU).</li>
  <li><b>XPU</b>: Use the specified XPU for inference (e.g., <code>xpu:0</code> for the first XPU).</li>
  <li><b>MLU</b>: Use the specified MLU for inference (e.g., <code>mlu:0</code> for the first MLU).</li>
  <li><b>DCU</b>: Use the specified DCU for inference (e.g., <code>dcu:0</code> for the first DCU).</li>
  <li><b>None</b>: If set to <code>None</code>, the default value used during pipeline initialization will be applied. During initialization, the local GPU 0 will be prioritized; if unavailable, the CPU will be used.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

(3) Process the prediction results. Each sample's prediction result is of type `dict` and supports operations such as printing, saving to a `csv` file, and saving to a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Explanation</th>
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
<td>Specifies the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save the result. When a directory is specified, the saved file name will match the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output <code>JSON</code> data, making it more readable. Only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_csv()</code></td>
<td>Save the result as a CSV file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save the result. Supports both directory and file paths</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the result to the terminal. The printed content is explained as follows:

    - `input_path`: `(str)` The input path of the time-series file to be predicted.

    - `forecast`: `(Pandas.DataFrame)` The time-series prediction result, including future time points and corresponding predicted values.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_ts_basename}_res.json`. If a file is specified, the result will be saved directly to that file. Since JSON files do not support saving NumPy arrays, `numpy.array` types will be converted to lists.

- Calling the `save_to_csv()` method will save the result to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_ts_basename}_res.csv`. If a file is specified, the result will be saved directly to that file.

* In addition, it also supports obtaining prediction results in different formats through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Obtain the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>csv</code></td>
<td rowspan="2">Obtain the prediction result in <code>csv</code> format</td>
</tr>
</table>

- The prediction result obtained through the `json` attribute is of type `dict`, and its content is consistent with the result saved by the `save_to_json()` method.
- The `csv` attribute returns a `Pandas.DataFrame` type data, which contains the time-series prediction results.

In addition, you can obtain the ts_forecast production line configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```bash
paddlex --get_pipeline_config ts_forecast --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the time-series forecasting pipeline by simply modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/ts_forecast.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/ts_forecast.yaml")
output = pipeline.predict("ts_fc.csv")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_csv("./output/") ## 保存csv格式结果
    res.save_to_json("./output/") ## 保存json格式结果
```

<b>Note:</b> The parameters in the configuration file are the initialization parameters for the production line. If you wish to change the initialization parameters for the `ts_forecasts` production line, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in a configuration file, simply specify the path to the configuration file with `--pipeline`.

## 3. Development Integration/Deployment
If the production line meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the production line directly into your Python project, you can refer to the example code in [2.2.2 Python Script Integration](#222-python脚本方式集成).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In practical production environments, many applications have strict performance requirements for deployment strategies, especially in terms of response speed, to ensure the efficient operation of the system and a smooth user experience. To this end, PaddleX provides a high-performance inference plugin, which aims to deeply optimize the performance of model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed information on high-performance inference, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service-Oriented Deployment</b>: Service-oriented deployment is a common form of deployment in practical production environments. By encapsulating the inference functionality into a service, clients can access these services via network requests to obtain inference results. PaddleX supports multiple service-oriented deployment solutions for production lines. For detailed information on service-oriented deployment, please refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references for basic service-oriented deployment and examples of multi-language service calls:

<details><summary>API Reference</summary>

<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and the response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
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
<th>Description</th>
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
<p>Perform time-series forecasting.</p>
<p><code>POST /time-series-forecasting</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
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
<li>When the request is processed successfully, the <code>result</code> in the response body has the following attributes:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>csv</code></td>
<td><code>string</code></td>
<td>The time-series forecasting result in CSV format. Encoded in UTF-8+Base64.</td>
</tr>
</tbody>
</table>
<p>An example of <code>result</code> is as follows:</p>
<pre><code class="language-json">{
&quot;csv&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>Multi-Language Service Call Examples</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/time-series-forecasting"  # Service URL
csv_path = "./test.csv"
output_csv_path = "./out.csv"

# Encode the local CSV file using Base64
with open(csv_path, "rb") as file:
    csv_bytes = file.read()
    csv_data = base64.b64encode(csv_bytes).decode("ascii")

payload = {"csv": csv_data}

# Call the API
response = requests.post(API_URL, json=payload)

# Process the returned data
assert response.status_code == 200
result = response.json()["result"]
with open(output_csv_path, "wb") as f:
    f.write(base64.b64decode(result["csv"]))
print(f"Output time-series data saved at {output_csv_path}")
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
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

    // Encode the CSV file using Base64
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
    auto response = client.Post("/time-series-forecasting", headers, body, "application/json");
    // Process the returned data
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse["result"];

        // Save the data
        std::string decodedString;
        encodedCsv = result["csv"];
        decodedString = base64::from_base64(encodedCsv);
        std::vector&lt;unsigned char&gt; decodedCsv(decodedString.begin(), decodedString.end());
        std::ofstream outputCsv(outputCsvPath, std::ios::binary | std::ios::out);
        if (outputCsv.is_open()) {
            outputCsv.write(reinterpret_cast&lt;char*&gt;(decodedCsv.data()), decodedCsv.size());
            outputCsv.close();
            std::cout &lt;&lt; "Output time-series data saved at " &lt;&lt; outputCsvPath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; "Unable to open file for writing: " &lt;&lt; outputCsvPath &lt;&lt; std::endl;
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
        String API_URL = &quot;http://localhost:8080/time-series-forecasting&quot;;
        String csvPath = &quot;./test.csv&quot;;
        String outputCsvPath = &quot;./out.csv&quot;;

        // Encode the local CSV file using Base64
        File file = new File(csvPath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String csvData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;csv&quot;, csvData);

        // Create an OkHttpClient instance
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get(&quot;application/json; charset=utf-8&quot;);
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // Call the API and process the response data
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
    API_URL := &quot;http://localhost:8080/time-series-forecasting&quot;
    csvPath := &quot;./test.csv&quot;;
    outputCsvPath := &quot;./out.csv&quot;;

    // Read the csv file and encode it in Base64
    csvBytes, err := ioutil.ReadFile(csvPath)
    if err != nil {
        fmt.Println(&quot;Error reading csv file:&quot;, err)
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

    // Decode the Base64-encoded csv data and save it as a file
    outputCsvData, err := base64.StdEncoding.DecodeString(respData.Result.Csv)
    if err != nil {
        fmt.Println(&quot;Error decoding base64 csv data:&quot;, err)
        return
    }
    err = ioutil.WriteFile(outputCsvPath, outputCsvData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing csv to file:&quot;, err)
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
    static readonly string API_URL = "http://localhost:8080/time-series-forecasting";
    static readonly string csvPath = "./test.csv";
    static readonly string outputCsvPath = "./out.csv";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // Encode the local CSV file using Base64
        byte[] csvBytes = File.ReadAllBytes(csvPath);
        string csvData = Convert.ToBase64String(csvBytes);

        var payload = new JObject{ { "csv", csvData } }; // Base64-encoded file content
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        // Call the API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // Process the returned data
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

const API_URL = 'http://localhost:8080/time-series-forecasting';
const csvPath = &quot;./test.csv&quot;;
const outputCsvPath = &quot;./out.csv&quot;;

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
.then((response) =&gt; {
    const result = response.data[&quot;result&quot;];

    // Save the CSV file
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

$API_URL = &quot;http://localhost:8080/time-series-forecasting&quot;; // Service URL
$csv_path = &quot;./test.csv&quot;;
$output_csv_path = &quot;./out.csv&quot;;

// Encode the local CSV file in Base64
$csv_data = base64_encode(file_get_contents($csv_path));
$payload = array(&quot;csv&quot; =&gt; $csv_data); // Base64-encoded file content

// Call the API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// Process the response data
$result = json_decode($response, true)[&quot;result&quot;];

file_put_contents($output_csv_path, base64_decode($result[&quot;csv&quot;]));
echo &quot;Output time-series data saved at &quot; . $output_csv_path . &quot;\n&quot;;

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on the user's device, allowing the device to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed instructions, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md). You can choose the appropriate deployment method based on your needs to integrate the model pipeline into subsequent AI applications.

## 4. Custom Development
If the default model weights provided by the time-series forecasting pipeline are not satisfactory in terms of accuracy or speed for your specific scenario, you can attempt to further <b>fine-tune</b> the existing models using <b>your own domain-specific or application data</b> to improve the performance of the time-series forecasting pipeline in your scenario.

#### 4.1 Model Fine-Tuning
Since the general time-series forecasting pipeline includes a time-series forecasting module, if the pipeline's performance does not meet expectations, you need to refer to the [Custom Development](../../../module_usage/tutorials/time_series_modules/time_series_forecasting.en.md#four-custom-development) section in the [Time-Series Forecasting Module Development Tutorial](../../../module_usage/tutorials/time_series_modules/time_series_forecasting.en.md) to fine-tune the time-series forecasting model using your private dataset.

#### 4.2 Model Application
After completing fine-tuning with your private dataset, you will obtain the local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by filling in the local path of the fine-tuned model weights to the `model_dir` in the pipeline configuration file:

```yaml
pipeline_name: ts_forecast

SubModules:
  TSForecast:
    module_name: ts_forecast
    model_name: DLinear
    model_dir: null # 此处替换为您训练后得到的模型权重本地路径
    batch_size: 1
```

Subsequently, refer to the command line method or Python script method in the local experience section to load the modified production line configuration file.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware devices.

For example, if you are using Ascend NPU for inference in the time-series forecasting production line, the Python command you would use is:

```bash
paddlex --pipeline ts_forecast --input ts_fc.csv --device npu:0
```

If you want to use the General Time-Series Forecasting Pipeline on a wider range of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
