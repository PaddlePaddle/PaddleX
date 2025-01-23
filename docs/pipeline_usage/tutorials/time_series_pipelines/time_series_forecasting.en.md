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
Before using the General Time Series Forecasting Pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

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

<details><summary> 👉Click to expand</summary>

<pre><code class="language-bash">paddlex --get_pipeline_config ts_fc --save_path ./my_path
</code></pre>
<p>After obtaining the pipeline configuration file, you can replace <code>--pipeline</code> with the configuration file save path to make the configuration file take effect. For example, if the configuration file save path is <code>./ts_fc.yaml</code>, simply execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./ts_fc.yaml --input ts_fc.csv --device gpu:0
</code></pre>
<p>Here, parameters such as <code>--model</code> and <code>--device</code> do not need to be specified, as they will use the parameters in the configuration file. If parameters are still specified, the specified parameters will take precedence.</p></details>

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
<td>The name of the production line or the path to the production line configuration file. If it is the name of the production line, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for production line model inference. Supports: "gpu", "cpu".</td>
<td><code>str</code></td>
<td>"gpu"</td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available when the production line supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
（2）Invoke the `predict` method of the  production line object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

<table>
<thead>
<tr>
<th>Parameter Type</th>
<th>Parameter Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>Supports directly passing in Python variables, such as numpy.ndarray representing image data.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in the path of the file to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in the URL of the file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv">Example</a>.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in a local directory, which should contain files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td>dict</td>
<td>Supports passing in a dictionary type, where the key needs to correspond to a specific task, such as "img" for image classification tasks. The value of the dictionary supports the above types of data, for example: <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td>list</td>
<td>Supports passing in a list, where the list elements need to be of the above types of data, such as <code>[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>
（3）Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Method Parameters</th>
</tr>
</thead>
<tbody>
<tr>
<td>print</td>
<td>Prints results to the terminal</td>
<td><code>- format_json</code>: bool, whether to format the output content with json indentation, default is True;<br/><code>- indent</code>: int, json formatting setting, only valid when format_json is True, default is 4;<br/><code>- ensure_ascii</code>: bool, json formatting setting, only valid when format_json is True, default is False;</td>
</tr>
<tr>
<td>save_to_json</td>
<td>Saves results as a json file</td>
<td><code>- save_path</code>: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;<br/><code>- indent</code>: int, json formatting setting, default is 4;<br/><code>- ensure_ascii</code>: bool, json formatting setting, default is False;</td>
</tr>
<tr>
<td>save_to_img</td>
<td>Saves results as an image file</td>
<td><code>- save_path</code>: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;</td>
</tr>
</tbody>
</table>
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

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for deployment strategy performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end process acceleration. For detailed high-performance inference procedures, refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Serving</b>: Serving is a common deployment strategy in real-world production environments. By encapsulating inference functions into services, clients can access these services via network requests to obtain inference results. PaddleX supports various solutions for serving pipelines. For detailed pipeline serving procedures, please refer to the [PaddleX Pipeline Serving Guide](../../../pipeline_deploy/serving.md).

Below are the API reference and multi-language service invocation examples for the basic serving solution:

<details><summary>API Reference</summary>

<p>For primary operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>The request body and the response body are both JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body properties are as follows:</li>
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
<td>UUID for the request.</td>
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
<td>Operation result.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the response body properties are as follows:</li>
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
<td>UUID for the request.</td>
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
<p>Primary operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Performs time-series forecasting.</p>
<p><code>POST /time-series-forecasting</code></p>
<ul>
<li>The request body properties are as follows:</li>
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
<td>The URL of a CSV file accessible by the server or the Base64 encoded result of the CSV file content. The CSV file must be encoded in UTF-8.</td>
<td>Yes</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> of the response body has the following properties:</li>
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

<details><summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/time-series-forecasting&quot;
csv_path = &quot;./test.csv&quot;
output_csv_path = &quot;./out.csv&quot;

with open(csv_path, &quot;rb&quot;) as file:
    csv_bytes = file.read()
    csv_data = base64.b64encode(csv_bytes).decode(&quot;ascii&quot;)

payload = {&quot;csv&quot;: csv_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_csv_path, &quot;wb&quot;) as f:
    f.write(base64.b64decode(result[&quot;csv&quot;]))
print(f&quot;Output time-series data saved at  {output_csv_path}&quot;)
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // https://github.com/Huiyicc/cpp-httplib
#include &quot;nlohmann/json.hpp&quot; // https://github.com/nlohmann/json
#include &quot;base64.hpp&quot; // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string csvPath = &quot;./test.csv&quot;;
    const std::string outputCsvPath = &quot;./out.csv&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

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

    auto response = client.Post(&quot;/time-series-forecasting&quot;, headers, body, &quot;application/json&quot;);
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedCsv = result[&quot;csv&quot;];
        decodedString = base64::from_base64(encodedCsv);
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
        String API_URL = &quot;http://localhost:8080/time-series-forecasting&quot;;
        String csvPath = &quot;./test.csv&quot;;
        String outputCsvPath = &quot;./out.csv&quot;;

        File file = new File(csvPath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String csvData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;csv&quot;, csvData);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get(&quot;application/json; charset=utf-8&quot;);
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get(&quot;result&quot;);

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

    csvBytes, err := ioutil.ReadFile(csvPath)
    if err != nil {
        fmt.Println(&quot;Error reading csv file:&quot;, err)
        return
    }
    csvData := base64.StdEncoding.EncodeToString(csvBytes)

    payload := map[string]string{&quot;csv&quot;: csvData}
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println(&quot;Error marshaling payload:&quot;, err)
        return
    }

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
    static readonly string API_URL = &quot;http://localhost:8080/time-series-forecasting&quot;;
    static readonly string csvPath = &quot;./test.csv&quot;;
    static readonly string outputCsvPath = &quot;./out.csv&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] csvBytes = File.ReadAllBytes(csvPath);
        string csvData = Convert.ToBase64String(csvBytes);

        var payload = new JObject{ { &quot;csv&quot;, csvData } };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Csv = jsonResponse[&quot;result&quot;][&quot;csv&quot;].ToString();
        byte[] outputCsvBytes = Convert.FromBase64String(base64Csv);
        File.WriteAllBytes(outputCsvPath, outputCsvBytes);
        Console.WriteLine($&quot;Output time-series data saved at {outputCsvPath}&quot;);
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/time-series-forecasting'
const csvPath = &quot;./test.csv&quot;;
const outputCsvPath = &quot;./out.csv&quot;;

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
.then((response) =&gt; {
    const result = response.data[&quot;result&quot;];

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

$API_URL = &quot;http://localhost:8080/time-series-forecasting&quot;;
$csv_path = &quot;./test.csv&quot;;
$output_csv_path = &quot;./out.csv&quot;;

$csv_data = base64_encode(file_get_contents($csv_path));
$payload = array(&quot;csv&quot; =&gt; $csv_data);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)[&quot;result&quot;];

file_put_contents($output_csv_path, base64_decode($result[&quot;csv&quot;]));
echo &quot;Output time-series data saved at &quot; . $output_csv_path . &quot;\n&quot;;

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computing and data processing functions on user devices themselves, enabling devices to directly process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.md).
Choose the appropriate deployment method for your model pipeline based on your needs, and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the General Time Series Forecasting Pipeline do not meet your requirements in terms of accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the pipeline in your scenario.

#### 4.1 Model Fine-tuning
Since the General Time Series Forecasting Pipeline includes a time series forecasting module, if the performance of the pipeline does not meet expectations, you need to refer to the [Customization](../../../module_usage/tutorials/time_series_modules/time_series_forecasting.en.md#iv-custom-development) section in the [Time Series Forecasting Module Development Tutorial](../../../module_usage/tutorials/time_series_modules/time_series_forecasting.en.md) and use your private dataset to fine-tune the time series forecasting model.

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
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for inference with the time series forecasting pipeline, the Python command would be:

```bash
paddlex --pipeline ts_fc --input ts_fc.csv --device gpu:0
``````
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the Python command to `npu:0`:

```bash
paddlex --pipeline ts_fc --input ts_fc.csv --device npu:0
```
If you want to use the General Time Series Forecasting Pipeline on a wider range of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
