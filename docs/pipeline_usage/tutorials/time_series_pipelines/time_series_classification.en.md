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
<td>TimesNet_cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/TimesNet_cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/TimesNet_cls_pretrained.pdparams">Trained Model</a></td>
<td>87.5</td>
<td>792K</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are measured on the <a href="https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TEST.csv">UWaveGestureLibrary</a> dataset. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## 2. Quick Start
PaddleX provides pre-trained model pipelines that can be quickly experienced. You can experience the effects of the General Time Series Classification Pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience online](https://aistudio.baidu.com/community/app/105707/webUI?source=appCenter) the effects of the General Time Series Classification Pipeline using the official demo for recognition, for example:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/time_series/02.png">

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to <b>fine-tune the model in the pipeline online</b>.

Note: Due to the close relationship between time series data and scenarios, the official built-in model for online experience of time series tasks is only a model solution for a specific scenario and is not a general solution applicable to other scenarios. Therefore, the experience method does not support using arbitrary files to experience the effect of the official model solution. However, after training a model for your own scenario data, you can select your trained model solution and use data from the corresponding scenario for online experience.

### 2.2 Local Experience
Before using the General Time Series Classification Pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience
Experience the effects of the time series classification pipeline quickly with a single command:

Experience the image anomaly detection pipeline with a single command，Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv), and replace `--input` with the local path to perform prediction.

```bash
paddlex --pipeline ts_cls --input ts_cls.csv --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it is the time series classification pipeline.
--input: The local path or URL of the input sequence to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the second and third GPUs). You can also choose to use CPU (--device cpu).
```

When executing the above command, the default time series classification pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details><summary> 👉Click to Expand</summary>

<pre><code class="language-bash">paddlex --get_pipeline_yaml ts_cls
</code></pre>
<p>After execution, the time series classification pipeline configuration file will be saved in the current path. If you wish to customize the save location, you can execute the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code class="language-bash">paddlex --get_pipeline_config ts_cls --save_path ./my_path
</code></pre>
<p>After obtaining the pipeline configuration file, you can replace <code>--pipeline</code> with the configuration file save path to make the configuration file take effect. For example, if the configuration file save path is <code>./ts_ad.yaml</code>, simply execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./ts_cls.yaml --input ts_cls.csv --device gpu:0
</code></pre>
<p>In this command, parameters such as <code>--model</code> and <code>--device</code> are not required to be specified, as they will use the parameters defined in the configuration file. If these parameters are specified, the specified values will take precedence.</p></details>

After execution, the result is:

```bash
{'input_path': 'ts_cls.csv', 'classification':         classid     score
sample
0             0  0.617688}
```

#### 2.2.2 Python Script Integration

A few lines of code can complete rapid inference for production lines. Taking the General Time Series Classification Pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="ts_cls")

output = pipeline.predict("ts_cls.csv")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_csv("./output/")  # Save results in CSV format
```

The results obtained are the same as those from the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: Specific parameter descriptions are as follows:

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
<td>The name of the pipeline or the path to the pipeline configuration file. If it's a pipeline name, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for pipeline model inference. Supports: "gpu", "cpu".</td>
<td><code>str</code></td>
<td>"gpu"</td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference. Available only if the pipeline supports it.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
(2) Call the `predict` method of the pipeline object for inference: The `predict` method takes `x` as a parameter, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

<table>
<thead>
<tr>
<th>Parameter Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>Supports directly passing Python variables, such as numpy.ndarray representing image data.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing the path of the file to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing the URL of the file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv">Example</a>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing a local directory, which should contain files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td><code>dict</code></td>
<td>Supports passing a dictionary type, where the key needs to correspond to the specific task, e.g., "img" for image classification tasks, and the value of the dictionary supports the above data types, e.g., <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td><code>list</code></td>
<td>Supports passing a list, where the list elements need to be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>, <code>[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>
(3) Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained by iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

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

For example, if your configuration file is saved at `./my_path/ts_cls.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/ts_cls.yaml")
output = pipeline.predict("ts_cls.csv")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_csv("./output/")  # Save results in CSV format
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy in production, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for deployment performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins that deeply optimize model inference and pre/post-processing to significantly speed up the end-to-end process. Refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md) for detailed high-performance inference procedures.

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
<p>Primary operations provided by the service:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Classify time-series data.</p>
<p><code>POST /time-series-classification</code></p>
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
<li>When the request is processed successfully, the <code>result</code> in the response body has the following properties:</li>
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
<td><code>label</code></td>
<td><code>string</code></td>
<td>Class label.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>Class score.</td>
</tr>
</tbody>
</table>
<p>An example of <code>result</code> is as follows:</p>
<pre><code class="language-json">{
&quot;label&quot;: &quot;running&quot;,
&quot;score&quot;: 0.97
}
</code></pre></details>

<details><summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/time-series-classification&quot;
csv_path = &quot;./test.csv&quot;

with open(csv_path, &quot;rb&quot;) as file:
    csv_bytes = file.read()
    csv_data = base64.b64encode(csv_bytes).decode(&quot;ascii&quot;)

payload = {&quot;csv&quot;: csv_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
print(f&quot;label: {result['label']}, score: {result['score']}&quot;)
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // https://github.com/Huiyicc/cpp-httplib
#include &quot;nlohmann/json.hpp&quot; // https://github.com/nlohmann/json
#include &quot;base64.hpp&quot; // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string csvPath = &quot;./test.csv&quot;;

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

    auto response = client.Post(&quot;/time-series-classification&quot;, headers, body, &quot;application/json&quot;);
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];
        std::cout &lt;&lt; &quot;label: &quot; &lt;&lt; result[&quot;label&quot;] &lt;&lt; &quot;, score: &quot; &lt;&lt; result[&quot;score&quot;] &lt;&lt; std::endl;
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
        String API_URL = &quot;http://localhost:8080/time-series-classification&quot;;
        String csvPath = &quot;./test.csv&quot;;

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
                System.out.println(&quot;label: &quot; + result.get(&quot;label&quot;).asText() + &quot;, score: &quot; + result.get(&quot;score&quot;).asText());
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
    API_URL := &quot;http://localhost:8080/time-series-classification&quot;
    csvPath := &quot;./test.csv&quot;;

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
            Label string `json:&quot;label&quot;`
            Score string `json:&quot;score&quot;`
        } `json:&quot;result&quot;`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println(&quot;Error unmarshaling response body:&quot;, err)
        return
    }

    fmt.Printf(&quot;label: %s, score: %s\n&quot;, respData.Result.Label, respData.Result.Score)
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
    static readonly string API_URL = &quot;http://localhost:8080/time-series-classification&quot;;
    static readonly string csvPath = &quot;./test.csv&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] csveBytes = File.ReadAllBytes(csvPath);
        string csvData = Convert.ToBase64String(csveBytes);

        var payload = new JObject{ { &quot;csv&quot;, csvData } };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string label = jsonResponse[&quot;result&quot;][&quot;label&quot;].ToString();
        string score = jsonResponse[&quot;result&quot;][&quot;score&quot;].ToString();
        Console.WriteLine($&quot;label: {label}, score: {score}&quot;);
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/time-series-classification'
const csvPath = &quot;./test.csv&quot;;

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
    console.log(`label: ${result[&quot;label&quot;]}, score: ${result[&quot;score&quot;]}`);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/time-series-classification&quot;;
$csv_path = &quot;./test.csv&quot;;

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
echo &quot;label: &quot; . $result[&quot;label&quot;] . &quot;, score: &quot; . $result[&quot;score&quot;];

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computing and data processing capabilities on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. Refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.md) for detailed edge deployment procedures.
Choose the appropriate deployment method based on your needs to proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the General Time Series Classification Pipeline do not meet your requirements for accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the General Time Series Classification Pipeline includes a time series classification module, if the performance of the pipeline does not meet expectations, you need to refer to the [Customization](../../../module_usage/tutorials/time_series_modules/time_series_classification.en.md#iv-custom-development) section in the [Time Series Classification Module Tutorial](../../../module_usage/tutorials/time_series_modules/time_series_classification.en.md) to fine-tune the time series classification model using your private dataset.

### 4.2 Model Application
After fine-tuning the model with your private dataset, you will obtain local model weights.

To use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```python
......
Pipeline:
  model: TimesNet_cls  # Replace with the local path of the fine-tuned model
  device: "gpu"
  batch_size: 1
......
```
Then, refer to the command line or Python script methods in the local experience section to load the modified pipeline configuration file.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for time series classification pipeline inference, the Python command is:

```bash
paddlex --pipeline ts_cls --input ts_cls.csv --device gpu:0
``````
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` flag in the Python command as follows:

```bash
paddlex --pipeline ts_cls --input ts_cls.csv --device npu:0
```

If you intend to use the General Time Series Classification Pipeline on a wider range of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
