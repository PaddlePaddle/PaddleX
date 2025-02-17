---
comments: true
---

# Multilingual Speech Recognition pipeline User Guide

## 1. Introduction to Multilingual Speech Recognition pipeline
Speech recognition is an advanced tool that can automatically convert spoken languages into corresponding text or commands. This technology plays an important role in various fields such as intelligent customer service, voice assistants, and meeting records. Multilingual speech recognition supports automatic language detection and recognition of multiple languages.

<p><b>Multilingual Speech Recognition Models (Optional):</b></p>
<table>
   <tr>
     <th>Model</th>
     <th>Model Download Link</th>
     <th>Training Data</th>
     <th>Model Size</th>
     <th>Word Error Rate</th>
     <th>Introduction</th>
   </tr>
   <tr>
     <td>whisper_large</td>
     <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_large.tar">whisper_large</a></td>
     <td>680kh</td>
     <td>5.8G</td>
     <td>2.7 (Librispeech)</td>
     <td rowspan="5">Whisper is a multilingual automatic speech recognition model developed by OpenAI, known for its high precision and robustness. It features an end-to-end architecture and can handle noisy audio environments, making it suitable for applications such as voice assistants and real-time subtitles.</td>
   </tr>
   <tr>
     <td>whisper_medium</td>
     <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_medium.tar">whisper_medium</a></td>
     <td>680kh</td>
     <td>2.9G</td>
     <td>-</td>
   </tr>
   <tr>
     <td>whisper_small</td>
     <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_small.tar">whisper_small</a></td>
     <td>680kh</td>
     <td>923M</td>
     <td>-</td>
   </tr>
   <tr>
     <td>whisper_base</td>
     <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_base.tar">whisper_base</a></td>
     <td>680kh</td>
     <td>277M</td>
     <td>-</td>
   </tr>
   <tr>
     <td>whisper_tiny</td>
     <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_tiny.tar">whisper_tiny</a></td>
     <td>680kh</td>
     <td>145M</td>
     <td>-</td>
   </tr>
 </table>

## 2. Quick Start
PaddleX supports experiencing the multilingual speech recognition pipeline locally using the command line or Python.

Before using the multilingual speech recognition pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 2.1 Local Experience

#### 2.1.1 Command Line Experience
You can quickly experience the effect of the document image preprocessing pipeline with a single command. Use the [example audio](https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav) and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline multilingual_speech_recognition \
        --input zh.wav \
        --save_path ./output \
        --device gpu:0
```

The relevant parameter descriptions can be found in the parameter descriptions in [2.1.2 Integration via Python Script]().

After running, the result will be printed to the terminal, as follows:

```bash
{'input_path': 'zh.wav', 'result': {'text': '我认为跑步最重要的就是给我带来了身体健康', 'segments': [{'id': 0, 'seek': 0, 'start': 0.0, 'end': 2.0, 'text': '我认为跑步最重要的就是', 'tokens': [50364, 1654, 7422, 97, 13992, 32585, 31429, 8661, 24928, 1546, 5620, 50464, 50464, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 50564], 'temperature': 0, 'avg_logprob': -0.22779104113578796, 'compression_ratio': 0.28169014084507044, 'no_speech_prob': 0.026114309206604958}, {'id': 1, 'seek': 200, 'start': 2.0, 'end': 31.0, 'text': '给我带来了身体健康', 'tokens': [50364, 49076, 4845, 99, 34912, 19847, 29485, 44201, 6346, 115, 51814], 'temperature': 0, 'avg_logprob': -0.21976988017559052, 'compression_ratio': 0.23684210526315788, 'no_speech_prob': 0.009023111313581467}], 'language': 'zh'}}
```

The explanation of the result parameters can refer to the result explanation in [2.1.2 Integration with Python Script](#212-integration-with-python-script).

#### 2.1.2 Integration with Python Script

The above command line is for quickly experiencing and viewing the effect. Generally speaking, in a project, it is often necessary to integrate through code. You can complete the rapid inference of the pipeline with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="multilingual_speech_recognition")
output = pipeline.predict(input="zh.wav")

for res in output:
    res.print()
    res.save_to_json(save_path="./output/")
```

In the above Python script, the following steps are executed:

(1) The <code>multilingual_speech_recognition</code> pipeline object is instantiated through <code>create_pipeline()</code>. The specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The inference device for the pipeline. It supports specifying the specific card number of the GPU, such as "gpu:0", the specific card number of other hardware, such as "npu:0", and the CPU, such as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
</tbody>
</table>

(2) The <code>predict()</code> method of the <code>multilingual_speech_recognition</code> pipeline object is called to perform inference and prediction. This method will return a <code>generator</code>. Below are the parameters and their descriptions for the <code>predict()</code> method:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted</td>
<td><code>str</code></td>
<td>
<ul>
  <li><b>File path</b>, such as the local path of an audio file: <code>/root/data/audio.wav</code></li>
  <li><b>URL link</b>, such as the network URL of an audio file: <a href="https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav">Example</a></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The inference device for the pipeline</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>CPU</b>: such as <code>cpu</code> indicates using the CPU for inference;</li>
  <li><b>GPU</b>: such as <code>gpu:0</code> indicates using the first GPU for inference;</li>
  <li><b>NPU</b>: such as <code>npu:0</code> indicates using the first NPU for inference;</li>
  <li><b>XPU</b>: such as <code>xpu:0</code> indicates using the first XPU for inference;</li>
  <li><b>MLU</b>: such as <code>mlu:0</code> indicates using the first MLU for inference;</li>
  <li><b>DCU</b>: such as <code>dcu:0</code> indicates using the first DCU for inference;</li>
  <li><b>None</b>: If set to <code>None</code>, the default value initialized for the pipeline will be used. During initialization, the local GPU device 0 will be prioritized. If it is not available, the CPU device will be used.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

(3) Process the prediction results. The prediction result for each sample is of the `dict` type and supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. When it is a directory, the saved file name is consistent with the input file type naming</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
</table>

- Calling the `print()` method will print the result to the terminal, with the printed content explained as follows:

    - `input_path`: The path where the input audio is stored
    - `result`: Recognition result
        -  `text`: The text result of speech recognition
        -  `segments`: The result text with timestamps
            * `id`: ID
            * `seek`: Audio segment pointer
            * `start`: Segment start time
            * `end`: Segment end time
            * `text`: Text recognized in the segment
            * `tokens`: Token IDs of the segment text
            * `temperature`: Speed variation ratio
            * `avg_logprob`: Average log probability
            * `compression_ratio`: Compression ratio
            * `no_speech_prob`: Non-speech probability
        - `language`: Recognized language

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_audio_basename}.json`; if specified as a file, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, the `numpy.array` types will be converted to lists.

* Additionally, it also supports obtaining visualized images and prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Get the predicted <code>json</code> format result</td>
</tr>
</table>

- The prediction result obtained by the `json` attribute is a dict type of data, with content consistent with the content saved by calling the `save_to_json()` method.

In addition, you can obtain the multilingual_speech_recognition pipeline configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config multilingual_speech_recognition --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the `multilingual_speech_recognition` pipeline. Simply modify the value of the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file. An example is as follows:

For example, if your configuration file is saved at `./my_path/multilingual_speech_recognition.yaml`, you just need to execute:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="multilingual_speech_recognition")
output = pipeline.predict(input="zh.wav")

for res in output:
    res.print()
    res.save_to_json(save_path="./output/")
```

<b>Note:</b> The parameters in the configuration file are the initialization parameters for the pipeline. If you want to change the initialization parameters of the <code>multilingual_speech_recognition</code> pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in a configuration file, simply specify the path of the configuration file with <code>--pipeline</code>.

<details><summary>Multilingual Service Call Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/video-classification&quot; # Service URL
video_path = &quot;./demo.mp4&quot;
output_video_path = &quot;./out.mp4&quot;

# Encode local video to Base64
with open(video_path, &quot;rb&quot;) as file:
    video_bytes = file.read()
    video_data = base64.b64encode(video_bytes).decode(&quot;ascii&quot;)

payload = {&quot;video&quot;: video_data}  # Base64 encoded file content or video URL

# Call API
response = requests.post(API_URL, json=payload)

# Process API response
assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_video_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;video&quot;]))
print(f&quot;Output video saved at {output_video_path}&quot;)
print(&quot;\nCategories:&quot;)
print(result[&quot;categories&quot;])
</code></pre></details>
<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // <url id="cu9qjr7f2ena5466v3o0" type="url" status="parsed" title="GitHub - Huiyicc/cpp-httplib: A C++ header-only HTTP/HTTPS server and client library" wc="15064">https://github.com/Huiyicc/cpp-httplib</url> 
#include &quot;nlohmann/json.hpp&quot; // <url id="cu9qjr7f2ena5466v3og" type="url" status="parsed" title="GitHub - nlohmann/json: JSON for Modern C++" wc="80311">https://github.com/nlohmann/json</url> 
#include &quot;base64.hpp&quot; // <url id="cu9qjr7f2ena5466v3p0" type="url" status="parsed" title="GitHub - tobiaslocker/base64: A modern C++ base64 encoder / decoder" wc="2293">https://github.com/tobiaslocker/base64</url> 

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string videoPath = &quot;./demo.mp4&quot;;
    const std::string outputImagePath = &quot;./out.mp4&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

    // Encode local video to Base64
    std::ifstream file(videoPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; &quot;Error reading file.&quot; &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj[&quot;video&quot;] = encodedImage;
    std::string body = jsonObj.dump();

    // Call API
    auto response = client.Post(&quot;/video-classification&quot;, headers, body, &quot;application/json&quot;);
    // Process API response
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedImage = result[&quot;video&quot;];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; &quot;Output video saved at &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        }

        auto categories = result[&quot;categories&quot;];
        std::cout &lt;&lt; &quot;\nCategories:&quot; &lt;&lt; std::endl;
        for (const auto&amp; category : categories) {
            std::cout &lt;&lt; category &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; &quot;Failed to send HTTP request.&quot; &lt;&lt; std::endl;
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
        String API_URL = &quot;http://localhost:8080/video-classification&quot;; // Service URL
        String videoPath = &quot;./demo.mp4&quot;; // Local video
        String outputImagePath = &quot;./out.mp4&quot;; // Output video

        // Encode local video to Base64
        File file = new File(videoPath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String videoData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;video&quot;, videoData); // Base64 encoded file content or video URL

        // Create OkHttpClient instance
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get(&quot;application/json; charset=utf-8&quot;);
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // Call API and process API response
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get(&quot;result&quot;);
                String base64Image = result.get(&quot;video&quot;).asText();
                JsonNode categories = result.get(&quot;categories&quot;);

                byte[] videoBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(videoBytes);
                }
                System.out.println(&quot;Output video saved at &quot; + outputImagePath);
                System.out.println(&quot;\nCategories: &quot; + categories.toString());
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
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
)

func main() {
    API_URL := "http://localhost:8080/video-classification"
    videoPath := "./demo.mp4"
    outputImagePath := "./out.mp4"

    // Base64 encode the local video
    videoBytes, err := ioutil.ReadFile(videoPath)
    if err != nil {
        fmt.Println("Error reading video file:", err)
        return
    }
    videoData := base64.StdEncoding.EncodeToString(videoBytes)

    payload := map[string]string{"video": videoData} // Base64 encoded file content or video URL
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println("Error marshaling payload:", err)
        return
    }

    // Call the API
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

    // Handle the API response
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }
    type Response struct {
        Result struct {
            Image      string   `json:"video"`
            Categories []map[string]interface{} `json:"categories"`
        } `json:"result"`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &respData)
    if err != nil {
        fmt.Println("Error unmarshaling response body:", err)
        return
    }

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println("Error decoding base64 video data:", err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println("Error writing video to file:", err)
        return
    }
    fmt.Printf("Image saved at %s.mp4\n", outputImagePath)
    fmt.Println("\nCategories:")
    for _, category := range respData.Result.Categories {
        fmt.Println(category)
    }
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
    static readonly string API_URL = "http://localhost:8080/video-classification";
    static readonly string videoPath = "./demo.mp4";
    static readonly string outputImagePath = "./out.mp4";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // Base64 encode the local video
        byte[] videoBytes = File.ReadAllBytes(videoPath);
        string video_data = Convert.ToBase64String(videoBytes);

        var payload = new JObject{ { "video", video_data } }; // Base64 encoded file content or video URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        // Call the API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // Handle the API response
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse["result"]["video"].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($"Output video saved at {outputImagePath}");
        Console.WriteLine("\nCategories:");
        Console.WriteLine(jsonResponse["result"]["categories"].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/video-classification'
const videoPath = './demo.mp4'
const outputImagePath = &quot;./out.mp4&quot;;

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'video': encodeImageToBase64(videoPath)  // Base64 encoded file content or video URL
  })
};

// Base64 encode the local video
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// Call the API
axios.request(config)
.then((response) =&gt; {
    // Process the API response
    const result = response.data[&quot;result&quot;];
    const videoBuffer = Buffer.from(result[&quot;video&quot;], 'base64');
    fs.writeFile(outputImagePath, videoBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output video saved at ${outputImagePath}`);
    });
    console.log(&quot;\nCategories:&quot;);
    console.log(result[&quot;categories&quot;]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>
<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/video-classification&quot;; // Service URL
$video_path = &quot;./demo.mp4&quot;;
$output_video_path = &quot;./out.mp4&quot;;

// Base64 encode the local video
$video_data = base64_encode(file_get_contents($video_path));
$payload = array(&quot;video&quot; =&gt; $video_data); // Base64 encoded file content or video URL

// Call the API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// Process the API response
$result = json_decode($response, true)[&quot;result&quot;];
file_put_contents($output_video_path, base64_decode($result[&quot;video&quot;]));
echo &quot;Output video saved at &quot; . $output_video_path . &quot;\n&quot;;
echo &quot;\nCategories:\n&quot;;
print_r($result[&quot;categories&quot;]);
?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on the user's device, allowing it to process data locally without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed procedures on edge deployment, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate method to deploy the model pipeline according to your needs and proceed with subsequent AI application integration.

## 3. Development Integration/Deployment

If the pipeline meets your requirements for inference speed and accuracy, you can directly proceed with development integration/deployment.

If you need to apply the pipeline directly in your Python project, you can refer to the example code in [2.2.2 Python Script Method](#222-python脚本方式集成).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have strict performance requirements for deployment strategies, especially in terms of response speed, to ensure the efficient operation of the system and the smoothness of the user experience. To this end, PaddleX provides a high-performance inference plugin, which aims to deeply optimize the performance of model inference and pre/post-processing to achieve significant acceleration of the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service-Oriented Deployment</b>: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports multiple pipeline service-oriented deployment solutions. For detailed pipeline service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references for basic service-oriented deployment and examples of multi-language service calls:

<details><summary>API Reference</summary>

<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the properties of the response body are as follows:</li>
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
<li>When the request is not processed successfully, the properties of the response body are as follows:</li>
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
<p>Perform multilingual speech recognition on audio.</p>
<p><code>POST /multilingual-speech-recognition</code></p>

<ul>
<li>The properties of the request body are as follows:</li>
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
<td><code>audio</code></td>
<td><code>string</code></td>
<td>The URL or path of the audio file accessible by the server.</td>
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
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>The text result of speech recognition.</td>
</tr>
<tr>
<td><code>segments</code></td>
<td><code>array</code></td>
<td>The result text with timestamps.</td>
</tr>
<tr>
<td><code>language</code></td>
<td><code>string</code></td>
<td>The recognized language.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>segments</code> is an <code>object</code> with the following properties:</p>
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
<td><code>id</code></td>
<td><code>integer</code></td>
<td>The ID of the audio segment.</td>
</tr>
<tr>
<td><code>seek</code></td>
<td><code>integer</code></td>
<td>The pointer of the audio segment.</td>
</tr>
<tr>
<td><code>start</code></td>
<td><code>number</code></td>
<td>The start time of the audio segment.</td>
</tr>
<tr>
<td><code>end</code></td>
<td><code>number</code></td>
<td>The end time of the audio segment.</td>
</tr>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>The recognized text of the audio segment.</td>
</tr>
<tr>
<td><code>tokens</code></td>
<td><code>array</code></td>
<td>The token IDs of the audio segment.</td>
</tr>
<tr>
<td><code>temperature</code></td>
<td><code>number</code></td>
<td>The speed change ratio.</td>
</tr>
<tr>
<td><code>avgLogProb</code></td>
<td><code>number</code></td>
<td>The average log probability.</td>
</tr>
<tr>
<td><code>compressionRatio</code></td>
<td><code>number</code></td>
<td>The compression ratio.</td>
</tr>
<tr>
<td><code>noSpeechProb</code></td>
<td><code>number</code></td>
<td>The probability of no speech.</td>
</tr>
</tbody>
</table>
</details>

<details><summary>Example of Multilingual Service Invocation</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">
import requests

API_URL = &quot;http://localhost:8080/multilingual-speech-recognition&quot; # Service URL
audio_path = &quot;./zh.wav&quot;

payload = {&quot;audio&quot;: audio_path}

# Invoke API
response = requests.post(API_URL, json=payload)

# Process API response
assert response.status_code == 200
result = response.json()[&quot;result&quot;]
print(result)
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computational and data processing capabilities directly on user devices, allowing them to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model into your pipeline and proceed with subsequent AI application integration.

## 4. Secondary Development
If the default model weights provided by the general video classification pipeline are not satisfactory in terms of accuracy or speed for your specific scenario, you can attempt to <b>fine-tune</b> the existing model using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the general video classification pipeline in your scenario.

### 4.1 Model Fine-Tuning

Since the general video classification pipeline only includes a video classification module, if the performance of the pipeline is not up to expectations, you can analyze the videos with poor recognition and refer to the corresponding fine-tuning tutorial links in the table below for model fine-tuning.


<table>
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Fine-Tuning Module</th>
      <th>Fine-Tuning Reference Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Inaccurate video classification</td>
      <td>Video Classification Module</td>
      <td><a href="../../../module_usage/tutorials/video_modules/video_classification.en.md">Link</a></td>
    </tr>

  </tbody>
</table>

### 4.2 Model Application
After completing the fine-tuning with your private dataset, you will obtain the local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the path to the fine-tuned model weights with the corresponding location in the pipeline configuration file:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/multilingual_speech_recognition.yaml")
output = pipeline.predict(input="zh.wav")
for res in output:
    res.print()
    res.save_to_json("./output/")
```

Subsequently, refer to the command-line method or Python script method in the local experience to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware devices.

For example, if you use Ascend NPU for video classification in the pipeline, the Python command used is: