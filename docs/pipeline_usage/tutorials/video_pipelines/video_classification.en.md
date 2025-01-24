---
comments: true
---

# General Video Classification Pipeline Tutorial

## 1. Introduction to the General Video Classification Pipeline
Video Classification is a technique that assigns video clips to predefined categories. It is widely applied in fields such as action recognition, event detection, and content recommendation. Video classification can recognize various dynamic events and scenes, such as sports activities, natural phenomena, traffic conditions, and categorize them based on their characteristics. By utilizing deep learning models, especially the combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN), video classification can automatically extract spatio-temporal features from videos and perform accurate classification. This technology holds significant applications in video surveillance, media retrieval, and personalized recommendation systems.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/video_classification/01.jpg">
<b>The General Video Classification Pipeline includes an video classification module. If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.</b>

<details><summary> 👉Details of Model List</summary>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top1 Acc(%)</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PPTSM_ResNet50_k400_8frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PPTSM_ResNet50_k400_8frames_uniform_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PPTSM_ResNet50_k400_8frames_uniform_pretrained.pdparams">Trained Model</a></td>
<td>74.36</td>
<td>93.4 M</td>
<td rowspan="1">
PP-TSM is a video classification model developed by Baidu PaddlePaddle's Vision Team. This model is optimized based on the ResNet-50 backbone network and undergoes model tuning in six aspects: data augmentation, network structure fine-tuning, training strategies, Batch Normalization (BN) layer optimization, pre-trained model selection, and model distillation. Under the center crop evaluation method, its accuracy on Kinetics-400 is improved by 3.95 points compared to the original paper's implementation.
</td>
</tr>

<tr>
<td>PPTSMv2_LCNet_k400_8frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PPTSMv2_LCNet_k400_8frames_uniform_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PPTSMv2_LCNet_k400_8frames_uniform_pretrained.pdparams">Trained Model</a></td>
<td>71.71</td>
<td>22.5 M</td>
<td rowspan="2">PP-TSMv2 is a lightweight video classification model optimized based on the CPU-oriented model PP-LCNetV2. It undergoes model tuning in seven aspects: backbone network and pre-trained model selection, data augmentation, TSM module tuning, input frame number optimization, decoding speed optimization, DML distillation, and LTA module. Under the center crop evaluation method, it achieves an accuracy of 75.16%, with an inference speed of only 456ms on the CPU for a 10-second video input.</td>
</tr>
<tr>
<td>PPTSMv2_LCNet_k400_16frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PPTSMv2_LCNet_k400_16frames_uniform_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PPTSMv2_LCNet_k400_16frames_uniform_pretrained.pdparams">Trained Model</a></td>
<td>73.11</td>
<td>22.5 M</td>
</tr>

</table>

<p><b>Note: The above accuracy metrics refer to Top-1 Accuracy on the <a href="https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/dataset/k400.md">K400</a> validation set. </b><b>All model GPU inference times are based on NVIDIA Tesla T4 machines, with precision type FP32. CPU inference speeds are based on Intel® Xeon® Gold 5117 CPU @ 2.00GHz, with 8 threads and precision type FP32.</b></p></details>

## 2. Quick Start

PaddleX supports experiencing the effects of pipelines locally using the command line or Python.

Before using the general video classification pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the PaddleX local installation tutorial.

#### 2.1 Command Line Experience
A single command is all you need to quickly experience the video classification pipeline, Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4), and replace `--input` with the local path to perform prediction.

```bash
paddlex --pipeline video_classification --input general_video_classification_001.mp4 --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it is the video classification pipeline.
--input: The local path or URL of the input video to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the second and third GPUs). You can also choose to use CPU (--device cpu).
```

When executing the above command, the default video classification pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details><summary> 👉Click to expand</summary>

<pre><code class="language-bash">paddlex --get_pipeline_config video_classification
</code></pre>
<p>After execution, the video classification pipeline configuration file will be saved in the current path. If you wish to customize the save location, you can execute the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code class="language-bash">paddlex --get_pipeline_config video_classification --save_path ./my_path
</code></pre>
<p>After obtaining the pipeline configuration file, replace <code>--pipeline</code> with the configuration file's save path to make the configuration file take effect. For example, if the configuration file's save path is <code>./video_classification.yaml</code>, simply execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./video_classification.yaml --input general_video_classification_001.mp4  --device gpu:0
</code></pre>
<p>Here, parameters such as <code>--model</code> and <code>--device</code> do not need to be specified, as they will use the parameters in the configuration file. If you still specify parameters, the specified parameters will take precedence.</p></details>

After running, the result will be:

```
{'input_path': 'general_video_classification_001.mp4', 'class_ids': [0], 'scores': array([0.91996]), 'label_names': ['abseiling']}
```
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/video_classification/02.jpg">


The visualized video not saved by default. You can customize the save path through `--save_path`, and then all results will be saved in the specified path.

#### 2.2  Python Script Integration
A few lines of code can complete the quick inference of the pipeline. Taking the general video classification pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="video_classification")

output = pipeline.predict("general_video_classification_001.mp4")
for res in output:
    res.print() # Print the structured output of the prediction
    res.save_to_video("./output/")  # Save the result visualization video
    res.save_to_json("./output/") # Save the structured output of the prediction
```
The results obtained are the same as those obtained through the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: The specific parameter descriptions are as follows:

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
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the name of the pipeline, it must be a pipeline supported by PaddleX.</td>
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
<td>Whether to enable high-performance inference, which is only available when the pipeline supports it.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
(2) Call the `predict` method of the video classification pipeline object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

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
<td>Supports directly passing Python variables, such as numpy.ndarray representing video data.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing the path of the file to be predicted, such as the local path of an video file: <code>/root/data/video.mp4。</code>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing the URL of the file to be predicted, such as the network URL of an video file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4 ">Example</a>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing a local directory, which should contain files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td><code>dict</code></td>
<td>Supports passing a dictionary type, where the key needs to correspond to the specific task, such as "video" for the video classification task, and the value of the dictionary supports the above data types, e.g., <code>{"video": "/root/data1"}</code>.</td>
</tr>
<tr>
<td><code>list</code></td>
<td>Supports passing a list, where the list elements need to be the above data types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/video1.mp4", "/root/data/video2.mp4"]</code>, <code>["/root/data1", "/root/data2"]</code>, <code>[{"video": "/root/data1"}, {"video": "/root/data2/video.mp4"}]</code>.</td>
</tr>
</tbody>
</table>
3）Obtain prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

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
<td>save_to_video</td>
<td>Saves results as an video file</td>
<td><code>- save_path</code>: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;</td>
</tr>
</tbody>
</table>
If you have a configuration file, you can customize the configurations of the video anomaly detection pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/video_classification.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/video_classification.yaml")
output = pipeline.predict("general_video_classification_001.mp4 ")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_video("./output/")  # Save the visualization video of the result
    res.save_to_json("./output/")  # Save the structured output of prediction
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the pipeline directly in your Python project, refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

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
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed as <code>"Success"</code>.</td>
</tr>
</tbody>
</table>
<p>The response body may also have a <code>result</code> property of type <code>object</code>, which stores the operation result information.</p>
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
<p>Classify videos.</p>
<p><code>POST /video-classification</code></p>
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
<td><code>video</code></td>
<td><code>string</code></td>
<td>The URL of a video file accessible by the server or the Base64 encoded result of the video file content.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>inferenceParams</code></td>
<td><code>object</code></td>
<td>Inference parameters.</td>
<td>No</td>
</tr>
</tbody>
</table>
<p>The properties of <code>inferenceParams</code> are as follows:</p>
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
<td><code>topK</code></td>
<td><code>integer</code></td>
<td>Only the top <code>topK</code> categories with the highest scores will be retained in the results.</td>
<td>No</td>
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
<td><code>categories</code></td>
<td><code>array</code></td>
<td>video category information.</td>
</tr>
<tr>
<td><code>video</code></td>
<td><code>string</code></td>
<td>The video classification result video. The video is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>categories</code> is an <code>object</code> with the following properties:</p>
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
<td><code>id</code></td>
<td><code>integer</code></td>
<td>Category ID.</td>
</tr>
<tr>
<td><code>name</code></td>
<td><code>string</code></td>
<td>Category name.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>Category score.</td>
</tr>
</tbody>
</table>
<p>An example of <code>result</code> is as follows:</p>
<pre><code class="language-json">{
&quot;categories&quot;: [
{
&quot;id&quot;: 5,
&quot;name&quot;: &quot;Rabbit&quot;,
&quot;score&quot;: 0.93
}
],
&quot;video&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/video-classification&quot;
video_path = &quot;./demo.mp4&quot;
output_video_path = &quot;./out.mp4&quot;

with open(video_path, &quot;rb&quot;) as file:
    video_bytes = file.read()
    video_data = base64.b64encode(video_bytes).decode(&quot;ascii&quot;)

payload = {&quot;video&quot;: video_data}

response = requests.post(API_URL, json=payload)

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
#include &quot;cpp-httplib/httplib.h&quot; // https://github.com/Huiyicc/cpp-httplib
#include &quot;nlohmann/json.hpp&quot; // https://github.com/nlohmann/json
#include &quot;base64.hpp&quot; // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string videoPath = &quot;./demo.mp4&quot;;
    const std::string outputvideoPath = &quot;./out.mp4&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

    std::ifstream file(videoPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; &quot;Error reading file.&quot; &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedVideo = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj[&quot;video&quot;] = encodedVideo;
    std::string body = jsonObj.dump();

    auto response = client.Post(&quot;/video-classification&quot;, headers, body, &quot;application/json&quot;);
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedVideo = result[&quot;video&quot;];
        std::string decodedString = base64::from_base64(encodedVideo);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutvideoPath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; &quot;Output video saved at &quot; &lt;&lt; outPutvideoPath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; outPutvideoPath &lt;&lt; std::endl;
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
        String API_URL = &quot;http://localhost:8080/video-classification&quot;;
        String videoPath = &quot;./demo.mp4&quot;;
        String outputvideoPath = &quot;./out.mp4&quot;;

        File file = new File(videoPath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String videoData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;video&quot;, videoData);

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
                String base64Image = result.get(&quot;video&quot;).asText();
                JsonNode categories = result.get(&quot;categories&quot;);

                byte[] videoBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputvideoPath)) {
                    fos.write(videoBytes);
                }
                System.out.println(&quot;Output video saved at &quot; + outputvideoPath);
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
    &quot;bytes&quot;
    &quot;encoding/base64&quot;
    &quot;encoding/json&quot;
    &quot;fmt&quot;
    &quot;io/ioutil&quot;
    &quot;net/http&quot;
)

func main() {
    API_URL := &quot;http://localhost:8080/video-classification&quot;
    videoPath := &quot;./demo.mp4&quot;
    outputvideoPath := &quot;./out.mp4&quot;

    videoBytes, err := ioutil.ReadFile(videoPath)
    if err != nil {
        fmt.Println(&quot;Error reading video file:&quot;, err)
        return
    }
    videoData := base64.StdEncoding.EncodeToString(videoBytes)

    payload := map[string]string{&quot;video&quot;: videoData}
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
            Image      string   `json:&quot;video&quot;`
            Categories []map[string]interface{} `json:&quot;categories&quot;`
        } `json:&quot;result&quot;`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println(&quot;Error unmarshaling response body:&quot;, err)
        return
    }

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println(&quot;Error decoding base64 video data:&quot;, err)
        return
    }
    err = ioutil.WriteFile(outputvideoPath, outputImageData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing video to file:&quot;, err)
        return
    }
    fmt.Printf(&quot;Image saved at %s.mp4\n&quot;, outputvideoPath)
    fmt.Println(&quot;\nCategories:&quot;)
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
    static readonly string API_URL = &quot;http://localhost:8080/video-classification&quot;;
    static readonly string videoPath = &quot;./demo.mp4&quot;;
    static readonly string outputvideoPath = &quot;./out.mp4&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] videoBytes = File.ReadAllBytes(videoPath);
        string video_data = Convert.ToBase64String(videoBytes);

        var payload = new JObject{ { &quot;video&quot;, video_data } };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse[&quot;result&quot;][&quot;video&quot;].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputvideoPath, outputImageBytes);
        Console.WriteLine($&quot;Output video saved at {outputvideoPath}&quot;);
        Console.WriteLine(&quot;\nCategories:&quot;);
        Console.WriteLine(jsonResponse[&quot;result&quot;][&quot;categories&quot;].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/video-classification'
const videoPath = './demo.mp4'
const outputvideoPath = &quot;./out.mp4&quot;;

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'video': encodeImageToBase64(videoPath)
  })
};

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

axios.request(config)
.then((response) =&gt; {
    const result = response.data[&quot;result&quot;];
    const videoBuffer = Buffer.from(result[&quot;video&quot;], 'base64');
    fs.writeFile(outputvideoPath, videoBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output video saved at ${outputvideoPath}`);
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

$API_URL = &quot;http://localhost:8080/video-classification&quot;;
$video_path = &quot;./demo.mp4&quot;;
$output_video_path = &quot;./out.mp4&quot;;

$video_data = base64_encode(file_get_contents($video_path));
$payload = array(&quot;video&quot; =&gt; $video_data);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)[&quot;result&quot;];
file_put_contents($output_video_path, base64_decode($result[&quot;video&quot;]));
echo &quot;Output video saved at &quot; . $output_video_path . &quot;\n&quot;;
echo &quot;\nCategories:\n&quot;;
print_r($result[&quot;categories&quot;]);
?&gt;
</code></pre></details>

</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method that places computing and data processing functions on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method for your model pipeline based on your needs and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the general video classification pipeline do not meet your requirements for accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using <b>data from your specific domain or application scenario</b> to improve the recognition performance of the general video classification pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the general video classification pipeline includes an video classification module, if the performance of the pipeline does not meet expectations, you need to refer to the [Customization](../../../module_usage/tutorials/video_modules/video_classification.en.md#iv-custom-development) section in the [Video Classification Module Development Tutorial](../../../module_usage/tutorials/video_modules/video_classification.en.md) and use your private dataset to fine-tune the video classification model.

### 4.2 Model Application
After you have completed fine-tuning training using your private dataset, you will obtain local model weight files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```yaml
......
Pipeline:
  model: PPTSMv2_LCNet_k400_8frames_uniform  # Can be modified to the local path of the fine-tuned model
  device: "gpu"
  batch_size: 1
......
```
Then, refer to the command line method or Python script method in the local experience section to load the modified pipeline configuration file.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for inference in the video classification pipeline, the Python command is:

```bash
paddlex --pipeline video_classification --input general_video_classification_001.mp4  --device gpu:0
``````
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the Python command to `npu:0`:

```bash
paddlex --pipeline video_classification --input general_video_classification_001.mp4  --device npu:0
```
If you want to use the General Video Classification Pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
