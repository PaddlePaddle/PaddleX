---
comments: true
---

# General Image Multi-Label Classification Pipeline Tutorial

## 1. Introduction to the General Image Multi-Label Classification Pipeline
Image multi-label classification is a technique that assigns multiple relevant categories to a single image simultaneously, widely used in image annotation, content recommendation, and social media analysis. It can identify multiple objects or features present in an image, for example, an image containing both "dog" and "outdoor" labels. By leveraging deep learning models, image multi-label classification automatically extracts image features and performs accurate classification, providing users with more comprehensive information. This technology is of great significance in applications such as intelligent search engines and automatic content generation.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_multi_label_classification/01.png">

<b>The General Image Multi-Label Classification Pipeline includes a module for image multi-label classification. If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, choose a model with faster inference. If you prioritize model storage size, choose a model with a smaller storage size.</b>

<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>mAP (%)</th>
<th>Model Storage Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIP_vit_base_patch16_448_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_448_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_448_ML_pretrained.pdparams">Trained Model</a></td>
<td>89.15</td>
<td>-</td>
</tr>
<tr>
<td>PP-HGNetV2-B0_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_ML_pretrained.pdparams">Trained Model</a></td>
<td>80.98</td>
<td>39.6</td>
</tr>
<tr>
<td>PP-HGNetV2-B4_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_ML_pretrained.pdparams">Trained Model</a></td>
<td>87.96</td>
<td>88.5</td>
</tr>
<tr>
<td>PP-HGNetV2-B6_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_ML_pretrained.pdparams">Trained Model</a></td>
<td>91.25</td>
<td>286.5</td>
</tr>
<tr>
<td>PP-LCNet_x1_0_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_ML_pretrained.pdparams">Trained Model</a></td>
<td>77.96</td>
<td>29.4</td>
</tr>
<tr>
<td>ResNet50_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_ML_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_ML_pretrained.pdparams">Trained Model</a></td>
<td>83.50</td>
<td>108.9</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are mAP for the multi-label classification task on </b><a href="https://cocodataset.org/#home">COCO2017</a><b>. The GPU inference time for all models is based on an NVIDIA Tesla T4 machine with FP32 precision. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## 2. Quick Start
PaddleX supports experiencing the effects of the General Image Multi-Label Classification Pipeline locally using command line or Python.

Before using the General Image Multi-Label Classification Pipeline locally, please ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 2.1 Experience via Command Line
Experience the effects of the image multi-label classification pipeline with a single command:

```bash
paddlex --pipeline multi_label_image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it is the image multi-label classification pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the second and third GPUs). You can also choose to use CPU (--device cpu).
```

When executing the above command, the default configuration file for the image multi-label classification pipeline is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details><summary> 👉Click to Expand</summary>

<pre><code class="language-bash">paddlex --get_pipeline_config multi_label_image_classification
</code></pre>
<p>After execution, the configuration file for the image multi-label classification pipeline will be saved in the current path. If you wish to customize the save location, you can execute the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code class="language-bash">paddlex --get_pipeline_config multi_label_image_classification --save_path ./my_path
</code></pre>
<p>After obtaining the pipeline configuration file, replace <code>--pipeline</code> with the saved path of the configuration file to make it effective. For example, if the configuration file is saved at <code>./multi_label_image_classification.yaml</code>, simply execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./multi_label_image_classification.yaml --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0
</code></pre>
<p>Where <code>--model</code>, <code>--device</code>, and other parameters are not specified, the parameters in the configuration file will be used. If parameters are specified, the specified parameters will take precedence.</p></details>

After running, the result obtained is:

```
{'input_path': 'general_image_classification_001.jpg', 'class_ids': [21, 0, 30, 24], 'scores': [0.99257, 0.70596, 0.63001, 0.57852], 'label_names': ['bear', 'person', 'skis', 'backpack']}
```
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_multi_label_classification/02.png">

The visualized image not saved by default. You can customize the save path through `--save_path`, and then all results will be saved in the specified path.

### 2.2 Integration via Python Script
A few lines of code can complete the rapid inference of the pipeline. Taking the general image multi-label classification pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="multi_label_image_classification")

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_img("./output/")  # Save the result visualization image
    res.save_to_json("./output/")  # Save the structured output of the prediction
```
The result obtained is the same as that of the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: Specific parameter descriptions are as follows:

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
<td>The name of the pipeline or the path of the pipeline configuration file. If it is the name of the pipeline, it must be a pipeline supported by PaddleX.</td>
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
(2) Call the `predict` method of the multi-label classification pipeline object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

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
<td>Supports directly passing in Python variables, such as numpy.ndarray representing image data.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in the file path of the data file to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in the URL of the data file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg">Example</a>.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in a local directory, which should contain the data files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td>dict</td>
<td>Supports passing in a dictionary type, where the key of the dictionary needs to correspond to the specific task, such as "img" for image classification tasks, and the value of the dictionary supports the above data types, for example: <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td>list</td>
<td>Supports passing in a list, where the list elements need to be the above data types, such as <code>[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>
（3）Obtain prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

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

For example, if your configuration file is saved at `./my_path/multi_label_image_classification.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/multi_label_image_classification.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_img("./output/")  # Save the visualization image of the result
    res.save_to_json("./output/")  # Save the structured output of prediction
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have strict standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins that aim to deeply optimize model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance inference procedures, refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

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
<td>Error code. Fixed to <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed to <code>"Success"</code>.</td>
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
<p>Classify images.</p>
<p><code>POST /multilabel-image-classification</code></p>
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
<td><code>image</code></td>
<td><code>string</code></td>
<td>The URL of the image file accessible by the server or the Base64 encoded result of the image file content.</td>
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
<td><code>threshold</code></td>
<td><code>number</code></td>
<td>Category score threshold.</td>
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
<td>Image category information.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>Image classification result image. The image is in JPEG format and encoded in Base64.</td>
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
&quot;image&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/multilabel-image-classification&quot;
image_path = &quot;./demo.jpg&quot;
output_image_path = &quot;./out.jpg&quot;

with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
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
    const std::string imagePath = &quot;./demo.jpg&quot;;
    const std::string outputImagePath = &quot;./out.jpg&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
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
    jsonObj[&quot;image&quot;] = encodedImage;
    std::string body = jsonObj.dump();

    auto response = client.Post(&quot;/multilabel-image-classification&quot;, headers, body, &quot;application/json&quot;);
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedImage = result[&quot;image&quot;];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; &quot;Output image saved at &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
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
        String API_URL = &quot;http://localhost:8080/multilabel-image-classification&quot;;
        String imagePath = &quot;./demo.jpg&quot;;
        String outputImagePath = &quot;./out.jpg&quot;;

        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;image&quot;, imageData);

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
                String base64Image = result.get(&quot;image&quot;).asText();
                JsonNode categories = result.get(&quot;categories&quot;);

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println(&quot;Output image saved at &quot; + outputImagePath);
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
    API_URL := &quot;http://localhost:8080/multilabel-image-classification&quot;
    imagePath := &quot;./demo.jpg&quot;
    outputImagePath := &quot;./out.jpg&quot;

    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println(&quot;Error reading image file:&quot;, err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{&quot;image&quot;: imageData}
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
            Image      string   `json:&quot;image&quot;`
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
        fmt.Println(&quot;Error decoding base64 image data:&quot;, err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing image to file:&quot;, err)
        return
    }
    fmt.Printf(&quot;Image saved at %s.jpg\n&quot;, outputImagePath)
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
    static readonly string API_URL = &quot;http://localhost:8080/multilabel-image-classification&quot;;
    static readonly string imagePath = &quot;./demo.jpg&quot;;
    static readonly string outputImagePath = &quot;./out.jpg&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { &quot;image&quot;, image_data } };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse[&quot;result&quot;][&quot;image&quot;].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($&quot;Output image saved at {outputImagePath}&quot;);
        Console.WriteLine(&quot;\nCategories:&quot;);
        Console.WriteLine(jsonResponse[&quot;result&quot;][&quot;categories&quot;].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/multilabel-image-classification'
const imagePath = './demo.jpg'
const outputImagePath = &quot;./out.jpg&quot;;

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)
  })
};

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

axios.request(config)
.then((response) =&gt; {
    const result = response.data[&quot;result&quot;];
    const imageBuffer = Buffer.from(result[&quot;image&quot;], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
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

$API_URL = &quot;http://localhost:8080/multilabel-image-classification&quot;;
$image_path = &quot;./demo.jpg&quot;;
$output_image_path = &quot;./out.jpg&quot;;

$image_data = base64_encode(file_get_contents($image_path));
$payload = array(&quot;image&quot; =&gt; $image_data);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)[&quot;result&quot;];
file_put_contents($output_image_path, base64_decode($result[&quot;image&quot;]));
echo &quot;Output image saved at &quot; . $output_image_path . &quot;\n&quot;;
echo &quot;\nCategories:\n&quot;;
print_r($result[&quot;categories&quot;]);
?&gt;
</code></pre></details>
</details>

<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a way to place computing and data processing functions on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method for your model pipeline based on your needs and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the general image multi-label classification pipeline do not meet your requirements in terms of accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the general image multi-label classification pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the general image multi-label classification pipeline includes an image multi-label classification module, if the performance of the pipeline does not meet expectations, you need to refer to the [Customization](../../../module_usage/tutorials/cv_modules/image_multilabel_classification.en.md#Customization) section in the [Image Multi-Label Classification Module Development Tutorial](../../../module_usage/tutorials/cv_modules/image_multilabel_classification.en.md) to fine-tune the image multi-label classification model using your private dataset.

### 4.2 Model Application
After you have completed fine-tuning training using your private dataset, you will obtain local model weights files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```python
......
Pipeline:
  model: PP-LCNet_x1_0_ML   # Can be modified to the local path of the fine-tuned model
  batch_size: 1
  device: "gpu:0"
......
```
Then, refer to the command line method or Python script method in the local experience section to load the modified pipeline configuration file.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for inference of the image multi-label classification pipeline, the Python command is:

```bash
paddlex --pipeline multi_label_image_classification --input https://paddle-model-ecology.bj.bcebos.com/padd
```

At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the Python command to `npu:0`:

```bash
paddlex --pipeline multi_label_image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device npu:0
```
If you want to use the General Image Multi-label Classification Pipeline on more diverse hardware, please refer to the [PaddleX Multi-device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
