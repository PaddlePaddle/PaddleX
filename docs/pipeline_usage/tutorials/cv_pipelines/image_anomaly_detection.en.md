---
comments: true
---

# Image Anomaly Detection Pipeline Tutorial

## 1. Introduction to Image Anomaly Detection Pipeline
Image anomaly detection is an image processing technique that identifies images that stand out or do not conform to normal patterns by analyzing the content within the images. It can automatically detect potential defects, anomalies, or abnormal behaviors in images, thereby helping us to identify problems in a timely manner and take appropriate measures.
This pipeline integrates the high-precision anomaly detection model STFPM, which extracts regions of anomalies or defects from images. The application scenarios cover various fields, including industrial manufacturing, food appearance quality inspection, and medical image analysis. The pipeline also offers flexible service-oriented deployment options, supporting the use of multiple programming languages on various hardware platforms. Moreover, it provides the capability for secondary development. You can train and fine-tune models on your own dataset based on this pipeline, and the trained models can be seamlessly integrated.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_anomaly_detection/01.png">

<b>The image anomaly detection pipeline includes an unsupervised anomaly detection module, with the following model benchmarks</b>:

<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>Avg (%)</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>STFPM</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/STFPM_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/STFPM_pretrained.pdparams">Trained Model</a></td>
<td>96.2</td>
<td>21.5 M</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are the average anomaly scores on the </b>[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)<b> validation set. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## 2. Quick Start
PaddleX provides pre-trained models for the anomaly detection pipeline, allowing for quick experience of its effects. You can use the command line or Python to experience the image anomaly detection pipeline locally.

Before using the image anomaly detection pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 2.1 Command-Line Experience
You can quickly experience the image anomaly detection pipeline with just one command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png), and replace `--input` with the local path for prediction.

Note: Due to network issues, the above URL could not be successfully parsed. If you need the content of this webpage, please check the validity of the URL and try again later. If you do not need the content of this link, you can proceed with the other instructions.

```bash
paddlex --pipeline anomaly_detection --input uad_grid.png --device gpu:0  --save_path ./output
```

The relevant parameter descriptions can be found in the [2.1.2 Python Script Integration](#212-python脚本方式集成) section.

After running, the results will be printed to the terminal as follows:

<pre><code>{'input_path': 'uad_grid.png', 'pred': '...'}</code></pre>

The explanation of the result parameters can be found in the [2.1.2 Python Script Integration](#212-python脚本方式集成) section.

The visualization results are saved under `save_path`, and the visualization results are as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_anomaly_detection/02.png">

### 2.2 Python Script Integration

The above command line is for quickly experiencing and checking the effect. Generally, in a project, it is often necessary to integrate through code. You can complete the quick inference of the pipeline with a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="anomaly_detection")
output = pipeline.predict(input="uad_grid.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img(save_path="./output/") ## 保存结果可视化图像
    res.save_to_json(save_path="./output/") ## 保存预测的结构化输出
```

In the above Python script, the following steps are executed:

(1) Instantiate the pipeline object through `create_pipeline()`: The specific parameter descriptions are as follows:

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
<td>Pipeline name or pipeline configuration file path. If it is a pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the pipeline (if set simultaneously with <code>pipeline</code>, it has higher priority than <code>pipeline</code>, and the pipeline name must be consistent with <code>pipeline</code>).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Pipeline inference device. Supports specifying the specific GPU card number, such as "gpu:0", other hardware specific card numbers, such as "npu:0", CPU such as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available when the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the anomaly_detection pipeline object for inference prediction. This method will return a `generator`. The following are the parameters and their descriptions of the `predict()` method:

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
<td>Data to be predicted, supports multiple input types, required</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
  <li><b>Python Var</b>: Such as <code>numpy.ndarray</code> representing image data</li>
  <li><b>str</b>: Such as the local path of the image file: <code>/root/data/img.jpg</code>; <b>such as URL link</b>, such as the network URL of the image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png">Example</a>; <b>such as local directory</b>, the directory must contain the images to be predicted, such as the local path: <code>/root/data/</code></li>
  <li><b>List</b>: The list elements must be the above types of data, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Pipeline inference device</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>CPU</b>: Such as <code>cpu</code> indicating using CPU for inference;</li>
  <li><b>GPU</b>: Such as <code>gpu:0</code> indicating using the 1st GPU for inference;</li>
  <li><b>NPU</b>: Such as <code>npu:0</code> indicating using the 1st NPU for inference;</li>
  <li><b>XPU</b>: Such as <code>xpu:0</code> indicating using the 1st XPU for inference;</li>
  <li><b>MLU</b>: Such as <code>mlu:0</code> indicating using the 1st MLU for inference;</li>
  <li><b>DCU</b>: Such as <code>dcu:0</code> indicating using the 1st DCU for inference;</li>
  <li><b>None</b>: If set to <code>None</code>, it will default to using the parameter value initialized by the pipeline. During initialization, it will preferentially use the local GPU 0 device, if not available, it will use the CPU device;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

(3) Process the prediction results. The prediction result for each sample is of `dict` type and supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Description</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">Print results to the terminal</td>
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
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save results as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. If it is a directory, the saved file will be named the same as the input file type</td>
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
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will retain the original characters. Only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save results as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. Supports directory or file path</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal. The content printed to the terminal is explained as follows:

    - `input_path`: `(str)` The input path of the image to be predicted

    - `pred`: `(str)` The prediction result. Due to the large number of pixel values, `...` is used here instead of printing.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.json`. If specified as a file, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, `numpy.array` types will be converted to lists.

- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.{your_img_extension}`. If specified as a file, it will be saved directly to that file. (Since the pipeline usually contains many result images, it is not recommended to specify a specific file path directly, otherwise multiple images will be overwritten, leaving only the last image)

* Additionally, it also supports obtaining visualized images and prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan = "2"><code>img</code></td>
<td rowspan = "2">Get visualized images in <code>dict</code> format</td>
</tr>
</table>

- The prediction result obtained by the `json` attribute is of dict type, and the content is consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is a dictionary type data. The key is `res`, and the corresponding value is an `Image.Image` object: used to display the visualized image of the anomaly_detection result.

In addition, you can obtain the anomaly_detection pipeline configuration file and load the configuration file for prediction. You can execute the following command to save the results in `my_path`:

```
paddlex --get_pipeline_config anomaly_detection --save_path ./my_path
```

If you have obtained the configuration file, you can customize the various configurations of the image anomaly detection pipeline. You only need to modify the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved in `./my_path/*anomaly_detection.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/anomaly_detection.yaml")
output = pipeline.predict("uad_grid.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the pipeline directly into your Python project, you can refer to the example code in [2.2 Python Script Integration](#22-python脚本方式集成).

In addition, PaddleX also provides three other deployment methods, which are detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have strict performance requirements for deployment strategies, especially in terms of response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin, which aims to deeply optimize the performance of model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service-Based Deployment</b>: Service-based deployment is a common form of deployment in actual production environments. By encapsulating inference capabilities into services, clients can access these services through network requests to obtain inference results. PaddleX supports various service-based deployment solutions for pipelines. For detailed procedures, please refer to the [PaddleX Service-Based Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references for basic service-based deployment and examples of multi-language service calls:

<details><summary>API Reference</summary>

<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is successfully processed, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
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
<td>Error code. Fixed to <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed to <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not successfully processed, the attributes of the response body are as follows:</li>
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
<td>Error description.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Perform anomaly detection on the image.</p>
<p><code>POST /anomaly-detection</code></p>
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
<td><code>image</code></td>
<td><code>string</code></td>| <code>null</code></td>
<td>The URL of the image file accessible by the server or the Base64 encoded result of the image file content.</td>
<td>Yes</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is successfully processed, the <code>result</code> of the response body has the following attributes:</li>
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
<td><code>labelMap</code></td>
<td><code>array</code></td>
<td>Records the category label of each pixel in the image (arranged in row-first order). Where <code>255</code> indicates an anomaly point, and <code>0</code> indicates a non-anomaly point.</td>
</tr>
<tr>
<td><code>size</code></td>
<td><code>array</code></td>
<td>Image shape. The elements in the array are the height and width of the image in order.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>Anomaly detection result image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p>An example of <code>result</code> is as follows:</p>
<pre><code class="language-json">{
&quot;labelMap&quot;: [
0,
0,
255,
0
],
&quot;size&quot;: [
2,
2
],
&quot;image&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>Multi-language Service Invocation Example</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/image-anomaly-detection"  # Service URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# Encode the local image using Base64
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64-encoded file content or image URL

# Call the API
response = requests.post(API_URL, json=payload)

# Process the response data
assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
# result.labelMap records the class labels for each pixel in the image (arranged in row-major order). See the API reference for details.
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include <iostream>
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client("localhost:8080");
    const std::string imagePath = "./demo.jpg";
    const std::string outputImagePath = "./out.jpg";

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    // Encode the local image using Base64
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

    // Call the API
    auto response = client.Post("/image-anomaly-detection", headers, body, "application/json");
    // Process the response data
    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        encodedImage = result["image"];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector<unsigned char> decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outputImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast<char*>(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout << "Output image saved at " << outputImagePath << std::endl;
            // result.labelMap records the class labels for each pixel in the image (arranged in row-major order). See the API reference for details.
        } else {
            std::cerr << "Unable to open file for writing: " << outputImagePath << std::endl;
        }
    } else {
        std::cout << "Failed to send HTTP request." << std::endl;
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
        String API_URL = &quot;http://localhost:8080/image-anomaly-detection&quot;; // Service URL
        String imagePath = &quot;./demo.jpg&quot;; // Local image
        String outputImagePath = &quot;./out.jpg&quot;; // Output image

        // Encode the local image using Base64
        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;image&quot;, imageData); // Base64-encoded file content or image URL

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
                String base64Image = result.get(&quot;image&quot;).asText();
                JsonNode labelMap = result.get(&quot;labelMap&quot;);

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println(&quot;Output image saved at &quot; + outputImagePath);
                // result.labelMap contains the class labels for each pixel in the image (arranged in row-major order). See the API reference documentation for details.
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
    API_URL := "http://localhost:8080/image-anomaly-detection"
    imagePath := "./demo.jpg"
    outputImagePath := "./out.jpg"

    // Base64 encode the local image
    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println("Error reading image file:", err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{"image": imageData} // Base64 encoded file content or image URL
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

    // Process the returned data from the API
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }
    type Response struct {
        Result struct {
            Image      string   `json:"image"`
            Labelmap []map[string]interface{} `json:"labelMap"`
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
        fmt.Println("Error decoding base64 image data:", err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println("Error writing image to file:", err)
        return
    }
    fmt.Printf("Image saved at %s.jpg\n", outputImagePath)
    // result.labelMap records the category label of each pixel in the image (arranged in row-first order). See the API reference documentation for details.
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
    static readonly string API_URL = &quot;http://localhost:8080/image-anomaly-detection&quot;;
    static readonly string imagePath = &quot;./demo.jpg&quot;;
    static readonly string outputImagePath = &quot;./out.jpg&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // Base64 encode the local image
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { &quot;image&quot;, image_data } }; // Base64 encoded file content or image URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        // Call the API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // Process the API response
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse[&quot;result&quot;][&quot;image&quot;].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($&quot;Output image saved at {outputImagePath}&quot;);
        // result.labelMap records the category label of each pixel in the image (arranged in row-first order). See API reference documentation for details.
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/image-anomaly-detection';
const imagePath = './demo.jpg';
const outputImagePath = './out.jpg';

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)  // Base64-encoded file content or image URL
  })
};

// Encode the local image using Base64
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// Call the API
axios.request(config)
.then((response) => {
    // Process the response data
    const result = response.data['result'];
    const imageBuffer = Buffer.from(result['image'], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) => {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    // result.labelMap records the class labels for each pixel in the image (arranged in row-major order). See the API reference for details.
})
.catch((error) => {
  console.log(error);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/image-anomaly-detection&quot;; // Service URL
$image_path = &quot;./demo.jpg&quot;;
$output_image_path = &quot;./out.jpg&quot;;

// Encode the local image using Base64
$image_data = base64_encode(file_get_contents($image_path));
$payload = array(&quot;image&quot; =&gt; $image_data); // Base64-encoded file content or image URL

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
file_put_contents($output_image_path, base64_decode($result[&quot;image&quot;]));
echo &quot;Output image saved at &quot; . $output_image_path . &quot;\n&quot;;
// result.labelMap contains the class labels for each pixel in the image (arranged in row-major order). See the API reference documentation for details.
?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing capabilities directly on user devices, allowing them to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method based on your needs to integrate the model pipeline into subsequent AI applications.

## 4. Custom Development
If the default model weights provided by the image anomaly detection pipeline are not satisfactory in terms of accuracy or speed for your specific scenario, you can attempt to <b>further fine-tune the existing models using your own domain-specific or application-specific data</b> to improve the recognition performance of the image anomaly detection pipeline in your scenario.

### 4.1 Model Fine-Tuning
Since the image anomaly detection pipeline includes an unsupervised image anomaly detection module, if the pipeline's performance does not meet expectations, you need to refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/anomaly_detection.en.md) section in the [Unsupervised Anomaly Detection Module Development Guide](../../../module_usage/tutorials/cv_modules/anomaly_detection.en.md) and use your private dataset to fine-tune the image anomaly detection model.

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain the local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file and enter the local path of the fine-tuned model weights into the `model_dir` field in the pipeline configuration file:

```python
pipeline_name: anomaly_detection

SubModules:
  AnomalyDetection:
    module_name: anomaly_detection
    model_name: STFPM
    model_dir: null  # Replace with the fine-tuned image anomaly detection model weight path
    batch_size: 1
```

Subsequently, refer to the command line method or Python script method in [2. Quick Start]() to load the modified pipeline configuration file.

##  5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices such as NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device`</b> parameter to achieve seamless switching between different hardware.

For example, if you are using Ascend NPU for image anomaly detection pipeline inference, the Python command used is:

```bash
paddlex --pipeline anomaly_detection --input uad_grid.png --device npu:0
```

If you want to use the image anomaly detection pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
