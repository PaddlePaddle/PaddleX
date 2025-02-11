---
comments: true
---

# General Video Classification Pipeline User Guide

## 1. Introduction to General Video Classification Pipeline
Video classification is a technology that assigns video clips to predefined categories. It is widely used in action recognition, event detection, and content recommendation. Video classification can identify various dynamic events and scenes, such as sports activities, natural phenomena, traffic conditions, etc., and classify them based on their characteristics. By using deep learning models, especially the combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN), video classification can automatically extract spatiotemporal features from videos and perform accurate classification. This technology has important applications in video surveillance, media retrieval, and personalized recommendation systems.

The general video classification pipeline is used to solve video classification tasks by extracting theme and category information from videos and outputting them as labels. This pipeline integrates the industry-renowned PP-TSM and PP-TSMv2 video classification systems, supporting the recognition of 400 video categories. Based on this pipeline, accurate classification of video content can be achieved, covering various fields such as media, security, education, and transportation. This pipeline also provides flexible service deployment options, supporting multiple programming languages on various hardware. Additionally, this pipeline offers secondary development capabilities, allowing you to train and fine-tune models on your own dataset, with seamless integration of the trained models.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/video_classification/01.jpg">

<b>The general video classification pipeline includes a video classification module. If you prioritize model accuracy, choose a model with higher accuracy; if you prioritize inference speed, choose a model with faster inference speed; if you prioritize storage size, choose a model with a smaller storage size.</b>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top1 Acc(%)</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-TSM-R50_8frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSM-R50_8frames_uniform_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSM-R50_8frames_uniform_pretrained.pdparams">Trained Model</a></td>
<td>74.36</td>
<td>93.4 M</td>
<td rowspan="1">
PP-TSM is a video classification model developed by Baidu PaddlePaddle's Vision Team. This model is optimized based on the ResNet-50 backbone network and undergoes model tuning in six aspects: data augmentation, network structure fine-tuning, training strategies, Batch Normalization (BN) layer optimization, pre-trained model selection, and model distillation. Under the center crop evaluation method, its accuracy on Kinetics-400 is improved by 3.95 points compared to the original paper's implementation.
</td>
</tr>

<tr>
<td>PP-TSMv2-LCNetV2_8frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSMv2-LCNetV2_8frames_uniform_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSMv2-LCNetV2_8frames_uniform_pretrained.pdparams">Trained Model</a></td>
<td>71.71</td>
<td>22.5 M</td>
<td rowspan="2">PP-TSMv2 is a lightweight video classification model optimized based on the CPU-oriented model PP-LCNetV2. It undergoes model tuning in seven aspects: backbone network and pre-trained model selection, data augmentation, TSM module tuning, input frame number optimization, decoding speed optimization, DML distillation, and LTA module. Under the center crop evaluation method, it achieves an accuracy of 75.16%, with an inference speed of only 456ms on the CPU for a 10-second video input.</td>
</tr>
<tr>
<td>PP-TSMv2-LCNetV2_16frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSMv2-LCNetV2_16frames_uniform_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSMv2-LCNetV2_16frames_uniform_pretrained.pdparams">Trained Model</a></td>
<td>73.11</td>
<td>22.5 M</td>
</tr>

</table>

<p><b>Note: The above accuracy metrics refer to Top-1 Accuracy on the <a href="https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/en/dataset/k400.md">K400</a> validation set. </b></p></details>

## 2. Quick Start

PaddleX supports experiencing the pipeline's effects locally using command line or Python.

Before using the general video classification pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the PaddleX Local Installation Guide.

#### 2.1 Command Line Experience
You can quickly experience the video classification pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4) and replace `--input` with your local path for prediction.

```bash
paddlex --pipeline video_classification \
    --input general_video_classification_001.mp4 \
    --topk 5 \
    --save_path ./output \
    --device gpu:0
```

The relevant parameter descriptions can be found in the parameter descriptions in [2.2 Integration via Python Script](#22-integration-with-python-script).

After running, the result will be printed to the terminal, as follows:

```bash
{'res': {'input_path': 'general_video_classification_001.mp4', 'class_ids': array([  0, ..., 162], dtype=int32), 'scores': [0.91997, 0.07052, 0.00237, 0.00214, 0.00158], 'label_names': ['abseiling', 'rock_climbing', 'climbing_tree', 'riding_mule', 'ice_climbing']}}
```

The explanation of the result parameters can refer to the result explanation in [2.2 Integration with Python Script](#22-integration-with-python-script).


The visualization results are saved under `save_path`, and the visualization result for video classification is as follows:
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/video_classification/02.jpg" style="width: 70%">


#### 2.2 Integration with Python Script
* The above command line is for quickly experiencing and viewing the effect. Generally speaking, in a project, it is often necessary to integrate through code. You can complete the rapid inference of the production line with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="video_classification")

output = pipeline.predict("general_video_classification_001.mp4", topk=5)
for res in output:
    res.print()
    res.save_to_video(save_path="./output/")
    res.save_to_json(save_path="./output/")
```

In the above Python script, the following steps are executed:

(1) The video classification pipeline object is instantiated via `create_pipeline()`. The specific parameter descriptions are as follows:

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
<td>The name of the pipeline or the path to the pipeline configuration file. If it is a pipeline name, it must be supported by PaddleX.</td>
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
<td>The inference device for the pipeline. It supports specifying the specific card number of the GPU, such as "gpu:0", other hardware card numbers, such as "npu:0", and CPU as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, which is only available when the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) The `predict()` method of the general video classification pipeline object is called for inference prediction. This method returns a `generator`. Here are the parameters and their descriptions for the `predict()` method:

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
<tbody>
<tr>
<td><code>input</code></td>
<td>The data to be predicted, supporting multiple input types (required).</td>
<td><code>str|list</code></td>
<td>
<ul>
  <li><b>str</b>: The local path of the video file, such as <code>/root/data/video.mp4</code>; <b>URL link</b>, such as the network URL of the video file: <a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4">Example</a>; <b>Local directory</b>, the directory must contain the videos to be predicted, such as the local path: <code>/root/data/</code></li></url>
  <li><b>List</b>: The elements of the list must be of the above types, such as <code>[\"/root/data/video1.mp4\", \"/root/data/video2.mp4\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
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
  <li><b>CPU</b>: For example, <code>cpu</code> indicates using the CPU for inference;</li>
  <li><b>GPU</b>: For example, <code>gpu:0</code> indicates using the first GPU for inference;</li>
  <li><b>NPU</b>: For example, <code>npu:0</code> indicates using the first NPU for inference;</li>
  <li><b>XPU</b>: For example, <code>xpu:0</code> indicates using the first XPU for inference;</li>
  <li><b>MLU</b>: For example, <code>mlu:0</code> indicates using the first MLU for inference;</li>
  <li><b>DCU</b>: For example, <code>dcu:0</code> indicates using the first DCU for inference;</li>
  <li><b>None</b>: If set to <code>None</code>, the value initialized for the pipeline will be used by default. During initialization, the local GPU 0 will be prioritized. If it is not available, the CPU will be used.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>topk</code></td>
<td>The top <code>topk</code> classes and their corresponding classification probabilities in the prediction results.</td>
<td><code>int|None</code></td>
<td>
<ul>
    <li><b>int</b>: Any integer greater than <code>0</code>.</li>
    <li><b>None</b>: If set to <code>None</code>, the value initialized for the pipeline will be used by default, which is <code>1</code>.</li>
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
<tr>
<td><code>save_to_video()</code></td>
<td>Save the result as a video file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file, supports directory or file path</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the result to the terminal, with the printed content explained as follows:

    - `input_path`: `(str)` The input path of the video to be predicted
    - `class_ids`: `(numpy.ndarray)` List of IDs for video classification
    - `scores`: `(List[float])` List of confidence scores for video classification
    - `label_names`: `(List[str])` List of categories for video classification

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_video_basename}_res.json`; if specified as a file, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, the `numpy.array` types will be converted to lists.

- Calling the `save_to_video()` method will save the visualization results to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_video_basename}_res.{your_video_extension}`; if specified as a file, it will be saved directly to that file. (The production line usually contains multiple result videos, so it is not recommended to specify a specific file path directly, as multiple videos will be overwritten and only the last video will be retained)

* Additionally, it also supports obtaining visualized videos and prediction results through attributes, as follows:

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
<tr>
<td rowspan = "2"><code>video</code></td>
<td rowspan = "2">Get the visualized video in <code>dict</code> format</td>
</tr>
</table>

- The prediction result obtained by the `json` attribute is a dict type of data, with content consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `video` attribute is a dictionary type of data. The key is `res`, and the corresponding value is a tuple. The first element of the tuple is the visualized video array with dimensions (number of frames, video height, video width, number of channels); the second element is the frame rate.

In addition, you can obtain the video classification production line configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config video_classification --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the video classification pipeline. Simply modify the value of the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file. An example is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/video_classification.yaml")

output = pipeline.predict("general_video_classification_001.mp4", topk=5)
for res in output:
    res.print()
    res.save_to_video(save_path="./output/")
    res.save_to_json(save_path="./output/")
```

<b>Note:</b> The parameters in the configuration file are for pipeline initialization. If you wish to change the initialization parameters of the general video classification pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in the configuration file by specifying the path with `--pipeline`.

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the pipeline directly to your Python project, you can refer to the example code in [2.2 Integration with Python Script](#22-integration-with-python-script).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. For this purpose, PaddleX provides a high-performance inference plugin, aimed at deeply optimizing the performance of model inference and pre/post-processing, significantly accelerating the end-to-end process. For detailed high-performance inference procedures, please refer to [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service Deployment</b>: Service deployment is a common form of deployment in actual production environments. By encapsulating the inference function as a service, clients can access these services via network requests to obtain inference results. PaddleX supports multiple pipeline service deployment solutions. For detailed pipeline service deployment procedures, please refer to [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references and multi-language service invocation examples for basic service deployment:

<details><summary>API Reference</summary>

<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body has the following attributes:</li>
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
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed at <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed at <code>"Success"</code>.</td>
</tr>
</tbody>
</table>
<p>The response body may also have a <code>result</code> attribute, which is an <code>object</code> containing the operation result information.</p>
<ul>
<li>When the request is not processed successfully, the response body has the following attributes:</li>
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
<p>Perform video classification.</p>
<p><code>POST /video-classification</code></p>
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
<td><code>video</code></td>
<td><code>string</code></td>
<td>The URL of the video file accessible by the server or the Base64 encoded result of the video file content.</td>
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
<p>The attributes of <code>inferenceParams</code> are as follows:</p>
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
<td><code>topK</code></td>
<td><code>integer</code></td>
<td>Only the top <code>topK</code> categories with the highest scores will be retained in the results.</td>
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
<td><code>categories</code></td>
<td><code>array</code></td>
<td>Video category information.</td>
</tr>
<tr>
<td><code>video</code></td>
<td><code>string</code></td>
<td>The video classification result image. The video is in JPEG format and encoded in Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>categories</code> is an <code>object</code> with the following attributes:</p>
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

#include &quot;cpp-httplib/httplib.h&quot; // https://github.com/Huiyicc/cpp-httplib
#include &quot;nlohmann/json.hpp&quot; // https://github.com/nlohmann/json
#include &quot;base64.hpp&quot; // https://github.com/tobiaslocker/base64

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

```yaml
...
SubModules:
  VideoClassification:
    module_name: video_classification
    model_name: PP-TSMv2-LCNetV2_8frames_uniform
    model_dir: null # Replace with the fine-tuned video classification model weights path
    batch_size: 1
    topk: 1

...
```

Subsequently, refer to the command-line method or Python script method in the local experience to load the modified production line configuration file.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware devices.

For example, if you use Ascend NPU for video classification in the production line, the Python command used is:

```bash
paddlex --pipeline video_classification \
    --input general_video_classification_001.mp4 \
    --topk 5 \
    --save_path ./output \
    --device npu:0
```

Of course, you can also specify the hardware device when calling `create_pipeline()` or `predict()` in a Python script.

If you want to use the General Video Classification Production Line on a wider variety of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
