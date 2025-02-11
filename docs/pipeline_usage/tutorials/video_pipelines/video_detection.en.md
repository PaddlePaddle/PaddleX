---
comments: true
---

# General Video Detection Pipeline User Guide

## 1. Introduction to General Video Detection Pipeline

Video detection is a technology that identifies and locates specific objects or events in video content. It is widely used in fields such as security surveillance, traffic management, and behavior analysis. This technology can capture and analyze dynamic changes in videos in real-time, such as human activities, vehicle movements, and abnormal events. Through deep learning models, video detection can efficiently extract spatial and temporal features from videos, achieving accurate recognition and localization. Video detection not only enhances the intelligence of surveillance systems but also provides important support for improving safety and operational efficiency. With the development of technology, video detection will play a key role in more scenarios.

<img src="https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/images/yowo.jpg">

<b>The video detection pipeline</b><b> includes a video detection module</b> with the following models.

<p><b>Video Detection Module (Optional):</b></p>

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Frame-mAP(@ IoU 0.5)</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>YOWO</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOWO_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOWO_pretrained.pdparams">训练模型</a></td>
<td>80.94</td>
<td>462.891M</td>
<td rowspan="1">
YOWO is a single-stage network with two branches. One branch extracts spatial features of the keyframe (i.e., the current frame) through 2D-CNN, while the other branch captures spatiotemporal features of the clip composed of previous frames through 3D-CNN. To accurately aggregate these features, YOWO uses a channel fusion and attention mechanism, maximizing the utilization of inter-channel dependencies. Finally, the fused features are used for frame-level detection.
</td>
</tr>

</table>

<p><b>Note: The above accuracy metrics are Frame-mAP (@ IoU 0.5) on the <a href="http://www.thumos.info/download.html">UCF101-24</a> test dataset. All model GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision, and CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## 2. Quick Start

PaddleX supports experiencing the pipeline's effects locally using command line or Python.

Before using the general video detection pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 2.1 Local Experience

#### 2.1 Command Line Experience
You can quickly experience the video detection pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi) and replace `--input` with your local path for prediction.

```bash
paddlex --pipeline video_detection --input HorseRiding.avi --device gpu:0 --save_path output
```

The relevant parameter description can be found in the parameter description in [2.1.2 Integration via Python Script](#212-python脚本方式集成).

After running, the result will be printed to the terminal, as follows:

<details><summary>👉Click to Expand</summary>

```bash
{'input_path': 'HorseRiding.avi', 'result': [[[[110, 40, 170, 171], 0.8385784886274905, 'HorseRiding']], [[[112, 31, 168, 167], 0.8587647461352432, 'HorseRiding']], [[[106, 28, 164, 165], 0.8579590929730969, 'HorseRiding']], [[[106, 24, 165, 171], 0.8743957465404151, 'HorseRiding']], [[[107, 22, 165, 172], 0.8488322619908999, 'HorseRiding']], [[[112, 22, 173, 171], 0.8446755521458691, 'HorseRiding']], [[[115, 23, 177, 176], 0.8454028365262367, 'HorseRiding']], [[[117, 22, 178, 179], 0.8484261880748285, 'HorseRiding']], [[[117, 22, 181, 181], 0.8319480115446183, 'HorseRiding']], [[[117, 39, 182, 183], 0.820551099084625, 'HorseRiding']], [[[117, 41, 183, 185], 0.8202395831914338, 'HorseRiding']], [[[121, 47, 185, 190], 0.8261058921745246, 'HorseRiding']], [[[123, 46, 188, 196], 0.8307278306829033, 'HorseRiding']], [[[125, 44, 189, 197], 0.8259781361122833, 'HorseRiding']], [[[128, 47, 191, 195], 0.8227593229866699, 'HorseRiding']], [[[127, 44, 192, 193], 0.8205373129456817, 'HorseRiding']], [[[129, 39, 192, 185], 0.8223318812628619, 'HorseRiding']], [[[127, 31, 196, 179], 0.8501208612019866, 'HorseRiding']], [[[128, 22, 193, 171], 0.8315708410681566, 'HorseRiding']], [[[130, 22, 192, 169], 0.8318588228062005, 'HorseRiding']], [[[132, 18, 193, 170], 0.8310494469100611, 'HorseRiding']], [[[132, 18, 194, 172], 0.8302132445350239, 'HorseRiding']], [[[133, 18, 194, 176], 0.8339063714162727, 'HorseRiding']], [[[134, 26, 200, 183], 0.8365876380675275, 'HorseRiding']], [[[133, 16, 198, 182], 0.8395230321418268, 'HorseRiding']], [[[133, 17, 199, 184], 0.8198139782724922, 'HorseRiding']], [[[140, 28, 204, 189], 0.8344166596681291, 'HorseRiding']], [[[139, 27, 204, 187], 0.8412694521771158, 'HorseRiding']], [[[139, 28, 204, 185], 0.8500098862888805, 'HorseRiding']], [[[135, 19, 199, 179], 0.8506627974981384, 'HorseRiding']], [[[132, 15, 201, 178], 0.8495054272547193, 'HorseRiding']], [[[136, 14, 199, 173], 0.8451630721500223, 'HorseRiding']], [[[136, 12, 200, 167], 0.8366456814214907, 'HorseRiding']], [[[133, 8, 200, 168], 0.8457252233401213, 'HorseRiding']], [[[131, 7, 197, 162], 0.8400586356358062, 'HorseRiding']], [[[131, 8, 195, 163], 0.8320492682901985, 'HorseRiding']], [[[129, 4, 194, 159], 0.8298043752822792, 'HorseRiding']], [[[127, 5, 194, 162], 0.8348390851948722, 'HorseRiding']], [[[125, 7, 190, 164], 0.8299688814865505, 'HorseRiding']], [[[125, 6, 191, 164], 0.8303107088154711, 'HorseRiding']], [[[123, 8, 190, 168], 0.8348342187965798, 'HorseRiding']], [[[125, 14, 189, 170], 0.8356523950497134, 'HorseRiding']], [[[127, 18, 191, 171], 0.8392671764931521, 'HorseRiding']], [[[127, 30, 193, 178], 0.8441704160826191, 'HorseRiding']], [[[128, 18, 190, 181], 0.8438125326146775, 'HorseRiding']], [[[128, 12, 189, 186], 0.8390128962093542, 'HorseRiding']], [[[129, 15, 190, 185], 0.8471056476788448, 'HorseRiding']], [[[129, 16, 191, 184], 0.8536121834731034, 'HorseRiding']], [[[129, 16, 192, 185], 0.8488154629800881, 'HorseRiding']], [[[128, 15, 194, 184], 0.8417711698421471, 'HorseRiding']], [[[129, 13, 195, 187], 0.8412510238991473, 'HorseRiding']], [[[129, 14, 191, 187], 0.8404350980083457, 'HorseRiding']], [[[129, 13, 190, 189], 0.8382891279858882, 'HorseRiding']], [[[129, 11, 187, 191], 0.8318282305903217, 'HorseRiding']], [[[128, 8, 188, 195], 0.8043430817880264, 'HorseRiding']], [[[131, 25, 193, 199], 0.826184954516826, 'HorseRiding']], [[[124, 35, 191, 203], 0.8270462809459467, 'HorseRiding']], [[[121, 38, 191, 206], 0.8350931715324705, 'HorseRiding']], [[[124, 41, 195, 212], 0.8331239341053625, 'HorseRiding']], [[[128, 42, 194, 211], 0.8343046153103657, 'HorseRiding']], [[[131, 40, 192, 203], 0.8309784496027532, 'HorseRiding']], [[[130, 32, 195, 202], 0.8316640083647542, 'HorseRiding']], [[[135, 30, 196, 197], 0.8272172409555161, 'HorseRiding']], [[[131, 16, 197, 186], 0.8388410406147955, 'HorseRiding']], [[[134, 15, 202, 184], 0.8485738297037244, 'HorseRiding']], [[[136, 15, 209, 182], 0.8529430205135213, 'HorseRiding']], [[[134, 13, 218, 182], 0.8601191479922718, 'HorseRiding']], [[[144, 10, 213, 183], 0.8591963099263467, 'HorseRiding']], [[[151, 12, 219, 184], 0.8617965108346937, 'HorseRiding']], [[[151, 10, 220, 186], 0.8631923599955371, 'HorseRiding']], [[[145, 10, 216, 186], 0.8800860885204287, 'HorseRiding']], [[[144, 10, 216, 186], 0.8858840451538228, 'HorseRiding']], [[[146, 11, 214, 190], 0.8773644144886106, 'HorseRiding']], [[[145, 24, 214, 193], 0.8605544385867248, 'HorseRiding']], [[[146, 23, 214, 193], 0.8727294882672254, 'HorseRiding']], [[[148, 22, 212, 198], 0.8713131467067079, 'HorseRiding']], [[[146, 29, 213, 197], 0.8579099324651196, 'HorseRiding']], [[[154, 29, 217, 199], 0.8547794072847914, 'HorseRiding']], [[[151, 26, 217, 203], 0.8641733722316758, 'HorseRiding']], [[[146, 24, 212, 199], 0.8613466257602624, 'HorseRiding']], [[[142, 25, 210, 194], 0.8492670944810214, 'HorseRiding']], [[[134, 24, 204, 192], 0.8428117300203049, 'HorseRiding']], [[[136, 25, 204, 189], 0.8486779356971397, 'HorseRiding']], [[[127, 21, 199, 179], 0.8513896296400709, 'HorseRiding']], [[[125, 10, 192, 192], 0.8510201771386576, 'HorseRiding']], [[[124, 8, 191, 192], 0.8493999019508465, 'HorseRiding']], [[[121, 8, 192, 193], 0.8487097098892171, 'HorseRiding']], [[[119, 6, 187, 193], 0.847543279648022, 'HorseRiding']], [[[118, 12, 190, 190], 0.8503535936620565, 'HorseRiding']], [[[122, 22, 189, 194], 0.8427901493276977, 'HorseRiding']], [[[118, 24, 188, 195], 0.8418835400352087, 'HorseRiding']], [[[120, 25, 188, 205], 0.847192725785284, 'HorseRiding']], [[[122, 25, 189, 207], 0.8444105420674433, 'HorseRiding']], [[[120, 23, 189, 208], 0.8470784016639392, 'HorseRiding']], [[[121, 23, 188, 205], 0.843428111269418, 'HorseRiding']], [[[117, 23, 186, 206], 0.8420809714166708, 'HorseRiding']], [[[119, 5, 199, 197], 0.8288265053231356, 'HorseRiding']], [[[121,
```

</details>

The explanation of the result parameters can refer to the result explanation in [2.1.2 Integration with Python Script](#212-integration-with-python-script).

The visualization results are saved under `save_path`, and the visualization results are as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/video_detection/HorseRiding_res.jpg">


#### 2.1.2 Integration with Python Script

The above command line is for quickly experiencing and viewing the effect. Generally speaking, in a project, it is often necessary to integrate through code. You can complete the rapid inference of the production line with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="video_detection")
output = pipeline.predict(input="HorseRiding.avi")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_video(save_path="./output/") ## 保存结果可视化视频
    res.save_to_json(save_path="./output/") ## 保存预测的结构化输出
```

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` instance to create a pipeline object. The specific parameter descriptions are as follows:

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
<td>None</td>
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
<td>Whether to enable high-performance inference. This is only available when the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the video detection pipeline object for inference prediction. This method will return a `generator`. Here are the parameters and their descriptions for the `predict()` method:

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
<td>The video data to be predicted, supports multiple input types (required).</td>
<td><code>Python str|list</code></td>
<td>
<ul>
  <li><b>str</b>: Local path of the video file: <code>/root/data/video.avi</code>; <b>URL link</b>, such as the network URL of the video file: <a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi">Example</a>; <b>Local directory</b>, the directory must contain the videos to be predicted, such as the local path: <code>/root/data/</code></li>
  <li><b>List</b>: The elements of the list must be of the above types, such as <code>[str, str]</code>, <code>[\"/root/data/video1.mp4\", \"/root/data/video2.avi\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
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
<td><code>nms_thresh</code></td>
<td>The IoU threshold parameter in the Non-Maximum Suppression (NMS) process</td>
<td><code>float|None</code></td>
<td>
<ul>
  <li><b>float</b>: A floating-point number greater than 0;</li>
  <li><b>None</b>: If set to <code>None</code>, the value initialized for the pipeline will be used by default, which is initialized to 0.4.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>score_thresh</code></td>
<td>The prediction confidence threshold</td>
<td><code>float|None</code></td>
<td>
<ul>
  <li><b>float</b>: A floating-point number greater than 0;</li>
  <li><b>None</b>: If set to <code>None</code>, the value initialized for the pipeline will be used by default, which is initialized to 0.8.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

(3) Process the prediction results. The prediction result for each sample is of the `dict` type and supports operations such as printing, saving as a video, and saving as a `json` file:

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
<td><code>save_to_video</code></td>
<td>Save the result as a video file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file, supports directory or file path</td>
<td>None</td>
</tr>
</table>

- Calling the `print()` method will print the result to the terminal, with the printed content explained as follows:

    - `input_path`: `(str)` The input path of the image to be predicted

    - `result`: `(List[List[List]])` Prediction results, where each list represents the prediction result of a frame, and each frame result includes the following content:

        - `[xmin, ymin, xmax, ymax]`: `(list)` Bounding box coordinates in the format [xmin, ymin, xmax, ymax], where (xmin, ymin) is the top-left coordinate and (xmax, ymax) is the bottom-right coordinate
        - `float`: Confidence score of the bounding box, a floating-point number
        - `str`: Category of the bounding box, a string

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}.json`; if specified as a file, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, the `numpy.array` types will be converted to lists.

- Calling the `save_to_video()` method will save the visualization results to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.{your_img_extension}`; if specified as a file, it will be saved directly to that file.

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
<tr>
</tr>
</table>

- The prediction result obtained by the `json` attribute is a dict type of data, with content consistent with the content saved by calling the `save_to_json()` method.

In addition, you can obtain the video_detection production line configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config video_detection --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the `video_detection` pipeline. Simply modify the value of the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file. An example is as follows:

For example, if your configuration file is saved at `./my_path/video_detection*.yaml`, you just need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/video_detection.yaml")
output = pipeline.predict(input="HorseRiding.avi")
for res in output:
    res.print()
    res.save_to_video("./output/")
    res.save_to_json("./output/")
```

<b>Note:</b> The parameters in the configuration file are for pipeline initialization. If you wish to change the initialization parameters of the video_detection pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Additionally, CLI prediction also supports passing in the configuration file by specifying the path with `--pipeline`.

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the pipeline directly to your Python project, you can refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

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
<td><code>logId</code></td>
<td><code>string</code></td>
<td>The UUID of the request.</td>
</tr>
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
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>The result of the operation.</td>
</tr>
</tbody>
</table>
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
<td><code>score_threshold</code></td>
<td><code>integer</code></td>
<td>Only boxes with scores higher than this threshold <code>score_threshold</code> will be retained in the results.</td>
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
<td>The video detection result image. The video is in JPEG format and encoded in Base64.</td>
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

<details><summary>Multi-language Service Invocation Example</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/video-classification"  # Service URL
video_path = "./demo.mp4"
output_video_path = "./out.mp4"

# Encode the local video using Base64
with open(video_path, "rb") as file:
    video_bytes = file.read()
    video_data = base64.b64encode(video_bytes).decode("ascii")

payload = {"video": video_data}  # Base64-encoded file content or video URL

# Call the API
response = requests.post(API_URL, json=payload)

# Process the response data
assert response.status_code == 200
result = response.json()["result"]
with open(output_video_path, "wb") as file:
    file.write(base64.b64decode(result["video"]))
print(f"Output video saved at {output_video_path}")
print("\nCategories:")
print(result["categories"])
</code></pre></details>
<details><summary>C++</summary>

<pre><code class="language-cpp">#include <iostream>
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client("localhost:8080");
    const std::string videoPath = "./demo.mp4";
    const std::string outputImagePath = "./out.mp4";

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    // Encode the local video using Base64
    std::ifstream file(videoPath, std::ios::binary | std::ios::ate);
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
    jsonObj["video"] = encodedImage;
    std::string body = jsonObj.dump();

    // Call the API
    auto response = client.Post("/video-classification", headers, body, "application/json");
    // Process the response data
    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        encodedImage = result["video"];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector<unsigned char> decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outputImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast<char*>(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout << "Output video saved at " << outputImagePath << std::endl;
        } else {
            std::cerr << "Unable to open file for writing: " << outputImagePath << std::endl;
        }

        auto categories = result["categories"];
        std::cout << "\nCategories:" << std::endl;
        for (const auto& category : categories) {
            std::cout << category << std::endl;
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
        String API_URL = "http://localhost:8080/video-classification"; // Service URL
        String videoPath = "./demo.mp4"; // Local video
        String outputImagePath = "./out.mp4"; // Output video

        // Encode the local video using Base64
        File file = new File(videoPath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String videoData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("video", videoData); // Base64-encoded file content or video URL

        // Create an OkHttpClient instance
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get("application/json; charset=utf-8");
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
                JsonNode result = resultNode.get("result");
                String base64Image = result.get("video").asText();
                JsonNode categories = result.get("categories");

                byte[] videoBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(videoBytes);
                }
                System.out.println("Output video saved at " + outputImagePath);
                System.out.println("\nCategories: " + categories.toString());
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
    outputImagePath := &quot;./out.mp4&quot;

    // Encode the local video in Base64
    videoBytes, err := ioutil.ReadFile(videoPath)
    if err != nil {
        fmt.Println(&quot;Error reading video file:&quot;, err)
        return
    }
    videoData := base64.StdEncoding.EncodeToString(videoBytes)

    payload := map[string]string{&quot;video&quot;: videoData} // Base64-encoded file content or video URL
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
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing video to file:&quot;, err)
        return
    }
    fmt.Printf(&quot;Image saved at %s.mp4\n&quot;, outputImagePath)
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
    static readonly string outputImagePath = &quot;./out.mp4&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // Encode the local video in Base64
        byte[] videoBytes = File.ReadAllBytes(videoPath);
        string video_data = Convert.ToBase64String(videoBytes);

        var payload = new JObject{ { &quot;video&quot;, video_data } }; // Base64-encoded file content or video URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        // Call the API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // Process the response data
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse[&quot;result&quot;][&quot;video&quot;].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($&quot;Output video saved at {outputImagePath}&quot;);
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
const outputImagePath = &quot;./out.mp4&quot;;

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'video': encodeImageToBase64(videoPath)  // Base64-encoded file content or video URL
  })
};

// Encode the local video in Base64
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// Call the API
axios.request(config)
.then((response) =&gt; {
    // Process the response data
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
If the default model weights provided by the general video detection pipeline are not satisfactory in terms of accuracy or speed for your specific scenario, you can attempt to <b>fine-tune</b> the existing model using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the general video detection pipeline in your scenario.

### 4.1 Model Fine-Tuning
Since the general video detection pipeline includes a video detection module, if the performance of the pipeline does not meet your expectations, you need to refer to the [Secondary Development](../../../module_usage/tutorials/video_modules/video_detection.en.md) section in the [Video Detection Module Development Tutorial](../../../module_usage/tutorials/video_modules/video_detection.en.md) and fine-tune the video detection model using your private dataset.

### 4.2 Model Application
After completing the fine-tuning with your private dataset, you will obtain the local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the path to the fine-tuned model weights with the corresponding location in the pipeline configuration file.

```
......
Pipeline:
  model: YOWO #可修改为微调后模型的本地路径
  device: "gpu"
  batch_size: 1
......
```

Subsequently, refer to the command-line method or Python script method in the local experience to load the modified production line configuration file.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware devices.

For example, if you use Ascend NPU for video detection in the production line, the command used is:

```bash
paddlex --pipeline video_detection --input HorseRiding.avi --device npu:0
```

If you want to use the General Video Detection Production Line on a wider variety of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
