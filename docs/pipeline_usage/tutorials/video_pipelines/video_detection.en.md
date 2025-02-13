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

The above command line is for quickly experiencing and viewing the effect. Generally speaking, in a project, it is often necessary to integrate through code. You can complete the rapid inference of the pipeline with just a few lines of code. The inference code is as follows:

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

In addition, you can obtain the video_detection pipeline configuration file and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

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
<li>Both the request body and the response body are JSON data (JSON objects).</li>
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
<td>Error code. Fixed at <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed at <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
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
<p>Classify videos.</p>
<p><code>POST /video-detection</code></p>
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
<td>The URL of the video file accessible by the server or the Base64-encoded content of the video file.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>nmsThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>nms_thresh</code> parameter description in the production <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>scoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Refer to the <code>score_thresh</code> parameter description in the production <code>predict</code> method.</td>
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
<td><code>frames</code></td>
<td><code>array</code></td>
<td>The detection result of each frame.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>frames</code> is an <code>object</code> with the following attributes:</p>
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
<td><code>index</code></td>
<td><code>integer</code></td>
<td>The frame number starting from 0.</td>
</tr>
<tr>
<td><code>detectedObjects</code></td>
<td><code>array</code></td>
<td>Information about the location, category, and other details of the detected objects.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>detectedObjects</code> is an <code>object</code> with the following attributes:</p>
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
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>The location of the detected object. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner.</td>
</tr>
<tr>
<td><code>categoryName</code></td>
<td><code>string</code></td>
<td>The name of the detected object category.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The score of the detected object.</td>
</tr>
</tbody>
</table>
</details>
<details><summary>Multi-language Service Call Example</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/video-detection" # Service URL
video_path = "./demo.mp4"

# Encode the local video using Base64
with open(video_path, "rb") as file:
    video_bytes = file.read()
    video_data = base64.b64encode(video_bytes).decode("ascii")

payload = {"video": video_data}  # Base64-encoded file content or video URL

# Call the API
response = requests.post(API_URL, json=payload)

# Process the API response
assert response.status_code == 200
result = response.json()["result"]
print("\nFrames:")
print(result["frames"])
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

Subsequently, refer to the command-line method or Python script method in the local experience to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware devices.

For example, if you use Ascend NPU for video detection in the pipeline, the command used is:

```bash
paddlex --pipeline video_detection --input HorseRiding.avi --device npu:0
```

If you want to use the General Video Detection pipeline on a wider variety of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
