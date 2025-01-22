---
comments: true
---

# Seal Recognition Pipeline Tutorial

## 1. Introduction to the Seal Recognition Pipeline
Seal recognition is a technology that automatically extracts and recognizes seal content from documents or images. The recognition of seal is part of document processing and has various applications in many scenarios, such as contract comparison, inventory access approval, and invoice reimbursement approval.

<img src="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/PP-ChatOCRv3_doc_seal/01.png">

The <b>Seal Recognition</b> pipeline includes a layout area analysis module, a seal detection module, and a text recognition module.

<b>If you prioritize model accuracy, please choose a model with higher accuracy. If you prioritize inference speed, please choose a model with faster inference. If you prioritize model storage size, please choose a model with a smaller storage footprint.</b>


<p><b>Layout Analysis Module Models:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Trained Model</a></td>
<td>86.8</td>
<td>13.0</td>
<td>91.3</td>
<td>7.4</td>
<td>An efficient layout area localization model trained on the PubLayNet dataset based on PicoDet-1x can locate five types of areas, including text, titles, tables, images, and lists.</td>
</tr>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet_layout_1x_table_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">Trained Model</a></td>
<td>95.7</td>
<td>12.623</td>
<td>90.8934</td>
<td>7.4 M</td>
<td>An efficient layout area localization model trained on the PubLayNet dataset based on PicoDet-1x can locate one type of tables.</td>
</tr>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Trained Model</a></td>
<td>87.1</td>
<td>13.5</td>
<td>45.8</td>
<td>4.8</td>
<td>An high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Trained Model</a></td>
<td>70.3</td>
<td>13.6</td>
<td>46.2</td>
<td>4.8</td>
<td>A high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Trained Model</a></td>
<td>89.3</td>
<td>15.7</td>
<td>159.8</td>
<td>22.6</td>
<td>An efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Trained Model</a></td>
<td>79.9</td>
<td>17.2</td>
<td>160.2</td>
<td>22.6</td>
<td>A efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Trained Model</a></td>
<td>95.9</td>
<td>114.6</td>
<td>3832.6</td>
<td>470.1</td>
<td>A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Trained Model</a></td>
<td>92.6</td>
<td>115.1</td>
<td>3827.2</td>
<td>470.2</td>
<td>A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
</tbody>
</table>
<p><b>Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built layout region analysis dataset, containing 10,000 images of common document types, including English and Chinese papers, magazines, research reports, etc. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Seal Detection Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Trained Model</a></td>
<td>98.21</td>
<td>84.341</td>
<td>2425.06</td>
<td>109</td>
<td>PP-OCRv4's server-side seal detection model, featuring higher accuracy, suitable for deployment on better-equipped servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Trained Model</a></td>
<td>96.47</td>
<td>10.5878</td>
<td>131.813</td>
<td>4.6</td>
<td>PP-OCRv4's mobile seal detection model, offering higher efficiency, suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are evaluated on a self-built dataset containing 500 circular seal images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Text Recognition Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>Average Recognition Accuracy (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Trained Model</a></td>
<td>78.20</td>
<td>7.95018</td>
<td>46.7868</td>
<td>10.6 M</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Trained Model</a></td>
<td>79.20</td>
<td>7.19439</td>
<td>140.179</td>
<td>71.2 M</td>
</tr>
</tbody>
</table>
<p><b>Note: The evaluation set for the above accuracy indicators is a self-built Chinese dataset from PaddleOCR, covering various scenarios such as street scenes, web images, documents, and handwriting. The text recognition subset includes 11,000 images. The GPU inference time for all models above is based on an NVIDIA Tesla T4 machine with a precision type of FP32. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads, and the precision type is also FP32.</b></p>

## 2.  Quick Start
The pre trained model production line provided by PaddleX can quickly experience the effect. You can experience the effect of the seal recognition production line online, or use the command line or Python locally to experience the effect of the seal recognition production line.


Before using the seal recognition production line locally, please ensure that you have completed the wheel package installation of PaddleX according to the  [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 2.1 Command line experience
One command can quickly experience the effect of seal recognition production line, use [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png), and replace ` --input ` with the local path for prediction

```
paddlex --pipeline seal_recognition --input seal_text_det.png --device gpu:0 --save_path output
```

Parameter description:

```
--Pipeline: Production line name, here is the seal recognition production line
--Input: The local path or URL of the input image to be processed
--The GPU serial number used by the device (e.g. GPU: 0 indicates the use of the 0th GPU, GPU: 1,2 indicates the use of the 1st and 2nd GPUs), or the CPU (-- device CPU) can be selected for use
```

When executing the above Python script, the default seal recognition production line configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details><summary>  👉 Click to expand</summary>

<pre><code class="language-bash">paddlex --get_pipeline_config seal_recognition
</code></pre>
<p>After execution, the seal recognition production line configuration file will be saved in the current path. If you want to customize the save location, you can execute the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code class="language-bash">paddlex --get_pipeline_config seal_recognition --save_path ./my_path --save_path output
</code></pre>
<p>After obtaining the production line configuration file, you can replace '-- pipeline' with the configuration file save path to make the configuration file effective. For example, if the configuration file save path is <code>/ seal_recognition.yaml</code>， Just need to execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./seal_recognition.yaml --input seal_text_det.png --save_path output
</code></pre>
<p>Among them, parameters such as <code>--model</code> and <code>--device</code> do not need to be specified and will use the parameters in the configuration file. If the parameters are still specified, the specified parameters will prevail.</p></details>

After running, the result obtained is:

<details><summary>  👉 Click to expand</summary>

<pre><code>
{'input_path': PosixPath('/root/.paddlex/temp/tmpa8eqnpus.png'), 'layout_result': {'input_path': PosixPath('/root/.paddlex/temp/tmpa8eqnpus.png'), 'boxes': [{'cls_id': 2, 'label': 'seal', 'score': 0.9813321828842163, 'coordinate': [0, 5.1820183, 639.59314, 637.7533]}]}, 'ocr_result': {'dt_polys': [array([[166, 468],
                        [206, 503],
                    [249, 523],
                    [312, 535],
                    [364, 529],
                    [390, 521],
                    [428, 505],
                    [465, 476],
                    [468, 474],
                    [473, 474],
                    [476, 475],
                    [478, 477],
                    [508, 507],
                    [510, 510],
                    [511, 514],
                    [509, 518],
                    [507, 521],
                    [458, 559],
                    [455, 560],
                    [399, 584],
                    [399, 584],
                    [369, 591],
                    [367, 592],
                    [308, 597],
                    [305, 596],
                    [240, 584],
                    [239, 584],
                    [220, 577],
                    [169, 552],
                    [166, 551],
                    [120, 510],
                    [117, 507],
                    [116, 503],
                    [117, 499],
                    [121, 495],
                    [153, 468],
                    [156, 467],
                    [161, 467]]), array([[439, 444],
                    [443, 444],
                    [446, 446],
                    [448, 448],
                    [450, 451],
                    [450, 454],
                    [448, 498],
                    [448, 502],
                    [445, 505],
                    [442, 507],
                    [439, 507],
                    [399, 505],
                    [196, 506],
                    [192, 505],
                    [189, 503],
                    [187, 500],
                    [187, 497],
                    [186, 458],
                    [186, 456],
                    [187, 451],
                    [188, 448],
                    [192, 444],
                    [194, 444],
                    [198, 443]]), array([[463, 347],
                    [468, 347],
                    [472, 350],
                    [474, 353],
                    [476, 360],
                    [477, 425],
                    [476, 429],
                    [474, 433],
                    [470, 436],
                    [466, 438],
                    [463, 438],
                    [175, 439],
                    [170, 438],
                    [166, 435],
                    [163, 432],
                    [161, 426],
                    [161, 361],
                    [161, 356],
                    [163, 352],
                    [167, 349],
                    [172, 347],
                    [184, 346],
                    [186, 346]]), array([[325,  38],
                    [485,  91],
                    [489,  94],
                    [493,  96],
                    [587, 225],
                    [588, 230],
                    [589, 234],
                    [592, 384],
                    [591, 389],
                    [588, 393],
                    [585, 397],
                    [581, 399],
                    [576, 399],
                    [572, 398],
                    [508, 380],
                    [503, 379],
                    [499, 375],
                    [498, 370],
                    [497, 367],
                    [493, 258],
                    [428, 171],
                    [421, 165],
                    [323, 136],
                    [225, 165],
                    [207, 175],
                    [144, 260],
                    [141, 365],
                    [141, 370],
                    [138, 374],
                    [134, 378],
                    [131, 379],
                    [ 66, 398],
                    [ 61, 398],
                    [ 56, 398],
                    [ 52, 395],
                    [ 48, 391],
                    [ 47, 386],
                    [ 47, 384],
                    [ 47, 235],
                    [ 48, 230],
                    [ 50, 226],
                    [146,  96],
                    [151,  92],
                    [154,  91],
                    [315,  38],
                    [320,  37]])], 'dt_scores': [0.99375725701319, 0.9871711582010613, 0.9937523531067023, 0.9911629231838204], 'rec_text': ['5263647368706', '吗繁物', '发票专天津君和缘商贸有限公司'], 'rec_score': [0.9933745265007019, 0.998288631439209, 0.9999362230300903, 0.9923253655433655], 'input_path': PosixPath('/Users/chenghong0temp/tmpa8eqnpus.png')}, 'src_file_name': 'https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png', 'page_id': 0}
</code></pre></details>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/seal_recognition/03.png">

The visualized image not saved by default. You can customize the save path through `--save_path`, and then all results will be saved in the specified path.


###  2.2 Python Script Integration
A few lines of code can complete the fast inference of the production line. Taking the seal recognition production line as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="seal_recognition")

output = pipeline.predict("seal_text_det.png")
for res in output:
    res.print()
    res.save_to_img("./output/") # Save the results in img
```

The result obtained is the same as the command line method.

In the above Python script, the following steps were executed:

（1）Instantiate the  production line object using `create_pipeline`: Specific parameter descriptions are as follows:

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
<td>The name of the production line or the path to the production line configuration file. If it is the name of the production line, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for production line model inference. Supports: "gpu", "cpu".</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available if the production line supports it.</td>
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
<td>Supports passing in the URL of the file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png">Example</a>.</td>
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
<td>save_to_img</td>
<td>Save the results as an img format file</td>
<td><code>- save_path</code>: str, the path to save the file. When it's a directory, the saved file name will be consistent with the input file type;</td>
</tr>
</tbody>
</table>
Where `save_to_img` can save visualization results (including OCR result images, layout analysis result images).

If you have a configuration file, you can customize the configurations of the seal recognition  pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved in `/ my_path/seal_recognition.yaml` ， Then only need to execute:


```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/seal_recognition.yaml")
output = pipeline.predict("seal_text_det.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存可视化结果
```

## 3. Development integration/deployment
If the production line can meet your requirements for inference speed and accuracy, you can directly develop integration/deployment.

If you need to directly apply the production line to your Python project, you can refer to the example code in [2.2.2 Python scripting] (# 222 python scripting integration).

In addition, PaddleX also offers three other deployment methods, detailed as follows:

🚀 ** High performance deployment: In actual production environments, many applications have strict standards for the performance indicators of deployment strategies, especially response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin aimed at deep performance optimization of model inference and pre-processing, achieving significant acceleration of end-to-end processes. For a detailed high-performance deployment process, please refer to the [PaddleX High Performance Deployment Guide] (../../../pipelin_deploy/high_performance_deploy. md).

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
<p>Obtain seal recognition results from an image.</p>
<p><code>POST /seal-recognition</code></p>
<ul>
<li>Request body properties:</li>
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
<td><code>file</code></td>
<td><code>string</code></td>
<td>The URL of an image file or PDF file accessible by the server, or the Base64 encoded result of the content of the above-mentioned file types. For PDF files with more than 10 pages, only the content of the first 10 pages will be used.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code></td>
<td>File type. <code>0</code> indicates a PDF file, and <code>1</code> indicates an image file. If this property is not present in the request body, the file type will be inferred based on the URL.</td>
<td>No</td>
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
<td><code>sealRecResults</code></td>
<td><code>array</code></td>
<td>Seal recognition results. The array length is 1 (for image input) or the smaller of the number of document pages and 10 (for PDF input). For PDF input, each element in the array represents the processing result of each page in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>sealRecResults</code> is an <code>object</code> with the following properties:</p>
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
<td><code>texts</code></td>
<td><code>array</code></td>
<td>Positions, contents, and scores of texts.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code></td>
<td>Input image. The image is in JPEG format and encoded using Base64.</td>
</tr>
<tr>
<td><code>layoutImage</code></td>
<td><code>string</code></td>
<td>Layout area detection result image. The image is in JPEG format and encoded using Base64.</td>
</tr>
<tr>
<td><code>ocrImage</code></td>
<td><code>string</code></td>
<td>OCR result image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>texts</code> is an <code>object</code> with the following properties:</p>
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
<td><code>poly</code></td>
<td><code>array</code></td>
<td>Text position. Elements in the array are the vertex coordinates of the polygon enclosing the text.</td>
</tr>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>Text content.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>Text recognition score.</td>
</tr>
</tbody>
</table></details>

<details><summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/seal-recognition&quot;
file_path = &quot;./demo.jpg&quot;

with open(file_path, &quot;rb&quot;) as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode(&quot;ascii&quot;)

payload = {&quot;file&quot;: file_data, &quot;fileType&quot;: 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
for i, res in enumerate(result[&quot;sealRecResults&quot;]):
    print(&quot;Detected texts:&quot;)
    print(res[&quot;texts&quot;])
    layout_img_path = f&quot;layout_{i}.jpg&quot;
    with open(layout_img_path, &quot;wb&quot;) as f:
        f.write(base64.b64decode(res[&quot;layoutImage&quot;]))
    ocr_img_path = f&quot;ocr_{i}.jpg&quot;
    with open(ocr_img_path, &quot;wb&quot;) as f:
        f.write(base64.b64decode(res[&quot;ocrImage&quot;]))
    print(f&quot;Output images saved at {layout_img_path} and {ocr_img_path}&quot;)
</code></pre></details>
</details>
<br/>

## 4.  Secondary development
If the default model weights provided by the seal recognition production line are not satisfactory in terms of accuracy or speed in your scenario, you can try using your own specific domain or application scenario data to further fine tune the existing model to improve the recognition performance of the seal recognition production line in your scenario.

### 4.1 Model fine-tuning
Due to the fact that the seal recognition production line consists of three modules, the performance of the model production line may not be as expected due to any of these modules.

You can analyze images with poor recognition performance and refer to the following rules for analysis and model fine-tuning:

* If the seal area is incorrectly located within the overall layout, the layout detection module may be insufficient. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/layout_detection.en.md#customization) section in the [Layout Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/layout_detection.en.md) and use your private dataset to fine-tune the layout detection model.
* If there is a significant amount of text that has not been detected (i.e. text miss detection phenomenon), it may be due to the shortcomings of the text detection model. You need to refer to the [Secondary Development](../../../module_usage/tutorials/ocr_modules/seal_text_detection.en.md#customization) section in the [Seal Text Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/seal_text_detection.en.md) to fine tune the text detection model using your private dataset.
* If seal texts are undetected (i.e., text miss detection), the text detection model may be insufficient. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/text_recognition.en.md#customization) section in the [Text Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_recognition.en.md) and use your private dataset to fine-tune the text detection model.

* If many detected texts contain recognition errors (i.e., the recognized text content does not match the actual text content), the text recognition model requires further improvement. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/text_recognition.en.md#customization) section.

### 4.2 Model Application
After completing fine-tuning training using a private dataset, you can obtain a local model weight file.

If you need to use the fine tuned model weights, simply modify the production line configuration file and replace the local path of the fine tuned model weights with the corresponding position in the production line configuration file

```python
......
 Pipeline:
  layout_model: RT-DETR-H_layout_3cls #can be modified to the local path of the fine tuned model
  text_det_model: PP-OCRv4_server_seal_det  #can be modified to the local path of the fine tuned model
  text_rec_model: PP-OCRv4_server_rec #can be modified to the local path of the fine tuned model
  layout_batch_size: 1
  text_rec_batch_size: 1
  device: "gpu:0"
......
```
Subsequently, refer to the command line or Python script in the local experience to load the modified production line configuration file.

##  5.  Multiple hardware support
PaddleX supports various mainstream hardware devices such as Nvidia GPU, Kunlun Core XPU, Ascend NPU, and Cambrian MLU, and can seamlessly switch between different hardware devices by simply modifying the <b>`--device`</b> parameter.

For example, if you use Nvidia GPU for inference on a seal recognition production line, the Python command you use is:

```
paddlex --pipeline seal_recognition --input seal_text_det.png --device gpu:0 --save_path output
```

At this point, if you want to switch the hardware to Ascend NPU, simply modify the ` --device ` in the Python command to NPU:

```
paddlex --pipeline seal_recognition --input seal_text_det.png --device npu:0 --save_path output
```

If you want to use the seal recognition production line on a wider range of hardware, please refer to the [PaddleX Multi Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md)。
