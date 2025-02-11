---
comments: true
---

# Open-Vocabulary Object Detection Module Usage Tutorial

## I. Overview
Open-vocabulary object detection is an advanced object detection technology aimed at overcoming the limitations of traditional object detection. Traditional methods can only recognize objects within predefined categories, while open-vocabulary object detection allows models to identify objects not seen during training. By integrating natural language processing techniques and using text descriptions to define new categories, the model can recognize and locate these new objects. This makes object detection more flexible and generalizable, with significant application potential.

## II. List of Supported Models

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>GroundingDINO-T</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/GroundingDINO-T_infer.tar">Inference Model</a></td>
<td>49.4</td>
<td>64.4</td>
<td>253.72</td>
<td>1807.4</td>
<td>658.3</td>
<td rowspan="3">This is an open-vocabulary object detection model trained on the O365, GoldG, and Cap4M datasets. It uses Bert for text encoding and DINO for the visual model, with additional cross-modal fusion modules, achieving good performance in open-vocabulary object detection.</td>
</tr>
</table>

**Note: The above accuracy metrics are based on the COCO val2017 validation set mAP(0.5:0.95). All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision, while CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package first. For details, please refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

After installing the wheel package, you can perform inference for the open-vocabulary object detection module with just a few lines of code. You can switch models under this module at will, and you can also integrate the model inference of the open-vocabulary object detection module into your project. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_detection.jpg) to your local machine.

**Note:** Due to network issues, the above URLs may not be accessible. If you need to access these links, please check the validity of the URLs and try again. If the problem persists, it may be related to the links themselves or the network connection.

```python
from paddlex import create_model
model = create_model('GroundingDINO-T')
results = model.predict('open_vocabulary_detection.jpg', prompt='bus . walking man . rearview mirror .')
for res in results:
    res.print()
    res.save_to_img(f"./output/")
    res.save_to_json(f"./output/res.json")
```

After running, the result obtained is:

```bash
{'res': "{'input_path': 'open_vocabulary_detection.jpg', 'boxes': [{'coordinate': [112.10542297363281, 117.93667602539062, 514.35693359375, 382.10150146484375], 'label': 'bus', 'score': 0.9348853230476379}, {'coordinate': [264.1828918457031, 162.6674346923828, 286.8844909667969, 201.86187744140625], 'label': 'rearview mirror', 'score': 0.6022508144378662}, {'coordinate': [606.1133422851562, 254.4973907470703, 622.56982421875, 293.7867126464844], 'label': 'walking man', 'score': 0.4384709894657135}, {'coordinate': [591.8192138671875, 260.2451171875, 607.3953247070312, 294.2210388183594], 'label': 'man', 'score': 0.3573091924190521}]}"}
```

The meanings of the parameters in the running results are as follows:
- `input_path`: The path of the input image to be predicted.
- `boxes`: Information about each predicted object.
  - `label`: The category name.
  - `score`: The prediction score.
  - `coordinate`: The coordinates of the prediction box, in the format <code>[xmin, ymin, xmax, ymax]</code>.

The visualization image is as follows:

<img src="https://raw.githubusercontent.com/BluebirdStory/PaddleX_doc_images/main/images/modules/open_vocabulary_detection/open_vocabulary_detection_res.jpg" />

Note: Due to network issues, the parsing of the above URL may not have been successful. If you need the content of this webpage, please check the validity of the URL and try again.

Related methods, parameters, and explanations are as follows:

* `create_model` instantiates an open-vocabulary object detection model (using `GroundingDINO-T` as an example). The specific explanations are as follows:
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
<td><code>model_name</code></td>
<td>The name of the model</td>
<td><code>str</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>The storage path of the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>thresholds</code></td>
<td>The filtering thresholds used by the model</td>
<td><code>dict/None</code></td>
<td>None</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the model parameters built into PaddleX will be used by default. If `model_dir` is specified, the user-defined model will be used.
* `thresholds` is the filtering threshold used by the model. The default is None, which means using the settings from the previous layer. The priority of parameter settings from high to low is: `predict parameter input > create_model initialization input > yaml configuration file setting`.
  * The GroundingDINO series of models require two thresholds during inference: box_threshold (default 0.3) and text_threshold (default 0.25). The parameter input format is `{"box_threshold": 0.3, "text_threshold": 0.25}`.

* The `predict()` method of the open-vocabulary object detection model is called for inference prediction. The parameters of the `predict()` method are `input`, `batch_size`, `thresholds`, and `prompt`, with specific explanations as follows:

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
<td>Data to be predicted, supporting multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>dict</code>/<code>list</code></td>
<td>
<ul>
  <li><b>Python variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
  <li><b>File path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
  <li><b>URL link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_detection.jpg">Example</a></li>
  <li><b>Local directory</b>, the directory should contain data files to be predicted, such as the local path: <code>/root/data/</code></li>
  <li><b>List</b>, the elements of the list must be the above types of data, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>Any integer</td>
<td>1</td>
</tr>
<tr>
<td><code>thresholds</code></td>
<td>The filtering thresholds used by the model</td>
<td><code>dict</code>/<code>None</code></td>
<td>
<ul>
  <li><b>None</b>, indicating the use of the settings from the previous layer. The priority of parameter settings from high to low is: <code>predict parameter input > create_model initialization input > yaml configuration file setting</code></li>
  <li><b>dict</b>, such as <code>{"box_threshold": 0.3, "text_threshold": 0.25}</code>, indicating that the box_threshold is set to 0.3 and the text_threshold is set to 0.25 during inference</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>prompt</code></td>
<td>The prompt used by the model for prediction</td>
<td><code>str</code></td>
<td>Any string</td>
<td>1</td>
</tr>
</table>

* The prediction results are processed, and the prediction result of each sample is of type `dict`, supporting operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Method Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">Print the results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the results as a file in JSON format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data and make it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the results as a file in image format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
</table>

* In addition, it also supports obtaining the visualization image with results and the prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Get the prediction results in <code>json</code> format</td>
</tr>
<tr>
<td rowspan = "1"><code>img</code></td>
<td rowspan = "1">Get the visualization image in <code>dict</code> format</td>
</tr>
</table>

For more information on the usage of PaddleX single-model inference APIs, please refer to [PaddleX Single-Model Python Script Usage Guide](../../instructions/model_python_API.en.md).

## IV. Secondary Development
The current module temporarily does not support fine-tuning training and only supports inference integration. Fine-tuning training for this module is planned to be supported in the future.
