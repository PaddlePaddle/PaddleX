---
comments: true
---

# Tutorial on Using the Open-Vocabulary Segmentation Module

## I. Overview
Open-vocabulary segmentation is an image segmentation task that aims to segment objects in an image based on additional information such as text descriptions, bounding boxes, keypoints, etc., rather than just the image itself. It allows the model to handle a wide range of object categories without a predefined list. This technology combines visual and multimodal techniques, significantly enhancing the flexibility and accuracy of image processing. Open-vocabulary segmentation has important applications in the field of computer vision, especially in object segmentation tasks in complex scenes.

## II. Supported Model List

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>SAM-H_box</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SAM-H_box_infer.tar">Inference Model</a></td>
<td>144.9</td>
<td>33920.7</td>
<td>2433.7</td>
<td rowspan="2">SAM (Segment Anything Model) is an advanced image segmentation model that can segment any object in an image based on simple user-provided prompts (such as points, boxes, or text). Trained on the SA-1B dataset with ten million images and 1.1 billion mask annotations, it performs well in most scenarios.</td>
</tr>
<tr>
<td>SAM-H_point</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SAM-H_point_infer.tar">Inference Model</a></td>
<td>144.9</td>
<td>33920.7</td>
<td>2433.7</td>
</tr>
</table>

<b>Note: All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision, while CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For details, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

After installing the whl package, you can complete the inference of the open-vocabulary segmentation module with just a few lines of code. You can switch between models under this module at will, and you can also integrate the model inference of the open-vocabulary segmentation module into your project. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_segmentation.jpg) to your local machine.

```python
from paddlex import create_model
model = create_model('SAM-H_box')
results = model.predict(
    "open_vocabulary_segmentation.jpg",
    prompts = {
        "box_prompt": [
            [112.9239273071289,118.38755798339844,513.7587890625,382.0570068359375],
            [4.597158432006836,263.5540771484375,92.20092010498047,336.5640869140625],
            [592.3548583984375,260.8838806152344,607.1813354492188,294.2261962890625]
        ],
    }
)
for res in results:
    res.print()
    res.save_to_img(f"./output/")
    res.save_to_json(f"./output/res.json")
```

After running, the result obtained is:

```bash
{'res': "{'input_path': '000000004505.jpg', 'prompts': {'box_prompt': [[112.9239273071289, 118.38755798339844, 513.7587890625, 382.0570068359375], [4.597158432006836, 263.5540771484375, 92.20092010498047, 336.5640869140625], [592.3548583984375, 260.8838806152344, 607.1813354492188, 294.2261962890625]]}, 'masks': '...', 'mask_infos': [{'label': 'box_prompt', 'prompt': [112.9239273071289, 118.38755798339844, 513.7587890625, 382.0570068359375]}, {'label': 'box_prompt', 'prompt': [4.597158432006836, 263.5540771484375, 92.20092010498047, 336.5640869140625]}, {'label': 'box_prompt', 'prompt': [592.3548583984375, 260.8838806152344, 607.1813354492188, 294.2261962890625]}]}"}
```

The meanings of the parameters in the running results are as follows:
- `input_path`: The path of the input image to be predicted.
- `prompts`: The original prompt information used for prediction.
- `masks`: The actual predicted masks. Since the data is too large to be conveniently printed directly, it is replaced with `...` here. You can save the prediction results as an image using `res.save_to_img()` or as a JSON file using `res.save_to_json()`.
- `mask_infos`: The prompt information corresponding to each predicted mask.
  - `label`: The prompt type corresponding to the predicted mask.
  - `prompt`: The original prompt input corresponding to the predicted mask.

The visualization image is as follows:

<img src="https://raw.githubusercontent.com/BluebirdStory/PaddleX_doc_images/main/images/modules/open_vocabulary_segmentation/open_vocabulary_segmentation_res.jpg" />

Note: Due to network issues, the parsing of the above URL may not have been successful. If you need the content of this webpage, please check the validity of the URL and try again.

Related methods and parameter explanations are as follows:

* `create_model` instantiates an open-vocabulary segmentation model (using `SAM-H_box` as an example). The specific explanations are as follows:
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
</table>

* The `model_name` must be specified. After specifying `model_name`, the model parameters built into PaddleX will be used by default. If `model_dir` is specified, the user-defined model will be used.

* The `predict()` method of the open-vocabulary segmentation model is called for inference prediction. The parameters of the `predict()` method are `input`, `batch_size`, and `prompts`, with specific explanations as follows:

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
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
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
<td><code>prompts</code></td>
<td>Prompts used by the model</td>
<td><code>dict</code></td>
<td>
<ul>
  <li><b>dict</b>, such as <code>{"box_prompt": [[float, float, float, float], ...]}</code>, representing multiple bboxes used as prompts during inference</li>
</ul>
</td>
<td>None</td>
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
