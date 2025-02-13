---
comments: true
---

# Semantic Segmentation Module Development Tutorial

## I. Overview
Semantic segmentation is a technique in computer vision that classifies each pixel in an image, dividing the image into distinct semantic regions, with each region corresponding to a specific category. This technique generates detailed segmentation maps, clearly revealing objects and their boundaries in the image, providing powerful support for image analysis and understanding.

## II. Supported Model List

<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>mIoU (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>OCRNet_HRNet-W48</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W48_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W48_pretrained.pdparams">Trained Model</a></td>
<td>82.15</td>
<td>627.36 / 170.76</td>
<td>3531.61 / 3531.61</td>
<td>249.8 M</td>
</tr>
<tr>
<td>PP-LiteSeg-T</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-T_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-T_pretrained.pdparams">Trained Model</a></td>
<td>73.10</td>
<td>30.16 / 14.03</td>
<td>420.07 / 235.01</td>
<td>28.5 M</td>
</tr>
</tbody>
</table>
> ❗ The above list features the <b>2 core models</b> that the image classification module primarily supports. In total, this module supports <b>18 models</b>. The complete list of models is as follows:

<details><summary> 👉Model List Details</summary>
<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>mIoU (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Deeplabv3_Plus-R50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R50_pretrained.pdparams">Trained Model</a></td>
<td>80.36</td>
<td>503.51 / 122.30</td>
<td>3543.91 / 3543.91</td>
<td>94.9 M</td>
</tr>
<tr>
<td>Deeplabv3_Plus-R101</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R101_pretrained.pdparams">Trained Model</a></td>
<td>81.10</td>
<td>803.79 / 175.45</td>
<td>5136.21 / 5136.21</td>
<td>162.5 M</td>
</tr>
<tr>
<td>Deeplabv3-R50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3-R50_pretrained.pdparams">Trained Model</a></td>
<td>79.90</td>
<td>647.56 / 121.67</td>
<td>3803.09 / 3803.09</td>
<td>138.3 M</td>
</tr>
<tr>
<td>Deeplabv3-R101</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3-R101_pretrained.pdparams">Trained Model</a></td>
<td>80.85</td>
<td>950.43 / 178.50</td>
<td>5517.14 / 5517.14</td>
<td>205.9 M</td>
</tr>
<tr>
<td>OCRNet_HRNet-W18</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W18_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W18_pretrained.pdparams">Trained Model</a></td>
<td>80.67</td>
<td>286.12 / 80.76</td>
<td>1794.03 / 1794.03</td>
<td>43.1 M</td>
</tr>
<tr>
<td>OCRNet_HRNet-W48</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W48_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W48_pretrained.pdparams">Trained Model</a></td>
<td>82.15</td>
<td>627.36 / 170.76</td>
<td>3531.61 / 3531.61</td>
<td>249.8 M</td>
</tr>
<tr>
<td>PP-LiteSeg-T</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-T_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-T_pretrained.pdparams">Trained Model</a></td>
<td>73.10</td>
<td>30.16 / 14.03</td>
<td>420.07 / 235.01</td>
<td>28.5 M</td>
</tr>
<tr>
<td>PP-LiteSeg-B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-B_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-B_pretrained.pdparams">Trained Model</a></td>
<td>75.25</td>
<td>40.92 / 20.18</td>
<td>494.32 / 310.34</td>
<td>47.0 M</td>
</tr>
<tr>
<td>SegFormer-B0 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B0 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B0 (slice)_pretrained.pdparams">Trained Model</a></td>
<td>76.73</td>
<td>11.1946</td>
<td>268.929</td>
<td>13.2 M</td>
</tr>
<tr>
<td>SegFormer-B1 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B1 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B1 (slice)_pretrained.pdparams">Trained Model</a></td>
<td>78.35</td>
<td>17.9998</td>
<td>403.393</td>
<td>48.5 M</td>
</tr>
<tr>
<td>SegFormer-B2 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B2 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B2 (slice)_pretrained.pdparams">Trained Model</a></td>
<td>81.60</td>
<td>48.0371</td>
<td>1248.52</td>
<td>96.9 M</td>
</tr>
<tr>
<td>SegFormer-B3 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B3 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B3 (slice)_pretrained.pdparams">Trained Model</a></td>
<td>82.47</td>
<td>64.341</td>
<td>1666.35</td>
<td>167.3 M</td>
</tr>
<tr>
<td>SegFormer-B4 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B4 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B4 (slice)_pretrained.pdparams">Trained Model</a></td>
<td>82.38</td>
<td>82.4336</td>
<td>1995.42</td>
<td>226.7 M</td>
</tr>
<tr>
<td>SegFormer-B5 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B5 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B5 (slice)_pretrained.pdparams">Trained Model</a></td>
<td>82.58</td>
<td>97.3717</td>
<td>2420.19</td>
<td>229.7 M</td>
</tr>
</tbody>
</table>
<p><b>The accuracy metrics of the above models are measured on the <a href="https://www.cityscapes-dataset.com/">Cityscapes</a> dataset. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>mIoU (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>SeaFormer_base(slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_base(slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_base(slice)_pretrained.pdparams">Trained Model</a></td>
<td>40.92</td>
<td>24.4073</td>
<td>397.574</td>
<td>30.8 M</td>
</tr>
<tr>
<td>SeaFormer_large (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_large (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_large (slice)_pretrained.pdparams">Trained Model</a></td>
<td>43.66</td>
<td>27.8123</td>
<td>550.464</td>
<td>49.8 M</td>
</tr>
<tr>
<td>SeaFormer_small (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_small (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_small (slice)_pretrained.pdparams">Trained Model</a></td>
<td>38.73</td>
<td>19.2295</td>
<td>358.343</td>
<td>14.3 M</td>
</tr>
<tr>
<td>SeaFormer_tiny (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_tiny (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_tiny (slice)_pretrained.pdparams">Trained Model</a></td>
<td>34.58</td>
<td>13.9496</td>
<td>330.132</td>
<td>6.1M</td>
</tr>
<tr>
<td>MaskFormer_small</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskFormer_small_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskFormer_small_pretrained.pdparams">Trained Model</a></td>
<td>49.70</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>MaskFormer_tiny</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MaskFormer_tiny_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MaskFormer_tiny_pretrained.pdparams">Trained Model</a></td>
<td>46.69</td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>
<p><b>The accuracy metrics of the above models are measured on the <a href="https://groups.csail.mit.edu/vision/datasets/ADE20K/">ADE20k</a> dataset. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p></details>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)


Just a few lines of code can complete the inference of the Semantic Segmentation module, allowing you to easily switch between models under this module. You can also integrate the model inference of the the Semantic Segmentation module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_semantic_segmentation_002.png) to your local machine.

```python
from paddlex import create_model
model = create_model("PP-LiteSeg-T")
output = model.predict("general_semantic_segmentation_002.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

After running, the result obtained is:

```bash
{'res': "{'input_path': 'general_semantic_segmentation_002.png', 'page_index': None, 'pred': '...'}"}
```

The meanings of the runtime parameters are as follows:
- `input_path`: Indicates the path of the input image to be predicted.
- `page_index`: If the input is a PDF file, it represents the current page number of the PDF; otherwise, it is `None`.
- `pred`: The actual mask predicted by the semantic segmentation model. Since the data is too large to be printed directly, it is replaced with `...` here. The prediction result can be saved as an image through `res.save_to_img()` and as a JSON file through `res.save_to_json()`.

The visualization image is as follows:

<img alt="Visualization Image" src="https://raw.githubusercontent.com/BluebirdStory/PaddleX_doc_images/main/images/modules/semantic_segmentation/general_semantic_segmentation_002_res.png"/>

Note: The image link may not be accessible due to network issues or problems with the link itself. If you need to access the image, please check the validity of the link and try again.

Related methods, parameters, and explanations are as follows:

* The `create_model` method instantiates a general semantic segmentation model (here using `PP-LiteSeg-T` as an example), with specific explanations as follows:
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
<td><code>target_size</code></td>
<td>The resolution used during model prediction</td>
<td><code>int/tuple</code></td>
<td><code>None/-1/int/tuple</code></td>
<td><code>None</code></td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the built-in model parameters of PaddleX are used by default. If `model_dir` is specified, the user-defined model is used.

* The `target_size` is specified during initialization to set the resolution for model inference. The default value is `None`. `-1` indicates that the original image size is used for inference, and `None` indicates that the settings from the lower priority are used. The priority order for parameter settings is: `predict parameter > create_model initialization > yaml configuration file`.

* The `predict()` method of the general semantic segmentation model is called for inference and prediction. The parameters of the `predict()` method are `input`, `batch_size`, and `target_size`, with specific explanations as follows:

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
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supports multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python Variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
<li><b>File Path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
<li><b>URL Link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_semantic_segmentation_001.png">Example</a></li>
<li><b>Local Directory</b>, the directory should contain data files to be predicted, such as the local path: <code>/root/data/</code></li>
<li><b>List</b>, elements of the list should be data of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
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
<td><code>target_size</code></td>
<td>Image size during inference (W, H)</td>
<td><code>int</code>/<code>tuple</code></td>
<td>
<ul>
<li><b>-1</b>, indicating inference using the original image size</li>
<li><b>None</b>, indicating the settings from the lower priority are used. The priority order for parameter settings is: <code>predict parameter &gt; create_model initialization &gt; yaml configuration file</code></li>
<li><b>int</b>, such as 512, indicating inference using a resolution of <code>(512, 512)</code></li>
<li><b>tuple</b>, such as (512, 1024), indicating inference using a resolution of <code>(512, 1024)</code></li>
</ul>
</td>
<td>None</td>
</tr>
</table>

* The prediction results are processed as `dict` type for each sample, and support operations such as printing, saving as an image, and saving as a `json` file:

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
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content with <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a file in <code>json</code> format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will match the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as a file in image format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will match the input file name</td>
<td>None</td>
</tr>
</table>

* Additionally, it also supports obtaining the visualization image with results and the prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="1"><code>img</code></td>
<td rowspan="1">Get the visualization image in <code>dict</code> format</td>
</tr>
</table>

For more information on using PaddleX's single-model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development

If you seek higher accuracy, you can leverage PaddleX's custom development capabilities to develop better Semantic Segmentation models. Before developing a Semantic Segmentation model with PaddleX, ensure you have installed PaddleClas plugin for PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 4.1 Dataset Preparation

Before model training, you need to prepare a dataset for the task. PaddleX provides data validation functionality for each module. <b>Only data that passes validation can be used for model training.</b> Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to [PaddleX Semantic Segmentation Task Module Data Preparation Tutorial](../../../data_annotations/cv_modules/semantic_segmentation.en.md).

#### 4.1.1 Demo Data Download

You can download the demo dataset to a specified folder using the following commands:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/seg_optic_examples.tar -P ./dataset
tar -xf ./dataset/seg_optic_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation

Data validation can be completed with a single command:

```bash
python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/seg_optic_examples
```

After executing the above command, PaddleX will verify the dataset and collect basic information about it. Once the command runs successfully, a message saying `Check dataset passed !` will be printed in the log. The verification results will be saved in `./output/check_dataset_result.json`, and related outputs will be stored in the `./output/check_dataset` directory, including visual examples of sample images and a histogram of sample distribution.

<details><summary>👉 <b>Verification Result Details (click to expand)</b></summary>
<p>The specific content of the verification result file is:</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_sample_paths": [
      "check_dataset/demo_img/P0005.jpg",
      "check_dataset/demo_img/P0050.jpg"
    ],
    "train_samples": 267,
    "val_sample_paths": [
      "check_dataset/demo_img/N0139.jpg",
      "check_dataset/demo_img/P0137.jpg"
    ],
    "val_samples": 76,
    "num_classes": 2
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "seg_optic_examples",
  "show_type": "image",
  "dataset_type": "SegDataset"
}
</code></pre>
<p>The verification results above indicate that <code>check_pass</code> being <code>True</code> means the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 2;</li>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 267;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 76;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to the visualization images of training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the visualization images of validation samples in this dataset;</li>
</ul>
<p>The dataset verification also analyzes the distribution of sample numbers across all classes and plots a histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/semanticseg/01.png"/></p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional) (Click to Expand)
<details><summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>
<p>After completing dataset verification, you can convert the dataset format or re-split the training/validation ratio by modifying the configuration file or appending hyperparameters.</p>
<p><b>(1) Dataset Format Conversion</b></p>
<p>Semantic segmentation supports converting <code>LabelMe</code> format datasets to the required format.</p>
<p>Parameters related to dataset verification can be set by modifying the <code>CheckDataset</code> fields in the configuration file. Example explanations for some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: Whether to enable dataset format conversion, supporting <code>LabelMe</code> format conversion, default is <code>False</code>;</li>
<li><code>src_dataset_type</code>: If dataset format conversion is enabled, the source dataset format needs to be set, default is <code>null</code>, and the supported source dataset format is <code>LabelMe</code>;</li>
</ul>
<p>For example, if you want to convert a <code>LabelMe</code> format dataset, you can download a sample <code>LabelMe</code> format dataset as follows:</p>
<pre><code class="language-bash">wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/seg_dataset_to_convert.tar -P ./dataset
tar -xf ./dataset/seg_dataset_to_convert.tar -C ./dataset/
</code></pre>
<p>After downloading, modify the <code>paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml</code> configuration as follows:</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: LabelMe
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/seg_dataset_to_convert
</code></pre>
<p>Of course, the above parameters also support being set by appending command-line arguments. For a <code>LabelMe</code> format dataset, the command is:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/seg_dataset_to_convert \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
</code></pre>
<p><b>(2) Dataset Splitting</b></p>
<p>Parameters for dataset splitting can be set by modifying the <code>CheckDataset</code> fields in the configuration file. Example explanations for some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to enable re-splitting the dataset, set to <code>True</code> to perform dataset splitting, default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, set the percentage of the training set, which should be an integer between 0 and 100, ensuring the sum with <code>val_percent</code> is 100;</li>
</ul>
<p>For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, modify the configuration file as follows:</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/seg_optic_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support setting through appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/seg_optic_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training

Model training can be completed with just one command. Here, we use the semantic segmentation model (PP-LiteSeg-T) as an example:

```bash
python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/seg_optic_examples
```

You need to follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `PP-LiteSeg-T.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md)).
* Set the mode to model training: `-o Global.mode=train`
* Specify the training dataset path: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to train using the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters Documentation](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<ul>
<li>During model training, PaddleX automatically saves model weight files, with the default path being <code>output</code>. To specify a different save path, use the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX abstracts the concepts of dynamic graph weights and static graph weights from you. During model training, both dynamic and static graph weights are produced, and static graph weights are used by default for model inference.</li>
<li>
<p>After model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including:</p>
</li>
<li>
<p><code>train_result.json</code>: Training result record file, including whether the training task completed successfully, produced weight metrics, and related file paths.</p>
</li>
<li><code>train.log</code>: Training log file, recording model metric changes, loss changes, etc.</li>
<li><code>config.yaml</code>: Training configuration file, recording the hyperparameters used for this training session.</li>
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, and static graph network structure.</li>
</ul></details>

### 4.3 Model Evaluation
After model training, you can evaluate the specified model weights on the validation set to verify model accuracy. Using PaddleX for model evaluation requires just one command:

```bash
python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/seg_optic_examples
```

Similar to model training, follow these steps:

* Specify the `.yaml` configuration file path for the model (here it's `PP-LiteSeg-T.yaml`).
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the validation dataset path: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For more details, refer to the [PaddleX Common Configuration Parameters Documentation](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<p>When evaluating the model, you need to specify the model weight file path. Each configuration file has a default weight save path. If you need to change it, simply append the command line parameter, e.g., <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After model evaluation, the following outputs are typically produced:</p>
<ul>
<li><code>evaluate_result.json</code>: Records the evaluation results, specifically whether the evaluation task completed successfully and the model's evaluation metrics, including mIoU.</li>
</ul></details>

### 4.4 Model Inference and Integration
After model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
To perform inference predictions via the command line, use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_semantic_segmentation_002.png) to your local machine.


```bash
python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model" \
    -o Predict.input="general_semantic_segmentation_002.png"
```

Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it's `PP-LCNet_x1_0_doc_ori.yaml`)

* Set the mode to model inference prediction: `-o Global.mode=predict`

* Specify the model weights path: -o Predict.model_dir="./output/best_accuracy/inference"

Specify the input data path: `-o Predict.inputh="..."` Other related parameters can be set by modifying the fields under Global and Predict in the `.yaml` configuration file. For details, refer to PaddleX Common Model Configuration File Parameter Description.

Alternatively, you can use the PaddleX wheel package for inference, easily integrating the model into your own projects.

#### 4.4.2 Model Integration

The model can be directly integrated into the PaddleX pipeline or into your own projects.

1. <b>Pipeline Integration</b>

The document semantic segmentation module can be integrated into PaddleX pipelines such as the [Semantic Segmentation Pipeline (Seg)](../../../pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.en.md). Simply replace the model path to update the The document semantic segmentation module's model.

2. <b>Module Integration</b>

The weights you produce can be directly integrated into the semantic segmentation module. You can refer to the Python sample code in [Quick Integration](#iii-quick-integration) and just replace the model with the path to the model you trained.

You can also use the PaddleX high-performance inference plugin to optimize the inference process of your model and further improve efficiency. For detailed procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).