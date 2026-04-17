---
comments: true
---

# Layout Analysis Module Tutorial

## I. Overview
The layout analysis task builds upon layout area detection by further introducing **instance segmentation** and **reading order prediction** capabilities. By analyzing input document images, it not only identifies various layout elements (such as text, charts, images, formulas, paragraph titles, abstracts, references, etc.) and outputs their bounding boxes, but also simultaneously outputs the **precise contour mask** and **reading order index** for each region, providing more complete structural information for document understanding and information extraction workflows.

The layout analysis module currently supports the PP-DocLayoutV3 model, which is based on the DETR architecture with PPHGNetV2-L as the backbone network. It adds a **reading order prediction** branch on top of the instance segmentation task, enabling end-to-end learning of reading order relationships among document elements.

![layout analysis](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/layout_analysis/layout_analysis.png)

## II. Supported Model List

> The inference time only includes the model inference time and does not include the time for pre- or post-processing.

* <b>The layout analysis model includes 25 common categories: abstract, algorithm, aside text, chart, content, display formula, document title, figure title, footer, footer image, footnote, formula number, header, header image, image, inline formula, number, paragraph title, reference, reference content, seal, table, text, vertical text, and vision footnote</b>

<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>GPU Inference Time (ms)<br/>A100 GPU</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayoutV3</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayoutV3_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayoutV3_pretrained.pdparams">Training Model</a></td>
<td>23.77</td>
<td>126</td>
<td>A layout analysis model trained on a self-built dataset containing Chinese and English papers, multi-column magazines, newspapers, PPT, contracts, books, exams, and research reports using DETR. It supports instance segmentation and reading order prediction for 25 layout element categories.</td>
</tr>
</tbody>
</table>

<b>Note: The evaluation set for the above accuracy metrics is a self-built layout analysis dataset containing images from various Chinese and English document scenarios.</b>

## III. Quick Integration  <a id="quick"> </a>
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to [PaddleX Local Installation Tutorial](../../../installation/installation.en.md)

After installing the wheel package, a few lines of code can complete the inference of the layout analysis module. You can switch models under this module freely, and you can also integrate the model inference of the layout analysis module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg) to your local machine.

```python
from paddlex import create_model

model_name = "PP-DocLayoutV3"
model = create_model(model_name=model_name)
output = model.predict("layout.jpg", batch_size=1)

for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

<b>Note: </b>The official models would be downloaded from HuggingFace by default. PaddleX also supports specifying the preferred source by setting the environment variable `PADDLE_PDX_MODEL_SOURCE`. The supported values are `huggingface`, `aistudio`, `bos`, and `modelscope`. For example, to prioritize using `bos`, set: `PADDLE_PDX_MODEL_SOURCE="bos"`.

<details><summary>👉 <b>After running, the result is: (Click to expand)</b></summary>

```bash
{'res': {'input_path': 'layout.jpg', 'page_index': None, 'boxes': [{'cls_id': 22, 'label': 'text', 'score': 0.98, 'coordinate': [34.1, 349.8, 358.5, 611.0], 'polygon_points': [[34.1, 349.8], [358.5, 349.8], [358.5, 611.0], [34.1, 611.0]], 'order': 3}, ...]}}
```

The meanings of the parameters are as follows:
- `input_path`: The path to the input image for prediction.
- `page_index`: If the input is a PDF file, it indicates which page of the PDF it is; otherwise, it is `None`.
- `boxes`: Information about the predicted bounding boxes, a list of dictionaries. Each dictionary represents a detected object and contains the following information:
  - `cls_id`: Class ID, an integer.
  - `label`: Class label, a string.
  - `score`: Confidence score of the bounding box, a float.
  - `coordinate`: Coordinates of the bounding box, a list of floats in the format <code>[xmin, ymin, xmax, ymax]</code>.
  - `polygon_points`: List of instance segmentation contour points, in the format <code>[[x1, y1], [x2, y2], ...]</code>.
  - `order`: Reading order index, an integer indicating the reading order of the region in the document (starting from 0).
</details>

After running, the visualization result saved by `save_to_img()` is shown below, with each region annotated with its category, confidence score, instance segmentation mask, and reading order index:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/layout_analysis/layout_analysis_demo_res.jpg" alt="版面分析可视化结果" width="30%" />

Relevant methods, parameters, and explanations are as follows:

* `create_model` instantiates a layout analysis model (here, `PP-DocLayoutV3` is used as an example). The detailed explanation is as follows:
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
<td><code>model_name</code></td>
<td>Name of the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Path to store the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for model inference</td>
<td><code>str</code></td>
<td>It supports specifying specific GPU card numbers, such as "gpu:0", other hardware card numbers, such as "npu:0", or CPU, such as "cpu".</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>engine</code></td>
<td>Inference engine</td>
<td><code>str | None</code></td>
<td>Optional <code>paddle</code>, <code>paddle_static</code>, <code>paddle_dynamic</code>, <code>flexible</code>, <code>transformers</code>.</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>engine_config</code></td>
<td>Inference engine configuration</td>
<td><code>dict | None</code></td>
<td>Different engines support different fields, please refer to <a href="../../instructions/model_python_API.en.md#4-inference-engine-and-configuration">Inference Engine and Configuration</a>.</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>pp_option</code></td>
<td>Used for changing runtime mode and other configuration items</td>
<td><code>PaddlePredictorOption</code></td>
<td>For detailed inference configuration, please refer to <a href="../../instructions/model_python_API.en.md#5-compatibility-configuration-paddlepredictoroption">Compatible Configuration (PaddlePredictorOption)</a>.</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>img_size</code></td>
<td>Size of the input image; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>int/list/None</code></td>
<td>
<ul>
<li><b>int</b>, e.g., 800, means resizing the input image to 800x800</li>
<li><b>List</b>, e.g., [800, 800], means resizing the input image to a width of 800 and a height of 800</li>
<li><b>None</b>, not specified, will use the default PaddleX official model configuration</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering low-confidence prediction results; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>float/dict/None</code></td>
<td>
<ul>
<li><b>float</b>, e.g., 0.5, means filtering out all bounding boxes with a confidence score less than 0.5</li>
<li><b>Dictionary</b>, with keys as <b>int</b> representing <code>cls_id</code> and values as <b>float</b> thresholds, e.g., <code>{0: 0.45, 2: 0.48}</code></li>
<li><b>None</b>, not specified, will use the default PaddleX official model configuration</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS post-processing to filter overlapping boxes; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>bool/None</code></td>
<td>
<ul>
<li><b>bool</b>, True/False, indicates whether to use NMS for post-processing to filter overlapping boxes</li>
<li><b>None</b>, not specified, will use the default PaddleX official model configuration</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Scaling factor for the side length of the detection box; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>float/list/dict/None</code></td>
<td>
<ul>
<li><b>float</b>, a positive float number, e.g., 1.1, means expanding the width and height of the detection box by 1.1 times while keeping the center unchanged</li>
<li><b>List</b>, e.g., [1.2, 1.5], means expanding the width by 1.2 times and the height by 1.5 times while keeping the center unchanged</li>
<li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code>, values as <b>tuple</b>, e.g., <code>{0: (1.1, 2.0)}</code></li>
<li><b>None</b>, not specified, will use the default PaddleX official model configuration</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Merging mode for the detection boxes output by the model; if not specified, the default PaddleX official model configuration will be used</td>
<td><code>string/dict/None</code></td>
<td>
<ul>
<li><b>large</b>, when set to large, only the largest external box will be retained for overlapping detection boxes, and the internal overlapping boxes will be deleted</li>
<li><b>small</b>, when set to small, only the smallest internal box will be retained for overlapping detection boxes, and the external overlapping boxes will be deleted</li>
<li><b>union</b>, no filtering of boxes will be performed, and both internal and external boxes will be retained</li>
<li><b>dict</b>, keys as <b>int</b> representing <code>cls_id</code> and values as merging modes, e.g., <code>{0: "large", 2: "small"}</code></li>
<li><b>None</b>, not specified, will use the default PaddleX official model configuration</li>
</ul>
</td>
<td>None</td>
</tr>
</table>

* Note that `model_name` must be specified. After specifying `model_name`, the default PaddleX built-in model parameters will be used. If `model_dir` is specified, the user-defined model will be used.

* The `predict()` method of the layout analysis model is called for inference prediction. The parameters of the `predict()` method are explained as follows:

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
<td>Data for prediction, supporting multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python Variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
<li><b>File Path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
<li><b>URL link</b>, such as the network URL of an image file</li>
<li><b>Local Directory</b>, the directory should contain the data files to be predicted, such as the local path: <code>/root/data/</code></li>
<li><b>List</b>, the elements of the list should be of the above-mentioned data types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code></li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>Any integer greater than 0</td>
<td>1</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering low-confidence prediction results</td>
<td><code>float/dict/None</code></td>
<td>
<ul>
<li><b>float</b>, e.g., 0.5, means filtering out all bounding boxes with a confidence score less than 0.5</li>
<li><b>Dictionary</b>, with keys as <b>int</b> representing <code>cls_id</code> and values as <b>float</b> thresholds</li>
<li><b>None</b>, not specified, will use the <code>threshold</code> parameter specified in <code>create_model</code>. If not specified in <code>create_model</code>, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<td>None</td>
</tr>
</table>

* The prediction results for each sample can be printed, saved as images, or saved as `json` files:

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
<td rowspan="3">Print results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level for prettifying the output <code>JSON</code> data, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether non-<code>ASCII</code> characters are escaped to Unicode, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save results as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When a directory is specified, the saved file is named consistently with the input file type</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level for prettifying the output <code>JSON</code> data, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether non-<code>ASCII</code> characters are escaped to Unicode, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save results as an image file (the visualized image includes instance segmentation masks and reading order indices)</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When a directory is specified, the saved file is named consistently with the input file type</td>
<td>None</td>
</tr>
</table>

* Additionally, it also supports obtaining the visualized image with results and the prediction results via attributes, as follows:

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
<td rowspan="1">Get the visualized image in <code>dict</code> format, with each region's category, confidence score, instance segmentation mask, and reading order index annotated</td>
</tr>
</table>

For more information on using PaddleX's single-model inference API, refer to [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better layout analysis models. Before developing a layout analysis model with PaddleX, ensure you have installed PaddleX's Detection-related model training capabilities. The installation process can be found in [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides a data validation function for each module, and <b>only data that passes the validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development based on the official demos. If you wish to use private datasets for subsequent model training, refer to the [PaddleX Object Detection Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/object_detection.en.md).

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/doclayoutv3_examples.tar -P ./dataset
tar -xf ./dataset/doclayoutv3_examples.tar -C ./dataset/
```

#### 4.1.2 Dataset Format Description
The layout analysis module uses the **COCOInstSegDataset** format, supplemented with reading order annotations. The dataset directory structure is as follows:

```
doclayoutv3_examples/
├── images/               # Original image directory
│   ├── train_0001.jpg
│   ├── val_0001.jpg
│   └── ...
├── images_mask/          # Image directory for training (same content as images)
│   └── ...
└── annotations/
    ├── instance_train.json   # Training set annotations (COCO instance segmentation format + read_order field)
    └── instance_val.json     # Validation set annotations (COCO instance segmentation format + read_order field)
```

The annotation files follow the COCO instance segmentation format, with an added `read_order` field in each annotation to record the reading order of the region in the document (a non-negative integer starting from 0; the `read_order` values of all annotations within the same image should form a consecutive sequence). An example annotation is as follows:

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 22,
      "bbox": [34.1, 349.8, 324.4, 261.2],
      "segmentation": [[34.1, 349.8, 358.5, 349.8, 358.5, 611.0, 34.1, 611.0]],
      "area": 84740.0,
      "iscrowd": 0,
      "read_order": 0
    }
  ]
}
```

The 25 categories supported by the model and their descriptions are as follows:

| Category Name (English) | Description |
|---|---|
| `abstract` | Abstract |
| `algorithm` | Algorithm |
| `aside_text` | Sidebar text |
| `chart` | Chart |
| `content` | Table of contents |
| `display_formula` | Display formula |
| `doc_title` | Document title |
| `figure_title` | Figure title |
| `footer` | Footer |
| `footer_image` | Footer image |
| `footnote` | Footnote |
| `formula_number` | Formula number |
| `header` | Header |
| `header_image` | Header image |
| `image` | Image |
| `inline_formula` | Inline formula |
| `number` | Page number |
| `paragraph_title` | Paragraph title |
| `reference` | Reference |
| `reference_content` | Reference content |
| `seal` | Seal |
| `table` | Table |
| `text` | Text |
| `vertical_text` | Vertical text |
| `vision_footnote` | Figure caption |

#### 4.1.3 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/doclayoutv3_examples
```

After executing the above command, PaddleX will validate the dataset and collect its basic information. Upon successful execution, the log will print the message `Check dataset passed !`. The validation result file will be saved in `./output/check_dataset_result.json`, and related outputs will be saved in the `./output/check_dataset` directory of the current directory. The output directory includes visualized example images (with instance segmentation masks and reading order indices annotated) and histograms of sample distributions.

<details><summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>
<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 11,
    "train_samples": 6351,
    "train_sample_paths": [
      "check_dataset\/demo_img\/train_4141.jpg",
      "check_dataset\/demo_img\/train_3699.jpg",
      "check_dataset\/demo_img\/train_3764.jpg",
      "check_dataset\/demo_img\/train_2279.jpg",
      "check_dataset\/demo_img\/train_4647.jpg",
      "check_dataset\/demo_img\/train_4442.jpg",
      "check_dataset\/demo_img\/train_2006.jpg",
      "check_dataset\/demo_img\/train_1463.jpg",
      "check_dataset\/demo_img\/train_3275.jpg",
      "check_dataset\/demo_img\/train_4509.jpg"
    ],
    "val_samples": 945,
    "val_sample_paths": [
      "check_dataset\/demo_img\/val_0105.jpg",
      "check_dataset\/demo_img\/val_0031.jpg",
      "check_dataset\/demo_img\/val_0755.jpg",
      "check_dataset\/demo_img\/val_0876.jpg",
      "check_dataset\/demo_img\/val_0374.jpg",
      "check_dataset\/demo_img\/val_0566.jpg",
      "check_dataset\/demo_img\/val_0748.jpg",
      "check_dataset\/demo_img\/val_0167.jpg",
      "check_dataset\/demo_img\/val_0345.jpg",
      "check_dataset\/demo_img\/val_0471.jpg"
    ],
    "read_order_validation": {
      "instance_train": {
        "total_images": 500,
        "valid_images": 500,
        "invalid_images": [],
        "pass_rate": 1.0
      },
      "instance_val": {
        "total_images": 100,
        "valid_images": 100,
        "invalid_images": [],
        "pass_rate": 1.0
      }
    }
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "doclayoutv3_examples",
  "show_type": "image",
  "dataset_type": "COCOInstSegDataset"
}
</code></pre>
<p>The verification results mentioned above indicate that <code>check_pass</code> being <code>True</code> means the dataset format meets the requirements. Details of other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 11;</li>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset;</li>
<li><code>attributes.train_sample_paths</code>: The list of relative paths to the visualization images of training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: The list of relative paths to the visualization images of validation samples in this dataset;</li>
<li><code>attributes.read_order_validation</code>: Validation statistics for the <code>read_order</code> field, including the total number of images, number of valid images, and pass rate for both training and validation sets.</li>
</ul>
<p>The dataset verification also analyzes the distribution of sample numbers across all classes and generates a histogram (histogram.png).</p>

<b>Note:</b> Layout analysis data validation additionally validates the <code>read_order</code> field in each image annotation:
<ul>
<li>Completeness: Each annotation must contain a <code>read_order</code> field;</li>
<li>Type validity: <code>read_order</code> must be a non-negative integer;</li>
<li>Consecutiveness: The <code>read_order</code> values of all annotations within the same image should form a consecutive integer sequence starting from 0 (a warning will be issued if non-consecutive).</li>
</ul>
</details>

#### 4.1.4 Dataset Format Conversion/Dataset Splitting (Optional)

After completing dataset verification, you can re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details on Dataset Splitting (Click to Expand)</b></summary>
<p><b>(1) Dataset Format Conversion</b></p>
<p>Layout analysis does not support data format conversion. Please use the COCO instance segmentation format directly (with the <code>read_order</code> field).</p>
<p><b>(2) Dataset Splitting</b></p>
<p>Parameters for dataset splitting can be set by modifying the <code>CheckDataset</code> section in the configuration file:</p>
<pre><code class="language-bash">CheckDataset:
  split:
    enable: True
    train_percent: 90
    val_percent: 10
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/doclayoutv3_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters can also be set by appending command-line arguments:</p>
<pre><code>python main.py -c paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/doclayoutv3_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training

A single command is sufficient to complete model training, taking the training of `PP-DocLayoutV3` as an example:

```bash
python main.py -c paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/doclayoutv3_examples
```
The steps required are:

* Specify the path to the `.yaml` configuration file of the model (here it is `PP-DocLayoutV3.yaml`)
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`
* Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters for Model Tasks](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<ul>
<li>During model training, PaddleX automatically saves model weight files, defaulting to <code>output</code>. To specify a save path, use the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.</li>
<li>
<p>After completing the model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including:</p>
</li>
<li>
<p><code>train_result.json</code>: Training result record file, recording whether the training task was completed normally, as well as the output weight metrics, related file paths, etc.;</p>
</li>
<li><code>train.log</code>: Training log file, recording changes in model metrics and loss during training;</li>
<li><code>config.yaml</code>: Training configuration file, recording the hyperparameter configuration for this training session;</li>
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.json</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;</li>
<li>Note: The layout analysis model uses an <code>order_loss</code> branch during training (with a weight coefficient of 50). This loss term supervises the reading order prediction, and changes in <code>order_loss</code> can be observed in the training log.</li>
</ul></details>

### <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation, you can complete the evaluation with a single command:

```bash
python main.py -c paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/doclayoutv3_examples
```
Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model (here it's `PP-DocLayoutV3.yaml`)
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to [PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model/model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully, and the model's evaluation metrics, including AP.</p></details>

### <b>4.4 Model Inference</b>
After completing model training and evaluation, you can use the trained model weights for inference predictions. In PaddleX, model inference predictions can be achieved through two methods: command line and wheel package.

#### 4.4.1 Model Inference
* To perform inference predictions through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg) to your local machine.
```bash
python main.py -c paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="layout.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it is `PP-DocLayoutV3.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

* Alternatively, you can use the PaddleX wheel package for inference, easily integrating the model into your own project. To integrate, simply add the `model_dir="/output/best_model/inference"` parameter to the `create_model` function in the quick integration method from Step 3.

#### 4.4.2 Weight Conversion

This module supports converting Paddle dynamic graph weights (`.pdparams`) to `safetensors` format for direct use with PaddleX's `paddle_dynamic` and `transformers` engines. Models supporting weight conversion in this module: `PP-DocLayoutV3`.

* To perform weight conversion via command line, taking `PP-DocLayoutV3` as an example:

```bash
python main.py -c paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml \
    -o Global.mode=pdparams2safetensors \
    -o Pdparams2safetensors.input_path=./path/to/model.pdparams \
    -o Pdparams2safetensors.output_dir=./output/safetensors/
```

* Parameter description:
    * `Global.mode`: Set the mode to weight conversion: `pdparams2safetensors`
    * `Pdparams2safetensors.input_path`: Path to the input `.pdparams` weight file (or a directory containing one)
    * `Pdparams2safetensors.output_dir`: Output directory for the converted `safetensors` model

After conversion, the output directory will contain `model.safetensors`, `config.json`, `preprocess_config.json`, `inference.yml`, and other files ready for inference.

For other related parameters, please refer to [PaddleX Common Model Configuration Parameters](../../instructions/config_parameters_common.en.md).


#### 4.4.3 Model Integration
The model can be directly integrated into PaddleX pipelines or into your own projects.

1. <b>Pipeline Integration</b>

The layout analysis module can be integrated into PaddleX pipelines such as the [Document Parsing Pipeline (PaddleOCR-VL and PaddleOCR-VL-1.5)](../../../pipeline_usage/tutorials/ocr_pipelines/PaddleOCR-VL.en.md). Simply replace the model path to update the layout analysis module.

2. <b>Module Integration</b>

The weights you produce can be directly integrated into the layout analysis module. You can refer to the Python example code in the [Quick Integration](#quick) section, simply replacing the model with the path to your trained model.
