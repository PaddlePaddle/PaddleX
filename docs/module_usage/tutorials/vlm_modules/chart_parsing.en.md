---
comments: true
---

# Chart Parsing Model Module Usage Tutorial

## I. Overview
Multimodal chart parsing is a cutting-edge technology in the OCR field, focusing on automatically converting various types of visual charts (such as bar charts, line charts, pie charts, etc.) into underlying data tables and formatting the output. Traditional methods rely on complex orchestration of models like chart key point detection, which involves many prior assumptions and lacks robustness. The models in this module utilize the latest VLM technology, driven by data, learning robust features from massive real-world data. Its application scenarios cover financial analysis, academic research, business reports, etc. — for instance, quickly extracting growth trend data from financial statements, experimental comparison values from scientific papers, or user distribution statistics from market research, assisting users in shifting from "viewing charts" to "using data."

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/chart_parsing/chart_parsing_01.png"/>

## II. Supported Model List

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Model parameter size（B）</th>
<th>Model Storage Size (GB)</th>
<th>Model Score </th>
<th>Description</th>
</tr>
<tr>
<td>PP-Chart2Table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-Chart2Table_infer.tar">Inference Model</a></td>
<td>0.58</td>
<td>1.4</td>
<th>75.98</th>
<td>PP-Chart2Table is a self-developed multimodal model by the PaddlePaddle team, focusing on chart parsing, demonstrating outstanding performance in both Chinese and English chart parsing tasks. The team adopted a carefully designed data generation strategy, constructing a high-quality multimodal dataset of nearly 700,000 entries covering common chart types like pie charts, bar charts, stacked area charts, and various application scenarios. They also designed a two-stage training method, utilizing large model distillation to fully leverage massive unlabeled OOD data. In internal business tests in both Chinese and English scenarios, PP-Chart2Table not only achieved the SOTA level among models of the same parameter scale but also reached accuracy comparable to 7B parameter scale VLM models in critical scenarios.</td>
</tr>
</table>

<b>Note: The above model scores are the results of internal evaluation set model testing, with a total of 1801 data points, including various chart types such as bar charts, line charts, and pie charts for testing samples under various scenarios such as financial reports, laws and regulations, contracts, etc. There are currently no plans to make them public.</b>



## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For details, please refer to [PaddleX Local Installation Tutorial](../../../installation/installation.md)

After completing the installation of the whl package, inference of the document-like visual language model module can be completed with just a few lines of code. You can freely switch models under this module, and you can also integrate model inference from the open document-like visual language model module into your project. Before running the following code, please download the [sample image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/chart_parsing_02.png) locally.

```python
from paddlex import create_model
model = create_model('PP-Chart2Table')
results = model.predict(
    input={"image": "chart_parsing_02.png"},
    batch_size=1
)
for res in results:
    res.print()
    res.save_to_json(f"./output/res.json")
```

After running, the result is:

```bash
{'res': {'image': 'chart_parsing_02.png', 'result': '年份 | 单家五星级旅游饭店年平均营收 (百万元) | 单家五星级旅游饭店年平均利润 (百万元)\n2018 | 104.22 | 9.87\n2019 | 99.11 | 7.47\n2020 | 57.87 | -3.87\n2021 | 68.99 | -2.9\n2022 | 56.29 | -9.48\n2023 | 87.99 | 5.96'}}
```
The meanings of the result parameters are as follows:
- `image`: Indicates the path of the input image to be predicted
- `result`: The result information predicted by the model

The visualized printed prediction result is as follows:

```bash
年份 | 单家五星级旅游饭店年平均营收 (百万元) | 单家五星级旅游饭店年平均利润 (百万元)
2018 | 104.22 | 9.87
2019 | 99.11 | 7.47
2020 | 57.87 | -3.87
2021 | 68.99 | -2.9
2022 | 56.29 | -9.48
2023 | 87.99 | 5.96
```

Related methods, parameters, and descriptions are as follows:

* `create_model` instantiates the document-like visual language model (taking `PP-Chart2Table` as an example here), with specific explanations as follows:
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>Model name</td>
<td><code>str</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Model storage path</td>
<td><code>str</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Model inference device</td>
<td><code>str</code></td>
<td>Support specifying specific GPU card number, such as "gpu:0", other hardware specific card numbers, such as "npu:0", CPU as "cpu".</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference plugins. Currently not supported.</td>
<td><code>bool</code></td>
<td>None</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>High-performance inference configuration. Currently not supported.</td>
<td><code>dict</code> | <code>None</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
</table>

* Among them, `model_name` must be specified. After specifying `model_name`, the default PaddleX built-in model parameters are used. On this basis, if `model_dir` is specified, the user-defined model is used.

* Call the `predict()` method of the document-like visual language model for inference prediction. The `predict()` method parameters include `input`, `batch_size`, with specific explanations as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted</td>
<td><code>dict</code></td>
<td>
<code>Dict</code>, as multimodal models have different input requirements, it needs to be determined based on the specific model. Specifically:
<li>The input form for PP-Chart2Table is <code>{'image': image_path}</code></li>
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>Integer</td>
<td>1</td>
</tr>
</table>

* Process the prediction results. The prediction result for each sample is the corresponding Result object, which supports operations like printing and saving as a `json` file:

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
<td rowspan = "3">Print results to terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data for better readability, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will keep the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the result as a json formatted file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. When it's a directory, the saved file name matches the input file type name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data for better readability, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> will keep the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
</table>

* Additionally, it is also possible to obtain prediction results through attributes, as follows:

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
</table>

For more information on using the API for single model inference in PaddleX, you can refer to [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.md).

## IV. Secondary Development
The current module temporarily does not support fine-tuning training, only inference integration. Support for fine-tuning training in this module is planned for the future.
