
# Instruction for Using Document Visual-Language Model Module

## 1. Overview
The document visual-language model is a cutting-edge multimodal processing technology aimed at overcoming the limitations of traditional document processing methods. Traditional methods are often restricted to handling documents of specific formats or predefined categories, whereas document visual-language models can integrate visual and linguistic information to comprehend and process diverse document content. By combining computer vision with natural language processing, the model can identify images, texts, and their interrelationships within documents, even understanding semantic information within complex layout structures. This makes document processing more intelligent and flexible, with enhanced generalization capabilities, showcasing broad application prospects in fields such as automated office and information extraction.

## 2. Supported Model List

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Storage Size (GB)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-DocBee-2B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocBee-2B_infer.tar">Inference Model</a></td>
<td>4.2</td>
<td rowspan="2">PP-DocBee is a multimodal large model developed by the PaddlePaddle team, focused on document understanding with excellent performance on Chinese document understanding tasks. The model is fine-tuned and optimized using nearly 5 million multimodal datasets for document understanding, including general VQA, OCR, table, text-rich, math and complex reasoning, synthetic, and pure text data, with different training data ratios. On several authoritative English document understanding evaluation leaderboards in academia, PP-DocBee has generally achieved SOTA at the same parameter level. In internal business Chinese scenarios, PP-DocBee also exceeds current popular open-source and closed-source models.</td>
</tr>
<tr>
<td>PP-DocBee-7B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-DocBee-7B_infer.tar">Inference Model</a></td>
<td>15.8</td>
</tr>
</table>

## 3. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For details, refer to [PaddleX Local Installation Guide](../../../installation/installation.md).

After completing the installation of the wheel package, a few lines of code can execute the inference of the document visual-language model module, allowing model switching within this module at will. You may also integrate inference from models in the open document visual-language model module into your project. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/medal_table.png) to your local.

```python
from paddlex import create_model
model = create_model('PP-DocBee-2B')
results = model.predict(
    input={"image": "medal_table.png", "query": "Identify the content of this table"},
    batch_size=1
)
for res in results:
    res.print()
    res.save_to_json(f"./output/res.json")
```

The results obtained will be:

```bash
{'res': {'image': 'medal_table.png', 'query': 'Identify the content of this table', 'result': '| Rank | Country/Region | Gold | Silver | Bronze | Total Medals |\n| --- | --- | --- | --- | --- | --- |\n| 1 | China (CHN) | 48 | 22 | 30 | 100 |\n| 2 | USA | 36 | 39 | 37 | 112 |\n| 3 | Russia (RUS) | 24 | 13 | 23 | 60 |\n| 4 | UK (GBR) | 19 | 13 | 19 | 51 |\n| 5 | Germany (GER) | 16 | 11 | 14 | 41 |\n| 6 | Australia (AUS) | 14 | 15 | 17 | 46 |\n| 7 | Korea (KOR) | 13 | 11 | 8 | 32 |\n| 8 | Japan (JPN) | 9 | 8 | 8 | 25 |\n| 9 | Italy (ITA) | 8 | 9 | 10 | 27 |\n| 10 | France (FRA) | 7 | 16 | 20 | 43 |\n| 11 | Netherlands (NED) | 7 | 5 | 4 | 16 |\n| 12 | Ukraine (UKR) | 7 | 4 | 11 | 22 |\n| 13 | Kenya (KEN) | 6 | 4 | 6 | 16 |\n| 14 | Spain (ESP) | 5 | 11 | 3 | 19 |\n| 15 | Jamaica (JAM) | 5 | 4 | 2 | 11 |\n'}}
```
The parameters in the results have the following meaning:

- `image`: Represents the path of the input image to be predicted
- `query`: Represents the input text information to be predicted
- `result`: The result information predicted by the model

The visualized prediction results are as follows:

```bash
| Rank | Country/Region | Gold | Silver | Bronze | Total Medals |
| --- | --- | --- | --- | --- | --- |
| 1 | China (CHN) | 48 | 22 | 30 | 100 |
| 2 | USA | 36 | 39 | 37 | 112 |
| 3 | Russia (RUS) | 24 | 13 | 23 | 60 |
| 4 | UK (GBR) | 19 | 13 | 19 | 51 |
| 5 | Germany (GER) | 16 | 11 | 14 | 41 |
| 6 | Australia (AUS) | 14 | 15 | 17 | 46 |
| 7 | Korea (KOR) | 13 | 11 | 8 | 32 |
| 8 | Japan (JPN) | 9 | 8 | 8 | 25 |
| 9 | Italy (ITA) | 8 | 9 | 10 | 27 |
| 10 | France (FRA) | 7 | 16 | 20 | 43 |
| 11 | Netherlands (NED) | 7 | 5 | 4 | 16 |
| 12 | Ukraine (UKR) | 7 | 4 | 11 | 22 |
| 13 | Kenya (KEN) | 6 | 4 | 6 | 16 |
| 14 | Spain (ESP) | 5 | 11 | 3 | 19 |
| 15 | Jamaica (JAM) | 5 | 4 | 2 | 11 |
```

The explanation of related methods and parameters are as follows:

* `create_model` instantiates the document visual-language model (using `PP-DocBee-2B` as an example), detailed as follows:

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
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>Model inference device</td>
<td><code>str</code></td>
<td>Supports specifying specific GPU card numbers, such as "gpu:0", specific hardware card numbers like "npu:0", or CPU as "cpu".</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Enable high-performance inference</td>
<td><code>bool</code></td>
<td>None, currently not supported</td>
<td><code>False</code></td>
</tr>
</table>

* `model_name` must be specified. After specifying it, the default model parameters built into PaddleX are used, and if `model_dir` is specified, the user-defined model is used.

* The `predict()` method of the document visual-language model is called for inference prediction. The `predict()` method parameters include `input` and `batch_size`, detailed as follows:

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
<code>Dict</code>, needs to be determined according to the specific model. For the PP-DocBee series, the input is {'image': image_path, 'query': query_text}
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>Integer (currently only supports 1)</td>
<td>1</td>
</tr>
</table>

* The prediction results are processed, with the prediction result for each sample being the corresponding Result object, supporting operations such as printing and saving as a `json` file:

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
<td>Whether to use <code>JSON</code> indentation to format output content</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify indentation levels to enhance the readability of output <code>JSON</code> data. This is effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters are escaped; <code>False</code> retains the original characters. This is effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save results as json file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path for saving the file. When specified as a directory, saved file names match the input file types</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify indentation levels to enhance the readability of output <code>JSON</code> data. This is effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters are escaped; <code>False</code> retains the original characters. This is effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
</table>

* In addition, the prediction results can also be accessed through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Get the predicted results in <code>json</code> format</td>
</tr>
</table>

For more usage instructions on the API for single model inference in PaddleX, you can refer to [Instructions for Using PaddleX Single Model Python API](../../instructions/model_python_API.md).

## 4. Secondary Development

The current module temporarily does not support fine-tuning training and only supports inference integration. Fine-tuning training for this module is planned to be supported in the future.
