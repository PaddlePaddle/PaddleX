# Document Understanding Pipeline User Guide

## 1. Introduction to Document Understanding Pipeline

The Document Understanding Pipeline is an advanced document processing technology based on Vision-Language Models (VLM), designed to overcome the limitations of traditional document processing. Traditional methods rely on fixed templates or predefined rules to parse documents, but this pipeline uses the multimodal capabilities of VLM to accurately answer user questions by integrating visual and linguistic information with just the input of document images and user queries. This technology does not require pre-training for specific document formats, allowing it to flexibly handle diverse document content and significantly enhance the generalization and practicality of document processing. It has broad application prospects in scenarios such as intelligent Q&A and information extraction. Currently, this pipeline does not support secondary development of VLM models, but future support is planned.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/doc_understanding/doc_understanding.png">

<b>The Document Understanding Pipeline includes document-based vision-language model modules. You can choose the model to use based on the benchmark test data below.</b>

### 1.1 Model benchmark data

<b>If you prioritize model accuracy, choose a model with higher accuracy; if you care more about inference speed, choose a faster model; if you are concerned about storage size, choose a model with a smaller storage footprint.</b>

> The inference time only includes the model inference time and does not include the time for pre- or post-processing.

<p><b>Document-based Vision-Language Model Modules (Optional):</b></p>

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Storage Size (GB)</th>
<th>Model Score</th>
<th>Description</th>
</tr>
<tr>
<td>PP-DocBee-2B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBee-2B_infer.tar">Inference Model</a></td>
<td>4.2</td>
<td>765</td>
<td rowspan="2">PP-DocBee is a multimodal large model developed by the PaddlePaddle team, focused on document understanding with excellent performance on Chinese document understanding tasks. The model is fine-tuned and optimized using nearly 5 million multimodal datasets for document understanding, including general VQA, OCR, table, text-rich, math and complex reasoning, synthetic, and pure text data, with different training data ratios. On several authoritative English document understanding evaluation leaderboards in academia, PP-DocBee has generally achieved SOTA at the same parameter level. In internal business Chinese scenarios, PP-DocBee also exceeds current popular open-source and closed-source models.</td>
</tr>
<tr>
<td>PP-DocBee-7B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBee-7B_infer.tar">Inference Model</a></td>
<td>15.8</td>
<td>-</td>
</tr>
<tr>
<td>PP-DocBee2-3B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBee2-3B_infer.tar">Inference Model</a></td>
<td>7.6</td>
<td>852</td>
<td>PP-DocBee2 is a multimodal large model independently developed by the PaddlePaddle team, specifically tailored for document understanding. Building upon PP-DocBee, the team has further optimized the foundational model and introduced a new data optimization scheme to enhance data quality. With just a relatively small dataset of 470,000 samples generated using the team's proprietary data synthesis strategy, PP-DocBee2 demonstrates superior performance in Chinese document understanding tasks. In terms of internal business metrics for Chinese-language scenarios, PP-DocBee2 has achieved an approximately 11.4% improvement over PP-DocBee, outperforming both current popular open-source and closed-source models of a similar scale.</td>
</tr>
</table>

<b>Note: The total scores of the above models are based on the test results from the internal evaluation set. All images in the internal evaluation set have a resolution (height, width) of (1680, 1204), with a total of 1,196 data entries. These entries cover various scenarios such as financial reports, laws and regulations, science and engineering papers, instruction manuals, liberal arts papers, contracts, research reports, etc. There are currently no plans to make this dataset publicly available.</b>

### 1.2 Pipeline benchmark data

<details>
<summary>Click to expand/collapse the table</summary>

<table border="1">
<tr><th>Pipeline configuration</th><th>Hardware</th><th>Avg. inference time (ms)</th><th>Peak CPU utilization (%)</th><th>Avg. CPU utilization (%)</th><th>Peak host memory (MB)</th><th>Avg. host memory (MB)</th><th>Peak GPU utilization (%)</th><th>Avg. GPU utilization (%)</th><th>Peak device memory (MB)</th><th>Avg. device memory (MB)</th></tr>
<tr>
<td rowspan="9">doc_preprocessor-default</td>
<td>Intel 6271C</td>
<td>1127.85</td>
<td>1012.50</td>
<td>789.07</td>
<td>1913.89</td>
<td>1650.30</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>764.23</td>
<td>1003.70</td>
<td>782.77</td>
<td>1916.81</td>
<td>1655.66</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>96.74</td>
<td>174.70</td>
<td>135.91</td>
<td>2023.54</td>
<td>1963.53</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>76.66</td>
<td>148.90</td>
<td>125.73</td>
<td>1748.06</td>
<td>1683.35</td>
<td>11</td>
<td>5.86</td>
<td>828.00</td>
<td>828.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>114.96</td>
<td>168.60</td>
<td>131.05</td>
<td>1837.66</td>
<td>1707.50</td>
<td>26</td>
<td>11.65</td>
<td>694.00</td>
<td>694.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>69.34</td>
<td>140.90</td>
<td>122.80</td>
<td>1992.86</td>
<td>1911.34</td>
<td>11</td>
<td>7.75</td>
<td>890.00</td>
<td>890.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>81.65</td>
<td>146.40</td>
<td>125.37</td>
<td>1896.46</td>
<td>1825.98</td>
<td>14</td>
<td>5.93</td>
<td>586.00</td>
<td>586.00</td>
</tr>
<tr>
<td>M4</td>
<td>368.93</td>
<td>118.20</td>
<td>103.40</td>
<td>1973.23</td>
<td>1756.00</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>122.40</td>
<td>156.90</td>
<td>123.75</td>
<td>1874.29</td>
<td>1727.01</td>
<td>73</td>
<td>23.81</td>
<td>450.00</td>
<td>450.00</td>
</tr>
<tr>
<td rowspan="9">doc_preprocessor-clsonly</td>
<td>Intel 6271C</td>
<td>55.12</td>
<td>1085.60</td>
<td>1057.12</td>
<td>1254.27</td>
<td>1203.36</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>59.48</td>
<td>1051.60</td>
<td>1043.39</td>
<td>1263.88</td>
<td>1206.67</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>47.44</td>
<td>202.70</td>
<td>173.75</td>
<td>1868.11</td>
<td>1810.80</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>43.27</td>
<td>167.80</td>
<td>150.56</td>
<td>1347.92</td>
<td>1325.32</td>
<td>2</td>
<td>1.25</td>
<td>514.00</td>
<td>514.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>53.72</td>
<td>202.70</td>
<td>170.72</td>
<td>1404.16</td>
<td>1352.74</td>
<td>2</td>
<td>1.20</td>
<td>394.00</td>
<td>394.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>41.89</td>
<td>153.80</td>
<td>141.46</td>
<td>1590.48</td>
<td>1540.25</td>
<td>2</td>
<td>1.12</td>
<td>578.00</td>
<td>578.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>44.61</td>
<td>179.80</td>
<td>156.96</td>
<td>1569.85</td>
<td>1514.64</td>
<td>2</td>
<td>0.88</td>
<td>302.00</td>
<td>302.00</td>
</tr>
<tr>
<td>M4</td>
<td>34.12</td>
<td>127.60</td>
<td>123.24</td>
<td>1459.06</td>
<td>1386.33</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>51.81</td>
<td>180.80</td>
<td>154.34</td>
<td>1515.43</td>
<td>1465.99</td>
<td>4</td>
<td>2.89</td>
<td>160.00</td>
<td>160.00</td>
</tr>
<tr>
<td rowspan="9">doc_preprocessor-unwarponly</td>
<td>Intel 6271C</td>
<td>1092.82</td>
<td>1002.10</td>
<td>749.82</td>
<td>1875.45</td>
<td>1680.20</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>747.32</td>
<td>951.30</td>
<td>754.01</td>
<td>1843.54</td>
<td>1645.73</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>95.22</td>
<td>101.20</td>
<td>100.08</td>
<td>1960.48</td>
<td>1863.72</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>78.05</td>
<td>100.90</td>
<td>100.15</td>
<td>1570.25</td>
<td>1446.01</td>
<td>9</td>
<td>5.57</td>
<td>788.00</td>
<td>788.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>130.84</td>
<td>100.90</td>
<td>100.25</td>
<td>1561.52</td>
<td>1459.73</td>
<td>24</td>
<td>11.35</td>
<td>656.00</td>
<td>656.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>68.62</td>
<td>106.90</td>
<td>100.62</td>
<td>1808.00</td>
<td>1715.35</td>
<td>6</td>
<td>4.83</td>
<td>684.00</td>
<td>684.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>84.19</td>
<td>101.90</td>
<td>100.14</td>
<td>1848.25</td>
<td>1719.24</td>
<td>19</td>
<td>7.73</td>
<td>544.00</td>
<td>544.00</td>
</tr>
<tr>
<td>M4</td>
<td>295.15</td>
<td>102.40</td>
<td>100.76</td>
<td>1957.42</td>
<td>1734.67</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>117.25</td>
<td>101.80</td>
<td>100.36</td>
<td>1674.25</td>
<td>1605.14</td>
<td>48</td>
<td>26.45</td>
<td>412.00</td>
<td>412.00</td>
</tr>
</table>


<table border="1">
<tr><th>Pipeline configuration</th><th>description</th></tr>
<tr>
<td>doc_preprocessor-default</td>
<td>Default configuration</td>
</tr>
<tr>
<td>doc_preprocessor-clsonly</td>
<td>Based on the default configuration, only document image orientation classification is enabled</td>
</tr>
<tr>
<td>doc_preprocessor-unwarponly</td>
<td>Based on the default configuration, only text image rectification is enabled</td>
</tr>
</table>
</details>


* Test environment:
    * PaddlePaddle 3.1.0、CUDA 11.8、cuDNN 8.9
    * PaddleX @ develop (f1eb28e23cfa54ce3e9234d2e61fcb87c93cf407)
    * Docker image: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda11.8-cudnn8.9

* Note:
    * Since we did not collect device memory data for NPU and XPU, the corresponding entries in the table are marked as N/A.

## 2. Quick Start

### 2.1 Local Experience

> ❗ Before using the Document Understanding Pipeline locally, ensure you have installed the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.md). If you wish to selectively install dependencies, refer to the relevant instructions in the installation guide. The dependency group for this pipeline is `multimodal`.

#### 2.1.1 Integration via Python Script

* The Document Understanding Pipeline can be quickly inferred with just a few lines of code as shown below:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="doc_understanding")
output = pipeline.predict(
    {
        "image": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/medal_table.png",
        "query": "识别这份表格的内容, 以markdown格式输出"
    }
)
for res in output:
    res.print()
    res.save_to_json("./output/")
```

In the above Python script, the following steps are performed:

1. Instantiate the Document Understanding Pipeline object through `create_pipeline()`. The parameter details are as follows:

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
<td>Pipeline name or configuration file path. If it's a pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the pipeline (if set simultaneously with <code>pipeline</code>, it has a higher priority and requires the pipeline name to be consistent with <code>pipeline</code>).</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Inference device for the pipeline. Supports specifying a specific GPU card number, such as "gpu:0", or other hardware card numbers, like "npu:0", or CPU as "cpu".</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable the high-performance inference plugin. If set to <code>None</code>, the setting from the configuration file or <code>config</code> will be used. Not supported for now.</td>
<td><code>bool</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>High-performance inference configuration. Not supported for now.</td>
<td><code>dict</code> | <code>None</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

2. Call the `predict()` method of the Document Understanding Pipeline object for inference prediction. This method returns a `generator`. Below are the parameters of the `predict()` method and their descriptions:

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
<td>Data to be predicted, currently only supports dictionary-type input</td>
<td><code>Python Dict</code></td>
<td>
<ul>
  <li><b>Python Dict</b>: For PP-DocBee, the input format is: <code>{"image":/path/to/image, "query": user question}</code>, representing the input image and the corresponding user question.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</table>

3. Process the prediction results. The prediction result for each sample is a corresponding Result object, and supports operations such as printing and saving as a `json` file:

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
<td rowspan = "3">Print results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content with <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify indentation level to beautify the <code>JSON</code> output, making it more readable, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save results as a json format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. If it is a directory, the saved file name is consistent with the input file type name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify indentation level to beautify the <code>JSON</code> output, making it more readable, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal. The printed content includes:

  - `image`: `(str)` The input path of the image

  - `query`: `(str)` The question related to the input image

  - `result`: `(str)` The output result from the model

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.json`. If specified as a file, it will be directly saved to that file.

* Additionally, you can also access the visualized image with results and prediction results through attributes, as follows:

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
<tr>
<td rowspan = "2"><code>img</code></td>
<td rowspan = "2">Get the visualized image in <code>dict</code> format</td>
</tr>
</table>

- The prediction result obtained from the `json` attribute is data of type dict, and the related content is consistent with the content saved by calling the `save_to_json()` method.

Additionally, you can obtain the configuration file for the Document Understanding Pipeline and load the configuration file for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config doc_understanding --save_path ./my_path
```

If you have obtained the configuration file, you can customize various configurations of the Document Understanding Pipeline, just modify the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file. For example:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/doc_understanding.yaml")
output = pipeline.predict(
    {
        "image": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/medal_table.png",
        "query": "识别这份表格的内容, 以markdown格式输出"
    }
)
for res in output:
    res.print()
    res.save_to_json("./output/")
```

<b>Note:</b> The parameters in the configuration file are the initialization parameters of the pipeline. If you want to change the initialization parameters of the Document Understanding Pipeline, you can directly modify the parameters in the configuration file and load the configuration file for prediction. Meanwhile, CLI prediction also supports passing in the configuration file, `--pipeline` specifies the path of the configuration file.

## 3. Development Integration/Deployment

If the pipeline meets your requirements for inference speed and accuracy, you can directly proceed to development integration/deployment.

If you need to directly apply the pipeline in your Python project, you can refer to the example code in [2.1.2 Python Script Method](#212-python-script-method-integration).

In addition, PaddleX also provides three other deployment methods, as detailed below:

🚀 <b>High-Performance Inference</b> (This pipeline does not support it currently): In real-world production environments, many applications have strict standards for performance indicators of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin aimed at deeply optimizing model inference and pre-post processing to achieve significant speed improvements in end-to-end processes. For detailed high-performance inference processes, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

☁️ <b>Service Deployment</b>: Service deployment is a common form of deployment in real-world production environments. By encapsulating inference functions into services, clients can access these services via network requests to obtain inference results. PaddleX supports various pipeline service deployment solutions. For detailed pipeline service deployment processes, please refer to the [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.md).

Below is a basic service deployment API reference and multilingual service call example:

<details><summary>API Reference</summary>

<p>Main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request and response bodies are in JSON format (JSON object).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body contains the following attributes:</li>
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
<li>When the request is not processed successfully, the response body contains the following attributes:</li>
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
<p>Generate a response based on the input message.</p>
<p><code>POST /document-understanding</code></p>
<p>Note: The above interface is an alias for /chat/completion and is compatible with OpenAI's interface.</p>

<ul>
<li>The request body contains the following attributes:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>model</code></td>
<td><code>string</code></td>
<td>Name of the model to use</td>
<td>Required</td>
<td>-</td>
</tr>
<tr>
<td><code>messages</code></td>
<td><code>array</code></td>
<td>List of dialogue messages</td>
<td>Required</td>
<td>-</td>
</tr>
<tr>
<td><code>max_tokens</code></td>
<td><code>integer</code></td>
<td>Maximum number of tokens to generate</td>
<td>Optional</td>
<td>1024</td>
</tr>
<tr>
<td><code>temperature</code></td>
<td><code>float</code></td>
<td>Sampling temperature</td>
<td>Optional</td>
<td>0.1</td>
</tr>
<tr>
<td><code>top_p</code></td>
<td><code>float</code></td>
<td>Top probability for nucleus sampling</td>
<td>Optional</td>
<td>0.95</td>
</tr>
<tr>
<td><code>stream</code></td>
<td><code>boolean</code></td>
<td>Whether to output in a streaming manner</td>
<td>Optional</td>
<td>false</td>
</tr>
<tr>
<td><code>max_image_tokens</code></td>
<td><code>int</code></td>
<td>Maximum number of tokens of input image</td>
<td>Optional</td>
<td>None</td>
</tr>
</tbody>
</table>

<p>Each element in <code>messages</code> is an <code>object</code> with the following attributes:</p>
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
<td><code>role</code></td>
<td><code>string</code></td>
<td>Role of the message (user/assistant/system)</td>
<td>Required</td>
</tr>
<tr>
<td><code>content</code></td>
<td><code>string</code> or <code>array</code></td>
<td>Message content (text or mixed content)</td>
<td>Required</td>
</tr>
</tbody>
</table>

<p>When <code>content</code> is an array, each element is an <code>object</code> with the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>type</code></td>
<td><code>string</code></td>
<td>Content type (text/image_url)</td>
<td>Required</td>
<td>-</td>
</tr>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>Text content (when type is text)</td>
<td>Conditionally Required</td>
<td>-</td>
</tr>
<tr>
<td><code>image_url</code></td>
<td><code>string</code> or <code>object</code></td>
<td>Image URL or object (when type is image_url)</td>
<td>Conditionally Required</td>
<td>-</td>
</tr>
</tbody>
</table>

<p>When <code>image_url</code> is an object, it contains the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>url</code></td>
<td><code>string</code></td>
<td>Image URL</td>
<td>Required</td>
<td>-</td>
</tr>
<tr>
<td><code>detail</code></td>
<td><code>string</code></td>
<td>Image detail processing method (low/high/auto)</td>
<td>Optional</td>
<td>auto</td>
</tr>
</tbody>
</table>

<p>When the request is processed successfully, the <code>result</code> in the response body contains the following attributes:</p>
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
<td><code>string</code></td>
<td>Request ID</td>
</tr>
<tr>
<td><code>object</code></td>
<td><code>string</code></td>
<td>Object type (chat.completion)</td>
</tr>
<tr>
<td><code>created</code></td>
<td><code>integer</code></td>
<td>Creation timestamp</td>
</tr>
<tr>
<td><code>choices</code></td>
<td><code>array</code></td>
<td>Generated result options</td>
</tr>
<tr>
<td><code>usage</code></td>
<td><code>object</code></td>
<td>Token usage details</td>
</tr>
</tbody>
</table>

<p>Each element in <code>choices</code> is a <code>Choice</code> object with the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Optional Values</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>finish_reason</code></td>
<td><code>string</code></td>
<td>Reason the model stopped generating tokens</td>
<td><code>stop</code> (naturally stopped)<br><code>length</code> (reached max token count)<br><code>tool_calls</code> (called a tool)<br><code>content_filter</code> (content filtered)<br><code>function_call</code> (called a function, deprecated)</td>
</tr>
<tr>
<td><code>index</code></td>
<td><code>integer</code></td>
<td>Index of the option in the list</td>
<td>-</td>
</tr>
<tr>
<td><code>logprobs</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Log probability information of the option</td>
<td>-</td>
</tr>
<tr>
<td><code>message</code></td>
<td><code>ChatCompletionMessage</code></td>
<td>Chat message generated by the model</td>
<td>-</td>
</tr>
</tbody>
</table>

<p>The <code>message</code> object contains the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Note</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>content</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Message content</td>
<td>May be null</td>
</tr>
<tr>
<td><code>refusal</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Refusal message generated by the model</td>
<td>Provided when content is refused</td>
</tr>
<tr>
<td><code>role</code></td>
<td><code>string</code></td>
<td>Role of the message author</td>
<td>Fixed at <code>"assistant"</code></td>
</tr>
<tr>
<td><code>audio</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Audio output data</td>
<td>Provided when audio output is requested<br><a href="https://platform.openai.com/docs/guides/audio">Learn more</a></td>
</tr>
<tr>
<td><code>function_call</code></td>
<td><code>object</code> | <code>null</code></td>
<td>Name and parameters of the function to be called</td>
<td>Deprecated, recommended to use <code>tool_calls</code></td>
</tr>
<tr>
<td><code>tool_calls</code></td>
<td><code>array</code> | <code>null</code></td>
<td>Tool calls generated by the model</td>
<td>e.g., function calls, etc.</td>
</tr>
</tbody>
</table>

<p>The <code>usage</code> object contains the following attributes:</p>
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
<td><code>prompt_tokens</code></td>
<td><code>integer</code></td>
<td>Number of prompt tokens</td>
</tr>
<tr>
<td><code>completion_tokens</code></td>
<td><code>integer</code></td>
<td>Number of generated tokens</td>
</tr>
<tr>
<td><code>total_tokens</code></td>
<td><code>integer</code></td>
<td>Total number of tokens</td>
</tr>
</tbody>
</table>
<p>An example of <code>result</code> is shown below:</p>
<pre><code class="language-json">{
    "id": "ed960013-eb19-43fa-b826-3c1b59657e35",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "| 名次 | 国家/地区 | 金牌 | 银牌 | 铜牌 | 奖牌总数 |\n| --- | --- | --- | --- | --- | --- |\n| 1 | 中国（CHN） | 48 | 22 | 30 | 100 |\n| 2 | 美国（USA） | 36 | 39 | 37 | 112 |\n| 3 | 俄罗斯（RUS） | 24 | 13 | 23 | 60 |\n| 4 | 英国（GBR） | 19 | 13 | 19 | 51 |\n| 5 | 德国（GER） | 16 | 11 | 14 | 41 |\n| 6 | 澳大利亚（AUS） | 14 | 15 | 17 | 46 |\n| 7 | 韩国（KOR） | 13 | 11 | 8 | 32 |\n| 8 | 日本（JPN） | 9 | 8 | 8 | 25 |\n| 9 | 意大利（ITA） | 8 | 9 | 10 | 27 |\n| 10 | 法国（FRA） | 7 | 16 | 20 | 43 |\n| 11 | 荷兰（NED） | 7 | 5 | 4 | 16 |\n| 12 | 乌克兰（UKR） | 7 | 4 | 11 | 22 |\n| 13 | 肯尼亚（KEN） | 6 | 4 | 6 | 16 |\n| 14 | 西班牙（ESP） | 5 | 11 | 3 | 19 |\n| 15 | 牙买加（JAM） | 5 | 4 | 2 | 11 |\n",
                "role": "assistant"
            }
        }
    ],
    "created": 1745218041,
    "model": "pp-docbee",
    "object": "chat.completion"
}
</code></pre></details>

<details><summary>Examples of Multilingual Service Calls</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
from openai import OpenAI

API_BASE_URL = "http://127.0.0.1:8080"

# Initialize OpenAI client
client = OpenAI(
    api_key='xxxxxxxxx',
    base_url=f'{API_BASE_URL}'
)

# Function to convert image to base64
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Input image path
image_path = "medal_table.png"

# Convert the original image to base64
base64_image = encode_image(image_path)

# Submit information to the PP-DocBee model
response = client.chat.completions.create(
    model="pp-docbee", # Select model
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content":[
                {
                    "type": "text",
                    "text": "Recognize the content of this table and output the content in HTML format."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                },
            ]
        },
    ],
)
content = response.choices[0].message.content
print('Reply:', content)
</code></pre></details>
</details>
<br/>

📱 <b>On-Device Deployment</b>: Edge deployment is a way of placing computing and data processing functions on the user device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment processes, please refer to the [PaddleX On-Device Deployment Guide](../../../pipeline_deploy/on_device_deployment.md).
You can choose the appropriate deployment method for your needs and proceed with subsequent AI application integration.


## 4. Secondary Development

Currently, this pipeline does not support fine-tuning training and only supports inference integration. Future support for fine-tuning training is planned.

## 5. Multi-Hardware Support

Currently, this pipeline only supports GPU and CPU inference. Future support for more hardware adaptations is planned.
