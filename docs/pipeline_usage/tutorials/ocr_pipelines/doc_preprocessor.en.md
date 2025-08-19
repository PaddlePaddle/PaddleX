---
comments: true
---

# Document Image Preprocessing Pipeline Tutorial

## 1. Introduction to the Do Pipeline

The document image preprocessing pipeline integrates two major functions: document orientation classification and geometric distortion correction. The document orientation classification can automatically identify the four orientations of a document (0°, 90°, 180°, 270°) to ensure that the document is processed in the correct direction for subsequent tasks. The geometric distortion correction model is used to correct geometric distortions that occur during the document's photographing or scanning process, restoring the document to its original shape and proportions. This is suitable for digital document management, preprocessing for doc_preprocessor recognition, and any scenario where improving document image quality is necessary. Through automated orientation correction and distortion correction, this module significantly enhances the accuracy and efficiency of document processing, providing users with a more reliable foundation for image analysis. The pipeline also offers flexible service deployment options, supporting invocation using various programming languages on multiple hardware platforms. Moreover, it provides the capability for further development, allowing you to train and fine-tune on your own dataset based on this pipeline, with the trained models being seamlessly integrable.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/doc_preprocessor/02.jpg">

**The general document image preprocessing pipeline includes optional document image orientation classification module and document image correction module**. Each module contains multiple models, and you can choose the model based on the benchmark test data below.

### 1.1 Model benchmark data

> The inference time only includes the model inference time and does not include the time for pre- or post-processing.

<p><b>Document Image Orientation Classification Module (Optional):</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
<td>99.06</td>
<td>2.62 / 0.59</td>
<td>3.24 / 1.19</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, with four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees.</td>
</tr>
</tbody>
</table>

<p><b>Text Image Correction Module (Optional):</b></p>

<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>CER</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (MB)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
<td>0.179</td>
<td>19.05 / 19.05</td>
<td>- / 869.82</td>
<td>30.3</td>
<td>High-accuracy text image rectification model</td>
</tr>
</tbody>
</table>

<strong>Test Environment Description:</strong>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
                    <li><strong>Test Dataset：</strong>
                        <ul>
                          <li>Document Image Orientation Classification Module: A self-built dataset using PaddleX, covering multiple scenarios such as ID cards and documents, containing 1000 images.</li>
                          <li>Text Image Rectification Module: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a>.</li>
                        </ul>
                    </li>
              <li><strong>Hardware Configuration:</strong>
                  <ul>
                      <li>GPU: NVIDIA Tesla T4</li>
                      <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
                  </ul>
              </li>
              <li><strong>Software Environment:</strong>
                  <ul>
                      <li>Ubuntu 20.04 / CUDA 11.8 / cuDNN 8.9 / TensorRT 8.6.1.6</li>
                      <li>paddlepaddle 3.0.0 / paddlex 3.0.3</li>
                  </ul>
              </li>
              </li>
          </ul>
      </li>
      <li><b>Inference Mode Description</b></li>
  </ul>

<table border="1">
    <thead>
        <tr>
            <th>Mode</th>
            <th>GPU Configuration </th>
            <th>CPU Configuration </th>
            <th>Acceleration Technology Combination</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Normal Mode</td>
            <td>FP32 Precision / No TRT Acceleration</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>High-Performance Mode</td>
            <td>Optimal combination of pre-selected precision types and acceleration strategies</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>Pre-selected optimal backend (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>

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

PaddleX supports experiencing the effects of the document image preprocessing pipeline locally via command line or Python.

Before using the document image preprocessing pipeline locally, please ensure you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.md). If you wish to selectively install dependencies, please refer to the relevant instructions in the installation guide. The dependency group corresponding to this pipeline is `ocr`.

### 2.1 Local Experience

#### 2.1.1 Command Line Experience
You can quickly experience the effects of the document image preprocessing pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/doc_test_rotated.jpg) and replace `--input` with the local path to perform predictions.

```bash
paddlex --pipeline doc_preprocessor \
        --input doc_test_rotated.jpg \
        --use_doc_orientation_classify True \
        --use_doc_unwarping True \
        --save_path ./output \
        --device gpu:0
```

<b>Note: </b>The official models would be download from HuggingFace by default. If can't access to HuggingFace, please set the environment variable `PADDLE_PDX_MODEL_SOURCE="BOS"` to change the model source to BOS. In the future, more model sources will be supported.

You can refer to the parameter descriptions in [2.1.2 Python Script Integration](#212-python-script-integration) for related parameter details. Supports specifying multiple devices simultaneously for parallel inference. For details, please refer to the documentation on pipeline parallel inference.

After running, the results will be printed to the terminal as follows:

<pre><code>{'res': {'input_path': 'doc_test_rotated.jpg', 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 180}}
</code></pre>

You can refer to the results explanation in [2.1.2 Python Script Integration](#212-python-script-integration) for a description of the output parameters.

The visualized results are saved under `save_path`. The visualized results are as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/doc_preprocessor/02.jpg">


#### 2.1.2 Python Script Integration

The above command line is for quickly experiencing and viewing the effect. Generally, in a project, it is often necessary to integrate through code. You can complete quick inference in a pipeline with just a few lines of code. The inference code is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="doc_preprocessor")
output = pipeline.predict(
    input="doc_test_rotated.jpg",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")
```

In the above Python script, the following steps were executed:


(1) Instantiate the `doc_preprocessor` pipeline object using `create_pipeline()`. The specific parameter descriptions are as follows:


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
<td>The pipeline name or the path to the pipeline configuration file. If it is a pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Inference device for the pipeline. Supports specifying the GPU card number, such as "gpu:0", other hardware card numbers, such as "npu:0", and CPU as "cpu". Supports specifying multiple devices simultaneously for parallel inference. For details, please refer to <a href="../../instructions/parallel_inference.en.md#specifying-multiple-inference-devices">Pipeline Parallel Inference</a>.</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable the high-performance inference plugin. If set to <code>None</code>, the setting from the configuration file or <code>config</code> will be used.</td>
<td><code>bool</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>High-performance inference configuration</td>
<td><code>dict</code> | <code>None</code></td>
<td>None</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>


(2) Call the `predict()` method of the doc_preprocessor pipeline object for inference prediction. This method will return a `generator`. Below are the parameters of the `predict()` method and their descriptions:


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
<td>Data to be predicted, supporting various input types, required</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
  <li><b>Python Var</b>: Such as image data represented by <code>numpy.ndarray</code></li>
  <li><b>str</b>: Such as the local path of an image file or PDF file: <code>/root/data/img.jpg</code>; <b>As URL link</b>, such as the network URL of an image file or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_doc_preprocessor_002.png">example</a>; <b>As a local directory</b>, which should contain images to be predicted, such as a local path: <code>/root/data/</code> (currently does not support directory prediction for PDFs, PDF files need to be specified to the specific file path)</li>
  <li><b>List</b>: List elements must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use the document orientation classification module</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the document unwarping correction module</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>: <code>True</code> or <code>False</code>;</li>
  <li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, initialized to <code>True</code>;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>


(3) Process the prediction results, where the prediction result for each sample is of `dict` type. Additionally, these results support operations such as printing, saving as an image, and saving as a `json` file.

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
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Prints the results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output <code>JSON</code> data for better readability, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-<code>ASCII</code> characters as <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Saves the results as a JSON format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save, naming consistent with the input file type when it is a directory</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specifies the indentation level to beautify the output <code>JSON</code> data for better readability, effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Controls whether to escape non-<code>ASCII</code> characters as <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Saves the results as an image format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path to save, supporting both directory or file path</td>
<td>None</td>
</tr>
</table>


- Calling the `print()` method will output the results to the terminal. The content printed to the terminal is explained as follows:

    - `input_path`: `(str)` The input path of the image to be predicted.

    - `model_settings`: `(Dict[str, bool])` Model parameters required for configuring the pipeline.

        - `use_doc_orientation_classify`: `(bool)` Controls whether to enable the document orientation classification module.
        - `use_doc_unwarping`: `(bool)` Controls whether to enable the document unwarping module.

    - `angle`: `(int)` The prediction result of the document orientation classification. When enabled, the values are [0, 90, 180, 270]; when not enabled, it is -1.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the path will be `save_path/{your_img_basename}.json`; if a file is specified, it will be saved directly to that file. Since JSON files do not support saving NumPy arrays, any `numpy.array` types will be converted to lists.

- Calling the `save_to_img()` method will save the visualized results to the specified `save_path`. If a directory is specified, the path will be `save_path/{your_img_basename}_doc_preprocessor_res_img.{your_img_extension}`; if a file is specified, it will be saved directly to that file. (Since the pipeline typically includes multiple result images, it is not recommended to specify a specific file path directly, as multiple images may be overwritten, leaving only the last image.)

* Additionally, it is also possible to obtain visualized images with results and prediction outcomes through attributes, as detailed below:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Retrieves the prediction results in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Retrieves visualized images in <code>dict</code> format</td>
</tr>
</table>


- The `json` attribute retrieves prediction results as a dictionary type of data, consistent with the content saved by calling the `save_to_json()` method.
- The `img` attribute returns prediction results as a dictionary type of data. Here, the key is `preprocessed_img`, and the corresponding value is an `Image.Image` object, which is a visualized image used to display the results of the `doc_preprocessor`.

Additionally, you can obtain the `doc_preprocessor` pipeline configuration file and load it for prediction. You can execute the following command to save the results in `my_path`:

```
paddlex --get_pipeline_config doc_preprocessor --save_path ./my_path
```

Once you have the configuration file, you can customize the various configurations of the `doc_preprocessor` pipeline by simply changing the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file. An example is as follows:

例如，若您的配置文件保存在 `./my_path/doc_preprocessor.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/doc_preprocessor.yaml")
output = pipeline.predict(
    input="doc_test_rotated.jpg"
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

<b>Note:</b> The parameters in the configuration file are for pipeline initialization. If you wish to modify the initialization parameters for the `doc_preprocessor` pipeline, you can directly edit the parameters in the configuration file and load the file for prediction. Additionally, CLI prediction also supports passing in a configuration file; simply specify the path to the configuration file using `--pipeline`.

## 3. Development Integration/Deployment

If the document image preprocessing pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the document image preprocessing pipeline directly to your Python project, you can refer to the sample code in [2.2 Python Script Method](#22-python-script-method-integration).

Additionally, PaddleX offers three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In real production environments, many applications have stringent performance standards for deployment strategies, especially regarding response speed, to ensure efficient system operation and a smooth user experience. To address this, PaddleX provides a high-performance inference plugin designed to deeply optimize model inference and pre/post-processing, resulting in significant end-to-end process acceleration. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
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
<td>The result of the operation.</td>
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
<p>Obtain the document image preprocessing results.</p>
<p><code>POST /document-preprocessing</code></p>
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
<td><code>file</code></td>
<td><code>string</code></td>
<td>The URL of an image or PDF file accessible by the server, or the Base64-encoded content of the file. By default, for PDF files exceeding 10 pages, only the first 10 pages will be processed.<br />
To remove the page limit, please add the following configuration to the pipeline configuration file:
<pre><code>Serving:
  extra:
    max_num_input_imgs: null
</code></pre></td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>The type of the file. <code>0</code> for PDF files, <code>1</code> for image files. If this attribute is missing, the file type will be inferred from the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_orientation_classify</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_unwarping</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>visualize</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>
Whether to return the final visualization image and intermediate images during the processing.<br/>
<ul style="margin: 0 0 0 1em; padding-left: 0em;">
<li>If <code>true</code> is provided: return images.</li>
<li>If <code>false</code> is provided: do not return any images.</li>
<li>If this parameter is omitted from the request body, or if <code>null</code> is explicitly passed, the behavior will follow the value of <code>Serving.visualize</code> in the pipeline configuration.</li>
</ul>
<br/>
For example, adding the following setting to the pipeline config file:<br/>
<pre><code>Serving:
  visualize: False
</code></pre>
will disable image return by default. This behavior can be overridden by explicitly setting the <code>visualize</code> parameter in the request.<br/>
If neither the request body nor the configuration file is set (If <code>visualize</code> is set to <code>null</code> in the request and  not defined in the configuration file), the image is returned by default.
</td>
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
<td><code>docPreprocessingResults</code></td>
<td><code>object</code></td>
<td>Document image preprocessing results. The array length is 1 (for image input) or the actual number of document pages processed (for PDF input). For PDF input, each element in the array represents the result of each page actually processed in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>docPreprocessingResults</code> is an <code>object</code> with the following attributes:</p>
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
<td><code>outputImage</code></td>
<td><code>string</code></td>
<td>The preprocessed image. The image is in PNG format and is Base64-encoded.</td>
</tr>
<tr>
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>A simplified version of the <code>res</code> field in the JSON representation of the result generated by the pipeline object's <code>predict</code> method, excluding the <code>input_path</code> and the <code>page_index</code> fields.</td>
</tr>
<tr>
<td><code>docPreprocessingImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>The visualization result image. The image is in JPEG format and is Base64-encoded.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>The input image. The image is in JPEG format and is Base64-encoded.</td>
</tr>
</tbody>
</table>
</details>
<details><summary>Multi-language Service Call Example</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/document-preprocessing"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["docPreprocessingResults"]):
    print(res["prunedResult"])
    output_img_path = f"out_{i}.png"
    with open(output_img_path, "wb") as f:
        f.write(base64.b64decode(res["outputImage"]))
    print(f"Output image saved at {output_img_path}")
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &lt;fstream&gt;
#include &lt;vector&gt;
#include &lt;string&gt;
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {

    httplib::Client client("localhost", 8080);
    const std::string filePath = "./demo.jpg";
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return 1;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }

    std::string bufferStr(buffer.data(), static_cast<size_t>(size));
    std::string encodedFile = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["file"] = encodedFile;
    jsonObj["fileType"] = 1;

    auto response = client.Post("/document-preprocessing", jsonObj.dump(), "application/json");

    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        if (!result.is_object() || !result["docPreprocessingResults"].is_array()) {
            std::cerr << "Unexpected response format." << std::endl;
            return 1;
        }

        for (size_t i = 0; i < result["docPreprocessingResults"].size(); ++i) {
            auto res = result["docPreprocessingResults"][i];

            if (res.contains("prunedResult")) {
                std::cout << "Preprocessed result: " << res["prunedResult"].dump() << std::endl;
            }

            if (res.contains("outputImage")) {
                std::string outputImgPath = "out_" + std::to_string(i) + ".png";
                std::string decodedImage = base64::from_base64(res["outputImage"].get<std::string>());

                std::ofstream outFile(outputImgPath, std::ios::binary);
                if (outFile.is_open()) {
                    outFile.write(decodedImage.c_str(), decodedImage.size());
                    outFile.close();
                    std::cout << "Saved image: " << outputImgPath << std::endl;
                } else {
                    std::cerr << "Failed to write image: " << outputImgPath << std::endl;
                }
            }
        }
    } else {
        std::cerr << "Request failed." << std::endl;
        if (response) {
            std::cerr << "HTTP status: " << response->status << std::endl;
            std::cerr << "Response body: " << response->body << std::endl;
        }
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
        String API_URL = "http://localhost:8080/document-preprocessing";
        String imagePath = "./demo.jpg";

        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String base64Image = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("file", base64Image);
        payload.put("fileType", 1);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.create(JSON, payload.toString());

        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode root = objectMapper.readTree(responseBody);
                JsonNode result = root.get("result");

                JsonNode docPreprocessingResults = result.get("docPreprocessingResults");
                for (int i = 0; i < docPreprocessingResults.size(); i++) {
                    JsonNode item = docPreprocessingResults.get(i);
                    int finalI = i;

                    JsonNode prunedResult = item.get("prunedResult");
                    System.out.println("Pruned Result [" + i + "]: " + prunedResult.toString());

                    String outputImgBase64 = item.get("outputImage").asText();
                    byte[] outputImgBytes = Base64.getDecoder().decode(outputImgBase64);
                    String outputImgPath = "out_" + finalI + ".png";
                    try (FileOutputStream fos = new FileOutputStream(outputImgPath)) {
                        fos.write(outputImgBytes);
                        System.out.println("Saved output image: " + outputImgPath);
                    }

                    JsonNode inputImageNode = item.get("inputImage");
                    if (inputImageNode != null && !inputImageNode.isNull()) {
                        String inputImageBase64 = inputImageNode.asText();
                        byte[] inputImageBytes = Base64.getDecoder().decode(inputImageBase64);
                        String inputImgPath = "inputImage_" + i + ".jpg";
                        try (FileOutputStream fos = new FileOutputStream(inputImgPath)) {
                            fos.write(inputImageBytes);
                            System.out.println("Saved input image to: " + inputImgPath);
                        }
                    }
                }
            } else {
                System.err.println("Request failed with HTTP code: " + response.code());
            }
        }
    }
}
</code></pre></details>

<details><summary>Go</summary>

<pre><code class="language-go">package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
    "os"
)

func main() {
    API_URL := "http://localhost:8080/document-preprocessing"
    filePath := "./demo.jpg"

    fileBytes, err := ioutil.ReadFile(filePath)
    if err != nil {
        fmt.Printf("Error reading file: %v\n", err)
        return
    }
    fileData := base64.StdEncoding.EncodeToString(fileBytes)

    payload := map[string]interface{}{
        "file":     fileData,
        "fileType": 1,
    }
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Printf("Error marshaling payload: %v\n", err)
        return
    }

    client := &http.Client{}
    req, err := http.NewRequest("POST", API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Printf("Error creating request: %v\n", err)
        return
    }
    req.Header.Set("Content-Type", "application/json")

    res, err := client.Do(req)
    if err != nil {
        fmt.Printf("Error sending request: %v\n", err)
        return
    }
    defer res.Body.Close()

    if res.StatusCode != http.StatusOK {
        fmt.Printf("Unexpected status code: %d\n", res.StatusCode)
        return
    }

    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Printf("Error reading response body: %v\n", err)
        return
    }

    type DocPreprocessingResult struct {
        PrunedResult         map[string]interface{} `json:"prunedResult"`
        OutputImage          string                 `json:"outputImage"`
        DocPreprocessingImage *string               `json:"docPreprocessingImage"`
        InputImage           *string                `json:"inputImage"`
    }

    type Response struct {
        Result struct {
            DocPreprocessingResults []DocPreprocessingResult `json:"docPreprocessingResults"`
            DataInfo                interface{}              `json:"dataInfo"`
        } `json:"result"`
    }

    var respData Response
    if err := json.Unmarshal(body, &respData); err != nil {
        fmt.Printf("Error unmarshaling response: %v\n", err)
        return
    }

    for i, res := range respData.Result.DocPreprocessingResults {
        fmt.Printf("Result %d - prunedResult: %+v\n", i, res.PrunedResult)

        imgBytes, err := base64.StdEncoding.DecodeString(res.OutputImage)
        if err != nil {
            fmt.Printf("Error decoding outputImage at index %d: %v\n", i, err)
            continue
        }

        filename := fmt.Sprintf("out_%d.png", i)
        if err := os.WriteFile(filename, imgBytes, 0644); err != nil {
            fmt.Printf("Error saving image %s: %v\n", filename, err)
            continue
        }
        fmt.Printf("Saved output image to %s\n", filename)
    }
}
</code></pre></details>

<details><summary>C#</summary>

<pre><code class="language-csharp">using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = "http://localhost:8080/document-preprocessing";
    static readonly string inputFilePath = "./demo.jpg";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] fileBytes = File.ReadAllBytes(inputFilePath);
        string fileData = Convert.ToBase64String(fileBytes);

        var payload = new JObject
        {
            { "file", fileData },
            { "fileType", 1 }
        };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        JArray docPreResults = (JArray)jsonResponse["result"]["docPreprocessingResults"];
        for (int i = 0; i < docPreResults.Count; i++)
        {
            var res = docPreResults[i];
            Console.WriteLine($"[{i}] prunedResult:\n{res["prunedResult"]}");

            string base64Image = res["outputImage"]?.ToString();
            if (!string.IsNullOrEmpty(base64Image))
            {
                string outputPath = $"out_{i}.png";
                byte[] imageBytes = Convert.FromBase64String(base64Image);
                File.WriteAllBytes(outputPath, imageBytes);
                Console.WriteLine($"Output image saved at {outputPath}");
            }
            else
            {
                Console.WriteLine($"outputImage at index {i} is null.");
            }
        }
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');
const path = require('path');

const API_URL = 'http://localhost:8080/document-preprocessing';
const imagePath = './demo.jpg';

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

const payload = {
  file: encodeImageToBase64(imagePath),
  fileType: 1
};

axios.post(API_URL, payload, {
  headers: {
    'Content-Type': 'application/json'
  },
  maxBodyLength: Infinity
})
.then((response) => {
  const results = response.data.result.docPreprocessingResults;

  results.forEach((res, index) => {
    console.log(`\n[${index}] prunedResult:`);
    console.log(res.prunedResult);

    const base64Image = res.outputImage;
    if (base64Image) {
      const outputImagePath = `out_${index}.png`;
      const imageBuffer = Buffer.from(base64Image, 'base64');
      fs.writeFileSync(outputImagePath, imageBuffer);
      console.log(`Output image saved at ${outputImagePath}`);
    } else {
      console.log(`outputImage at index ${index} is null.`);
    }
  });
})
.catch((error) => {
  console.error('API error:', error.message);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/document-preprocessing";
$image_path = "./demo.jpg";
$output_image_path = "./out_0.png";

$image_data = base64_encode(file_get_contents($image_path));
$payload = array("file" => $image_data, "fileType" => 1);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)["result"]["docPreprocessingResults"];

foreach ($result as $i => $item) {
    echo "[$i] prunedResult:\n";
    print_r($item["prunedResult"]);

    if (!empty($item["outputImage"])) {
        $output_image_path = "out_" . $i . ".png";
        file_put_contents($output_image_path, base64_decode($item["outputImage"]));
        echo "Output image saved at $output_image_path\n";
    } else {
        echo "No outputImage found for item $i\n";
    }
}
?&gt;
</code></pre></details>
</details>
<br/>

☁️ <b>Service Deployment</b>: Service deployment is a common form of deployment in real production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports multiple pipeline service deployment solutions. For detailed pipeline service deployment procedures, please refer to the [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.en.md).




## 4. Custom Development

If the default model weights provided by the document image preprocessing pipeline do not meet your accuracy or speed requirements in your specific scenario, you can try to further fine-tune the existing model using data from your specific domain or application scenario to enhance the recognition performance of the document image preprocessing pipeline in your context.

### 4.1 Model Fine-Tuning

Since the document image preprocessing pipeline consists of several modules, if the pipeline's performance does not meet expectations, it may be due to any one of these modules. You can analyze the images with poor recognition results to identify which module has issues, and then refer to the corresponding fine-tuning tutorial link in the table below to fine-tune the model.


<table>
  <thead>
    <tr>
      <th>situation</th>
      <th>Fine-tuning model </th>
      <th>Fine-tuning reference link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>The overall image rotation correction is inaccurate.</td>
      <td>Image orientation classification module</td>
      <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html">Link</a></td>
    </tr>
    <tr>
      <td>The image distortion correction is inaccurate.</td>
      <td>Image Unwarping</td>
      <td>Fine-tuning is not supported at the moment.</td>
    </tr>
  </tbody>
</table>

### 4.2 Model Application

After completing fine-tuning training with a private dataset, you can obtain a local model weights file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by entering the local path of the fine-tuned model weights into the `model_dir` field in the pipeline configuration file.

```yaml
......
  DocOrientationClassify:
    module_name: doc_text_orientation
    model_name: PP-LCNet_x1_0_doc_ori
    model_dir: ./output/best_model/inference  # Replace it with the path of the fine-tuned document image orientation classification model weights.
......
```

Then, refer to the command line method or Python script method in [2. Quick Start](#2-quick-start) to load the modified pipeline configuration file.

## 5. Multi-Hardware Support

PaddleX supports a variety of mainstream hardware devices such as NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. You can achieve seamless switching between different hardware by simply modifying the `--device` parameter.

For example, if you are using an Ascend NPU for inference in a document image preprocessing pipeline, the Python command you would use is:

```bash
paddlex --pipeline doc_preprocessor \
        --input doc_test_rotated.jpg \
        --use_doc_orientation_classify True \
        --use_doc_unwarping True \
        --save_path ./output \
        --device npu:0
```

If you want to use the document image preprocessing pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
