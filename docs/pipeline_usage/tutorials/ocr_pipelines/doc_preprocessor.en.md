---
comments: true
---

# Document Image Preprocessing Pipeline Tutorial

## 1. Introduction to the Do Pipeline

The document image preprocessing pipeline integrates two major functions: document orientation classification and geometric distortion correction. The document orientation classification can automatically identify the four orientations of a document (0°, 90°, 180°, 270°) to ensure that the document is processed in the correct direction for subsequent tasks. The geometric distortion correction model is used to correct geometric distortions that occur during the document's photographing or scanning process, restoring the document to its original shape and proportions. This is suitable for digital document management, preprocessing for doc_preprocessor recognition, and any scenario where improving document image quality is necessary. Through automated orientation correction and distortion correction, this module significantly enhances the accuracy and efficiency of document processing, providing users with a more reliable foundation for image analysis. The pipeline also offers flexible service deployment options, supporting invocation using various programming languages on multiple hardware platforms. Moreover, it provides the capability for further development, allowing you to train and fine-tune on your own dataset based on this pipeline, with the trained models being seamlessly integrable.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/doc_preprocessor/02.jpg">

**The general document image preprocessing pipeline includes optional document image orientation classification module and document image correction module** with the following models included.

<p><b>Document Image Orientation Classification Module (Optional):</b></p>

<table>
<thead>
<tr>
<th>Model</th><th>Model download link</th>
<th>Top-1 Acc（%）</th>
<th>GPU inference time (ms)</th>
<th>CPU inference time (ms)</th>
<th>Model storage size（M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_doc_ori_infer.tar">                Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Train Model</a></td>
<td>99.06</td>
<td>3.84845</td>
<td>9.23735</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, containing four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees.</td>
</tr>
</tbody>
</table>
<b>Note: The accuracy metrics mentioned above are evaluated on a self-constructed dataset, which covers various scenarios such as IDs and documents, including 1000 images. The GPU inference time is based on an NVIDIA Tesla T4 machine with a precision type of FP32. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and a precision type of FP32.</b>

<p><b>Text Image Unwarping Module (Optional)：</b></p>

<table>
<thead>
<tr>
<th>Model</th><th>Model download link</th>
<th>CER </th>
<th>Model storage size（M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Train Model</a></td>
<td>0.179</td>
<td>30.3 M</td>
<td>High-Precision Text Image Correction Model</td>
</tr>
</tbody>
</table>
<b>The accuracy metrics of the model are measured from <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet benchmark</a> </b>



## 2. Quick Start

PaddleX supports experiencing the effects of the document image preprocessing pipeline locally via command line or Python.

Before using the document image preprocessing pipeline locally, please ensure you have completed the installation of the PaddleX wheel package according to the [PaddleX Local Installation Guide](../../../installation/installation.md).

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
You can refer to the parameter descriptions in [2.1.2 Python Script Integration](#212-python-script-integration) for related parameter details.

After running, the results will be printed to the terminal as follows:

<pre><code>{'res': {'input_path': 'doc_test_rotated.jpg', 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 180}}
</code></pre>

You can refer to the results explanation in [2.1.2 Python Script Integration](#212-python-script-integration) for a description of the output parameters.

The visualized results are saved under `save_path`. The visualized results are as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/doc_preprocessor/02.jpg">


#### 2.1.2 Python Script Integration

The above command line is for quickly experiencing and viewing the effect. Generally, in a project, it is often necessary to integrate through code. You can complete quick inference in a production line with just a few lines of code. The inference code is as follows:

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
<td>Inference device for the pipeline. Supports specifying the GPU card number, such as "gpu:0", other hardware card numbers, such as "npu:0", and CPU as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, available only when the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
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
<td><code>device</code></td>
<td>Inference device for the pipeline</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>CPU</b>: Like <code>cpu</code>, indicating inference using CPU;</li>
  <li><b>GPU</b>: Like <code>gpu:0</code>, indicating inference using the first GPU;</li>
  <li><b>NPU</b>: Like <code>npu:0</code>, indicating inference using the first NPU;</li>
  <li><b>XPU</b>: Like <code>xpu:0</code>, indicating inference using the first XPU;</li>
  <li><b>MLU</b>: Like <code>mlu:0</code>, indicating inference using the first MLU;</li>
  <li><b>DCU</b>: Like <code>dcu:0</code>, indicating inference using the first DCU;</li>
  <li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used. During initialization, it will preferentially use the local GPU device 0, if none, then the CPU device;</li>
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
      <td><a href="../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.en.md">链接</a></td>
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
