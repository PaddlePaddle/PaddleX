---
comments: true
---

# Text Recognition Module Development Tutorial

## I. Overview
The text recognition module is the core component of an OCR (Optical Character Recognition) system, responsible for extracting text information from text regions within images. The performance of this module directly impacts the accuracy and efficiency of the entire OCR system. The text recognition module typically receives bounding boxes of text regions output by the text detection module as input. Through complex image processing and deep learning algorithms, it converts the text in images into editable and searchable electronic text. The accuracy of text recognition results is crucial for subsequent applications such as information extraction and data mining.

## II. Supported Model List

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td>6.65 / 6.65</td>
<td>32.92 / 32.92</td>
<td></td>
<td>PP-OCRv4_server_rec_doc is trained on a mixed dataset of more Chinese document data and PP-OCR training data based on PP-OCRv4_server_rec. It has added the ability to recognize some traditional Chinese characters, Japanese, and special characters, and can support the recognition of more than 15,000 characters. In addition to improving the text recognition capability related to documents, it also enhances the general text recognition capability.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>78.20</td>
<td>4.82 / 4.82</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td>The PP-OCRv4 recognition model is further upgraded based on PP-OCRv3. Under comparable speed conditions, the effect in Chinese and English scenarios is further improved, and the average recognition accuracy of the 80-language multilingual model is increased by more than 8%.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Training Model</a></td>
<td>79.20</td>
<td>6.58 / 6.58</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
<td>A high-precision server-side text recognition model, featuring high accuracy, fast speed, and multilingual support. It is suitable for text recognition tasks in various scenarios.</td>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td>4.81 / 4.81</td>
<td>16.10 / 5.31</td>
<td></td>
<td>The ultra-lightweight English text recognition model released by PaddleOCR in May 2023. It is small in size and fast in speed, and can achieve millisecond-level prediction on CPU. Compared with the PP-OCRv3 English model, the recognition accuracy is improved by 6%, and it is suitable for text recognition tasks in various scenarios.</td>
</tr>
</table>
<b>Note: The evaluation set for the above accuracy indicators is the Chinese dataset built by PaddleOCR, covering multiple scenarios such as street view, web images, documents, and handwriting, with 11,000 images included in text recognition. All models' GPU inference time is based on NVIDIA Tesla T4 machine, with precision type of FP32. CPU inference speed is based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz, with 8 threads and precision type of FP32.</b>

&gt; ❗ The above list features the <b>4 core models</b> that the text recognition module primarily supports. In total, this module supports <b>18 models</b>. The complete list of models is as follows:

<details><summary> 👉Model List Details</summary>

* <b>Chinese Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td>6.65 / 6.65</td>
<td>32.92 / 32.92</td>
<td></td>
<td>PP-OCRv4_server_rec_doc is trained on a mixed dataset of more Chinese document data and PP-OCR training data based on PP-OCRv4_server_rec. It has added the recognition capabilities for some traditional Chinese characters, Japanese, and special characters. The number of recognizable characters is over 15,000. In addition to the improvement in document-related text recognition, it also enhances the general text recognition capability.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Training Model</a></td>
<td>78.20</td>
<td>4.82 / 4.82</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td>The PP-OCRv4 recognition model is an upgrade from PP-OCRv3. Under comparable speed conditions, the effect in Chinese and English scenarios is further improved. The average recognition accuracy of the 80 multilingual models is increased by more than 8%.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Trained Model</a></td>
<td>79.20</td>
<td>6.58 / 6.58</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
<td>A high-precision server text recognition model, featuring high accuracy, fast speed, and multilingual support. It is suitable for text recognition tasks in various scenarios.</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td>5.87 / 5.87</td>
<td>9.07 / 4.28</td>
<td></td>
<td>An ultra-lightweight OCR model suitable for mobile applications. It adopts an encoder-decoder structure based on Transformer and enhances recognition accuracy and efficiency through techniques such as data augmentation and mixed precision training. The model size is 10.6M, making it suitable for deployment on resource-constrained devices. It can be used in scenarios such as mobile photo translation and business card recognition.</td>
</tr>
</table>
<p><b>Note: The evaluation set for the above accuracy indicators is the Chinese dataset built by PaddleOCR, covering multiple scenarios such as street view, web images, documents, and handwriting. The text recognition includes 11,000 images. The GPU inference time for all models is based on NVIDIA Tesla T4 machines with FP32 precision type. The CPU inference speed is based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision type.</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Training Model</a></td>
<td>68.81</td>
<td>8.36801</td>
<td>165.706</td>
<td>73.9 M</td>
<td rowspan="1">
SVTRv2 is a server text recognition model developed by the OpenOCR team of Fudan University's Visual and Learning Laboratory (FVL). It won the first prize in the PaddleOCR Algorithm Model Challenge - Task One: OCR End-to-End Recognition Task. The end-to-end recognition accuracy on the A list is 6% higher than that of PP-OCRv4.
</td>
</tr>
</table>
<p><b>Note: The evaluation set for the above accuracy indicators is the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a> - Task One: OCR End-to-End Recognition Task A list. The GPU inference time for all models is based on NVIDIA Tesla T4 machines with FP32 precision type. The CPU inference speed is based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision type.</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Training Model</a></td>
<td>65.07</td>
<td>10.5047</td>
<td>51.5647</td>
<td>22.1 M</td>
<td rowspan="1">    The RepSVTR text recognition model is a mobile text recognition model based on SVTRv2. It won the first prize in the PaddleOCR Algorithm Model Challenge - Task One: OCR End-to-End Recognition Task. The end-to-end recognition accuracy on the B list is 2.5% higher than that of PP-OCRv4, with the same inference speed.</td>
</tr>
</table>
<p><b>Note: The evaluation set for the above accuracy indicators is the <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Model Challenge</a> - Task One: OCR End-to-End Recognition Task B list. The GPU inference time for all models is based on NVIDIA Tesla T4 machines with FP32 precision type. The CPU inference speed is based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision type.</b></p>

* <b>English Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>[Latest] Further upgraded based on PP-OCRv3, with improved accuracy under comparable speed conditions.</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/en_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Ultra-lightweight model, supporting English and numeric recognition.</td>
</tr>
</table>

* <b>Multilingual Recognition Model</b>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy(%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/korean_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Korean Recognition</td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/japan_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Japanese Recognition</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Traditional Chinese Recognition</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/te_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Telugu Recognition</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ka_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Kannada Recognition</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ta_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Tamil Recognition</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/latin_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Latin Recognition</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/arabic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Arabic Script Recognition</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Cyrillic Script Recognition</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="">Training Model</a></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Devanagari Script Recognition</td>
</tr>
</table>
</details>

## III. Quick Integration
Before quick integration, you need to install the PaddleX wheel package. For the installation method, please refer to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md). After installing the wheel package, a few lines of code can complete the inference of the text recognition module. You can switch models under this module freely, and you can also integrate the model inference of the text recognition module into your project.

Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png) to your local machine.

```python
from paddlex import create_model
model = create_model("PP-OCRv4_mobile_rec")
output = model.predict("general_ocr_rec_001.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference APIs, please refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

After running, the result obtained is:
```bash
{'res': {'input_path': 'general_ocr_rec_001.png', 'page_index': None, 'rec_text': '绿洲仕格维花园公寓', 'rec_score': 0.9875497817993164}}
````
The meanings of the running results parameters are as follows:
- `input_path`：Represents the path to the image of the text line to be predicted.
- `page_index`：If the input is a PDF file, this indicates the current page number of the PDF. Otherwise, it is `None`
- `rec_text`：Represents the predicted text of the text line image.
- `rec_score`：Represents the confidence score of the predicted text line image.

The visualized image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/text_recog/general_ocr_rec_001.png"/>

In the above Python script, the following steps are executed:
* `create_model` instantiates the text recognition model (here, `PP-OCRv4_mobile_rec` is taken as an example)
* The `predict` method of the text recognition model is called for inference prediction. The parameter of the `predict` method is `x`, which is used to input the data to be predicted. It supports multiple input types, and the specific instructions are as follows:

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
<td><code>x</code></td>
<td>Data to be predicted, supporting multiple input types</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
<li><b>Python variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
<li><b>File path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
<li><b>URL link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png">Example</a></li>
<li><b>Local directory</b>, this directory should contain the data files to be predicted, such as the local path: <code>/root/data/</code></li>
<li><b>List</b>, the elements of the list should be the above types of data, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>, <code>[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code></li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>module_name</code></td>
<td>Name of the single-function module</td>
<td><code>str</code></td>
<td>None</td>
<td><code>text_recognition</code></td>
</tr>
<tr>
<td><code>model_name</code></td>
<td>Name of the model</td>
<td><code>str</code></td>
<td>None</td>
<td><code>PP-OCRv4_mobile_rec</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Path where the model is stored</td>
<td><code>str</code></td>
<td>None</td>
<td><code>null</code></td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>None</td>
<td>1</td>
</tr>
<tr>
<td><code>score_thresh</code></td>
<td>Score threshold</td>
<td><code>int</code></td>
<td>None</td>
<td><code>0</code></td>
</tr>
</table>

* Process the prediction results. The prediction result for each sample is of `dict` type, and supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td rowspan="3"><code>print</code></td>
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>json</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>JSON formatting setting, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>JSON formatting setting, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json</code></td>
<td rowspan="3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path where the file is saved. If it is a directory, the saved file name is consistent with the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>JSON formatting setting</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>JSON formatting setting</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path where the file is saved. If it is a directory, the saved file name is consistent with the input file name</td>
<td>None</td>
</tr>
</table>


## IV. Custom Development

If you are seeking higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better  text recognition models. Before using PaddleX to develop text recognition models, please ensure that you have installed the relevant model training plugins for OCR in PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, it is necessary to prepare the corresponding dataset for each task module. PaddleX provides a data validation function for each module, and <b>only data that passes the validation can be used for model training</b>. Additionally, PaddleX offers Demo datasets for each module, allowing you to complete subsequent development based on the officially provided Demo data. If you wish to use a private dataset for subsequent model training, you can refer to the [PaddleX Text Detection/Text Recognition Task Module Data Annotation Tutorial](../../../data_annotations/ocr_modules/text_detection_recognition.en.md).

#### 4.1.1 Download Demo Data
You can use the following commands to download the Demo dataset to a specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ocr_rec_dataset_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples
```
After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.

<details><summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>
<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 4468,
    "train_sample_paths": [
      "check_dataset\/demo_img\/train_word_1.png",
      "check_dataset\/demo_img\/train_word_2.png",
      "check_dataset\/demo_img\/train_word_3.png",
      "check_dataset\/demo_img\/train_word_4.png",
      "check_dataset\/demo_img\/train_word_5.png",
      "check_dataset\/demo_img\/train_word_6.png",
      "check_dataset\/demo_img\/train_word_7.png",
      "check_dataset\/demo_img\/train_word_8.png",
      "check_dataset\/demo_img\/train_word_9.png",
      "check_dataset\/demo_img\/train_word_10.png"
    ],
    "val_samples": 2077,
    "val_sample_paths": [
      "check_dataset\/demo_img\/val_word_1.png",
      "check_dataset\/demo_img\/val_word_2.png",
      "check_dataset\/demo_img\/val_word_3.png",
      "check_dataset\/demo_img\/val_word_4.png",
      "check_dataset\/demo_img\/val_word_5.png",
      "check_dataset\/demo_img\/val_word_6.png",
      "check_dataset\/demo_img\/val_word_7.png",
      "check_dataset\/demo_img\/val_word_8.png",
      "check_dataset\/demo_img\/val_word_9.png",
      "check_dataset\/demo_img\/val_word_10.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "ocr_rec_dataset_examples",
  "show_type": "image",
  "dataset_type": "MSTextRecDataset"
}
</code></pre>
<p>In the above validation result, <code>check_pass</code> being <code>true</code> indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.train_samples</code>: The number of training set samples in this dataset is 4468;</li>
<li><code>attributes.val_samples</code>: The number of validation set samples in this dataset is 2077;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to the visualized training set samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the visualized validation set samples in this dataset;
Additionally, the dataset validation also analyzes the distribution of character length ratios in the dataset and generates a distribution histogram (histogram.png):</li>
</ul>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/text_recog/01.png"/></p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)


After completing data validation, you can convert the dataset format or re-split the training/validation ratio of the dataset by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Dataset Format Conversion/Dataset Splitting Details (Click to Expand)</b></summary>
<p><b>(1) Dataset Format Conversion</b></p>
<p>Text recognition does not currently support data conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>The parameters for dataset splitting can be set by modifying the <code>CheckDataset</code> section in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. Set to <code>True</code> to enable dataset splitting, default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, set the percentage of the training set. The type is any integer between 0-100, and it must sum up to 100 with <code>val_percent</code>;
For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, modify the configuration file as follows:</li>
</ul>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples
</code></pre>
<p>After data splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support setting through appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
Model training can be completed with a single command. Here's an example of training the PP-OCRv4 mobile text recognition model (PP-OCRv4_mobile_rec):

```bash
python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples
```
The steps required are:

* Specify the path to the model's `.yaml` configuration file (here it's `PP-OCRv4_mobile_rec.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`.
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file or adjusted by appending parameters in the command line. For example, to specify training on the first 2 GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common.en.md).


<details><summary>👉 <b>More Information (Click to Expand)</b></summary>
<ul>
<li>During model training, PaddleX automatically saves the model weight files, with the default being <code>output</code>. If you need to specify a save path, you can set it through the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.</li>
<li>
<p>After completing the model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including:</p>
</li>
<li>
<p><code>train_result.json</code>: Training result record file, recording whether the training task was completed normally, as well as the output weight metrics, related file paths, etc.;</p>
</li>
<li><code>train.log</code>: Training log file, recording changes in model metrics and loss during training;</li>
<li><code>config.yaml</code>: Training configuration file, recording the hyperparameter configuration for this training session;</li>
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;</li>
</ul></details>

## <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash

```bash
python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples

```
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it's `PP-OCRv4_mobile_rec.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).


<details><summary>👉 <b>More Information (Click to Expand)</b></summary>
<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path. If you need to change it, simply append the command line parameter to set it, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully and the model's evaluation metrics, including  acc、norm_edit_dis；</p></details>

### <b>4.4 Model Inference and Model Integration</b>
After completing model training and evaluation, you can use the trained model weights for inference prediction or Python integration.

#### 4.4.1 Model Inference
To perform inference prediction via the command line, simply use the following command:

Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png) to your local machine.


```bash
python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_accuracy/inference" \
    -o Predict.input="general_ocr_rec_001.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-OCRv4_mobile_rec.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_accuracy/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
Models can be directly integrated into the PaddleX pipelines or into your own projects.

1.<b>Pipeline Integration</b>

The text recognition module can be integrated into PaddleX pipelines such as the [General OCR Pipeline](../../../pipeline_usage/tutorials/ocr_pipelines/OCR.en.md), [General Table Recognition Pipeline](../../../pipeline_usage/tutorials/ocr_pipelines/table_recognition.en.md), and [Document Scene Information Extraction Pipeline v3 (PP-ChatOCRv3)](../../../pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.en.md). Simply replace the model path to update the text recognition module of the relevant pipeline.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the text recognition module. Refer to the [Quick Integration](#iii-quick-integration) Python example code. Simply replace the model with the path to your trained model.
