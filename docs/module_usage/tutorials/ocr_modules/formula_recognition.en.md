---
comments: true
---

# Formula Recognition Module Development Tutorial

## I. Overview
The formula recognition module is a crucial component of OCR (Optical Character Recognition) systems, responsible for converting mathematical formulas in images into editable text or computer-readable formats. The performance of this module directly impacts the accuracy and efficiency of the entire OCR system. The module typically outputs LaTeX or MathML codes of mathematical formulas, which are then passed on to the text understanding module for further processing.

## II. Supported Model List


<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Avg-BLEU(%)</th>
<th>GPU Inference Time (ms)</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<td>UniMERNet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/UniMERNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">Training Model</a></td>
<td>86.13</td>
<td>2266.96</td>
<td>1.4 G</td>
<td>UniMERNet is a formula recognition model developed by Shanghai AI Lab. It uses Donut Swin as the encoder and MBartDecoder as the decoder. The model is trained on a dataset of one million samples, including simple formulas, complex formulas, scanned formulas, and handwritten formulas, significantly improving the recognition accuracy of real-world formulas.</td>
<tr>
<td>PP-FormulaNet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-FormulaNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">Training Model</a></td>
<td>87.12</td>
<td>202.25</td>
<td>167.9 M</td>
<td rowspan="2">PP-FormulaNet is an advanced formula recognition model developed by the Baidu PaddlePaddle Vision Team. The PP-FormulaNet-S version uses PP-HGNetV2-B4 as its backbone network. Through parallel masking and model distillation techniques, it significantly improves inference speed while maintaining high recognition accuracy, making it suitable for applications requiring fast inference. The PP-FormulaNet-L version, on the other hand, uses Vary_VIT_B as its backbone network and is trained on a large-scale formula dataset, showing significant improvements in recognizing complex formulas compared to PP-FormulaNet-S.</td>
</tr>
<td>PP-FormulaNet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-FormulaNet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">Training Model</a></td>
<td>92.13</td>
<td>1976.52</td>
<td>535.2 M</td>
<tr>
<td>LaTeX_OCR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Training Model</a></td>
<td>71.63</td>
<td>-</td>
<td>89.7 M</td>
<td>LaTeX-OCR is a formula recognition algorithm based on an autoregressive large model. It uses Hybrid ViT as the backbone network and a transformer as the decoder, significantly improving the accuracy of formula recognition.</td>
</tr>
</table>



<b>Note: The above accuracy metrics are measured using an internally built formula recognition test set within PaddleX. The BLEU score of LaTeX_OCR_rec on the LaTeX-OCR formula recognition test set is 0.8821. All model GPU inference times are based on machines with Tesla V100 GPUs, with precision type FP32.</b>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For details, please refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After installing the wheel package, you can complete the inference of the formula recognition module with just a few lines of code. You can switch models under this module at will, and you can also integrate the model inference of the formula recognition module into your project. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png) to your local machine.

```python
from paddlex import create_model
model = create_model(model_name="PP-FormulaNet-S")
output = model.predict(input="general_formula_rec_001.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the result obtained is:

````
{'res': {'input_path': 'general_formula_rec_001.png', 'page_index': None, 'rec_formula': '\\zeta_{0}(\\nu)=-{\\frac{\\nu\\varrho^{-2\\nu}}{\\pi}}\\int_{\\mu}^{\\infty}d\\omega\\int_{C_{+}}d z{\\frac{2z^{2}}{(z^{2}+\\omega^{2})^{\\nu+1}}}\\ \\ {vec\\Psi}(\\omega;z)e^{i\\epsilon z}\\quad,'}}
````

The meanings of the running results parameters are as follows:
- `input_path`: Indicates the path to the input image of the formula to be predicted.
- `page_index`：If the input is a PDF file, this indicates the current page number of the PDF. Otherwise, it is `None`
- `rec_formula`: Indicates the predicted LaTeX source code of the formula image.

The visualization image is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/formula_recog/general_formula_rec_001_res.png">

<b>Note: If you need to visualize the formula recognition pipeline, you need to run the following commands to install the LaTeX rendering environment. Currently, the formula recognition visualization pipeline only supports the Ubuntu environment; other environments are not supported at this time. For complex formulas, the LaTeX results may include some advanced representations that might not be displayed successfully in environments like Markdown:</b>
```bash
sudo apt-get update
sudo apt-get install texlive texlive-latex-base texlive-latex-extra -y
````

The explanations for the methods, parameters, etc., are as follows:

* The `create_model` method instantiates the formula recognition model (here, `PP-FormulaNet-S` is used as an example), and the specific explanations are as follows:
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
<td>Name of the model</td>
<td><code>str</code></td>
<td>All model names supported by PaddleX</td>
<td>None</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Path to store the model</td>
<td><code>str</code></td>
<td>None</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the default model parameters built into PaddleX are used. If `model_dir` is specified, the user-defined model is used.

* The `predict()` method of the formula recognition model is called for inference prediction. The `predict()` method has parameters `input` and `batch_size`, which are explained as follows:

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
  <li><b>URL link</b>, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png">Example</a></li>
  <li><b>Local directory</b>, the directory should contain data files to be predicted, such as the local path: <code>/root/data/</code></li>
  <li><b>List</b>, elements of the list must be of the above types of data, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code>, <code>[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code></li>
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
</table>

* The prediction results are processed, and the prediction result for each sample is of type `dict`. It supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the results as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. If it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable, only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. If set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters, only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the results as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. If it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
</table>

* Additionally, it supports obtaining the visualization image with results and the prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
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
If you aim for higher accuracy with existing models, you can leverage PaddleX's custom development capabilities to develop better formula recognition models. Before developing formula recognition models with PaddleX, ensure you have installed the PaddleOCR-related model training plugins for PaddleX. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides a data validation function for each module, and <b>only data that passes the validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to the [LaTeX-OCR Formula Recognition Project](https://github.com/lukas-blecher/LaTeX-OCR).

#### 4.1.1 Demo Data Download
You can download the demo dataset to a specified folder using the following command:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_latexocr_dataset_example.tar -P ./dataset
tar -xf ./dataset/ocr_rec_latexocr_dataset_example.tar -C ./dataset/
```

#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/modules/formula_recognition/PP-FormulaNet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example
```
After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.
<details><summary>👉 <b>Details of Validation Results (Click to Expand)</b></summary>

<p>The specific content of the validation result file is:</p>

<pre><code class="language-bash">
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 10001,
    "train_sample_paths": [
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/train_0077809.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/train_0161600.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/train_0002077.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/train_0178425.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/train_0010959.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/train_0079266.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/train_0142495.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/train_0196376.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/train_0185513.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/train_0217146.png"
    ],
    "val_samples": 501,
    "val_sample_paths": [
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/val_0053264.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/val_0100521.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/val_0146333.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/val_0072788.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/val_0002022.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/val_0203664.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/val_0082217.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/val_0208199.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/val_0111236.png",
      "..\/dataset\/ocr_rec_latexocr_dataset_example\/images\/val_0204453.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "ocr_rec_latexocr_dataset_example",
  "show_type": "image",
  "dataset_type": "FormulaRecDataset"
}
</code></pre>
<p>In the above validation results, <code>check_pass</code> being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:
<ul>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 9452;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 1050;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to the visualized training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the visualized validation samples in this dataset;</li>
</ul>
<p>Additionally, the dataset verification also analyzes the distribution of sample numbers across all categories in the dataset and generates a distribution histogram (<code>histogram.png</code>):
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/formula_recognition/01.jpg"></p></details>


### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the data verification, you can convert the dataset format and re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details of Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>The formula recognition supports converting <code>FormulaRecDataset</code> format datasets to <code>LaTeXOCRDataset</code> format ( <code>PKL</code> format ). The parameters for dataset format conversion can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: Whether to perform dataset format conversion. Formula recognition supports converting <code>FormulaRecDataset</code> format datasets to <code>LaTeXOCRDataset</code> format, default is <code>True</code>;</li>
<li><code>src_dataset_type</code>: If dataset format conversion is performed, the source dataset format needs to be set, default is <code>FormulaRecDataset</code>;</li>
</ul>
<p>For example, if you want to convert a <code>FormulaRecDataset</code> format dataset to <code>LaTeXOCRDataset</code> format, you need to modify the configuration file as follows:</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: FormulaRecDataset
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/formula_recognition/PP-FormulaNet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example
</code></pre>
<p>After the data conversion is executed, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support being set by appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c  paddlex/configs/modules/formula_recognition/PP-FormulaNet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=FormulaRecDataset
</code></pre>
<p><b>(2) Dataset Splitting</b></p>
<p>The parameters for dataset splitting can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. When set to <code>True</code>, dataset splitting is performed, default is <code>False</code>;</li>
<li><code>train_percent</code>: If the dataset is re-split, the percentage of the training set needs to be set, which is an integer between 0 and 100, and the sum with <code>val_percent</code> should be 100;</li>
</ul>
<p>For example, if you want to re-split the dataset with 90% for the training set and 10% for the validation set, you need to modify the configuration file as follows:</p>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/formula_recognition/PP-FormulaNet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example
</code></pre>
<p>After the data splitting is executed, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support being set by appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c  paddlex/configs/modules/formula_recognition/PP-FormulaNet-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
Model training can be completed with a single command, taking the training of the formula recognition model PP-FormulaNet-S as an example:

```bash
FLAGS_json_format_model=1 python main.py -c paddlex/configs/modules/formula_recognition/PP-FormulaNet-S.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example
```
The following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-FormulaNet-S.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Set the mode to model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`.
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file instructions for the corresponding task module of the model [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common.en.md).
*  Except for LaTeX_OCR_rec, the formula recognition models only support exporting models in JSON format. Therefore, during training, you need to set the parameter `FLAGS_json_format_model=1`.
*  For the PP-FormulaNet-S, PP-FormulaNet-L, and UniMERNet models, additional Linux packages need to be installed during training. The specific command is as follows:

```bash
sudo apt-get update
sudo apt-get install libmagickwand-dev
python -m pip install Wand
```


<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

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
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/modules/formula_recognition/PP-FormulaNet-S.yaml  \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example
```
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-FormulaNet-S.yaml`)
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`.
Other related parameters can be set by modifying the `Global` and `Evaluate` fields in the `.yaml` configuration file, detailed instructions can be found in [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_accuracy/best_accuracy.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be produced, which records the evaluation results, specifically, whether the evaluation task was completed successfully and the model's evaluation metrics, including recall1、recall5、mAP；</p></details>


### <b>4.4 Model Inference and Integration</b>
After completing model training and evaluation, you can use the trained model weights for inference prediction or Python integration.


#### 4.4.1 Model Inference
To perform inference prediction through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png) to your local machine.
```bash
python main.py -c paddlex/configs/modules/formula_recognition/PP-FormulaNet-S.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_accuracy/inference" \
    -o Predict.input="general_formula_rec_001.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-FormulaNet-S.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_accuracy/inference"`
* Specify the input data path: `-o Predict.input="..."`.
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).


#### 4.4.2 Model Integration

The weights you produce can be directly integrated into the formula recognition module. Refer to the Python example code in [Quick Integration](#iii-quick-integration), and simply replace the model with the path to your trained model.
