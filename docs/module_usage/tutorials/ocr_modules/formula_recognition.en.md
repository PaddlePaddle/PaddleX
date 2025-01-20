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
<th>BLEU Score</th>
<th>Normed Edit Distance</th>
<th>ExpRate (%)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-FormulaNet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-FormulaNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">Trained Model</a></td>
<td>0.8821</td>
<td>0.0823</td>
<td>40.01</td>
<td>89.7 M</td>
<td>LaTeX-OCR is a formula recognition algorithm based on an autoregressive large model. By adopting Hybrid ViT as the backbone network and transformer as the decoder, it significantly improves the accuracy of formula recognition.</td>
</tr>
</table>

<b>Note: The above accuracy metrics are measured on the LaTeX-OCR formula recognition test set.</b>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

After installing the wheel package, a few lines of code can complete the inference of the formula recognition module. You can switch models under this module freely, and you can also integrate the model inference of the formula recognition module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png) to your local machine.

```bash
from paddlex import create_model
model = create_model("PP-FormulaNet-S")
output = model.predict("general_formula_rec_001.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_json("./output/res.json")
```
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
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;train_samples&quot;: 9452,
    &quot;train_sample_paths&quot;: [
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0109284.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0217434.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0166758.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0022294.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/val_0071799.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0017043.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0026204.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0209202.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/val_0157332.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0232582.png&quot;
    ],
    &quot;val_samples&quot;: 1050,
    &quot;val_sample_paths&quot;: [
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0070221.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0157901.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0085392.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0196480.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0096180.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0136149.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0143310.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0004560.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0115191.png&quot;,
      &quot;../dataset/ocr_rec_latexocr_dataset_example/images/train_0015323.png&quot;
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/ocr_rec_latexocr_dataset_example&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;FormulaRecDataset&quot;
}
</code></pre>
<p>In the above validation results, <code>check_pass</code> being True indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:
* <code>attributes.train_samples</code>: The number of training samples in this dataset is 9452;
* <code>attributes.val_samples</code>: The number of validation samples in this dataset is 1050;
* <code>attributes.train_sample_paths</code>: A list of relative paths to the visualized training samples in this dataset;
* <code>attributes.val_sample_paths</code>: A list of relative paths to the visualized validation samples in this dataset;</p>
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
python main.py -c paddlex/configs/modules/formula_recognition/PP-FormulaNet-S.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ocr_rec_latexocr_dataset_example
```
The following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-FormulaNet-S.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Set the mode to model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`.
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file instructions for the corresponding task module of the model [PaddleX Common Configuration File Parameters](../../instructions/config_parameters_common.en.md).

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
