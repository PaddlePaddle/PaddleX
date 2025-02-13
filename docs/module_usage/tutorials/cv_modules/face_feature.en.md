---
comments: true
---

# Face Feature Module Usage Tutorial

## I. Overview
Face feature models typically take standardized face images processed through detection, extraction, and keypoint correction as input. These models extract highly discriminative facial features from these images for subsequent modules, such as face matching and verification tasks.

## II. Supported Model List

<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Output Feature Dimension</th>
<th>Acc (%)<br/>AgeDB-30/CFP-FP/LFW</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>MobileFaceNet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/MobileFaceNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileFaceNet_pretrained.pdparams">Trained Model</a></td>
<td>128</td>
<td>96.28/96.71/99.58</td>
<td>3.16 / 0.48</td>
<td>6.49 / 6.49</td>
<td>4.1</td>
<td>Face feature model trained on MobileFaceNet with MS1Mv3 dataset</td>
</tr>
<tr>
<td>ResNet50_face</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_face_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_face_pretrained.pdparams">Trained Model</a></td>
<td>512</td>
<td>98.12/98.56/99.77</td>
<td>5.68 / 1.09</td>
<td>14.96 / 11.90</td>
<td>87.2</td>
<td>Face feature model trained on ResNet50 with MS1Mv3 dataset</td>
</tr>
</tbody>
</table>
<p>Note: The above accuracy metrics are Accuracy scores measured on the AgeDB-30, CFP-FP, and LFW datasets, respectively. All model GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</p>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For details, refer to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md)

After installing the whl package, a few lines of code can complete the inference of the face feature module. You can switch models under this module freely, and you can also integrate the model inference of the face feature module into your project. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_recognition_001.jpg) to your local machine.

```python
from paddlex import create_model

model_name = "MobileFaceNet"

model = create_model(model_name)
output = model.predict("face_recognition_001.jpg", batch_size=1)

for res in output:
    res.print(json_format=False)
    res.save_to_json("./output/res.json")
```
<details><summary>👉 <b>The result obtained after running is: (Click to expand)</b></summary>

```bash
{'res': {'input_path': 'face_recognition_001.jpg', 'feature': [0.04121152311563492, 0.0010890548583120108, -0.03561094403266907, 0.05722084641456604, 0.05919725075364113, -0.007132374215871096, -0.061298906803131104, -0.10843975096940994, -0.02871585637331009, 0.03347175195813179, 0.13309064507484436, 0.05309445410966873, 0.004820522852241993, -0.11700531840324402, 0.03240801766514778, 0.0639009103178978, 0.17841649055480957, 0.006999856326729059, -0.052513156086206436, 0.14528249204158783, 0.013314608484506607, -0.04820159450173378, -0.04795005917549133, 0.184268519282341, -0.15508289635181427, -0.01048946287482977, -0.103487029671669, 0.020606128498911858, 0.11970002949237823, 0.07393684983253479, -0.05581602826714516, -0.10253427177667618, -0.015256273560225964, 0.06347685307264328, 0.0893929973244667, -0.01050905603915453, -0.025690989568829536, -0.10570172965526581, -0.11608698219060898, -0.04072513058781624, 0.05093423277139664, 0.044215817004442215, 0.1629297435283661, -0.06339056044816971, -0.07671815156936646, 0.09480706602334976, -0.15456975996494293, -0.021657753735780716, 0.12482058256864548, -0.1267298310995102, 0.002465370809659362, -0.05374367907643318, -0.07079283148050308, 0.1325870305299759, -0.006946612149477005, 0.047657083719968796, 0.06102422997355461, -0.18113569915294647, -0.15677541494369507, -0.05817852169275284, -0.007711497135460377, -0.03407919406890869, 0.04798268899321556, -0.036309171468019485, 0.10679583996534348, -0.1858624368906021, -0.06799137592315674, 0.008694482035934925, 0.026530278846621513, -0.06917411088943481, 0.13533912599086761, -0.08762945234775543, -0.17223820090293884, -0.024798616766929626, -0.03390877693891525, -0.17003266513347626, -0.08045653998851776, -0.21928688883781433, -0.08328460901975632, 0.0745469480752945, -0.05523530766367912, -0.08471746742725372, -0.06595447659492493, 0.11475134640932083, 0.12401033192873001, 0.09317877888679504, -0.08352484554052353, 0.0247682835906744, 0.0008310621487908065, -0.09977596998214722, -0.002699024509638548, -0.23338164389133453, -0.1783595234155655, -0.08259879052639008, 0.14328709244728088, 0.024862702935934067, 0.008164866827428341, 0.06340813636779785, 0.1028614193201065, -0.038397643715143204, -0.05210508778691292, 0.0389365553855896, 0.12757952511310577, 0.05326246842741966, 0.06695418804883957, -0.0052042435854673386, 0.035264499485492706, 0.00990584772080183, -0.05249840393662453, 0.06972697377204895, -0.06477969884872437, -0.003332878928631544, 0.0449349470436573, 0.020609190687537193, 0.074540875852108, -0.03608720749616623, 0.04876900464296341, -0.06063542515039444, -0.07829384505748749, -0.1220116913318634, 0.05064597725868225, 0.07839702069759369, 0.06130668520927429, -0.13095220923423767, 0.0888662114739418, -0.029464716091752052, 0.030264943838119507, 0.04124804586172104]}}
```

Parameter meanings are as follows:
- `input_path`: The path of the input image to be predicted.
- `feature`: The feature vector extracted by the model.

</details>

The explanations for the methods, parameters, etc., are as follows:

* `create_model` instantiates a face feature model (here, `MobileFaceNet` is used as an example), and the specific explanations are as follows:
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
<td><code>flip</code></td>
<td>Whether to perform flipped inference; if True, the model will infer the horizontally flipped input image and fuse the results of both inferences to improve the accuracy of face features</td>
<td><code>bool</code></td>
<td>None</td>
<td><code>False</code></td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the default model parameters built into PaddleX are used. If `model_dir` is specified, the user-defined model is used.

* The `predict()` method of the face feature model is called for inference prediction. The `predict()` method has parameters `input` and `batch_size`, which are explained as follows:

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
<li><b>Python Variable</b>, such as image data represented by <code>numpy.ndarray</code></li>
<li><b>File Path</b>, such as the local path of an image file: <code>/root/data/img.jpg</code></li>
<li><b>URL Link</b>, such as the web URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">Example</a></li>
<li><b>Local Directory</b>, the directory should contain the data files to be predicted, such as the local path: <code>/root/data/</code></li>
<li><b>List</b>, the elements of the list should be of the above-mentioned data types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
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
</table>

* Additionally, it supports obtaining the prediction results through attributes, as follows:

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
</table>

For more information on using the PaddleX single-model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you aim for higher accuracy with existing models, you can leverage PaddleX's custom development capabilities to develop better face feature models. Before developing face feature models with PaddleX, ensure you have installed the PaddleX PaddleClas plugin. The installation process can be found in the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md)

### 4.1 Data Preparation
Before model training, you need to prepare the dataset for the corresponding task module. PaddleX provides data validation functionality for each module, and <b>only data that passes validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, allowing you to complete subsequent development based on the official demo data. If you wish to use a private dataset for subsequent model training, the training dataset for the face feature module is organized in a general image classification dataset format. You can refer to the [PaddleX Image Classification Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/image_classification.en.md). If you wish to use a private dataset for subsequent model evaluation, note that the validation dataset format for the face feature module differs from the training dataset format. Please refer to [Section 4.1.4 Data Organization Face Feature Module](#414-Data-Organization-for-Face-Feature-Module)

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/face_rec_examples.tar -P ./dataset
tar -xf ./dataset/face_rec_examples.tar -C ./dataset/
```
#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/modules/face_feature/MobileFaceNet.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/face_rec_examples
```

After executing the above command, PaddleX will validate the dataset and collect its basic information. Upon successful execution, the log will print the message `Check dataset passed !`. The validation result file will be saved in `./output/check_dataset_result.json`, and related outputs will be saved in the `./output/check_dataset` directory of the current directory. The output directory includes visualized example images and histograms of sample distributions.

<details><summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>
<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_label_file": "../../dataset/face_rec_examples/train/label.txt",
    "train_num_classes": 995,
    "train_samples": 1000,
    "train_sample_paths": [
      "check_dataset/demo_img/01378592.jpg",
      "check_dataset/demo_img/04331410.jpg",
      "check_dataset/demo_img/03485713.jpg",
      "check_dataset/demo_img/02382123.jpg",
      "check_dataset/demo_img/01722397.jpg",
      "check_dataset/demo_img/02682349.jpg",
      "check_dataset/demo_img/00272794.jpg",
      "check_dataset/demo_img/03151987.jpg",
      "check_dataset/demo_img/01725764.jpg",
      "check_dataset/demo_img/02580369.jpg"
    ],
    "val_label_file": "../../dataset/face_rec_examples/val/pair_label.txt",
    "val_num_classes": 2,
    "val_samples": 500,
    "val_sample_paths": [
      "check_dataset/demo_img/Don_Carcieri_0001.jpg",
      "check_dataset/demo_img/Eric_Fehr_0001.jpg",
      "check_dataset/demo_img/Harry_Kalas_0001.jpg",
      "check_dataset/demo_img/Francis_Ford_Coppola_0001.jpg",
      "check_dataset/demo_img/Amer_al-Saadi_0001.jpg",
      "check_dataset/demo_img/Sergei_Ivanov_0001.jpg",
      "check_dataset/demo_img/Erin_Runnion_0003.jpg",
      "check_dataset/demo_img/Bill_Stapleton_0001.jpg",
      "check_dataset/demo_img/Daniel_Bruehl_0001.jpg",
      "check_dataset/demo_img/Clare_Short_0004.jpg"
    ]
  },
  "analysis": {},
  "dataset_path": "./dataset/face_rec_examples",
  "show_type": "image",
  "dataset_type": "ClsDataset"
}
</code></pre>
<p>The verification results mentioned above indicate that <code>check_pass</code> being <code>True</code> means the dataset format meets the requirements. Details of other indicators are as follows:</p>
<ul>
<li><code>attributes.train_num_classes</code>: The number of classes in this training dataset is 995;</li>
<li><code>attributes.val_num_classes</code>: The number of classes in this validation dataset is 2;</li>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 1000;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 500;</li>
<li><code>attributes.train_sample_paths</code>: The list of relative paths to the visualization images of training samples in this dataset;</li>
</ul></details>

#### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the data validation, you can convert the dataset format and re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>adding hyperparameters</b>.

<details><summary>👉 <b>Details on Format Conversion / Dataset Splitting (Click to Expand)</b></summary>
<p>The face feature module does not support data format conversion or dataset splitting.</p></details>

#### 4.1.4 Data Organization for Face Feature Module

The format of the validation dataset for the face feature module differs from the training dataset. If you need to evaluate model accuracy on private data, please organize your dataset as follows:

```bash
face_rec_dataroot      # Root directory of the dataset, the directory name can be changed
├── train              # Directory for saving the training dataset, the directory name cannot be changed
   ├── images          # Directory for saving images, the directory name can be changed but should correspond to the content in label.txt
      ├── xxx.jpg      # Face image file
      ├── xxx.jpg      # Face image file
      ...
   └──label.txt       # Training set annotation file, the file name cannot be changed. Each line gives the relative path of the image to `train` and the face image class (face identity) id, separated by a space. Example content: images/image_06765.jpg 0
├── val                # Directory for saving the validation dataset, the directory name cannot be changed
   ├── images          # Directory for saving images, the directory name can be changed but should correspond to the content in pair_label.txt
      ├── xxx.jpg      # Face image file
      ├── xxx.jpg      # Face image file
      ...
   └── pair_label.txt  # Validation dataset annotation file, the file name cannot be changed. Each line gives the paths of two images to be compared and a 0 or 1 label indicating whether the pair of images belong to the same person, separated by spaces.
```

Example content of the validation set annotation file `pair_label.txt`:

```bash
# Face image 1.jpg Face image 2.jpg Label (0 indicates the two face images do not belong to the same person, 1 indicates they do)
images/Angela_Merkel_0001.jpg images/Angela_Merkel_0002.jpg 1
images/Bruce_Gebhardt_0001.jpg images/Masao_Azuma_0001.jpg 0
images/Francis_Ford_Coppola_0001.jpg images/Francis_Ford_Coppola_0002.jpg 1
images/Jason_Kidd_0006.jpg images/Jason_Kidd_0008.jpg 1
images/Miyako_Miyazaki_0002.jpg images/Munir_Akram_0002.jpg 0
```

### 4.2 Model Training
Model training can be completed with a single command. Here is an example of training MobileFaceNet:

```bash
python main.py -c paddlex/configs/modules/face_feature/MobileFaceNet.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/face_rec_examples
```
The steps required are:

* Specify the path to the `.yaml` configuration file for the model (here it is `MobileFaceNet.yaml`)
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file or by appending parameters in the command line. For example, to specify the first two GPUs for training: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file instructions for the corresponding task module [PaddleX Common Configuration Parameters for Model Tasks](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>
<ul>
<li>During model training, PaddleX automatically saves model weight files, defaulting to <code>output</code>. To specify a save path, use the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.</li>
<li>When training other models, specify the corresponding configuration file. The correspondence between models and configuration files can be found in the <a href="../../../support_list/models_list.en.md">PaddleX Model List (CPU/GPU)</a>.
After completing model training, all outputs are saved in the specified output directory (default is <code>./output/</code>). Typically, the following outputs are included:</li>
<li><code>train_result.json</code>: A file that records the training results, indicating whether the training task was successfully completed, and includes metrics, paths to related files, etc.</li>
<li><code>train.log</code>: A log file that records changes in model metrics, loss variations, and other details during the training process.</li>
<li><code>config.yaml</code>: A configuration file that logs the hyperparameter settings for the current training session.</li>
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: Files related to model weights, including network parameters, optimizer, EMA (Exponential Moving Average), static graph network parameters, and static graph network structure.</li>
</ul>
<details>

### <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation, you can complete the evaluation with a single command:


<pre><code class="language-bash">python main.py -c paddlex/configs/modules/face_detection/MobileFaceNet.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/face_rec_examples
</code></pre>

Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model（here it's `MobileFaceNet.yaml`）
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to [PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common.en.md)。

<details>

<summary>👉 <b>More Details (Click to Expand)</b></summary>

During model evaluation, the path to the model weights file needs to be specified. Each configuration file has a default weight save path built in. If you need to change it, you can set it by appending a command line parameter, such as `-o Evaluate.weight_path="./output/best_model/best_model/model.pdparams"`.

After completing the model evaluation, an `evaluate_result.json` file will be produced, which records the evaluation results. Specifically, it records whether the evaluation task was completed normally and the model's evaluation metrics, including Accuracy.

</details>

### <b>4.4 Model Inference</b>
After completing model training and evaluation, you can use the trained model weights for inference predictions. In PaddleX, model inference predictions can be implemented through two methods: command line and wheel package.

#### 4.4.1 Model Inference
* To perform inference predictions through the command line, you only need the following command. Before running the following code, please download the [example image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_recognition_001.jpg) to your local machine.
```bash
python main.py -c paddlex/configs/modules/face_feature/MobileFaceNet.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="face_recognition_001.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it is `MobileFaceNet.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the path to the model weights: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the path to the input data: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or into your own project.

1. <b>Pipeline Integration</b>

The face feature module can be integrated into the PaddleX pipeline for [<b>Face Recognition</b>](../../../pipeline_usage/tutorials/face_recognition_pipelines/face_recognition.en.md). You only need to replace the model path to update the face feature module of the relevant pipeline. In pipeline integration, you can use high-performance deployment and service-oriented deployment to deploy the model you obtained.

2. <b>Module Integration</b>

The weights you produced can be directly integrated into the face feature module. You can refer to the Python example code in [Quick Integration](#III.-Quick-Integration) and only need to replace the model with the path to the model you trained.

You can also use the PaddleX high-performance inference plugin to optimize the inference process of your model and further improve efficiency. For detailed procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).