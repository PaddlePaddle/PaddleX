---
comments: true
---

# Face Detection Module Development Tutorial

## I. Overview
Face detection is a fundamental task in object detection, aiming to automatically identify and locate the position and size of faces in input images. It serves as the prerequisite and foundation for subsequent tasks such as face recognition and face analysis. Face detection accomplishes this by constructing deep neural network models that learn the feature representations of faces, enabling efficient and accurate face detection.

## II. Supported Model List


<table>
<thead>
<tr>
<th style="text-align: center;">Model</th><th>Model Download Link</th>
<th style="text-align: center;">AP (%)<br/>Easy/Medium/Hard</th>
<th style="text-align: center;">GPU Inference Time (ms)</th>
<th style="text-align: center;">CPU Inference Time (ms)</th>
<th style="text-align: center;">Model Size (M)</th>
<th style="text-align: center;">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;">BlazeFace</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/BlazeFace_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace_pretrained.pdparams">Trained Model</a></td>
<td style="text-align: center;">77.7/73.4/49.5</td>
<td style="text-align: center;">49.9</td>
<td style="text-align: center;">68.2</td>
<td style="text-align: center;">0.447</td>
<td style="text-align: center;">A lightweight and efficient face detection model</td>
</tr>
<tr>
<td style="text-align: center;">BlazeFace-FPN-SSH</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/BlazeFace-FPN-SSH_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace-FPN-SSH_pretrained.pdparams">Trained Model</a></td>
<td style="text-align: center;">83.2/80.5/60.5</td>
<td style="text-align: center;">52.4</td>
<td style="text-align: center;">73.2</td>
<td style="text-align: center;">0.606</td>
<td style="text-align: center;">An improved model of BlazeFace, incorporating FPN and SSH structures</td>
</tr>
<tr>
<td style="text-align: center;">PicoDet_LCNet_x2_5_face</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet_LCNet_x2_5_face_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_LCNet_x2_5_face_pretrained.pdparams">Trained Model</a></td>
<td style="text-align: center;">93.7/90.7/68.1</td>
<td style="text-align: center;">33.7</td>
<td style="text-align: center;">185.1</td>
<td style="text-align: center;">28.9</td>
<td style="text-align: center;">Face Detection model based on PicoDet_LCNet_x2_5</td>
</tr>
<tr>
<td style="text-align: center;">PP-YOLOE_plus-S_face</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-YOLOE_plus-S_face_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_face_pretrained.pdparams">Trained Model</a></td>
<td style="text-align: center;">93.9/91.8/79.8</td>
<td style="text-align: center;">25.8</td>
<td style="text-align: center;">159.9</td>
<td style="text-align: center;">26.5</td>
<td style="text-align: center;">Face Detection model based on PP-YOLOE_plus-S</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are evaluated on the WIDER-FACE validation set with an input size of 640*640. GPU inference time is based on an NVIDIA V100 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz and FP32 precision.</b>
</details>

## III. Quick Integration  <a id="quick"> </a>
Before quick integration, you need to install the PaddleX wheel package. For the installation method of the wheel package, please refer to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md). After installing the wheel package, a few lines of code can complete the inference of the face detection module. You can switch models under this module freely, and you can also integrate the model inference of the face detection module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_detection.png) to your local machine.

```python
from paddlex import create_model

model_name = "PicoDet_LCNet_x2_5_face"

model = create_model(model_name)
output = model.predict("face_detection.png", batch_size=1)

for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

For more information on the usage of PaddleX's single-model inference API, please refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better face detection models. Before using PaddleX to develop face detection models, ensure you have installed the PaddleDetection plugin for PaddleX. The installation process can be found in the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides a data validation function for each module, and <b>only data that passes the validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development based on the official demos. If you wish to use private datasets for subsequent model training, refer to the [PaddleX Object Detection Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/object_detection.en.md).

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/widerface_coco_examples.tar -P ./dataset
tar -xf ./dataset/widerface_coco_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
```

After executing the above command, PaddleX will validate the dataset and collect its basic information. Upon successful execution, the log will print the message `Check dataset passed !`. The validation result file will be saved in `./output/check_dataset_result.json`, and related outputs will be saved in the `./output/check_dataset` directory of the current directory. The output directory includes visualized example images and histograms of sample distributions.

<details><summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>

<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;num_classes&quot;: 1,
    &quot;train_samples&quot;: 500,
    &quot;train_sample_paths&quot;: [
      &quot;check_dataset/demo_img/0--Parade/0_Parade_marchingband_1_849.jpg&quot;,
      &quot;check_dataset/demo_img/0--Parade/0_Parade_Parade_0_904.jpg&quot;,
      &quot;check_dataset/demo_img/0--Parade/0_Parade_marchingband_1_799.jpg&quot;
    ],
    &quot;val_samples&quot;: 100,
    &quot;val_sample_paths&quot;: [
      &quot;check_dataset/demo_img/1--Handshaking/1_Handshaking_Handshaking_1_384.jpg&quot;,
      &quot;check_dataset/demo_img/1--Handshaking/1_Handshaking_Handshaking_1_538.jpg&quot;,
      &quot;check_dataset/demo_img/1--Handshaking/1_Handshaking_Handshaking_1_429.jpg&quot;
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/example_data/widerface_coco_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;COCODetDataset&quot;
}
</code></pre>
<p>The verification results mentioned above indicate that <code>check_pass</code> being <code>True</code> means the dataset format meets the requirements. Details of other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 1;</li>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 500;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 100;</li>
<li><code>attributes.train_sample_paths</code>: The list of relative paths to the visualization images of training samples in this dataset;</li>
<li><code>attributes.val_sample_paths</code>: The list of relative paths to the visualization images of validation samples in this dataset;</li>
</ul>
<p>The dataset verification also analyzes the distribution of sample numbers across all classes and generates a histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/face_det/01.png"></p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)

After completing dataset verification, you can convert the dataset format or re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>Face detection does not support data format conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>Parameters for dataset splitting can be set by modifying the <code>CheckDataset</code> section in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. Set to <code>True</code> to enable dataset splitting, default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, set the percentage of the training set. The type is any integer between 0-100, ensuring the sum with <code>val_percent</code> is 100;</li>
</ul>
<p>For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, modify the configuration file as follows:</p>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters can also be set by appending command-line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/widerface_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training

A single command is sufficient to complete model training, taking the training of PicoDet_LCNet_x2_5_face as an example:

```bash
python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
```
The steps required are:

* Specify the path to the `.yaml` configuration file of the model (here it is `PicoDet_LCNet_x2_5_face.yaml`，When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters for Model Tasks](../../instructions/config_parameters_common.en.md).

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
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;</li>
</ul></details>

### <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation, you can complete the evaluation with a single command:

```bash
python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
```
Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model（here it's `PicoDet_LCNet_x2_5_face.yaml`）
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to [PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common.en.md)。

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model/model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully, and the model's evaluation metrics, including AP.</p></details>

### <b>4.4 Model Inference</b>
After completing model training and evaluation, you can use the trained model weights for inference prediction. In PaddleX, model inference prediction can be achieved through two methods: command line and wheel package.

#### 4.4.1 Model Inference
* To perform inference prediction through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_detection.png) to your local machine.
```bash
python main.py -c paddlex/configs/modules/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="face_detection.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it is `PicoDet_LCNet_x2_5_face.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weight path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or into your own project.

1. <b>Pipeline Integration</b>

The face detection module can be integrated into PaddleX pipelines such as [<b>Face Recognition</b>](../../../pipeline_usage/tutorials/cv_pipelines/face_recognition.en.md). Simply replace the model path to update the face detection module of the relevant pipeline. In pipeline integration, you can use high-performance inference and service-oriented deployment to deploy your model.

2. <b>Module Integration</b>

The weights you produce can be directly integrated into the face detection module. You can refer to the Python example code in [Quick Integration](#quick), simply replace the model with the path to your trained model.
