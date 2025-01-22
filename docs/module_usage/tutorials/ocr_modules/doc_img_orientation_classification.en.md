---
comments: true
---

# Document Image Orientation Classification Module Development Tutorial

## I. Overview
The document image orientation classification module is aim to distinguish the orientation of document images and correct them through post-processing. In processes such as document scanning and ID card photography, capturing devices are sometimes rotated to obtain clearer images, resulting in images with varying orientations. Standard OCR pipelines cannot effectively handle such data. By utilizing image classification technology, we can pre-judge the orientation of document or ID card images containing text regions and adjust their orientations, thereby enhancing the accuracy of OCR processing.

## II. Supported Model List


<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Accuracy (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Trained Model</a></td>
<td>99.06</td>
<td>3.84845</td>
<td>9.23735</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, with four categories: 0°, 90°, 180°, 270°</td>
</tr>
</tbody>
</table>
<b>Note: The above accuracy metrics are evaluated on a self-built dataset covering various scenarios such as IDs and documents, containing 1000 images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## III. Quick Integration

> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to [PaddleX Local Installation Tutorial](../../../installation/installation.en.md)

Just a few lines of code can complete the inference of the document image orientation classification module, allowing you to easily switch between models under this module. You can also integrate the model inference of the the document image orientation classification module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg) to your local machine.

```bash
from paddlex import create_model
model = create_model("PP-LCNet_x1_0_doc_ori")
output = model.predict("img_rot180_demo.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/demo.png")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single model inference API, refer to [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you seek higher accuracy, you can leverage PaddleX's custom development capabilities to develop better document image orientation classification models. Before developing a document image orientation classification model with PaddleX, ensure you have installed PaddleClas plugin for PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare a dataset for the task. PaddleX provides data validation functionality for each module. <b>Only data that passes validation can be used for model training.</b> Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to [PaddleX Image Classification Task Module Data Preparation Tutorial](../../../data_annotations/cv_modules/image_classification.en.md).

#### 4.1.1 Demo Data Download
You can download the demo dataset to a specified folder using the following commands:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/text_image_orientation.tar -P ./dataset
tar -xf ./dataset/text_image_orientation.tar  -C ./dataset/
```

#### 4.1.2 Data Validation
Data validation can be completed with a single command:

```bash
python main.py -c paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/text_image_orientation
```

After executing the above command, PaddleX will verify the dataset and collect basic information about it. Once the command runs successfully, a message saying `Check dataset passed !` will be printed in the log. The verification results will be saved in `./output/check_dataset_result.json`, and related outputs will be stored in the `./output/check_dataset` directory, including visual examples of sample images and a histogram of sample distribution.

<details><summary>👉 <b>Verification Result Details (click to expand)</b></summary>

<p>The specific content of the verification result file is:</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;label_file&quot;: &quot;..\/..\/text_image_orientation\/label.txt&quot;,
    &quot;num_classes&quot;: 4,
    &quot;train_samples&quot;: 1553,
    &quot;train_sample_paths&quot;: [
      &quot;check_dataset\/demo_img\/img_rot270_10351.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot0_3908.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot180_7712.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot0_7480.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot270_9599.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot90_10323.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot90_4885.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot180_3939.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot90_7153.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot180_1747.jpg&quot;
    ],
    &quot;val_samples&quot;: 2593,
    &quot;val_sample_paths&quot;: [
      &quot;check_dataset\/demo_img\/img_rot270_3190.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot0_10272.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot0_9930.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot90_918.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot180_2079.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot90_8574.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot90_7595.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot90_1751.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot180_1573.jpg&quot;,
      &quot;check_dataset\/demo_img\/img_rot90_4401.jpg&quot;
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset\/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;.\/text_image_orientation&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;ClsDataset&quot;
}
</code></pre>
<p>In the verification results above, <code>check_pass</code> being True indicates that the dataset format meets the requirements. Explanations of other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 4;</li>
<li><code>attributes.train_samples</code>: The number of training samples in this dataset is 1552;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in this dataset is 2593;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to visual sample images for the training set of this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the visual samples in the validation set of this dataset;</li>
</ul>
<p>Additionally, the dataset validation analyzes the sample number distribution across all classes in the dataset and generates a distribution histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/doc_img_ori_classification/01.png"></p></details>


#### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format and re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details of Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>Document image orientation classification does not currently support dataset format conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>Parameters for dataset splitting can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/text_image_orientation
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters also support setting through appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/text_image_orientation \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>


### 4.2 Model Training

Model training can be completed with just one command. Here, we use the document image orientation classification model (PP-LCNet_x1_0_doc_ori) as an example:

```bash
python main.py -c paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/text_image_orientation
```

You need to follow these steps:

* Specify the path to the model's `.yaml` configuration file (here, `PP-LCNet_x1_0_doc_ori.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md)).
* Set the mode to model training: `-o Global.mode=train`.
* Specify the training dataset path: `-o Global.dataset_dir`.

Other relevant parameters can be set by modifying fields under `Global` and `Train` in the `.yaml` configuration file, or by appending arguments to the command line. For example, to specify the first two GPUs for training: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and detailed explanations, refer to the [PaddleX General Model Configuration File Parameters](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Information (click to expand)</b></summary>

<ul>
<li>During model training, PaddleX automatically saves the model weight files, defaulting to <code>output</code>. If you want to specify a different save path, you can set it using the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX abstracts away the concept of dynamic graph weights and static graph weights. During model training, it produces both dynamic and static graph weights. For model inference, it defaults to using static graph weights.</li>
<li>
<p>After completing model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including the following:</p>
</li>
<li>
<p><code>train_result.json</code>: Training result record file, which records whether the training task was completed normally, as well as the output weight metrics and related file paths.</p>
</li>
<li><code>train.log</code>: Training log file, which records changes in model metrics and loss during training.</li>
<li><code>config.yaml</code>: Training configuration file, which records the hyperparameter configuration for this training.</li>
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.</li>
</ul></details>

### <b>4.3 Model Evaluation</b>

After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. With PaddleX, model evaluation can be done with just one command:

```bash
python main.py -c paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/text_image_orientation
```
Similar to model training and evaluation, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0_doc_ori.yaml`).
* Set the mode to model inference prediction: `-o Global.mode=predict`.
* Specify the path to the model weights: `-o Predict.model_dir="./output/best_model/inference"`.
* Specify the input data path: `-o Predict.input="..."`.
Other relevant parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX General Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Information (click to expand)</b></summary>

<ul>
<li>
<p>When conducting model evaluation, it is necessary to specify the model weight file path. Each configuration file has a built-in default path for saving weights. If you need to change this path, you can simply append a command line argument to set it, for example: <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
</li>
<li>
<p>After the model evaluation is completed, typically, the following outputs are generated:</p>
</li>
<li>
<p>Upon finishing the model evaluation, an evaluate_result.json file is produced, which records the results of the evaluation. Specifically, it logs whether the evaluation task was successfully completed and the evaluation metrics of the model, including <code>Top1 Accuracy (Top1 Acc)</code>.</p>
</li>
</ul></details>

### <b>4.4 Model Inference and Model Integration</b>

After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference

To perform inference predictions via the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg) to your local machine.

```bash
python main.py -c paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="img_rot180_demo.jpg"
```

Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it's `PP-LCNet_x1_0_doc_ori.yaml`)

* Set the mode to model inference prediction: `-o Global.mode=predict`

* Specify the model weights path: -o Predict.model_dir="./output/best_accuracy/inference"

Specify the input data path: `-o Predict.input="..."` Other related parameters can be set by modifying the fields under Global and Predict in the `.yaml` configuration file. For details, refer to PaddleX Common Model Configuration File Parameter Description.

Alternatively, you can use the PaddleX wheel package for inference, easily integrating the model into your own projects.

#### 4.4.2 Model Integration

The model can be directly integrated into the PaddleX pipeline or into your own projects.

1.<b>Pipeline Integration</b>

The document image classification module can be integrated into PaddleX pipelines such as the [Document Scene Information Extraction Pipeline (PP-ChatOCRv3)](../../..//pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.en.md). Simply replace the model path to update the The document image classification module's model.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the document image orientation classification module. You can refer to the Python sample code in [Quick Integration](#iii-quick-integration) and just replace the model with the path to the model you trained.
