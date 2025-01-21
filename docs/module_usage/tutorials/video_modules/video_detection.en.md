---
comments: true
---

# Video Detection Module Development Tutorial

## I. Overview
Video detection tasks are a critical component of computer vision systems, focusing on identifying and locating objects or events within video sequences. Video detection involves decomposing the video into individual frame sequences and then analyzing these frames to recognize detected objects or actions, such as detecting pedestrians in surveillance videos or identifying specific activities like "running," "jumping," or "playing guitar" in sports or entertainment videos.

The output of the video detection module includes bounding boxes and class labels for each detected object or event. This information can be used by other modules or systems for further analysis, such as tracking the movement of detected objects, generating alerts, or compiling statistics for decision-making processes. Therefore, video detection plays an important role in various applications ranging from security surveillance and autonomous driving to sports analytics and content moderation.

## II. List of Supported Models


<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Frame-mAP (@ IoU 0.5)</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>YOWO</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/YOWO_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOWO_pretrained.pdparams">训练模型</a></td>
<td>80.94</td>
<td>462.891M</td>
<td rowspan="1">
YOWO is a single-stage network with two branches. One branch extracts spatial features of key frames (i.e., the current frame) through a 2D-CNN, while the other branch captures spatiotemporal features of a clip composed of previous frames using a 3D-CNN. To accurately aggregate these features, YOWO employs a channel fusion and attention mechanism to maximize the utilization of inter-channel dependencies. Finally, the fused features are used for frame-level detection.
</td>
</tr>

</table>

<p><b>Note: The above accuracy metrics refer to Frame-mAP (@ IoU 0.5) Accuracy on the  <a href="http://www.thumos.info/download.html">UCF101-24</a> test set. </b><b>All model GPU inference times are based on NVIDIA Tesla T4 machines, with precision type FP32. CPU inference speeds are based on Intel® Xeon® Gold 5117 CPU @ 2.00GHz, with 8 threads and precision type FP32.</b></p></details>

## <span id="lable">III. Quick Integration</span>
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

After installing the wheel package, you can complete video Detection module inference with just a few lines of code. You can switch between models in this module freely, and you can also integrate the model inference of the video Detection module into your project. Before running the following code, please download the [demo video](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi) to your local machine.

```python
from paddlex import create_model
model = create_model("YOWO")
output = model.predict("HorseRiding.avi", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_video("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference APIs, please refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you are seeking higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better video Detection models. Before using PaddleX to develop video Detection models, please ensure that you have installed the relevant model training plugins for video Detection in PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the dataset for the corresponding task module. PaddleX provides data validation functionality for each module, and <b>only data that passes data validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use your own private dataset for subsequent model training, please refer to the [PaddleX Video Detection Task Module Data Annotation Guide](../../../data_annotations/video_modules/video_detection.en.md).

#### 4.1.1 Demo Data Download
You can use the following command to download the demo dataset to a specified folder:
```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/video_det_examples.tar -P ./dataset
tar -xf ./dataset/video_det_examples.tar -C ./dataset/
```
#### 4.1.2 Data Validation
One command is all you need to complete data validation:

```bash
python main.py -c paddlex/configs/video_detection/YOWO.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/video_det_examples
```
After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.

<details><summary>👉 <b>Validation Results Details (Click to Expand)</b></summary>

<pre><code class="language-bash">
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "..\/..\/dataset\/video_det_examples\/label_map.txt",
    "num_classes": 24,
    "train_samples": 6878,
    "train_sample_paths": [
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/SoccerJuggling\/v_SoccerJuggling_g19_c06\/00296.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/SkateBoarding\/v_SkateBoarding_g17_c04\/00026.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/RopeClimbing\/v_RopeClimbing_g01_c03\/00055.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/HorseRiding\/v_HorseRiding_g11_c05\/00132.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/PoleVault\/v_PoleVault_g13_c03\/00089.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/Basketball\/v_Basketball_g13_c04\/00050.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/PoleVault\/v_PoleVault_g01_c05\/00024.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/RopeClimbing\/v_RopeClimbing_g03_c04\/00118.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/GolfSwing\/v_GolfSwing_g01_c06\/00231.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/TrampolineJumping\/v_TrampolineJumping_g02_c02\/00134.jpg"
    ],
    "val_samples": 3916,
    "val_sample_paths": [
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/IceDancing\/v_IceDancing_g22_c02\/00017.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/TennisSwing\/v_TennisSwing_g04_c02\/00046.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/SoccerJuggling\/v_SoccerJuggling_g08_c03\/00169.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/Fencing\/v_Fencing_g24_c02\/00009.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/Diving\/v_Diving_g16_c02\/00110.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/HorseRiding\/v_HorseRiding_g08_c02\/00079.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/PoleVault\/v_PoleVault_g17_c07\/00008.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/Skiing\/v_Skiing_g20_c06\/00221.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/PoleVault\/v_PoleVault_g17_c07\/00137.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/GolfSwing\/v_GolfSwing_g24_c01\/00093.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "video_det_examples",
  "show_type": "video",
  "dataset_type": "VideoDetDataset"
}
</code></pre>
<p>The above validation results, with check_pass being True, indicate that the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 24;</li>
<li><code>attributes.train_samples</code>: The number of training set samples in this dataset is 6878;</li>
<li><code>attributes.val_samples</code>: The number of validation set samples in this dataset is 3916;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to the visual samples in the training set of this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the visual samples in the validation set of this dataset;</li>
</ul>
<p>Additionally, the dataset validation analyzes the sample number distribution across all classes in the dataset and generates a distribution histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/video_detection/01.png"></p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format or re-split the training/validation ratio of the dataset by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Dataset Format Conversion/Dataset Splitting Details (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>Image Detection does not currently support data conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>Image Detection does not currently support data conversion.</p>

### 4.2 Model Training
A single command can complete the model training. Taking the training of the video Detection model YOWO as an example:
```
python main.py -c paddlex/configs/video_det_examples/YOWO.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/video_det_examples
```

the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `YOWO.yaml`. When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path of the training dataset: `-o Global.dataset_dir`. Other related parameters can be set by modifying the fields under `Global` and `Train` in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the second GPU: `-o Global.device=gpu:2`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file parameter instructions for the corresponding task module of the model [PaddleX Common Model Configuration File Parameters](../../instructions/config_parameters_common.en.md).


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
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model accuracy. Using PaddleX for model evaluation, a single command can complete the model evaluation:
```bash
python main.py -c  paddlex/configs/video_detection/YOWO.yaml  \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/video_det_examples
```
Similar to model training, the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `YOWO.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path of the validation dataset: `-o Global.dataset_dir`. Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration. Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the model weight file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results. Specifically, it records whether the evaluation task was completed successfully and the model's evaluation metrics, including mAP;</p></details>

### <b>4.4 Model Inference and Model Integration</b>
After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
To perform inference prediction through the command line, simply use the following command. Before running the following code, please download the [demo video](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi) to your local machine.

```bash
python main.py -c paddlex/configs/video_detection/YOWO.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="HorseRiding.avi"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `YOWO.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weight path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipelines or directly into your own project.

1.<b>Pipeline Integration</b>


The video Detection module can be integrated into the [General Video Detection Pipeline](../../../pipeline_usage/tutorials/video_pipelines/video_detection.en.md) of PaddleX. Simply replace the model path to update the video Detection module of the relevant pipeline. In pipeline integration, you can use high-performance inference and service-oriented deployment to deploy your obtained model.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the video Detection module. You can refer to the Python example code in <a href="#lable">Quick Integration</a>  and simply replace the model with the path to your trained model.
