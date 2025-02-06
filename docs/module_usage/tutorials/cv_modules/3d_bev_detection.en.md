---
comments: true
---

# Tutorial on Using the 3D Multimodal Fusion Detection Module

## I. Overview
The 3D multimodal fusion detection module is a key component in the fields of computer vision and autonomous driving, responsible for locating and marking the 3D coordinates and detection box information of regions containing specific targets in images or videos. The performance of this module directly affects the accuracy and efficiency of the entire vision or autonomous driving perception system. The 3D multimodal fusion detection module typically outputs 3D bounding boxes of target regions, which are then passed as inputs to the target recognition module for further processing.

## II. List of Supported Models

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>NDS</th>
<th>Introduction</th>
</tr>
<tr>
<td>BEVFusion</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BEVFusion_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BEVFusion_pretrained.pdparams">Training Model</a></td>
<td>53.9</td>
<td>60.9</td>

<td rowspan="2">BEVFusion is a multimodal fusion model in the Bird's Eye View (BEV) perspective. It uses two branches to process data from different modalities to obtain features of lidar and camera in the BEV perspective. The camera branch uses the LSS (Lift, Splat, and Scatter) bottom-up approach to explicitly generate image BEV features, while the lidar branch uses a classic point cloud detection network. Finally, the BEV features of the two modalities are aligned and fused for application in detection heads or segmentation heads.</td>
</tr>
<tr>
</table>

<p><b>Note: The above accuracy metrics are based on the <a href="https://www.nuscenes.org/nuscenes">nuscenes</a> validation set with mAP(0.5:0.95) and NDS 60.9, and the precision type is FP32.</b></p>

## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package first. For details, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

After installing the wheel package, you can complete the inference of the object detection module with just a few lines of code. You can switch between models under this module at will, and you can also integrate the model inference of the 3D multimodal fusion detection module into your project. Before running the following code, please download the [sample input](https://paddle-model-ecology.bj.bcebos.com/paddlex/det_3d/demo_det_3d/nuscenes_infos_val.pkl) to your local machine. 

```python
from paddlex import create_model
model = create_model(model_name="BEVFusion")
output = model.predict(input="nuscenes_infos_val.pkl", batch_size=1)
for res in output:
    res.print()
    res.save_to_json(save_path="./output/res.json")
```

After running, the result obtained is:

```bash
{"res":
  {
    "input_path": "./data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151616947490.pcd.bin", "input_img_paths": ["./data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151616904806.jpg", "./data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151616912404.jpg", "./data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151616920482.jpg", "./data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151616928113.jpg", "./data/nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151616937558.jpg", "./data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151616947405.jpg"], "sample_id": "cc57c1ea80fe46a7abddfdb15654c872", "boxes_3d": [[-8.913962364196777, 13.30993366241455, -1.7353310585021973, 1.9886571168899536, 4.886075019836426, 1.877254605293274, 6.317165374755859, -0.00018131558317691088, 0.022375036031007767]], "labels_3d": [0], "scores_3d": [0.9951273202896118]
  }
}
```

The meanings of the run results parameters are as follows:
- `input_path`: Indicates the input point cloud data path of the input sample to be predicted.
- `sample_id`: Indicates the unique identifier of the input sample to be predicted.
- `input_img_paths`: Indicates the input image data path of the input sample to be predicted.
- `boxes_3d`: Indicates the prediction box information of the 3D sample. Each prediction box information is a list of length 9, with each element representing:
  - 0: x coordinate of the center point
  - 1: y coordinate of the center point
  - 2: z coordinate of the center point
  - 3: Width of the detection box
  - 4: Length of the detection box
  - 5: Height of the detection box
  - 6: Rotation angle
  - 7: Velocity in the x direction of the coordinate system
  - 8: Velocity in the y direction of the coordinate system
- `labels_3d`: Indicates the predicted categories corresponding to all prediction boxes of the 3D sample.
- `scores_3d`: Indicates the confidence scores corresponding to all prediction boxes of the 3D sample.

The following is an explanation of relevant methods and parameters:

* `create_model` instantiates a 3D detection model (here, `BEVFusion` is used as an example), with specific explanations as follows:
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Optional</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>Name of the model</td>
<td><code>str</code></td>
<td>No</td>
<td><code>BEVFusion</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Path where the model is stored</td>
<td><code>str</code></td>
<td>No</td>
<td>None</td>
</tr>
</table>

* The `model_name` must be specified. After specifying `model_name`, the default model parameters built into PaddleX will be used. If `model_dir` is specified, the user-defined model will be used.

* The `predict()` method of the 3D detection model is called for inference prediction. The parameters of the `predict()` method are `input` and `batch_size`, with specific explanations as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Optional</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types</td>
<td><code>str</code>/<code>list</code></td>
<td>
<ul>
  <li><b>File path</b>, such as the local path of a 3D annotation file: <code>/root/data/anno_file.pkl</code></li>
  <li><b>List</b>, elements of the list must be of the above type, such as <code>[\"/root/data/anno_file1.pkl\", \"/root/data/anno_file2.pkl\"]</code></li>
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

* Process the prediction results. Each sample's prediction result is a corresponding Result object, and it supports operations such as printing and saving as a `json` file:

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
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. It is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. It is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the result as a JSON-formatted file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. When it is a directory, the saved file name will be consistent with the input file name</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. It is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. It is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
</table>

* Additionally, it supports obtaining visual images with results and prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">Get the prediction result in <code>json</code> format</td>
</tr>
</table>

For more information on the usage of PaddleX single-model inference API, please refer to [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Secondary Development
If you are pursuing higher precision in existing models, you can utilize the secondary development capabilities of PaddleX to develop better object detection models. Before using PaddleX to develop object detection models, please make sure to install the object detection model training plugin for PaddleX. The installation process can be referred to in the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before training a model, you need to prepare the dataset for the corresponding task module. PaddleX provides a data validation feature for each module, and <b>only data that passes the validation can be used for model training</b>. In addition, PaddleX offers a Demo dataset for each module, and you can complete subsequent development based on the official Demo data.

#### 4.1.1 Downloading Demo Data
You can refer to the following command to download the Demo dataset to the specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/nuscenes_demo.tar -P ./dataset
tar -xf ./dataset/nuscenes_demo.tar -C ./dataset/
```

#### 4.1.2 Data Validation
Data validation can be completed with a single command:

```bash
python main.py -c paddlex/configs/modules/3d_bev_detection/BEVFusion.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/nuscenes_demo
```

<details><summary>👉 <b>Verification Result Details (Click to Expand)</b></summary>

<p>The specific content of the verification result file is:</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;num_classes&quot;: 11,
    &quot;train_mate&quot;: [
      {
        &quot;sample_idx&quot;: &quot;f9878012c3f6412184c294c13ba4bac3&quot;,
        &quot;lidar_path&quot;: &quot;./data/nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin&quot;,
        &quot;image_paths&quot; [
          &quot;./data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-05-21-11-06-59-0400__CAM_FRONT_LEFT__1526915243004917.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-05-21-11-06-59-0400__CAM_FRONT_RIGHT__1526915243019956.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-05-21-11-06-59-0400__CAM_BACK_RIGHT__1526915243027813.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_BACK/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-05-21-11-06-59-0400__CAM_BACK_LEFT__1526915243047295.jpg&quot;
        ]
      },
    ],
    &quot;val_mate&quot;: [
      {
        &quot;sample_idx&quot;: &quot;30e55a3ec6184d8cb1944b39ba19d622&quot;,
        &quot;lidar_path&quot;: &quot;./data/nuscenes/samples/LIDAR_TOP/n015-2018-07-11-11-54-16+0800__LIDAR_TOP__1531281439800013.pcd.bin&quot;,
        &quot;image_paths&quot;: [
          &quot;./data/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_LEFT__1531281439754844.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439762460.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_RIGHT__1531281439770339.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_BACK_RIGHT__1531281439777893.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_BACK/n015-2018-07-11-11-54-16+0800__CAM_BACK__1531281439787525.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_BACK_LEFT/n015-2018-07-11-11-54-16+0800__CAM_BACK_LEFT__1531281439797423.jpg&quot;
        ]
      },
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;/workspace/bevfusion/Paddle3D/data/nuscenes&quot;,
  &quot;show_type&quot;: &quot;path for images and lidar&quot;,
  &quot;dataset_type&quot;: &quot;NuscenesMMDataset&quot;
}
</code></pre>
<p>In the verification results above, <code>check_pass</code> being true indicates that the dataset format meets the requirements.</p>
</details>

#### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After you complete the dataset verification, you can convert the dataset format or re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>adding hyperparameters</b>.

<details><summary>👉 <b>Details of Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

<p>The 3D multimodal fusion detection module does not support dataset format conversion or dataset splitting.</p></details>

### 4.2 Model Training
A single command can complete the model training. Taking the training of the 3D multimodal fusion detection model <code>BEVFusion</code> as an example:

```bash
python main.py -c paddlex/configs/modules/3d_bev_detection/BEVFusion.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/nuscenes_demo \
```

* Specify the path of the model's `.yaml` configuration file (here it is `bevf_pp_2x8_1x_nusc.yaml`. When training other models, you need to specify the corresponding configuration file. The correspondence between models and configuration files can be found in [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md)).
* Set the mode to model training: `-o Global.mode=train`
* Specify the training dataset path: `-o Global.dataset_dir`
Other related parameters can be set by modifying the fields under `Global` and `Train` in the `.yaml` configuration file, or by appending parameters in the command line. For example, to specify training on the first 2 GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file instructions for the corresponding model task module [PaddleX Common Model Configuration Parameters](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Information (Click to Expand)</b></summary>

<ul>
<li>During model training, PaddleX will automatically save the model weight files, with the default directory being <code>output</code>. If you need to specify a save path, you can set it through the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX abstracts away the concepts of dynamic graph weights and static graph weights from you. During model training, both dynamic and static graph weights will be generated. By default, static graph weights are used for model inference.</li>
<li>
<p>After model training is completed, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including the following:</p>
</li>
<li>
<p><code>train_result.json</code>: The training result record file, which logs whether the training task was completed normally, as well as the weight metrics, related file paths, etc.</p>
</li>
<li><code>train.log</code>: The training log file, which records the changes in model metrics and loss during the training process.</li>
<li><code>config.yaml</code>: The training configuration file, which records the hyperparameter settings for this training session.</li>
<li><code>.pdparams</code>, <code>.pdopt</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: Model weight-related files, including network parameters, optimizer, static graph network parameters, and static graph network structure, etc.</li>
</ul></details>

## <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. With PaddleX, model evaluation can be completed with a single command:

```bash
python main.py -c paddlex/configs/modules/3d_bev_detection/BEVFusion.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/nuscenes_demo \
```

Similar to model training, the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `bevf_pp_2x8_1x_nusc.yaml`)
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path of the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For details, please refer to [PaddleX General Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Information (Click to Expand)</b></summary>

<p>When evaluating the model, the path of the model weight file needs to be specified. Each configuration file has a default weight save path built-in. If you need to change it, you can simply set it by appending a command-line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After the model evaluation is completed, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results. Specifically, it records whether the evaluation task was completed normally and the evaluation metrics of the model, including mAP and NDS.</p></details>

### <b>4.4 Model Inference and Model Integration</b>
After completing the training and evaluation of the model, you can use the trained model weights for inference prediction or integrate them into Python.

#### 4.4.1 Model Inference

* To perform inference prediction via the command line, you only need the following command. Before running the code below, please download the [sample data] (<url id="cui26flkfv3k7i3hp2og" type="url" status="failed" title="" wc="0">https://paddle-model-ecology.bj.bcebos.com/paddlex/det_3d/demo_det_3d/nuscenes_demo_infer.tar) to your local machine.</url> Note: The link above may not be accessible due to network issues or an invalid URL. Please check the validity of the link and try again if necessary.

```bash
python main.py -c paddlex/configs/modules/3d_bev_detection/BEVFusion.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="nuscenes_demo_infer.tar"
```

Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `bevf_pp_2x8_1x_nusc.yaml`)
* Specify the mode for model inference prediction: `-o Global.mode=predict`
* Specify the model weight path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`

Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX production line or into your own project.

1.<b>Production Line Integration</b>

The 3D multimodal fusion detection module can be integrated into the 3D detection production line of PaddleX. Simply replacing the model path will complete the model update for the target detection module in the relevant production line. In production line integration, you can deploy your model using high-performance deployment and service-oriented deployment.

2.<b>Module Integration</b>

The weights you generate can be directly integrated into the 3D multimodal fusion detection module. You can refer to the Python example code in [Quick Integration](). Just replace the model with the path of the model you have trained.

