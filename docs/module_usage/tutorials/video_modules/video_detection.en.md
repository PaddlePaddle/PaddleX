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
<td>YOWO</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOWO_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOWO_pretrained.pdparams">训练模型</a></td>
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
model = create_model(model_name="YOWO")
output = model.predict(input="HorseRiding.avi", batch_size=1)
for res in output:
    res.print()
    res.save_to_video(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

<details><summary>👉 <b>After running, the result is: (Click to expand)</b></summary>

```bash
{'res': {'input_path': 'HorseRiding.avi', 'result': [[[[110, 40, 170, 171], 0.8385784886274905, 'HorseRiding']], [[[112, 31, 168, 167], 0.8587647461352432, 'HorseRiding']], [[[106, 28, 164, 165], 0.8579590929730969, 'HorseRiding']], [[[106, 24, 165, 171], 0.8743957465404151, 'HorseRiding']], [[[107, 22, 165, 172], 0.8488322619908999, 'HorseRiding']], [[[112, 22, 173, 171], 0.8446755521458691, 'HorseRiding']], [[[115, 23, 177, 176], 0.8454028365262367, 'HorseRiding']], [[[117, 22, 178, 179], 0.8484261880748285, 'HorseRiding']], [[[117, 22, 181, 181], 0.8319480115446183, 'HorseRiding']], [[[117, 39, 182, 183], 0.820551099084625, 'HorseRiding']], [[[117, 41, 183, 185], 0.8202395831914338, 'HorseRiding']], [[[121, 47, 185, 190], 0.8261058921745246, 'HorseRiding']], [[[123, 46, 188, 196], 0.8307278306829033, 'HorseRiding']], [[[125, 44, 189, 197], 0.8259781361122833, 'HorseRiding']], [[[128, 47, 191, 195], 0.8227593229866699, 'HorseRiding']], [[[127, 44, 192, 193], 0.8205373129456817, 'HorseRiding']], [[[129, 39, 192, 185], 0.8223318812628619, 'HorseRiding']], [[[127, 31, 196, 179], 0.8501208612019866, 'HorseRiding']], [[[128, 22, 193, 171], 0.8315708410681566, 'HorseRiding']], [[[130, 22, 192, 169], 0.8318588228062005, 'HorseRiding']], [[[132, 18, 193, 170], 0.8310494469100611, 'HorseRiding']], [[[132, 18, 194, 172], 0.8302132445350239, 'HorseRiding']], [[[133, 18, 194, 176], 0.8339063714162727, 'HorseRiding']], [[[134, 26, 200, 183], 0.8365876380675275, 'HorseRiding']], [[[133, 16, 198, 182], 0.8395230321418268, 'HorseRiding']], [[[133, 17, 199, 184], 0.8198139782724922, 'HorseRiding']], [[[140, 28, 204, 189], 0.8344166596681291, 'HorseRiding']], [[[139, 27, 204, 187], 0.8412694521771158, 'HorseRiding']], [[[139, 28, 204, 185], 0.8500098862888805, 'HorseRiding']], [[[135, 19, 199, 179], 0.8506627974981384, 'HorseRiding']], [[[132, 15, 201, 178], 0.8495054272547193, 'HorseRiding']], [[[136, 14, 199, 173], 0.8451630721500223, 'HorseRiding']], [[[136, 12, 200, 167], 0.8366456814214907, 'HorseRiding']], [[[133, 8, 200, 168], 0.8457252233401213, 'HorseRiding']], [[[131, 7, 197, 162], 0.8400586356358062, 'HorseRiding']], [[[131, 8, 195, 163], 0.8320492682901985, 'HorseRiding']], [[[129, 4, 194, 159], 0.8298043752822792, 'HorseRiding']], [[[127, 5, 194, 162], 0.8348390851948722, 'HorseRiding']], [[[125, 7, 190, 164], 0.8299688814865505, 'HorseRiding']], [[[125, 6, 191, 164], 0.8303107088154711, 'HorseRiding']], [[[123, 8, 190, 168], 0.8348342187965798, 'HorseRiding']], [[[125, 14, 189, 170], 0.8356523950497134, 'HorseRiding']], [[[127, 18, 191, 171], 0.8392671764931521, 'HorseRiding']], [[[127, 30, 193, 178], 0.8441704160826191, 'HorseRiding']], [[[128, 18, 190, 181], 0.8438125326146775, 'HorseRiding']], [[[128, 12, 189, 186], 0.8390128962093542, 'HorseRiding']], [[[129, 15, 190, 185], 0.8471056476788448, 'HorseRiding']], [[[129, 16, 191, 184], 0.8536121834731034, 'HorseRiding']], [[[129, 16, 192, 185], 0.8488154629800881, 'HorseRiding']], [[[128, 15, 194, 184], 0.8417711698421471, 'HorseRiding']], [[[129, 13, 195, 187], 0.8412510238991473, 'HorseRiding']], [[[129, 14, 191, 187], 0.8404350980083457, 'HorseRiding']], [[[129, 13, 190, 189], 0.8382891279858882, 'HorseRiding']], [[[129, 11, 187, 191], 0.8318282305903217, 'HorseRiding']], [[[128, 8, 188, 195], 0.8043430817880264, 'HorseRiding']], [[[131, 25, 193, 199], 0.826184954516826, 'HorseRiding']], [[[124, 35, 191, 203], 0.8270462809459467, 'HorseRiding']], [[[121, 38, 191, 206], 0.8350931715324705, 'HorseRiding']], [[[124, 41, 195, 212], 0.8331239341053625, 'HorseRiding']], [[[128, 42, 194, 211], 0.8343046153103657, 'HorseRiding']], [[[131, 40, 192, 203], 0.8309784496027532, 'HorseRiding']], [[[130, 32, 195, 202], 0.8316640083647542, 'HorseRiding']], [[[135, 30, 196, 197], 0.8272172409555161, 'HorseRiding']], [[[131, 16, 197, 186], 0.8388410406147955, 'HorseRiding']], [[[134, 15, 202, 184], 0.8485738297037244, 'HorseRiding']], [[[136, 15, 209, 182], 0.8529430205135213, 'HorseRiding']], [[[134, 13, 218, 182], 0.8601191479922718, 'HorseRiding']], [[[144, 10, 213, 183], 0.8591963099263467, 'HorseRiding']], [[[151, 12, 219, 184], 0.8617965108346937, 'HorseRiding']], [[[151, 10, 220, 186], 0.8631923599955371, 'HorseRiding']], [[[145, 10, 216, 186], 0.8800860885204287, 'HorseRiding']], [[[144, 10, 216, 186], 0.8858840451538228, 'HorseRiding']], [[[146, 11, 214, 190], 0.8773644144886106, 'HorseRiding']], [[[145, 24, 214, 193], 0.8605544385867248, 'HorseRiding']], [[[146, 23, 214, 193], 0.8727294882672254, 'HorseRiding']], [[[148, 22, 212, 198], 0.8713131467067079, 'HorseRiding']], [[[146, 29, 213, 197], 0.8579099324651196, 'HorseRiding']], [[[154, 29, 217, 199], 0.8547794072847914, 'HorseRiding']], [[[151, 26, 217, 203], 0.8641733722316758, 'HorseRiding']], [[[146, 24, 212, 199], 0.8613466257602624, 'HorseRiding']], [[[142, 25, 210, 194], 0.8492670944810214, 'HorseRiding']], [[[134, 24, 204, 192], 0.8428117300203049, 'HorseRiding']], [[[136, 25, 204, 189], 0.8486779356971397, 'HorseRiding']], [[[127, 21, 199, 179], 0.8513896296400709, 'HorseRiding']], [[[125, 10, 192, 192], 0.8510201771386576, 'HorseRiding']], [[[124, 8, 191, 192], 0.8493999019508465, 'HorseRiding']], [[[121, 8, 192, 193], 0.8487097098892171, 'HorseRiding']], [[[119, 6, 187, 193], 0.847543279648022, 'HorseRiding']], [[[118, 12, 190, 190], 0.8503535936620565, 'HorseRiding']], [[[122, 22, 189, 194], 0.8427901493276977, 'HorseRiding']], [[[118, 24, 188, 195], 0.8418835400352087, 'HorseRiding']], [[[120, 25, 188, 205], 0.847192725785284, 'HorseRiding']], [[[122, 25, 189, 207], 0.8444105420674433, 'HorseRiding']], [[[120, 23, 189, 208], 0.8470784016639392, 'HorseRiding']], [[[121, 23, 188, 205], 0.843428111269418, 'HorseRiding']], [[[117, 23, 186, 206], 0.8420809714166708, 'HorseRiding']], [[[119, 5, 199, 197], 0.8288265053231356, 'HorseRiding']], [[[121, 8, 192, 195], 0.8197548738023599, 'HorseRiding']]]}}
```

Parameter meanings are as follows:
- `input_path`: Represents the path of the input video to be predicted.
- `result`: The target bounding box information of the video frames, which is a list. Each element in the list represents the detection result of an image frame, which is still a list. Each element contains the information of a detection box, which are:
  - <code>[xmin, ymin, xmax, ymax]</code>: The coordinates of the top-left and bottom-right corners of the bounding box, both are floating-point numbers.
  - float: The confidence score of the bounding box, a floating-point number.
  - string: The category to which the bounding box belongs, a string.

</details>

The visualized video is as follows:
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/video_detection/HorseRiding_res.jpg" />

In the above Python script, the following steps are executed:

* `create_model` instantiates the video detection model (`YOWO`), with specific explanations as follows:

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
<tr>
<td><code>nms_thresh</code></td>
<td>The IoU threshold parameter in the Non-Maximum Suppression (NMS) process; if not specified, the default configuration of the PaddleX official model will be used</td>
<td><code>float/None</code></td>
<td>Floating-point number greater than 0/None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>score_thresh</code></td>
<td>The prediction confidence threshold; if not specified, the default configuration of the PaddleX official model will be used</td>
<td><code>float/None</code></td>
<td>Floating-point number greater than 0/None</td>
<td><code>None</code></td>
</tr>
</table>

* The `predict` method of the video classification model is called to perform inference and prediction. The parameter of the `predict` method is `input`, which is used to input the data to be predicted and supports multiple input types, with specific explanations as follows:

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
  <li><b>Python variable</b>, such as a <code>str</code> representing the local path of a video file</li>
  <li><b>File path</b>, such as the local path of a video file: <code>/root/data/video.mp4</code></li>
  <li><b>URL link</b>, such as the network URL of a video file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4">Example</a></li>
  <li><b>Local directory</b>, the directory should contain data files to be predicted, such as the local path: <code>/root/data/</code></li>
  <li><b>List</b>, elements of the list should be of the above types, such as <code>[\"/root/data/video1.mp4\", \"/root/data/video2.mp4\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size</td>
<td><code>int</code></td>
<td>None</td>
<td>1</td>
</tr>
<tr>
<td><code>nms_thresh</code></td>
<td>The IoU threshold parameter in the Non-Maximum Suppression (NMS) process</td>
<td><code>float|None</code></td>
<td>
<ul>
  <li><b>float</b>: Floating-point number greater than 0;</li>
  <li><b>None</b>: If set to <code>None</code>, the default value initialized in the production environment will be used, initialized to 0.4;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>score_thresh</code></td>
<td>Prediction confidence threshold</td>
<td><code>float|None</code></td>
<td>
<ul>
  <li><b>float</b>: Floating-point number greater than 0;</li>
  <li><b>None</b>: If set to <code>None</code>, the default value initialized in the production environment will be used, initialized to 0.8;</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</table>

* Process the prediction results. Each sample's prediction result is a corresponding Result object, and it supports operations such as printing, saving as an image, and saving as a `json` file:

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
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. When it is a directory, the saved file name is consistent with the input file name</td>
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
<td><code>save_to_video()</code></td>
<td>Save the result as a video file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The path to save the file. When it is a directory, the saved file name is consistent with the input file name</td>
<td>None</td>
</tr>
</table>

* Additionally, it supports obtaining the result visualization video and `json` results through attributes:

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
<tr>
<td rowspan = "1"><code>video</code></td>
<td rowspan = "1">Get the visualization video and frame rate in <code>dict</code> format. Here, the visualization video is an np.array array with dimensions (number of video frames, video height, video width, number of video channels)</td>
</tr>
</table>

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
python main.py -c paddlex/configs/modules/video_detection/YOWO.yaml \
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
python main.py -c paddlex/configs/modules/video_det_examples/YOWO.yaml  \
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
python main.py -c  paddlex/configs/modules/video_detection/YOWO.yaml  \
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
python main.py -c paddlex/configs/modules/video_detection/YOWO.yaml \
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
