---
comments: true
---
# Human Keypoint Detection Pipeline User Guide

## 1. Introduction to Human Keypoint Detection Pipeline

Human keypoint detection aims to analyze human posture and movements by identifying and locating specific joints and parts of the human body. This task requires not only detecting humans in images but also accurately obtaining the positions of keypoints such as shoulders, elbows, knees, etc., for pose estimation and behavior recognition. Human keypoint detection is widely used in sports analysis, health monitoring, animation production, and human-computer interaction.

PaddleX's Human Keypoint Detection Pipeline is a Top-Down solution consisting of pedestrian detection and keypoint detection modules, optimized for mobile devices. It can accurately and smoothly perform multi-person pose estimation tasks on mobile devices.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/human_keypoint_detection/01.jpg"/>
<b>The Human Keypoint Detection Pipeline includes pedestrian detection and human keypoint detection modules</b>, with several models available. You can choose the model based on the benchmark data below. <b>If you prioritize model accuracy, choose a model with higher accuracy; if you prioritize inference speed, choose a model with faster inference speed; if you prioritize storage size, choose a model with a smaller storage size</b>.

<details><summary> 👉Model List Details</summary>
<b>Pedestrian Detection Module:</b>
<table>
<tr>
<th>Model</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-YOLOE-L_human</td>
<td>48.0</td>
<td>81.9</td>
<td>33.27 / 9.19</td>
<td>173.72 / 173.72</td>
<td>196.02</td>
<td rowspan="2">Pedestrian detection model based on PP-YOLOE</td>
</tr>
<tr>
<td>PP-YOLOE-S_human</td>
<td>42.5</td>
<td>77.9</td>
<td>9.94 / 3.42</td>
<td>54.48 / 46.52</td>
<td>28.79</td>
</tr>
</table>
<b>Note: The above accuracy metrics are based on the CrowdHuman dataset mAP(0.5:0.95). All model GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision, and CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>
<b>Human Keypoint Detection Module:</b>
<table>
<tr>
<th>Model</th>
<th>Solution</th>
<th>Input Size</th>
<th>AP(0.5:0.95)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-TinyPose_128x96</td>
<td>Top-Down</td>
<td>128*96</td>
<td>58.4</td>
<td></td>
<td></td>
<td>4.9</td>
<td rowspan="2">PP-TinyPose is a real-time keypoint detection model developed by Baidu PaddlePaddle Vision Team, optimized for mobile devices, capable of smoothly performing multi-person pose estimation tasks on mobile devices</td>
</tr>
<tr>
<td>PP-TinyPose_256x192</td>
<td>Top-Down</td>
<td>256*192</td>
<td>68.3</td>
<td></td>
<td></td>
<td>4.9</td>
</tr>
</table>
<b>Note: The above accuracy metrics are based on the COCO dataset AP(0.5:0.95), with detection boxes obtained from ground truth annotations. All model GPU inference times are based on NVIDIA Tesla T4 machines with FP32 precision, and CPU inference speeds are based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>
</details>

## 2. Quick Start

The pre-trained model pipelines provided by PaddleX can be quickly experienced. You can use Python locally to experience the effects of the general image recognition pipeline.

### 2.1 Online Experience

Not supported for online experience.

### 2.2 Local Experience

> ❗ Before using the Human Keypoint Detection Pipeline locally, please ensure that you have completed the installation of the PaddleX wheel package according to the [PaddleX Installation Guide](../../../installation/installation.en.md).

#### 2.2.1 Command Line Experience

You can quickly experience the Human Keypoint Detection Pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/keypoint_detection_001.jpg) and replace `--input` with your local path for prediction.

Due to network issues, the above web page parsing was not successful. If you need the content of the web page, please check the validity of the web page link and try again. If you do not need the parsing of this link, you can proceed with other questions.

```bash
paddlex --pipeline human_keypoint_detection \
        --input keypoint_detection_001.jpg \
        --det_threshold 0.5 \
        --save_path ./output/ \
        --device gpu:0
```

The relevant parameter descriptions and results explanations can be referred to in the parameter explanations and results explanations of [2.2.2 Integration via Python Script](#222-integration-via-python-script).

The visualization results are saved to `save_path`, as shown below:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/human_keypoint_detection/01.jpg"/>

#### 2.2.2 Integration via Python Script
The above command line method allows you to quickly experience and view the results. In a project, code integration is often required. You can complete the quick inference of the pipeline with the following lines of code:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="human_keypoint_detection")

output = pipeline.predict("keypoint_detection_001.jpg"， det_threshold=0.5)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

In the above Python script, the following steps are executed:

(1) Call the `create_pipeline` function to instantiate a pipeline object. The specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is a pipeline name, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>config</code></td>
<td>Specific configuration information for the pipeline (if set simultaneously with the <code>pipeline</code>, it takes precedence over the <code>pipeline</code>, and the pipeline name must match the <code>pipeline</code>).
</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for pipeline inference. It supports specifying the specific card number of GPU, such as "gpu:0", other hardware card numbers, such as "npu:0", or CPU, such as "cpu".</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference. This is only available if the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

(2) Call the `predict()` method of the human keypoint detection pipeline object for inference prediction. This method returns a `generator`. Below are the parameters and their descriptions for the `predict()` method:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>Data to be predicted. It supports multiple input types and is a required parameter.</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code>.</li>
<li><b>str</b>: Local path of an image file, such as <code>/root/data/img.jpg</code>; <b>URL link</b>, such as a network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png">example</a>; <b>Local directory</b>, which should contain images to be predicted, such as <code>/root/data/</code>.</li>
<li><b>List</b>: Elements of the list must be of the above types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, or <code>[\"/root/data1\", \"/root/data2\"]</code>.</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for the human detection model.</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>: For example, <code>0.5</code> means filtering out all bounding boxes with a score lower than <code>0.5</code>.</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used, which is <code>0.5</code>.</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

(3) Process the prediction results. The prediction result for each sample is of type `dict`, and supports operations such as printing, saving as an image, and saving as a `json` file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will match the input file name</td>
<td>N/A</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. This is only effective when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non-<code>ASCII</code> characters are escaped to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. This is only effective when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the result as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting both directory and file paths</td>
<td>N/A</td>
</tr>
</table>
<ul><li><details><summary>👉 Calling the <code>print()</code> method will print the following result to the terminal (click to expand):</summary>

```bash
{'res': {'input_path': 'keypoint_detection_001.jpg', 'boxes': [{'coordinate': [325.65088, 74.46718, 391.5512, 209.46529], 'det_score': 0.9316536784172058, 'keypoints': array([[3.52227936e+02, 8.88497543e+01, 5.41676700e-01],
       [3.51974579e+02, 8.66366196e+01, 6.21515572e-01],
       [3.52865662e+02, 8.64902344e+01, 5.60755610e-01],
       [3.50862457e+02, 8.75393066e+01, 5.66961825e-01],
       [3.57415802e+02, 8.63235092e+01, 5.54250121e-01],
       [3.39434937e+02, 9.99389191e+01, 6.28665030e-01],
       [3.69297821e+02, 1.01169853e+02, 6.00828469e-01],
       [3.36788544e+02, 1.18020966e+02, 5.31029820e-01],
       [3.71721039e+02, 1.22033646e+02, 6.07613802e-01],
       [3.66371948e+02, 1.40476746e+02, 1.52681962e-01],
       [3.67885651e+02, 1.39969559e+02, 3.90044987e-01],
       [3.47553253e+02, 1.42274353e+02, 5.21435857e-01],
       [3.60833710e+02, 1.42503479e+02, 4.64817137e-01],
       [3.40133362e+02, 1.67570465e+02, 5.92474759e-01],
       [3.74433594e+02, 1.69982712e+02, 5.48423827e-01],
       [3.37616333e+02, 1.92737564e+02, 5.85887253e-01],
       [3.82684723e+02, 1.96479385e+02, 6.19615853e-01]], dtype=float32), 'kpt_score': 0.53462815}, {'coordinate': [271.96713, 69.02892, 336.77832, 217.54662], 'det_score': 0.9304604530334473, 'keypoints': array([[2.96400787e+02, 8.58611679e+01, 7.14319646e-01],
       [2.97301758e+02, 8.28755493e+01, 7.04051554e-01],
       [2.94497406e+02, 8.29398193e+01, 6.89844370e-01],
       [3.00162109e+02, 8.35247955e+01, 5.55446565e-01],
       [2.93188751e+02, 8.33744202e+01, 6.51386738e-01],
       [3.12854675e+02, 9.81457520e+01, 7.32430100e-01],
       [2.86463226e+02, 1.01262375e+02, 6.10454917e-01],
       [3.19350311e+02, 1.18383713e+02, 5.93547344e-01],
       [2.82401520e+02, 1.21164886e+02, 5.71586847e-01],
       [3.23966248e+02, 1.39364532e+02, 5.18607676e-01],
       [2.82263916e+02, 1.44509521e+02, 2.59432912e-01],
       [3.09791840e+02, 1.43603912e+02, 6.89817309e-01],
       [2.94868561e+02, 1.44677597e+02, 5.97069323e-01],
       [3.14365845e+02, 1.74088943e+02, 6.36058152e-01],
       [2.92653442e+02, 1.75070770e+02, 5.97140312e-01],
       [3.19849792e+02, 2.02162598e+02, 6.80035114e-01],
       [2.94255920e+02, 2.03049500e+02, 6.08293772e-01]], dtype=float32), 'kpt_score': 0.6123249}, {'coordinate': [293.55933, 188.65804, 419.47382, 305.4712], 'det_score': 0.9179267883300781, 'keypoints': array([[3.33173096e+02, 2.05474487e+02, 5.18341482e-01],
       [3.36098663e+02, 2.03492996e+02, 5.60671568e-01],
       [3.37248505e+02, 2.03364868e+02, 5.00729203e-01],
       [3.40604095e+02, 2.02393539e+02, 4.98033434e-01],
       [3.43625671e+02, 2.01539536e+02, 6.13991261e-01],
       [3.24516022e+02, 2.18521667e+02, 3.14208776e-01],
       [3.52951965e+02, 2.12051971e+02, 4.42923039e-01],
       [3.14448853e+02, 2.22776672e+02, 2.55664617e-01],
       [3.65774384e+02, 2.25498718e+02, 8.26140717e-02],
       [3.06869843e+02, 2.34243729e+02, 1.35185301e-01],
       [3.08855865e+02, 2.36824249e+02, 1.00039296e-01],
       [3.82195679e+02, 2.42062302e+02, 5.45506418e-01],
       [3.88757233e+02, 2.42933960e+02, 5.79574823e-01],
       [3.50280792e+02, 2.56009766e+02, 7.92343557e-01],
       [3.70955750e+02, 2.74127930e+02, 5.06902397e-01],
       [3.61553833e+02, 2.83896454e+02, 6.03924632e-01],
       [3.94064087e+02, 2.88825836e+02, 3.72752368e-01]], dtype=float32), 'kpt_score': 0.43667096}, {'coordinate': [238.98825, 182.67476, 372.81628, 307.61395], 'det_score': 0.914400041103363, 'keypoints': array([[281.63354   , 197.61014   ,   0.76263565],
       [283.38297   , 195.05458   ,   0.8535259 ],
       [277.73865   , 192.96712   ,   0.7486459 ],
       [282.2258    , 197.37396   ,   0.5293759 ],
       [267.41278   , 193.56656   ,   0.6177453 ],
       [269.7986    , 215.17424   ,   0.6587688 ],
       [259.27332   , 214.76183   ,   0.7745857 ],
       [277.53683   , 237.42062   ,   0.48790172],
       [260.1237    , 240.18477   ,   0.44012186],
       [293.51572   , 250.89894   ,   0.49827316],
       [290.2274    , 252.19504   ,   0.58322966],
       [279.096     , 260.06042   ,   0.6834831 ],
       [269.9528    , 265.9034    ,   0.74632865],
       [313.1393    , 260.79523   ,   0.6337413 ],
       [310.01425   , 262.5001    ,   0.7376388 ],
       [348.17996   , 283.56802   ,   0.6096815 ],
       [334.12622   , 292.06284   ,   0.805234  ]], dtype=float32), 'kpt_score': 0.6571127}, {'coordinate': [66.23172, 93.531204, 124.48463, 217.99655], 'det_score': 0.9086756110191345, 'keypoints': array([[ 91.31993   , 108.413284  ,   0.615049  ],
       [ 92.08924   , 106.03603   ,   0.63400346],
       [ 88.434235  , 105.775925  ,   0.6342117 ],
       [ 94.41964   , 106.27531   ,   0.5885308 ],
       [ 84.07658   , 105.80367   ,   0.6773151 ],
       [100.38561   , 118.80038   ,   0.74734527],
       [ 79.46563   , 119.58027   ,   0.7821885 ],
       [102.27228   , 136.0127    ,   0.5907435 ],
       [ 73.76375   , 135.51848   ,   0.7132327 ],
       [101.763245  , 148.3819    ,   0.35871926],
       [ 72.33199   , 148.83861   ,   0.5405212 ],
       [ 99.1199    , 151.6756    ,   0.83278877],
       [ 86.4599    , 152.03287   ,   0.78065455],
       [106.40269   , 176.46979   ,   0.75466657],
       [ 84.909454  , 178.44617   ,   0.76010597],
       [110.97942   , 201.19633   ,   0.7917331 ],
       [ 79.87372   , 202.87093   ,   0.79150075]], dtype=float32), 'kpt_score': 0.68195945}, {'coordinate': [160.1294, 78.35935, 212.01868, 153.2241], 'det_score': 0.8295672535896301, 'keypoints': array([[1.8115924e+02, 1.0371443e+02, 2.5303254e-01],
       [1.8318883e+02, 9.6959526e+01, 1.5748371e-01],
       [1.8303551e+02, 9.8071350e+01, 2.1673845e-01],
       [1.8769695e+02, 9.1113632e+01, 1.4220884e-01],
       [1.8665227e+02, 9.1424126e+01, 1.2998220e-01],
       [1.9881558e+02, 9.9527107e+01, 2.6830634e-01],
       [1.8619264e+02, 9.9051491e+01, 1.8842754e-01],
       [1.9972902e+02, 1.2386106e+02, 4.7812667e-01],
       [1.8038458e+02, 1.2146417e+02, 1.7550260e-01],
       [1.8155409e+02, 1.3735040e+02, 3.2514629e-01],
       [1.7971712e+02, 1.3371999e+02, 1.1313542e-01],
       [1.9606516e+02, 1.4140919e+02, 2.3604973e-01],
       [1.8650092e+02, 1.4260675e+02, 1.3515554e-01],
       [1.9617020e+02, 1.2273723e+02, 9.6943676e-02],
       [1.6671684e+02, 1.2564886e+02, 1.2711491e-01],
       [1.8317670e+02, 1.3923177e+02, 1.0834377e-01],
       [1.7997801e+02, 1.3850316e+02, 9.3360245e-02]], dtype=float32), 'kpt_score': 0.19088578}, {'coordinate': [52.482475, 59.36664, 96.47121, 135.45993], 'det_score': 0.7726763486862183, 'keypoints': array([[7.38075943e+01, 7.33277664e+01, 1.63257480e-01],
       [7.37732239e+01, 7.15934525e+01, 1.55248597e-01],
       [7.20166702e+01, 7.13588028e+01, 1.96659654e-01],
       [6.95982971e+01, 7.10932083e+01, 1.26999229e-01],
       [6.98693237e+01, 7.16391983e+01, 1.35578454e-01],
       [8.22228088e+01, 7.77278976e+01, 2.35379949e-01],
       [6.47475586e+01, 7.88423233e+01, 2.10787609e-01],
       [8.33889618e+01, 9.05893707e+01, 2.98420697e-01],
       [8.30510330e+01, 9.10888824e+01, 1.13309503e-01],
       [8.09216843e+01, 9.85093231e+01, 1.84670463e-01],
       [7.77405396e+01, 1.01128220e+02, 1.49517819e-01],
       [7.58817215e+01, 1.02311646e+02, 7.63842911e-02],
       [6.97640839e+01, 1.01978600e+02, 9.00617689e-02],
       [8.89746017e+01, 9.87928925e+01, 2.00097740e-01],
       [6.45541153e+01, 1.04130150e+02, 1.00787796e-01],
       [8.81069489e+01, 1.19858818e+02, 1.84717909e-01],
       [7.08108673e+01, 1.38108154e+02, 9.07213315e-02]], dtype=float32), 'kpt_score': 0.15956473}, {'coordinate': [7.081953, 80.3705, 46.81927, 161.72012], 'det_score': 0.6587498784065247, 'keypoints': array([[2.68747215e+01, 9.24635468e+01, 3.17601502e-01],
       [2.71188889e+01, 9.08834305e+01, 2.46545449e-01],
       [2.69905357e+01, 9.03851013e+01, 3.27626228e-01],
       [2.34424419e+01, 8.87997513e+01, 2.75899678e-01],
       [3.25261230e+01, 8.93143845e+01, 3.42958122e-01],
       [1.98818531e+01, 9.67693405e+01, 2.83849210e-01],
       [3.82729301e+01, 9.91002884e+01, 3.19851965e-01],
       [1.63669338e+01, 1.10157967e+02, 1.96907550e-01],
       [4.11151352e+01, 1.10147133e+02, 2.70823181e-01],
       [1.86983719e+01, 1.17376358e+02, 1.15746662e-01],
       [1.98090267e+01, 1.16526924e+02, 1.02475688e-01],
       [2.51145611e+01, 1.23519379e+02, 3.24807376e-01],
       [3.34709854e+01, 1.24678688e+02, 2.65269905e-01],
       [2.82129307e+01, 1.42196121e+02, 2.98054874e-01],
       [2.94088726e+01, 1.42497360e+02, 3.57838601e-01],
       [2.95637341e+01, 1.57201065e+02, 2.14882523e-01],
       [3.03766575e+01, 1.57535706e+02, 2.10423440e-01]], dtype=float32), 'kpt_score': 0.26303306}, {'coordinate': [126.131096, 30.263107, 168.5759, 134.09885], 'det_score': 0.6441988348960876, 'keypoints': array([[148.10135   ,  40.584896  ,   0.44685563],
       [150.00848   ,  38.423157  ,   0.5721373 ],
       [146.84933   ,  38.88104   ,   0.5450204 ],
       [153.57166   ,  38.53051   ,   0.58872294],
       [144.609     ,  38.854267  ,   0.54383296],
       [158.78825   ,  51.609245  ,   0.6847385 ],
       [141.20293   ,  49.92705   ,   0.52605945],
       [157.85371   ,  70.32525   ,   0.7879656 ],
       [136.42497   ,  68.15052   ,   0.4752547 ],
       [144.46915   ,  79.161385  ,   0.5807479 ],
       [135.84734   ,  75.86984   ,   0.32583416],
       [155.16513   ,  78.74157   ,   0.56849873],
       [141.66093   ,  77.88423   ,   0.45576522],
       [152.68689   , 100.64953   ,   0.5331878 ],
       [130.97485   ,  87.03784   ,   0.73304355],
       [153.57033   , 123.742294  ,   0.39946193],
       [132.91501   , 114.52923   ,   0.36085486]], dtype=float32), 'kpt_score': 0.5369401}, {'coordinate': [112.50212, 64.127, 150.35353, 125.85529], 'det_score': 0.5013833045959473, 'keypoints': array([[1.34417511e+02, 7.67317352e+01, 8.11196864e-02],
       [1.33561203e+02, 7.61824722e+01, 5.88811524e-02],
       [1.33302322e+02, 7.54709702e+01, 3.77583615e-02],
       [1.33238602e+02, 7.65276260e+01, 8.76586884e-02],
       [1.27832169e+02, 7.29993439e+01, 5.68802767e-02],
       [1.32234711e+02, 8.55900650e+01, 6.25893995e-02],
       [1.29263702e+02, 8.66081772e+01, 7.35298842e-02],
       [1.17821297e+02, 6.38808823e+01, 1.47604376e-01],
       [1.17373665e+02, 6.40265808e+01, 1.84295043e-01],
       [1.39441742e+02, 9.73589020e+01, 6.45915419e-02],
       [1.24748589e+02, 1.04544739e+02, 5.86665794e-02],
       [1.35098206e+02, 7.81749954e+01, 8.30232948e-02],
       [1.34638489e+02, 7.91068802e+01, 8.16871747e-02],
       [1.36119888e+02, 8.93165436e+01, 1.34096310e-01],
       [1.30067749e+02, 9.34937820e+01, 8.98712277e-02],
       [1.36004654e+02, 1.16780487e+02, 1.60800099e-01],
       [1.28087891e+02, 1.15956802e+02, 1.99588016e-01]], dtype=float32), 'kpt_score': 0.097802415}]}}
```

</details></li></ul>

- The output result parameters are as follows:
    - `input_path`: Indicates the path of the input image
    - `boxes`: Detected human body information, a list of dictionaries, each dictionary contains the following information:
        - `coordinate`: Coordinates of the human body target box, in the format [xmin, ymin, xmax, ymax]
        - `det_score`: Confidence score of the human body target box
        - `keypoints`: Keypoint coordinate information, a numpy array with shape [num_keypoints, 3], where each keypoint consists of [x, y, score], and score is the confidence score of the keypoint
        - `kpt_score`: Overall confidence score of the keypoints, which is the average confidence score of the keypoints

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.json`; if specified as a file, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, the `numpy.array` types will be converted to lists.
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If specified as a directory, the saved path will be `save_path/{your_img_basename}_res.{your_img_extension}`; if specified as a file, it will be saved directly to that file. (The production line usually contains many result images, it is not recommended to specify a specific file path directly, otherwise multiple images will be overwritten, leaving only the last image)

* Additionally, it also supports obtaining visualized images and prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the predicted <code>json</code> format result</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get the visualized image in <code>dict</code> format</td>
</tr>
</table>

- The prediction result obtained by the `json` attribute is a dict type of data, with content consistent with the content saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is a dictionary type of data. The key is `res`, and the corresponding value is an `Image.Image` object used for visualizing the human keypoint detection results.

The above Python script integration method uses the parameter settings from the PaddleX official configuration file by default. If you need to customize the configuration file, you can execute the following command to obtain the official configuration file and save it in `my_path`:

```bash
paddlex --get_pipeline_config human_keypoint_detection --save_path ./my_path
```

If you have obtained the configuration file, you can customize the settings for the human keypoint detection pipeline. Simply modify the value of the `pipeline` parameter in the `create_pipeline` method to the path of your custom pipeline configuration file.

For example, if your custom configuration file is saved at `./my_path/human_keypoint_detection.yaml`, you just need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/human_keypoint_detection.yaml")
output = pipeline.predict("keypoint_detection_001.jpg")
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

## 3. Development Integration/Deployment

If the human keypoint detection pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the general image recognition pipeline directly to your Python project, you can refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. For this purpose, PaddleX provides a high-performance inference plugin, aimed at deeply optimizing the performance of model inference and pre/post-processing, significantly accelerating the end-to-end process. For detailed high-performance inference procedures, please refer to [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.en.md).

☁️ <b>Service Deployment</b>: Service deployment is a common form of deployment in actual production environments. By encapsulating the inference function as a service, clients can access these services via network requests to obtain inference results. PaddleX supports multiple pipeline service deployment solutions. For detailed pipeline service deployment procedures, please refer to [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.en.md).

Below are the API references and multi-language service invocation examples for basic service deployment:

<details><summary>API Reference</summary>
</details>
<details><summary>Multi-language Service Invocation Examples</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/ocr" # Service URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# Base64 encode the local image
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64 encoded file content or image URL

# Call the API
response = requests.post(API_URL, json=payload)

# Process the API response
assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected texts:")
print(result["texts"])
</code></pre>
</details>
</details>


📱 <b>Edge Deployment</b>: Edge deployment is a method where computation and data processing functions are placed on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose the appropriate deployment method based on your needs to integrate the AI application subsequently.


## 4. Secondary Development

If the default model weights provided by the human keypoint detection pipeline do not meet your accuracy or speed requirements in your scenario, you can try further <b>fine-tuning</b> the existing model using <b>your own specific domain or application data</b> to improve the recognition performance of the pipeline in your scenario.

### 4.1 Model Fine-Tuning

Since the human keypoint detection pipeline consists of two modules (pedestrian detection module and human keypoint detection module), the suboptimal performance of the model pipeline may stem from either module.

You can analyze the images with poor recognition performance. If you find that many pedestrian targets are not detected during the analysis, it may indicate a deficiency in the pedestrian detection model. You need to refer to the [Pedestrian Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/human_detection.en.md) in the [Secondary Development](../../../module_usage/tutorials/cv_modules/human_detection.en.md) section to fine-tune the pedestrian detection model using your private dataset. If keypoint detection errors occur in detected pedestrians, it indicates that the keypoint detection model needs further improvement. You need to refer to the [Keypoint Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/human_keypoint_detection.en.md) in the [Secondary Development](../../../module_usage/tutorials/cv_modules/human_keypoint_detection.en.md#secondary-development) section to fine-tune the keypoint detection model.

### 4.2 Model Application

After completing the fine-tuning training with your private dataset, you will obtain a local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights in the corresponding position of the pipeline configuration file:

```yaml
pipeline_name: human_keypoint_detection

SubModules:
  ObjectDetection:
    module_name: object_detection
    model_name: PP-YOLOE-S_human
    model_dir: null # Can be modified to the local path of the fine-tuned human detection model
    batch_size: 1
    threshold: null
    img_size: null
  KeypointDetection:
    module_name: keypoint_detection
    model_name: PP-TinyPose_128x96
    model_dir: # Can be modified to the local path of the fine-tuned keypoint detection model
    batch_size: 1
    flip: False
    use_udp: null
```

Then, refer to the command-line method or Python script method in [2.2 Local Experience](#22-Local-Experience) to load the modified production configuration file.

## 5. Multi-Hardware Support

PaddleX supports a variety of mainstream hardware devices, including NVIDIA GPU, Kunlunxin XPU, Ascend NPU, and Cambricon MLU. <b>Simply modify the `--device` parameter</b> to seamlessly switch between different hardware.

For example, using Ascend NPU for fast inference of human keypoint detection in production:

```bash
paddlex --pipeline human_keypoint_detection \
        --input keypoint_detection_001.jpg \
        --det_threshold 0.5 \
        --save_path ./output/ \
        --device npu:0
```

If you want to use the general image recognition production line on more types of hardware, please refer to the <a href="../../../other_devices_support/multi_devices_use_guide.en.md">PaddleX Multi-Hardware Usage Guide</a>.
