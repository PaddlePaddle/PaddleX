---
comments: true
---

# 3D Multi-modal Fusion Detection Pipeline Usage Tutorial

## 1. Introduction to 3D Multi-modal Fusion Detection Pipeline

The 3D multi-modal fusion detection pipeline supports input from multiple sensors (LiDAR, surround RGB cameras, etc.), processes the data using deep learning methods, and outputs information such as the position, shape, orientation, and category of objects in three-dimensional space. It has a wide range of applications in fields such as autonomous driving, robot navigation, and industrial automation.

BEVFusion is a multi-modal 3D object detection model that fuses surround camera images and LiDAR point cloud data into a unified Bird's Eye View (BEV) representation, aligning and fusing features from different sensors, overcoming the limitations of a single sensor, and significantly improving detection accuracy and robustness. It is suitable for complex scenarios such as autonomous driving.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/3d_bev_detection/01.png">

<b>The 3D multi-modal fusion detection pipeline includes a 3D multi-modal fusion detection module</b>，which contains <b>a BEVFusion model</b>. We provide benchmark data for this model:

<p><b>3D Multi-modal Fusion Detection Module:</b></p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>NDS</th>
<th>Description</th>
</tr>
<tr>
<td>BEVFusion</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BEVFusion_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BEVFusion_pretrained.pdparams">Training Model</a></td>
<td>53.9</td>
<td>60.9</td>
<td rowspan="2">BEVFusion is a multi-modal fusion detection model from a BEV perspective. It uses two branches to process data from different modalities, obtaining features for LiDAR and camera in the BEV perspective. The camera branch uses the LSS bottom-up approach to explicitly generate image BEV features, while the LiDAR branch uses a classic point cloud detection network. Finally, the BEV features from both modalities are aligned and fused, applied to the detection head or segmentation head.
</td>
</tr>
<tr>
</table>

<p>Note: The above accuracy metrics are for the <a href="https://www.nuscenes.org/nuscenes">nuscenes</a> validation set with mAP(0.5:0.95) and NDS 60.9, with an accuracy type of FP32.</p></details>

## 2. Quick Start

The pre-trained model pipelines provided by PaddleX allow for quick experimentation. You can experience the effects of the 3D multi-modal fusion detection pipeline online or locally using the command line or Python.

### 2.1 Online Experience

Online experience is currently not supported.

### 2.2 Local Experience
> ❗ Before using the 3D multi-modal fusion detection pipeline locally, please ensure you have completed the PaddleX wheel package installation according to [the PaddleX Installation Tutorial](../../../installation/installation.md).

Demo dataset download: You can use the following command to download the demo dataset to a specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/nuscenes_demo.tar -P ./data

tar -xf ./data/nuscenes_demo.tar -C ./data/
```

#### 2.2.1 Command Line Experience

You can quickly experience the 3D multi-modal fusion detection pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/det_3d/demo_det_3d/nuscenes_demo_infer.tar)，and  `--input` replace with the local path for prediction.
```bash
paddlex --pipeline 3d_bev_detection \
        --input nuscenes_demo_infer.tar \
        --device gpu:0
```
Parameter description:

```
--pipeline: The name of the pipeline, here it is the 3D multi-modal fusion detection pipeline.

--input: The input path to the .tar file containing image and lidar data to be processed. 3D multi-modal fusion detection pipeline is a multi-input pipeline depending on images, pointclouds and transition matrix information. Tar file contains "samples" directory with all images and pointclouds data, "sweeps" directories with pointclouds data of relative frames and nuscnes_infos_val.pkl file containing relataive data path from "samples" and "sweeps" directories and transition matrix infomation.

--device: The GPU index to be used (e.g., gpu:0 means using the 0th GPU, gpu:1,2 means using the 1st and 2nd GPUs), or you can choose to use CPU (--device cpu).
```

After running, the results will be printed on the terminal as follows:

```bash
{"res":
  {
    'input_path': 'samples/LIDAR_TOP/n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984253447765.pcd.bin',
    'sample_id': 'b4ff30109dd14c89b24789dc5713cf8c',
    'input_img_paths': [
      'samples/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984253404844.jpg',
      'samples/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984253412460.jpg',
      'samples/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984253420339.jpg',
      'samples/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984253427893.jpg',
      'samples/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984253437525.jpg',
      'samples/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984253447423.jpg'
    ]
    "boxes_3d": [
        [
            14.5425386428833,
            22.142045974731445,
            -1.2903141975402832,
            1.8441576957702637,
            4.433370113372803,
            1.7367216348648071,
            6.367165565490723,
            0.0036598597653210163,
            -0.013568558730185032
        ]
    ],
    "labels_3d": [
        0
    ],
    "scores_3d": [
        0.9920279383659363
    ]
  }
}
```

The meanings of the result parameters are as follows:

- `input_path`: Indicates the path to the input point cloud data of the sample to be predicted.
- `sample_id`: Indicates the unique identifier of the input sample to be predicted.
- `input_img_paths`: Indicates the paths to the input image data of the sample to be predicted.
- `boxes_3d`: Represents all the predicted bounding box information for the 3D sample. Each bounding box information is a list of length 9, with each element representing:
  - 0: x-coordinate of the center point
  - 1: y-coordinate of the center point
  - 2: z-coordinate of the center point
  - 3: Width of the detection box
  - 4: Length of the detection box
  - 5: Height of the detection box
  - 6: Rotation angle
  - 7: Velocity in the x-direction of the coordinate system
  - 8: Velocity in the y-direction of the coordinate system
- `labels_3d`: Represents the predicted categories corresponding to all the predicted bounding boxes of the 3D sample.
- `scores_3d`: Represents the confidence levels corresponding to all the predicted bounding boxes of the 3D sample.

#### 2.2.2 Python Script Integration
* The above command line is for quick experience. Generally, in projects, integration through code is often required. You can complete quick inference of the pipeline with a few lines of code as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="3d_bev_detection")
output = pipeline.predict("nuscenes_demo_infer.tar")

for res in output:
    res.print()  ## Print the structured output of the prediction
    res.save_to_json("./output/")  ## Save the results to a json file
    res.visualize(save_path="./output/", show=True) ## 3D result visualization. If the runtime environment has a graphical interface, set `show=True`; otherwise, set it to `False`.
```

<b>Note: </b>  
1、To visualize 3D detection results, you need to install the open3d package first. The installation command is as follows:
```bash
pip install open3d
```
2、If the runtime environment does not have a graphical interface, visualization will not be possible, but the results will still be saved. You can run the script in an environment that supports a graphical interface to visualize the saved results:
```bash
python paddlex/inference/models/3d_bev_detection/visualizer_3d.py --save_path="./output/"
```

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/images/pipelines/3d_bev_detection/02.png">

In the above Python script, the following steps are executed:

（1）Call `create_pipeline` to instantiate the 3D multi-modal fusion detection pipeline object. Specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is a pipeline name, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for pipeline model inference. Supports: "gpu", "cpu".</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available when the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

（2）Call the `predict` method of the 3D multi-modal fusion detection pipeline object for inference prediction: The `predict` method parameter is `input`, used to input data to be predicted, supporting multiple input methods. Specific examples are as follows:

<table>
<thead>
<tr>
<th>Parameter Type</th>
<th>Parameter Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>str</td>
<td><b>tar file path</b>，e.g., <code>/root/data/nuscenes_demo_infer.tar</code></td>
</tr>
<tr>
<td>list</td>
<td><b>List</b>，list elements need to be data of the above type, e.g., <code>["/root/data/nuscenes_demo_infer1.tar", "/root/data/nuscenes_demo_infer2.tar"]</td>
</tr>
</tbody>
</table>

<p><b>Note: pkl files can be created according to the <a href="https://github.com/ADLab-AutoDrive/BEVFusion/blob/main/tools/create_data.py">script</a>.</b></p></details>

（3）Call the `predict` method to obtain prediction results: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are represented as a list of prediction results.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving as a json file, as follows:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Method Parameters</th>
</tr>
</thead>
<tbody>
<tr>
<td>print</td>
<td>Print results to the terminal</td>
<td><code>- format_json</code>：bool, whether to format the output content using json indentation, default is True;<br/><code>- indent</code>：int, json formatting setting, only effective when format_json is True, default is 4;<br/><code>- ensure_ascii</code>：bool, json formatting setting, only effective when format_json is True, default is False;</td>
</tr>
<tr>
<td>save_to_json</td>
<td>Save the results as a json format file</td>
<td><code>- save_path</code>：str, the file path to save, when it is a directory, the saved file naming is consistent with the input file type naming;<br/><code>- indent</code>：int, json formatting setting, default is 4;<br/><code>- ensure_ascii</code>：bool, json formatting setting, default is False;</td>
</tr>
</tbody>
</table>

Additionally, you can obtain the 3D multi-modal fusion detection pipeline configuration file and load it for prediction. You can execute the following command to save the result in `my_path`:

```
paddlex --get_pipeline_config 3d_bev_detection --save_path ./my_path
```

If you have obtained the configuration file, you can customize various configurations of the 3D multi-modal fusion detection pipeline. Simply modify the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file. An example is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/3d_bev_detection.yaml")

output = pipeline.predict("nuscenes_demo_infer.tar")

for res in output:
    res.print()  ## Print the structured output of the prediction
    res.save_to_json("./output/")  ## Save the results to a json file
    res.visualize(save_path="./output/", show=True) ## 3D result visualization. If the runtime environment has a graphical interface, set `show=True`; otherwise, set it to `False`.
```

<b>Note: </b>The parameters in the configuration file are pipeline initialization parameters. If you want to change the 3D multi-modal fusion detection pipeline initialization parameters, you can directly modify the parameters in the configuration file and load the configuration file for prediction. At the same time, CLI prediction also supports passing in a configuration file by specifying the path with `--pipeline`.

## 3. Development Integration/Deployment
If the 3D multi-modal fusion detection pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the general 3D multi-modal fusion detection pipeline directly to your Python project, you can refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

In addition, PaddleX also provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent performance indicators (especially response speed) for deployment strategies to ensure efficient system operation and smooth user experience. Therefore, PaddleX provides a high-performance inference plugin aimed at deeply optimizing model inference and pre/post-processing to achieve significant speedups in the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

☁️ <b>Service Deployment</b>: Service deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports multiple pipeline service deployment solutions. For detailed pipeline service deployment procedures, please refer to the [PaddleX Service Deployment Guide](../../../pipeline_deploy/serving.md).

<details><summary>API Reference</summary>

<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>The result of the operation.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the attributes of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Same as the response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Perform 3D multi-modal fusion detection.</p>
<p><code>POST /bev-3d-object-detection</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>tar</code></td>
<td><code>string</code></td>
<td>The URL or path of the tar file accessible by the server.</td>
<td>Yes</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> in the response body has the following attributes:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>detectedObjects</code></td>
<td><code>array</code></td>
<td>Information about the location, category, and other details of the detected objects.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>detectedObjects</code> is an <code>object</code> with the following attributes:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>A list of length 9: 0: x-coordinate of the center point, 1: y-coordinate of the center point, 2: z-coordinate of the center point, 3: width of the detection box, 4: length of the detection box, 5: height of the detection box, 6: rotation angle, 7: velocity in the x-direction of the coordinate system, 8: velocity in the y-direction of the coordinate system</td>
</tr>
<tr>
<td><code>categoryId</code></td>
<td><code>integer</code></td>
<td>The category ID of the detected object.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The score of the detected object.</td>
</tr>
</tbody>
</table>
</details>

<details><summary>Multi-language Service Invocation Examples</summary>

<details>
<summary>Python</summary>

<pre><code class="language-python">
import requests

API_URL = &quot;http://localhost:8080/bev-3d-object-detection&quot; # Service URL
tar_path = &quot;./nuscenes_demo_infer.tar&quot;

payload = {&quot;tar&quot;: tar_path}

# Call the API
response = requests.post(API_URL, json=payload)

# Process the response data
assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
print(&quot;Detected objects:&quot;)
print(result[&quot;detectedObjects&quot;])
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method of placing computing and data processing functions on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.md).

You can choose an appropriate deployment method for your model pipeline based on your needs, and then proceed with subsequent AI application integration.

## 4. Secondary Development
If the default model weights provided by the 3D multi-modal fusion detection pipeline do not meet your requirements for accuracy or speed in your scenario, you can attempt to further <b>fine-tune</b> the existing model using <b>your own data from specific domains or application scenarios</b> to improve the recognition performance of the 3D multi-modal fusion detection pipeline in your scenario.

### 4.1 Model Fine-Tuning

Refer to the [Secondary Development](../../../module_usage/tutorials/cv_modules/3d_bev_detection.md#四二次开发) section in the [3D Multi-modal Fusion Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/3d_bev_detection.md) and use your private dataset to fine-tune the model.

### 4.2 Model Application
After completing fine-tuning training using your private dataset, you will obtain local model weight files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file, replacing the local path of the fine-tuned 3D object detection model to the corresponding position in the pipeline configuration file:

```bash
......
Pipeline:
  device: "gpu:0"
  det_model: "BEVFusion"        # Can be modified to the local path of the fine-tuned 3D object detection model
  det_batch_size: 1
  device: gpu
......
```
Subsequently, refer to the command line or Python script methods in [2.2 Local Experience](#22-local-experience) and load the modified pipeline configuration file.

##  5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPUs, Ascend NPUs, and Cambricon MLUs. <b>Simply modify the `--device` parameter</b> to achieve seamless switching between different hardware.

For example, when running the 3D multi-modal fusion detection pipeline in Python, to change the running device from an NVIDIA GPU to an Ascend NPU, you only need to modify the `device` in the script to npu:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="3d_bev_detection",
    device="npu:0" # gpu:0 --> npu:0
    )
```
If you want to use the 3D multi-modal fusion detection pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Use Guide](../../../other_devices_support/multi_devices_use_guide.md).
