---
comments: true
---

# 3D Object Detection Pipeline Tutorial

## 1. Introduction to 3D Object Detection Pipeline
The 3D object detection pipeline leverages 3D object detection technology to process data from various sensors (LIDAR, RGB cameras, etc.) using deep learning methods. It outputs information about the position, shape, orientation, and category of objects in 3D space. This pipeline is widely used in autonomous driving, robot navigation, and industrial automation.

BEVFusion is a multi-modal 3D object detection framework that fuses camera images and LiDAR point cloud data into a unified Bird's Eye View (BEV) representation. This overcomes the limitations of single sensors and significantly improves detection accuracy and robustness, making it suitable for complex scenarios like autonomous driving.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/3d_bev_detection/01.png">

<b>The 3D object detection pipeline includes a 3D multi-modal fusion detection module</b>，which contains a BEVFusion model. We provide benchmark data for this model:：

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>NDS</th>
<th>Introduction</th>
</tr>
<tr>
<td>BEVFusion</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/BEVFusion_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BEVFusion_pretrained.pdparams">Training Model</a></td>
<td>53.9</td>
<td>60.9</td>
<td rowspan="2">BEVFusion is a multi-modal fusion model in BEV perspective. It processes data from different modalities in two branches to obtain lidar and camera features in BEV perspective. The camera branch uses LSS, a bottom-up approach, to explicitly generate image BEV features. The lidar branch uses a classic point cloud detection network. Finally, it aligns and fuses the BEV features from both modalities for head detection or segmentation.
</td>
</tr>
<tr>
</table>

<p><b>Note: The above accuracy indicators are for the <a href="https://www.nuscenes.org/nuscenes">nuscenes</a> validation set with mAP(0.5:0.95) of 53.9 and NDS of 60.9. The accuracy type is FP32.</b></p></details>

## 2. Quick Start
PaddleX provides pre-trained models that can be quickly experienced. You can experience the 3D object detection pipeline online or locally using command line or Python.

### 2.1 Online Experience

Online experience is not supported at the moment.

### 2.2 Local Experience
> ❗ Before using the 3D object detection pipeline locally, make sure you have installed PaddleX's wheel package following the [PaddleX installation tutorial](../../../installation/installation.md).

#### 2.2.1 Command Line Experience

Command line experience is not supported at the moment.

By default, the built-in 3D object detection pipeline configuration file is used. If you need a custom configuration file, you can execute the following command:

<details><summary> 👉Click to expand</summary>

<pre><code class="language-bash">paddlex --get_pipeline_config 3d_bev_detection
</code></pre>
<p>After execution, the 3D object detection pipeline configuration file will be saved in the current path. If you wish to save it to a custom location, you can use the following command (assuming the custom location is <code>./my_path</code>）：</p>
<pre><code class="language-bash">paddlex --get_pipeline_config 3d_bev_detection --save_path ./my_path
</code></pre></details>

#### 2.2.2 Python Script Integration
* The command line is mainly for quick experience. Typically, in a project, integration is done through code. You can complete pipeline inference with just a few lines of code:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="3d_bev_detection")
output = pipeline.predict("./data/nuscenes/nuscenes_infos_val.pkl")

for res in output:
    print(res)
    res.print()  ## Print structured output
    res.save_to_json("./output/")  ## Save results to a JSON file
```

In the above Python script, the following steps are performed:：

（1）Instantiate the `create_pipeline` to create a 3D object detection pipeline object. The parameter details are as follows:
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>Pipeline name or configuration file path. If it's a pipeline name, it must be a supported pipeline by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>Inference device for the pipeline model. Supported: "gpu", "cpu".</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available when the pipeline supports high-performance inference.	</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
（2）Call the `predict` method of the 3D object detection pipeline object for inference. The `predict` method takes `input` as a parameter, used to input data for prediction. It supports multiple input types, as shown in the following example:
<table>
<thead>
<tr>
<th>Parameter Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>str</td>
<td><b>File path</b>，e.g., local path of a 3D annotation file：<code>/root/data/anno_file.pkl</code></td>
</tr>
<tr>
<td>list</td>
<td><b>List</b>，list elements must be of the above type, e.g., <code>["/root/data/anno_file1.pkl", "/root/data/anno_file2.pkl"]</td>
</tr>
</tbody>
</table>
（3）Call the `predict` method to obtain prediction results. The `predict` method is a `generator`, so you need to iterate to get the prediction results. The `predict` method predicts data in batches, so the prediction results are a list representing a set of prediction results.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and can be printed or saved as a JSON file, as shown below:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description	</th>
<th>Parameters</th>
</tr>
</thead>
<tbody>
<tr>
<td>print</td>
<td>Print results to the terminal</td>
<td><code>- format_json</code>：bool, whether to format the output content with JSON indentation, default is True;<br/><code>- indent</code>：int, JSON formatting setting, only effective when format_json is True, default is 4;</code>：bool, JSON formatting setting, only effective when format_json is True, default is False;</td>
</tr>
<tr>
<td>save_to_json</td>
<td>Save results to a JSON file</td>
<td><code>- save_path</code>：str, file path for saving, when it's a directory, the saved file name is consistent with the input file type naming;<br/><code>- indent</code>：int, JSON formatting setting, default is 4;<br/><code>- ensure_ascii</code>：bool, JSON formatting setting, default is False;</td>
</tr>
</tbody>
</table>
If you have obtained a configuration file, you can customize various configurations of the 3D object detection pipeline by modifying the `pipeline` parameter value in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/3d_bev_detection.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/3d_bev_detection.yaml")
output = pipeline.predict("./data/nuscenes/nuscenes_infos_val.pkl")

for res in output:
    print(res)
    res.print()  ## Print structured output
    res.save_to_json("./output/")  ## Save results to a CSV file
```
## 3. Development Integration/Deployment
If the 3D object detection pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the general 3D object detection pipeline directly to your Python project, you can refer to the example code in [2.2.2 Python Script Integration](#222-command-line-experience)

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>：In real production environments, many applications have stringent performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. Therefore, PaddleX offers a high-performance inference plugin aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

☁️ <b>Service Deployment</b>：Service deployment is a common deployment form in real production environments. By encapsulating inference functionality as a service, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service deployment of pipelines. For detailed service deployment procedures, please refer to the [PaddleX Service Deployment Guide](../../../pipeline_deploy/service_deploy.md).

📱 <b>Edge Deployment</b>：Edge deployment is a method of placing computation and data processing functions on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.md).

You can choose the appropriate deployment method based on your needs to deploy the model pipeline and proceed with subsequent AI application integration.

## 4. Secondary Development
If the default model weights provided by the 3D object detection pipeline do not meet your scene's accuracy or speed requirements, you can attempt to further <b>fine-tune</b> the existing model using <b> your own data from specific domains or application scenarios </b> to improve the pipeline's recognition performance in your scene.

### 4.1 Model Fine-Tuning

Refer to the [Secondary Development chapter](../../../module_usage/tutorials/cv_modules/3d_bev_detection.md#) in the [3D Multi-Modal Fusion Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/3d_bev_detection.md) to fine-tune the model using your private dataset.

### 4.2 Model Application
After completing fine-tuning using your private dataset, you will obtain the local model weight file.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned 3D object detection model in the corresponding position of the pipeline configuration file:

```bash
......
Pipeline:
  device: "gpu:0"
  det_model: "BEVFusion"        #Can be modified to the local path of the fine-tuned 3D object detection model
  det_batch_size: 1
  device: gpu
......
```
Subsequently, refer to the command line or Python script methods in [2.2 Local Experience]((#22-local-experience)) to load the modified pipeline configuration file.
