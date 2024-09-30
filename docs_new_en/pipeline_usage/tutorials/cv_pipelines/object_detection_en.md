# General Object Detection Pipeline Tutorial

## 1. Introduction to General Object Detection Pipeline
Object detection aims to identify the categories and locations of multiple objects in images or videos by generating bounding boxes to mark these objects. Unlike simple image classification, object detection not only requires recognizing what objects are present in an image, such as people, cars, and animals, but also accurately determining the specific position of each object within the image, typically represented by rectangular boxes. This technology is widely used in autonomous driving, surveillance systems, smart photo albums, and other fields, relying on deep learning models (e.g., YOLO, Faster R-CNN) that can efficiently extract features and perform real-time detection, significantly enhancing the computer's ability to understand image content.

![](/tmp/images/pipelines/object_detection/01.png)


<details>
   <summary> 👉Details of Model List</summary>

<table>
  <tr>
    <th>Model</th>
    <th>mAP(%)</th>
    <th>GPU Inference Time (ms)</th>
    <th>CPU Inference Time</th>
    <th>Model Size (M)</th>
    <th>Description</th>
  </tr>
 <tr>
    <td>Cascade-FasterRCNN-ResNet50-FPN</td>
    <td>41.1</td>
    <td>-</td>
    <td>-</td>
    <td>245.4 M</td>
    <td rowspan="2">Cascade-FasterRCNN is an improved version of the Faster R-CNN object detection model. By coupling multiple detectors and optimizing detection results using different IoU thresholds, it addresses the mismatch problem between training and prediction stages, enhancing the accuracy of object detection.</td>
  </tr>
  <tr>
    <td>Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
    <td>45.0</td>
    <td>-</td>
    <td>-</td>
    <td>246.2 M</td>
    <td></td>
  </tr>
  <tr>
    <td>CenterNet-DLA-34</td>
    <td>37.6</td>
    <td>-</td>
    <td>-</td>
    <td>75.4 M</td>
    <td rowspan="2">CenterNet is an anchor-free object detection model that treats the keypoints of the object to be detected as a single point—the center point of its bounding box, and performs regression through these keypoints.</td>
  </tr>
  <tr>
    <td>CenterNet-ResNet50</td>
    <td>38.9</td>
    <td>-</td>
    <td>-</td>
    <td>319.7 M</td>
    <td></td>
  </tr>
  <tr>
    <td>DETR-R50</td>
    <td>42.3</td>
    <td>59.2132</td>
    <td>5334.52</td>
    <td>159.3 M</td>
    <td >DETR is a transformer-based object detection model proposed by Facebook. It achieves end-to-end object detection without the need for predefined anchor boxes or NMS post-processing strategies.</td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet34-FPN</td>
    <td>37.8</td>
    <td>-</td>
    <td>-</td>
    <td>137.5 M</td>
    <td rowspan="9">Faster R-CNN is a typical two-stage object detection model that first generates region proposals and then performs classification and regression on these proposals. Compared to its predecessors R-CNN and Fast R-CNN, Faster R-CNN's main improvement lies in the region proposal aspect, using a Region Proposal Network (RPN) to provide region proposals instead of traditional selective search. RPN is a Convolutional Neural Network (CNN) that shares convolutional features with the detection network, reducing the computational overhead of region proposals.</td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet50-FPN</td>
    <td>38.4</td>
    <td>-</td>
    <td>-</td>
    <td>148.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet50-vd-FPN</td>
    <td>39.5</td>
    <td>-</td>
    <td>-</td>
    <td>148.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet50-vd-SSLDv2-FPN</td>
    <td>41.4</td>
    <td>-</td>
    <td>-</td>
    <td>148.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet50</td>
    <td>36.7</td>
    <td>-</td>
    <td>-</td>
    <td>120.2 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet101-FPN</td>
    <td>41.4</td>
    <td>-</td>
    <td>-</td>
    <td>216.3 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNet101</td>
    <td>39.0</td>
    <td>-</td>
    <td>-</td>
    <td>188.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-ResNeXt101-vd-FPN</td>
    <td>43.4</td>
    <td>-</td>
    <td>-</td>
    <td>360.6 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FasterRCNN-Swin-Tiny-FPN</td>
    <td>42.6</td>
    <td>-</td>
    <td>-</td>
    <td>159.8 M</td>
    <td></td>
  </tr>
  <tr>
    <td>FCOS-ResNet50</td>
    <td>39.6</td>
    <td>103.367</td>
    <td>3424.91</td>
    <td>124.2 M</td>
    <td>FCOS is an anchor-free object detection model that performs dense predictions. It uses the backbone of RetinaNet and directly regresses the width and height of the target object on the feature map, predicting the object's category and centerness (the degree of offset of pixels on the feature map from the object's center), which is eventually used as a weight to adjust the object score.</td>
  </tr>
  <tr>
    <td>PicoDet-L</td>
    <td>42.6</td>
    <td>16.6715</td>
    <td>169.904</td>
    <td>20.9 M</td>
    <td rowspan="4">PP-PicoDet is a lightweight object detection algorithm designed for full-size and wide-aspect-ratio targets, with a focus on mobile device computation. Compared to traditional object detection algorithms, PP-PicoDet boasts smaller model sizes and lower computational complexity, achieving higher speeds and lower latency while maintaining detection accuracy.</td>
  </tr>
  <tr>
    <td>PicoDet-M</td>
    <td>37.5</td>
    <td>16.2311</td>
    <td>71.7257</td>
    <td>16.8 M</td>
    <td></td>
  </tr>
  <tr>
    <td>PicoDet-S</td>
    <td>29.1</td>
    <td>14.097</td>
    <td>37.6563</td>
    <td>4.4 M</td>
    <td></td>
  </tr>
  <tr>
    <td>PicoDet-XS</td>
    <td>26.2</td>
    <td>13.8102</td>
    <td>48.3139</td>
    <td>5.7 M</td>
    <td></td>
  </tr>
    <tr>
    <td>PP-YOLOE_plus-L</td>
    <td>52.9</td>
    <td>33.5644</td>
    <td>814.825</td>
    <td>185.3 M</td>
    <td rowspan="4">PP-YOLOE_plus is an iteratively optimized and upgraded version of PP-YOLOE, a high-precision cloud-edge integrated model developed by Baidu PaddlePaddle's Vision Team. By leveraging the large-scale Objects365 dataset and optimizing preprocessing, it significantly enhances the end-to-end inference speed of the model.</td>
  </tr>
  <tr>
    <td>PP-YOLOE_plus-M</td>
    <td>49.8</td>
    <td>19.843</td>
    <td>449.261</td>
    <td>82.3 M</td>
    <td></td>
  </tr>
  <tr>
    <td>PP-YOLOE_plus-S</td>
    <td>43.7</td>
    <td>16.8884</td>
    <td>223.059</td>
    <td>28.3 M</td>
    <td></td>
  </tr>
  <tr>
    <td>PP-YOLOE_plus-X</td>
    <td>54.7</td>
    <td>57.8995</td>
    <td>1439.93</td>
    <td>349.4 M</td>
    <td></td>
  </tr>
  <tr>
    <td>RT-DETR-H</td>
    <td>56.3</td>
    <td>114.814</td>
    <td>3933.39</td>
    <td>435.8 M</td>
    <td rowspan="5">RT-DETR is the first real-time end-to-end object detector. It features an efficient hybrid encoder that balances model performance and throughput, efficiently processes multi-scale features, and introduces an accelerated and optimized query selection mechanism to dynamize decoder queries. RT-DETR supports flexible end-to-end inference speeds through the use of different decoders.</td>
  </tr>
  <tr>
    <td>RT-DETR-L</td>
    <td>53.0</td>
    <td>34.5252</td>
    <td>1454.27</td>
    <td>113.7 M</td>
    <td></td>
  </tr>
  <tr>
    <td>RT-DETR-R18</td>
    <td>46.5</td>
    <td>19.89</td>
    <td>784.824</td>
    <td>70.7 M</td>
    <td></td>
  </tr>
  <tr>
    <td>RT-DETR-R50</td>
    <td>53.1</td>
    <td>41.9327</td>
    <td>1625.95</td>
    <td>149.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>RT-DETR-X</td>
    <td>54.8</td>
    <td>61.8042</td>
    <td>2246.64</td>
    <td>232.9 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOv3-DarkNet53</td>
    <td>39.1</td>
    <td>40.1055</td>
    <td>883.041</td>
    <td>219.7 M</td>
    <td rowspan="3">YOLOv3 is a real-time end-to-end object detector that utilizes a unique single Convolutional Neural Network (CNN) to frame the object detection problem as a regression task, enabling real-time detection. The model employs multi-scale detection to enhance performance across different object sizes.</td>
  </tr>
  <tr>
    <td>YOLOv3-MobileNetV3</td>
    <td>31.4</td>
    <td>18.6692</td>
    <td>267.214</td>
    <td>83.8 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOv3-ResNet50_vd_DCN</td>
    <td>40.6</td>
    <td>31.6276</td>
    <td>856.047</td>
    <td>163.0 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-L</td>
    <td>50.1</td>
    <td>185.691</td>
    <td>1250.58</td>
    <td>192.5 M</td>
    <td rowspan="6">Building upon YOLOv3's framework, YOLOX significantly boosts detection performance in complex scenarios by incorporating Decoupled Head, Data Augmentation, Anchor Free, and SimOTA components.</td>
  </tr>
  <tr>
    <td>YOLOX-M</td>
    <td>46.9</td>
    <td>123.324</td>
    <td>688.071</td>
    <td>90.0 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-N</td>
    <td>26.1</td>
    <td>79.1665</td>
    <td>155.59</td>
    <td>3.4 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-S</td>
    <td>40.4</td>
    <td>184.828</td>
    <td>474.446</td>
    <td>32.0 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-T</td>
    <td>32.9</td>
    <td>102.748</td>
    <td>212.52</td>
    <td>18.1 M</td>
    <td></td>
  </tr>
  <tr>
    <td>YOLOX-X</td>
    <td>51.8</td>
    <td>227.361</td>
    <td>2067.84</td>
    <td>351.5 M</td>
    <td></td>
  </tr>
</table>

**Note: The precision metrics mentioned are based on the [COCO2017](https://cocodataset.org/#home) validation set mAP(0.5:0.95). All model GPU inference times are measured on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

</details>

## 2. Quick Start
PaddleX's pre-trained model pipelines allow for quick experience of their effects. You can experience the effects of the General Object Detection Pipeline online or locally using command line or Python.

### 2.1 Online Experience
You can [experience the General Object Detection Pipeline online](https://aistudio.baidu.com/community/app/70230/webUI) using the demo images provided by the official source, for example:

![](/tmp/images/pipelines/object_detection/02.png)

If you are satisfied with the pipeline's performance, you can directly integrate and deploy it. If not, you can also use your private data to **fine-tune the model within the pipeline**.

### 2.2 Local Experience
Before using the General Object Detection Pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation.md).

#### 2.2.1 Command Line Experience
A single command can quickly experience the effects of the object detection pipeline:

```bash
paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it is the object detection pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 indicates using the first GPU, gpu:1,2 indicates using the second and third GPUs). You can also choose to use CPU (--device cpu).
```

When executing the above command, the default object detection pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details>
   <summary> Click to expand 👉</summary>

```bash
paddlex --get_pipeline_config object_detection
```

After execution, the object detection pipeline configuration file will be saved in the current path. If you wish to customize the save location, you can execute the following command (assuming the custom save location is `./my_path`):

```bash
paddlex --get_pipeline_config object_detection --config_save_path ./my_path
```

After obtaining the pipeline configuration file, replace `--pipeline` with the configuration file save path to make the configuration file effective. For example, if the configuration file save path is `./object_detection.yaml`, simply execute:

```bash
paddlex --pipeline ./object_detection.yaml --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png
```

Here, parameters such as `--model` and `--device` do not need to be specified, as they will use the parameters in the configuration file. If these parameters are still specified, the specified parameters will take precedence.

</details>

After running, the result will be:

```
{'img_path': '/root/.paddlex/predict_input/general_object_detection_002.png', 'boxes': [{'cls_id': 49, 'label': 'orange', 'score': 0.8188097476959229, 'coordinate': [661, 93, 870, 305]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7743489146232605, 'coordinate': [76, 274, 330, 520]}, {'cls_id': 47, 'label': 'apple', 'score': 0.7270504236221313, 'coordinate': [285, 94, 469, 297]}, {'cls_id': 46, 'label': 'banana', 'score': 0.5570532083511353, 'coordinate': [310, 361, 685, 712]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5484835505485535, 'coordinate': [764, 285, 924, 440]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5160726308822632, 'coordinate': [853, 169, 987, 303]}, {'cls_id': 60, 'label': 'dining table', 'score': 0.5142655968666077, 'coordinate': [0, 0, 1072, 720]}, {'cls_id': 47, 'label': 'apple', 'score': 0.5101479291915894, 'coordinate': [57, 23, 213, 176]}]}
```

![](/tmp/images/pipelines/object_detection/03.png)

The visualized images are saved in the `output` directory by default, but you can customize this with `--save_path`.

#### 2.2.2 Integration via Python Scripts
A few lines of code are all you need to quickly perform inference on your production line. Taking General Object Detection as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="object_detection")

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_img("./output/")  # Save the visualized image of the result
    res.save_to_json("./output/")  # Save the structured output of the prediction
```
The results obtained are the same as those from the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: The specific parameter descriptions are as follows:

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `pipeline` | The name of the pipeline or the path to the pipeline configuration file. If it's a pipeline name, it must be supported by PaddleX. | `str` | None |
| `device` | The device for pipeline model inference. Supports: "gpu", "cpu". | `str` | "gpu" |
| `enable_hpi` | Whether to enable high-performance inference, only available if the pipeline supports it. | `bool` | `False` |

(2) Call the `predict` method of the pipeline object to perform inference: The `predict` method parameter `x` is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Description |
|----------------|-------------|
| Python Var | Supports directly passing Python variables, such as numpy.ndarray representing image data. |
| `str` | Supports passing the path of the file to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| `str` | Supports passing the URL of the file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png). |
| `str` | Supports passing a local directory, which should contain files to be predicted, such as the local path: `/root/data/`. |
| `dict` | Supports passing a dictionary type, where the key needs to correspond to the specific task, such as "img" for image classification tasks, and the value of the dictionary supports the above data types, e.g., `{"img": "/root/data1"}`. |
| `list` | Supports passing a list, where the list elements need to be of the above types, such as `[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]`. |

(3) Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

| Method         | Description                     | Method Parameters |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | Prints results to the terminal  | `- format_json`: bool, whether to format the output content with json indentation, default is True;<br>`- indent`: int, json formatting setting, only valid when format_json is True, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, only valid when format_json is True, default is False; |
| save_to_json | Saves results as a json file   | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;<br>`- indent`: int, json formatting setting, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, default is False; |
| save_to_img  | Saves results as an image file | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type; |

If you have a configuration file, you can customize the configurations of the image anomaly detection pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/object_detection.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/object_detection.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_img("./output/")  # Save the visualized image of the result
    res.save_to_json("./output/")  # Save the structured output of prediction
```

## 3. Development Integration/Deployment

If the pipeline meets your requirements for inference speed and accuracy, you can proceed with development integration/deployment.

If you need to directly apply the pipeline in your Python project, refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Deployment**: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies, especially response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. Refer to the [PaddleX High-Performance Deployment Guide](../../../pipeline_deploy/high_performance_deploy.md) for detailed high-performance deployment procedures.

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service-oriented deployment of pipelines. Refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy.md) for detailed service-oriented deployment procedures.

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing functions on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. Refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/lite_deploy.md) for detailed edge deployment procedures.

Choose the appropriate deployment method for your model pipeline based on your needs, and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the General Object Detection pipeline do not meet your requirements for precision or speed in your specific scenario, you can try to further **fine-tune** the existing model using **your own domain-specific or application-specific data** to improve the recognition performance of the General Object Detection pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the General Object Detection pipeline includes an object detection module, if the performance of the pipeline does not meet expectations, you need to refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/object_detection.md#四二次开发) section in the [Object Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/object_detection.md) and use your private dataset to fine-tune the object detection model.

### 4.2 Model Application
After fine-tuning the model with your private dataset, you will obtain local model weights files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```python
......
Pipeline:
  model: PicoDet-S  # Can be modified to the local path of the fine-tuned model
  device: "gpu"
  batch_size: 1
......
```
Then, refer to the command line method or Python script method in the local experience, and load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modify the `--device` parameter** to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for inference of the General Object Detection pipeline, the Python command is:

```bash
paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device gpu:0
``````
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the Python command to `npu`:

```bash
paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device npu:0
```
If you want to use the General Object Detection Pipeline on more types of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../installation/installation_other_devices.md).