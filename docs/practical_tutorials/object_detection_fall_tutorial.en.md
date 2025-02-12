---
comments: true
---

# PaddleX 3.0 General Object Detection Pipeline — Tutorial on Pedestrian Fall Detection

PaddleX offers a rich set of pipelines, each consisting of one or more models tailored to solve specific scenario tasks. All PaddleX pipelines support quick trials, and if the results are not satisfactory, you can fine-tune the models with your private data. PaddleX also provides Python APIs for easy integration into personal projects. Before proceeding, ensure you have installed PaddleX. For installation instructions, refer to [PaddleX Installation](../installation/installation.en.md). This tutorial introduces the usage of the pipeline tool with an example of pedestrian fall detection.

## 1. Select a Pipeline

First, choose the appropriate PaddleX pipeline based on your task scenario. For pedestrian fall detection, this falls under the General Object Detection pipeline in PaddleX. If unsure about the task-pipeline correspondence, consult the [Pipeline List](../support_list/pipelines_list.en.md) for capabilities of each pipeline.

## 2. Quick Start

PaddleX offers two ways to experience the pipelines: locally through the PaddleX wheel package or on the <b>Baidu AIStudio Community</b>.

- Local Experience:
    ```bash
    paddlex --pipeline object_detection \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/fall.png
    ```

- AIStudio Community Experience: Navigate to [Baidu AIStudio Community](https://aistudio.baidu.com/pipeline/mine), click "Create Pipeline," and create a <b>General Object Detection</b> pipeline for a quick trial.

Quick trial output example:
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/obeject_detection/01.png" width=600>

</center>

After the trial, determine if the pipeline meets your expectations (including accuracy, speed, etc.). If the model's speed or accuracy is unsatisfactory, test alternative models or proceed with fine-tuning. Since the default weights (trained on the COCO dataset) are unlikely to meet the requirements for detecting pedestrian falls, you'll need to collect and annotate data for training and fine-tuning.

## 3. Choose a Model

PaddleX provides 37 end-to-end object detection models. Refer to the [Model List](../support_list/models_list.en.md) for details. Here's a benchmark of some models:

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(%)</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PicoDet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_pretrained.pdparams">Trained Model</a></td>
<td>42.6</td>
<td>14.68 / 5.81</td>
<td>47.32 / 47.32</td>
<td>20.9 M</td>
<td rowspan="2">PP-PicoDet is a lightweight object detection algorithm for full-size, wide-angle targets, considering the computational capacity of mobile devices. Compared to traditional object detection algorithms, PP-PicoDet has a smaller model size and lower computational complexity, achieving higher speed and lower latency while maintaining detection accuracy.</td>
</tr>
<tr>
<td>PicoDet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PicoDet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_pretrained.pdparams">Trained Model</a></td>
<td>29.1</td>
<td>7.98 / 2.33</td>
<td>14.82 / 5.60</td>
<td>4.4 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-L_pretrained.pdparams">Trained Model</a></td>
<td>52.9</td>
<td>33.55 / 10.46</td>
<td>189.05 / 189.05</td>
<td>185.3 M</td>
<td rowspan="2">PP-YOLOE_plus is an upgraded version of the high-precision cloud-edge integrated model PP-YOLOE, developed by Baidu's PaddlePaddle vision team. By using the large-scale Objects365 dataset and optimizing preprocessing, it significantly enhances the model's end-to-end inference speed.</td>
</tr>
<tr>
<td>PP-YOLOE_plus-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_pretrained.pdparams">Trained Model</a></td>
<td>43.7</td>
<td>12.16 / 4.58</td>
<td>73.86 / 52.90</td>
<td>28.3 M</td>
</tr>
<tr>
<td>RT-DETR-H</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-H_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_pretrained.pdparams">Trained Model</a></td>
<td>56.3</td>
<td>115.92 / 28.16</td>
<td>971.32 / 971.32</td>
<td>435.8 M</td>
<td rowspan="2">RT-DETR is the first real-time end-to-end object detector. The model features an efficient hybrid encoder to meet both model performance and throughput requirements, efficiently handling multi-scale features, and proposes an accelerated and optimized query selection mechanism to optimize the dynamics of decoder queries. RT-DETR supports flexible end-to-end inference speeds by using different decoders.</td>
</tr>
<tr>
<td>RT-DETR-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_pretrained.pdparams">Trained Model</a></td>
<td>53.0</td>
<td>35.00 / 10.45</td>
<td>495.51 / 495.51</td>
<td>113.7 M</td>
</tr>
</table>

> <b>Note: The above accuracy metrics are based on the mAP(0.5:0.95) of the [COCO2017](https://cocodataset.org/#home) validation set. GPU inference time is measured on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

In summary, models listed from top to bottom offer faster inference speeds, while those from bottom to top offer higher accuracy. This tutorial uses the PP-YOLOE_plus-S model as an example to complete the full model development process. Choose a suitable model based on your actual usage scenario, train it, evaluate the model weights within the pipeline, and finally deploy

## 4. Data Preparation and Verification
### 4.1 Data Preparation

This tutorial uses the "Pedestrian Fall Detection Dataset" as an example dataset. You can obtain the example dataset using the following commands. If you use your own annotated dataset, you need to adjust it according to the PaddleX format requirements to meet PaddleX's data format specifications. For data format introductions, you can refer to the [PaddleX Object Detection Task Module Data Preparation Tutorial](../data_annotations/cv_modules/object_detection.en.md).

Dataset acquisition commands:
```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/fall_det.tar -P ./dataset
tar -xf ./dataset/fall_det.tar -C ./dataset/
```

### 4.2 Dataset Verification

To verify the dataset, simply use the following command:

```bash
python main.py -c paddlex/configs/modules/object_detection/PP-YOLOE_plus-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/fall_det
```

After executing the above command, PaddleX will verify the dataset and count its basic information. Upon successful execution, the log will print out `Check dataset passed !` information, and relevant outputs will be saved in the current directory's `./output/check_dataset` folder. The output directory includes visualized example images and sample distribution histograms. The verification result file is saved in `./output/check_dataset_result.json`, and the specific content of the verification result file is:

```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 1,
    "train_samples": 1224,
    "train_sample_paths": [
      "check_dataset/demo_img/fall_1168.jpg",
      "check_dataset/demo_img/fall_1113.jpg"
    ],
    "val_samples": 216,
    "val_sample_paths": [
      "check_dataset/demo_img/fall_349.jpg",
      "check_dataset/demo_img/fall_394.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/fall_det",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}
```
The above verification results indicate that the `check_pass` being `True` means the dataset format meets the requirements. Explanations for other indicators are as follows:

- `attributes.num_classes`: The number of classes in this dataset is 1, which is the number of classes that need to be passed in for subsequent training.
- `attributes.train_samples`: The number of training samples in this dataset is 1224.
- `attributes.val_samples`: The number of validation samples in this dataset is 216.
- `attributes.train_sample_paths`: A list of relative paths to the visualization images of training samples in this dataset.
- `attributes.val_sample_paths`: A list of relative paths to the visualization images of validation samples in this dataset.

Additionally, the dataset verification also analyzes the distribution of sample numbers across all classes and generates a histogram (`histogram.png`) for visualization:
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/obeject_detection/02.png" width=600>

</center>

<b>Note</b>: Only data that passes the verification can be used for training and evaluation.


## 4.3 Dataset Format Conversion/Dataset Splitting (Optional)

If you need to convert the dataset format or re-split the dataset, you can modify the configuration file or append hyperparameters for settings.

Parameters related to dataset verification can be set by modifying the fields under `CheckDataset` in the configuration file. Examples of some parameters in the configuration file are as follows:

* `CheckDataset`:
    * `convert`:
        * `enable`: Whether to convert the dataset format. Set to `True` to enable dataset format conversion, default is `False`.
        * `src_dataset_type`: If dataset format conversion is performed, the source dataset format must be specified. Available source formats are `LabelMe` and `VOC`.
    * `split`:
        * `enable`: Whether to re-split the dataset. Set to `True` to enable dataset splitting, default is `False`.
        * `train_percent`: If dataset splitting is performed, the percentage of the training set must be set. The value should be an integer between 0 and 100, and the sum with `val_percent` must be 100.
        * `val_percent`: If dataset splitting is performed, the percentage of the validation set must be set. The value should be an integer between 0 and 100, and the sum with `train_percent` must be 100.

Data conversion and data splitting can be enabled simultaneously. For data splitting, the original annotation files will be renamed to `xxx.bak` in their original paths. The above parameters also support setting through appending command-line arguments, for example, to re-split the dataset and set the training and validation set ratios: `-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`.

## 5. Model Training and Evaluation

### 5.1 Model Training

Before training, ensure that you have validated your dataset. To complete the training of a PaddleX model, simply use the following command:

```bash
python main.py -c paddlex/configs/modules/object_detection/PP-YOLOE_plus-S.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/fall_det \
    -o Train.num_classes=1
```

PaddleX supports modifying training hyperparameters, single/multi-GPU training, and more, simply by modifying the configuration file or appending command-line parameters.

Each model in PaddleX provides a configuration file for model development, which is used to set relevant parameters. Model training-related parameters can be set by modifying the `Train` fields in the configuration file. Some example explanations of parameters in the configuration file are as follows:

* `Global`:
    * `mode`: Mode, supporting dataset validation (`check_dataset`), model training (`train`), and model evaluation (`evaluate`);
    * `device`: Training device, options include `cpu`, `gpu`, `xpu`, `npu`, `mlu`. For multi-GPU training, specify card numbers, e.g., `gpu:0,1,2,3`;
* `Train`: Training hyperparameter settings;
    * `epochs_iters`: Number of training epochs;
    * `learning_rate`: Training learning rate;

For more hyperparameter introductions, refer to [PaddleX General Model Configuration File Parameter Explanation](../module_usage/instructions/config_parameters_common.en.md).

<b>Note</b>:
- The above parameters can be set by appending command-line parameters, e.g., specifying the mode as model training: `-o Global.mode=train`; specifying the first two GPUs for training: `-o Global.device=gpu:0,1`; setting the number of training epochs to 10: `-o Train.epochs_iters=10`.
- During model training, PaddleX automatically saves model weight files, with the default being `output`. To specify a save path, use the `-o Global.output` field in the configuration file.
- PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.

<b>Explanation of Training Outputs</b>:

After completing model training, all outputs are saved in the specified output directory (default is `./output/`), typically including the following:

* train_result.json: Training result record file, recording whether the training task completed normally, as well as the output weight metrics, relevant file paths, etc.;
* train.log: Training log file, recording changes in model metrics, loss, etc., during training;
* config.yaml: Training configuration file, recording the hyperparameter configuration for this training session;
* .pdparams, .pdema, .pdopt.pdstate, .pdiparams, .pdmodel: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;

### 5.2 Model Evaluation

After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. To evaluate a model using PaddleX, simply use the following command:

```bash
python main.py -c paddlex/configs/modules/object_detection/PP-YOLOE_plus-S.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/fall_det
```

Similar to model training, model evaluation supports setting parameters by modifying the configuration file or appending command-line parameters.

<b>Note</b>: When evaluating a model, you need to specify the model weight file path. Each configuration file has a default weight save path. If you need to change it, simply set it by appending a command-line parameter, e.g., `-o Evaluate.weight_path=./output/best_model/best_model.pdparams`.

### 5.3 Model Tuning

After learning about model training and evaluation, we can improve model accuracy by adjusting hyperparameters. By reasonably adjusting the number of training epochs, you can control the depth of model training to avoid overfitting or underfitting. The learning rate setting affects the speed and stability of model convergence. Therefore, when optimizing model performance, carefully consider the values of these two parameters and adjust them flexibly based on actual conditions to achieve the best training results.

It is recommended to follow the control variable method when debugging parameters:

1. First, fix the number of training epochs to 10 and the batch size to 8.
2. Start three experiments based on the PP-YOLOE_plus-S model with learning rates of: 0.00002, 0.0001, 0.0005.
3. You may find that the configuration with the highest accuracy in Experiment 2 is a learning rate of 0.0001. Based on this training hyperparameter, change the number of epochs and observe the accuracy results of different epochs. It is found that the best accuracy is basically achieved at 100 epochs.

Learning Rate Exploration Results:
<center>

<table>
<thead>
<tr>
<th>Experiment</th>
<th>Epochs</th>
<th>Learning Rate</th>
<th>batch_size</th>
<th>Training Environment</th>
<th>mAP@0.5</th>
</tr>
</thead>
<tbody>
<tr>
<td>Experiment 1</td>
<td>10</td>
<td>0.00002</td>
<td>8</td>
<td>4 GPUs</td>
<td>0.880</td>
</tr>
<tr>
<td>Experiment 2</td>
<td>10</td>
<td>0.0001</td>
<td>8</td>
<td>4 GPUs</td>
<td><strong>0.910</strong></td>
</tr>
<tr>
<td>Experiment 3</td>
<td>10</td>
<td>0.0005</td>
<td>8</td>
<td>4 GPUs</td>
<td>0.888</td>
</tr>
</tbody>
</table>
</center>

Changing Epochs Results:
<center>

<table>
<thead>
<tr>
<th>Experiment</th>
<th>Epochs</th>
<th>Learning Rate</th>
<th>batch_size</th>
<th>Training Environment</th>
<th>mAP@0.5</th>
</tr>
</thead>
<tbody>
<tr>
<td>Experiment 2</td>
<td>10</td>
<td>0.0001</td>
<td>8</td>
<td>4 GPUs</td>
<td>0.910</td>
</tr>
<tr>
<td>Experiment 2 (Increased Epochs)</td>
<td>50</td>
<td>0.0001</td>
<td>8</td>
<td>4 GPUs</td>
<td>0.944</td>
</tr>
<tr>
<td>Experiment 2 (Increased Epochs)</td>
<td>100</td>
<td>0.0001</td>
<td>8</td>
<td>4 GPUs</td>
<td><strong>0.947</strong></td>
</tr>
</tbody>
</table>
</center>

> <b>Note: The above accuracy metrics are based on the mAP(0.5:0.95) of the [COCO2017](https://cocodataset.org/#home) validation set. GPU inference time is measured on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

## 6. Production Line Testing

Replace the model in the production line with the fine-tuned model for testing, for example:

```bash
python main.py -c paddlex/configs/modules/object_detection/PP-YOLOE_plus-S.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/fall.png"
```

The prediction results will be generated under `./output`, and the prediction result for `fall.png` is shown below:
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/obeject_detection/03.png" width="600"/>

</center>

## 7. Development Integration/Deployment
If the General Object Detection Pipeline meets your requirements for inference speed and precision in the production line, you can proceed directly with development integration/deployment.

1. If you need to use the fine-tuned model weights, you can obtain the pipeline configuration file for object detection and load it for prediction. You can execute the following command to save the results in `my_path`:

```bash
paddlex --get_pipeline_config object_detection --save_path ./my_path
```

Fill in the local path of the fine-tuned model weights in the `model_dir` of the pipeline configuration file. If you want to directly apply the general object detection pipeline in your Python project, you can refer to the example below:

```yaml
pipeline_name: object_detection

SubModules:
  ObjectDetection:
    module_name: object_detection
    model_name: PicoDet-S
    model_dir: null # Replace this with the local path to your trained model weights
    batch_size: 1
    img_size: null
    threshold: null
```

Then, in your Python code, you can use the pipeline as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="my_path/object_detection.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/fall.png")
for res in output:
    res.print() # Print the structured output of the prediction
    res.save_to_img("./output/") # Save the result as a visualized image
    res.save_to_json("./output/") # Save the structured output of the prediction
```

For more parameters, please refer to [General Object Detection Pipeline Usage Tutorial](../pipeline_usage/tutorials/cv_pipelines/object_detection.en.md).

2. Additionally, PaddleX offers three other deployment methods, detailed as follows:

* high-performance inference: In actual production environments, many applications have stringent standards for deployment strategy performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end process acceleration. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../pipeline_deploy/high_performance_inference.en.md).
* Service-Oriented Deployment: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving cost-effective service-oriented deployment of production lines. For detailed service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../pipeline_deploy/serving.en.md).
* Edge Deployment: Edge deployment is a method that places computing and data processing capabilities directly on user devices, allowing devices to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../pipeline_deploy/edge_deploy.en.md).

You can select the appropriate deployment method for your model pipeline according to your needs, and proceed with subsequent AI application integration.
