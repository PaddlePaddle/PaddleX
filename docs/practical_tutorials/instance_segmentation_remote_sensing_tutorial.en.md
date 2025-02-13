---
comments: true
---

# PaddleX 3.0 General Instance Segmentation Pipeline — Tutorial for Remote Sensing Image Instance Segmentation

PaddleX offers a rich set of pipelines, each consisting of one or more models that can tackle specific scenario tasks. All PaddleX pipelines support quick trials, and if the results do not meet expectations, fine-tuning with private data is also supported. PaddleX provides Python APIs for easy integration into personal projects. Before use, you need to install PaddleX. For installation instructions, refer to [PaddleX Installation](../installation/installation.en.md). This tutorial introduces the usage of the pipeline tool with an example of remote sensing image segmentation.

## 1. Select a Pipeline

First, choose the corresponding PaddleX pipeline based on your task scenario. For remote sensing image segmentation, this falls under the category of instance segmentation, corresponding to PaddleX's Universal Instance Segmentation Pipeline. If unsure about the task-pipeline correspondence, refer to the [Pipeline List](../support_list/pipelines_list.en.md) for an overview of pipeline capabilities.

## 2. Quick Start

PaddleX offers two ways to experience the pipeline: locally through the PaddleX wheel package or on the <b>Baidu AIStudio Community</b>.

- Local Experience:
    ```bash
    paddlex --pipeline instance_segmentation \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/remotesensing_demo.png
    ```

- AIStudio Community Experience: Go to [Baidu AIStudio Community](https://aistudio.baidu.com/pipeline/mine), click "Create Pipeline", and create a <b>Universal Instance Segmentation</b> pipeline for a quick trial.

Quick trial output example:
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/instance_segmentation/01.png" width=600>

</center>

After experiencing the pipeline, determine if it meets your expectations (including accuracy, speed, etc.). If the included model needs further fine-tuning due to unsatisfactory speed or accuracy, select alternative models for testing until satisfied. If the final results are unsatisfactory, fine-tuning the model is necessary. This tutorial aims to produce a model that segments geospatial objects, and the default weights (trained on the COCO dataset) cannot meet this requirement. Data collection and annotation are required for training and fine-tuning.

## 3. Select a Model

PaddleX provides 15 end-to-end instance segmentation models. Refer to the [Model List](../support_list/models_list.en.md) for details. Benchmarks for some models are as follows:

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Mask AP</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>Mask-RT-DETR-H</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-H_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-H_pretrained.pdparams">Trained Model</a></td>
<td>50.6</td>
<td>172.36 / 172.36</td>
<td>1615.75 / 1615.75</td>
<td>449.9 M</td>
<td rowspan="5">Mask-RT-DETR is an instance segmentation model based on RT-DETR. By adopting the high-performance PP-HGNetV2 as the backbone network and constructing a MaskHybridEncoder encoder, along with introducing IOU-aware Query Selection technology, it achieves state-of-the-art (SOTA) instance segmentation accuracy with the same inference time.</td>
</tr>
<tr>
<td>Mask-RT-DETR-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Mask-RT-DETR-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Mask-RT-DETR-L_pretrained.pdparams">Trained Model</a></td>
<td>45.7</td>
<td>88.18 / 88.18</td>
<td>1090.84 / 1090.84</td>
<td>113.6 M</td>
</tr>
</table>

> <b>Note: The above accuracy metrics are mAP(0.5:0.95) on the [COCO2017](https://cocodataset.org/#home) validation set. GPU inference time is based on an NVIDIA V100 machine with FP32 precision.</b>

In summary, models listed from top to bottom offer faster inference speeds, while those from bottom to top offer higher accuracy. This tutorial uses the `Mask-RT-DETR-H` model as an example to complete the full model development process. Choose a suitable model based on your actual usage scenario, train it, evaluate the model weights within the pipeline, and finally apply them in real-world scenarios.

## 4. Data Preparation and Validation
### 4.1 Data Preparation

This tutorial uses the "Remote Sensing Image Instance Segmentation Dataset" as an example dataset. You can obtain the example dataset using the following commands. If you are using your own annotated dataset, you need to adjust it according to PaddleX's format requirements to meet PaddleX's data format specifications. For an introduction to data formats, you can refer to [PaddleX Instance Segmentation Task Module Data Annotation Tutorial](../data_annotations/cv_modules/instance_segmentation.en.md).

Dataset acquisition commands:
```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/intseg_remote_sense_coco.tar -P ./dataset
tar -xf ./dataset/intseg_remote_sense_coco.tar -C ./dataset/
```

### 4.2 Dataset Verification

When verifying the dataset, you only need one command:

```bash
python main.py -c paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-H.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/intseg_remote_sense_coco
```

After executing the above command, PaddleX will verify the dataset and collect basic information about it. Upon successful execution, the log will print `Check dataset passed !`, and relevant outputs will be saved in the current directory's `./output/check_dataset` folder. The output directory includes visualized sample images and sample distribution histograms. The verification result file is saved in `./output/check_dataset_result.json`, and the specific content of the verification result file is as follows:

```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 10,
    "train_samples": 2989,
    "train_sample_paths": [
      "check_dataset/demo_img/524.jpg",
      "check_dataset/demo_img/024.jpg",
    ],
    "val_samples": 932,
    "val_sample_paths": [
      "check_dataset/demo_img/326.jpg",
      "check_dataset/demo_img/596.jpg",
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/intseg_remote_sense_coco/",
  "show_type": "image",
  "dataset_type": "COCOInstSegDataset"
}
```

In the above verification results, `check_pass` being `true` indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:

- `attributes.num_classes`: The number of classes in this dataset is 10, which is the number of classes that need to be passed in for subsequent training.
- `attributes.train_samples`: The number of training samples in this dataset is 2989.
- `attributes.val_samples`: The number of validation samples in this dataset is 932.
- `attributes.train_sample_paths`: A list of relative paths to the visualized training samples.
- `attributes.val_sample_paths`: A list of relative paths to the visualized validation samples.

Additionally, the dataset verification also analyzes the sample number distribution of all categories in the dataset and generates a distribution histogram (`histogram.png`):

<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/instance_segmentation/02.png" width=600>

</center>

<b>Note</b>: Only data that passes verification can be used for training and evaluation.

### 4.3 Dataset Format Conversion / Dataset Splitting (Optional)

If you need to convert the dataset format or re-split the dataset, you can modify the configuration file or append hyperparameters for settings.

Parameters related to dataset verification can be set by modifying the fields under `CheckDataset` in the configuration file. The following are example explanations of some parameters in the configuration file:

* `CheckDataset`:
    * `convert`:
        * `enable`: Whether to convert the dataset format. Set to `True` to enable dataset format conversion, default is `False`;
        * `src_dataset_type`: If dataset format conversion is enabled, the source dataset format needs to be set. Available source formats are `LabelMe` and `VOC`;
    * `split`:
        * `enable`: Whether to re-split the dataset. Set to `True` to enable dataset splitting, default is `False`;
        * `train_percent`: If re-splitting the dataset, the percentage of the training set needs to be set, which is an integer between 0-100, and the sum with `val_percent` must be 100;
        * `val_percent`: If re-splitting the dataset, the percentage of the validation set needs to be set, which is an integer between 0-100, and the sum with `train_percent` must be 100;

Data conversion and data splitting can be enabled simultaneously. For data splitting, the original annotation files will be renamed to `xxx.bak` in the original path. The above parameters also support setting through appending command line arguments, for example, to re-split the dataset and set the training and validation set ratios: `-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`.

## 5. Model Training and Evaluation
### 5.1 Model Training

Before training, ensure that you have verified the dataset. To complete PaddleX model training, simply use the following command:

```bash
python main.py -c paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-H.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/intseg_remote_sense_coco \
    -o Train.num_classes=10
```

PaddleX model training supports modifying training hyperparameters, single/multi-GPU training, etc., simply by modifying the configuration file or appending command line arguments.

Each model in PaddleX provides a configuration file for model development to set relevant parameters. Parameters related to model training can be set by modifying the fields under `Train` in the configuration file. The following are example explanations of some parameters in the configuration file:

* `Global`:
    * `mode`: Mode, supporting dataset verification (`check_dataset`), model training (`train`), and model evaluation (`evaluate`);
    * `device`: Training device, options include `cpu`, `gpu`, `xpu`, `npu`, `mlu`. For multi-GPU training, specify card numbers, e.g., `gpu:0,1,2,3`;
* `Train`: Training hyperparameter settings;
    * `epochs_iters`: Number of training epochs;
    * `learning_rate`: Training learning rate;

For more hyperparameter introductions, please refer to [PaddleX General Model Configuration File Parameter Explanation](../module_usage/instructions/config_parameters_common.en.md).

<b>Note</b>:
- The above parameters can be set by appending command line arguments, e.g., specifying the mode as model training: `-o Global.mode=train`; specifying the first 2 GPUs for training: `-o Global.device=gpu:0,1`; setting the number of training epochs to 10: `-o Train.epochs_iters=10`.
- During model training, PaddleX automatically saves model weight files, with the default being `output`. If you need to specify a save path, you can use the `-o Global.output` field in the configuration file.
- PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced. During model inference, static graph weights are selected by default.

<b>Explanation of Training Outputs</b>:

After completing the model training, all outputs are saved in the specified output directory (default is `./output/`), typically including the following:

* train_result.json: Training result record file, which logs whether the training task completed successfully, as well as the output weight metrics, relevant file paths, etc.;
* train.log: Training log file, which records changes in model metrics, loss variations, etc. during the training process;
* config.yaml: Training configuration file, which records the hyperparameter configurations for this training session;
* .pdparams, .pdema, .pdopt.pdstate, .pdiparams, .pdmodel: Model weight-related files, including network parameters, optimizer states, EMA (Exponential Moving Average), static graph network parameters, and static graph network structures;

### 5.2 Model Evaluation

After completing model training, you can evaluate the specified model weight files on the validation set to verify the model's accuracy. To perform model evaluation using PaddleX, simply use the following command:

```bash
python main.py -c paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-H.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/intseg_remote_sense_coco
```

Similar to model training, model evaluation supports setting configurations by modifying the configuration file or appending command-line parameters.

<b>Note</b>: When evaluating the model, you need to specify the model weight file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command-line parameter, such as `-o Evaluate.weight_path=./output/best_model/best_model.pdparams`.

### 5.3 Model Tuning

After learning about model training and evaluation, we can enhance model accuracy by adjusting hyperparameters. By carefully tuning the number of training epochs, you can control the depth of model training, avoiding overfitting or underfitting. Meanwhile, the setting of the learning rate is crucial to the speed and stability of model convergence. Therefore, when optimizing model performance, it is essential to consider the values of these two parameters prudently and adjust them flexibly based on actual conditions to achieve the best training results.

It is recommended to follow the controlled variable method when debugging parameters:

1. First, fix the number of training epochs at 80 and the batch size at 2.
2. Launch three experiments based on the `Mask-RT-DETR-H` model with learning rates of: 0.0005, 0.005, 0.0001
3. You may find that the configuration with the highest accuracy in Experiment 2 is a learning rate of 0.0001. Based on this training hyperparameter, change the number of epochs and observe the accuracy results for different epochs.

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
<td>80</td>
<td>0.0005</td>
<td>2</td>
<td>4 GPUs</td>
<td>0.695</td>
</tr>
<tr>
<td>Experiment 2</td>
<td>80</td>
<td>0.0001</td>
<td>2</td>
<td>4 GPUs</td>
<td><strong>0.825</strong></td>
</tr>
<tr>
<td>Experiment 3</td>
<td>80</td>
<td>0.00005</td>
<td>2</td>
<td>4 GPUs</td>
<td>0.706</td>
</tr>
</tbody>
</table>
</center>

Epoch Variation Results:
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
<td>80</td>
<td>0.0001</td>
<td>2</td>
<td>4 GPUs</td>
<td>0.825</td>
</tr>
<tr>
<td>Reduced Epochs in Experiment 2</td>
<td>30</td>
<td>0.0001</td>
<td>2</td>
<td>4 GPUs</td>
<td>0.287</td>
</tr>
<tr>
<td>Reduced Epochs in Experiment 2</td>
<td>50</td>
<td>0.0001</td>
<td>2</td>
<td>4 GPUs</td>
<td>0.545</td>
</tr>
<tr>
<td>Increased Epochs in Experiment 2</td>
<td>100</td>
<td>0.0001</td>
<td>2</td>
<td>4 GPUs</td>
<td>0.813</td>
</tr>
</tbody>
</table>
</center>

<b>Note: This tutorial is designed for 4 GPUs. If you only have 1 GPU, you can adjust the number of training GPUs to complete the experiments, but the final metrics may not align with the above, which is normal.</b>

## 6. Production Line Testing

Replace the model in the production line with the fine-tuned model for testing, e.g.:

```bash
python main.py -c paddlex/configs/modules/instance_segmentation/Mask-RT-DETR-H.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/remotesensing_demo.png"
```

The prediction results will be generated under `./output`, and the prediction result for `remotesensing_demo.png` is as follows:
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/instance_segmentation/03.png" width="600"/>

</center>

## 7. Development Integration/Deployment

If the general instance segmentation pipeline meets your requirements for inference speed and accuracy, you can proceed with development integration/deployment.

1. If you need to use the fine-tuned model weights, you can obtain the pipeline configuration file for instance segmentation and load it for prediction. You can execute the following command to save the results in `my_path`:

```bash
paddlex --get_pipeline_config instance_segmentation --save_path ./my_path
```

Fill in the local path of the fine-tuned model weights in the `model_dir` of the pipeline configuration file. If you want to directly apply the general instance segmentation pipeline in your Python project, you can refer to the example below:

```yaml
pipeline_name: instance_segmentation

SubModules:
  InstanceSegmentation:
    module_name: instance_segmentation
    model_name: Mask-RT-DETR-S
    model_dir: null # Replace this with the local path to your trained model weights
    batch_size: 1
    threshold: 0.5
```

Then, in your Python code, you can use the pipeline as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="my_path/instance_segmentation.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/remotesensing_demo.png")
for res in output:
    res.print() # Print the structured output of the prediction
    res.save_to_img("./output/") # Save the result as a visualized image
    res.save_to_json("./output/") # Save the structured output of the prediction
```
For more parameters, please refer to the [General Instance Segmentation Pipline User Guide](../pipeline_usage/tutorials/cv_pipelines/instance_segmentation.en.md)。

2. Additionally, PaddleX offers three other deployment methods, detailed as follows:

* high-performance inference: In actual production environments, many applications have stringent standards for deployment strategy performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end process acceleration. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../pipeline_deploy/high_performance_inference.en.md).
* Service-Oriented Deployment: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving cost-effective service-oriented deployment of production lines. For detailed service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../pipeline_deploy/serving.en.md).
* Edge Deployment: Edge deployment is a method that places computing and data processing capabilities directly on user devices, allowing devices to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../pipeline_deploy/edge_deploy.en.md).

You can select the appropriate deployment method for your model pipeline according to your needs, and proceed with subsequent AI application integration.
