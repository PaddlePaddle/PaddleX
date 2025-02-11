---
comments: true
---

# PaddleX 3.0 General Image Classification Pipeline — Garbage Classification Tutorial

PaddleX offers a rich set of pipelines, each consisting of one or more models that can tackle specific scenario tasks. All PaddleX pipelines support quick trials, and if the results are not satisfactory, you can fine-tune the models with your private data. PaddleX also provides Python APIs for easy integration into personal projects. Before use, you need to install PaddleX. For installation instructions, please refer to [PaddleX Installation](../installation/installation.en.md). This tutorial introduces the usage of the pipeline tool with a garbage classification task as an example.

## 1. Select a Pipeline

First, choose the corresponding PaddleX pipeline based on your task scenario. For garbage classification, it falls under the general image classification task, corresponding to PaddleX's universal image classification pipeline. If you're unsure about the task-pipeline correspondence, you can check the capabilities of relevant pipelines in the [PaddleX Supported Pipelines List](../support_list/pipelines_list.en.md).

## 2. Quick Start

PaddleX offers two ways to experience the pipelines: one is through the PaddleX wheel package locally, and the other is on the <b>Baidu AIStudio Community</b>.

- Local Experience:
    ```bash
    paddlex --pipeline image_classification \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/garbage_demo.png \
        --device gpu:0 \
        --save_path ./output/
    ```

- AIStudio Community Experience: Go to [Baidu AIStudio Community](https://aistudio.baidu.com/pipeline/mine), click "Create Pipeline", and create a <b>General Image Classification</b> pipeline for a quick trial.

Quick Trial Output Example:
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/image_classification/01.png" width=600>

</center>

After trying the pipeline, determine if it meets your expectations (including accuracy, speed, etc.). If the model's speed or accuracy is not satisfactory, you can test alternative models and decide if further fine-tuning is needed. Since this tutorial aims to classify specific garbage types, the default weights (trained on the ImageNet-1k dataset) are insufficient. You need to collect and annotate data for training and fine-tuning.

## 3. Choosing a Model

PaddleX provides 80 end-to-end image classification models, which can be referenced in the [Model List](../support_list/models_list.en.md). Some of the benchmarks for these models are as follows:

<table>
<thead>
<tr>
<th>Model List</th>
<th>Top-1 Accuracy (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-HGNetV2_B6</td>
<td>86.30</td>
<td>10.46</td>
<td>240.18</td>
<td>288</td>
</tr>
<tr>
<td>CLIP_vit_base_patch16_224</td>
<td>85.39</td>
<td>12.03</td>
<td>234.85</td>
<td>331</td>
</tr>
<tr>
<td>PP-HGNetV2_B4</td>
<td>83.57</td>
<td>2.45</td>
<td>38.10</td>
<td>76</td>
</tr>
<tr>
<td>SwinTransformer_base_patch4_window7_224</td>
<td>83.37</td>
<td>12.35</td>
<td>-</td>
<td>342</td>
</tr>
<tr>
<td>PP-HGNet_small</td>
<td>81.51</td>
<td>4.24</td>
<td>108.21</td>
<td>94</td>
</tr>
<tr>
<td>PP-HGNetV2_B0</td>
<td>77.77</td>
<td>0.68</td>
<td>6.41</td>
<td>23</td>
</tr>
<tr>
<td>ResNet50</td>
<td>76.50</td>
<td>3.12</td>
<td>50.90</td>
<td>98</td>
</tr>
<tr>
<td>PP-LCNet_x1_0</td>
<td>71.32</td>
<td>1.01</td>
<td>3.39</td>
<td>7</td>
</tr>
<tr>
<td>MobileNetV3_small_x1_0</td>
<td>68.24</td>
<td>1.09</td>
<td>3.65</td>
<td>12</td>
</tr>
</tbody>
</table>
> <b>Note: The above accuracy metrics are Top-1 Accuracy on the ImageNet-1k validation set. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>

In short, the models listed from top to bottom have faster inference speeds, while those from bottom to top have higher accuracy. This tutorial will use the `PP-LCNet_x1_0` model as an example to complete the full model development process. You can select an appropriate model for training based on your actual usage scenarios. After training, you can evaluate the suitable model weights within your pipeline and ultimately use them in real-world scenarios.

## 4. Data Preparation and Verification
### 4.1 Data Preparation

This tutorial uses the `Garbage Classification Dataset` as an example dataset. You can obtain the example dataset using the following commands. If you are using your own annotated dataset, you need to adjust it according to PaddleX's format requirements to meet PaddleX's data format specifications. For an introduction to data formats, you can refer to the [PaddleX Image Classification Task Module Data Annotation Tutorial](../data_annotations/cv_modules/image_classification.en.md).

Dataset acquisition commands:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/trash40.tar -P ./dataset
tar -xf ./dataset/trash40.tar -C ./dataset/
```

### 4.2 Dataset Verification

To verify the dataset, simply run the following command:

```bash
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/trash40/
```

After executing the above command, PaddleX will verify the dataset and count the basic information of the dataset. If the command runs successfully, it will print `Check dataset passed !` in the log, and the relevant outputs will be saved in the current directory's `./output/check_dataset` directory. The output directory includes visualized example images and a histogram of sample distribution. The verification result file is saved in `./output/check_dataset_result.json`, and the specific content of the verification result file is as follows:

```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": ../../dataset/trash40/label.txt",
    "num_classes": 40,
    "train_samples": 1605,
    "train_sample_paths": [
      "check_dataset/demo_img/img_14950.jpg",
      "check_dataset/demo_img/img_12045.jpg",
    ],
    "val_samples": 3558,
    "val_sample_paths": [
      "check_dataset/demo_img/img_198.jpg",
      "check_dataset/demo_img/img_19627.jpg",
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "trash40",
  "show_type": "image",
  "dataset_type": "ClsDataset"
}
```

In the above verification results, `check_pass` being `True` indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:

- `attributes.num_classes`: The number of classes in this dataset is 40, which is the number of classes that need to be passed in for subsequent training;
- `attributes.train_samples`: The number of training set samples in this dataset is 1605;
- `attributes.val_samples`: The number of validation set samples in this dataset is 3558;
- `attributes.train_sample_paths`: A list of relative paths to the visualized images of the training set samples in this dataset;
- `attributes.val_sample_paths`: A list of relative paths to the visualized images of the validation set samples in this dataset;

In addition, the dataset verification also analyzes the sample number distribution of all categories in the dataset and draws a distribution histogram (`histogram.png`):
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/image_classification/02.png" width=600>

</center>

<b>Note</b>: Only data that passes the verification can be used for training and evaluation.


### 4.3 Dataset Splitting (Optional)

If you need to convert the dataset format or re-split the dataset, you can set it by modifying the configuration file or appending hyperparameters.

The parameters related to dataset verification can be set by modifying the fields under `CheckDataset` in the configuration file. Some example explanations of the parameters in the configuration file are as follows:

* `CheckDataset`:
    * `split`:
        * `enable`: Whether to re-split the dataset. When set to `True`, the dataset format will be converted. The default is `False`;
        * `train_percent`: If the dataset is to be re-split, you need to set the percentage of the training set. The type is any integer between 0-100, and it needs to ensure that the sum with `val_percent` is 100;
        * `val_percent`: If the dataset is to be re-split, you need to set the percentage of the validation set. The type is any integer between 0-100, and it needs to ensure that the sum with `train_percent` is 100;

## Data Splitting
When splitting data, the original annotation files will be renamed as `xxx.bak` in the original path. The above parameters can also be set by appending command line arguments, for example, to re-split the dataset and set the ratio of training set to validation set: `-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`.

## 5. Model Training and Evaluation
### 5.1 Model Training

Before training, please ensure that you have validated the dataset. To complete PaddleX model training, simply use the following command:

```bash
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/trash40 \
    -o Train.num_classes=40
```

PaddleX supports modifying training hyperparameters, single/multi-GPU training, etc., by modifying the configuration file or appending command line arguments.

Each model in PaddleX provides a configuration file for model development to set relevant parameters. Model training-related parameters can be set by modifying the `Train` fields in the configuration file. Some example explanations of parameters in the configuration file are as follows:

* `Global`:
    * `mode`: Mode, supporting dataset validation (`check_dataset`), model training (`train`), and model evaluation (`evaluate`);
    * `device`: Training device, options include `cpu`, `gpu`, `xpu`, `npu`, `mlu`. For multi-GPU training, specify the card numbers, e.g., `gpu:0,1,2,3`;
* `Train`: Training hyperparameter settings;
    * `epochs_iters`: Number of training epochs;
    * `learning_rate`: Training learning rate;

For more hyperparameter introductions, please refer to [PaddleX General Model Configuration File Parameter Explanation](../module_usage/instructions/config_parameters_common.en.md).

<b>Note</b>:
- The above parameters can be set by appending command line arguments, e.g., specifying the mode as model training: `-o Global.mode=train`; specifying the first two GPUs for training: `-o Global.device=gpu:0,1`; setting the number of training epochs to 10: `-o Train.epochs_iters=10`.
- During model training, PaddleX automatically saves model weight files, with the default being `output`. If you need to specify a save path, you can use the `-o Global.output` field in the configuration file.
- PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.

<b>Explanation of Training Outputs</b>:

After completing model training, all outputs are saved in the specified output directory (default is `./output/`), typically including the following:

* train_result.json: Training result record file, recording whether the training task was completed normally, as well as the output weight metrics and related file paths;
* train.log: Training log file, recording changes in model metrics and loss during training;
* config.yaml: Training configuration file, recording the hyperparameter configuration for this training session;
* .pdparams, .pdopt, .pdstates, .pdiparams, .pdmodel: Model weight-related files, including network parameters, optimizer, static graph network parameters, static graph network structure, etc.;

### 5.2 Model Evaluation

After completing model training, you can evaluate the specified model weight file on the validation set to verify the model accuracy. To evaluate a model using PaddleX, simply use the following command:

```bash
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/trash40
```

Similar to model training, model evaluation supports setting by modifying the configuration file or appending command line arguments.

<b>Note</b>: When evaluating the model, you need to specify the model weight file path. Each configuration file has a default weight save path. If you need to change it, simply set it by appending a command line argument, e.g., `-o Evaluate.weight_path=./output/best_model/best_model.pdparams`.

### 5.3 Model Optimization

After learning about model training and evaluation, we can enhance model accuracy by adjusting hyperparameters. By carefully tuning the number of training epochs, you can control the depth of model training to avoid overfitting or underfitting. Meanwhile, the setting of the learning rate is crucial to the speed and stability of model convergence. Therefore, when optimizing model performance, it is essential to consider the values of these two parameters prudently and adjust them flexibly based on actual conditions to achieve the best training results.

It is recommended to follow the controlled variable method when debugging parameters:

1. First, fix the number of training epochs at 20, and set the batch size to 64 due to the small size of the training dataset.
2. Initiate three experiments based on the `PP-LCNet_x1_0` model, with learning rates of: 0.01, 0.001, 0.1.
3. It can be observed that the configuration with the highest accuracy in Experiment 1 is a learning rate of 0.01. Based on this training hyperparameter, change the number of epochs and observe the accuracy results at different epochs, finding that the best accuracy is generally achieved at 100 epochs.

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
<th>Top-1 Acc</th>
</tr>
</thead>
<tbody>
<tr>
<td>Experiment 1</td>
<td>20</td>
<td>0.01</td>
<td>64</td>
<td>4 GPUs</td>
<td><strong>73.83%</strong></td>
</tr>
<tr>
<td>Experiment 2</td>
<td>20</td>
<td>0.001</td>
<td>64</td>
<td>4 GPUs</td>
<td>30.64%</td>
</tr>
<tr>
<td>Experiment 3</td>
<td>20</td>
<td>0.1</td>
<td>64</td>
<td>4 GPUs</td>
<td>71.53%</td>
</tr>
</tbody>
</table>
</center>

Changing Epochs Experiment Results:
<center>

<table>
<thead>
<tr>
<th>Experiment</th>
<th>Epochs</th>
<th>Learning Rate</th>
<th>batch_size</th>
<th>Training Environment</th>
<th>Top-1 Acc</th>
</tr>
</thead>
<tbody>
<tr>
<td>Experiment 1</td>
<td>20</td>
<td>0.01</td>
<td>64</td>
<td>4 GPUs</td>
<td>73.83%</td>
</tr>
<tr>
<td>Experiment 1 (Increased Epochs)</td>
<td>50</td>
<td>0.01</td>
<td>64</td>
<td>4 GPUs</td>
<td>77.32%</td>
</tr>
<tr>
<td>Experiment 1 (Increased Epochs)</td>
<td>80</td>
<td>0.01</td>
<td>64</td>
<td>4 GPUs</td>
<td>77.60%</td>
</tr>
<tr>
<td>Experiment 1 (Increased Epochs)</td>
<td>100</td>
<td>0.01</td>
<td>64</td>
<td>4 GPUs</td>
<td><strong>77.80%</strong></td>
</tr>
</tbody>
</table>
</center>

> <b>Note: The above accuracy metrics are Top-1 Accuracy on the [ImageNet-1k](https://www.image-net.org/index.php) validation set. GPU inference time is based on an NVIDIA Tesla T4 machine, with FP32 precision. CPU inference speed is based on an Intel® Xeon® Gold 5117 CPU @ 2.00GHz, with 8 threads and FP32 precision.</b>

## 6. Production Line Testing

Replace the model in the production line with the fine-tuned model for testing. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/garbage_demo.png) to perform predictions:

```bash
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/garbage_demo.png"
```

The prediction results will be generated under `./output`, and the prediction result for `garbage_demo.png` is shown below:
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/image_classification/03.png" width="600"/>

</center>

## 7. Development Integration/Deployment
If the General Image Classification Pipeline meets your requirements for inference speed and accuracy in the production line, you can proceed directly with development integration/deployment.

1. If you need to use the fine-tuned model weights, you can obtain the configuration file for the image_classification pipeline and load it for prediction. You can execute the following command to save the results in my_path:

```
paddlex --get_pipeline_config image_classification --save_path ./my_path
```

Fill in the local path of the fine-tuned model weights into the `model_dir` field in the configuration file. If you need to directly apply the general image classification pipeline to your Python project, you can refer to the following example:

```yaml
pipeline_name: image_classification

SubModules:
  ImageClassification:
    module_name: image_classification
    model_name: PP-LCNet_x0_5
    model_dir: null # Replace this with the local path to your trained model weights
    batch_size: 4
    topk: 5
```
Subsequently, in your Python code, you can utilize the pipeline as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="my_path/image_classification.yaml")
output = pipeline.predict("./dataset/trash40/images/test/0/img_154.jpg")
for res in output:
    res.print() # Print the structured output of the prediction
    res.save_to_img("./output/") # Save the visualized image of the result
    res.save_to_json("./output/") # Save the structured output of the prediction
```

For more parameters, please refer to the [General Image Classification Pipeline Usage Tutorial](../pipeline_usage/tutorials/cv_pipelines/image_classification.en.md).

2. Additionally, PaddleX offers three other deployment methods, detailed as follows:

* high-performance inference: In actual production environments, many applications have stringent standards for deployment strategy performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugin aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end process acceleration. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../pipeline_deploy/high_performance_inference.en.md).
* Service-Oriented Deployment: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving cost-effective service-oriented deployment of production lines. For detailed service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../pipeline_deploy/serving.en.md).
* Edge Deployment: Edge deployment is a method that places computing and data processing capabilities directly on user devices, allowing devices to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../pipeline_deploy/edge_deploy.en.md).

You can select the appropriate deployment method for your model pipeline according to your needs, and proceed with subsequent AI application integration.
