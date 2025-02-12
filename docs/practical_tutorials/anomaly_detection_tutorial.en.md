---
comments: true
---

# PaddleX 3.0 Image Anomaly Detection Pipeline — Food Appearance Quality Inspection Tutorial

PaddleX offers a rich set of pipelines, each consisting of one or more models that can solve specific scenario tasks. All PaddleX pipelines support quick trials, and if the results do not meet expectations, fine-tuning the models with private data is also supported. PaddleX provides Python APIs for easy integration into personal projects. Before use, you need to install PaddleX. For installation instructions, refer to [PaddleX Installation](../installation/installation.en.md). This tutorial introduces the usage of the pipeline tool with an example of a Food Appearance Quality Inspection task.

## 1. Select a Pipeline

First, choose the corresponding PaddleX pipeline based on your task scenario. For Food Appearance Quality Inspection, this falls under the category of anomaly detection tasks, corresponding to PaddleX's Image Anomaly Detection Pipeline. If you are unsure about the correspondence between tasks and pipelines, you can refer to the [Pipeline List](../support_list/pipelines_list.en.md) for an overview of pipeline capabilities.

## 2. Quick Start

PaddleX offers 1 ways to experience the pipeline: one is through the PaddleX wheel package locally.

- Local Experience:
  ```bash
  paddlex --pipeline anomaly_detection \
      --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_hazelnut.png \
      --save_path output
  ```


Quick trial output example:
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/image_anomaly_detection/01.png" width=600>

</center>

After experiencing the pipeline, determine if it meets your expectations (including accuracy, speed, etc.). If the model's speed or accuracy does not meet your requirements, you can select alternative models for further testing. If the final results are unsatisfactory, you may need to fine-tune the model. This tutorial aims to produce a model that segments lane lines, and the default weights (trained on the Cityscapes dataset) cannot meet this requirement. Therefore, you need to collect and annotate data for training and fine-tuning.

## 3. Choose a Model

PaddleX provides 1 end-to-end anomaly detection models. For details, refer to the [Model List](../support_list/models_list.en.md). Some model benchmarks are as follows:

<table>
<thead>
<tr>
<th>Model</th>
<th>mIoU</th>
<th>Model Storage Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>STFPM</td>
<td>0.9901</td>
<td>22.5</td>
</tr>
</tbody>
</table>
<b>The above model accuracy metrics are measured from the grid category in the MVTec_AD dataset.</b>

> **Note: The above accuracy metrics are measured on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.**

## 4. Data Preparation and Verification
### 4.1 Data Preparation

This tutorial uses the "Food Appearance Quality Inspection Dataset" as an example dataset. You can obtain the example dataset using the following commands. If you use your own annotated dataset, you need to adjust it according to PaddleX's format requirements to meet PaddleX's data format specifications. For an introduction to data formats, you can refer to [PaddleX anomaly detection Task Module Data Annotation Tutorial](../data_annotations/cv_modules/semantic_segmentation.en.md).

Dataset acquisition commands:
```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/anomaly_detection_hazelnut.tar -P ./dataset
tar -xf ./dataset/anomaly_detection_hazelnut.tar -C ./dataset/
```

### 4.2 Dataset Verification

To verify the dataset, simply use the following command:

```bash
python main.py -c paddlex/configs/modules/image_anomaly_detection/STFPM.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/anomaly_detection_hazelnut
```

After executing the above command, PaddleX will verify the dataset and collect basic information about it. Upon successful execution, the log will print "Check dataset passed !" information, and relevant outputs will be saved in the current directory's `./output/check_dataset` directory, including visualized sample images and sample distribution histograms. The verification result file is saved in `./output/check_dataset_result.json`, and the specific content of the verification result file is

```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_sample_paths": [
      "check_dataset/demo_img/294.png",
      "check_dataset/demo_img/260.png",
      "check_dataset/demo_img/297.png",
      "check_dataset/demo_img/170.png",
      "check_dataset/demo_img/068.png",
      "check_dataset/demo_img/212.png",
      "check_dataset/demo_img/204.png",
      "check_dataset/demo_img/233.png",
      "check_dataset/demo_img/367.png",
      "check_dataset/demo_img/383.png"
    ],
    "train_samples": 391,
    "val_sample_paths": [
      "check_dataset/demo_img/012.png",
      "check_dataset/demo_img/017.png",
      "check_dataset/demo_img/006.png",
      "check_dataset/demo_img/013.png",
      "check_dataset/demo_img/014.png",
      "check_dataset/demo_img/010.png",
      "check_dataset/demo_img/007.png",
      "check_dataset/demo_img/001.png",
      "check_dataset/demo_img/002.png",
      "check_dataset/demo_img/009.png"
    ],
    "val_samples": 70,
    "num_classes": 1
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "anomaly_detection_hazelnut",
  "show_type": "image",
  "dataset_type": "SegDataset"
}
```

In the verification results above, `check_pass` being `True` indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:

- `attributes.num_classes`: The number of classes in this dataset is 1, which is the number of classes that need to be passed for subsequent training;
- `attributes.train_samples`: The number of samples in the training set of this dataset is 391;
- `attributes.val_samples`: The number of samples in the validation set of this dataset is 70;
- `attributes.train_sample_paths`: A list of relative paths to the visualization images of samples in the training set of this dataset;
- `attributes.val_sample_paths`: A list of relative paths to the visualization images of samples in the validation set of this dataset;

Additionally, the dataset verification also analyzes the sample distribution across all classes and plots a histogram (`histogram.png`):
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/image_anomaly_detection/02.png" width=600>

</center>

**Note**: Only data that passes verification can be used for training and evaluation.


### 4.3 Dataset Format Conversion / Dataset Splitting (Optional)

If you need to convert the dataset format or re-split the dataset, you can set it by modifying the configuration file or appending hyperparameters.

Parameters related to dataset verification can be set by modifying the fields under `CheckDataset` in the configuration file. Examples of some parameters in the configuration file are as follows:

* `CheckDataset`:
    * `convert`:
        * `enable`: Whether to convert the dataset format. Set to `True` to enable dataset format conversion, default is `False`;
        * `src_dataset_type`: If dataset format conversion is enabled, the source dataset format must be set. Available source formats are `LabelMe` and `VOC`;
    * `split`:
        * `enable`: Whether to re-split the dataset. Set to `True` to enable dataset splitting, default is `False`;
        * `train_percent`: If dataset splitting is enabled, the percentage of the training set must be set. The type is any integer between 0-100, and the sum with `val_percent` must be 100;
        * `val_percent`: If dataset splitting is enabled, the percentage of the validation set must be set. The type is any integer between 0-100, and the sum with `train_percent` must be 100;

Data conversion and splitting can be enabled simultaneously. For data splitting, the original annotation files will be renamed to `xxx.bak` in their original paths. These parameters also support being set by appending command-line arguments, for example, to re-split the dataset and set the training and validation set ratios: `-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`.

## 5. Model Training and Evaluation

### 5.1 Model Training

Before training, ensure that you have validated your dataset. To complete the training of a PaddleX model, simply use the following command:

```bash
python main.py -c paddlex/configs/modules/image_anomaly_detection/STFPM.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/anomaly_detection_hazelnut \
    -o Train.epochs_iters=4000
```

PaddleX supports modifying training hyperparameters, single/multi-GPU training, and more, simply by modifying the configuration file or appending command line arguments.

Each model in PaddleX provides a configuration file for model development, which is used to set relevant parameters. Model training-related parameters can be set by modifying the `Train` fields in the configuration file. Some example parameter descriptions in the configuration file are as follows:

* `Global`:
    * `mode`: Mode, supports data verification (`check_dataset`), model training (`train`), model evaluation (`evaluate`), and model inference (`predict`);
    * `device`: Training device, options include `cpu`, `gpu`, `xpu`, `npu`, `mlu`. For multi-GPU training, specify card numbers, e.g., `gpu:0,1,2,3`;
* `Train`: Training hyperparameter settings;
    * `epochs_iters`: Number of training iterations;
    * `learning_rate`: Training learning rate;

For more hyperparameter introductions, refer to [PaddleX General Model Configuration File Parameter Explanation](../module_usage/instructions/config_parameters_common.en.md).

**Note**:
- The above parameters can be set by appending command line arguments, e.g., specifying the mode as model training: `-o Global.mode=train`; specifying the first two GPUs for training: `-o Global.device=gpu:0,1`; setting the number of training iterations to 5000: `-o Train.epochs_iters=5000`.
- During model training, PaddleX automatically saves model weight files, with the default being `output`. To specify a save path, use the `-o Global.output` field in the configuration file.
- PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.

**Training Outputs Explanation**:

After completing model training, all outputs are saved in the specified output directory (default is `./output/`), typically including the following:

* train_result.json: Training result record file, recording whether the training task completed normally, as well as the output weight metrics, relevant file paths, etc.;
* train.log: Training log file, recording changes in model metrics, loss, etc. during training;
* config.yaml: Training configuration file, recording the hyperparameter configuration for this training session;
* .pdparams, .pdema, .pdopt.pdstate, .pdiparams, .pdmodel: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;

### 5.2 Model Evaluation

After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. To evaluate a model using PaddleX, simply use the following command:

```bash
python main.py -c paddlex/configs/modules/image_anomaly_detection/STFPM.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/anomaly_detection_hazelnut
```

Similar to model training, model evaluation supports setting parameters by modifying the configuration file or appending command line arguments.

**Note**: When evaluating a model, you need to specify the model weight file path. Each configuration file has a default weight save path. If you need to change it, simply set it by appending a command line argument, e.g., `-o Evaluate.weight_path=./output/best_model/model.pdparams`.

### 5.3 Model Optimization

After learning about model training and evaluation, we can enhance model accuracy by adjusting hyperparameters. By carefully tuning the number of training epochs, you can control the depth of model training, avoiding overfitting or underfitting. Meanwhile, the setting of the learning rate is crucial to the speed and stability of model convergence. Therefore, when optimizing model performance, it is essential to consider the values of these two parameters prudently and adjust them flexibly based on actual conditions to achieve the best training results.

It is recommended to follow the method of controlled variables when debugging parameters:

1. First, fix the number of training iterations at 4000 and the batch size at 1.
2. Initiate three experiments based on the STFPM model, with learning rates of: 0.01, 0.1, 0.4.
3. It can be observed that the configuration with the highest accuracy in Experiment 3 is a learning rate of 0.4. Based on this training hyperparameter, change the number of training epochs and observe the accuracy results of different iterations, finding that the optimal accuracy is basically achieved at 5000 iterations.

Learning Rate Exploration Results:
<center>

| Experiment | Iterations | Learning Rate | batch\_size | Training Environment | mIoU |
|-----------|------------|-------------|-----------|--------------------|------|
| Experiment 1 | 4000 | 0\.01 | 1        | 4    | 0\.9646   |
| Experiment 2 |4000 | 0\.1 | 1        | 4    |0\.9707|
| Experiment 3 | 4000 | 0\.4  | 1        | 4   | **0\.9797**   |

</center>

Changing Epoch Results:
<center>

| Experiment                   | Iterations | Learning Rate | batch\_size | Training Environment | mIoU |
|--------------------|---------|-------|------------|------|----------|
| Experiment 3                 | 4000    | 0\.4 | 1         | 4   | 0\.9797   |
| Experiment 3 with more epochs  | 5000   | 0\.4 | 1         | 4   | **0\.9826**|

</center>

**Note: This tutorial is designed for 4 GPUs. If you have only 1 GPU, you can adjust the number of training GPUs to complete the experiment, but the final metrics may not align with the above indicators, which is normal.**

## 6. Production Line Testing

Replace the model in the production line with the fine-tuned model for testing. You can obtain the anomaly_detection production configuration file and load the configuration file for prediction. You can execute the following command to save the configuration in `my_path`:

```
paddlex --get_pipeline_config anomaly_detection --save_path ./my_path
```

Modify the `SubModules.AnomalyDetection.model_dir` in the configuration file `my_path/anomaly_detection.yaml` to your model path `output/best_model/inference`:

pipeline_name: anomaly_detection

```yaml
SubModules:
  AnomalyDetection:
    module_name: anomaly_detection
    model_name: STFPM
    model_dir: output/best_model/inference  # Replace with the fine-tuned image anomaly detection model weight path
    batch_size: 1
```

Subsequently, in the Python code, you can call the production line as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/anomaly_detection.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_hazelnut.png")
for res in output:
    res.print()  ## Print the structured output of the prediction
    res.save_to_img("./output/")  ## Save the visualized result image
    res.save_to_json("./output/")  ## Save the structured output of the prediction
```

The prediction results will be generated under `./output`, where the prediction result for `uad_hazelnut.png` is shown below:
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/image_anomaly_detection/03.png" width="600"/>

</center>

## 7. Development Integration/Deployment
If the anomaly detection pipeline meets your requirements for inference speed and accuracy in the production line, you can proceed directly with development integration/deployment.
1. Directly apply the trained model in your Python project. You can refer to the following example:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="my_path/anomaly_detection.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_hazelnut.png")
for res in output:
    res.print() # Print the structured output of the prediction
    res.save_to_img("./output/") # Save the visualized image of the result
    res.save_to_json("./output/") # Save the structured output of the prediction
```
For more parameters, please refer to [Anomaly Detection Pipeline Usage Tutorial](../pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.en.md).

1. Additionally, PaddleX offers three other deployment methods, detailed as follows:

* high-performance inference: In actual production environments, many applications have stringent standards for deployment strategy performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end process acceleration. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../pipeline_deploy/high_performance_inference.en.md).
* Service-Oriented Deployment: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving cost-effective service-oriented deployment of production lines. For detailed service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../pipeline_deploy/serving.en.md).
* Edge Deployment: Edge deployment is a method that places computing and data processing capabilities directly on user devices, allowing devices to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../pipeline_deploy/edge_deploy.en.md).

You can select the appropriate deployment method for your model pipeline according to your needs, and proceed with subsequent AI application integration.
