# PaddleX 3.0 General OCR Pipeline — Handwritten Chinese Recognition Tutorial

PaddleX offers a rich set of pipelines, each consisting of one or more models that collectively solve specific scenario tasks. All PaddleX pipelines support quick trials, and if the results do not meet expectations, they also support fine-tuning with private data. PaddleX provides Python APIs for easy integration into personal projects. Before use, you need to install PaddleX. For installation instructions, please refer to [PaddleX Installation](../installation/installation_en.md). This tutorial introduces the usage of the pipeline tool with an example of handwritten Chinese recognition.

## 1. Select a Pipeline

First, choose the corresponding PaddleX pipeline based on your task scenario. For handwritten Chinese recognition, this task falls under the Text Recognition category, corresponding to PaddleX's Universal OCR Pipeline. If you are unsure about the correspondence between tasks and pipelines, you can refer to the [Pipeline List](../support_list/pipelines_list_en.md) for an overview of pipeline capabilities.

## 2. Quick Start

PaddleX offers two ways to experience the pipeline: one is through the PaddleX wheel package locally, and the other is on the **Baidu AIStudio Community**.

- Local Experience:
    ```bash
    paddlex --pipeline OCR \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/OCR_rec/case.png
    ```

- AIStudio Community Experience: Go to [Baidu AIStudio Community](https://aistudio.baidu.com/pipeline/mine), click "Create Pipeline", and create a **Universal OCR** pipeline for a quick trial.

Quick trial output example:
<center>

<img src="https://github.com/user-attachments/assets/a3210910-a76c-4552-b7a5-fe3670205584" width=600>

</center>

After experiencing the pipeline, determine if it meets your expectations (including accuracy, speed, etc.). If the pipeline's models need further fine-tuning due to unsatisfactory speed or accuracy, select alternative models for continued testing to determine satisfaction. If the final results are unsatisfactory, fine-tuning the model is necessary.

## 3. Select a Model

PaddleX provides four end-to-end OCR models. For details, refer to the [Model List](../support_list/models_list_en.md). Benchmarks for some models are as follows:

| Model List         | Detection Hmean(%) | Recognition Avg Accuracy(%) | GPU Inference Time(ms) | CPU Inference Time(ms) | Model Size(M) |
| --------------- | ----------- | ------------------- | --------------- | --------------- |---------------|
|PP-OCRv4_server |     82.69     | 79.20     | 22.20346     | 2662.158     | 198|
|PP-OCRv4_mobile     | 77.79     | 78.20 |     2.719474 |     79.1097     | 15|

**Note: The evaluation set is a self-built Chinese dataset by PaddleOCR, covering street scenes, web images, documents, and handwritten texts. The text recognition set contains 11,000 images, and the detection set contains 500 images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

In summary, models listed from top to bottom have faster inference speeds, while those from bottom to top have higher accuracy. This tutorial uses the `PP-OCRv4_server` model as an example to complete a full model development process. Based on your actual usage scenario, choose a suitable model for training. After training, evaluate the appropriate model weights within the pipeline and use them in practical scenarios.

## 4. Data Preparation and Verification

### 4.1 Data Preparation

This tutorial uses the "Handwritten Chinese Recognition Dataset" as an example dataset. You can obtain the example dataset using the following commands. If you use your own annotated dataset, you need to adjust it according to the PaddleX format requirements to meet PaddleX's data format specifications. For an introduction to data formats, you can refer to [PaddleX Text Detection/Text Recognition Task Module Data Annotation Tutorial](../data_annotations/ocr_modules/text_detection_recognition_en.md).

Dataset acquisition commands:
```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/handwrite_chinese_text_rec.tar -P ./dataset
tar -xf ./dataset/handwrite_chinese_text_rec.tar -C ./dataset/
```

### 4.2 Dataset Verification

To verify the dataset, simply use the following command:

```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_server_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/handwrite_chinese_text_rec
```

After executing the above command, PaddleX will verify the dataset and collect basic information about it. Upon successful execution, the log will print "Check dataset passed !" information, and relevant outputs will be saved in the current directory's `./output/check_dataset` directory, including visualized sample images and sample distribution histograms. The verification result file is saved in `./output/check_dataset_result.json`, and the specific content of the verification result file is:
```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 23965,
    "train_sample_paths": [
      "..\/..\/handwrite_chinese_text_rec\/train_data\/64957.png",
      "..\/..\/handwrite_chinese_text_rec\/train_data\/138926.png",
      "..\/..\/handwrite_chinese_text_rec\/train_data\/86760.png",
      "..\/..\/handwrite_chinese_text_rec\/train_data\/83191.png",
      "..\/..\/handwrite_chinese_text_rec\/train_data\/79882.png",
      "..\/..\/handwrite_chinese_text_rec\/train_data\/58639.png",
      "..\/..\/handwrite_chinese_text_rec\/train_data\/1187-P16_1.jpg",
      "..\/..\/handwrite_chinese_text_rec\/train_data\/8199.png",
      "..\/..\/handwrite_chinese_text_rec\/train_data\/1225-P19_9.jpg",
      "..\/..\/handwrite_chinese_text_rec\/train_data\/183335.png"
    ],
    "val_samples": 17259,
    "val_sample_paths": [
      "..\/..\/handwrite_chinese_text_rec\/test_data\/11.png",
      "..\/..\/handwrite_chinese_text_rec\/test_data\/12.png",
      "..\/..\/handwrite_chinese_text_rec\/test_data\/13.png",
      "..\/..\/handwrite_chinese_text_rec\/test_data\/14.png",
      "..\/..\/handwrite_chinese_text_rec\/test_data\/15.png",
      "..\/..\/handwrite_chinese_text_rec\/test_data\/16.png",
      "..\/..\/handwrite_chinese_text_rec\/test_data\/17.png",
      "..\/..\/handwrite_chinese_text_rec\/test_data\/18.png",
      "..\/..\/handwrite_chinese_text_rec\/test_data\/19.png",
      "..\/..\/handwrite_chinese_text_rec\/test_data\/20.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "\/mnt\/liujiaxuan01\/new\/new2\/handwrite_chinese_text_rec",
  "show_type": "image",
  "dataset_type": "MSTextRecDataset"
}
```

In the above verification results, `check_pass` being `True` indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:

- `attributes.train_samples`: The number of samples in the training set of this dataset is 23965;
- `attributes.val_samples`: The number of samples in the validation set of this dataset is 17259;
- `attributes.train_sample_paths`: The list of relative paths to the visualization images of the samples in the training set of this dataset;
- `attributes.val_sample_paths`: The list of relative paths to the visualization images of the samples in the validation set of this dataset;

Additionally, the dataset verification also analyzes the distribution of sample numbers across all categories in the dataset and plots a histogram (`histogram.png`):
<center>

<img src="https://github.com/user-attachments/assets/1734db3d-59f1-4278-ace1-741cf57755db" width=600>

</center>

**Note**: Only data that passes the verification can be used for training and evaluation.


### 4.3 Dataset Splitting (Optional)

If you need to convert the dataset format or re-split the dataset, you can set it by modifying the configuration file or appending hyperparameters.

Parameters related to dataset verification can be set by modifying the fields under `CheckDataset` in the configuration file. Examples of some parameters in the configuration file are as follows:

* `CheckDataset`:
    * `split`:
        * `enable`: Whether to re-split the dataset. Set to `True` to convert the dataset format, default is `False`;
        * `train_percent`: If re-splitting the dataset, you need to set the percentage of the training set, which is an integer between 0-100, and needs to ensure that the sum with `val_percent` is 100;
        * `val_percent`: If re-splitting the dataset, you need to set the percentage of the validation set, which is an integer between 0-100, and needs to ensure that the sum with `train_percent` is 100;

During data splitting, the original annotation files will be renamed to `xxx.bak` in their original paths. The above parameters also support being set by appending command-line arguments, for example, to re-split the dataset and set the training and validation set ratios: `-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`.

## 5. Model Training and Evaluation

### 5.1 Model Training

Before training, ensure that you have validated your dataset. To complete PaddleX model training, simply use the following command:

```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_server_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/handwrite_chinese_text_rec
```

PaddleX supports modifying training hyperparameters, single/multi-GPU training, etc., by modifying the configuration file or appending command-line arguments.

Each model in PaddleX provides a configuration file for model development to set relevant parameters. Model training-related parameters can be set by modifying the `Train` fields in the configuration file. Some example parameter descriptions in the configuration file are as follows:

* `Global`:
    * `mode`: Mode, supporting dataset validation (`check_dataset`), model training (`train`), and model evaluation (`evaluate`);
    * `device`: Training device, options include `cpu`, `gpu`, `xpu`, `npu`, `mlu`. For multi-GPU training, specify card numbers, e.g., `gpu:0,1,2,3`;
* `Train`: Training hyperparameter settings;
    * `epochs_iters`: Number of training epochs;
    * `learning_rate`: Training learning rate;

For more hyperparameter introductions, refer to [PaddleX Hyperparameter Introduction](../module_usage/instructions/config_parameters_common_en.md).

**Note**:
- The above parameters can be set by appending command-line arguments, e.g., specifying the mode as model training: `-o Global.mode=train`; specifying the first two GPUs for training: `-o Global.device=gpu:0,1`; setting the number of training epochs to 10: `-o Train.epochs_iters=10`.
- During model training, PaddleX automatically saves model weight files, with the default being `output`. To specify a save path, use the `-o Global.output` field in the configuration file.
- PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.

**Training Output Explanation**:  

After completing model training, all outputs are saved in the specified output directory (default is `./output/`), typically including the following:

* train_result.json: Training result record file, recording whether the training task completed normally, as well as the output weight metrics, relevant file paths, etc.;
* train.log: Training log file, recording changes in model metrics and loss during training;
* config.yaml: Training configuration file, recording the hyperparameter configuration for this training session;
* .pdparams, .pdopt, .pdstates, .pdiparams, .pdmodel: Model weight-related files, including network parameters, optimizer, static graph network parameters, static graph network structure, etc.;

### 5.2 Model Evaluation

After completing model training, you can evaluate the specified model weight file on the validation set to verify the model accuracy. To evaluate a model using PaddleX, simply use the following command:

```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_server_rec.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/handwrite_chinese_text_rec
```

Similar to model training, model evaluation supports setting parameters by modifying the configuration file or appending command-line arguments.

**Note**: When evaluating a model, you need to specify the model weight file path. Each configuration file has a default weight save path. If you need to change it, simply set it by appending a command-line argument, e.g., `-o Evaluate.weight_path=./output/best_accuracy/best_accuracy.pdparams`.

### 5.3 Model Optimization

After learning about model training and evaluation, we can enhance model accuracy by adjusting hyperparameters. By carefully tuning the number of training epochs, you can control the depth of model training to avoid overfitting or underfitting. Meanwhile, the setting of the learning rate is crucial for the speed and stability of model convergence. Therefore, when optimizing model performance, it is essential to consider the values of these two parameters carefully and adjust them flexibly based on actual conditions to achieve the best training results.

It is recommended to follow the method of controlled variables when debugging parameters:

1. First, fix the number of training epochs at 20, the batch size at 8, select 4 GPUs, and the total batch size is 32.
2. Initiate four experiments based on the PP-OCRv4_server_rec model with learning rates of: 0.001, 0.005, 0.0002, 0.0001.
3. It can be observed that Experiment 3, with a learning rate of 0.0002, yields the highest accuracy, and the validation set score indicates that accuracy continues to increase in the last few epochs. Therefore, increasing the number of training epochs to 30, 50, and 80 will further improve model accuracy.

Learning Rate Exploration Results:
<center>

| Experiment ID | Learning Rate | Recognition Acc (%) |
|---------------|-------------|-------------------|
| 1             | 0.001       | 43.28             |
| 2             | 0.005       | 32.63             |
| 3             | 0.0002      | 49.64             |
| 4             | 0.0001      | 46.32             |
</center>

Next, based on a learning rate of 0.0002, we can increase the number of training epochs. Comparing Experiments [4, 5, 6, 7] below, it can be seen that increasing the number of training epochs further improves model accuracy.
<center>

| Experiment ID | Number of Training Epochs | Recognition Acc (%) |
|---------------|---------------------------|-------------------|
| 4             | 20                        | 49.64             |
| 5             | 30                        | 52.03             |
| 6             | 50                        | 54.15             |
| 7             | 80                        | 54.35             |
</center>

**Note: This tutorial is designed for 4 GPUs. If you only have 1 GPU, you can adjust the number of training GPUs to complete the experiments, but the final metrics may not align with the above indicators, which is normal.**

## 6. Production Line Testing

Replace the model in the production line with the fine-tuned model for testing, for example:

```bash
paddlex --pipeline OCR \
        --model PP-OCRv4_server_det PP-OCRv4_server_rec \
        --model_dir None output/best_accuracy/inference \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/OCR_rec/case.png
```

The prediction results will be generated under `./output`, and the prediction result for `case.jpg` is shown below:
<center>

<img src="https://github.com/user-attachments/assets/a0c28495-6352-4c64-b53e-9903da3e002a" width="600"/>

</center>

## 7. Development Integration/Deployment
If the general OCR pipeline meets your requirements for inference speed and accuracy in the production line, you can proceed directly with development integration/deployment.
1. If you need to apply the general OCR pipeline directly in your Python project, you can refer to the following sample code:
```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="paddlex/pipelines/OCR.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/OCR_rec/case.png")
for res in output:
    res.print() # Print the structured output of the prediction
    res.save_to_img("./output/") # Save the visualized image of the result
    res.save_to_json("./output/") # Save the structured output of the prediction
```
For more parameters, please refer to the [General OCR Pipeline Usage Tutorial](../pipeline_usage/tutorials/ocr_pipelines/OCR_en.md).

2. Additionally, PaddleX also offers service-oriented deployment methods, detailed as follows:

* High-Performance Deployment: In actual production environments, many applications have stringent standards for deployment strategy performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end process acceleration. For detailed high-performance deployment procedures, please refer to the [PaddleX High-Performance Deployment Guide](../pipeline_deploy/high_performance_deploy_en.md).
* Service-Oriented Deployment: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving cost-effective service-oriented deployment of production lines. For detailed service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../pipeline_deploy/service_deploy_en.md).
* Edge Deployment: Edge deployment is a method that places computing and data processing capabilities directly on user devices, allowing devices to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../pipeline_deploy/lite_deploy_en.md).

You can select the appropriate deployment method for your model pipeline according to your needs, and proceed with subsequent AI application integration.
