---
comments: true
---

# PaddleX 3.0 General OCR Pipeline — License Plate Recognition Tutorial

PaddleX provides a rich set of pipelines, each consisting of one or more models that work together to solve specific scenario tasks. All PaddleX pipelines support quick trials, and if the results do not meet expectations, you can also fine-tune the models with private data. PaddleX provides Python APIs to easily integrate pipelines into personal projects. Before use, you need to install PaddleX. For installation instructions, refer to [PaddleX Installation](../installation/installation.en.md). This tutorial introduces the usage of the pipeline tool with a license plate recognition task as an example.

## 1. Select a Pipeline

First, choose the corresponding PaddleX pipeline based on your task scenario. For license plate recognition, this task falls under text detection, corresponding to PaddleX's Universal OCR pipeline. If you are unsure about the correspondence between tasks and pipelines, you can refer to the [Pipeline List](../support_list/pipelines_list.en.md) supported by PaddleX to understand the capabilities of relevant pipelines.

## 2. Quick Start

PaddleX offers two ways to experience the pipeline: one is through the PaddleX wheel package locally, and the other is on the <b>Baidu AIStudio Community</b>.

- Local Experience:
    ```bash
    paddlex --pipeline OCR \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/OCR/case1.jpg
    ```

- AIStudio Community Experience: Go to [Baidu AIStudio Community](https://aistudio.baidu.com/pipeline/mine), click "Create Pipeline", and create a <b>Universal OCR</b> pipeline for a quick trial.

Quick trial output example:
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/ocr/01.png" width=600>

</center>

After experiencing the pipeline, determine if it meets your expectations (including accuracy, speed, etc.), and whether the models included in the pipeline need further fine-tuning. If the speed or accuracy of the models does not meet expectations, select replaceable models for continued testing to determine satisfaction. If the final results are unsatisfactory, fine-tune the models.

## 3. Select a Model

PaddleX provides 4 end-to-end text detection models. For details, refer to the [Model List](../support_list/models_list.en.md). The benchmarks of the models are as follows:

<table>
<thead>
<tr>
<th>Model</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_det</td>
<td>82.56</td>
<td>83.3501</td>
<td>2434.01</td>
<td>109</td>
<td>The server-side text detection model of PP-OCRv4, with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td>77.35</td>
<td>10.6923</td>
<td>120.177</td>
<td>4.7</td>
<td>The mobile text detection model of PP-OCRv4, with higher efficiency, suitable for deployment on edge devices</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_det</td>
<td>78.68</td>
<td>-</td>
<td>-</td>
<td>2.1</td>
<td>The mobile text detection model of PP-OCRv3, with higher efficiency, suitable for deployment on edge devices</td>
</tr>
<tr>
<td>PP-OCRv3_server_det</td>
<td>80.11</td>
<td>-</td>
<td>-</td>
<td>102.1</td>
<td>The server-side text detection model of PP-OCRv3, with higher accuracy, suitable for deployment on high-performance servers</td>
</tr>
</tbody>
</table>

In short, the models listed from top to bottom have faster inference speeds, while from bottom to top, they have higher accuracy. This tutorial uses the `PP-OCRv4_server` model as an example to complete a full model development process. Depending on your actual usage scenario, choose a suitable model for training. After training, evaluate the appropriate model weights within the pipeline and use them in practical scenarios.

## 4. Data Preparation and Validation
### 4.1 Data Preparation

This tutorial uses the "License Plate Recognition Dataset" as an example dataset. You can obtain the example dataset using the following commands. If you use your own annotated dataset, you need to adjust it according to the PaddleX format requirements to meet PaddleX's data format specifications. For information on data format, you can refer to [PaddleX Text Detection/Text Recognition Task Module Data Annotation Tutorial](../data_annotations/ocr_modules/text_detection_recognition.en.md).

Dataset acquisition commands:
```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ccpd_text_det.tar -P ./dataset
tar -xf ./dataset/ccpd_text_det.tar -C ./dataset/
```

### 4.2 Dataset Validation

To validate the dataset, simply use the following command:

```bash
python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_server_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ccpd_text_det
```

After executing the above command, PaddleX will validate the dataset and collect basic information about it. Upon successful execution, the log will print "Check dataset passed !" information, and relevant outputs will be saved in the current directory's `./output/check_dataset` directory, including visualized sample images and sample distribution histograms. The validation result file is saved in `./output/check_dataset_result.json`, and the specific content of the validation result file is

```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 5769,
    "train_sample_paths": [
      "check_dataset\/demo_img\/0274305555556-90_266-204&460_520&548-516&548_209&547_204&464_520&460-0_0_3_25_24_24_24_26-63-89.jpg",
      "check_dataset\/demo_img\/0126171875-90_267-294&424_498&486-498&486_296&485_294&425_496&424-0_0_3_24_33_32_30_31-157-29.jpg",
      "check_dataset\/demo_img\/0371516927083-89_254-178&423_517&534-517&534_204&525_178&431_496&423-1_0_3_24_33_31_29_31-117-667.jpg",
      "check_dataset\/demo_img\/03349609375-90_268-211&469_526&576-526&567_214&576_211&473_520&469-0_0_3_27_31_32_29_32-174-48.jpg",
      "check_dataset\/demo_img\/0388454861111-90_269-138&409_496&518-496&518_138&517_139&410_491&409-0_0_3_24_27_26_26_30-174-148.jpg",
      "check_dataset\/demo_img\/0198741319444-89_112-208&517_449&600-423&593_208&600_233&517_449&518-0_0_3_24_28_26_26_26-87-268.jpg",
      "check_dataset\/demo_img\/3027782118055555555-91_92-186&493_532&574-529&574_199&565_186&497_532&493-0_0_3_27_26_30_33_32-73-336.jpg",
      "check_dataset\/demo_img\/034375-90_258-168&449_528&546-528&542_186&546_168&449_525&449-0_0_3_26_30_30_26_33-94-221.jpg",
      "check_dataset\/demo_img\/0286501736111-89_92-290&486_577&587-576&577_290&587_292&491_577&486-0_0_3_17_25_28_30_33-134-122.jpg",
      "check_dataset\/demo_img\/02001953125-92_103-212&486_458&569-458&569_224&555_212&486_446&494-0_0_3_24_24_25_24_24-88-24.jpg"
    ],
    "val_samples": 1001,
    "val_sample_paths": [
      "check_dataset\/demo_img\/3056141493055555554-88_93-205&455_603&597-603&575_207&597_205&468_595&455-0_0_3_24_32_27_31_33-90-213.jpg",
      "check_dataset\/demo_img\/0680295138889-88_94-120&474_581&623-577&605_126&623_120&483_581&474-0_0_5_24_31_24_24_24-116-518.jpg",
      "check_dataset\/demo_img\/0482421875-87_265-154&388_496&530-490&495_154&530_156&411_496&388-0_0_5_25_33_33_33_33-84-104.jpg",
      "check_dataset\/demo_img\/0347504340278-105_106-235&443_474&589-474&589_240&518_235&443_473&503-0_0_3_25_30_33_27_30-162-4.jpg",
      "check_dataset\/demo_img\/0205338541667-93_262-182&428_410&519-410&519_187&499_182&428_402&442-0_0_3_24_26_29_32_24-83-63.jpg",
      "check_dataset\/demo_img\/0380913628472-97_250-234&403_529&534-529&534_250&480_234&403_528&446-0_0_3_25_25_24_25_25-185-85.jpg",
      "check_dataset\/demo_img\/020598958333333334-93_267-256&471_482&563-478&563_256&546_262&471_482&484-0_0_3_26_24_25_32_24-102-115.jpg",
      "check_dataset\/demo_img\/3030323350694444445-86_131-170&495_484&593-434&569_170&593_226&511_484&495-11_0_5_30_30_31_33_24-118-59.jpg",
      "check_dataset\/demo_img\/3016158854166666667-86_97-243&471_462&546-462&527_245&546_243&479_453&471-0_0_3_24_30_27_24_29-98-40.jpg",
      "check_dataset\/demo_img\/0340831163194-89_264-177&412_488&523-477&506_177&523_185&420_488&412-0_0_3_24_30_29_31_31-109-46.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "ccpd_text_det",
  "show_type": "image",
  "dataset_type": "TextDetDataset"
}
```

In the above verification results, `check_pass` being `True` indicates that the dataset format meets the requirements. Explanations for other indicators are as follows:

- `attributes.train_samples`: The number of samples in the training set of this dataset is 5769;
- `attributes.val_samples`: The number of samples in the validation set of this dataset is 1001;
- `attributes.train_sample_paths`: A list of relative paths to the visualization images of samples in the training set of this dataset;
- `attributes.val_sample_paths`: A list of relative paths to the visualization images of samples in the validation set of this dataset;

Additionally, the dataset verification also analyzes the distribution of sample numbers across all categories in the dataset and plots a histogram (`histogram.png`):
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/ocr/02.png" width=600>

</center>

<b>Note</b>: Only data that passes the verification can be used for training and evaluation.


### 4.3 Dataset Splitting (Optional)

If you need to convert the dataset format or re-split the dataset, you can set it by modifying the configuration file or appending hyperparameters.

Parameters related to dataset verification can be set by modifying the fields under `CheckDataset` in the configuration file. Examples of some parameters in the configuration file are as follows:

* `CheckDataset`:
    * `split`:
        * `enable`: Whether to re-split the dataset. Set to `True` to perform dataset format conversion, default is `False`;
        * `train_percent`: If re-splitting the dataset, you need to set the percentage of the training set, which is an integer between 0-100, ensuring the sum with `val_percent` is 100;
        * `val_percent`: If re-splitting the dataset, you need to set the percentage of the validation set, which is an integer between 0-100, ensuring the sum with `train_percent` is 100;

During data splitting, the original annotation files will be renamed to `xxx.bak` in their original paths. The above parameters also support being set by appending command-line arguments, for example, to re-split the dataset and set the training and validation set ratios: `-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`.

## 5. Model Training and Evaluation
### 5.1 Model Training

Before training, ensure you have verified the dataset. To complete PaddleX model training, simply use the following command:

```bash
python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_server_det.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ccpd_text_det
```

PaddleX supports modifying training hyperparameters, single-machine single/multi-GPU training, etc., by modifying the configuration file or appending command line parameters.

Each model in PaddleX provides a configuration file for model development to set relevant parameters. Parameters related to model training can be set by modifying the fields under `Train` in the configuration file. Some example explanations of the parameters in the configuration file are as follows:

* `Global`:
    * `mode`: Mode, supporting dataset verification (`check_dataset`), model training (`train`), and model evaluation (`evaluate`);
    * `device`: Training device, options include `cpu`, `gpu`, `xpu`, `npu`, `mlu`. For multi-GPU training, specify card numbers, e.g., `gpu:0,1,2,3`;
* `Train`: Training hyperparameter settings;
    * `epochs_iters`: Number of training epochs;
    * `learning_rate`: Training learning rate;

For more hyperparameter introductions, please refer to [PaddleX General Model Configuration File Parameter Explanation](../module_usage/instructions/config_parameters_common.en.md).

<b>Note</b>:
- The above parameters can be set by appending command line arguments, e.g., specifying the mode as model training: `-o Global.mode=train`; specifying the first 2 GPUs for training: `-o Global.device=gpu:0,1`; setting the number of training epochs to 10: `-o Train.epochs_iters=10`.
- During model training, PaddleX automatically saves model weight files, defaulting to `output`. To specify a save path, use the `-o Global.output` field in the configuration file.
- PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced. During model inference, static graph weights are selected by default.

<b>Training Output Explanation</b>:

After completing model training, all outputs are saved in the specified output directory (default is `./output/`), typically including:

* train_result.json: Training result record file, recording whether the training task completed normally, as well as the output weight metrics, related file paths, etc.;
* train.log: Training log file, recording model metric changes, loss changes, etc., during training;
* config.yaml: Training configuration file, recording the hyperparameter configuration for this training session;
* .pdparams, .pdopt, .pdstates, .pdiparams, .pdmodel: Model weight-related files, including network parameters, optimizer, static graph network parameters, static graph network structure, etc.;

### 5.2 Model Evaluation

After completing model training, you can evaluate the specified model weights file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation requires only one command:

```bash
python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_server_det.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ccpd_text_det
```

Similar to model training, model evaluation supports setting through modifying the configuration file or appending command-line parameters.

<b>Note</b>: When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command-line parameter, such as `-o Evaluate.weight_path=./output/best_accuracy/best_accuracy.pdparams`.

### 5.3 Model Tuning

After learning about model training and evaluation, we can improve the model's accuracy by adjusting hyperparameters. By reasonably adjusting the number of training epochs, you can control the depth of model training to avoid overfitting or underfitting. The setting of the learning rate is related to the speed and stability of model convergence. Therefore, when optimizing model performance, it is essential to carefully consider the values of these two parameters and adjust them flexibly according to the actual situation to achieve the best training effect.

It is recommended to follow the controlled variable method when debugging parameters:

1. First, fix the number of training epochs to 10, the batch size to 8, the number of GPUs to 4, and the total batch size to 32.
2. Start four experiments based on the PP-OCRv4_server_det model with learning rates of: 0.00005, 0.0001, 0.0005, 0.001.
3. You can find that Experiment 4 with a learning rate of 0.001 has the highest accuracy, and by observing the validation set score, the accuracy continues to increase in the last few epochs. Therefore, increasing the number of training epochs to 20 will further improve the model accuracy.

Learning Rate Exploration Results:
<center>

<table>
<thead>
<tr>
<th>Experiment ID</th>
<th>Learning Rate</th>
<th>Detection Hmean (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>0.00005</td>
<td>99.06</td>
</tr>
<tr>
<td>2</td>
<td>0.0001</td>
<td>99.55</td>
</tr>
<tr>
<td>3</td>
<td>0.0005</td>
<td>99.60</td>
</tr>
<tr>
<td>4</td>
<td>0.001</td>
<td>99.70</td>
</tr>
</tbody>
</table>
</center>

Next, based on a learning rate of 0.001, we can increase the number of training epochs. Comparing Experiments [4, 5] below, it can be seen that increasing the number of training epochs further improves the model accuracy.
<center>

<table>
<thead>
<tr>
<th>Experiment ID</th>
<th>Number of Training Epochs</th>
<th>Detection Hmean (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>4</td>
<td>10</td>
<td>99.70</td>
</tr>
<tr>
<td>5</td>
<td>20</td>
<td>99.80</td>
</tr>
</tbody>
</table>
</center>

<b>Note: This tutorial is designed for 4 GPUs. If you only have 1 GPU, you can complete the experiment by adjusting the number of training GPUs, but the final metrics may not align with the above indicators, which is normal.</b>

## 6. Production Line Testing

Replace the model in the production line with the fine-tuned model for testing. You can obtain the OCR production configuration file and load the configuration file for prediction. You can execute the following command to save the results in `my_path`:

```bash
paddlex --get_pipeline_config OCR --save_path ./my_path
```

Modify the `SubModules.TextDetection.model_dir` in the configuration file to your model path.

```yaml
SubModules:
  TextDetection:
    module_name: text_detection
    model_name: PP-OCRv4_mobile_det
    model_dir: output/best_accuracy/inference # 替换为微调后的文本检测模型权重路径
    ...
```

Subsequently, based on the Python script method, you can load the modified production configuration file:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="my_path/OCR.yaml")

output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/OCR/case1.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")

```

This will generate prediction results under `./output`, where the prediction result for `case1.jpg` is shown below:
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/practical_tutorials/ocr/03.png" width="600"/>

</center>

## 7. Development Integration/Deployment
If the general OCR pipeline meets your requirements for inference speed and accuracy in the production line, you can proceed directly with development integration/deployment.
1. Directly apply the trained model in your Python project. You can refer to the following example:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="paddlex/configs/pipelines/OCR.yaml")

output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/OCR/case1.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")

```
For more parameters, please refer to the [General OCR Pipeline Usage Tutorial](../pipeline_usage/tutorials/ocr_pipelines/OCR.en.md).

1. Additionally, PaddleX offers three other deployment methods, detailed as follows:

* high-performance inference: In actual production environments, many applications have stringent standards for deployment strategy performance metrics (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end process acceleration. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../pipeline_deploy/high_performance_inference.en.md).
* Service-Oriented Deployment: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving cost-effective service-oriented deployment of production lines. For detailed service-oriented deployment procedures, please refer to the [PaddleX Service-Oriented Deployment Guide](../pipeline_deploy/serving.en.md).
* Edge Deployment: Edge deployment is a method that places computing and data processing capabilities directly on user devices, allowing devices to process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../pipeline_deploy/edge_deploy.en.md).

You can select the appropriate deployment method for your model pipeline according to your needs, and proceed with subsequent AI application integration.
