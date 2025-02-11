
# PaddleX 3.0 Small Object Detection Pipeline——Remote Sensing Image Small Object Detection

PaddleX provides a rich set of model pipelines, which are implemented by combining one or more models. Each model pipeline can solve specific task problems in different scenarios. The model pipelines provided by PaddleX all support quick experience. If the effect does not meet expectations, it also supports fine-tuning the model with private data. Moreover, PaddleX provides a Python API, making it convenient to integrate the pipeline into individual projects. Before using it, you first need to install PaddleX. Please refer to [PaddleX Installation](../installation/installation.md) for installation methods. Here, we take a remote sensing small object detection task as an example to introduce the usage process of the model pipeline tool.

## 1. Select Pipeline

First, you need to choose the corresponding PaddleX pipeline based on your task scenario. Here, for remote sensing small object detection, you need to understand that this task belongs to small object detection, corresponding to the small object detection pipeline of PaddleX. If you cannot determine the correspondence between the task and the pipeline, you can learn about the capabilities of related pipelines in the [Model Pipeline List](../support_list/pipelines_list.md) supported by PaddleX.

## 2. Quick Experience

PaddleX provides the following quick experience methods, which can be directly experienced locally through the PaddleX wheel package.

  - Local experience method:
    ```python
    import paddlex
    from paddlex import create_model

    model = create_model("PP-YOLOE_plus_SOD-largesize-L")
    image_url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/remote-scene-det_example.png"
    rst_save_dir = "./output"
    output = model.predict([image_url], batch_size=1)

    for res in output:
        res.print(json_format=False)
        res.save_to_img(rst_save_dir)
        res.save_to_json(rst_save_dir)
    ```

  The quick experience produces the following inference result example:
  <center>

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/smallobj_det/02.png" width=600>

  </center>

After experiencing the pipeline, you need to determine whether the pipeline meets expectations (including accuracy, speed, etc.), and whether the models included in the pipeline need to be further fine-tuned. If the speed or accuracy of the model does not meet expectations, you need to choose a replaceable model based on the model selection to continue testing and determine whether the effect is satisfactory. If the final effect is not satisfactory, you need to fine-tune the model. This tutorial aims to produce a small object detection model in the field of remote sensing. Obviously, the default weights (weights produced by training on the UAV aerial photography dataset) cannot meet the requirements. You need to collect and annotate data, and then train and fine-tune.

## 3. Select Model

PaddleX provides 3 high-precision and high-efficiency end-to-end small object detection models. For details, please refer to the [Model List](../support_list/models_list.md). The benchmarks of some models are as follows:

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-YOLOE_plus_SOD-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-L_pretrained.pdparams">Training Model</a></td>
<td>31.9</td>
<td>52.1</td>
<td>57.1</td>
<td>1007.0</td>
<td>324.93</td>
<td rowspan="3">PP-YOLOE_plus small object detection model trained based on VisDrone. VisDrone is a benchmark dataset for UAV vision data, which is used for training and evaluation of small object detection tasks due to its small targets and certain challenges.</td>
</tr>
<tr>
<td>PP-YOLOE_plus_SOD-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-S_pretrained.pdparams">Training Model</a></td>
<td>25.1</td>
<td>42.8</td>
<td>65.5</td>
<td>324.4</td>
<td>77.29</td>
</tr>
<tr>
<td>PP-YOLOE_plus_SOD-largesize-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE_plus_SOD-largesize-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-largesize-L_pretrained.pdparams">Training Model</a></td>
<td>42.7</td>
<td>65.9</td>
<td>458.5</td>
<td>11172.7</td>
<td>340.42</td>
</tr>
</table>

**Note: The above accuracy metrics are mAP(0.5:0.95) on VisDrone-DET validation set. All models' GPU inference time is based on NVIDIA Tesla T4 machine with FP32 precision, CPU inference speed is based on Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads, and FP32 precision.**

## 4. Data Preparation and Validation
### 4.1 Data Preparation

In this tutorial, the `Remote Sensing Small Object Detection Dataset` is used as an example dataset, which can be obtained through the following commands. If you are using your own annotated dataset, you need to adjust your dataset according to PaddleX's format requirements to meet PaddleX's data format requirements. For data format introduction, you can refer to [PaddleX Detection Task Module Data Annotation Tutorial](../data_annotations/cv_modules/object_detection.md).

Dataset acquisition command:
```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/remote_scene_object_detection.tar -P ./dataset
tar -xf ./dataset/remote_scene_object_detection.tar -C ./dataset/
```

### 4.2 Dataset Validation

When validating the dataset, you only need one line of command:

```bash
python main.py -c paddlex/configs/modules/small_object_detection/PP-YOLOE_plus_SOD-largesize-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/remote_scene_object_detection
```

After executing the above command, PaddleX will validate the dataset and count the basic information of the dataset. After the command runs successfully, `Check dataset passed!` message will be printed in the log, and the related outputs will be saved in the `./output/check_dataset` directory in the current directory. The output directory includes visualized sample images and sample distribution histograms. The validation result file is saved in `./output/check_dataset_result.json`, the specific content of the validation result file is
```
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 15,
    "train_samples": 172,
    "train_sample_paths": [
      "check_dataset\/demo_img\/P1751__1.0__824___824.png",
      "check_dataset\/demo_img\/P1560__1.0__0___0.png",
      "check_dataset\/demo_img\/P1051__1.0__0___0.png",
      "check_dataset\/demo_img\/P1751__1.0__1648___0.png",
      "check_dataset\/demo_img\/P0117__1.0__0___0.png",
      "check_dataset\/demo_img\/P2365__1.0__1648___824.png",
      "check_dataset\/demo_img\/P1750__1.0__0___1648.png",
      "check_dataset\/demo_img\/P1560__1.0__824___1648.png",
      "check_dataset\/demo_img\/P2152__1.0__0___0.png",
      "check_dataset\/demo_img\/P0047__1.0__102___1143.png"
    ],
    "val_samples": 80,
    "val_sample_paths": [
      "check_dataset\/demo_img\/P0551__1.0__288___354.png",
      "check_dataset\/demo_img\/P0858__1.0__1360___441.png",
      "check_dataset\/demo_img\/P1560__1.0__1648___2472.png",
      "check_dataset\/demo_img\/P2722__1.0__1648___0.png",
      "check_dataset\/demo_img\/P2462__1.0__824___824.png",
      "check_dataset\/demo_img\/P0262__1.0__824___0.png",
      "check_dataset\/demo_img\/P0047__1.0__0___0.png",
      "check_dataset\/demo_img\/P2722__1.0__1648___1422.png",
      "check_dataset\/demo_img\/P2645__1.0__0___1648.png",
      "check_dataset\/demo_img\/P0858__1.0__0___441.png"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "rdet_dota_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}
```
In the above validation result, `check_pass` being `True` indicates that the dataset format meets the requirements. The explanations of other indicators are as follows:

- `attributes.num_classes`: The number of classes in the dataset is 15, which is the number of categories to be passed in for subsequent training;
- `attributes.train_samples`: The number of training samples in the dataset is 172;
- `attributes.val_samples`: The number of validation samples in the dataset is 80;
- `attributes.train_sample_paths`: The relative path list of visualized images of training samples in the dataset;
- `attributes.val_sample_paths`: The relative path list of visualized images of validation samples in the dataset;

In addition, the dataset validation also analyzed the sample number distribution of all categories in the dataset and plotted the distribution histogram (`histogram.png`):
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/smallobj_det/03.png" width=600>

</center>

**Note**: Only datasets that pass validation can be trained and evaluated.

### 4.3 Dataset Format Conversion/Dataset Splitting (Optional)

If you need to convert the dataset format or re-split the dataset, you can set it by modifying the configuration file or adding hyperparameters.

Dataset validation related parameters can be set by modifying the fields under `CheckDataset` in the configuration file. The example descriptions of some parameters in the configuration file are as follows:

* `CheckDataset`:
    * `convert`:
        * `enable`: Whether to perform dataset format conversion. When set to `True`, perform dataset format conversion. The default is `False`;
        * `src_dataset_type`: If performing dataset format conversion, you need to set the source dataset format. The selectable source formats are `LabelMe` and `VOC`;
    * `split`:
        * `enable`: Whether to re-split the dataset. When set to `True`, perform dataset splitting. The default is `False`;
        * `train_percent`: If re-splitting the dataset, you need to set the percentage of the training set. The type is any integer between 0-100, and needs to ensure that the sum with `val_percent` is 100;
        * `val_percent`: If re-splitting the dataset, you need to set the percentage of the validation set. The type is any integer between 0-100, and needs to ensure that the sum with `train_percent` is 100;

Dataset conversion and dataset splitting support being enabled at the same time. For dataset splitting, the original annotation files will be renamed as `xxx.bak` in the original path. The above parameters also support being set by adding command line parameters, such as re-splitting the dataset and setting the ratio of training set to validation set: `-o CheckDataset.split.enable=True -o CheckDataset.split.train_percent=80 -o CheckDataset.split.val_percent=20`.

## 5. Model Training and Evaluation
### 5.1 Model Training

Before training, please ensure that you have validated the dataset. To complete the training of the PaddleX model, you only need the following command:

```bash
python main.py -c paddlex/configs/modules/small_object_detection/PP-YOLOE_plus_SOD-largesize-L.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/remote_scene_object_detection \
    -o Train.epochs_iters=80
```

In PaddleX, model training supports modifying training hyperparameters, single-machine single-card/multi-card training, etc., by modifying the configuration file or adding command line parameters.

Each model in PaddleX provides a configuration file for model development, which is used to set related parameters. Model training related parameters can be set by modifying the fields under `Train` in the configuration file. The example descriptions of some parameters in the configuration file are as follows:

* `Global`:
    * `mode`: Mode, supports dataset validation (`check_dataset`), model training (`train`), model evaluation (`evaluate`);
    * `device`: Training device, optional `cpu`, `gpu`, `xpu`, `npu`, `mlu`. Except for `cpu`, multi-card training can specify card numbers, such as `gpu:0,1,2,3`;
* `Train`: Training hyperparameter settings;
    * `epochs_iters`: Setting of the number of training iterations;
    * `learning_rate`: Setting of the training learning rate;

For more hyperparameter introductions, please refer to [PaddleX Universal Model Configuration File Parameter Description](../module_usage/instructions/config_parameters_common.md).

**Note:**
- The above parameters can be set by adding command line parameters, such as specifying the mode to model training: `-o Global.mode=train`; specifying the first 2 GPUs for training: `-o Global.device=gpu:0,1`; setting the number of training iterations to 80: `-o Train.epochs_iters=80`.
- During model training, PaddleX will automatically save the model weight files. The default is `output`. If you need to specify the save path, you can do so through the field `-o Global.output` in the configuration file.
- PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During the model training process, both dynamic graph and static graph weights will be produced. When model inference, static graph weights are selected by default.

**Explanation of Training Output:**

After completing the model training, all outputs are saved under the specified output directory (default is `./output/`), usually including the following outputs:

* `train_result.json`: Training result record file, records whether the training task was completed normally, as well as the indicators of the produced weights, related file paths, etc.;
* `train.log`: Training log file, records the changes in model indicators during training, loss changes, etc.;
* `config.yaml`: Training configuration file, records the configuration of hyperparameters in this training;
* `.pdparams`, `.pdema`, `.pdopt`, `.pdstate`, `.pdiparams`, `.pdmodel`: Model weight related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;

### 5.2 Model Evaluation

After completing the model training, you can evaluate the specified model weight file on the validation set to verify the model accuracy. Using PaddleX for model evaluation requires only one command:

```bash
python main.py -c paddlex/configs/modules/small_object_detection/PP-YOLOE_plus_SOD-largesize-L.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/remote_scene_object_detection
```

Similar to model training, model evaluation supports setting by modifying the configuration file or adding command line parameters.

**Note:** When evaluating the model, you need to specify the model weight file path. Each configuration file has a default weight save path built-in. If you need to change it, you only need to set it by adding command line parameters, such as `-o Evaluate.weight_path=./output/best_model/model.pdparams`.

### 5.3 Model Tuning

After learning model training and evaluation, we can improve the model accuracy by adjusting hyperparameters. By properly adjusting the number of training epochs, you can control the training depth of the model and avoid overfitting or underfitting; the setting of the learning rate is related to the speed and stability of model convergence. Therefore, when optimizing model performance, it is necessary to carefully consider the values of these two parameters and adjust flexibly according to the actual situation to obtain the best training effect.

It is recommended to follow the controlled variable method when debugging parameters:

1. First, fix the number of training iterations to 40, and batch size to 1.
2. Start three experiments based on the PP-YOLOE+ SOD model, with learning rates of 0.000625, 0.00125, and 0.0125 respectively.
3. It can be found that the configuration with the highest accuracy in Experiment Three is the learning rate of 0.0125. Based on this training hyperparameter, increasing the number of training epochs to 80 can see better accuracy.

Learning rate exploration experiment results:
<center>

| Experiment  | Number of Epochs | Learning Rate | batch\_size | Training Environment | mAP(0.5:0.95) |
|-------|------|--------|----------|-------|----------|
| Experiment One | 40 | 0.000625 | 1        | 4 cards   | 0.476   |
| Experiment Two | 40 | 0.00125 | 1        | 4 cards   |0.412|
| Experiment Three | 40 | 0.0125  | 1        | 4 cards   | **0.501**   |

</center>

Experiment results after changing epoch:
<center>

| Experiment                | Number of Epochs | Learning Rate | batch\_size| Training Environment | mAP(0.5:0.95) |
|--------------------|---------|-------|------------|------|----------|
| Experiment Three              | 40    | 0.0125 | 1         | 4 cards   | 0.501   |
| Experiment Three with increased training iterations | 80   | 0.0125 | 1         | 4 cards   | **0.512**|
</center>

**Note: This tutorial is an 8-card tutorial. If you have only one GPU, you can complete this experiment by adjusting the number of training cards, but the final indicator may not align with the above indicators, which is normal.**

## 6. Pipeline Testing

Replace the model in the pipeline with the fine-tuned model for testing, such as:

```bash
python main.py -c paddlex/configs/modules/small_object_detection/PP-YOLOE_plus_SOD-largesize-L.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/remote-scene-det_example.png"
```

Through the above, the prediction results can be generated under `./output`, as follows:
<center>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/smallobj_det/04.png" width="600"/>

</center>

## 7. Development Integration/Deployment
If the small object detection pipeline can meet your requirements for pipeline inference speed and accuracy, you can directly proceed with development integration/deployment.

1. If you need to use the fine-tuned model weights, you can obtain the configuration file for the small_object_detection pipeline and load it for prediction. You can execute the following command to save the results in my_path:

```
paddlex --get_pipeline_config small_object_detection --save_path ./my_path
```

Fill in the local path of the fine-tuned model weights into the `model_dir` field in the configuration file. If you need to directly apply the general small_object_detection pipeline to your Python project, you can refer to the following example:

```yaml
pipeline_name: small_object_detection

SubModules:
  SmallObjectDetection:
    module_name: small_object_detection
    model_name: PP-YOLOE_plus_SOD-L
    model_dir: null # Replace this with the local path to your trained model weights.
    batch_size: 1
    threshold: 0.5
```

Subsequently, in your Python code, you can utilize the pipeline as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="my_path/small_object_detection.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/remote-scene-det_example.png")
for res in output:
    res.print() # Print the structured output of the prediction
    res.save_to_img("./output/") # Save the result visualization image
    res.save_to_json("./output/") # Save the structured output of the prediction
```
For more parameters, please refer to [Small Object Detection Pipeline Usage Tutorial](../pipeline_usage/tutorials/cv_pipelines/small_object_detection.md).

2. In addition, PaddleX also provides three other deployment methods, detailed explanations are as follows:

* High-performance deployment: In actual production environments, many applications have strict standards for the performance indicators of deployment strategies (especially response speed) to ensure the efficient operation of the system and the smooth user experience. For this reason, PaddleX provides a high-performance inference plugin, aiming to deeply optimize the performance of model inference and pre/post-processing, achieving significant acceleration of the end-to-end process. For detailed high-performance deployment processes, please refer to [PaddleX High-Performance Inference Guide](../pipeline_deploy/high_performance_inference.en.md).
* Service deployment: Service deployment is a common deployment form in actual production environments. By encapsulating inference functions into services, clients can access these services through network requests to obtain inference results. PaddleX supports users to achieve service deployment of the pipeline at a low cost. For detailed service deployment processes, please refer to [PaddleX Service Deployment Guide](../pipeline_deploy/serving.en.md).
* Edge deployment: Edge deployment is a way of placing computing and data processing functions on user devices themselves, where devices can directly process data without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment processes, please refer to [PaddleX Edge Deployment Guide](../pipeline_deploy/edge_deploy.en.md).

You can choose an appropriate method to deploy the model pipeline according to your needs and proceed with subsequent AI application integration.
