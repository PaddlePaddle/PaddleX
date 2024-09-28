# Document Image Orientation Classification Module Development Tutorial

## I. Overview
The document image orientation classification module is aim to distinguish the orientation of document images and correct them through post-processing. In processes such as document scanning and ID card photography, capturing devices are sometimes rotated to obtain clearer images, resulting in images with varying orientations. Standard OCR pipelines cannot effectively handle such data. By utilizing image classification technology, we can pre-judge the orientation of document or ID card images containing text regions and adjust their orientations, thereby enhancing the accuracy of OCR processing.

## II. Supported Model List

<details>
   <summary> 👉 Model List Details</summary>

| Model | Top-1 Accuracy (%) | GPU Inference Time (ms) | CPU Inference Time | Model Size (M) | Description |
|-|-|-|-|-|-|
| PP-LCNet_x1_0_doc_ori | 99.06 | | | 7 | A document image classification model based on PP-LCNet_x1_0, with four categories: 0°, 90°, 180°, 270° |

**Note: The above accuracy metrics are evaluated on a self-built dataset covering various scenarios such as IDs and documents, containing 1000 images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**
</details>

## III. Quick Integration

> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to [PaddleX Local Installation Tutorial](../../../installation/installation.md)

Just a few lines of code can complete the inference of the document image orientation classification module, allowing you to easily switch between models under this module. You can also integrate the model inference of the the document image orientation classification module into your project.

```bash
from paddlex import create_model
model = create_model("PP-LCNet_x1_0_doc_ori")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/demo.png")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single model inference API, refer to [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.md).

## IV. Custom Development
If you seek higher accuracy, you can leverage PaddleX's custom development capabilities to develop better document image orientation classification models. Before developing a document image orientation classification model with PaddleX, ensure you have installed PaddleClas plugin for PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Tutorial](https://github.com/AmberC0209/PaddleX/blob/docs_change/docs_new/installation/installation.md).

### 4.1 Data Preparation
Before model training, you need to prepare a dataset for the task. PaddleX provides data validation functionality for each module. **Only data that passes validation can be used for model training.** Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to [PaddleX Image Classification Task Module Data Preparation Tutorial](/docs_new/data_annotations/cv_modules/image_classification.md).

#### 4.1.1 Demo Data Download
You can download the demo dataset to a specified folder using the following commands:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/text_image_orientation.tar -P ./dataset
tar -xf ./dataset/text_image_orientation.tar  -C ./dataset/
```

#### 4.1.2 Data Validation
Data validation can be completed with a single command:

```bash
python main.py -c paddlex/configs/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/text_image_orientation
```

After executing the above command, PaddleX will verify the dataset and collect basic information about it. Once the command runs successfully, a message saying `Check dataset passed !` will be printed in the log. The verification results will be saved in `./output/check_dataset_result.json`, and related outputs will be stored in the `./output/check_dataset` directory, including visual examples of sample images and a histogram of sample distribution.

<details>
  <summary>👉 <b>Verification Result Details (click to expand)</b></summary>

The specific content of the verification result file is:

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "..\/..\/text_image_orientation\/label.txt",
    "num_classes": 4,
    "train_samples": 1553,
    "train_sample_paths": [
      "check_dataset\/demo_img\/img_rot270_10351.jpg",
      "check_dataset\/demo_img\/img_rot0_3908.jpg",
      "check_dataset\/demo_img\/img_rot180_7712.jpg",
      "check_dataset\/demo_img\/img_rot0_7480.jpg",
      "check_dataset\/demo_img\/img_rot270_9599.jpg",
      "check_dataset\/demo_img\/img_rot90_10323.jpg",
      "check_dataset\/demo_img\/img_rot90_4885.jpg",
      "check_dataset\/demo_img\/img_rot180_3939.jpg",
      "check_dataset\/demo_img\/img_rot90_7153.jpg",
      "check_dataset\/demo_img\/img_rot180_1747.jpg"
    ],
    "val_samples": 2593,
    "val_sample_paths": [
      "check_dataset\/demo_img\/img_rot270_3190.jpg",
      "check_dataset\/demo_img\/img_rot0_10272.jpg",
      "check_dataset\/demo_img\/img_rot0_9930.jpg",
      "check_dataset\/demo_img\/img_rot90_918.jpg",
      "check_dataset\/demo_img\/img_rot180_2079.jpg",
      "check_dataset\/demo_img\/img_rot90_8574.jpg",
      "check_dataset\/demo_img\/img_rot90_7595.jpg",
      "check_dataset\/demo_img\/img_rot90_1751.jpg",
      "check_dataset\/demo_img\/img_rot180_1573.jpg",
      "check_dataset\/demo_img\/img_rot90_4401.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": ".\/text_image_orientation",
  "show_type": "image",
  "dataset_type": "ClsDataset"
}
```

In the verification results above, `check_pass` being True indicates that the dataset format meets the requirements. Explanations of other indicators are as follows:

* `attributes.num_classes`: The number of classes in this dataset is 4;
* `attributes.train_samples`: The number of training samples in this dataset is 1552;
* `attributes.val_samples`: The number of validation samples in this dataset is 2593;
* `attributes.train_sample_paths`: A list of relative paths to visual sample images for the training set of this dataset;
* ````
Then, execute the following command:

```bash
python main.py -c paddlex/configs/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/text_image_orientation
```

After data splitting, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters can also be set by appending command line arguments:

```bash
python main.py -c paddlex/configs/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/text_image_orientation \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```

</details>


#### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format and re-split the training/validation ratio by **modifying the configuration file** or **appending hyperparameters**.

<details>
  <summary>👉 <b>Details of Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

Document image orientation classification does not currently support dataset format conversion.

**(2) Dataset Splitting**

Parameters for dataset splitting can be set by modifying the fields under `CheckDataset` in the configuration file. Examples of some parameters in the configuration file are as follows:

* `CheckDataset`:
  * `split`:
    * `enable`: Whether to re-split the dataset. Set to `True` to enable dataset splitting, default is `False`;
    * `train_percent`: If re-splitting the dataset, set the percentage of the training set. The type is any integer between 0-100, ensuring the sum with `val_percent` is 100;

For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, modify the configuration file as follows:

```bash
......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
```
Then execute the command:

```bash
python main.py -c paddlex/configs/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/text_image_orientation
```
After dataset splitting, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters also support setting through appending command line arguments:

```bash
python main.py -c paddlex/configs/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/text_image_orientation \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>


### 4.2 Model Training

Model training can be completed with just one command. Here, we use the document image orientation classification model (PP-LCNet_x1_0_doc_ori) as an example:

```bash
python main.py -c paddlex/configs/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/text_image_orientation
```

You need to follow these steps:

* Specify the path to the model's `.yaml` configuration file (here, `PP-LCNet_x1_0_doc_ori.yaml`).
* Set the mode to model training: `-o Global.mode=train`.
* Specify the training dataset path: `-o Global.dataset_dir`.

Other relevant parameters can be set by modifying fields under `Global` and `Train` in the `.yaml` configuration file, or by appending arguments to the command line. For example, to specify the first two GPUs for training: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and detailed explanations, refer to the [PaddleX General Model Configuration File Parameters](../../instructions/config_parameters_common.md).

<details>
  <summary>👉 <b>More Information (click to expand)</b></summary>

* During model training, PaddleX automatically saves the model weight files, defaulting to `output`. If you want to specify a different save path, you can set it using the `-o Global.output` field in the configuration file.
* PaddleX abstracts away the concept of dynamic graph weights and static graph weights. During model training, it produces both dynamic and static graph weights. For model inference, it defaults to using static graph weights.
* To train other models, specify the corresponding configuration file. The relationship between models and configuration files can be found in the [PaddleX Model List (CPU/GPU)](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/GvMbk70MZz/0PKFjfhs0UN4Qs?t=mention&mt=doc&dt=doc).

After completing model training, all outputs are saved in the specified output directory (default is `./output/`), typically including the following:

* `train_result.json`: Training result record file, which records whether the training task was completed normally, as well as the output weight metrics and related file paths.
* `train.log`: Training log file, which records changes in model metrics and loss during training.
* `config.yaml`: Training configuration file, which records the hyperparameter configuration for this training.
* `.pdparams`, `.pdema`, `.pdopt.pdstate`, `.pdiparams`, `.pdmodel`: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.

</details>

### **4.3 Model Evaluation**

After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. With PaddleX, model evaluation can be done with just one command:

```bash
python main.py -c paddlex/configs/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/text_image_orientation
```
Similar to model training and evaluation, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (here it is `PP-LCNet_x1_0_doc_ori.yaml`).
* Set the mode to model inference prediction: `-o Global.mode=predict`.
* Specify the path to the model weights: `-o Predict.model_dir="./output/best_model/inference"`.
* Specify the input data path: `-o Predict.input="..."`.
Other relevant parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX General Model Configuration File Parameter Description](../../instructions/config_parameters_common.md).

<details>
  <summary>👉 <b>More Information (click to expand)</b></summary>

* When conducting model evaluation, it is necessary to specify the model weight file path. Each configuration file has a built-in default path for saving weights. If you need to change this path, you can simply append a command line argument to set it, for example: `-o Evaluate.weight_path=./output/best_model/best_model.pdparams`.

* After the model evaluation is completed, typically, the following outputs are generated:

* Upon finishing the model evaluation, an evaluate_result.json file is produced, which records the results of the evaluation. Specifically, it logs whether the evaluation task was successfully completed and the evaluation metrics of the model, including `Top1 Accuracy (Top1 Acc)`.

</details>

### **4.4 Model Inference and Model Integration**

After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference

To perform inference predictions via the command line, simply use the following command:

```bash
python main.py -c paddlex/configs/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg"
```

Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it's `PP-LCNet_x1_0_doc_ori.yaml`)

* Set the mode to model inference prediction: `-o Global.mode=predict`

* Specify the model weights path: -o Predict.model_dir="./output/best_accuracy/inference"

Specify the input data path: `-o Predict.inputh="..."` Other related parameters can be set by modifying the fields under Global and Predict in the `.yaml` configuration file. For details, refer to PaddleX Common Model Configuration File Parameter Description.

Alternatively, you can use the PaddleX wheel package for inference, easily integrating the model into your own projects.

#### 4.4.2 Model Integration

The model can be directly integrated into the PaddleX pipeline or into your own projects.

1. **Pipeline Integration**

The document image classification module can be integrated into PaddleX pipelines such as the [Document Scene Information Extraction Pipeline (PP-ChatOCRv3)](/docs_new/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction.md). Simply replace the model path to update the The document image classification module's model.

2. **Module Integration**

The weights you produce can be directly integrated into the document image orientation classification module. You can refer to the Python sample code in [Quick Integration](#quick-integration) and just replace the model with the path to the model you trained.
