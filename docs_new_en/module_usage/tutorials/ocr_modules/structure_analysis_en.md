# Structure Analysis Module Development Tutorial

## I. Overview
The core task of structure analysis is to parse and segment the content of input document images. By identifying different elements in the image (such as text, charts, images, etc.), they are classified into predefined categories (e.g., pure text area, title area, table area, image area, list area, etc.), and the position and size of these regions in the document are determined.

## II. Supported Model List

<details>
   <summary> 👉Model List Details</summary>

| Model | mAP(0.5) (%) | GPU Inference Time (ms) | CPU Inference Time | Model Size (M) | Description |
|-|-|-|-|-|-|
| PicoDet-L_layout_3cls | 89.3 | 15.7 | 159.8 | 22.6 | High-efficiency structure analysis model based on PicoDet-L, including 3 classes: table, image, and seal |
| PicoDet_layout_1x | 86.8 | 13.0 | 91.3 | 7.4 | High-efficiency structure analysis model based on PicoDet-1x, including text, title, table, image, and list |
| RT-DETR-H_layout_17cls | 92.6 | 115.1 | 3827.2 | 470.2 | High-precision structure analysis model based on RT-DETR-H, including 17 common layout categories. |
| RT-DETR-H_layout_3cls | 95.9 | 114.6 | 3832.6 | 470.1 | High-precision structure analysis model based on RT-DETR-H, including 3 classes: table, image, and seal |

**Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built layout region analysis dataset, containing 10,000 images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**
</details>

## III. Quick Integration  <a id="quick"> </a> 
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to [PaddleX Local Installation Tutorial](../../../installation/installation_en.md)

After installing the wheel package, a few lines of code can complete the inference of the structure analysis module. You can switch models under this module freely, and you can also integrate the model inference of the structure analysis module into your project.

```python
from paddlex.inference import create_model 

model_name = "PicoDet-L_layout_3cls"

model = create_model(model_name)
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg", batch_size=1)

for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")

```

For more information on using PaddleX's single-model inference API, refer to [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API_en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better structure analysis models. Before developing a structure analysis model with PaddleX, ensure you have installed PaddleX's Detection-related model training capabilities. The installation process can be found in [PaddleX Local Installation Tutorial](../../../installation/installation_en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides a data validation function for each module, and **only data that passes the validation can be used for model training**. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development based on the official demos. If you wish to use private datasets for subsequent model training, refer to the [PaddleX Object Detection Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/object_detection_en.md).

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_layout_examples.tar -P ./dataset
tar -xf ./dataset/det_layout_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/structure_analysis/PicoDet-L_layout_3cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_layout_examples
```

After executing the above command, PaddleX will validate the dataset and collect its basic information. Upon successful execution, the log will print the message `Check dataset passed !`. The validation result file will be saved in `./output/check_dataset_result.json`, and related outputs will be saved in the `./output/check_dataset` directory of the current directory. The output directory includes visualized example images and histograms of sample distributions.

<details>
  <summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>

The specific content of the validation result file is:

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 11,
    "train_samples": 90,
    "train_sample_paths": [
      "check_dataset/demo_img/JPEGImages/train_0077.jpg",
      "check_dataset/demo_img/JPEGImages/train_0028.jpg",
      "check_dataset/demo_img/JPEGImages/train_0012.jpg"
    ],
    "val_samples": 20,
    "val_sample_paths": [
      "check_dataset/demo_img/JPEGImages/val_0007.jpg",
      "check_dataset/demo_img/JPEGImages/val_0019.jpg",
      "check_dataset/demo_img/JPEGImages/val_0010.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/example_data/det_layout_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}
```

The verification results mentioned above indicate that `check_pass` being `True` means the dataset format meets the requirements. Details of other indicators are as follows:

* `attributes.num_classes`: The number of classes in this dataset is 11;
* `attributes.train_samples`: The number of training samples in this dataset is 90;
* `attributes.val_samples`: The number of validation samples in this dataset is 20;
* `attributes.train_sample_paths`: The list of relative paths to the visualization images of training samples in this dataset;
* `attributes.val_sample_paths`: The list of relative paths to the visualization images of validation samples in this dataset;

The dataset verification also analyzes the distribution of sample numbers across all classes and generates a histogram (histogram.png):

![](/tmp/images/modules/layout_dec/01.png)

</details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)

After completing dataset verification, you can convert the dataset format or re-split the training/validation ratio by **modifying the configuration file** or **appending hyperparameters**.

<details>
  <summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

Structure analysis does not support data format conversion.

**(2) Dataset Splitting**

Parameters for dataset splitting can be set by modifying the `CheckDataset` section in the configuration file. Examples of some parameters in the configuration file are as follows:

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
python main.py -c paddlex/configs/structure_analysis/PicoDet-L_layout_3cls.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_layout_examples
```
After dataset splitting, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters can also be set by appending command-line arguments:

```bash
python main.py -c paddlex/configs/structure_analysis/PicoDet-L_layout_3cls.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_layout_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### 4.2 Model Training

A single command is sufficient to complete model training, taking the training of PicoDet-L_layout_3cls as an example:

```bash
python main.py -c paddlex/configs/structure_analysis/PicoDet-L_layout_3cls.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/det_layout_examples
```
The steps required are:

* Specify the path to the `.yaml` configuration file of the model (here it is `PicoDet-L_layout_3cls.yaml`)
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters for Model Tasks](../../instructions/config_parameters_common_en.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

* During model training, PaddleX automatically saves model weight files, defaulting to `output`. To specify a save path, use the `-o Global.output` field in the configuration file.
* PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.
* When training other models, specify the corresponding configuration file. The correspondence between models and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list_en.md).
After completing model training, all outputs are saved in the specified output directory (default is `./output/`), typically```markdown
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it is `PicoDet-L_layout_3cls.yaml`)
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path of the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common_en.md).
</details>

### **4.3 Model Evaluation**
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation, you can complete the evaluation with a single command:

```bash
python main.py -c paddlex/configs/structure_analysis/PicoDet-L_layout_3cls.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/det_layout_examples
```
Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model（here it's `PicoDet-L_layout_3cls.yaml`）
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to [PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common_en.md)。

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>


When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as `-o Evaluate.weight_path=./output/best_model/best_model/model.pdparams`.

After completing the model evaluation, an `evaluate_result.json` file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully, and the model's evaluation metrics, including AP.

</details>

### **4.4 Model Inference**
After completing model training and evaluation, you can use the trained model weights for inference predictions. In PaddleX, model inference predictions can be achieved through two methods: command line and wheel package.

#### 4.4.1 Model Inference
* To perform inference predictions through the command line, simply use the following command:
```bash
python main.py -c paddlex/configs/structure_analysis/PicoDet-L_layout_3cls.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it is `PicoDet-L_layout_3cls.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common_en.md).

* Alternatively, you can use the PaddleX wheel package for inference, easily integrating the model into your own project. To integrate, simply add the `model_dir="/output/best_model/inference"` parameter to the `create_model(model_name=model_name, kernel_option=kernel_option)` function in the quick integration method from Step 3.

#### 4.4.2 Model Integration
The model can be directly integrated into PaddleX pipelines or into your own projects.

1. **Pipeline Integration**
The structure analysis module can be integrated into PaddleX pipelines such as the [General Table Recognition Pipeline](../../../pipeline_usage/tutorials/ocr_pipelies/table_recognition_en.md) and the [Document Scene Information Extraction Pipeline v3 (PP-ChatOCRv3)](../../..//pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction_en.md). Simply replace the model path to update the layout area localization module. In pipeline integration, you can use high-performance deployment and service-oriented deployment to deploy your model.

1. **Module Integration**
The weights you produce can be directly integrated into the layout area localization module. You can refer to the Python example code in the [Quick Integration](#quick) section, simply replacing the model with the path to your trained model.
