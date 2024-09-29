# Face Detection Module Development Tutorial

## I. Overview
Face detection is a fundamental task in object detection, aiming to automatically identify and locate the position and size of faces in input images. It serves as the prerequisite and foundation for subsequent tasks such as face recognition and face analysis. Face detection accomplishes this by constructing deep neural network models that learn the feature representations of faces, enabling efficient and accurate face detection.

## II. Supported Model List

<details>
   <summary> 👉Model List Details</summary>

| Model | mAP（%）| GPU Inference Time (ms) | CPU Inference Time | Model Size (M) | Description |
|-|-|-|-|-|-|
| PicoDet_LCNet_x2_5_face | 35.8 | 33.7 | 537.0 | 28.9 | Face detection model based on PicoDet_LCNet_x2_5 |

**Note: The evaluation set for the above accuracy metrics is wider_face dataset mAP(0.5:0.95). GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**
</details>

## III. Quick Integration
Before quick integration, you need to install the PaddleX wheel package. For the installation method of the wheel package, please refer to the [PaddleX Local Installation Tutorial](../../../installation/installation_en.md). After installing the wheel package, a few lines of code can complete the inference of the face detection module. You can switch models under this module freely, and you can also integrate the model inference of the face detection module into your project.

```python
from paddlex.inference import create_model 

model_name = "PicoDet_LCNet_x2_5_face"

model = create_model(model_name)
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_detection.png", batch_size=1)

for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

For more information on the usage of PaddleX's single-model inference API, please refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API_en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better face detection models. Before using PaddleX to develop face detection models, ensure you have installed the PaddleDetection plugin for PaddleX. The installation process can be found in the [PaddleX Local Installation Tutorial](../../../installation/installation_en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the corresponding dataset for the task module. PaddleX provides a data validation function for each module, and **only data that passes the validation can be used for model training**. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development based on the official demos. If you wish to use private datasets for subsequent model training, refer to the [PaddleX Object Detection Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/object_detection_en.md).

#### 4.1.1 Demo Data Download
You can use the following commands to download the demo dataset to a specified folder:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/widerface_coco_examples.tar -P ./dataset
tar -xf ./dataset/widerface_coco_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
A single command can complete data validation:

```bash
python main.py -c paddlex/configs/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
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
    "num_classes": 1,
    "train_samples": 500,
    "train_sample_paths": [
      "check_dataset/demo_img/0--Parade/0_Parade_marchingband_1_849.jpg",
      "check_dataset/demo_img/0--Parade/0_Parade_Parade_0_904.jpg",
      "check_dataset/demo_img/0--Parade/0_Parade_marchingband_1_799.jpg"
    ],
    "val_samples": 100,
    "val_sample_paths": [
      "check_dataset/demo_img/1--Handshaking/1_Handshaking_Handshaking_1_384.jpg",
      "check_dataset/demo_img/1--Handshaking/1_Handshaking_Handshaking_1_538.jpg",
      "check_dataset/demo_img/1--Handshaking/1_Handshaking_Handshaking_1_429.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/example_data/widerface_coco_examples",
  "show_type": "image",
  "dataset_type": "COCODetDataset"
}
```

The verification results mentioned above indicate that `check_pass` being `True` means the dataset format meets the requirements. Details of other indicators are as follows:

* `attributes.num_classes`: The number of classes in this dataset is 1;
* `attributes.train_samples`: The number of training samples in this dataset is 500;
* `attributes.val_samples`: The number of validation samples in this dataset is 100;
* `attributes.train_sample_paths`: The list of relative paths to the visualization images of training samples in this dataset;
* `attributes.val_sample_paths`: The list of relative paths to the visualization images of validation samples in this dataset;

The dataset verification also analyzes the distribution of sample numbers across all classes and generates a histogram (histogram.png):

![](/tmp/images/modules/face_det/01.png)

</details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)

After completing dataset verification, you can convert the dataset format or re-split the training/validation ratio by **modifying the configuration file** or **appending hyperparameters**.

<details>
  <summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>

**(1) Dataset Format Conversion**

Face detection does not support data format conversion.

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
python main.py -c paddlex/configs/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
```
After dataset splitting, the original annotation files will be renamed to `xxx.bak` in the original path.

The above parameters can also be set by appending command-line arguments:

```bash
python main.py -c paddlex/configs/face_detection/PicoDet_LCNet_x2_5_face.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/widerface_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>

### 4.2 Model Training

A single command is sufficient to complete model training, taking the training of PicoDet_LCNet_x2_5_face as an example:

```bash
python main.py -c paddlex/configs/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
```
The steps required are:

* Specify the path to the `.yaml` configuration file of the model (here it is `PicoDet_LCNet_x2_5_face.yaml`)
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`

Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters for Model Tasks](../../instructions/config_parameters_common.md).

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>

* During model training, PaddleX automatically saves model weight files, defaulting to `output`. To specify a save path, use the `-o Global.output` field in the configuration file.
* PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.
* When training other models, specify the corresponding configuration file. The correspondence between models and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.md).
After completing model training, all outputs are saved in the specified output directory (default is `./output/`), typically```markdown
Similar to model training, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it is `PicoDet_LCNet_x2_5_face.yaml`)
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path of the validation dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.md).
</details>

### **4.3 Model Evaluation**
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation, you can complete the evaluation with a single command:

```bash
python main.py -c paddlex/configs/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/widerface_coco_examples
```
Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model（here it's `PicoDet_LCNet_x2_5_face.yaml`）
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to [PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common.md)。

<details>
  <summary>👉 <b>More Details (Click to Expand)</b></summary>


When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as `-o Evaluate.weight_path=./output/best_model/best_model/model.pdparams`.

After completing the model evaluation, an `evaluate_result.json` file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully, and the model's evaluation metrics, including AP.

</details>

### **4.4 Model Inference**
After completing model training and evaluation, you can use the trained model weights for inference prediction. In PaddleX, model inference prediction can be achieved through two methods: command line and wheel package.

#### 4.4.1 Model Inference
* To perform inference prediction through the command line, simply use the following command:
```bash
python main.py -c paddlex/configs/face_detection/PicoDet_LCNet_x2_5_face.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_detection.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it is `PicoDet_LCNet_x2_5_face.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weight path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipeline or into your own project.

1. **Pipeline Integration**

The face detection module can be integrated into PaddleX pipelines such as **Face Recognition** (coming soon). Simply replace the model path to update the face detection module of the relevant pipeline. In pipeline integration, you can use high-performance deployment and service-oriented deployment to deploy your model.

2. **Module Integration**

The weights you produce can be directly integrated into the face detection module. You can refer to the Python example code in [Quick Integration](#三快速集成), simply replace the model with the path to your trained model.
