[简体中文](pipeline_CLI_usage.md) | English

# PaddleX Pipeline CLI Usage Instructions

Before using the CLI command line for rapid inference of the pipeline, please ensure that you have completed the installation of PaddleX according to the [PaddleX Local Installation Tutorial](../../installation/installation_en.md).

## I. Usage Example

### 1. Quick Experience

Taking the image classification pipeline as an example, the usage is as follows:

```bash
paddlex --pipeline image_classification \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
        --device gpu:0 \
        --save_path ./output/
```
This single step completes the inference prediction and saves the prediction results. Explanations for the relevant parameters are as follows:

* `pipeline`: The name of the pipeline or the local path to the pipeline configuration file, such as the pipeline name "image_classification", or the path to the pipeline configuration file "path/to/image_classification.yaml";
* `input`: The path to the data file to be predicted, supporting local file paths, local directories containing data files to be predicted, and file URL links;
* `device`: Used to set the model inference device. If set for GPU, you can specify the card number, such as "cpu", "gpu:2". When not specified, if GPU is available, it will be used; otherwise, CPU will be used;
* `save_path`: The save path for prediction results. When not specified, the prediction results will not be saved;

### 2. Custom Pipeline Configuration

If you need to modify the pipeline configuration, you can retrieve the configuration file and modify it. Still taking the image classification pipeline as an example, the way to retrieve the configuration file is as follows:

```bash
paddlex --get_pipeline_config image_classification

# Please enter the path that you want to save the pipeline config file: (default `./`)
./configs/

# The pipeline config has been saved to: configs/image_classification.yaml
```

After modifying the production line configuration file `configs/image_classification.yaml`, such as the content for the image classification configuration file:

```yaml
Global:
  pipeline_name: image_classification
  input: https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg

Pipeline:
  model: PP-LCNet_x0_5
  batch_size: 1
  device: "gpu:0"
```

Once the modification is completed, you can use this configuration file to perform model pipeline inference prediction as follows:

```bash
paddlex --pipeline configs/image_classification.yaml \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
        --save_path ./output/

# {'input_path': '/root/.paddlex/predict_input/general_image_classification_001.jpg', 'class_ids': [296, 170, 356, 258, 248], 'scores': array([0.62817, 0.03729, 0.03262, 0.03247, 0.03196]), 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Irish wolfhound', 'weasel', 'Samoyed, Samoyede', 'Eskimo dog, husky']}
```
