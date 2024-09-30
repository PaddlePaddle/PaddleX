# Overview of PaddleX Model Pipeline Usage

If you have already experienced the pre-trained model pipeline effects in PaddleX and wish to proceed directly with model fine-tuning, you can jump to [Model Selection](#3-optional-model-selection).

The complete PaddleX model pipeline development process is illustrated in the following diagram:

**<center>PaddleX Model Pipeline Development Flowchart</center>**

```mermaid
graph LR
    select_pipeline(Select Pipeline) --> online_experience[Quick Experience]
    online_experience --> online_ok{Satisfied with Results?}
    online_ok --No--> select_model[Select Model]
    select_model --> model_finetune[Model Fine-tuning]
    online_ok --Yes--> development_integration(Development Integration/Deployment)
    model_finetune --> pipeline_test[Pipeline Testing]
    pipeline_test --> test_ok{Satisfied with Results?}
    test_ok --No--> select_model
    test_ok --Yes--> development_integration
```

The pre-trained model pipelines provided by PaddleX allow for **quick experience of effects**. If the pipeline effects meet your requirements, you can directly proceed with **development integration/deployment** of the pre-trained model pipeline. If the effects are not as expected, you can use your private data to **fine-tune** the models within the pipeline until satisfactory results are achieved.

Below, let's take the task of boarding pass recognition as an example to introduce the local usage process of the PaddleX model pipeline tool. Before use, please ensure you have completed the installation of PaddleX according to the [PaddleX Local Installation Tutorial](../installation/installation_en.md).

## 1. Select Pipeline

Each pipeline in PaddleX can solve specific task scenarios such as object detection, time series prediction, semantic segmentation, etc. You need to select the pipeline for subsequent development based on the specific task. For example, for the boarding pass recognition task, the corresponding PaddleX pipeline is the **General OCR Pipeline**. More task-pipeline correspondences can be found in the [PaddleX Pipeline List (CPU/GPU)](../support_list/pipelines_list_en.md).

## 2. Quick Start

Each pipeline in PaddleX integrates numerous pre-trained models. You can first experience the effects of the PaddleX pre-trained model pipeline. If the effects of the pre-trained model pipeline meet your expectations, you can proceed directly with [Development Integration/Deployment](#6-development-integration-deployment). If not, optimize the pipeline effects according to the subsequent steps.

PaddleX provides three ways to quickly experience pipeline effects. You can choose the appropriate method based on your needs:

* Online Quick Experience URL: [PaddleX Pipeline List (CPU/GPU)](../support_list/pipelines_list_en.md)
* Command Line Quick Experience: [PaddleX Pipeline Command Line Usage Instructions](../pipeline_usage/instructions/pipeline_CLI_usage_en.md)
* Python Script Quick Experience: [PaddleX Pipeline Python API Usage Instructions](../pipeline_usage/instructions/pipeline_python_API_en.md)

Taking the General OCR pipeline for boarding pass recognition as an example, a single command can quickly experience the pipeline effects:

```bash
paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0
```
Parameter Explanation:

```bash
--pipeline: Name of the pipeline, here it is OCR pipeline
--input: Local path or URL of the input image to be processed
--device: GPU index to use (e.g., gpu:0 indicates using the first GPU, gpu:1,2 indicates using the 1st and 2nd GPUs), or choose to use CPU (--device cpu)
```
After execution, you will be prompted to select the OCR pipeline configuration file save path, which defaults to the *current directory*, or you can *customize the path*.

Additionally, by adding the `-y` parameter to the command, you can skip the path selection and directly save the pipeline configuration file to the current directory.

After obtaining the pipeline configuration file, you can replace `--pipeline` with the saved configuration file path to make the configuration file take effect. For example, if the configuration file save path is `./ocr.yaml`, simply execute:

```bash
paddlex --pipeline ./ocr.yaml --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png
```
Where `--model`, `--device`, and other parameters are not specified, the parameters in the configuration file will be used. If parameters are still specified, the specified parameters will take precedence.

After running, the results are:

```bash
The prediction result is:
['登机口于起飞前10分钟关闭']
The prediction result is:
['GATES CLOSE 10 MINUTES BEFORE DEPARTURE TIME']
The prediction result is:
['ETKT7813699238489/1']
......
```
The visualization result is as follows:

![](/tmp/images/boardingpass.png)

## 3. Model Selection (Optional)

Since a pipeline may contain one or more models, when fine-tuning models, you need to determine which model to fine-tune based on testing results. Taking the OCR pipeline for boarding pass recognition as an example, this pipeline includes a text detection model (e.g., `PP-OCRv4_mobile_det`) and a text recognition model (e.g., `PP-OCRv4_mobile_rec`). If the text positioning is inaccurate, you need to fine-tune the text detection model. If the text recognition is inaccurate, you need to fine-tune the text recognition model. If you are unsure which models are included in the pipeline, you can refer to the [PaddleX Pipeline List (CPU/GPU)](../support_list/pipelines_list_en.md)

## 4. Model Fine-tuning (Optional)

After determining the model to fine-tune, you need to train the model with your private dataset. PaddleX provides a single-model development tool that can complete model training with a single command:

```bash
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=your/dataset_dir
```
In addition, PaddleX provides detailed tutorials for preparing private datasets for model fine-tuning, single-model inference, and more. For details, please refer to the [PaddleX Single-Model Development Tool Usage Guide](../module_usage/module_develop_guide_en.md)

## 5. Production Line Testing (Optional)

After fine-tuning your model with a private dataset, you will obtain local model weight files.

To use the fine-tuned model weights, simply modify the production line configuration file by replacing the local paths of the fine-tuned model weights with the corresponding paths in the configuration file:

```bash
......
Pipeline:
  det_model: PP-OCRv4_server_det  # Can be modified to the local path of the fine-tuned text detection model
  det_device: "gpu"
  rec_model: PP-OCRv4_server_rec  # Can be modified to the local path of the fine-tuned text recognition model
  rec_batch_size: 1
  rec_device: "gpu"
......
```
Then, refer to the command line method or Python script method in [Quick Start](#2-Quick-Start) to load the modified production line configuration file.

If the results are satisfactory, proceed with [Development Integration/Deployment](#6-Development-Integration-and-Deployment). If not, return to [Model Selection](#3-Model-Selection-(Optional)) to continue fine-tuning other task modules until you achieve satisfactory results.

## 6. Development Integration and Deployment

PaddleX provides concise Python APIs that allow you to integrate model pipelines into your projects with just a few lines of code. Here's an example code snippet for integrating a boarding pass recognition OCR pipeline:

```bash
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="OCR")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png")
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more detailed Python integration methods for model pipelines, refer to [PaddleX Pipeline Python API Usage Instructions](../pipeline_usage/instructions/pipeline_python_API_en.md)

PaddleX also provides three deployment methods with detailed deployment tutorials:

🚀 **High-Performance Deployment**: In actual production environments, many applications have stringent standards for the performance metrics (especially response speed) of deployment strategies to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins that aim to deeply optimize model inference and pre/post-processing for significant speedups in the end-to-end process. Refer to the [PaddleX High-Performance Deployment Guide](../pipeline_deploy/high_performance_deploy_en.md) for detailed high-performance deployment procedures.

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service-oriented deployment of pipelines. Refer to the [PaddleX Service-Oriented Deployment Guide](../pipeline_deploy/service_deploy_en.md) for detailed service-oriented deployment procedures.

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing capabilities on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. Refer to the [PaddleX Edge Deployment Guide](../pipeline_deploy/lite_deploy_en.md) for detailed edge deployment procedures.

Choose the appropriate deployment method for your model pipeline based on your needs, and proceed with subsequent AI application integration.

> ❗ Warm Reminder: PaddleX provides detailed usage instructions for each pipeline. You can choose according to your needs. The usage instructions for all pipelines are as follows:

| Pipeline Name          | Detailed Description                                                                                            |
|------------------------|-------------------------------------------------------------------------------------------------------------------|
| Document Scene Information Extraction v3 | [Python Script Usage Instructions for Document Scene Information Extraction v3 Pipeline](/docs_new_en/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction_en.md) |
| General Image Classification       | [Python Script Usage Instructions for General Image Classification Pipeline](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_classification_en.md) |
| General Object Detection       | [Python Script Usage Instructions for General Object Detection Pipeline](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/object_detection_en.md) |
| General Instance Segmentation       | [Python Script Usage Instructions for General Instance Segmentation Pipeline](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/instance_segmentation_en.md) |
| General Semantic Segmentation       | [Python Script Usage Instructions for General Semantic Segmentation Pipeline](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation_en.md) |
| General Image Multi-Label Classification | [Python Script Usage Instructions for General Image Multi-Label Classification Pipeline](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification_en.md) |
| Small Object Detection         | [Python Script Usage Instructions for Small Object Detection Pipeline](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/small_object_detection_en.md) |
| Image Anomaly Detection       | [Python Script Usage Instructions for Image Anomaly Detection Pipeline](/docs_new_en/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection_en.md) |
| General OCR            | [Python Script Usage Instructions for General OCR Pipeline](/docs_new_en/pipeline_usage/tutorials/ocr_pipelines/OCR_en.md) |
| General Table Recognition       | [Python Script Usage Instructions for General Table Recognition Pipeline](/docs_new_en/pipeline_usage/tutorials/ocr_pipelines/table_recognition_en.md) |
| General Time Series Forecasting       | [Python Script Usage Instructions for General Time Series Forecasting Pipeline](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting_en.md) |
| General Time Series Anomaly Detection   | [Python Script Usage Instructions for General Time Series Anomaly Detection Pipeline](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection_en.md) |
| General Time Series Classification       | [Python Script Usage Instructions for General Time Series Classification Pipeline](/docs_new_en/pipeline_usage/tutorials/time_series_pipelines/time_series_classification_en.md) |
