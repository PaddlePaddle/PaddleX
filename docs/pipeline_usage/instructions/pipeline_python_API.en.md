---
comments: true
---

# PaddleX Model Pipeline Python Usage Instructions

Before using Python scripts for rapid inference on model pipelines, please ensure you have installed PaddleX following the [PaddleX Local Installation Guide](../../installation/installation.en.md).

## I. Usage Example

Taking the image classification pipeline as an example, the usage is as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline("image_classification")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg", batch_size=1, topk=5)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

In short, there are only three steps:

* Call the `create_pipeline()` method to instantiate the prediction model pipeline object;
* Call the `predict()` method of the prediction model pipeline object for inference;
* Call `print()`, `save_to_xxx()` and other related methods to print or save the prediction results.

## II. API Description

### 1. Instantiate the Prediction Model Pipeline Object by Calling `create_pipeline()`
* `create_pipeline`: Instantiates the prediction model pipeline object;
  * Parameters:
    * `pipeline`: `str` type, the pipeline name or the local pipeline configuration file path, such as "image_classification", "/path/to/image_classification.yaml";
    * `device`: `str` type, used to set the inference device. If set for GPU, you can specify the card number, such as "cpu", "gpu:2". By default, using 0 id GPU if available, otherwise CPU;
    * `pp_option`: `PaddlePredictorOption` type, used to change inference settings (e.g. the operating mode). Please refer to [4-Inference Configuration](#4-inference-configuration) for more details;
    * `use_hpip`：`bool | None` type, whether to enable the high-performance inference plugin (`None` for using the setting from the configuration file);
    * `hpi_config`：`dict | None` type, high-performance inference configuration;
  * Return Value: `BasePipeline` type.

### 2. Perform Inference by Calling the `predict()` Method of the Prediction Model Pipeline Object

* `predict`: Uses the defined prediction model pipeline to predict input data;
  * Parameters:
    * `input`: Any type, supporting str representing the path of the file to be predicted, or a directory containing files to be predicted, or a network URL; for CV tasks, supports numpy.ndarray representing image data; for TS tasks, supports pandas.DataFrame type data; also supports lists of the above types;
  * Return Value: `generator`, returns the prediction result of one sample per call;

### 3. Visualize the Prediction Results

The prediction results of the pipeline support to be accessed and saved, which can be achieved through corresponding attributes or methods, specifically as follows:

#### Attributes:

* `str`: `str` type representation of the prediction result;
  * Return Value: `str` type, string representation of the prediction result;
* `json`: Prediction result in JSON format;
  * Return Value: `dict` type;
* `img`: Visualization image of the prediction result;
  * Return Value: `PIL.Image` type;
* `html`: HTML representation of the prediction result;
  * Return Value: `str` type;
* _`more attrs`_: The prediction result of different pipelines support different representation methods. Please refer to the specific pipeline tutorial documentation for details.

#### Methods:

* `print()`: Outputs the prediction result. Note that when the prediction result is not convenient for direct output, relevant content will be omitted;
  * Parameters:
    * `json_format`: `bool` type, default is `False`, indicating that json formatting is not used;
    * `indent`: `int` type, default is `4`, valid when `json_format` is `True`, indicating the indentation level for json formatting;
    * `ensure_ascii`: `bool` type, default is `False`, valid when `json_format` is `True`;
  * Return Value: None;
* `save_to_json()`: Saves the prediction result as a JSON file. Note that when the prediction result contains data that cannot be serialized in JSON, automatic format conversion will be performed to achieve serialization and saving;
  * Parameters:
    * `save_path`: `str` type, the path to save the result;
    * `indent`: `int` type, default is `4`, valid when `json_format` is `True`, indicating the indentation level for json formatting;
    * `ensure_ascii`: `bool` type, default is `False`, valid when `json_format` is `True`;
  * Return Value: None;
* `save_to_img()`: Visualizes the prediction result and saves it as an image;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* `save_to_csv()`: Saves the prediction result as a CSV file;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* `save_to_html()`: Saves the prediction result as an HTML file;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* `save_to_xlsx()`: Saves the prediction result as an XLSX file;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* _`more funcs`_: The prediction result of different pipelines support different saving methods. Please refer to the specific pipeline tutorial documentation for details.

### 4. Inference Configuration

PaddleX supports modifying the inference configuration through `PaddlePredictorOption`. Relevant APIs are as follows:

#### Attributes:

* `device`: Inference device.
  * Supports setting the device type and card number represented by `str`. Device types include 'gpu', 'cpu', 'npu', 'xpu', 'mlu', 'dcu'. When using an accelerator card, you can specify the card number, e.g., 'gpu:0' for GPU 0. By default, if a GPU is available, the one with the smallest ID number will be used; otherwise, the CPU will be used.
  * Return value: `str` type, the currently set inference device.
* `run_mode`: Operating mode.
  * Supports setting the operating mode as a `str` type, options include 'paddle', 'trt_fp32', 'trt_fp16', 'trt_int8', 'mkldnn', 'mkldnn_bf16'. Note that 'trt_fp32' and 'trt_fp16' correspond to using the TensorRT subgraph engine for inference with FP32 and FP16 precision respectively; these options are only available when the inference device is a GPU. Additionally, 'mkldnn' is only available when the inference device is a CPU. The default value is 'paddle'.
  * Return value: `str` type, the currently set operating mode.
* `cpu_threads`: Number of CPU threads for the acceleration library, only valid when the inference device is 'cpu'.
  * Supports setting an `int` type for the number of CPU threads for the acceleration library during CPU inference.
  * Return value: `int` type, the currently set number of threads for the acceleration library.
* `trt_dynamic_shapes`: TensorRT dynamic shape configuration, only effective when `run_mode` is set to 'trt_fp32' or 'trt_fp16'.
  * Supports setting a value of type `dict` or `None`. If it is a `dict`, the keys are the input tensor names and the values are two-level nested lists formatted as `[{minimum shape}, {optimal shape}, {maximum shape}]`, for example `[[1, 2], [1, 2], [2, 2]]`.
  * Return value: `dict` type or `None`, the current TensorRT dynamic shape configuration.
* `trt_dynamic_shape_input_data`: For TensorRT usage, this parameter provides the fill data for the input tensors used to build the engine, and it is only valid when `run_mode` is set to 'trt_fp32' or 'trt_fp16'.
  * Supports setting a value of type `dict` or `None`. If it is a `dict`, the keys are the input tensor names and the values are two-level nested lists formatted as `[{fill data corresponding to the minimum shape}, {fill data corresponding to the optimal shape}, {fill data corresponding to the maximum shape}]`, for example `[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]`. The data are floating point numbers filled in row-major order.
  * Return value: `dict` type or `None`, the currently set input tensor fill data.

#### Methods:

* `get_support_run_mode`: Get supported operating modes;
  * Parameters: None;
  * Return value: List type, the available operating modes.
* `get_support_device`: Get supported device types for running;
  * Parameters: None;
  * Return value: List type, the available device types.
* `get_device`: Get the currently set device;
  * Parameters: None;
  * Return value: `str` type.
