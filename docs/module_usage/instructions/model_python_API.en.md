---
comments: true
---

# PaddleX Single Model Python Usage Instructions

Before using Python scripts for single model quick inference, please ensure you have completed the installation of PaddleX following the [PaddleX Local Installation Tutorial](../../installation/installation.en.md).

## I. Usage Example

Taking the image classification model as an example, the usage is as follows:

```python
from paddlex import create_model
model = create_model(model_name="PP-LCNet_x1_0")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
In short, just three steps:

* Call the `create_model()` method to instantiate the prediction model object;
* Call the `predict()` method of the prediction model object to perform inference prediction;
* Call `print()`, `save_to_xxx()` and other related methods to print or save the prediction results.

## II. API Description

### 1. Instantiate the Prediction Model Object by Calling the `create_model()` Method

* `create_model`: Instantiate the prediction model object;
  * Parameters:
    * `model_name`: `str` type, model name, such as "PP-LCNet_x1_0", "/path/to/PP-LCNet_x1_0_infer/";
    * `model_dir`: `str` type, local path to directory of inference model files ，such as "/path/to/PP-LCNet_x1_0_infer/", default to `None`, means that use the official model specified by `model_name`;
    * `batch_size`: `int` type, default to `1`;
    * `device`: `str` type, used to set the inference device, such as "cpu", "gpu:2" for GPU settings. By default, using 0 id GPU if available, otherwise CPU;
    * `pp_option`: `PaddlePredictorOption` type, used to set the inference engine. Please refer to [4-Inference Backend Configuration](#4-inference-backend-configuration) for more details;
    * _`inference hyperparameters`_: used to set common inference hyperparameters. Please refer to specific model description document for details.
  * Return Value: `BasePredictor` type.

### 2. Perform Inference Prediction by Calling the `predict()` Method of the Prediction Model Object

* `predict`: Use the defined prediction model to predict the input data;
  * Parameters:
    * `input`: Any type, supports str type representing the path of the file to be predicted, or a directory containing files to be predicted, or a network URL; for CV models, supports numpy.ndarray representing image data; for TS models, supports pandas.DataFrame type data; also supports list types composed of the above types;
  * Return Value: `generator`, using `for-in` or `next()` to iterate, and the prediction result of one sample would be returned per call.

### 3. Visualize the Prediction Results

The prediction results support to be accessed, visualized, and saved, which can be achieved through corresponding attributes or methods, specifically as follows:

#### Attributes:

* `str`: Representation of the prediction result in `str` type;
  * Returns: A `str` type, the string representation of the prediction result.
* `json`: The prediction result in JSON format;
  * Returns: A `dict` type.
* `img`: The visualization image of the prediction result. Available only when the results support visual representation;
  * Returns: A `PIL.Image` type.
* `html`: The HTML representation of the prediction result. Available only when the results support representation in HTML format;
  * Returns: A `str` type.
* _`more attrs`_: The prediction result of different models support different representation methods. Please refer to the specific model tutorial documentation for details.

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
* `save_to_img()`: Visualizes the prediction result and saves it as an image. Available only when the results support representation in the form of images;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* `save_to_csv()`: Saves the prediction result as a CSV file. Available only when the results support representation in CSV format;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* `save_to_html()`: Saves the prediction result as an HTML file. Available only when the results support representation in HTML format;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* `save_to_xlsx()`: Saves the prediction result as an XLSX file. Available only when the results support representation in XLSX format;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.

### 4. Inference Backend Configuration

PaddleX supports configuring the inference backend through `PaddlePredictorOption`. Relevant APIs are as follows:

#### Attributes:

* `device`: Inference device;
  * Supports setting the device type and card number represented by `str`. Device types include 'gpu', 'cpu', 'npu', 'xpu', 'mlu', 'dcu'. When using an accelerator card, you can specify the card number, e.g., 'gpu:0' for GPU 0. By default, using 0 id GPU if available, otherwise CPU;
  * Return value: `str` type, the currently set inference device.
* `run_mode`: Inference backend;
  * Supports setting the inference backend as a `str` type, options include 'paddle', 'trt_fp32', 'trt_fp16', 'trt_int8', 'mkldnn', 'mkldnn_bf16'. 'mkldnn' is only selectable when the inference device is 'cpu'. The default is 'paddle';
  * Return value: `str` type, the currently set inference backend.
* `cpu_threads`: Number of CPU threads for the acceleration library, only valid when the inference device is 'cpu';
  * Supports setting an `int` type for the number of CPU threads for the acceleration library during CPU inference;
  * Return value: `int` type, the currently set number of threads for the acceleration library.

#### Methods:

* `get_support_run_mode`: Get supported inference backend configurations;
  * Parameters: None;
  * Return value: List type, the available inference backend configurations.
* `get_support_device`: Get supported device types for running;
  * Parameters: None;
  * Return value: List type, the available device types.
* `get_device`: Get the currently set device;
  * Parameters: None;
  * Return value: `str` type.
```
