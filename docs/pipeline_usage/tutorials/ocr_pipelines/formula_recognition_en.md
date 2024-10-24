[简体中文](formula_recognition.md) | English

# Formula Recognition Pipeline Tutorial

## 1. Introduction to the Formula Recognition Pipeline

Formula recognition is a technology that automatically identifies and extracts LaTeX formula content and its structure from documents or images. It is widely used in document editing and data analysis in fields such as mathematics, physics, and computer science. Leveraging computer vision and machine learning algorithms, formula recognition converts complex mathematical formula information into editable LaTeX format, facilitating further data processing and analysis for users.

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/01.jpg)

**The Formula Recognition Pipeline comprises a layout detection module and a formula recognition module.**

**If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model size, choose a model with a smaller storage footprint.**

<details>
   <summary> 👉Model List Details</summary>

**Layout Detection Module Models**:

| Model Name | mAP (%) | GPU Inference Time (ms) | CPU Inference Time (ms) | Model Size (M) |
|-|-|-|-|-|
| RT-DETR-H_layout_17cls | 92.6 | 115.126 | 3827.25 | 470.2M |

**Note: The above accuracy metrics are evaluated on PaddleX's self-built layout detection dataset, containing 10,000 images. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

**Formula Recognition Module Models**:

| Model Name | BLEU Score | Normed Edit Distance | ExpRate (%) | GPU Inference Time (ms) | CPU Inference Time (ms) | Model Size |
|-|-|-|-|-|-|-|
| LaTeX_OCR_rec | 0.8821 | 0.0823 | 40.01 | - | - | 89.7 M |

**Note: The above accuracy metrics are measured on the [LaTeX-OCR Formula Recognition Test Set](https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO). All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

</details>

## 2. Quick Start
PaddleX supports experiencing the effects of the formula recognition pipeline through command line or Python locally.

Before using the formula recognition pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Guide](../../../installation/installation_en.md).

### 2.1 Experience via Command Line
Experience the formula recognition pipeline with a single command, using the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png), and replace `--input` with your local path for prediction:

```bash
paddlex --pipeline formula_recognition --input general_formula_recognition.png --device gpu:0
```

Parameter Explanation:

```
--pipeline: The pipeline name, which is formula_recognition for this case.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the second and third GPUs). Alternatively, use CPU (--device cpu).
```

When executing the above command, the default formula recognition pipeline configuration file is loaded. If you need to customize the configuration file, you can run the following command to obtain it:

<details>
   <summary> 👉Click to Expand</summary>

```bash
paddlex --get_pipeline_config formula_recognition
```

After execution, the formula recognition pipeline configuration file will be saved in the current directory. If you wish to customize the save location, you can run the following command (assuming the custom save location is `./my_path`):

```bash
paddlex --get_pipeline_config formula_recognition --save_path ./my_path
```

After obtaining the Pipeline configuration file, replace `--pipeline` with the configuration file's save path to make the configuration file effective. For example, if the configuration file is saved as  `./formula_recognition.yaml`, simply execute:
```bash
paddlex --pipeline ./formula_recognition.yaml --input general_formula_recognition.png --device gpu:0
```
Here, parameters such as `--model` and `--device` do not need to be specified, as they will use the parameters in the configuration file. If parameters are still specified, the specified parameters will take precedence.

</details>

After execution, the result is:
<details>
   <summary> 👉Click to Expand</summary>

```
{'input_path': 'general_formula_recognition.png', 'layout_result': {'input_path': 'general_formula_recognition.png', 'boxes': [{'cls_id': 3, 'label': 'number', 'score': 0.7580855488777161, 'coordinate': [1028.3635, 205.46213, 1038.953, 222.99033]}, {'cls_id': 0, 'label': 'paragraph_title', 'score': 0.8882032632827759, 'coordinate': [272.75305, 204.50894, 433.7473, 226.17996]}, {'cls_id': 2, 'label': 'text', 'score': 0.9685840606689453, 'coordinate': [272.75928, 282.17773, 1041.9316, 374.44687]}, {'cls_id': 2, 'label': 'text', 'score': 0.9559416770935059, 'coordinate': [272.39056, 385.54114, 1044.1521, 443.8598]}, {'cls_id': 2, 'label': 'text', 'score': 0.9610629081726074, 'coordinate': [272.40817, 467.2738, 1045.1033, 563.4855]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8916195034980774, 'coordinate': [503.45743, 594.6236, 1040.6804, 619.73895]}, {'cls_id': 2, 'label': 'text', 'score': 0.973675549030304, 'coordinate': [272.32007, 648.8599, 1040.8702, 775.15686]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9038916230201721, 'coordinate': [554.2307, 803.5825, 1040.4657, 855.3159]}, {'cls_id': 2, 'label': 'text', 'score': 0.9025381803512573, 'coordinate': [272.535, 875.1402, 573.1086, 898.3587]}, {'cls_id': 2, 'label': 'text', 'score': 0.8336610794067383, 'coordinate': [317.48013, 909.60864, 966.8498, 933.7868]}, {'cls_id': 2, 'label': 'text', 'score': 0.8779091238975525, 'coordinate': [19.704018, 653.322, 72.433235, 1215.1992]}, {'cls_id': 2, 'label': 'text', 'score': 0.8832409977912903, 'coordinate': [272.13028, 958.50806, 1039.7928, 1019.476]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9088466167449951, 'coordinate': [517.1226, 1042.3978, 1040.2208, 1095.7457]}, {'cls_id': 2, 'label': 'text', 'score': 0.9587949514389038, 'coordinate': [272.03336, 1112.9269, 1041.0201, 1206.8417]}, {'cls_id': 2, 'label': 'text', 'score': 0.8885666131973267, 'coordinate': [271.7495, 1231.8752, 710.44495, 1255.7981]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8907185196876526, 'coordinate': [581.2295, 1287.4525, 1039.8014, 1312.772]}, {'cls_id': 2, 'label': 'text', 'score': 0.9559596180915833, 'coordinate': [273.1827, 1341.421, 1041.0299, 1401.7255]}, {'cls_id': 2, 'label': 'text', 'score': 0.875311553478241, 'coordinate': [272.8338, 1427.3711, 789.7108, 1451.1359]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9152213931083679, 'coordinate': [524.9582, 1474.8136, 1041.6333, 1530.7142]}, {'cls_id': 2, 'label': 'text', 'score': 0.9584835767745972, 'coordinate': [272.81665, 1549.524, 1042.9962, 1608.7157]}]}, 'ocr_result': {}, 'table_result': [], 'dt_polys': [array([[ 503.45743,  594.6236 ],
       [1040.6804 ,  594.6236 ],
       [1040.6804 ,  619.73895],
       [ 503.45743,  619.73895]], dtype=float32), array([[ 554.2307,  803.5825],
       [1040.4657,  803.5825],
       [1040.4657,  855.3159],
       [ 554.2307,  855.3159]], dtype=float32), array([[ 517.1226, 1042.3978],
       [1040.2208, 1042.3978],
       [1040.2208, 1095.7457],
       [ 517.1226, 1095.7457]], dtype=float32), array([[ 581.2295, 1287.4525],
       [1039.8014, 1287.4525],
       [1039.8014, 1312.772 ],
       [ 581.2295, 1312.772 ]], dtype=float32), array([[ 524.9582, 1474.8136],
       [1041.6333, 1474.8136],
       [1041.6333, 1530.7142],
       [ 524.9582, 1530.7142]], dtype=float32)], 'rec_formula': ['F({\bf x})=C(F_{1}(x_{1}),\cdot\cdot\cdot,F_{N}(x_{N})).\qquad\qquad\qquad(1)', 'p(\mathbf{x})=c(\mathbf{u})\prod_{i}p(x_{i}).\qquad\qquad\qquad\qquad\qquad\quad\quad~~\quad~~~~~~~~~~~~~~~(2)', 'H_{c}({\bf x})=-\int_{{\bf{u}}}c({\bf{u}})\log c({\bf{u}})d{\bf{u}}.~~~~~~~~~~~~~~~~~~~~~(3)', 'I({\bf x})=-H_{c}({\bf x}).\qquad\qquad\qquad\qquad(4)', 'H({\bf x})=\sum_{i}H(x_{i})+H_{c}({\bf x}).\eqno\qquad\qquad\qquad(5)']}
```

Where `dt_polys` represents the coordinates of the detected formula area, and `rec_formula` is the detected formula.
</details>

The visualization result is as follows:
![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/02.jpg)

The visualized image not saved by default. You can customize the save path through `--save_path`, and then all results will be saved in the specified path. Formula recognition visualization requires a separate environment configuration. Please refer to [2.3 Formula Recognition Pipeline Visualization](#23-formula-recognition-pipeline-visualization) to install the LaTeX rendering engine.


#### 2.2 Python Script Integration
* Quickly perform inference on the pipeline with just a few lines of code, taking the formula recognition pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="formula_recognition")

output = pipeline.predict("general_formula_recognition.png")
for res in output:
    res.print()
```
> ❗ The results obtained from running the Python script are the same as those from the command line.

The Python script above executes the following steps:

（1）Instantiate the formula recognition pipeline object using `create_pipeline`: Specific parameter descriptions are as follows:

| Parameter | Description | Type | Default |
|-|-|-|-|
|`pipeline`| The name of the pipeline or the path to the pipeline configuration file. If it is the name of the pipeline, it must be supported by PaddleX. |`str`|None|
|`device`| The device for pipeline model inference. Supports: "gpu", "cpu". |`str`|`gpu`|
|`use_hpip`| Whether to enable high-performance inference, only available if the pipeline supports it. |`bool`|`False`|

（2）Invoke the `predict` method of the formula recognition pipeline object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Parameter Description |
|---------------|-----------------------------------------------------------------------------------------------------------|
| Python Var    | Supports directly passing in Python variables, such as numpy.ndarray representing image data. |
| str         | Supports passing in the path of the file to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| str           | Supports passing in the URL of the file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png). |
| str           | Supports passing in a local directory, which should contain files to be predicted, such as the local path: `/root/data/`. |
| dict          | Supports passing in a dictionary type, where the key needs to correspond to a specific task, such as "img" for image classification tasks. The value of the dictionary supports the above types of data, for example: `{"img": "/root/data1"}`. |
| list          | Supports passing in a list, where the list elements need to be of the above types of data, such as `[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]`. |

（3）Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

| Method         | Description                     | Method Parameters |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | Prints results to the terminal  | `- format_json`: bool, whether to format the output content with json indentation, default is True;<br>`- indent`: int, json formatting setting, only valid when format_json is True, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, only valid when format_json is True, default is False; |
| save_to_json | Saves results as a json file   | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;<br>`- indent`: int, json formatting setting, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, default is False; |
| save_to_img  | Saves results as an image file | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type; |

If you have a configuration file, you can customize the configurations of the formula recognition pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/formula_recognition.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/formula_recognition.yaml")
output = pipeline.predict("general_formula_recognition.png")
for res in output:
    res.print()
```

### 2.3 Formula Recognition Pipeline Visualization
If you need to visualize the formula recognition pipeline, you need to run the following command to install the LaTeX rendering environment:
```python
apt-get install sudo
sudo apt-get update
sudo apt-get install texlive
sudo apt-get install texlive-latex-base
sudo apt-get install texlive-latex-extra
python -m pip install PyMuPDF==1.24.12
```
After that, use the `save_to_img` method to save the visualization image. The specific command is as follows:
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="formula_recognition")

output = pipeline.predict("general_formula_recognition.png")
for res in output:
    res.print()
    res.save_to_img("./output/")
```
**Note**: Since the formula recognition visualization process requires rendering each formula image, it may take a relatively long time. Please be patient.

## 3. Development Integration/Deployment
If the formula recognition pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to apply the formula recognition pipeline directly in your Python project, refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 **High-Performance Inference**: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing for significant end-to-end speedups. For detailed high-performance inference procedures, refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference_en.md).

☁️ **Service-Oriented Deployment**: Service-oriented deployment is a common deployment form in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users in achieving low-cost service-oriented deployment of pipelines. For detailed service-oriented deployment procedures, refer to the [PaddleX Service-Oriented Deployment Guide](../../../pipeline_deploy/service_deploy_en.md).

Below are the API references and multi-language service invocation examples:

<details>
<summary>API Reference</summary>

For all operations provided by the service:

- Both the response body and the request body for POST requests are JSON data (JSON objects).
- When the request is processed successfully, the response status code is `200`, and the response body properties are as follows:

    | Name | Type | Description |
    |------|------|-------------|
    |`errorCode`|`integer`|Error code. Fixed as `0`.|
    |`errorMsg`|`string`|Error description. Fixed as `"Success"`.|

    The response body may also have a `result` property of type `object`, which stores the operation result information.

- When the request is not processed successfully, the response body properties are as follows:

    | Name | Type | Description |
    |------|------|-------------|
    |`errorCode`|`integer`|Error code. Same as the response status code.|
    |`errorMsg`|`string`|Error description.|

Operations provided by the service:

- **`infer`**

    Obtain formula recognition results from an image.

    `POST /formula-recognition`

    - Request body properties:

        | Name | Type | Description | Required |
        |------|------|-------------|----------|
        |`image`|`string`|The URL of an image file accessible by the service or the Base64 encoded result of the image file content.|Yes|
        |`inferenceParams`|`object`|Inference parameters.|No|

        Properties of `inferenceParams`:

        | Name | Type | Description | Required |
        |------|------|-------------|----------|
        |`maxLongSide`|`integer`|During inference, if the length of the longer side of the input image for the layout detection model is greater than `maxLongSide`, the image will be scaled so that the length of the longer side equals `maxLongSide`.|No|

    - When the request is processed successfully, the `result` in the response body has the following properties:

        | Name | Type | Description |
        |------|------|-------------|
        |`formulas`|`array`|Positions and contents of formulas.|
        |`image`|`string`|Formula recognition result image with detected formula positions annotated. The image is in JPEG format and encoded in Base64.|

        Each element in `formulas` is an `object` with the following properties:

        | Name | Type | Description |
        |------|------|-------------|
        |`poly`|`array`|Formula position. Elements in the array are the vertex coordinates of the polygon enclosing the formula.|
        |`latex`|`string`|Formula content.|

        Example of `result`:

        ```json
        {
          "formulas": [
            {
              "poly": [
                [
                  444.0,
                  244.0
                ],
                [
                  705.4,
                  244.5
                ],
                [
                  705.8,
                  311.3
                ],
                [
                  444.1,
                  311.0
                ]
              ],
              "latex": "F({\bf x})=C(F_{1}(x_{1}),\cdot\cdot\cdot,F_{N}(x_{N})).\qquad\qquad\qquad(1)"
            }
          ],
          "image": "xxxxxx"
        }
        ```

</details>

<details>
<summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>

```python
import base64
import requests

API_URL = "http://localhost:8080/formula-recognition"
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected formulas:")
print(result["formulas"])
```

</details>

<details>
<summary>C++</summary>

```cpp
#include <iostream>
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client("localhost:8080");
    const std::string imagePath = "./demo.jpg";
    const std::string outputImagePath = "./out.jpg";

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["image"] = encodedImage;
    std::string body = jsonObj.dump();

    auto response = client.Post("/formula-recognition", headers, body, "application/json");
    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        encodedImage = result["image"];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector<unsigned char> decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast<char*>(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout << "Output image saved at " << outPutImagePath << std::endl;
        } else {
            std::cerr << "Unable to open file for writing: " << outPutImagePath << std::endl;
        }

        auto formulas = result["formulas"];
        std::cout << "\nDetected formulas:" << std::endl;
        for (const auto& formula : formulas) {
            std::cout << formula << std::endl;
        }
    } else {
        std::cout << "Failed to send HTTP request." << std::endl;
        return 1;
    }

    return 0;
}
```

</details>

<details>
<summary>Java</summary>

```java
import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/formula-recognition";
        String imagePath = "./demo.jpg";
        String outputImagePath = "./out.jpg";

        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("image", imageData);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get("result");
                String base64Image = result.get("image").asText();
                JsonNode formulas = result.get("formulas");

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println("Output image saved at " + outputImagePath);
                System.out.println("\nDetected formulas: " + formulas.toString());
            } else {
                System.err.println("Request failed with code: " + response.code());
            }
        }
    }
}
```

</details>

<details>
<summary>Go</summary>

```go
package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
)

func main() {
    API_URL := "http://localhost:8080/formula-recognition"
    imagePath := "./demo.jpg"
    outputImagePath := "./out.jpg"

    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println("Error reading image file:", err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{"image": imageData}
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println("Error marshaling payload:", err)
        return
    }

    client := &http.Client{}
    req, err := http.NewRequest("POST", API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println("Error creating request:", err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println("Error sending request:", err)
        return
    }
    defer res.Body.Close()

    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }
    type Response struct {
        Result struct {
            Image      string   `json:"image"`
            Formulas []map[string]interface{} `json:"formulas"`
        } `json:"result"`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &respData)
    if err != nil {
        fmt.Println("Error unmarshaling response body:", err)
        return
    }

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println("Error decoding base64 image data:", err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println("Error writing image to file:", err)
        return
    }
    fmt.Printf("Image saved at %s.jpg\n", outputImagePath)
    fmt.Println("\nDetected formulas:")
    for _, formula := range respData.Result.Formulas {
        fmt.Println(formula)
    }
}
```

</details>

<details>
<summary>C#</summary>

```csharp
using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = "http://localhost:8080/formula-recognition";
    static readonly string imagePath = "./demo.jpg";
    static readonly string outputImagePath = "./out.jpg";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { "image", image_data } };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse["result"]["image"].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($"Output image saved at {outputImagePath}");
        Console.WriteLine("\nDetected formulas:");
        Console.WriteLine(jsonResponse["result"]["formulas"].ToString());
    }
}
```

</details>

<details>
<summary>Node.js</summary>

```js
const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/formula-recognition'
const imagePath = './demo.jpg'
const outputImagePath = "./out.jpg";

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)
  })
};

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

axios.request(config)
.then((response) => {
    const result = response.data["result"];
    const imageBuffer = Buffer.from(result["image"], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) => {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log("\nDetected formulas:");
    console.log(result["formulas"]);
})
.catch((error) => {
  console.log(error);
});
```

</details>

<details>
<summary>PHP</summary>

```php
<?php

$API_URL = "http://localhost:8080/formula-recognition";
$image_path = "./demo.jpg";
$output_image_path = "./out.jpg";

$image_data = base64_encode(file_get_contents($image_path));
$payload = array("image" => $image_data);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)["result"];
file_put_contents($output_image_path, base64_decode($result["image"]));
echo "Output image saved at " . $output_image_path . "\n";
echo "\nDetected formulas:\n";
print_r($result["formulas"]);

?>
```

</details>
</details>
<br/>

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing capabilities on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy_en.md).
You can choose the appropriate deployment method based on your needs to proceed with subsequent AI application integration.


## 4. Custom Development
If the default model weights provided by the formula recognition pipeline do not meet your requirements for accuracy or speed in your specific scenario, you can try to further fine-tune the existing models using **your own domain-specific or application-specific data** to improve the recognition performance of the formula recognition pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the formula recognition pipeline consists of two modules (layout detection and formula recognition), unsatisfactory performance may stem from either module.

You can analyze images with poor recognition results. If you find that many formula are undetected (i.e., formula miss detection), it may indicate that the layout detection model needs improvement. You should refer to the [Customization](../../../module_usage/tutorials/ocr_modules/layout_detection_en.md#iv-custom-development) section in the [Layout Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/layout_detection_en.md) and use your private dataset to fine-tune the layout detection model. If many recognition errors occur in detected formula (i.e., the recognized formula content does not match the actual formula content), it suggests that the formula recognition model requires further refinement. You should refer to the [Customization](../../../module_usage/tutorials/ocr_modules/formula_recognition_en.md#iv-custom-development) section in the [Formula Recognition Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/formula_recognition_en.md) and fine-tune the formula recognition model.

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain local model weights files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local paths of the fine-tuned model weights to the corresponding positions in the pipeline configuration file:

```bash
......
Pipeline:
  layout_model: RT-DETR-H_layout_17cls # Can be replaced with the local path of the fine-tuned layout detection model
  formula_rec_model: LaTeX_OCR_rec # Can be replaced with the local path of the fine-tuned formula recognition model
  formula_rec_batch_size: 5
  device: "gpu:0"
......
```

Then, refer to the command line method or Python script method in [2. Quick Start](#2-quick-start) to load the modified pipeline configuration file.

## 5. Multi-Hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPU, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modifying the `--device` parameter** allows seamless switching between different hardware.

For example, if you are using an NVIDIA GPU for formula pipeline inference, the Python command would be:

```bash
paddlex --pipeline formula_recognition --input general_formula_recognition.png --device gpu:0
```
Now, if you want to switch the hardware to Ascend NPU, you only need to modify the `--device` in the Python command:


```bash
paddlex --pipeline formula_recognition --input general_formula_recognition.png --device npu:0
```

If you want to use the formula recognition pipeline on more types of hardware, please refer to the [PaddleX Multi-Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide_en.md).
