[简体中文](small_object_detection.md) | English

# Small Object Detection Pipeline Tutorial

## 1. Introduction to Small Object Detection Pipeline
Small object detection is a specialized technique for identifying tiny objects within images, widely applied in fields such as surveillance, autonomous driving, and satellite image analysis. It can accurately locate and classify small-sized objects like pedestrians, traffic signs, or small animals within complex scenes. By leveraging deep learning algorithms and optimized Convolutional Neural Networks (CNNs), small object detection significantly enhances the recognition capabilities for small objects, ensuring no critical information is overlooked in practical applications. This technology plays a pivotal role in enhancing safety and automation levels.

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/small_object_detection/01.png)

**The small object detection pipeline includes a small object detection module. If you prioritize model accuracy, choose a model with higher accuracy. If you prioritize inference speed, select a model with faster inference. If you prioritize model storage size, opt for a model with a smaller storage size.**

<details>
   <summary> 👉Model List Details</summary>

|Model Name|mAP (%)|GPU Inference Time (ms)|CPU Inference Time (ms)|Model Size (M)|
|-|-|-|-|-|
|PP-YOLOE_plus_SOD-S|25.1|65.4608|324.37|77.3 M|
|PP-YOLOE_plus_SOD-L|31.9|57.1448|1006.98|325.0 M|
|PP-YOLOE_plus_SOD-largesize-L|42.7|458.521|11172.7|340.5 M|

**Note: The above accuracy metrics are based on the **[VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset)** validation set mAP(0.5:0.95). All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.**

</details>

## 2. Quick Start
PaddleX supports experiencing the small object detection pipeline's effects through command line or Python locally.

Before using the small object detection pipeline locally, ensure you have installed the PaddleX wheel package following the [PaddleX Local Installation Tutorial](../../../installation/installation_en.md).

### 2.1 Experience via Command Line
Experience the small object detection pipeline with a single command, Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg), and replace `--input` with the local path to perform prediction.


```bash
paddlex --pipeline small_object_detection --input small_object_detection.jpg --device gpu:0
```
Parameter Explanation:

```
--pipeline: The name of the pipeline, here it's the small object detection pipeline.
--input: The local path or URL of the input image to be processed.
--device: The GPU index to use (e.g., gpu:0 for the first GPU, gpu:1,2 for the second and third GPUs). Alternatively, you can choose to use CPU (--device cpu).
```

When executing the above command, the default small object detection pipeline configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details>
   <summary> 👉Click to Expand</summary>

```bash
paddlex --get_pipeline_config small_object_detection
```
After execution, the small object detection pipeline configuration file will be saved in the current directory. If you wish to customize the save location, execute the following command (assuming the custom save location is `./my_path`):

```bash
paddlex --get_pipeline_config small_object_detection --save_path ./my_path
```

After obtaining the pipeline configuration file, replace `--pipeline` with the configuration file's save path to make the configuration file effective. For example, if the configuration file's save path is `./small_object_detection.yaml`, simply execute:

```bash
paddlex --pipeline ./small_object_detection.yaml --input small_object_detection.jpg --device gpu:0
```
Here, parameters like `--model` and `--device` do not need to be specified, as they will use the parameters in the configuration file.```markdown

</details>

After running, the result will be:

```
{'input_path': 'small_object_detection.jpg', 'boxes': [{'cls_id': 3, 'label': 'car', 'score': 0.9243856072425842, 'coordinate': [624, 638, 682, 741]}, {'cls_id': 3, 'label': 'car', 'score': 0.9206348061561584, 'coordinate': [242, 561, 356, 613]}, {'cls_id': 3, 'label': 'car', 'score': 0.9194547533988953, 'coordinate': [670, 367, 705, 400]}, {'cls_id': 3, 'label': 'car', 'score': 0.9162291288375854, 'coordinate': [459, 259, 523, 283]}, {'cls_id': 4, 'label': 'van', 'score': 0.9075379371643066, 'coordinate': [467, 213, 498, 242]}, {'cls_id': 4, 'label': 'van', 'score': 0.9066920876502991, 'coordinate': [547, 351, 577, 397]}, {'cls_id': 3, 'label': 'car', 'score': 0.9041045308113098, 'coordinate': [502, 632, 562, 736]}, {'cls_id': 3, 'label': 'car', 'score': 0.8934890627861023, 'coordinate': [613, 383, 647, 427]}, {'cls_id': 3, 'label': 'car', 'score': 0.8803309202194214, 'coordinate': [640, 280, 671, 309]}, {'cls_id': 3, 'label': 'car', 'score': 0.8727016448974609, 'coordinate': [1199, 256, 1259, 281]}, {'cls_id': 3, 'label': 'car', 'score': 0.8705748915672302, 'coordinate': [534, 410, 570, 461]}, {'cls_id': 3, 'label': 'car', 'score': 0.8654043078422546, 'coordinate': [669, 248, 702, 271]}, {'cls_id': 3, 'label': 'car', 'score': 0.8555219769477844, 'coordinate': [525, 243, 550, 270]}, {'cls_id': 3, 'label': 'car', 'score': 0.8522038459777832, 'coordinate': [526, 220, 553, 243]}, {'cls_id': 3, 'label': 'car', 'score': 0.8392605185508728, 'coordinate': [557, 141, 575, 158]}, {'cls_id': 3, 'label': 'car', 'score': 0.8353804349899292, 'coordinate': [537, 120, 553, 133]}, {'cls_id': 3, 'label': 'car', 'score': 0.8322211503982544, 'coordinate': [585, 132, 603, 147]}, {'cls_id': 3, 'label': 'car', 'score': 0.8298957943916321, 'coordinate': [701, 283, 736, 313]}, {'cls_id': 3, 'label': 'car', 'score': 0.8217393159866333, 'coordinate': [885, 347, 943, 377]}, {'cls_id': 3, 'label': 'car', 'score': 0.820313572883606, 'coordinate': [493, 150, 511, 168]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.8183429837226868, 'coordinate': [203, 701, 224, 743]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.815082848072052, 'coordinate': [185, 710, 201, 744]}, {'cls_id': 6, 'label': 'tricycle', 'score': 0.7892289757728577, 'coordinate': [311, 371, 344, 407]}, {'cls_id': 6, 'label': 'tricycle', 'score': 0.7812919020652771, 'coordinate': [345, 380, 388, 405]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7748346328735352, 'coordinate': [295, 500, 309, 532]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7688500285148621, 'coordinate': [851, 436, 863, 466]}, {'cls_id': 3, 'label': 'car', 'score': 0.7466475367546082, 'coordinate': [565, 114, 580, 128]}, {'cls_id': 3, 'label': 'car', 'score': 0.7156463265419006, 'coordinate': [483, 66, 495, 78]}, {'cls_id': 3, 'label': 'car', 'score': 0.704211950302124, 'coordinate': [607, 138, 642, 152]}, {'cls_id': 3, 'label': 'car', 'score': 0.7021926045417786, 'coordinate': [505, 72, 518, 83]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.6897469162940979, 'coordinate': [802, 460, 815, 488]}, {'cls_id': 3, 'label': 'car', 'score': 0.671891450881958, 'coordinate': [574, 123, 593, 136]}, {'cls_id': 9, 'label': 'motorcycle', 'score': 0.6712754368782043, 'coordinate': [445, 317, 472, 334]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.6695684790611267, 'coordinate': [479, 309, 489, 332]}, {'cls_id': 3, 'label': 'car', 'score': 0.6273623704910278, 'coordinate': [654, 210, 677, 234]}, {'cls_id': 3, 'label': 'car', 'score': 0.6070230603218079, 'coordinate': [640, 166, 667, 185]}, {'cls_id': 3, 'label': 'car', 'score': 0.6064521670341492, 'coordinate': [461, 59, 476, 71]}, {'cls_id': 3, 'label': 'car', 'score': 0.5860581398010254, 'coordinate': [464, 87, 484, 100]}, {'cls_id': 9, 'label': 'motorcycle', 'score': 0.5792551636695862, 'coordinate': [390, 390, 419, 408]}, {'cls_id': 3, 'label': 'car', 'score': 0.5559225678443909, 'coordinate': [481, 125, 496, 140]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.5531904697418213, 'coordinate': [869, 306, 880, 331]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.5468509793281555, 'coordinate': [895, 294, 904, 319]}, {'cls_id': 3, 'label': 'car', 'score': 0.5451828241348267, 'coordinate': [505, 94, 518, 108]}, {'cls_id': 3, 'label': 'car', 'score': 0.5398445725440979, 'coordinate': [657, 188, 681, 208]}, {'cls_id': 4, 'label': 'van', 'score': 0.5318890810012817, 'coordinate': [518, 88, 534, 102]}, {'cls_id': 3, 'label': 'car', 'score': 0.5296525359153748, 'coordinate': [527, 71, 540, 81]}, {'cls_id': 6, 'label': 'tricycle', 'score': 0.5168400406837463, 'coordinate': [528, 320, 563, 346]}, {'cls_id': 3, 'label': 'car', 'score': 0.5088561177253723, 'coordinate': [511, 84, 530, 95]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.502006471157074, 'coordinate': [841, 266, 850, 283]}]}
```

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/small_object_detection/02.png)

The visualized image not saved by default. You can customize the save path through `--save_path`, and then all results will be saved in the specified path.

### 2.2 Integration via Python Script
A few lines of code can quickly enable inference on the production line. Taking the General Small Object Detection Pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="small_object_detection")

output = pipeline.predict("small_object_detection.jpg")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_img("./output/")  # Save the visualized image of the result
    res.save_to_json("./output/")  # Save the structured output of prediction
```
The results obtained are the same as those from the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: Specific parameter descriptions are as follows:

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `pipeline` | The name of the pipeline or the path to the pipeline configuration file. If it's a pipeline name, it must be supported by PaddleX. | `str` | None |
| `device` | The device for pipeline model inference. Supports: "gpu", "cpu". | `str` | "gpu" |
| `use_hpip` | Whether to enable high-performance inference, only available when the pipeline supports it. | `bool` | `False` |

(2) Call the `predict` method of the pipeline object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

| Parameter Type | Description |
|----------------|-------------|
| Python Var | Supports directly passing Python variables, such as numpy.ndarray representing image data. |
| `str` | Supports passing the path of the file to be predicted, such as the local path of an image file: `/root/data/img.jpg`. |
| `str` | Supports passing the URL of the file to be predicted, such as the network URL of an image file: [Example](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg). |
| `str` | Supports passing a local directory, which should contain files to be predicted, such as the local path: `/root/data/`. |
| `dict` | Supports passing a dictionary type, where the key needs to correspond to a specific task, such as "img" for image classification tasks. The value of the dictionary supports the above data types, for example: `{"img": "/root/data1"}`. |
| `list` | Supports passing a list, where the list elements need to be the above data types, such as `[numpy.ndarray, numpy.ndarray]`, `["/root/data/img1.jpg", "/root/data/img2.jpg"]`, `["/root/data1", "/root/data2"]`, `[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]`. |

(3) Obtain prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained by iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list representing a set of prediction results.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

| Method         | Description                     | Method Parameters |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | Prints results to the terminal  | `- format_json`: bool, whether to format the output content with json indentation, default is True;<br>`- indent`: int, json formatting setting, only valid when format_json is True, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, only valid when format_json is True, default is False; |
| save_to_json | Saves results as a json file   | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type;<br>`- indent`: int, json formatting setting, default is 4;<br>`- ensure_ascii`: bool, json formatting setting, default is False; |
| save_to_img  | Saves results as an image file | `- save_path`: str, the path to save the file, when it's a directory, the saved file name is consistent with the input file type; |

If you have a configuration file, you can customize the configurations of the image anomaly detection pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved at `./my_path/small_object_detection`, you only need to execute:


```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/small_object_detection.yaml")
output = pipeline.predict("small_object_detection.jpg")
for res in output:
    res.print()  # Print the structured output of prediction
    res.save_to_img("./output/")  # Save the visualization image of the result
    res.save_to_json("./output/")  # Save the structured output of prediction
```

## 3. Development Integration/Deployment
If the pipeline meets your requirements for inference speed and accuracy in production, you can proceed with development integration/deployment directly.

If you need to apply the pipeline directly in your Python project, refer to the example code in [2.2 Python Script Integration](#22-python-script-integration).

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
    | `errorCode` | `integer` | Error code. Fixed as `0`. |
    | `errorMsg` | `string` | Error description. Fixed as `"Success"`. |

    The response body may also have a `result` property of type `object`, which stores the operation result information.

- When the request is not processed successfully, the response body properties are as follows:

    | Name | Type | Description |
    |------|------|-------------|
    | `errorCode` | `integer` | Error code. Same as the response status code. |
    | `errorMsg` | `string` | Error description. |

Operations provided by the service are as follows:

- **`infer`**

    Performs object detection on an image.

    `POST /object-detection`

    - The request body properties are as follows:

        | Name | Type | Description | Required |
        |------|------|-------------|----------|
        | `image` | `string` | The URL of an image file accessible by the service or the Base64 encoded result of the image file content. | Yes |

    - When the request is processed successfully, the `result` of the response body has the following properties:

        | Name | Type | Description |
        |------|------|-------------|
        | `detectedObjects` | `array` | Information about the location and category of the detected objects. |
        | `image` | `string` | The image of the object detection result. The image is in JPEG format and encoded in Base64. |

        Each element in `detectedObjects` is an `object` with the following properties:

        | Name | Type | Description |
        |------|------|-------------|
        | `bbox` | `array` | The location of the object. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner of the bounding box, respectively. |
        | `categoryId` | `integer` | The ID of the object category. |
        | `score` | `number` | The score of the object. |

        An example of `result` is as follows:

        ```json
        {
          "detectedObjects": [
            {
              "bbox": [
                404.4967956542969,
                90.15770721435547,
                506.2465515136719,
                285.4187316894531
              ],
              "categoryId": 0,
              "score": 0.7418514490127563
            },
            {
              "bbox": [
                155.33145141601562,
                81.10954284667969,
                199.71136474609375,
                167.4235382080078
              ],
              "categoryId": 1,
              "score": 0.7328268885612488
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

API_URL = "http://localhost:8080/object-detection"
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
print("\nDetected objects:")
print(result["detectedObjects"])
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

    auto response = client.Post("/object-detection", headers, body, "application/json");
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

        auto detectedObjects = result["detectedObjects"];
        std::cout << "\nDetected objects:" << std::endl;
        for (const auto& obj : detectedObjects) {
            std::cout << obj << std::endl;
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
        String API_URL = "http://localhost:8080/object-detection";
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
                JsonNode detectedObjects = result.get("detectedObjects");

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println("Output image saved at " + outputImagePath);
                System.out.println("\nDetected objects: " + detectedObjects.toString());
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
    API_URL := "http://localhost:8080/object-detection"
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
            DetectedObjects []map[string]interface{} `json:"detectedObjects"`
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
    fmt.Println("\nDetected objects:")
    for _, obj := range respData.Result.DetectedObjects {
        fmt.Println(obj)
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
    static readonly string API_URL = "http://localhost:8080/object-detection";
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
        Console.WriteLine("\nDetected objects:");
        Console.WriteLine(jsonResponse["result"]["detectedObjects"].ToString());
    }
}
```

</details>

<details>
<summary>Node.js</summary>

```js
const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/object-detection'
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
    console.log("\nDetected objects:");
    console.log(result["detectedObjects"]);
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

$API_URL = "http://localhost:8080/object-detection";
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
echo "\nDetected objects:\n";
print_r($result["detectedObjects"]);

?>
```

</details>

</details>
<br/>

📱 **Edge Deployment**: Edge deployment is a method that places computing and data processing capabilities on user devices themselves, allowing devices to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/lite_deploy_en.md).
Choose the appropriate deployment method for your model pipeline based on your needs, and proceed with subsequent AI application integration.

## 4. Custom Development
If the default model weights provided by the General Small Object Detection Pipeline do not meet your requirements for accuracy or speed in your specific scenario, you can try to further fine-tune the existing model using **your own domain-specific or application-specific data** to improve the recognition performance of the small object detection pipeline in your scenario.

### 4.1 Model Fine-tuning
Since the General Small Object Detection Pipeline includes a small object detection module, if the performance of the pipeline does not meet expectations, you need to refer to the [Customization](../../../module_usage/tutorials/cv_modules/small_object_detection_en.md#iv-custom-development) section in the [Small Object Detection Module Tutorial](../../../module_usage/tutorials/cv_modules/small_object_detection_en.md) and use your private dataset to fine-tune the small object detection model.

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain local model weight files.

If you need to use the fine-tuned model weights, simply modify the pipeline configuration file by replacing the local path of the fine-tuned model weights to the corresponding location in the pipeline configuration file:

```python
......
Pipeline:
  model: PP-YOLOE_plus_SOD-L  # Can be modified to the local path of the fine-tuned model
  batch_size: 1
  device: "gpu:0"
......
```
Then, refer to the command line or Python script methods in the local experience section to load the modified pipeline configuration file.

## Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. **Simply modify the `--device` parameter** to seamlessly switch between different hardware.

For example, if you use an NVIDIA GPU for inference with the small object detection pipeline, the Python command would be:

```bash
paddlex --pipeline multilabel_classification --input small_object_detection.jpg --device gpu:0
``````
At this point, if you wish to switch the hardware to Ascend NPU, simply modify the `--device` in the Python command to `npu:0`:

```bash
paddlex --pipeline multilabel_classification --input small_object_detection.jpg --device npu:0
```

If you want to use the General Small Object Detection Pipeline on a wider range of hardware, please refer to the [PaddleX Multi-Device Usage Guide](../../../other_devices_support/multi_devices_use_guide_en.md).
