简体中文 | [English](time_series_classification_en.md)

# 时序分类产线使用教程

## 1. 通用时序分类产线介绍
时序分类是一种将时间序列数据归类到预定义类别的技术，广泛应用于行为识别、金融趋势分析等领域。它通过分析随时间变化的特征，识别出不同的模式或事件，例如将一段语音信号分类为“问候”或“请求”，或将股票价格走势划分为“上涨”或“下跌”。时序分类通常使用机器学习和深度学习模型，能够有效捕捉时间依赖性和变化规律，以便为数据提供准确的分类标签。这项技术在智能监控、市场预测等应用中起着关键作用。

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/time_series/01.png)


**通用****时序分类****产线中包含了****时序分类****模块**。

<details>
   <summary> 👉模型列表详情</summary>

|模型名称|acc(%)|模型存储大小（M)|
|-|-|-|
|TimesNet_cls|87.5|792K|

**注：以上精度指标测量自 [UWaveGestureLibrary](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TEST.csv) 数据集。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

</details>

## 2. 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验通用时序分类产线的效果，也可以在本地使用命令行或 Python 体验通用时序分类产线的效果。

### 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/105707/webUI?source=appCenter)通用时序分类产线的效果，用官方提供的 demo 进行识别，例如：

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/time_series/02.png)

如果您对产线运行的效果满意，可以直接对产线进行集成部署，如果不满意，您也可以利用私有数据**对产线中的模型进行在线微调**。

注：由于时序数据和场景紧密相关，时序任务的在线体验官方内置模型仅是在一个特定场景下的模型方案，并非通用方案，不适用其他场景，因此体验方式不支持使用任意的文件来体验官方模型方案效果。但是，在完成自己场景数据下的模型训练之后，可以选择自己训练的模型方案，并使用对应场景的数据进行在线体验。

### 2.2 本地体验
在本地使用通用时序分类产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.2.1 命令行方式体验
一行命令即可快速体验时序分类产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline ts_cls --input ts_cls.csv --device gpu:0
```
参数说明：

```
--pipeline：产线名称，此处为时序分类产线
--input：待处理的输入序列的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```

在执行上述 Python 脚本时，加载的是默认的时序分类产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

<details>
   <summary> 👉点击展开</summary>

```
paddlex --get_pipeline_yaml ts_cls
```
执行后，时序分类产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为* ./my_path*）：

```
paddlex --get_pipeline_config ts_cls --save_path ./my_path
```

获取产线配置文件后，可将 `--pipeline` 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 `./ts_cls.yaml`，只需执行：

```bash
paddlex --pipeline ./ts_cls.yaml --input ts_cls.csv --device gpu:0
```
其中，`--model`、`--device` 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。

</details>

运行后，得到的结果为：

```
{'input_path': 'ts_cls.csv', 'classification':         classid     score
sample
0             0  0.617688}
```

#### 2.2.2 Python脚本方式集成
几行代码即可完成产线的快速推理，以通用时序分类产线为例：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="ts_cls")

output = pipeline.predict("ts_cls.csv")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_csv("./output/") ## 保存csv格式结果
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

（1）实例化 `create_pipeline` 实例化产线对象：具体参数说明如下：

|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|`pipeline`|产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。|`str`|无|
|`device`|产线模型推理设备。支持：“gpu”，“cpu”。|`str`|`gpu`|
|`use_hpip`|是否启用高性能推理，仅当该产线支持高性能推理时可用。|`bool`|`False`|

（2）调用产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

| 参数类型      | 参数说明                                                                                                  |
|---------------|-----------------------------------------------------------------------------------------------------------|
| Python Var    | 支持直接传入Python变量，如numpy.ndarray表示的图像数据。                                               |
| str         | 支持传入待预测数据文件路径，如图像文件的本地路径：`/root/data/img.jpg`。                                   |
| str           | 支持传入待预测数据文件URL，如图像文件的网络URL：[示例](https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv)。|
| str           | 支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：`/root/data/`。                               |
| dict          | 支持传入字典类型，字典的key需与具体任务对应，如图像分类任务对应\"img\"，字典的val支持上述类型数据，例如：`{\"img\": \"/root/data1\"}`。|
| list          | 支持传入列表，列表元素需为上述类型数据，如`[numpy.ndarray, numpy.ndarray]，[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]`，`[\"/root/data1\", \"/root/data2\"]`，`[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]`。|

（3）调用`predict`方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以batch为单位对数据进行预测，因此预测结果为list形式表示的一组预测结果。

（4）对预测结果进行处理：每个样本的预测结果均为`dict`类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

|方法|说明|方法参数|
|-|-|-|
|save_to_csv|将结果保存为csv格式的文件|`- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；|
|save_to_html|将结果保存为html格式的文件|`- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；|
|save_to_xlsx|将结果保存为表格格式的文件|`- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；|

若您获取了配置文件，即可对时序分类产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 `./my_path/ts_cls.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/ts_forecast.yaml")
output = pipeline.predict("ts_cls.csv")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_csv("./output/") ## 保存csv格式结果
```

## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 **高性能推理**：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ **服务化部署**：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/service_deploy.md)。

下面是API参考和多语言服务调用示例：

<details>
<summary>API参考</summary>

对于服务提供的所有操作：

- 响应体以及POST请求的请求体均为JSON数据（JSON对象）。
- 当请求处理成功时，响应状态码为`200`，响应体的属性如下：

    |名称|类型|含义|
    |-|-|-|
    |`errorCode`|`integer`|错误码。固定为`0`。|
    |`errorMsg`|`string`|错误说明。固定为`"Success"`。|

    响应体还可能有`result`属性，类型为`object`，其中存储操作结果信息。

- 当请求处理未成功时，响应体的属性如下：

    |名称|类型|含义|
    |-|-|-|
    |`errorCode`|`integer`|错误码。与响应状态码相同。|
    |`errorMsg`|`string`|错误说明。|

服务提供的操作如下：

- **`infer`**

    对时序数据进行分类。

    `POST /time-series-classification`

    - 请求体的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`csv`|`string`|服务可访问的CSV文件的URL或CSV文件内容的Base64编码结果。CSV文件需要使用UTF-8编码。|是|

    - 请求处理成功时，响应体的`result`具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`label`|`string`|类别标签。|
        |`score`|`number`|类别得分。|

        `result`示例如下：

        ```json
        {
          "label": "running",
          "score": 0.97
        }
        ```

</details>

<details>
<summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>

```python
import base64
import requests

API_URL = "http://localhost:8080/time-series-classification" # 服务URL
csv_path = "./test.csv"

# 对本地图像进行Base64编码
with open(csv_path, "rb") as file:
    csv_bytes = file.read()
    csv_data = base64.b64encode(csv_bytes).decode("ascii")

payload = {"csv": csv_data}

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()["result"]
print(f"label: {result['label']}, score: {result['score']}")
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
    const std::string csvPath = "./test.csv";

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    // 进行Base64编码
    std::ifstream file(csvPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    std::string encodedCsv = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["csv"] = encodedCsv;
    std::string body = jsonObj.dump();

    // 调用API
    auto response = client.Post("/time-series-classification", headers, body, "application/json");
    // 处理接口返回数据
    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];
        std::cout << "label: " << result["label"] << ", score: " << result["score"] << std::endl;
    } else {
        std::cout << "Failed to send HTTP request." << std::endl;
        std::cout << response->body << std::endl;
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
        String API_URL = "http://localhost:8080/time-series-classification";
        String csvPath = "./test.csv";

        // 对本地csv进行Base64编码
        File file = new File(csvPath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String csvData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("csv", csvData);

        // 创建 OkHttpClient 实例
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get("application/json; charset=utf-8");
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // 调用API并处理接口返回数据
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get("result");
                System.out.println("label: " + result.get("label").asText() + ", score: " + result.get("score").asText());
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
    API_URL := "http://localhost:8080/time-series-classification"
    csvPath := "./test.csv";

    // 读取csv文件并进行Base64编码
    csvBytes, err := ioutil.ReadFile(csvPath)
    if err != nil {
        fmt.Println("Error reading csv file:", err)
        return
    }
    csvData := base64.StdEncoding.EncodeToString(csvBytes)

    payload := map[string]string{"csv": csvData} // Base64编码的文件内容
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println("Error marshaling payload:", err)
        return
    }

    // 调用API
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

    // 处理返回数据
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }
    type Response struct {
        Result struct {
            Label string `json:"label"`
            Score string `json:"score"`
        } `json:"result"`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &respData)
    if err != nil {
        fmt.Println("Error unmarshaling response body:", err)
        return
    }

    fmt.Printf("label: %s, score: %s\n", respData.Result.Label, respData.Result.Score)
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
    static readonly string API_URL = "http://localhost:8080/time-series-classification";
    static readonly string csvPath = "./test.csv";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // 对本地csv文件进行Base64编码
        byte[] csveBytes = File.ReadAllBytes(csvPath);
        string csvData = Convert.ToBase64String(csveBytes);

        var payload = new JObject{ { "csv", csvData } }; // Base64编码的文件内容
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        // 调用API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // 处理接口返回数据
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string label = jsonResponse["result"]["label"].ToString();
        string score = jsonResponse["result"]["score"].ToString();
        Console.WriteLine($"label: {label}, score: {score}");
    }
}
```

</details>

<details>
<summary>Node.js</summary>

```js
const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/time-series-classification'
const csvPath = "./test.csv";

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'csv': encodeFileToBase64(csvPath)  // Base64编码的文件内容
  })
};

// 读取csv文件并转换为Base64
function encodeFileToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

axios.request(config)
.then((response) => {
    const result = response.data["result"];
    console.log(`label: ${result["label"]}, score: ${result["score"]}`);
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

$API_URL = "http://localhost:8080/time-series-classification"; // 服务URL
$csv_path = "./test.csv";

// 对本地csv文件进行Base64编码
$csv_data = base64_encode(file_get_contents($csv_path));
$payload = array("csv" => $csv_data); // Base64编码的文件内容

// 调用API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// 处理接口返回数据
$result = json_decode($response, true)["result"];
echo "label: " . $result["label"] . ", score: " . $result["score"];

?>
```

</details>
</details>
<br/>

📱 **端侧部署**：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/lite_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。


## 4. 二次开发
如果通用时序分类产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用**您自己拥有的特定领域或应用场景的数据**对现有模型进行进一步的**微调**，以提升通用时序分类产线的在您的场景中的识别效果。

### 4.1 模型微调
由于通用时序分类产线包含时序分类模块，如果模型产线的效果不及预期，那么您需要参考[时序分类模块开发教程](../../../module_usage/tutorials/time_series_modules/time_series_classification.md)中的[二次开发](../../../module_usage/tutorials/time_series_modules/time_series_classification.md#四二次开发)章节，使用您的私有数据集对时序分类模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```python
......
Pipeline:
  model: TimesNet_cls  #可修改为微调后模型的本地路径
  device: "gpu"
  batch_size: 1
......
```
随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，**仅需修改 `--device` 参数**即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行时序分类产线的推理，使用的 Python 命令为：

```bash
paddlex --pipeline ts_cls --input ts_cls.csv --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需对 Python 命令中的 `--device` 进行修改即可：

```bash
paddlex --pipeline ts_cls --input ts_cls.csv --device npu:0
```
若您想在更多种类的硬件上使用通用时序分类产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
