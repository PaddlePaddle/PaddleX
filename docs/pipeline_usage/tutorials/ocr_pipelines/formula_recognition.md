简体中文 | [English](formula_recognition_en.md)

# 公式识别产线使用教程

## 1. 公式识别产线介绍

公式识别是一种自动从文档或图像中识别和提取LaTeX公式内容及其结构的技术，广泛应用于数学、物理、计算机科学等领域的文档编辑和数据分析。通过使用计算机视觉和机器学习算法，公式识别能够将复杂的数学公式信息转换为可编辑的LaTeX格式，方便用户进一步处理和分析数据。

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/01.jpg)

**通用****公式识别****产线中包含版面区域检测模块和公式识别模块**。

**如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型**。

<details>
   <summary> 👉模型列表详情</summary>

**版面区域检测模块模型：**

|模型名称|mAP（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小（M)|
|-|-|-|-|-|
|RT-DETR-H_layout_17cls|92.6|115.126|3827.25|470.2M|

**注：以上精度指标的评估集是 PaddleX 自建的版面区域检测数据集，包含 1w 张图片。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

**公式识别模块模型：**
|模型名称|BLEU score|normed edit distance|ExpRate （%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|
|-|-|-|-|-|-|-|
|LaTeX_OCR_rec|0.8821|0.0823|40.01|-|-|89.7 M|

**注：以上精度指标测量自 [LaTeX-OCR公式识别测试集](https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO)。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**


</details>

## 2. 快速开始
PaddleX 支持在本地使用命令行或 Python 体验公式识别产线的效果。

在本地使用公式识别产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

### 2.1 命令行方式体验
一行命令即可快速体验公式识别产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline formula_recognition --input general_formula_recognition.png --device gpu:0
```
参数说明：

```
--pipeline：产线名称，此处为公式识别产线
--input：待处理的输入图片的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```

在执行上述 Python 脚本时，加载的是默认的公式识别产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

<details>
   <summary> 👉点击展开</summary>

```
paddlex --get_pipeline_config formula_recognition
```
执行后，公式识别产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为 `./my_path` ）：

```
paddlex --get_pipeline_config formula_recognition --save_path ./my_path
```

获取产线配置文件后，可将 `--pipeline` 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 `./formula_recognition.yaml`，只需执行：

```bash
paddlex --pipeline ./formula_recognition.yaml --input general_formula_recognition.png --device gpu:0
```
其中，`--model`、`--device` 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。

</details>


运行后，得到的结果为：

<details>
   <summary> 👉点击展开</summary>

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
其中，dt_polys为检测到的公式区域坐标， rec_formula为检测到的公式。
</details>

可视化结果如下：

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/02.jpg)

可视化图片默认不进行保存，您可以通过 `--save_path` 自定义保存路径，随后所有结果将被保存在指定路径下。公式识别可视化需要单独配置环境，请您参考[2.3 公式识别产线可视化](#23-公式识别产线可视化) 对LaTeX渲染引擎进行安装。

### 2.2 Python脚本方式集成
几行代码即可完成产线的快速推理，以公式识别产线为例：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="formula_recognition")

output = pipeline.predict("general_formula_recognition.png")
for res in output:
    res.print()
```

> ❗ Python脚本运行得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

（1）实例化 `create_pipeline` 实例化 公式识别产线对象：具体参数说明如下：

|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|`pipeline`|产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。|`str`|无|
|`device`|产线模型推理设备。支持：“gpu”，“cpu”。|`str`|`gpu`|
|`use_hpip`|是否启用高性能推理，仅当该产线支持高性能推理时可用。|`bool`|`False`|

（2）调用公式识别产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

| 参数类型      | 参数说明                                                                                                  |
|---------------|-----------------------------------------------------------------------------------------------------------|
| Python Var    | 支持直接传入Python变量，如numpy.ndarray表示的图像数据。                                               |
| str         | 支持传入待预测数据文件路径，如图像文件的本地路径：`/root/data/img.jpg`。                                   |
| str           | 支持传入待预测数据文件URL，如图像文件的网络URL：[示例](https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png)。|
| str           | 支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：`/root/data/`。                               |
| dict          | 支持传入字典类型，字典的key需与具体任务对应，如图像分类任务对应\"img\"，字典的val支持上述类型数据，例如：`{\"img\": \"/root/data1\"}`。|
| list          | 支持传入列表，列表元素需为上述类型数据，如`[numpy.ndarray, numpy.ndarray]，[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]`，`[\"/root/data1\", \"/root/data2\"]`，`[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]`。|

（3）调用`predict`方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以batch为单位对数据进行预测，因此预测结果为list形式表示的一组预测结果。

（4）对预测结果进行处理：每个样本的预测结果均为`dict`类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

| 方法         | 说明                        | 方法参数                                                                                               |
|--------------|-----------------------------|--------------------------------------------------------------------------------------------------------|
| print        | 打印结果到终端              | `- format_json`：bool类型，是否对输出内容进行使用json缩进格式化，默认为True；<br>`- indent`：int类型，json格式化设置，仅当format_json为True时有效，默认为4；<br>`- ensure_ascii`：bool类型，json格式化设置，仅当format_json为True时有效，默认为False； |
| save_to_json | 将结果保存为json格式的文件   | `- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；<br>`- indent`：int类型，json格式化设置，默认为4；<br>`- ensure_ascii`：bool类型，json格式化设置，默认为False； |
| save_to_img  | 将结果保存为图像格式的文件  | `- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致； |

若您获取了配置文件，即可对公式识别产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 `./my_path/formula_recognition.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/formula_recognition.yaml")
output = pipeline.predict("general_formula_recognition.png")
for res in output:
    res.print()
```
### 2.3 公式识别产线可视化
如果您需要对公式识别产线进行可视化，需要运行如下命令来对LaTeX渲染环境进行安装：
```python
apt-get install sudo
sudo apt-get update
sudo apt-get install texlive
sudo apt-get install texlive-latex-base
sudo apt-get install texlive-latex-extra
python -m pip install PyMuPDF==1.24.12
```
之后，使用 `save_to_img` 方法对可视化图片进行保存。具体命令如下：
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="formula_recognition")

output = pipeline.predict("general_formula_recognition.png")
for res in output:
    res.print()
    res.save_to_img("./output/")
```
**备注**： 由于公式识别可视化过程中需要对每张公式图片进行渲染，因此耗时较长，请您耐心等待。

## 3. 开发集成/部署
如果公式识别产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将公式识别产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

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

    获取图像公式识别结果。

    `POST /formula-recognition`

    - 请求体的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`image`|`string`|服务可访问的图像文件的URL或图像文件内容的Base64编码结果。|是|
        |`inferenceParams`|`object`|推理参数。|否|

        `inferenceParams`的属性如下：

        |名称|类型|含义|是否必填|
        |-|-|-|-|
        |`maxLongSide`|`integer`|推理时，若文本检测模型的输入图像较长边的长度大于`maxLongSide`，则将对图像进行缩放，使其较长边的长度等于`maxLongSide`。|否|

    - 请求处理成功时，响应体的`result`具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`formulas`|`array`|公式位置和内容。|
        |`image`|`string`|公式识别结果图，其中标注检测到的公式位置。图像为JPEG格式，使用Base64编码。|

        `formulas`中的每个元素为一个`object`，具有如下属性：

        |名称|类型|含义|
        |-|-|-|
        |`poly`|`array`|公式位置。数组中元素依次为包围文本的多边形的顶点坐标。|
        |`latex`|`string`|公式内容。|

        `result`示例如下：

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
<summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>

```python
import base64
import requests

API_URL = "http://localhost:8080/formula-recognition" # 服务URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64编码的文件内容或者图像URL

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
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

    // 对本地图像进行Base64编码
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

    // 调用API
    auto response = client.Post("/formula-recognition", headers, body, "application/json");
    // 处理接口返回数据
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
        String API_URL = "http://localhost:8080/formula-recognition"; // 服务URL
        String imagePath = "./demo.jpg"; // 本地图像
        String outputImagePath = "./out.jpg"; // 输出图像

        // 对本地图像进行Base64编码
        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put("image", imageData); // Base64编码的文件内容或者图像URL

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

    // 对本地图像进行Base64编码
    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println("Error reading image file:", err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{"image": imageData} // Base64编码的文件内容或者图像URL
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

    // 处理接口返回数据
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

        // 对本地图像进行Base64编码
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { "image", image_data } }; // Base64编码的文件内容或者图像URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        // 调用API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // 处理接口返回数据
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
    'image': encodeImageToBase64(imagePath)  // Base64编码的文件内容或者图像URL
  })
};

// 对本地图像进行Base64编码
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// 调用API
axios.request(config)
.then((response) => {
    // 处理接口返回数据
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

$API_URL = "http://localhost:8080/formula-recognition"; // 服务URL
$image_path = "./demo.jpg";
$output_image_path = "./out.jpg";

// 对本地图像进行Base64编码
$image_data = base64_encode(file_get_contents($image_path));
$payload = array("image" => $image_data); // Base64编码的文件内容或者图像URL

// 调用API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// 处理接口返回数据
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

📱 **端侧部署**：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。


## 4. 二次开发
如果公式识别产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用**您自己拥有的特定领域或应用场景的数据**对现有模型进行进一步的**微调**，以提升公式识别产线的在您的场景中的识别效果。

### 4.1 模型微调
由于公式识别产线包含两个模块（版面区域检测模块和公式识别），模型产线的效果不及预期可能来自于其中任何一个模块。

您可以对识别效果差的图片进行分析，如果在分析过程中发现有较多的公式未被检测出来（即公式漏检现象），那么可能是版面区域检测模型存在不足，您需要参考[版面区域检测模块开发教程](../../../module_usage/tutorials/ocr_modules/layout_detection.md)中的[二次开发](../../../module_usage/tutorials/ocr_modules/layout_detection.md#四二次开发)章节，使用您的私有数据集对版面区域检测模型进行微调；如果在已检测到的公式中出现较多的识别错误（即识别出的公式内容与实际公式内容不符），这表明公式识别模型需要进一步改进，您需要参考[公式识别模块开发教程](../../../module_usage/tutorials/ocr_modules/formula_recognition.md)中的中的[二次开发](../../../module_usage/tutorials/ocr_modules/formula_recognition.md#四二次开发)章节,对公式识别模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```bash
......
Pipeline:
  layout_model: RT-DETR-H_layout_17cls #可修改为微调后版面区域检测模型的本地路径
  formula_rec_model: LaTeX_OCR_rec #可修改为微调后公式识别模型的本地路径
  formula_rec_batch_size: 5
  device: "gpu:0"
......
```
随后， 参考[2. 快速开始](#2-快速开始)中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，**仅需修改 `--device`参数**即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行公式识别产线的推理，使用的 Python 命令为：

```bash
paddlex --pipeline formula_recognition --input general_formula_recognition.png --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需对 Python 命令中的 `--device` 进行修改即可：

```bash
paddlex --pipeline formula_recognition --input general_formula_recognition.png --device npu:0
```
若您想在更多种类的硬件上使用公式识别产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
