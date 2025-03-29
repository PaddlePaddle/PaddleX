---
comments: true
---

# 多目标跟踪产线使用教程

## 1. 多目标跟踪产线介绍
多目标跟踪(Multiple Object Tracking, MOT)是计算机视觉领域的一个重要研究方向，其目标是在视频序列中同时跟踪多个目标物体，并保持其身份的一致性。它在视频监控、自动驾驶、人机交互等领域有着广泛的应用。

多目标跟踪面临着许多挑战，包括目标遮挡、目标形变、光照变化、目标出现与消失等复杂情况。近年来，随着深度学习的发展，基于深度学习的MOT方法取得了显著进展，如DeepSORT、JDE、FairMOT等算法。现代MOT系统通常包含目标检测、特征提取、运动预测和数据关联等模块。评估MOT系统性能的主要指标包括MOTA(多目标跟踪准确度)、IDF1(识别F1分数)和MOTP(多目标跟踪精度)等。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/multiobject_tracking/mot.gif"/>

<b>通用</b><b>多目标跟踪</b><b>产线中包含了</b><b>联合检测嵌入</b><b>模块，如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>MOTA (%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>FairMOT-DLA-34_1088x608</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/FairMOT-DLA-34_1088x608_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FairMOT-DLA-34_1088x608_pretrained.pdparams">训练模型</a></td>
<td>75.0</td>
<td> - </td>
<td> - </td>
<td>77 M</td>
<td>FairMOT是一种基于锚点自由的实时多目标跟踪算法，采用联合检测和嵌入表示方法。它使用CenterNet作为检测器，并为每个目标生成像素级特征嵌入，有效解决了特征不对齐问题，在多个MOT基准测试中取得了优异的性能。
</td>
</tr>

</table>

**测试环境说明：**

- **性能测试环境**
  - **测试数据集**：<a href="https://motchallenge.net/data/MOT16/">MOT</a>测试集。

## 2. 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验多目标产线的效果，也可以在本地使用命令行或 Python 体验多目标跟踪产线的效果。

### 2.1 在线体验

暂不支持在线体验

### 2.2 本地体验
在本地使用多目标跟踪产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.2.1 命令行方式体验
一行命令即可快速体验多目标跟踪产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/multiobject_tracking_000.mp4)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline multiobject_tracking \
        --input multiobject_tracking_000.mp4 \
        --save_path ./output/ \
        --device gpu:0
```
相关参数和运行结果说明可以参考[2.2.2 Python脚本方式集成](#222-python脚本方式集成)中的参数说明和结果解释。

可视化结果保存至`save_path`，如下所示：

https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/videos/pipelines/multiobject_tracking/01.mp4

#### 2.2.2 Python脚本方式集成
通过上述命令行方式可快速体验查看效果，在项目中往往需要代码集成，您可以通过如下几行代码完成产线的快速推理：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="multiobject_tracking")

output = pipeline.predict("multiobject_tracking_000.mp4")
for res in output:
    res.print()
    res.save_to_video("./output/")
    res.save_to_json("./output/")
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）调用 `create_pipeline` 实例化产线对象：具体参数说明如下：
<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<tr>
<td><code>config</code></td>
<td>产线具体的配置信息（如果和<code>pipeline</code>同时设置，优先级高于<code>pipeline</code>，且要求产线名和<code>pipeline</code>一致）。</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<td><code>device</code></td>
<td>产线推理设备。支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理，仅当该产线支持高性能推理时可用。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

2）调用多目标跟踪产线对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>可选项</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>待预测视频数据，支持多种输入类型，必填</td>
<td><code>Python str|list</code></td>
<td>
<ul>
  <li><b>str</b>：视频文件本地路径：<code>/root/data/video.mp4</code>；<b>如URL链接</b>，如视频文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/multiobject_tracking_000.mp4">示例</a>；<b>如本地目录</b>，该目录下需包含待预测视频，如本地路径：<code>/root/data/</code></li>
  <li><b>List</b>：列表元素需为上述类型数据，如<code>[str, str]</code>，<code>[\"/root/data/video1.mp4\", \"/root/data/video2.avi\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线推理设备</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
  <li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
  <li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
  <li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
  <li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
  <li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
  <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

（3）对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为视频、保存为`json`文件的操作:

<table>
<thead>
<tr>
<th>方法</th>
<th>方法说明</th>
<th>参数</th>
<th>参数类型</th>
<th>参数说明</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">打印结果到终端</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>是否对输出内容进行使用 <code>JSON</code> 缩进格式化</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">将结果保存为json格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_video()</code></td>
<td>将结果保存为视频格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
</table>

- 调用 <code>print()</code> 方法会将如下结果打印到终端：

```bash
{'res': {
    "input_path": "mot17_demo.mp4",
    "result": [
        {
            "frame_id": 0,
            "track_res": [
                {
                   "cls_id": 0,
                   "label": "pedestrian",
                   "track_id": 1,
                   "score": 0.8757550716400146,
                   "coordinate": [ 119.05133819580078, 189.6024169921875, 359.66656494140625, 796.8548583984375]
                },
                ...
            ]
        }
        ...
    ]
}
```

- 输出结果参数含义如下：
   - `input_path`: `(str)` 待预测视频的输入路径

    - `result`: `(List[dict])` 预测结果，每个字典表示一帧图像的预测结果，每帧结果包含以下内容：
        - `frame_id`: 帧的编号
        - `track_res`: `(List[dict])` 目标框的预测结果，每个字典表示一个目标框的预测结果，包含以下内容：
            - `cls_id`: 类别编号
            - `label`: 类别名称
            - `track_id`: 目标框的跟踪ID
            - `score`: 跟踪的置信度
            - `coordinate`: 目标框的坐标，格式为`[xmin, ymin, xmax, ymax]`

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。
- 调用`save_to_video()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_video_basename}_res.{your_video_extension}`，如果指定为文件，则直接保存到该文件中。

* 此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">获取预测的 <code>json</code> 格式的结果</td>
</tr>
</table>

- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。

上述Python脚本集成方式默认使用 PaddleX 官方配置文件中的参数设置，若您需要自定义配置文件，可先执行如下命令获取官方配置文件，并保存在 `my_path` 中：

```bash
paddlex --get_pipeline_config multiobject_tracking --save_path ./my_path
```

若您获取了配置文件，即可对多目标跟踪产线各项配置进行自定义。只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为自定义产线配置文件路径即可。

例如，若您的自定义配置文件保存在 `./my_path/multiobject_tracking.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/multiobject_tracking.yaml")
output = pipeline.predict("multiobject_tracking_000.mp4")
for res in output:
    res.print()
    res.save_to_video("./output/")
    res.save_to_json("./output/")
```
<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改多目标跟踪产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。

## 3. 开发集成/部署

如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 <b>高性能推理</b>：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ <b>服务化部署</b>：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持多种产线服务化部署方案，详细的产线服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/serving.md)。

以下是基础服务化部署的API参考与多语言服务调用示例：

<details><summary>API参考</summary>
<p>对于服务提供的主要操作：</p>
<ul>
<li>HTTP请求方法为POST。</li>
<li>请求体和响应体均为JSON数据（JSON对象）。</li>
<li>当请求处理成功时，响应状态码为<code>200</code>，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。固定为<code>0</code>。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。固定为<code>"Success"</code>。</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>操作结果。</td>
</tr>
</tbody>
</table>
<ul>
<li>当请求处理未成功时，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。与响应状态码相同。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。</td>
</tr>
</tbody>
</table>
<p>服务提供的主要操作如下：</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>对视频进行目标检测。</p>
<p><code>POST /object-detection</code></p>
<ul>
<li>请求体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>服务器可访问的图像文件的URL或图像文件内容的Base64编码结果。</td>
<td>是</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td><code>number</code> | <code>object</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>threshold</code> 参数相关说明。</td>
<td>否</td>
</tr>
</tbody>
</table>
<ul>
<li>请求处理成功时，响应体的<code>result</code>具有如下属性：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>detectedObjects</code></td>
<td><code>array</code></td>
<td>目标的位置、类别等信息。</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code> | <code>null</code></td>
<td>目标检测结果图。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<p><code>detectedObjects</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>目标位置。数组中元素依次为边界框左上角x坐标、左上角y坐标、右下角x坐标以及右下角y坐标。</td>
</tr>
<tr>
<td><code>categoryId</code></td>
<td><code>integer</code></td>
<td>目标类别ID。</td>
</tr>
<tr>
<td><code>categoryName</code></td>
<td><code>string</code></td>
<td>目标类别名称。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>目标得分。</td>
</tr>
</tbody>
</table>
<p><code>result</code>示例如下：</p>
<pre><code class="language-json">{
"detectedObjects": [
{
"bbox": [
404.4967956542969,
90.15770721435547,
506.2465515136719,
285.4187316894531
],
"categoryId": 0,
"categoryName": "oranage",
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
"categoryName": "banana",
"score": 0.7328268885612488
}
],
"image": "xxxxxx"
}
</code></pre></details>
<details><summary>多语言调用服务示例</summary>
<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/object-detection" # 服务URL
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
print("\nDetected objects:")
print(result["detectedObjects"])
</code></pre></details>
<details><summary>C++</summary>
<pre><code class="language-cpp">#include &lt;iostream&gt;
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

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; "Error reading file." &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["image"] = encodedImage;
    std::string body = jsonObj.dump();

    // 调用API
    auto response = client.Post("/object-detection", headers, body, "application/json");
    // 处理接口返回数据
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse["result"];

        encodedImage = result["image"];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; "Output image saved at " &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; "Unable to open file for writing: " &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        }

        auto detectedObjects = result["detectedObjects"];
        std::cout &lt;&lt; "\nDetected objects:" &lt;&lt; std::endl;
        for (const auto&amp; obj : detectedObjects) {
            std::cout &lt;&lt; obj &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; "Failed to send HTTP request." &lt;&lt; std::endl;
        return 1;
    }

    return 0;
}
</code></pre></details>
<details><summary>Java</summary>
<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/object-detection"; // 服务URL
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
</code></pre></details>
<details><summary>Go</summary>
<pre><code class="language-go">package main

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
    client := &amp;http.Client{}
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
            DetectedObjects []map[string]interface{} `json:"detectedObjects"`
        } `json:"result"`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
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
</code></pre></details>
<details><summary>C#</summary>
<pre><code class="language-csharp">using System;
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
        Console.WriteLine("\nDetected objects:");
        Console.WriteLine(jsonResponse["result"]["detectedObjects"].ToString());
    }
}
</code></pre></details>
<details><summary>Node.js</summary>
<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/object-detection'
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
.then((response) =&gt; {
    // 处理接口返回数据
    const result = response.data["result"];
    const imageBuffer = Buffer.from(result["image"], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log("\nDetected objects:");
    console.log(result["detectedObjects"]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>
<details><summary>PHP</summary>
<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/object-detection"; // 服务URL
$image_path = "./demo.jpg";
$output_image_path = "./out.jpg";

// 对本地图像进行Base64编码
$image_data = base64_encode(file_get_contents($image_path));
$payload = array("image" =&gt; $image_data); // Base64编码的文件内容或者图像URL

// 调用API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// 处理接口返回数据
$result = json_decode($response, true)["result"];
file_put_contents($output_image_path, base64_decode($result["image"]));
echo "Output image saved at " . $output_image_path . "\n";
echo "\nDetected objects:\n";
print_r($result["detectedObjects"]);

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果多目标跟踪产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升多目标跟踪产线的在您的场景中的识别效果。

### 4.1 模型微调
由于多目标跟踪产线包含联合检测嵌入模块，如果模型产线的效果不及预期，那么您需要参考[联合检测嵌入模块开发教程](../../../module_usage/tutorials/cv_modules/joint_detection_embedding.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/joint_detection_embedding.md#四二次开发)章节，使用您的私有数据集对联合检测嵌入模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```yaml
pipeline_name: multiobject_tracking

SubModules:
  Detector:
    module_name: joint_detection_embedding
    model_name: FairMOT-DLA-34_1088x608
    model_dir: null #可修改为微调后模型的本地路径
    batch_size: 1
    img_size: null
    threshold: null
```
随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

## 5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU 和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改  `--device` 参数</b>即可完成不同硬件之间的无缝切换。

例如，使用昇腾 NPU 进行多目标跟踪产线快速推理：

```bash
paddlex --pipeline multiobject_tracking \
        --input multiobject_tracking_000.mp4 \
        --save_path ./output/ \
        --device npu:0
```

当然，您也可以在 Python 脚本中 `create_pipeline()` 时或者 `predict()` 时指定硬件设备。

若您想在更多种类的硬件上使用多目标跟踪产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
