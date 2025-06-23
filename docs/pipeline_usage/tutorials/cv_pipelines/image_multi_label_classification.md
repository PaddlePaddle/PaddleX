---
comments: true
---

# 通用图像多标签分类产线使用教程

## 1. 通用图像多标签分类产线介绍
图像多标签分类是一种将一张图像同时分配到多个相关类别的技术，广泛应用于图像标注、内容推荐和社交媒体分析等领域。它能够识别图像中存在的多个物体或特征，例如一张图片中同时包含“狗”和“户外”这两个标签。通过使用深度学习模型，图像多标签分类能够自动提取图像特征并进行准确分类，以便为用户提供更加全面的信息。这项技术在智能搜索引擎和自动内容生成等应用中具有重要意义。本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_multi_label_classification/01.png">

<b>通用图像多标签分类产线中包含了图像多标签分类模块，如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。

<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP（%）</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIP_vit_base_patch16_448_ML</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/CLIP_vit_base_patch16_448_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_448_ML_pretrained.pdparams">训练模型</a></td>
<td>89.15</td>
<td>325.6 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B0_ML</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNetV2-B0_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_ML_pretrained.pdparams">训练模型</a></td>
<td>80.98</td>
<td>39.6</td>
</tr>
<tr>
<td>PP-HGNetV2-B4_ML</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNetV2-B4_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_ML_pretrained.pdparams">训练模型</a></td>
<td>87.96</td>
<td>88.5</td>
</tr>
<tr>
<td>PP-HGNetV2-B6_ML</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-HGNetV2-B6_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_ML_pretrained.pdparams">训练模型</a></td>
<td>91.25</td>
<td>286.5</td>
</tr>
<tr>
<td>PP-LCNet_x1_0_ML</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_ML_pretrained.pdparams">训练模型</a></td>
<td>77.96</td>
<td>29.4</td>
</tr>
<tr>
<td>ResNet50_ML</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet50_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_ML_pretrained.pdparams">训练模型</a></td>
<td>83.50</td>
<td>108.9</td>
</tr>
</tbody>
</table>

<strong>测试环境说明:</strong>

  <ul>
      <li><b>性能测试环境</b>
          <ul>
            <li><strong>测试数据集：
             </strong>
               <a href="https://cocodataset.org/#home">COCO2017</a><b> 的多标签分类任务</b>
             </li>
              <li><strong>硬件配置：</strong>
                  <ul>
                      <li>GPU：NVIDIA Tesla T4</li>
                      <li>CPU：Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>其他环境：Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
                  </ul>
              </li>
          </ul>
      </li>
      <li><b>推理模式说明</b></li>
  </ul>

<table border="1">
    <thead>
        <tr>
            <th>模式</th>
            <th>GPU配置</th>
            <th>CPU配置</th>
            <th>加速技术组合</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>常规模式</td>
            <td>FP32精度 / 无TRT加速</td>
            <td>FP32精度 / 8线程</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>高性能模式</td>
            <td>选择先验精度类型和加速策略的最优组合</td>
            <td>FP32精度 / 8线程</td>
            <td>选择先验最优后端（Paddle/OpenVINO/TRT等）</td>
        </tr>
    </tbody>
</table>


## 2. 快速开始
PaddleX 所提供的模型产线均可以快速体验效果，你可以在星河社区线体验通用图像多标签分类产线的效果，也可以在本地使用命令行或 Python 体验通用图像多标签分类产线的效果。

### 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/387974/webUI?source=appCenter)通用图像多标签分类产线的效果，用官方提供的 Demo 图片进行识别，例如：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_multi_label_classification/multi_label_cls.png"/>

如果您对产线运行的效果满意，可以直接进行集成部署。您可以选择从云端下载部署包，也可以参考[2.2节本地体验](#22-本地体验)中的方法进行本地部署。如果对效果不满意，您可以利用私有数据<b>对产线中的模型进行微调训练</b>。如果您具备本地训练的硬件资源，可以直接在本地开展训练；如果没有，星河零代码平台提供了一键式训练服务，无需编写代码，只需上传数据后，即可一键启动训练任务。

### 2.2 本地体验
❗ 在本地使用通用图像多标签分类产线前，请确保您已经按照[PaddleX安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。如果您希望选择性安装依赖，请参考安装教程中的相关说明。该产线对应的依赖分组为 `cv`。

#### 2.2.1 命令行方式体验
一行命令即可快速体验图像多标签分类产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline image_multilabel_classification --input general_image_classification_001.jpg --device gpu:0
```
相关的参数说明可以参考[2.2.2 Python脚本方式集成](#222-python脚本方式集成)中的参数说明。支持同时指定多个设备以进行并行推理，详情请参考 [产线并行推理](../../instructions/parallel_inference.md#指定多个推理设备)。

运行后，会将结果打印到终端上，结果如下：

```bash
{'res': {'input_path': 'test_imgs/general_image_classification_001.jpg', 'page_index': None, 'class_ids': array([21]), 'scores': array([0.99962]), 'label_names': ['bear']}}
```

运行结果参数说明可以参考[2.2.2 Python脚本方式集成](#222-python脚本方式集成)中的结果解释。

可视化结果保存在`save_path`下，可视化结果如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/image_multi_label_classification/02.png">

#### 2.2.1 命令行方式体验
* 上述命令行是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="image_multilabel_classification")

output = pipeline.predict("general_image_classification_001.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

（1）通过 `create_pipeline()` 实例化通用图像多标签分类产线对象，具体参数说明如下：

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
<td>None</td>
</tr>
<tr>
<td><code>config</code></td>
<td>产线具体的配置信息（如果和<code>pipeline</code>同时设置，优先级高于<code>pipeline</code>，且要求产线名和<code>pipeline</code>一致）。</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线推理设备。支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。支持同时指定多个设备以进行并行推理，详情请参考产线并行推理文档。</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理插件。如果为 <code>None</code>，则使用配置文件或 <code>config</code> 中的配置。</td>
<td><code>bool</code> | <code>None</code></td>
<td>无</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>高性能推理配置</td>
<td><code>dict</code> | <code>None</code></td>
<td>无</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>
（2）调用通用图像多标签分类产线对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：

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
<td>待预测数据，支持多种输入类型，必填</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
  <li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
  <li><b>str</b>：如图像文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code></li>
  <li><b>List</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>多标签分类阈值</td>
<td><code>float | dict | list｜ None</code></td>
<td>
<ul>
  <li><b>float</b>：表示多标签分类统一阈值；</li>
  <li><b>list</b>：如<code>[0.5, 0.45, 0.48, 0.4]</code>，表示按照<code>label list</code>顺序的不同类别阈值；</code>；</li>
  <li><b>dict</b>：字典的key 为 <code>default</code> 和 <code>int</code> 类型，val 为 <code>float</code> 类型阈值，如<code>{"default": 0.5, 0: 0.45, 2: 0.48, 7: 0.4}</code>，<code>default</code> 表示多标签分类统一阈值，其他 <code>int</code> 类型表示对 cls_id 为0的类别应用阈值 0.45、cls_id 为 1 的类别应用阈值 0.48、cls_id 为 7 的类别应用阈值 0.4；</li>
  <li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>0.7</code>；</li>
</ul>
</td>
<td><code>0.5</code></td>
</tr>
</table>
3）对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为`json`文件的操作:

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
<td><code>save_to_img()</code></td>
<td>将结果保存为图像格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
</table>

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：

    - `input_path`: `(str)` 待预测图像的输入路径。
    - `page_index`: `(Union[int, None])` 如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`。
    - `class_ids`: `(List[numpy.ndarray])` 表示预测结果的类别id。
    - `scores`: `(List[numpy.ndarray])` 表示预测结果的置信度。
    - `label_names`: `(List[str])` 表示预测结果的类别名称。

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)

* 此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">获取预测的 <code>json</code> 格式的结果</td>
</tr>
<tr>
<td rowspan = "2"><code>img</code></td>
<td rowspan = "2">获取格式为 <code>dict</code> 的可视化图像</td>
</tr>
</table>

- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。
- `img` 属性返回的预测结果是一个字典类型的数据。其中，键为 `res` 对应的值是`Image.Image` 对象：一个用于显示属性识别结果的可视化图像。

此外，您可以获取通用图像多标签分类产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config image_multilabel_classification --save_path ./my_path
```

若您获取了配置文件，即可对通用图像多标签分类产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。示例如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/image_multilabel_classification.yaml")

output = pipeline.predict(
    input="./general_image_classification_001.jpg",
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")

```

<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改通用图像多标签分类产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。同时，CLI 预测也支持传入配置文件，`--pipeline` 指定配置文件的路径即可。

## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

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
<p>对图像进行分类。</p>
<p><code>POST /multilabel-image-classification</code></p>
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
<td><code>number</code> | <code>array</code> | <code>object</code> | <code>null</code></td>
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
<td><code>categories</code></td>
<td><code>array</code></td>
<td>图像类别信息。</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code> | <code>null</code></td>
<td>图像分类结果图。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<p><code>categories</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>id</code></td>
<td><code>integer</code></td>
<td>类别ID。</td>
</tr>
<tr>
<td><code>name</code></td>
<td><code>string</code></td>
<td>类别名称。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>类别得分。</td>
</tr>
</tbody>
</table>
<p><code>result</code>示例如下：</p>
<pre><code class="language-json">{
&quot;categories&quot;: [
{
&quot;id&quot;: 5,
&quot;name&quot;: &quot;兔子&quot;,
&quot;score&quot;: 0.93
}
],
&quot;image&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/multilabel-image-classification&quot; # 服务URL
image_path = &quot;./demo.jpg&quot;
output_image_path = &quot;./out.jpg&quot;

# 对本地图像进行Base64编码
with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}  # Base64编码的文件内容或者图像URL

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
print(&quot;\nCategories:&quot;)
print(result[&quot;categories&quot;])
</code></pre></details>
<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // https://github.com/Huiyicc/cpp-httplib
#include &quot;nlohmann/json.hpp&quot; // https://github.com/nlohmann/json
#include &quot;base64.hpp&quot; // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string imagePath = &quot;./demo.jpg&quot;;
    const std::string outputImagePath = &quot;./out.jpg&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

    // 对本地图像进行Base64编码
    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; &quot;Error reading file.&quot; &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj[&quot;image&quot;] = encodedImage;
    std::string body = jsonObj.dump();

    // 调用API
    auto response = client.Post(&quot;/multilabel-image-classification&quot;, headers, body, &quot;application/json&quot;);
    // 处理接口返回数据
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedImage = result[&quot;image&quot;];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; &quot;Output image saved at &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        }

        auto categories = result[&quot;categories&quot;];
        std::cout &lt;&lt; &quot;\nCategories:&quot; &lt;&lt; std::endl;
        for (const auto&amp; category : categories) {
            std::cout &lt;&lt; category &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; &quot;Failed to send HTTP request.&quot; &lt;&lt; std::endl;
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
        String API_URL = &quot;http://localhost:8080/multilabel-image-classification&quot;; // 服务URL
        String imagePath = &quot;./demo.jpg&quot;; // 本地图像
        String outputImagePath = &quot;./out.jpg&quot;; // 输出图像

        // 对本地图像进行Base64编码
        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;image&quot;, imageData); // Base64编码的文件内容或者图像URL

        // 创建 OkHttpClient 实例
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get(&quot;application/json; charset=utf-8&quot;);
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
                JsonNode result = resultNode.get(&quot;result&quot;);
                String base64Image = result.get(&quot;image&quot;).asText();
                JsonNode categories = result.get(&quot;categories&quot;);

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println(&quot;Output image saved at &quot; + outputImagePath);
                System.out.println(&quot;\nCategories: &quot; + categories.toString());
            } else {
                System.err.println(&quot;Request failed with code: &quot; + response.code());
            }
        }
    }
}
</code></pre></details>

<details><summary>Go</summary>

<pre><code class="language-go">package main

import (
    &quot;bytes&quot;
    &quot;encoding/base64&quot;
    &quot;encoding/json&quot;
    &quot;fmt&quot;
    &quot;io/ioutil&quot;
    &quot;net/http&quot;
)

func main() {
    API_URL := &quot;http://localhost:8080/multilabel-image-classification&quot;
    imagePath := &quot;./demo.jpg&quot;
    outputImagePath := &quot;./out.jpg&quot;

    // 对本地图像进行Base64编码
    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println(&quot;Error reading image file:&quot;, err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{&quot;image&quot;: imageData} // Base64编码的文件内容或者图像URL
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println(&quot;Error marshaling payload:&quot;, err)
        return
    }

    // 调用API
    client := &amp;http.Client{}
    req, err := http.NewRequest(&quot;POST&quot;, API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println(&quot;Error creating request:&quot;, err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println(&quot;Error sending request:&quot;, err)
        return
    }
    defer res.Body.Close()

    // 处理接口返回数据
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println(&quot;Error reading response body:&quot;, err)
        return
    }
    type Response struct {
        Result struct {
            Image      string   `json:&quot;image&quot;`
            Categories []map[string]interface{} `json:&quot;categories&quot;`
        } `json:&quot;result&quot;`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println(&quot;Error unmarshaling response body:&quot;, err)
        return
    }

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println(&quot;Error decoding base64 image data:&quot;, err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing image to file:&quot;, err)
        return
    }
    fmt.Printf(&quot;Image saved at %s.jpg\n&quot;, outputImagePath)
    fmt.Println(&quot;\nCategories:&quot;)
    for _, category := range respData.Result.Categories {
        fmt.Println(category)
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
    static readonly string API_URL = &quot;http://localhost:8080/multilabel-image-classification&quot;;
    static readonly string imagePath = &quot;./demo.jpg&quot;;
    static readonly string outputImagePath = &quot;./out.jpg&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // 对本地图像进行Base64编码
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { &quot;image&quot;, image_data } }; // Base64编码的文件内容或者图像URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        // 调用API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // 处理接口返回数据
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse[&quot;result&quot;][&quot;image&quot;].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($&quot;Output image saved at {outputImagePath}&quot;);
        Console.WriteLine(&quot;\nCategories:&quot;);
        Console.WriteLine(jsonResponse[&quot;result&quot;][&quot;categories&quot;].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/multilabel-image-classification'
const imagePath = './demo.jpg'
const outputImagePath = &quot;./out.jpg&quot;;

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
    const result = response.data[&quot;result&quot;];
    const imageBuffer = Buffer.from(result[&quot;image&quot;], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log(&quot;\nCategories:&quot;);
    console.log(result[&quot;categories&quot;]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>
<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/multilabel-image-classification&quot;; // 服务URL
$image_path = &quot;./demo.jpg&quot;;
$output_image_path = &quot;./out.jpg&quot;;

// 对本地图像进行Base64编码
$image_data = base64_encode(file_get_contents($image_path));
$payload = array(&quot;image&quot; =&gt; $image_data); // Base64编码的文件内容或者图像URL

// 调用API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// 处理接口返回数据
$result = json_decode($response, true)[&quot;result&quot;];
file_put_contents($output_image_path, base64_decode($result[&quot;image&quot;]));
echo &quot;Output image saved at &quot; . $output_image_path . &quot;\n&quot;;
echo &quot;\nCategories:\n&quot;;
print_r($result[&quot;categories&quot;]);
?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/on_device_deployment.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果通用图像多标签分类产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升通用图像多标签分类产线的在您的场景中的识别效果。

### 4.1 模型微调

由于通用图像多标签分类产线包含图像多标签分类模块，如果模型产线的效果不及预期，需要参考以下表格中对应的微调教程链接进行模型微调。

<table>
  <thead>
    <tr>
      <th>情形</th>
      <th>微调模块</th>
      <th>微调参考链接</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>多标签分类效果不准</td>
      <td>多标签分类模块</td>
      <td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/image_multilabel_classification.html">链接</a></td>
    </tr>
  </tbody>
</table>

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

``` yaml
pipeline_name: image_multilabel_classification

SubModules:
  ImageMultiLabelClassification:
    module_name: image_multilabel_classification
    model_name: PP-HGNetV2-B6_ML
    model_dir: null # 替换为微调后的多标签分类模型权重路径
    batch_size: 4
```
随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device`参数</b>即可完成不同硬件之间的无缝切换。

例如，您使用昇腾 NPU 进行通用图像多标签分类产线的推理，使用的 CLI 命令为：

```bash
paddlex --pipeline image_multilabel_classification \
        --input general_image_classification_001.jpg \
        --device npu:0
```

当然，您也可以在 Python 脚本中 `create_pipeline()` 时或者 `predict()` 时指定硬件设备。

若您想在更多种类的硬件上使用通用通用图像多标签分类产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
