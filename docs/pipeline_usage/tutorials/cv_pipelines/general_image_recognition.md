---
comments: true
---

# 通用图像识别产线使用教程

## 1. 通用图像识别产线介绍

通用图像识别产线旨在解决开放域目标定位及识别问题，目前 PaddleX 的通用图像识别产线支持 PP-ShiTuV2。

PP-ShiTuV2 是一个实用的通用图像识别系统，主要由主体检测、特征学习和向量检索三个模块组成。该系统从骨干网络选择和调整、损失函数的选择、数据增强、学习率变换策略、正则化参数选择、预训练模型使用以及模型裁剪量化多个方面，融合改进多种策略，对各个模块进行优化，最终在多个实际应用场景上的检索性能均有较好效果。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/general_image_recognition/pp_shitu_v2.jpg"/>
<b>通用图像识别产线中包含了主体检测模块和图像特征模块</b>，有若干模型可供选择，您可以根据下边的 benchmark 数据来选择使用的模型。<b>如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。


<b>主体检测模块：</b>
<table>
<tr>
<th>模型</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-ShiTuV2_det</td>
<td>41.5</td>
<td>62.0</td>
<td>12.79 / 4.51</td>
<td>44.14 / 44.14</td>
<td>27.54</td>
<td>基于PicoDet_LCNet_x2_5的主体检测模型，模型可能会同时检测出多个常见主体。</td>
</tr>
</table>

注：以上精度指标为 PaddleClas 主体检测数据集。

<b>图像特征模块：</b>
<table>
<tr>
<th>模型</th>
<th>recall@1 (%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-ShiTuV2_rec</td>
<td>84.2</td>
<td>3.48 / 0.55</td>
<td>8.04 / 4.04</td>
<td>16.3 M</td>
<td rowspan="3">PP-ShiTuV2是一个通用图像特征系统，由主体检测、特征提取、向量检索三个模块构成，这些模型是其中的特征提取模块的模型之一，可以根据系统的情况选择不同的模型。</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_base</td>
<td>88.69</td>
<td>12.94 / 2.88</td>
<td>58.36 / 58.36</td>
<td>306.6 M</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_large</td>
<td>91.03</td>
<td>51.65 / 11.18</td>
<td>255.78 / 255.78</td>
<td>1.05 G</td>
</tr>
</table>

注：以上精度指标为 AliProducts recall@1。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。

## 2. 快速开始

PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在本地使用 Python 体验通用图像识别产线的效果。

### 2.1 在线体验

暂不支持在线体验。

### 2.2 本地体验

> ❗ 在本地使用通用图像识别产线前，请确保您已经按照[PaddleX安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.2.1 命令行方式体验

该产线暂不支持命令行体验。

#### 2.2.2 Python脚本方式集成

* 在该产线的运行示例中需要预先构建索引库，您可以下载官方提供的饮料识别测试数据集[drink_dataset_v2.0](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/drink_dataset_v2.0.tar) 构建索引库。若您希望用私有数据集，可以参考[2.3节 构建索引库的数据组织方式](#23-构建索引库的数据组织方式)。之后通过几行代码即可完成建立索引库和通用图像识别产线的快速推理。

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-ShiTuV2")

index_data = pipeline.build_index(gallery_imgs="drink_dataset_v2.0/", gallery_label="drink_dataset_v2.0/gallery.txt")
index_data.save("drink_index")

output = pipeline.predict("./drink_dataset_v2.0/test_images/001.jpeg", index=index_data)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）调用 `create_pipeline` 实例化通用图像识别产线对象。具体参数说明如下：

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
<td>无</td>
</tr>
<tr>
<td><code>config</code></td>
<td>产线具体的配置信息（如果和<code>pipeline</code>同时设置，优先级高于<code>pipeline</code>，且要求产线名和<code>pipeline</code>一致）。</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
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

（2）调用通用图像识别产线对象的 `build_index` 方法，构建索引库。具体参数为说明如下：

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
<tbody>
<tr>
<td><code>gallery_imgs</code></td>
<td>要添加的底库图片，必需参数</td>
<td><code>str</code>|<code>list</code></td>
<td>
<ul>
<li><b>str</b>：数据集的根目录，数据组织方式参考<a href="#2.3-构建索引库的数据组织方式">2.3节 构建索引库的数据组织方式</a></li>
<li><b>List[numpy.ndarray]</b>：numpy.array列表类型的底库图片数据</li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>gallery_label</code></td>
<td>底库图片的标注信息，必需参数</td>
<td><code>str|list</code></td>
<td>
<ul>
<li><b>str</b>：数据标注文件路径，数据组织方式参考<a href="#2.3-构建索引库的数据组织方式">2.3节 构建索引库的数据组织方式</a></li>
<li><b>List[str]</b>：str列表类型的底库图片标注</li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>metric_type</code></td>
<td>特征度量方式，可选参数</td>
<td><code>str</code></td>
<td>
<ul>
<li><code>"IP"</code>：内积（Inner Product）</li>
<li><code>"L2"</code>：欧几里得距离（Euclidean Distance）</li>
</ul>
</td>
<td><code>"IP"</code></td>
</tr>
<tr>
<td><code>index_type</code></td>
<td>索引类型，可选参数</td>
<td><code>str</code></td>
<td>
<ul>
<li><code>"HNSW32"</code>：检索速度较快且精度较高，但不支持 <code>remove_index()</code> 操作</li>
<li><code>"IVF"</code>：检索速度较快但精度相对较低，支持 <code>append_index()</code> 和 <code>remove_index()</code> 操作</li>
<li><code>"Flat"</code>：检索速度较低精度较高，支持 <code>append_index()</code> 和 <code>remove_index()</code> 操作</li>
</ul>
</td>
<td><code>"HNSW32"</code></td>
</tr>
</tbody>
</table>

索引库对象 `index` 支持 `save` 方法，用于将索引库保存到磁盘：

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
<td><code>save_path</code></td>
<td>索引库文件的保存目录，如<code>drink_index</code>。</td>
<td><code>str</code></td>
<td>无</td>
</tr>
</tbody>
</table>

（3）调用通用图像识别产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为 `input`，用于输入待预测数据，支持多种输入方式，具体示例如下：

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
<td>待预测数据，支持多种输入类型，必需参数</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
<li><b>str</b>：如图像文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code></li>
<li><b>List</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>index</code></td>
<td>产线推理预测所用的特征库，可选参数。如不传入该参数，则默认使用产线配置文件中指定的索引库。</td>
<td><code>str|paddlex.inference.components.retrieval.faiss.IndexData|None</code></td>
<td>
<ul>
<li><b>str</b>类型表示的目录（该目录下需要包含特征库文件，包括<code>vector.index</code>和<code>index_info.yaml</code>）</li>
<li><code>build_index</code>方法创建的<b>IndexData</b>对象</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</table>

（4）对预测结果进行处理：每个样本的预测结果均为对应的Result对象，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

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
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">打印结果到终端</td>
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
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">将结果保存为json格式的文件</td>
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

- 调用 `print()` 方法会将如下结果打印到终端：
```bash
{'res': {'input_path': './drink_dataset_v2.0/test_images/001.jpeg', 'boxes': [{'labels': ['红牛-强化型', '红牛-强化型', '红牛-强化型', '红牛-强化型', '红牛-强化型'], 'rec_scores': [0.720183789730072, 0.7044230699539185, 0.6812724471092224, 0.6583285927772522, 0.6578206419944763], 'det_score': 0.6135568618774414, 'coordinate': [343.8184, 98.96374, 528.0366, 593.3813]}]}}
```

- 输出结果参数含义如下：
    - `input_path`：表示输入图像的路径
    - `boxes`：检测到的物体信息，一个字典列表，每个字典包含以下信息：
        - `labels`：识别标签列表，按照分数从高到低排序
        - `rec_scores`：识别分数列表，其中元素与`labels`一一对应
        - `det_score`：检测得分
        - `coordinate`：目标框坐标，格式为[xmin, ymin, xmax, ymax]

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}.json`，如果指定为文件，则直接保存到该文件中。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)，上述示例中，可视化结果如下所示：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/general_image_recognition/01.jpg"/>

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
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">获取格式为 <code>dict</code> 的可视化图像</td>
</tr>
</table>

- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。
- `img` 属性返回的预测结果是一个字典类型的数据。键为 `res` ，对应的值是一个用于可视化通用图像识别结果的 `Image.Image` 对象。

上述Python脚本集成方式默认使用 PaddleX 官方配置文件中的参数设置，若您需要自定义配置文件，可先执行如下命令获取官方配置文件，并保存在 `my_path` 中：

```bash
paddlex --get_pipeline_config PP-ShiTuV2 --save_path ./my_path
```

若您获取了配置文件，即可对通用图像识别产线各项配置进行自定义。只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为自定义产线配置文件路径即可。

例如，若您的自定义配置文件保存在 `./my_path/PP-ShiTuV2.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/PP-ShiTuV2.yaml")

output = pipeline.predict("./drink_dataset_v2.0/test_images/001.jpeg", index="drink_index")
for res in output:
    res.print()
    res.save_to_json("./output/")
    res.save_to_img("./output/")
```

<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改通用图像识别产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。

#### 2.2.3 索引库的添加和删除操作

若您希望将更多的图像添加到索引库中，则可以调用 `append_index` 方法；删除图像特征，则可以调用 `remove_index` 方法。

```python
from paddlex import create_pipeline

pipeline = create_pipeline("PP-ShiTuV2")
index_data = pipeline.build_index(gallery_imgs="drink_dataset_v2.0/", gallery_label="drink_dataset_v2.0/gallery.txt", index_type="IVF", metric_type="IP")
index_data = pipeline.append_index(gallery_imgs="drink_dataset_v2.0/", gallery_label="drink_dataset_v2.0/gallery.txt", index=index_data)
index_data = pipeline.remove_index(remove_ids="drink_dataset_v2.0/remove_ids.txt", index=index_data)
index_data.save("drink_index")
```

上述方法参数说明如下：
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
<tbody>
<tr>
<td><code>gallery_imgs</code></td>
<td>要添加的底库图片，必需参数</td>
<td><code>str</code>|<code>list</code></td>
<td>
<ul>
<li><b>str</b>：图片根目录，数据组织方式参考<a href="#2.3-构建索引库的数据组织方式">2.3节 构建索引库的数据组织方式</a></li>
<li><b>List[numpy.ndarray]</b>：numpy.array列表类型的底库图片数据</li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>gallery_label</code></td>
<td>底库图片的标注信息，必需参数</td>
<td><code>str|list</code></td>
<td>
<ul>
<li><b>str</b>：标注文件的路径，数据组织方式与构建特征库时相同，参考<a href="#2.3-构建索引库的数据组织方式">2.3节 构建索引库的数据组织方式</a></li>
<li><b>List[str]</b>：str列表类型的底库图片标注</li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>metric_type</code></td>
<td>特征度量方式，可选参数</td>
<td><code>str</code></td>
<td>
<ul>
<li><code>"IP"</code>：内积（Inner Product）</li>
<li><code>"L2"</code>：欧几里得距离（Euclidean Distance）</li>
</ul>
</td>
<td><code>"IP"</code></td>
</tr>
<tr>
<td><code>index_type</code></td>
<td>索引类型，可选参数</td>
<td><code>str</code></td>
<td>
<ul>
<li><code>"HNSW32"</code>：检索速度较快且精度较高，但不支持 <code>remove_index()</code> 操作</li>
<li><code>"IVF"</code>：检索速度较快但精度相对较低，支持 <code>append_index()</code> 和 <code>remove_index()</code> 操作</li>
<li><code>"Flat"</code>：检索速度较低精度较高，支持 <code>append_index()</code> 和 <code>remove_index()</code> 操作</li>
</ul>
</td>
<td><code>"HNSW32"</code></td>
</tr>
<tr>
<td><code>remove_ids</code></td>
<td>待删除的索引序号，</td>
<td><code>str</code>|<code>list</code></td>
<td>
<ul>
<li><b>str</b>：表示的txt文件的路径，内容为待删除的索引id，每行一个“id”；</li>
<li><b>List[int]</b>：表示的待删除的索引序号。仅在 <code>remove_index</code> 中有效。</li></ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>index</code></td>
<td>产线推理预测所用的特征库</td>
<td><code>str|paddlex.inference.components.retrieval.faiss.IndexData</code></td>
<td>
<ul>
<li><b>str</b>类型表示的目录（该目录下需要包含特征库文件，包括<code>vector.index</code>和<code>index_info.yaml</code>）</li>
<li><code>build_index</code>方法创建的<b>IndexData</b>对象</li>
</ul>
</td>
<td>无</td>
</tr>
</tbody>
</table>
<b>注意</b>：<code>HNSW32</code>在windows平台存在兼容性问题，可能导致索引库无法构建、加载。

### 2.3 构建索引库的数据组织方式

PaddleX 的通用图像识别产线示例需要使用预先构建好的索引库进行特征检索。如果您希望用私有数据构建索引库，则需要按照如下方式组织数据：

```bash
data_root             # 数据集根目录，目录名称可以改变
├── images            # 图像的保存目录，目录名称可以改变
│   │   ...
└── gallery.txt       # 索引库数据集标注文件，文件名称可以改变。每行给出待检索图像路径和图像标签，使用空格分隔，内容举例： “0/0.jpg 脉动”
```

## 3. 开发集成/部署

如果通用图像识别产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将通用图像识别产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

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
<li><b><code>buildIndex</code></b></li>
</ul>
<p>构建特征向量索引。</p>
<p><code>POST /shitu-index-build</code></p>
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
<td><code>imageLabelPairs</code></td>
<td><code>array</code></td>
<td>用于构建索引的图像-标签对。</td>
<td>是</td>
</tr>
</tbody>
</table>
<p><code>imageLabelPairs</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>image</code></td>
<td><code>string</code></td>
<td>服务器可访问的图像文件的URL或图像文件内容的Base64编码结果。</td>
</tr>
<tr>
<td><code>label</code></td>
<td><code>string</code></td>
<td>标签。</td>
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
<td><code>indexKey</code></td>
<td><code>string</code></td>
<td>索引对应的键，用于标识建立的索引。可用作其他操作的输入。</td>
</tr>
<tr>
<td><code>idMap</code></td>
<td><code>object</code></td>
<td>向量ID到标签的映射。</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>addImagesToIndex</code></b></li>
</ul>
<p>将图像（对应的特征向量）加入索引。</p>
<p><code>POST /shitu-index-add</code></p>
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
<td><code>imageLabelPairs</code></td>
<td><code>array</code></td>
<td>用于构建索引的图像-标签对。</td>
<td>是</td>
</tr>
<tr>
<td><code>indexKey</code></td>
<td><code>string</code></td>
<td>索引对应的键。由<code>buildIndex</code>操作提供。</td>
<td>是</td>
</tr>
</tbody>
</table>
<p><code>imageLabelPairs</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>image</code></td>
<td><code>string</code></td>
<td>服务器可访问的图像文件的URL或图像文件内容的Base64编码结果。</td>
</tr>
<tr>
<td><code>label</code></td>
<td><code>string</code></td>
<td>标签。</td>
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
<td><code>idMap</code></td>
<td><code>object</code></td>
<td>向量ID到标签的映射。</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>removeImagesFromIndex</code></b></li>
</ul>
<p>从索引中移除图像（对应的特征向量）。</p>
<p><code>POST /shitu-index-remove</code></p>
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
<td><code>ids</code></td>
<td><code>array</code></td>
<td>需要从索引中移除的向量的ID。</td>
<td>是</td>
</tr>
<tr>
<td><code>indexKey</code></td>
<td><code>string</code></td>
<td>索引对应的键。由<code>buildIndex</code>操作提供。</td>
<td>是</td>
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
<td><code>idMap</code></td>
<td><code>object</code></td>
<td>向量ID到标签的映射。</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>进行图像识别。</p>
<p><code>POST /shitu-infer</code></p>
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
<td><code>indexKey</code></td>
<td><code>string</code></td>
<td>索引对应的键。由<code>buildIndex</code>操作提供。</td>
<td>否</td>
</tr>
<tr>
<td><code>detThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>det_threshold</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>recThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>rec_threshold</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>hammingRadius</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>hamming_radius</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>topk</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>topk</code> 参数说明。</td>
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
<td>检测到的目标的信息。</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code> | <code>null</code></td>
<td>识别结果图。图像为JPEG格式，使用Base64编码。</td>
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
<td><code>recResults</code></td>
<td><code>array</code></td>
<td>识别结果。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>检测得分。</td>
</tr>
</tbody>
</table>
<p><code>recResults</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>label</code></td>
<td><code>string</code></td>
<td>标签。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>识别得分。</td>
</tr>
</tbody>
</table>
</details>
<details><summary>多语言调用服务示例</summary>
<details>
<summary>Python</summary>
<pre><code class="language-python">import base64
import pprint
import sys

import requests

API_BASE_URL = "http://0.0.0.0:8080"

base_image_label_pairs = [
    {"image": "./demo0.jpg", "label": "兔子"},
    {"image": "./demo1.jpg", "label": "兔子"},
    {"image": "./demo2.jpg", "label": "小狗"},
]
image_label_pairs_to_add = [
    {"image": "./demo3.jpg", "label": "小狗"},
]
ids_to_remove = [1]
infer_image_path = "./demo4.jpg"
output_image_path = "./out.jpg"

for pair in base_image_label_pairs:
    with open(pair["image"], "rb") as file:
        image_bytes = file.read()
        image_data = base64.b64encode(image_bytes).decode("ascii")
    pair["image"] = image_data

payload = {"imageLabelPairs": base_image_label_pairs}
resp_index_build = requests.post(f"{API_BASE_URL}/shitu-index-build", json=payload)
if resp_index_build.status_code != 200:
    print(f"Request to shitu-index-build failed with status code {resp_index_build}.")
    pprint.pp(resp_index_build.json())
    sys.exit(1)
result_index_build = resp_index_build.json()["result"]
print(f"Number of images indexed: {len(result_index_build['idMap'])}")

for pair in image_label_pairs_to_add:
    with open(pair["image"], "rb") as file:
        image_bytes = file.read()
        image_data = base64.b64encode(image_bytes).decode("ascii")
    pair["image"] = image_data

payload = {"imageLabelPairs": image_label_pairs_to_add, "indexKey": result_index_build["indexKey"]}
resp_index_add = requests.post(f"{API_BASE_URL}/shitu-index-add", json=payload)
if resp_index_add.status_code != 200:
    print(f"Request to shitu-index-add failed with status code {resp_index_add}.")
    pprint.pp(resp_index_add.json())
    sys.exit(1)
result_index_add = resp_index_add.json()["result"]
print(f"Number of images indexed: {len(result_index_add['idMap'])}")

payload = {"ids": ids_to_remove, "indexKey": result_index_build["indexKey"]}
resp_index_remove = requests.post(f"{API_BASE_URL}/shitu-index-remove", json=payload)
if resp_index_remove.status_code != 200:
    print(f"Request to shitu-index-remove failed with status code {resp_index_remove}.")
    pprint.pp(resp_index_remove.json())
    sys.exit(1)
result_index_remove = resp_index_remove.json()["result"]
print(f"Number of images indexed: {len(result_index_remove['idMap'])}")

with open(infer_image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data, "indexKey": result_index_build["indexKey"]}
resp_infer = requests.post(f"{API_BASE_URL}/shitu-infer", json=payload)
if resp_infer.status_code != 200:
    print(f"Request to shitu-infer failed with status code {resp_infer}.")
    pprint.pp(resp_infer.json())
    sys.exit(1)
result_infer = resp_infer.json()["result"]

with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result_infer["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected objects:")
pprint.pp(result_infer["detectedObjects"])
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。


## 4. 二次开发

如果通用图像识别产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升通用该产线的在您的场景中的识别效果。

### 4.1 模型微调

由于通用图像识别产线包含两个模块（主体检测模块和图像特征模块），模型产线的效果不及预期可能来自于其中任何一个模块。

您可以对识别效果差的图片进行分析，如果在分析过程中发现有较多的主体目标未被检测出来，那么可能是主体检测模型存在不足，您需要参考[主体检测模块开发教程](../../../module_usage/tutorials/cv_modules/mainbody_detection.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/mainbody_detection.md#四二次开发)章节，使用您的私有数据集对主体检测模型进行微调；如果在已检测到的主体出现匹配错误，这表明图像特征模型需要进一步改进，您需要参考[图像特征模块开发教程](../../../module_usage/tutorials/cv_modules/image_feature.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/image_feature.md#四二次开发)章节,对图像特征模型进行微调。

### 4.2 模型应用

当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```yaml

...

SubModules:
  Detection:
    module_name: text_detection
    model_name: PP-ShiTuV2_det
    model_dir: null #可修改为微调后主体检测模型的本地路径
    batch_size: 1
  Recognition:
    module_name: text_recognition
    model_name: PP-ShiTuV2_rec
    model_dir: null #可修改为微调后图像特征模型的本地路径
    batch_size: 1
```
随后， 参考[2.2 本地体验](#22-本地体验)中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持

PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU 和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device`参数</b>即可完成不同硬件之间的无缝切换。

例如，使用Python运行通用图像识别产线时，将运行设备从英伟达 GPU 更改为昇腾 NPU，仅需将脚本中的 `device` 修改为 npu 即可：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="PP-ShiTuV2",
    device="npu:0" # gpu:0 --> npu:0
    )
```

若您想在更多种类的硬件上使用通用图像识别产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
