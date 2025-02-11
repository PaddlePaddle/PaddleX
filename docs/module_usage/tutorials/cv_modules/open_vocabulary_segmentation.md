---
comments: true
---

# 开放词汇分割模块使用教程

## 一、概述
开放词汇分割是一项图像分割任务，旨在根据文本描述、边框、关键点等除图像以外的信息作为提示，分割图像中对应的物体。它允许模型处理广泛的对象类别，而无需预定义的类别列表。这项技术结合了视觉和多模态技术，极大地提高了图像处理的灵活性和精度。开放词汇分割在计算机视觉领域具有重要应用价值，尤其在复杂场景下的对象分割任务中表现突出。

## 二、支持模型列表


<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>SAM-H_box</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SAM-H_box_infer.tar">推理模型</a></td>
<td>144.9</td>
<td>33920.7</td>
<td>2433.7</td>
<td rowspan="2">SAM（Segment Anything Model）是一种先进的图像分割模型，能够根据用户提供的简单提示（如点、框或文本）对图像中的任意对象进行分割。基于SA-1B数据集训练，有一千万的图像数据和十一亿掩码标注，在大部分场景均有较好的效果。其中SAM-H_box表示使用框作为分割提示输入，SAM会分割被框包裹主的主体；SAM-H_point表示使用点作为分割提示输入，SAM会分割点所在的主体。</td>
</tr>
<tr>
<td>SAM-H_point</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SAM-H_point_infer.tar">推理模型</a></td>
<td>144.9</td>
<td>33920.7</td>
<td>2433.7</td>
</tr>
</table>

<b>注：所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32</b>。


## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成whl包的安装后，几行代码即可完成开放词汇分割模块的推理，可以任意切换该模块下的模型，您也可以将开放词汇分割的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_segmentation.jpg)到本地。

```python
from paddlex import create_model
model = create_model(model_name='SAM-H_box')
results = model.predict(
    input="open_vocabulary_segmentation.jpg",
    prompts={
        "box_prompt": [
            [112.9239273071289,118.38755798339844,513.7587890625,382.0570068359375],
            [4.597158432006836,263.5540771484375,92.20092010498047,336.5640869140625],
            [592.3548583984375,260.8838806152344,607.1813354492188,294.2261962890625]
        ],
    },
    batch_size=1
)
for res in results:
    res.print()
    res.save_to_img(f"./output/")
    res.save_to_json(f"./output/res.json")
```

运行后，得到的结果为：
```bash
{'res': "{'input_path': 'open_vocabulary_segmentation.jpg', 'prompts': {'box_prompt': [[112.9239273071289, 118.38755798339844, 513.7587890625, 382.0570068359375], [4.597158432006836, 263.5540771484375, 92.20092010498047, 336.5640869140625], [592.3548583984375, 260.8838806152344, 607.1813354492188, 294.2261962890625]]}, 'masks': '...', 'mask_infos': [{'label': 'box_prompt', 'prompt': [112.9239273071289, 118.38755798339844, 513.7587890625, 382.0570068359375]}, {'label': 'box_prompt', 'prompt': [4.597158432006836, 263.5540771484375, 92.20092010498047, 336.5640869140625]}, {'label': 'box_prompt', 'prompt': [592.3548583984375, 260.8838806152344, 607.1813354492188, 294.2261962890625]}]}"}
```
运行结果参数含义如下：
- `input_path`: 表示输入待预测图像的路径
- `prompts`: 预测使用的原始prompt信息
- `masks`: 实际预测的mask，由于数据过大不便于直接print，所以此处用`...`替换，可以通过`res.save_to_img()`将预测结果保存为图片，通过`res.save_to_json()`将预测结果保存为json文件。
- `mask_infos`: 每个预测的mask对应的prompt信息
  - `label`: 预测的mask对应的prompt类型
  - `prompt`: 预测的mask对应的原始prompt输入

可视化图片如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/open_vocabulary_segmentation/open_vocabulary_segmentation_res.jpg">

相关方法、参数等说明如下：

* `create_model`实例化开放词汇分割模型（此处以`SAM-H_box`为例），具体说明如下：
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
<td><code>model_name</code></td>
<td>模型名称</td>
<td><code>str</code></td>
<td>无</td>
<td><code>无</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>模型存储路径</td>
<td><code>str</code></td>
<td>无</td>
<td>无</td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 调用开放词汇分割模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input` 、 `batch_size` 和 `prompts`，具体说明如下：

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
<td>待预测数据，支持多种输入类型</td>
<td><code>Python Var</code>/<code>str</code>/<code>list</code></td>
<td>
<ul>
  <li><b>Python变量</b>，如<code>numpy.ndarray</code>表示的图像数据</li>
  <li><b>文件路径</b>，如图像文件的本地路径：<code>/root/data/img.jpg</code></li>
  <li><b>URL链接</b>，如图像文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_detection.jpg">示例</a></li>
  <li><b>本地目录</b>，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code></li>
  <li><b>列表</b>，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>批大小</td>
<td><code>int</code></td>
<td>任意整数</td>
<td>1</td>
</tr>
<tr>
<td><code>prompts</code></td>
<td>模型使用提示词</td>
<td><code>dict</code></td>
<td>
<ul>
  <li><b>dict</b>，如<code>{"box_prompt": [[float, float, float, foat], ...]}</code>，表示推理时使用的多个bbox作为prompt</li>
</ul>
</td>
<td>无</td>
</tr>
</table>

* 对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为`json`文件的操作:

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
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
</table>

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
<td rowspan = "1">获取预测的<code>json</code>格式的结果</td>
</tr>
<tr>
<td rowspan = "1"><code>img</code></td>
<td rowspan = "1">获取格式为<code>dict</code>的可视化图像</td>
</tr>
</table>


关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
当前模块暂时不支持微调训练，仅支持推理集成。关于该模块的微调训练，计划在未来支持。
