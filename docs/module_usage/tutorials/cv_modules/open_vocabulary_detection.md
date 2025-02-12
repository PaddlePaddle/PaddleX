---
comments: true
---

# 开放词汇目标检测模块使用教程

## 一、概述
开放词汇目标检测是当前一种先进的目标检测技术，旨在突破传统目标检测的局限性。传统方法仅能识别预定义类别的物体，而开放词汇目标检测允许模型识别未在训练中出现的物体。通过结合自然语言处理技术，利用文本描述来定义新的类别，模型能够识别和定位这些新物体。这使得目标检测更具灵活性和泛化能力，具有重要的应用前景。

## 二、支持模型列表


<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>GroundingDINO-T</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/GroundingDINO-T_infer.tar">推理模型</a></td>
<td>49.4</td>
<td>64.4</td>
<td>253.72</td>
<td>1807.4</td>
<td>658.3</td>
<td rowspan="3">基于O365,GoldG,Cap4M三个数据集训练的开放词汇目标目标检测模型。文本编码器采用Bert，视觉模型部份整体采用DINO，额外设计了一些跨模态融合模块，在开放词汇目标检测领域取得了较好的效果。</td>
</tr>
</table>

<b>注：以上精度指标为 COCO val2017 验证集 mAP(0.5:0.95)。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32</b>。


## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成whl包的安装后，几行代码即可完成开放词汇目标检测模块的推理，可以任意切换该模块下的模型，您也可以将开放词汇目标检测的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_detection.jpg)到本地。

```python
from paddlex import create_model
model = create_model(model_name='GroundingDINO-T')
results = model.predict(input='open_vocabulary_detection.jpg', prompt = 'bus . walking man . rearview mirror .', batch_size=1)
for res in results:
    res.print()
    res.save_to_img(f"./output/")
    res.save_to_json(f"./output/res.json")
```

运行后，得到的结果为：

```bash
{'res': "{'input_path': 'open_vocabulary_detection.jpg', 'boxes': [{'coordinate': [112.10542297363281, 117.93667602539062, 514.35693359375, 382.10150146484375], 'label': 'bus', 'score': 0.9348853230476379}, {'coordinate': [264.1828918457031, 162.6674346923828, 286.8844909667969, 201.86187744140625], 'label': 'rearview mirror', 'score': 0.6022508144378662}, {'coordinate': [606.1133422851562, 254.4973907470703, 622.56982421875, 293.7867126464844], 'label': 'walking man', 'score': 0.4384709894657135}, {'coordinate': [591.8192138671875, 260.2451171875, 607.3953247070312, 294.2210388183594], 'label': 'man', 'score': 0.3573091924190521}]}"}
```
运行结果参数含义如下：
- `input_path`: 表示输入待预测图像的路径
- `boxes`: 每个预测出的object的信息
  - `label`: 类别名称
  - `score`: 预测得分
  - `coordinate`: 预测框的坐标，格式为<code>[xmin, ymin, xmax, ymax]</code>

可视化图片如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/open_vocabulary_detection/open_vocabulary_detection_res.jpg">


相关方法、参数等说明如下：

* `create_model`实例化开放词汇目标检测模型（此处以`GroundingDINO-T`为例），具体说明如下：
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
<tr>
<td><code>thresholds</code></td>
<td>模型使用的过滤阈值</td>
<td><code>dict/None</code></td>
<td>无</td>
<td>None</td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。
* `thresholds`模型使用的过滤阈值，默认为None，表示使用上一层设置，参数设置的优先级从高到低为：`predict参数传入 > create_model初始化传入 > yaml配置文件设置`。
  * GroundingDINO系列模型在推理时需要设置两个阈值，分别为box_threshold（默认0.3）和text_threshold（默认0.25），参数传入形式为`{"box_threshold": 0.3, "text_threshold": 0.25}`。

* 调用开放词汇目标目标检测模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input` 、 `batch_size` 、 `thresholds` 和 `prompt`，具体说明如下：

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
<td><code>thresholds</code></td>
<td>模型使用的过滤阈值</td>
<td><code>dict</code>/<code>None</code></td>
<td>
<ul>
  <li><b>None</b>，表示沿用上一层设置, 参数设置优先级从高到低为: <code>predict参数传入 > create_model初始化传入 > yaml配置文件设置</code></li>
  <li><b>dict</b>，如<code>{"box_threshold": 0.3, "text_threshold": 0.25}</code>，表示推理时box_threshold设置为0.3，text_threshold设置为0.25</li>
</ul>
</td>
<td>None</td>
</tr>
<tr>
<td><code>prompt</code></td>
<td>模型预测使用的提示词</td>
<td><code>str</code></td>
<td>任意字符串</td>
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
