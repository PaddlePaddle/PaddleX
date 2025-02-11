---
comments: true
---

# 旋转目标检测模块使用教程

## 一、概述
旋转目标检测是目标检测模块中的一种衍生，它专门针对旋转目标进行检测。旋转框（Rotated Bounding Boxes）常用于检测带有角度信息的矩形框，即矩形框的宽和高不再与图像坐标轴平行。相较于水平矩形框，旋转矩形框一般包括更少的背景信息。旋转框检测常用于遥感等场景中。

## 二、支持模型列表

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(%)</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-YOLOE-R-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-YOLOE-R-L_infer.tar">推理模型</a>/<a href="https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams">训练模型</a></td>
<td>78.14</td>
<td>20.7039</td>
<td>157.942</td>
<td>211.0 M</td>
<td rowspan="1">PP-YOLOE-R是一个高效的单阶段Anchor-free旋转框检测模型。基于PP-YOLOE, PP-YOLOE-R以极少的参数量和计算量为代价，引入了一系列有用的设计来提升检测精度。</td>
</tr>
</table>

<p><b>注：以上精度指标为<a href="https://captain-whu.github.io/DOTA/">DOTA</a>验证集 mAP(0.5:0.95)。所有模型 GPU 推理耗时基于 NVIDIA TRX2080 Ti 机器，精度类型为 F16， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>



## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成 wheel 包的安装后，几行代码即可完成旋转目标检测模块的推理，可以任意切换该模块下的模型，您也可以将旋转目标检测的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/rotated_object_detection_001.png)到本地。
```python
from paddlex import create_model
model = create_model(model_name="PP-YOLOE-R-L")
output = model.predict(input="rotated_object_detection_001.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

运行后，得到的结果为：
```bash
{'res': {'input_path': 'rotated_object_detection_001.png', 'page_index': None, 'boxes': [{'cls_id': 4, 'label': 'small-vehicle', 'score': 0.7409099340438843, 'coordinate': [92.88687, 763.1569, 85.163124, 749.5868, 116.07975, 731.99414, 123.8035, 745.5643]}, {'cls_id': 4, 'label': 'small-vehicle', 'score': 0.7393015623092651, 'coordinate': [348.2332, 177.55974, 332.77704, 150.24973, 345.2183, 143.21028, 360.67444, 170.5203]}, {'cls_id': 11, 'label': 'roundabout', 'score': 0.8101699948310852, 'coordinate': [537.1732, 695.5475, 204.4297, 612.9735, 286.71338, 281.48022, 619.4569, 364.05426]}]}}
```
运行结果参数含义如下：
- `input_path`: 表示输入待预测图像的路径
- `boxes`: 每个预测出的object的信息
  - `cls_id`: 类别ID
  - `label`: 类别名称
  - `score`: 预测得分
  - `coordinate`: 预测框的坐标，格式为<code>[x1, y1, x2, y2, x3, y3, x4, y4]</code>

可视化图片如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/rotated_object_detection/rotated_object_detection_001_res.png">


相关方法、参数等说明如下：

* `create_model`实例化旋转目标检测模型（此处以`PP-YOLOE-R-L`为例），具体说明如下：
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
<td><code>threshold</code></td>
<td>低分object过滤阈值</td>
<td><code>float/None/dict[int, float]</code></td>
<td>无</td>
<td>None</td>
</tr>
<tr>
<td><code>img_size</code></td>
<td>模型实际预测使用的分辨率</td>
<td><code>int/tuple/None</code></td>
<td>无</td>
<td>None</td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* `threshold`为低分object过滤阈值，默认为None，表示使用上一层设置，参数设置的优先级从高到低为：`predict参数传入 > create_model初始化传入 > yaml配置文件设置`。目前支持float和dict两种阈值设置方式：
  * `float`, 对于所有的类别使用同一个阈值。
  * `dict[int, float]`, key为类别ID，value为阈值，对于不同的类别使用不同的阈值。

* `img_size`为模型实际预测使用的分辨率，默认为None，表示使用上一层设置，参数设置的优先级从高到低为：`create_model初始化 > yaml配置文件设置`。

* 调用旋转目标检测模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input` 、 `batch_size` 和 `threshold`，具体说明如下：

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
  <li><b>URL链接</b>，如图像文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png">示例</a></li>
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
<td><code>threshold</code></td>
<td>低分object过滤阈值</td>
<td><code>float</code>/<code>dict[int, float]</code>/<code>None</code></td>
<td>
<ul>
  <li><b>None</b>，表示沿用上一层设置, 参数设置优先级从高到低为: <code>predict参数传入 > create_model初始化传入 > yaml配置文件设置</code></li>
  <li><b>float</b>，对于所有的类别使用同一个阈值。如0.5，表示推理时使用0.5作为所有类别的低分object过滤阈值</li>
  <li><b>dict[int, float]</b>，如<code>{0: 0.5, 1: 0.35}</code>，表示推理时对类别0使用0.5低分过滤阈值，对类别1使用0.35低分过滤阈值。</li>
</ul>
</td>
<td>None</td>
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
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的旋转目标检测模型。在使用 PaddleX 开发旋转目标检测模型之前，请务必安装 PaddleX的旋转目标检测相关模型训练插件，安装过程可以参考 [PaddleX本地安装教程](../../../installation/installation.md)

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX目标检测任务模块数据标注教程](../../../data_annotations/cv_modules/object_detection.md)。

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/rdet_dota_examples.tar -P ./dataset
tar -xf ./dataset/rdet_dota_examples.tar -C ./dataset/
```
解压后，数据集目录结构如下：
```bash
- dataset/rdet_dota_examples
  - annotations
    - instance_train.json
    - instance_val.json
  - images
    - img1.png
    - img2.png
    - img3.png
    ...
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/rdet_dota_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>

<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;num_classes&quot;: 15,
    &quot;train_samples&quot;: 194,
    &quot;train_sample_paths&quot;: [
      &quot;check_dataset\/demo_img\/P0457__1.0__379___0.png&quot;,
      &quot;check_dataset\/demo_img\/P1560__1.0__0___0.png&quot;,
      &quot;check_dataset\/demo_img\/P2722__1.0__0___1422.png&quot;,
      &quot;check_dataset\/demo_img\/P1750__1.0__824___1648.png&quot;,
      &quot;check_dataset\/demo_img\/P1560__1.0__1648___824.png&quot;,
      &quot;check_dataset\/demo_img\/P1751__1.0__2472___1648.png&quot;,
      &quot;check_dataset\/demo_img\/P1560__1.0__2976___2976.png&quot;,
      &quot;check_dataset\/demo_img\/P2766__1.0__4421___0.png&quot;,
      &quot;check_dataset\/demo_img\/P2365__1.0__1807___0.png&quot;,
      &quot;check_dataset\/demo_img\/P0117__1.0__0___138.png&quot;
    ],
    &quot;val_samples&quot;: 21,
    &quot;val_sample_paths&quot;: [
      &quot;check_dataset\/demo_img\/P0844__1.0__0___0.png&quot;,
      &quot;check_dataset\/demo_img\/P0457__1.0__0___0.png&quot;,
      &quot;check_dataset\/demo_img\/P2645__1.0__0___0.png&quot;,
      &quot;check_dataset\/demo_img\/P1651__1.0__824___824.png&quot;,
      &quot;check_dataset\/demo_img\/P1529__1.0__824___2976.png&quot;,
      &quot;check_dataset\/demo_img\/P1750__1.0__3260___824.png&quot;,
      &quot;check_dataset\/demo_img\/P0725__1.0__634___0.png&quot;,
      &quot;check_dataset\/demo_img\/P2722__1.0__2472___0.png&quot;,
      &quot;check_dataset\/demo_img\/P0262__1.0__0___1414.png&quot;,
      &quot;check_dataset\/demo_img\/P1750__1.0__0___2472.png&quot;,
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;rdet_dota_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;COCODetDataset&quot;
}
</code></pre>
<p>上述校验结果中，check_pass 为 true 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.num_classes</code>：该数据集类别数为 15；</li>
<li><code>attributes.train_samples</code>：该数据集训练集样本数量为 194</li>
<li><code>attributes.val_samples</code>：该数据集验证集样本数量为 21</li>
<li><code>attributes.train_sample_paths</code>：该数据集训练集样本可视化图片相对路径列表；</li>
<li><code>attributes.val_sample_paths</code>：该数据集验证集样本可视化图片相对路径列表；</li>
</ul>
<p>另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/rotated_object_detection/01.png"></p></details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过<b>修改配置文件</b>或是<b>追加超参数</b>的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>

<p><b>（1）数据集格式转换</b></p>

旋转目标检测暂不支持数据格式转换，只支持标准DOTA的COCO数据格式。

<p><b>（2）数据集划分</b></p>
<p>数据集划分的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: 是否进行重新划分数据集，为 <code>True</code> 时进行数据集格式转换，默认为 <code>False</code>；</li>
<li><code>train_percent</code>: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证和 <code>val_percent</code> 值加和为100；</li>
<li><code>val_percent</code>: 如果重新划分数据集，则需要设置验证集的百分比，类型为0-100之间的任意整数，需要保证和 <code>train_percent</code> 值加和为100；
例如，您想重新划分数据集为 训练集占比90%、验证集占比10%，则需将配置文件修改为：</li>
</ul>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
</code></pre>
<p>随后执行命令：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/rdet_dota_examples
</code></pre>
<p>数据划分执行之后，原有标注文件会被在原路径下重命名为 <code>xxx.bak</code>。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/rdet_dota_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处旋转目标检测模型 `PP-YOLOE-R-L` 的训练为例：

```bash
python main.py -c paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/rdet_dota_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-YOLOE-R-L.yaml`，训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅模型对应任务模块的配置文件说明[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<ul>
<li>模型训练过程中，PaddleX 会自动保存模型权重文件，默认为<code>output</code>，如需指定保存路径，可通过配置文件中 <code>-o Global.output</code> 字段进行设置。</li>
<li>PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。</li>
<li>
<p>在完成模型训练后，所有产出保存在指定的输出目录（默认为<code>./output/</code>）下，通常有以下产出：</p>
</li>
<li>
<p><code>train_result.json</code>：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；</p>
</li>
<li><code>train.log</code>：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；</li>
<li><code>config.yaml</code>：训练配置文件，记录了本次训练的超参数的配置；</li>
<li><code>.pdparams</code>、<code>.pdema</code>、<code>.pdopt.pdstate</code>、<code>.pdiparams</code>、<code>.pdmodel</code>：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；</li>
</ul></details>

## <b>4.3 模型评估</b>
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：

```bash
python main.py -c paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/rdet_dota_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-YOLOE-R-L.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>。</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json，其记录了</code>评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 AP；</p></details>

### <b>4.4 模型推理和模型集成</b>
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理

* 通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/rotated_object_detection_001.png)到本地。
```bash
python main.py -c paddlex/configs/modules/rotated_object_detection/PP-YOLOE-R-L.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="rotated_object_detection_001.png"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-YOLOE-R-L.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>模块集成</b>

您产出的权重可以直接集成到旋转目标检测模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。
