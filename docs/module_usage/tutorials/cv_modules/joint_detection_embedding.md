---
comments: true
---

# 联合检测嵌入模块使用教程

## 一、概述
联合检测嵌入（Joint Detection and Embedding, JDE）模块是一种基于深度学习的多目标跟踪（MOT）方法，旨在通过联合检测和 REID 任务，实现高效、实时的目标跟踪。JDE模块的核心在于将目标检测和外观嵌入（embedding）提取结合到一个统一的网络框架中。传统目标跟踪方法通常将检测和 REID 分为两个独立的步骤，分别使用不同的模型来完成，而JDE通过共享特征提取器和预测模块，减少了模型参数和计算量

## 二、支持模型列表

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

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成 wheel 包的安装后，几行代码即可完成联合检测嵌入模块的推理，可以任意切换该模块下的模型，您也可以将联合检测嵌入模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/joint_detection_embedding_000.png)到本地。

```python
from paddlex import create_model

model = create_model(model_name="FairMOT-DLA-34_1088x608")
output = model.predict("joint_detection_embedding_000.png")

for res in output:
    res.print()
    res.save_to_json("./output/")
```

运行后，得到的结果为:

```bash
{'res': {'input_path': 'joint_detection_embedding_000.png', 'pred_dets': array([[   0.    , ..., 1011.8289],
       ...,
       [   0.    , ...,  403.8857]], dtype=float32), 'pred_embs': array([[ 0.0138496 , ..., -0.05584364],
       ...,
       [ 0.02329516, ..., -0.01240982]], dtype=float32)}}
```

参数含义如下：
- `input_path`：表示输入图像的路径
- `pred_dets`：JDE模块预测的目标框信息，一个numpy数组，形状为[N, 6]，N为检测到的目标数量，6为每个目标框的坐标信息（cls_id, score, xmin, ymin, xmax, ymax）。
- `pred_embs`：预测的目标框信息，一个numpy数组，形状为[N, 128]，N为检测到的目标数量，128为对应目标框目标的embedding信息。

相关方法、参数等说明如下：

* `create_model`实例化联合检测嵌入模型，具体说明如下：
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
<td>无</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>模型存储路径</td>
<td><code>str</code></td>
<td>无</td>
<td>无</td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理</td>
<td><code>bool</code></td>
<td>无</td>
<td><code>False</code></td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 调用联合检测嵌入模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input`，具体说明如下：

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
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
<li><b>List</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
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
</table>

* 此外，也支持通过属性获取预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">获取预测的<code>json</code>格式的结果</td>
</tr>
</table>

关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的联合检测嵌入模型。在使用 PaddleX 开发联合检测嵌入模型之前，请务必安装 PaddleX的目标检测相关模型训练插件，安装过程可以参考 [PaddleX本地安装教程](../../../installation/installation.md)

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/mot_examples.tar -P ./dataset
tar -xf ./dataset/mot_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/joint_detection_embeding/FairMOT-DLA-34_1088x608.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/mot_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>
<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_num_classes": 1,
    "train_num_identities": 62,
    "train_samples": 600,
    "train_sample_paths": [
      "check_dataset\/demo_img\/train\/000066.jpg",
      "check_dataset\/demo_img\/train\/000202.jpg",
      "check_dataset\/demo_img\/train\/000080.jpg",
      "check_dataset\/demo_img\/train\/000251.jpg",
      "check_dataset\/demo_img\/train\/000392.jpg",
      "check_dataset\/demo_img\/train\/000126.jpg",
      "check_dataset\/demo_img\/train\/000335.jpg",
      "check_dataset\/demo_img\/train\/000171.jpg",
      "check_dataset\/demo_img\/train\/000360.jpg",
      "check_dataset\/demo_img\/train\/000201.jpg"
    ],
    "val_num_classes": 1,
    "val_num_identities": 29,
    "val_samples": 525,
    "val_sample_paths": [
      "check_dataset\/demo_img\/val\/000161.jpg",
      "check_dataset\/demo_img\/val\/000026.jpg",
      "check_dataset\/demo_img\/val\/000206.jpg",
      "check_dataset\/demo_img\/val\/000353.jpg",
      "check_dataset\/demo_img\/val\/000153.jpg",
      "check_dataset\/demo_img\/val\/000022.jpg",
      "check_dataset\/demo_img\/val\/000055.jpg",
      "check_dataset\/demo_img\/val\/000127.jpg",
      "check_dataset\/demo_img\/val\/000247.jpg",
      "check_dataset\/demo_img\/val\/000332.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "mot_examples",
  "show_type": "image",
  "dataset_type": "MOTDataset"
}
</code></pre>
<p>上述校验结果中，check_pass 为 true 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.train_num_classes</code>：训练数据集类别数为 1；</li>
<li><code>attributes.train_num_identities</code>：训练数据集不同实例总数为 62；</li>
<li><code>attributes.train_samples</code>：训练数据集样本数量为 600；</li>
<li><code>attributes.val_num_classes</code>：验证数据集类别数为 1；</li>
<li><code>attributes.val_num_identities</code>：验证数据集不同实例总数为 29；</li>
<li><code>attributes.val_samples</code>：验证数据集样本数量为 525；</li>
</ul>
<p>另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/jde/01.png"/></p></details>


### 4.2 模型训练
一条命令即可完成模型的训练:

```bash
python main.py -c paddlex/configs/modules/joint_detection_embeding/FairMOT-DLA-34_1088x608.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/mot_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`FairMOT-DLA-34_1088x608.yaml`，训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
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
python main.py -c paddlex/configs/modules/joint_detection_embeding/FairMOT-DLA-34_1088x608.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/mot_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`FairMOT-DLA-34_1088x608.yaml.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>
<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>。</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json，其记录了</code>评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 MOTA；</p></details>

### <b>4.4 模型推理和模型集成</b>
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理

* 通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/joint_detection_embedding_000.png)到本地。
```bash
python main.py -c paddlex/configs/modules/joint_detection_embeding/FairMOT-DLA-34_1088x608.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="./joint_detection_embedding_000.png"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`FairMOT-DLA-34_1088x608.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

联合检测嵌入模块可以集成的PaddleX产线有[多目标跟踪产线](../../../pipeline_usage/tutorials/cv_pipelines/multiobject_tracking.md)，只需要替换模型路径即可完成相关产线的联合检测嵌入模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.<b>模块集成</b>

您产出的权重可以直接集成到联合检测嵌入模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。

您也可以利用 PaddleX 高性能推理插件来优化您模型的推理过程，进一步提升效率，详细的流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。
