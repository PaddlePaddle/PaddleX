---
comments: true
---

# 视频分类模块使用教程

## 一、概述
视频分类模块是计算机视觉系统中的关键组成部分，负责对输入的视频进行分类。该模块的性能直接影响到整个计算机视觉系统的准确性和效率。视频分类模块通常会接收视频作为输入，然后通过深度学习或其他机器学习算法，根据视频的特性和内容，将其分类到预定义的类别中。例如，对于一个动作识别系统，视频分类模块可能需要将输入的视频分类为“攀绳下降”、“空中打鼓”、“回答问题”等类别。视频分类模块的分类结果将作为输出，供其他模块或系统使用。

## 二、支持模型列表


<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top1 Acc(%)</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-TSM-R50_8frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSM-R50_8frames_uniform_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSM-R50_8frames_uniform_pretrained.pdparams">训练模型</a></td>
<td>74.36</td>
<td>93.4 M</td>
<td rowspan="1">
PP-TSM是一种百度飞桨视觉团队自研的视频分类模型。该模型基于ResNet-50骨干网络进行优化，从数据增强、网络结构微调、训练策略、BN层优化、预训练模型选择、模型蒸馏等6个方面进行模型调优，在中心采样评估方式下，Kinetics-400上精度较原论文实现提升3.95个点
</td>
</tr>

<tr>
<td>PP-TSMv2-LCNetV2_8frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSMv2-LCNetV2_8frames_uniform_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSMv2-LCNetV2_8frames_uniform_pretrained.pdparams">训练模型</a></td>
<td>71.71</td>
<td>22.5 M</td>
<td rowspan="2">PP-TSMv2是轻量化的视频分类模型，基于CPU端模型PP-LCNetV2进行优化，从骨干网络与预训练模型选择、数据增强、tsm模块调优、输入帧数优化、解码速度优化、DML蒸馏、LTA模块等7个方面进行模型调优，在中心采样评估方式下，精度达到75.16%，输入10s视频在CPU端的推理速度仅需456ms。</td>
</tr>
<tr>
<td>PP-TSMv2-LCNetV2_16frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TSMv2-LCNetV2_16frames_uniform_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TSMv2-LCNetV2_16frames_uniform_pretrained.pdparams">训练模型</a></td>
<td>73.11</td>
<td>22.5 M</td>
</tr>

</table>



<p><b>注：以上精度指标为 <a href="https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/dataset/k400.md">K400</a> 验证集 Top1 Acc。</b></p></details>


## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)。

完成 wheel 包的安装后，几行代码即可完成视频分类模块的推理，可以任意切换该模块下的模型，您也可以将视频分类的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例视频](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4)到本地。

```python
from paddlex import create_model
model = create_model(model_name="PP-TSMv2-LCNetV2_8frames_uniform")
output = model.predict(input="general_video_classification_001.mp4", batch_size=1)
for res in output:
    res.print()
    res.save_to_video(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

运行后，得到的结果为：
```bash
{'res': {'input_path': 'general_video_classification_001.mp4', 'class_ids': array([0], dtype=int32), 'scores': [0.91996], 'label_names': ['abseiling']}}
```

参数含义如下：
- `input_path`：表示输入待预测视频的路径
- `class_ids`：表示视频的分类id
- `scores`：表示视频的分类分数
- `label_names`：表示视频的分类标签名称

可视化视频如下：
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/video_classification/general_video_classification_001.jpg">
上述Python脚本中，执行了如下几个步骤：
* `create_model`实例化视频分类模型（此处以`PP-TSMv2-LCNetV2_8frames_uniform`为例），具体说明如下：


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
<td>所有PaddleX支持的模型名称</td>
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
<td><code> topk</code></td>
<td>预测结果的前 <code>topk</code> 个类别和对应的分类概率；如果不指定，将默认使用PaddleX官方模型配置</td>
<td><code>int</code></td>
<td>无</td>
<td><code>1</code></td>
</tr>
</table>

* 调用视频分类模型的`predict`方法进行推理预测，`predict` 方法参数为`input`，用于输入待预测数据，支持多种输入类型，具体说明如下：

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
  <li><b>Python变量</b>，如<code>str</code>表示的视频文件的本地路径</li>
  <li><b>文件路径</b>，如视频文件的本地路径：<code>/root/data/video.mp4</code></li>
  <li><b>URL链接</b>，如视频文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4">示例</a></li>
  <li><b>本地目录</b>，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code></li>
  <li><b>列表</b>，列表元素需为上述类型数据，如 <code>[\"/root/data/video1.mp4\", \"/root/data/video2.mp4\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>批大小</td>
<td><code>int</code></td>
<td>无</td>
<td>1</td>
</tr>
<tr>
<td><code> topk</code></td>
<td>预测结果的前 <code>topk</code> 个类别和对应的分类概率；如果不指定，将默认使用 creat_model 指定的 <code>topk</code> 参数，如果creat_model 也没有指定， 则默认使用PaddleX官方模型配置</td>
<td><code>int</code></td>
<td>无</td>
<td><code>1</code></td>
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
<td>是否对输出内容进行使用<code>json</code>缩进格式化</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>json格式化设置，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>json格式化设置，仅当<code>format_json</code>为<code>True</code>时有效</td>
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
<td>json格式化设置</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>json格式化设置</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_video()</code></td>
<td>将结果保存为视频格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
</table>

* 此外，也支持通过属性获取结果可视化视频和`json`结果:

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
<td rowspan = "1"><code>video</code></td>
<td rowspan = "1">获取格式为<code>dict</code>的可视化视频和视频帧率。这里，可视化视频是np.array数组，维度是（视频帧数，视频高度，视频宽度，视频通道数）</td>
</tr>

</table>

关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的视频分类模型。在使用 PaddleX 开发视频分类模型之前，请务必安装 PaddleX 的 视频分类  [PaddleX本地安装教程](../../../installation/installation.md)中的二次开发部分。

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX视频分类任务模块数据标注教程](../../../data_annotations/video_modules/video_classification.md)

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/k400_examples.tar -P ./dataset
tar -xf ./dataset/k400_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/video_classification/PP-TSMv2-LCNetV2_8frames_uniform.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/k400_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>
<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "..\/..\/dataset\/k400_examples\/label.txt",
    "num_classes": 5,
    "train_samples": 250,
    "train_sample_paths": [
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/Wary2ON3aSo_000079_000089.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/_LHpfh0rXjk_000012_000022.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/dyoiNbn80q0_000039_000049.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/brBw6cFwock_000049_000059.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/-o4X5Z_Isyc_000085_000095.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/e24p-4W3TiU_000011_000021.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/2Grg_zwmYZE_000004_000014.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/aZY_0UqRNgA_000098_000108.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/WZlsi4nQHOo_000025_000035.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/rRh-lkFj4Tw_000001_000011.mp4"
    ],
    "val_samples": 50,
    "val_sample_paths": [
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/7Mga5kywfU4.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/w5UCdQ2NmfY.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/Qbo_tnzfjOY.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/LgW8pMDtylE.mkv",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/BY0883Dvt1c.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/PHQkMPu-KNo.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/7LSJ2Ryv1a8.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/oBYZWvlI8Uk.mp4",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/dpn2eg9O3Rs.mkv",
      "check_dataset\/..\/..\/dataset\/k400_examples\/videos\/hXtsZAaZ3yc.mkv"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "k400_examples",
  "show_type": "video",
  "dataset_type": "VideoClsDataset"
}
</code></pre>
<p>上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.num_classes</code>：该数据集类别数为 5；</li>
<li><code>attributes.train_samples</code>：该数据集训练集样本数量为 250；</li>
<li><code>attributes.val_samples</code>：该数据集验证集样本数量为 50；</li>
<li><code>attributes.train_sample_paths</code>：该数据集训练集样本可视化视频相对路径列表；</li>
<li><code>attributes.val_sample_paths</code>：该数据集验证集样本可视化视频相对路径列表；</li>
</ul>
<p>另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/video_classification/01.png"></p></details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过<b>修改配置文件</b>或是<b>追加超参数</b>的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>

<p><b>（1）数据集格式转换</b></p>
<p>视频分类暂不支持数据转换。</p>
<p><b>（2）数据集划分</b></p>
<p>数据集划分的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: 是否进行重新划分数据集，为 <code>True</code> 时进行数据集格式转换，默认为 <code>False</code>；</li>
<li><code>train_percent</code>: 如果重新划分数据集，则需要设置训练集的百分比，类型为 0-100 之间的任意整数，需要保证和 <code>val_percent</code> 值加和为100；</li>
</ul>
<p>例如，您想重新划分数据集为 训练集占比90%、验证集占比10%，则需将配置文件修改为：</p>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/video_classification/PP-TSMv2-LCNetV2_8frames_uniform.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/k400_examples
</code></pre>
<p>数据划分执行之后，原有标注文件会被在原路径下重命名为 <code>xxx.bak</code>。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/video_classification/PP-TSMv2-LCNetV2_8frames_uniform.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/k400_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处视频分类模型 PP-TSMv2-LCNetV2_8frames_uniform 的训练为例：

```
python main.py -c paddlex/configs/modules/video_classification/PP-TSMv2-LCNetV2_8frames_uniform.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/k400_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-TSMv2-LCNetV2_8frames_uniform.yaml`,训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
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
python main.py -c  paddlex/configs/modules/video_classification/PP-TSMv2-LCNetV2_8frames_uniform.yaml  \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/k400_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-TSMv2-LCNetV2_8frames_uniform.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>。</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json</code>，其记录了评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 val.top1、val.top5；</p></details>

### <b>4.4 模型推理和模型集成</b>

在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理
通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例视频](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4)到本地。

```bash
python main.py -c paddlex/configs/modules/video_classification/PP-TSMv2-LCNetV2_8frames_uniform.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="general_video_classification_001.mp4"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-TSMv2-LCNetV2_8frames_uniform.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

视频分类模块可以集成的 PaddleX 产线有[通用视频分类产线](../../../pipeline_usage/tutorials/video_pipelines/video_classification.md)，只需要替换模型路径即可完成相关产线的视频分类模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.<b>模块集成</b>

您产出的权重可以直接集成到视频分类模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。
