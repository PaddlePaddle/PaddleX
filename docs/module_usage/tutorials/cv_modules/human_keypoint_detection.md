---
comments: true
---

# 人体关键点检测模块使用教程

## 一、概述
人体关键点检测是计算机视觉领域中的一个重要任务，旨在识别图像或视频中人体的特定关键点位置。通过检测这些关键点，可以实现姿态估计、动作识别、人机交互、动画生成等多种应用。人体关键点检测在增强现实、虚拟现实、运动捕捉等领域都有广泛的应用。

关键点检测算法主要包括 Top-Down 和 Bottom-Up 两种方案。Top-Down 方案通常依赖一个目标检测算法识别出感兴趣物体的边界框，关键点检测模型的输入为经过裁剪的单个目标，输出为这个目标的关键点预测结果，模型的准确率会更高，但速度会随着对象数量的增加而变慢。不同的是，Bottom-Up 方法不依赖于先进行目标检测，而是直接对整个图像进行关键点检测，然后对这些点进行分组或连接以形成多个姿势实例，其速度是固定的，不会随着物体数量的增加而变慢，但精度会更低。

## 二、支持模型列表

<table>
  <tr>
    <th >模型</th><th>模型下载链接</th>
    <th >方案</th>
    <th >输入尺寸</th>
    <th >AP(0.5:0.95)</th>
    <th >GPU推理耗时（ms）</th>
    <th >CPU推理耗时 (ms)</th>
    <th >模型存储大小（M）</th>
    <th >介绍</th>
  </tr>
  <tr>
    <td>PP-TinyPose_128x96</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TinyPose_128x96_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TinyPose_128x96_pretrained.pdparams">训练模型</a></td>
    <td>Top-Down</td>
    <td>128*96</td>
    <td>58.4</td>
    <td></td>
    <td></td>
    <td>4.9</td>
    <td rowspan="2">PP-TinyPose 是百度飞桨视觉团队自研的针对移动端设备优化的实时关键点检测模型，可流畅地在移动端设备上执行多人姿态估计任务</td>
  </tr>
  <tr>
    <td>PP-TinyPose_256x192</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-TinyPose_256x192_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-TinyPose_256x192_pretrained.pdparams">训练模型</a></td>
    <td>Top-Down</td>
    <td>256*192</td>
    <td>68.3</td>
    <td></td>
    <td></td>
    <td>4.9</td>
  </tr>
</table>

**注：以上精度指标为COCO数据集 AP(0.5:0.95)，所依赖的检测框为ground truth标注得到。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**


## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成wheel包的安装后，几行代码即可完成人体关键点检测模块的推理，可以任意切换该模块下的模型，您也可以将人体关键点检测模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/keypoint_detection_002.jpg)到本地。

```python
from paddlex import create_model

model = create_model(model_name="PP-TinyPose_128x96")
output = model.predict("keypoint_detection_002.jpg", batch_size=1)

for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

<details><summary>👉 运行后，得到的结果为：（点击展开）</summary>

```bash
{'res': {'input_path': 'keypoint_detection_002.jpg', 'kpts': [{'keypoints': array([[175.28381   ,  56.04361   ,   0.6522829 ],
       [181.32794   ,  49.64205   ,   0.7338211 ],
       [169.46002   ,  50.59111   ,   0.68370765],
       [193.34212   ,  51.919697  ,   0.86765444],
       [164.50787   ,  55.65199   ,   0.8232859 ],
       [219.72359   ,  90.2871    ,   0.8812915 ],
       [152.90378   ,  95.078064  ,   0.9093066 ],
       [233.10957   , 149.67049   ,   0.77069044],
       [139.55766   , 144.38327   ,   0.75550145],
       [245.2283    , 202.4244    ,   0.7065905 ],
       [117.837944  , 188.5641    ,   0.8892116 ],
       [203.29543   , 200.2967    ,   0.83833086],
       [172.00792   , 201.19939   ,   0.7636936 ],
       [181.18797   , 273.06693   ,   0.8719099 ],
       [185.175     , 278.47977   ,   0.687819  ],
       [171.55069   , 362.4273    ,   0.7994317 ],
       [201.69414   , 354.59534   ,   0.67892176]], dtype=float32), 'kpt_score': 0.7831442}]}}
```

参数含义如下：
- `input_path`：输入的待预测图像的路径
- `kpts`：预测的关键点信息，一个字典列表。每个字典包含以下信息：
  - `keypoints`：关键点坐标信息，一个numpy数组，形状为[num_keypoints, 3]，其中每个关键点由[x, y, score]组成，score为该关键点的置信度
  - `kpt_score`：关键点整体的置信度，即关键点的平均置信度

</details>

可视化图像如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/keypoint_det/keypoint_detection_002_res.jpg">

相关方法、参数等说明如下：

* `create_model`实例化人体关键点检测模型（此处以`PP-TinyPose_128x96`为例），具体说明如下：
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
<td><code>flip</code></td>
<td>是否进行图像水平反转推理结果融合； 如果为True，模型会对输入图像水平翻转后再次推理，并融合两次推理结果以增加关键点预测的准确性</td>
<td><code>bool</code></td>
<td>无</td>
<td><code>False</code></td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 调用人体关键点检测模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input`和`batch_size`，具体说明如下：

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
  <li><b>URL链接</b>，如图像文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">示例</a></li>
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
如果你追求更高精度的现有模型，可以使用PaddleX的二次开发能力，开发更好的关键点检测模型。在使用PaddleX开发关键点检测模型之前，请务必安装PaddleX的PaddleDetection插件，安装过程可以参考 [PaddleX本地安装教程](../../../installation/installation.md)。

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，**只有通过数据校验的数据才可以进行模型训练**。此外，PaddleX为每一个模块都提供了demo数据集，您可以基于官方提供的 Demo 数据完成后续的开发。
#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/keypoint_coco_examples.tar -P ./dataset
tar -xf ./dataset/keypoint_coco_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/keypoint_detection/PP-TinyPose_128x96.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/keypoint_coco_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details>
  <summary>👉 <b>校验结果详情（点击展开）</b></summary>


校验结果文件具体内容为：

```bash
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "num_classes": 1,
    "train_samples": 500,
    "train_sample_paths": [
      "check_dataset/demo_img/000000560108.jpg",
      "check_dataset/demo_img/000000434662.jpg",
      "check_dataset/demo_img/000000540556.jpg",
      ...
    ],
    "val_samples": 100,
    "val_sample_paths": [
      "check_dataset/demo_img/000000463730.jpg",
      "check_dataset/demo_img/000000085329.jpg",
      "check_dataset/demo_img/000000459153.jpg",
      ...
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "keypoint_coco_examples",
  "show_type": "image",
  "dataset_type": "KeypointTopDownCocoDetDataset"
}
```
上述校验结果中，`check_pass` 为 `True` 表示数据集格式符合要求，其他部分指标的说明如下：

* `attributes.num_classes`：该数据集类别数为 1；
* `attributes.train_samples`：该数据集训练集样本数量为500；
* `attributes.val_samples`：该数据集验证集样本数量为 100；
* `attributes.train_sample_paths`：该数据集训练集样本可视化图片相对路径列表；
* `attributes.val_sample_paths`：该数据集验证集样本可视化图片相对路径列表；


数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/keypoint_det/01.png)
</details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过**修改配置文件**或是**追加超参数**的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details>
  <summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>



**（1）数据集格式转换**

关键点检测不支持数据格式转换。

**（2）数据集划分**

数据集划分的参数可以通过修改配置文件中 `CheckDataset` 下的字段进行设置，配置文件中部分参数的示例说明如下：

* `CheckDataset`:
  * `split`:
    * `enable`: 是否进行重新划分数据集，为 `True` 时进行数据集格式转换，默认为 `False`；
    * `train_percent`: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证与 `val_percent` 的值之和为100；


例如，您想重新划分数据集为 训练集占比90%、验证集占比10%，则需将配置文件修改为：

```bash
......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
```
随后执行命令：

```bash
python main.py -c paddlex/configs/modules/keypoint_detection/PP-TinyPose_128x96.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/keypoint_coco_examples
```
数据划分执行之后，原有标注文件会被在原路径下重命名为 `xxx.bak`。

以上参数同样支持通过追加命令行参数的方式进行设置：

```bash
python main.py -c paddlex/configs/modules/keypoint_detection/PP-TinyPose_128x96.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/keypoint_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
```
</details>



### 4.2 模型训练
一条命令即可完成模型的训练，以此处`PP-TinyPose_128x96`的训练为例：

```bash
python main.py -c paddlex/configs/modules/keypoint_detection/PP-TinyPose_128x96.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/keypoint_coco_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-TinyPose_128x96.yaml`，训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅模型对应任务模块的配置文件说明[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details>
  <summary>👉 <b>更多说明（点击展开）</b></summary>


* 模型训练过程中，PaddleX 会自动保存模型权重文件，默认为`output`，如需指定保存路径，可通过配置文件中 `-o Global.output` 字段进行设置。
* PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。
* 在完成模型训练后，所有产出保存在指定的输出目录（默认为`./output/`）下，通常有以下产出：

* `train_result.json`：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；
* `train.log`：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；
* `config.yaml`：训练配置文件，记录了本次训练的超参数的配置；
* `.pdparams`、`.pdema`、`.pdopt.pdstate`、`.pdiparams`、`.pdmodel`：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；
</details>

## **4.3 模型评估**
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：

```bash
python main.py -c paddlex/configs/modules/keypoint_detection/PP-TinyPose_128x96.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/keypoint_coco_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-TinyPose_128x96.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details>
  <summary>👉 <b>更多说明（点击展开）</b></summary>


在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如`-o Evaluate.weight_path=./output/best_model/best_model/model.pdparams`。

在完成模型评估后，会产出`evaluate_result.json，其记录了`评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 AP；

</details>

### **4.4 模型推理**
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测。在PaddleX中实现模型推理预测可以通过两种方式：命令行和wheel 包。

#### 4.4.1 模型推理
* 通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/keypoint_detection_002.jpg)到本地。
```bash
python main.py -c paddlex/configs/modules/keypoint_detection/PP-TinyPose_128x96.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="keypoint_detection_002.jpg"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-TinyPose_128x96.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成

模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.**产线集成**

人体关键点检测模块可以集成的PaddleX产线有[**人体关键点检测**](../../../pipeline_usage/tutorials/cv_pipelines/human_keypoint_detection.md)，只需要替换模型路径即可完成相关产线的人体关键点检测模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.**模块集成**

您产出的权重可以直接集成到人体关键点检测模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。
