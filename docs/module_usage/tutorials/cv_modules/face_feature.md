---
comments: true
---

# 人脸特征模块使用教程

## 一、概述
人脸特征模块通常以经过检测提取和关键点矫正处理的标准化人脸图像作为输入，从这些图像中提取具有高度辨识性的人脸特征，以便供后续模块使用，如人脸匹配和验证等任务。

## 二、支持模型列表

<details><summary> 👉模型列表详情</summary>

<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>输出特征维度</th>
<th>Acc (%)<br/>AgeDB-30/CFP-FP/LFW</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>MobileFaceNet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/MobileFaceNet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileFaceNet_pretrained.pdparams">训练模型</a></td>
<td>128</td>
<td>96.28/96.71/99.58</td>
<td>5.7</td>
<td>101.6</td>
<td>4.1</td>
<td>基于MobileFaceNet在MS1Mv3数据集上训练的人脸特征提取模型</td>
</tr>
<tr>
<td>ResNet50_face</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/ResNet50_face_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_face_pretrained.pdparams">训练模型</a></td>
<td>512</td>
<td>98.12/98.56/99.77</td>
<td>8.7</td>
<td>200.7</td>
<td>87.2</td>
<td>基于ResNet50在MS1Mv3数据集上训练的人脸特征提取模型</td>
</tr>
</tbody>
</table>
<p>注：以上精度指标是分别在AgeDB-30、CFP-FP和LFW数据集上测得的Accuracy。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</p></details>

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成whl包的安装后，几行代码即可完成人脸特征模块的推理，可以任意切换该模块下的模型，您也可以将人脸特征模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_recognition_001.jpg)到本地。

```python
from paddlex import create_model

model_name = "MobileFaceNet"

model = create_model(model_name)
output = model.predict("face_recognition_001.jpg", batch_size=1)

for res in output:
    res.print(json_format=False)
    res.save_to_json("./output/res.json")
```

<details><summary>👉 <b>运行后，得到的结果为：（点击展开）</b></summary>

```bash
{'res': {'input_path': 'face_recognition_001.jpg', 'feature': [0.04121152311563492, 0.0010890548583120108, -0.03561094403266907, 0.05722084641456604, 0.05919725075364113, -0.007132374215871096, -0.061298906803131104, -0.10843975096940994, -0.02871585637331009, 0.03347175195813179, 0.13309064507484436, 0.05309445410966873, 0.004820522852241993, -0.11700531840324402, 0.03240801766514778, 0.0639009103178978, 0.17841649055480957, 0.006999856326729059, -0.052513156086206436, 0.14528249204158783, 0.013314608484506607, -0.04820159450173378, -0.04795005917549133, 0.184268519282341, -0.15508289635181427, -0.01048946287482977, -0.103487029671669, 0.020606128498911858, 0.11970002949237823, 0.07393684983253479, -0.05581602826714516, -0.10253427177667618, -0.015256273560225964, 0.06347685307264328, 0.0893929973244667, -0.01050905603915453, -0.025690989568829536, -0.10570172965526581, -0.11608698219060898, -0.04072513058781624, 0.05093423277139664, 0.044215817004442215, 0.1629297435283661, -0.06339056044816971, -0.07671815156936646, 0.09480706602334976, -0.15456975996494293, -0.021657753735780716, 0.12482058256864548, -0.1267298310995102, 0.002465370809659362, -0.05374367907643318, -0.07079283148050308, 0.1325870305299759, -0.006946612149477005, 0.047657083719968796, 0.06102422997355461, -0.18113569915294647, -0.15677541494369507, -0.05817852169275284, -0.007711497135460377, -0.03407919406890869, 0.04798268899321556, -0.036309171468019485, 0.10679583996534348, -0.1858624368906021, -0.06799137592315674, 0.008694482035934925, 0.026530278846621513, -0.06917411088943481, 0.13533912599086761, -0.08762945234775543, -0.17223820090293884, -0.024798616766929626, -0.03390877693891525, -0.17003266513347626, -0.08045653998851776, -0.21928688883781433, -0.08328460901975632, 0.0745469480752945, -0.05523530766367912, -0.08471746742725372, -0.06595447659492493, 0.11475134640932083, 0.12401033192873001, 0.09317877888679504, -0.08352484554052353, 0.0247682835906744, 0.0008310621487908065, -0.09977596998214722, -0.002699024509638548, -0.23338164389133453, -0.1783595234155655, -0.08259879052639008, 0.14328709244728088, 0.024862702935934067, 0.008164866827428341, 0.06340813636779785, 0.1028614193201065, -0.038397643715143204, -0.05210508778691292, 0.0389365553855896, 0.12757952511310577, 0.05326246842741966, 0.06695418804883957, -0.0052042435854673386, 0.035264499485492706, 0.00990584772080183, -0.05249840393662453, 0.06972697377204895, -0.06477969884872437, -0.003332878928631544, 0.0449349470436573, 0.020609190687537193, 0.074540875852108, -0.03608720749616623, 0.04876900464296341, -0.06063542515039444, -0.07829384505748749, -0.1220116913318634, 0.05064597725868225, 0.07839702069759369, 0.06130668520927429, -0.13095220923423767, 0.0888662114739418, -0.029464716091752052, 0.030264943838119507, 0.04124804586172104]}}
```

参数含义如下：
- `input_path`：输入的待预测图像的路径
- `feature`：表示特征向量的浮点数列表，长度为128

</details>

相关方法、参数等说明如下：

* `create_model`实例化人脸特征模型（此处以`MobileFaceNet`为例），具体说明如下：
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
<td>是否进行反转推理； 如果为True，模型会对输入图像水平翻转后再次推理，并融合两次推理结果以提升人脸特征的准确性</td>
<td><code>bool</code></td>
<td>无</td>
<td><code>False</code></td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 调用人脸特征模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input`和`batch_size`具体说明如下：

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
<td><code>Python Var</code>/<code>str</code>/<code>dict</code>/<code>list</code></td>
<td>
<ul>
  <li><b>Python变量</b>，如<code>numpy.ndarray</code>表示的图像数据</li>
  <li><b>文件路径</b>，如图像文件的本地路径：<code>/root/data/img.jpg</code></li>
  <li><b>URL链接</b>，如图像文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">示例</a></li>
  <li><b>本地目录</b>，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code></li>
  <li><b>字典</b>，字典的<code>key</code>需与具体任务对应，如图像分类任务对应<code>\"img\"</code>，字典的<code>val</code>支持上述类型数据，例如：<code>{\"img\": \"/root/data1\"}</code></li>
  <li><b>列表</b>，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code>，<code>[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]</code></li>
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

* 对预测结果进行处理，每个样本的预测结果均为`dict`类型，且支持打印、保存为图片、保存为`json`文件的操作:

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
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">获取预测的<code>json</code>格式的结果</td>
</tr>

</table>

关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。
## 四、二次开发
如果你追求更高精度的现有模型，可以使用PaddleX的二次开发能力，开发更好的人脸特征模型。在使用PaddleX开发人脸特征模型之前，请务必安装PaddleX的PaddleClas插件，安装过程可以参考 [PaddleX本地安装教程](../../../installation/installation.md)

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX为每一个模块都提供了demo数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，人脸特征模块的训练数据集采取通用图像分类数据集格式组织，可以参考[PaddleX图像分类任务模块数据标注教程](../../../data_annotations/cv_modules/image_classification.md)。若您希望用私有数据集进行后续的模型评估，请注意人脸特征模块的验证数据集格式与训练数据集的方式有所不同，请参考[4.1.4节 人脸特征模块数据集组织方式](#414-人脸特征模块数据集组织方式)

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/face_rec_examples.tar -P ./dataset
tar -xf ./dataset/face_rec_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/face_feature/MobileFaceNet.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/face_rec_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>

<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;train_label_file&quot;: &quot;../../dataset/face_rec_examples/train/label.txt&quot;,
    &quot;train_num_classes&quot;: 995,
    &quot;train_samples&quot;: 1000,
    &quot;train_sample_paths&quot;: [
      &quot;check_dataset/demo_img/01378592.jpg&quot;,
      &quot;check_dataset/demo_img/04331410.jpg&quot;,
      &quot;check_dataset/demo_img/03485713.jpg&quot;,
      &quot;check_dataset/demo_img/02382123.jpg&quot;,
      &quot;check_dataset/demo_img/01722397.jpg&quot;,
      &quot;check_dataset/demo_img/02682349.jpg&quot;,
      &quot;check_dataset/demo_img/00272794.jpg&quot;,
      &quot;check_dataset/demo_img/03151987.jpg&quot;,
      &quot;check_dataset/demo_img/01725764.jpg&quot;,
      &quot;check_dataset/demo_img/02580369.jpg&quot;
    ],
    &quot;val_label_file&quot;: &quot;../../dataset/face_rec_examples/val/pair_label.txt&quot;,
    &quot;val_num_classes&quot;: 2,
    &quot;val_samples&quot;: 500,
    &quot;val_sample_paths&quot;: [
      &quot;check_dataset/demo_img/Don_Carcieri_0001.jpg&quot;,
      &quot;check_dataset/demo_img/Eric_Fehr_0001.jpg&quot;,
      &quot;check_dataset/demo_img/Harry_Kalas_0001.jpg&quot;,
      &quot;check_dataset/demo_img/Francis_Ford_Coppola_0001.jpg&quot;,
      &quot;check_dataset/demo_img/Amer_al-Saadi_0001.jpg&quot;,
      &quot;check_dataset/demo_img/Sergei_Ivanov_0001.jpg&quot;,
      &quot;check_dataset/demo_img/Erin_Runnion_0003.jpg&quot;,
      &quot;check_dataset/demo_img/Bill_Stapleton_0001.jpg&quot;,
      &quot;check_dataset/demo_img/Daniel_Bruehl_0001.jpg&quot;,
      &quot;check_dataset/demo_img/Clare_Short_0004.jpg&quot;
    ]
  },
  &quot;analysis&quot;: {},
  &quot;dataset_path&quot;: &quot;./dataset/face_rec_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;ClsDataset&quot;
}
</code></pre>
<p>上述校验结果中，<code>check_pass</code> 为 <code>True</code> 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.train_num_classes</code>：该数据集训练类别数为 995；</li>
<li><code>attributes.val_num_classes</code>：该数据集验证类别数为 2；</li>
<li><code>attributes.train_samples</code>：该数据集训练集样本数量为 1000；</li>
<li><code>attributes.val_samples</code>：该数据集验证集样本数量为 500；</li>
<li><code>attributes.train_sample_paths</code>：该数据集训练集样本可视化图片相对路径列表；</li>
</ul></details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过<b>修改配置文件</b>或是<b>追加超参数</b>的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>

<p>人脸特征模块不支持数据格式转换与数据集划分。</p></details>

#### 4.1.4 人脸特征模块数据集组织方式

人脸特征模块验证数据集与训练数据集格式不同，若需要在私有数据上训练模型和评估模型精度，请按照如下方式组织自己的数据集：

```bash
face_rec_dataroot      # 数据集根目录，目录名称可以改变
├── train              # 训练数据集的保存目录，目录名称不可以改变
   ├── images          # 图像的保存目录，目录名称可以改变，但要注意与label.txt中的内容对应
      ├── xxx.jpg      # 人脸图像文件
      ├── xxx.jpg      # 人脸图像文件
      ...
   └── label.txt       # 训练集标注文件，文件名称不可改变。每行给出图像相对`train`的路径和人脸图像类别（人脸身份）id，使用空格分隔，内容举例：images/image_06765.jpg 0
├── val                # 验证数据集的保存目录，目录名称不可以改变
   ├── images          # 图像的保存目录，目录名称可以改变，但要注意与pari_label.txt中的内容对应
      ├── xxx.jpg      # 人脸图像文件
      ├── xxx.jpg      # 人脸图像文件
      ...
   └── pair_label.txt  # 验证数据集标注文件，文件名称不可改变。每行给出两个要比对的人脸图像路径和一个表示该对图像是否属于同一个人的0、1标签，使用空格分隔。
```

验证集标注文件`pair_label.txt`的内容示例:

```bash
# 人脸图像1.jpg 人脸图像2.jpg 标签(0表示该行的两个人脸图像文件不属于同一个人，1表示属于同一个人)
images/Angela_Merkel_0001.jpg images/Angela_Merkel_0002.jpg 1
images/Bruce_Gebhardt_0001.jpg images/Masao_Azuma_0001.jpg 0
images/Francis_Ford_Coppola_0001.jpg images/Francis_Ford_Coppola_0002.jpg 1
images/Jason_Kidd_0006.jpg images/Jason_Kidd_0008.jpg 1
images/Miyako_Miyazaki_0002.jpg images/Munir_Akram_0002.jpg 0
```

### 4.2 模型训练
一条命令即可完成模型的训练，以此处MobileFaceNet的训练为例：

```bash
python main.py -c paddlex/configs/modules/face_feature/MobileFaceNet.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/face_rec_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`MobileFaceNet.yaml`）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅模型对应任务模块的配置文件说明[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<ul>
<li>模型训练过程中，PaddleX 会自动保存模型权重文件，默认为<code>output</code>，如需指定保存路径，可通过配置文件中 <code>-o Global.output</code> 字段进行设置。</li>
<li>PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。</li>
<li>
<p>训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅<a href="../../../support_list/models_list.md">PaddleX模型列表（CPU/GPU）</a>。
在完成模型训练后，所有产出保存在指定的输出目录（默认为<code>./output/</code>）下，通常有以下产出：</p>
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
python main.py -c paddlex/configs/modules/face_feature/MobileFaceNet.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/face_rec_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`MobileFaceNet.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=./output/best_model/best_model/model.pdparams</code>。</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json</code>，其记录了评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 Accuracy；</p></details>

### <b>4.4 模型推理</b>
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测。在PaddleX中实现模型推理预测可以通过两种方式：命令行和wheel 包。

#### 4.4.1 模型推理
* 通过命令行的方式进行推理预测，只需如下一条命令，运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/face_recognition_001.jpg)到本地。
```bash
python main.py -c paddlex/configs/modules/face_feature/MobileFaceNet.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="face_recognition_001.jpg"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`MobileFaceNet.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

人脸特征模块可以集成的PaddleX产线有[<b>人脸识别</b>](../../../pipeline_usage/tutorials/cv_pipelines/face_recognition.md)，只需要替换模型路径即可完成相关产线的人脸特征模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.<b>模块集成</b>

您产出的权重可以直接集成到人脸特征模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。
