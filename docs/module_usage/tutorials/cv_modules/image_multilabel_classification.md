---
comments: true
---

# 图像多标签分类模块使用教程

## 一、概述
图像多标签分类模块是计算机视觉系统中的重要组件，负责对输入的图像进行多标签的分类。与传统的图像分类任务只将图像分到一个类别不同，图像多标签分类任务需要将图像分到多个相关的类别。该模块的性能直接影响到整个计算机视觉系统的准确性和效率。图像多标签分类模块通常会接收图像作为输入，然后通过深度学习或其他机器学习算法，根据图像的特性和内容，将其分类到多个预定义的类别中。例如，对于一张包含猫和狗的图像，图像多标签分类模块可能需要将其同时标记为“猫”和“狗”。这些分类标签将作为输出，供其他模块或系统进行后续的处理和分析。

## 二、支持模型列表


<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(%)</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>CLIP_vit_base_patch16_448_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/CLIP_vit_base_patch16_448_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CLIP_vit_base_patch16_448_ML_pretrained.pdparams">训练模型</a></td>
<td>89.15</td>
<td>325.6 M</td>
<td>CLIP_ML是一种基于CLIP的图像多标签分类模型，通过结合ML-Decoder，显著提升了模型在图像多标签分类任务上的准确性。</td>
</tr>
<tr>
<td>PP-HGNetV2-B0_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B0_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B0_ML_pretrained.pdparams">训练模型</a></td>
<td>80.98</td>
<td>39.6 M</td>
<td rowspan="3">PP-HGNetV2_ML是一种基于PP-HGNetV2的图像多标签分类模型，通过结合ML-Decoder，显著提升了模型在图像多标签分类任务上的准确性。</td>
</tr>
<tr>
<td>PP-HGNetV2-B4_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B4_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B4_ML_pretrained.pdparams">训练模型</a></td>
<td>87.96</td>
<td>88.5 M</td>
</tr>
<tr>
<td>PP-HGNetV2-B6_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-HGNetV2-B6_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-HGNetV2-B6_ML_pretrained.pdparams">训练模型</a></td>
<td>91.25</td>
<td>286.5 M</td>
</tr>
<tr>
<td>PP-LCNet_x1_0_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LCNet_x1_0_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_ML_pretrained.pdparams">训练模型</a></td>
<td>77.96</td>
<td>29.4 M</td>
<td>PP-LCNet_ML是一种基于PP-LCNet的图像多标签分类模型，通过结合ML-Decoder，显著提升了模型在图像多标签分类任务上的准确性。</td>
</tr>
<tr>
<td>ResNet50_ML</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/ResNet50_ML_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_ML_pretrained.pdparams">训练模型</a></td>
<td>83.50</td>
<td>108.9 M</td>
<td>ResNet50_ML是一种基于ResNet50的图像多标签分类模型，通过结合ML-Decoder，显著提升了模型在图像多标签分类任务上的准确性。</td>
</tr>
</table>


<b>注：以上精度指标为[COCO2017](https://cocodataset.org/#home)的多标签分类任务mAP。</b>

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

wheel 包的安装后，几行代码即可完成图像多标签分类模块的推理，可以任意切换该模块下的模型，您也可以将图像多标签分类的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/multilabel_classification_005.png)到本地。

```python
from paddlex import create_model
model = create_model(model_name="PP-LCNet_x1_0_ML")
output = model.predict(input="multilabel_classification_005.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
运行后，得到的结果为：
```bash
{'res': {'input_path': 'multilabel_classification_005.png', 'class_ids': [46, 49, 47, 45, 60, 43, 39], 'scores': [0.99972, 0.99601, 0.99277, 0.65718, 0.56914, 0.56436, 0.52865], 'label_names': ['banana', 'orange', 'apple', 'bowl', 'dining table', 'knife', 'bottle']}}
```

运行结果参数含义如下：
- `input_path`：表示输入待预测多类别图像的路径
- `class_ids`：表示多类别图像的预测标签ID
- `scores`：表示多类别图像的预测标签置信度
- `label_names`：表示多类别图像的预测标签名称


可视化图片如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/image_multilabel_classification/multilabel_classification_005_result.png">

相关方法、参数等说明如下：

* `create_model`实例化多标签分类模型（此处以`PP-LCNet_x1_0_ML`为例），具体说明如下：
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
<td><code>PP-LCNet_x1_0_ML</code></td>
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
<td>多标签分类阈值</td>
<td><code>float/list/dict</code></td>
<td><li><b>float类型变量</b>，任意[0-1]之间浮点数：<code>0.5</code></li>
<li><b>list类型变量</b>，由多个[0-1]之间浮点数组成的列表：<code>[0.5,0.5,...]</code></li>
<li><b>dict类型变量</b>,指定不同类别使用不同的阈值，其中"default"为必须包含的键:<code>{"default":0.5,1:0.1,...}</code>
</li>
</td>
<td>0.5</td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 其中，`threshold` 参数用于设置多标签分类的阈值，默认为0.5。当设置为浮点数时，表示所有类别均使用该阈值；当设置为列表时，表示不同类别使用不同的阈值,此时需保持列表长度与类别数量一致；当设置为字典时，`default` 为必须包含的键， 表示所有类别的默认阈值，其它类别使用各自的阈值。例如：{"default":0.5,1:0.1}。

* 调用多标签分类模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input` , `batch_size` 和  `threshold`，具体说明如下：

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
  <li><b>URL链接</b>，如图像文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/multilabel_classification_005.png">示例</a></li>
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
<tr>
<td><code>threshold</code></td>
<td>多标签分类阈值</td>
<td><code>float/list/dict</code></td>
<td><li><b>float类型变量</b>，任意[0-1]之间浮点数：<code>0.5</code></li>
<li><b>list类型变量</b>，由多个[0-1]之间浮点数组成的列表：<code>[0.5,0.5,...]</code></li>
<li><b>dict类型变量</b>,指定不同类别使用不同的阈值，其中"default"为必须包含的键:<code>{"default":0.5,1:0.1,...}</code>
</li>
</td>
<td>0.5</td>
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


关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的图像多标签分类模型。在使用 PaddleX 开发图像多标签分类模型之前，请务必安装 PaddleX 的 图像分类 相关模型训练插件，安装过程可以参考 [PaddleX本地安装教程](../../../installation/installation.md)

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX多标签分类任务模块数据标注教程](../../../data_annotations/cv_modules/ml_classification.md)

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/mlcls_nus_examples.tar -P ./dataset
tar -xf ./dataset/mlcls_nus_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/image_multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/mlcls_nus_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在 log 中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>

<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;label_file&quot;: &quot;../../dataset/mlcls_nus_examples/label.txt&quot;,
    &quot;num_classes&quot;: 33,
    &quot;train_samples&quot;: 17463,
    &quot;train_sample_paths&quot;: [
      &quot;check_dataset/demo_img/0543_4338693.jpg&quot;,
      &quot;check_dataset/demo_img/0272_347806939.jpg&quot;,
      &quot;check_dataset/demo_img/0069_2291994812.jpg&quot;,
      &quot;check_dataset/demo_img/0012_1222850604.jpg&quot;,
      &quot;check_dataset/demo_img/0238_53773041.jpg&quot;,
      &quot;check_dataset/demo_img/0373_541261977.jpg&quot;,
      &quot;check_dataset/demo_img/0567_519506868.jpg&quot;,
      &quot;check_dataset/demo_img/0023_289621557.jpg&quot;,
      &quot;check_dataset/demo_img/0581_484524659.jpg&quot;,
      &quot;check_dataset/demo_img/0325_120753036.jpg&quot;
    ],
    &quot;val_samples&quot;: 17463,
    &quot;val_sample_paths&quot;: [
      &quot;check_dataset/demo_img/0546_130758157.jpg&quot;,
      &quot;check_dataset/demo_img/0284_2230710138.jpg&quot;,
      &quot;check_dataset/demo_img/0090_1491261559.jpg&quot;,
      &quot;check_dataset/demo_img/0013_392798436.jpg&quot;,
      &quot;check_dataset/demo_img/0246_2248376356.jpg&quot;,
      &quot;check_dataset/demo_img/0377_1349296474.jpg&quot;,
      &quot;check_dataset/demo_img/0570_2457645006.jpg&quot;,
      &quot;check_dataset/demo_img/0027_309333946.jpg&quot;,
      &quot;check_dataset/demo_img/0584_132639537.jpg&quot;,
      &quot;check_dataset/demo_img/0329_206031527.jpg&quot;
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/mlcls_nus_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;MLClsDataset&quot;
}
</code></pre>
<p>上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.num_classes</code>：该数据集类别数为 33；</li>
<li><code>attributes.train_samples</code>：该数据集训练集样本数量为 17463；</li>
<li><code>attributes.val_samples</code>：该数据集验证集样本数量为 17463；</li>
<li><code>attributes.train_sample_paths</code>：该数据集训练集样本可视化图片相对路径列表；</li>
<li><code>attributes.val_sample_paths</code>：该数据集验证集样本可视化图片相对路径列表；</li>
</ul>
<p>另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/ml_classification/01.png"></p></details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过<b>修改配置文件</b>或是<b>追加超参数</b>的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>

<p><b>（1）数据集格式转换</b></p>
<p>图像多标签分类支持 <code>COCO</code>格式的数据集转换为 <code>MLClsDataset</code>格式，数据集格式转换的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: 是否进行数据集格式转换，图像多标签分类支持 <code>COCO</code>格式的数据集转换为 <code>MLClsDataset</code>格式，默认为 <code>False</code>;</li>
<li><code>src_dataset_type</code>: 如果进行数据集格式转换，则需设置源数据集格式，默认为 <code>null</code>，可选值为 <code>COCO</code> ；</li>
</ul>
<p>例如，您想将<code>COCO</code>格式的数据集转换为 <code>MLClsDataset</code>格式，则需将配置文件修改为：</p>
<pre><code class="language-bash">cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_coco_examples.tar -P ./dataset
tar -xf ./dataset/det_coco_examples.tar -C ./dataset/
</code></pre>
<pre><code class="language-bash">......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: COCO
  ......
</code></pre>
<p>随后执行命令：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples
</code></pre>
<p>数据转换执行之后，原有标注文件会被在原路径下重命名为 <code>xxx.bak</code>。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=COCO
</code></pre>
<p><b>（2）数据集划分</b></p>
<p>数据集划分的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: 是否进行重新划分数据集，为 <code>True</code> 时进行数据集格式转换，默认为 <code>False</code>；</li>
<li><code>train_percent</code>: 如果重新划分数据集，则需要设置训练集的百分比，类型为 0-100 之间的任意整数，需要保证和 <code>val_percent</code> 值加和为 100；</li>
<li><code>val_percent</code>: 如果重新划分数据集，则需要设置验证集的百分比，类型为 0-100 之间的任意整数，需要保证和 <code>train_percent</code> 值加和为 100；</li>
</ul>
<p>例如，您想重新划分数据集为 训练集占比 90%、验证集占比 10%，则需将配置文件修改为：</p>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples
</code></pre>
<p>数据划分执行之后，原有标注文件会被在原路径下重命名为 <code>xxx.bak</code>。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处图像多标签分类模型 PP-LCNet_x1_0_ML 的训练为例：

```bash
python main.py -c paddlex/configs/modules/image_multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/mlcls_nus_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-LCNet_x1_0_ML.yaml`，训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
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
python main.py -c paddlex/configs/modules/image_multilabel_classification/PP-LCNet_x1_0_ML.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/mlcls_nus_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-LCNet_x1_0_ML.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>。</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json，其记录了</code>评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包括 MultiLabelMAP；</p></details>

### <b>4.4 模型推理和模型集成</b>
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理

* 通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/multilabel_classification_005.png)到本地。
```bash
python main.py -c paddlex/configs/modules/image_multilabel_classification/PP-LCNet_x1_0_ML.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="multilabel_classification_005.png"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-LCNet_x1_0_ML.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

图像多标签分类模块可以集成的PaddleX产线有[通用图像多标签分类产线](../../../pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.md)，只需要替换模型路径即可完成相关产线的图像多标签分类模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.<b>模块集成</b>

您产出的权重可以直接集成到图像多标签分类模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。

您也可以利用 PaddleX 高性能推理插件来优化您模型的推理过程，进一步提升效率，详细的流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。
