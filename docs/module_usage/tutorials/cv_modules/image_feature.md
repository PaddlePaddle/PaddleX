---
comments: true
---

# 图像特征模块使用教程

## 一、概述
图像特征模块是计算机视觉中的一项重要任务之一，主要指的是通过深度学习方法自动从图像数据中提取有用的特征，以便于后续的图像检索任务。该模块的性能直接影响到后续任务的准确性和效率。在实际应用中，图像特征通常会输出一组特征向量，这些向量能够有效地表示图像的内容、结构、纹理等信息，并将作为输入传递给后续的检索模块进行处理。

## 二、支持模型列表


<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>recall@1 (%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-ShiTuV2_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-ShiTuV2_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_pretrained.pdparams">训练模型</a></td>
<td>84.2</td>
<td>3.91 / 1.06</td>
<td>6.82 / 2.89</td>
<td>16.3</td>
<td rowspan="3">PP-ShiTuV2是一个通用图像特征系统，由主体检测、特征提取、向量检索三个模块构成，这些模型是其中的特征提取模块的模型之一，可以根据系统的情况选择不同的模型。</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_base</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-ShiTuV2_rec_CLIP_vit_base_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_base_pretrained.pdparams">训练模型</a></td>
<td>88.69</td>
<td>12.57 / 11.62</td>
<td>67.09 / 67.09</td>
<td>306.6</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_large</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-ShiTuV2_rec_CLIP_vit_large_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_large_pretrained.pdparams">训练模型</a></td>
<td>91.03</td>
<td>49.85 / 49.85</td>
<td>229.14 / 229.14</td>
<td>1050</td>
</tr>
</table>

<strong>测试环境说明:</strong>

  <ul>
      <li><b>性能测试环境</b>
          <ul>
            <li><strong>测试数据集：</strong>PaddleX自建数据集。</li>
              <li><strong>硬件配置：</strong>
                  <ul>
                      <li>GPU：NVIDIA Tesla T4</li>
                      <li>CPU：Intel Xeon Gold 6271C @ 2.60GHz</li>
                  </ul>
              </li>
              <li><strong>软件环境：</strong>
                  <ul>
                      <li>Ubuntu 20.04 / CUDA 11.8 / cuDNN 8.9 / TensorRT 8.6.1.6</li>
                      <li>paddlepaddle 3.0.0 / paddlex 3.0.2</li>
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

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成 wheel 包的安装后，几行代码即可完成图像特征模块的推理，可以任意切换该模块下的模型，您也可以将图像特征的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_recognition_001.jpg)到本地。

```python
from paddlex import create_model

model_name = "PP-ShiTuV2_rec"
model = create_model(model_name)
output = model.predict("general_image_recognition_001.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_json("./output/res.json")
```
运行后，得到的结果为：

```bash
{'res': {'input_path': 'general_image_recognition_001.jpg', 'page_index': None, 'feature': array([ 0.04910231, ..., -0.07126085], dtype=float32)}}
```

参数含义如下：
- `input_path`：输入的待预测图像的路径
- `feature`：提取的图像特征向量，维度为模型输出特征维度，此处为512维。

相关方法、参数等说明如下：

* `create_model`实例化图像特征模型（此处以`PP-ShiTuV2_rec`为例），具体说明如下：
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
<td><code>device</code></td>
<td>模型推理设备</td>
<td><code>str</code></td>
<td>支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理插件</td>
<td><code>bool</code></td>
<td>无</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>高性能推理配置</td>
<td><code>dict</code> | <code>None</code></td>
<td>无</td>
<td><code>None</code></td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 调用图像特征模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input`和`batch_size`具体说明如下：

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
  <li><b>列表</b>，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code>/li>
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

## 四、二次开发
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的图像特征模型。在使用 PaddleX 开发图像特征模型之前，请务必安装 PaddleX的分类相关模型训练插件，安装过程可以参考 [PaddleX本地安装教程](../../../installation/installation.md)

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX图像特征任务模块数据标注教程](../../../data_annotations/cv_modules/image_feature.md)

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Inshop_examples.tar -P ./dataset
tar -xf ./dataset/Inshop_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>
<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "train_samples": 1000,
    "train_sample_paths": [
      "check_dataset/demo_img/05_1_front.jpg",
      "check_dataset/demo_img/02_1_front.jpg",
      "check_dataset/demo_img/02_3_back.jpg",
      "check_dataset/demo_img/04_3_back.jpg",
      "check_dataset/demo_img/04_2_side.jpg",
      "check_dataset/demo_img/12_1_front.jpg",
      "check_dataset/demo_img/07_2_side.jpg",
      "check_dataset/demo_img/04_7_additional.jpg",
      "check_dataset/demo_img/04_4_full.jpg",
      "check_dataset/demo_img/01_1_front.jpg"
    ],
    "gallery_samples": 110,
    "gallery_sample_paths": [
      "check_dataset/demo_img/06_2_side.jpg",
      "check_dataset/demo_img/01_4_full.jpg",
      "check_dataset/demo_img/04_7_additional.jpg",
      "check_dataset/demo_img/02_1_front.jpg",
      "check_dataset/demo_img/02_3_back.jpg",
      "check_dataset/demo_img/02_3_back.jpg",
      "check_dataset/demo_img/02_4_full.jpg",
      "check_dataset/demo_img/03_4_full.jpg",
      "check_dataset/demo_img/02_2_side.jpg",
      "check_dataset/demo_img/03_2_side.jpg"
    ],
    "query_samples": 125,
    "query_sample_paths": [
      "check_dataset/demo_img/08_7_additional.jpg",
      "check_dataset/demo_img/01_7_additional.jpg",
      "check_dataset/demo_img/02_4_full.jpg",
      "check_dataset/demo_img/04_4_full.jpg",
      "check_dataset/demo_img/09_7_additional.jpg",
      "check_dataset/demo_img/04_3_back.jpg",
      "check_dataset/demo_img/02_1_front.jpg",
      "check_dataset/demo_img/06_2_side.jpg",
      "check_dataset/demo_img/02_7_additional.jpg",
      "check_dataset/demo_img/02_2_side.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/Inshop_examples",
  "show_type": "image",
  "dataset_type": "ShiTuRecDataset"
}
</code></pre>
<p>上述校验结果中，check_pass  为 true 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.train_samples</code>：该数据集训练样本数量为 1000；</li>
<li><code>attributes.gallery_samples</code>：该数据集被查询样本数量为 110；</li>
<li><code>attributes.query_samples</code>：该数据集查询样本数量为 125；</li>
<li><code>attributes.train_sample_paths</code>：该数据集训练样本可视化图片相对路径列表；</li>
<li><code>attributes.gallery_sample_paths</code>：该数据集被查询样本可视化图片相对路径列表；</li>
<li><code>attributes.query_sample_paths</code>：该数据集查询样本可视化图片相对路径列表；
另外，数据集校验还对数据集中图像数量和图像类别情况进行了分析，并绘制了分布直方图（histogram.png）：</li>
</ul>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/img_recognition/01.png"/></p></details>

### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过<b>修改配置文件</b>或是<b>追加超参数</b>的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>
<p><b>（1）数据集格式转换</b></p>
<p>图像特征任务支持 <code>LabelMe</code>格式的数据集转换为 <code>ShiTuRecDataset</code>格式，数据集格式转换的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: 是否进行数据集格式转换，图像特征任务支持 <code>LabelMe</code>格式的数据集转换为 <code>ShiTuRecDataset</code>格式，默认为 <code>False</code>;</li>
<li><code>src_dataset_type</code>: 如果进行数据集格式转换，则需设置源数据集格式，默认为 <code>null</code>，可选值为 <code>LabelMe</code> ；
例如，您想将<code>LabelMe</code>格式的数据集转换为 <code>ShiTuRecDataset</code>格式，则需将配置文件修改为：</li>
</ul>
<pre><code class="language-bash">cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/image_classification_labelme_examples.tar -P ./dataset
tar -xf ./dataset/image_classification_labelme_examples.tar -C ./dataset/
</code></pre>
<pre><code class="language-bash">......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: LabelMe
  ......
</code></pre>
<p>随后执行命令：</p>
<pre><code class="language-bash">python main.py -c  paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/image_classification_labelme_examples
</code></pre>
<p>数据转换执行之后，原有标注文件会被在原路径下重命名为 <code>xxx.bak</code>。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/image_classification_labelme_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
</code></pre>
<p><b>（2）数据集划分</b></p>
<p>数据集划分的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: 是否进行重新划分数据集，为 <code>True</code> 时进行数据集格式转换，默认为 <code>False</code>；</li>
<li><code>train_percent</code>: 如果重新划分数据集，则需要设置训练集的百分比，类型为 0-100 之间的任意整数，需要保证和 <code>gallery_percent 、query_percent</code> 值加和为100；</li>
</ul>
<p>例如，您想重新划分数据集为 训练集占比70%、被查询数据集占比20%，查询数据集占比10%，则需将配置文件修改为：</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 70
    gallery_percent: 20
    query_percent: 10
  ......
</code></pre>
<p>随后执行命令：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples
</code></pre>
<p>数据划分执行之后，原有标注文件会被在原路径下重命名为 <code>xxx.bak</code>。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=70 \
    -o CheckDataset.split.gallery_percent=20 \
    -o CheckDataset.split.query_percent=10
</code></pre>
<blockquote>
<p>❗注意 ：由于图像特征模型评估的特殊性，当且仅当 train、query、gallery 集合属于同一类别体系下，数据切分才有意义，在图像特征模的评估过程中，必须满足 gallery 集合和 query 集合属于同一类别体系，其允许和 train 集合不在同一类别体系， 如果 gallery 集合和 query 集合与 train 集合不在同一类别体系，则数据划分后的评估没有意义，建议谨慎操作。</p>
</blockquote></details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处图像特征模型 PP-ShiTuV2_rec 的训练为例：

```
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-ShiTuV2_rec.yaml`，训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
* 其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅模型对应任务模块的配置文件说明[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)
* 新特性：Paddle 3.0 版本支持了 CINN 神经网络编译器，在使用 GPU 设备训练时，不同模型有不同程度的训练加速效果。在 PaddleX 中训练模型时，可通过指定参数 `-o Train.dy2st=True` 开启。

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
<li><code>.pdparams</code>、<code>.pdema</code>、<code>.pdopt.pdstate</code>、<code>.pdiparams</code>、<code>.json</code>：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；</li>
<li>【注意】：Paddle 3.0.0 对于静态图网络结构信息的存储格式，由protobuf（原<code>.pdmodel</code>后缀文件）升级为json（现<code>.json</code>后缀文件），以兼容PIR体系，并获得更好的灵活性与扩展性。</li>
</ul></details>

## <b>4.3 模型评估</b>
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：

```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-ShiTuV2_rec.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`.
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>
<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>。</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json，其记录了</code>评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 recall1、recall5、mAP；</p></details>

### <b>4.4 模型推理</b>
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行 Python 集成。

#### 4.4.1 模型推理
通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_recognition_001.jpg)到本地。

```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="general_image_recognition_001.jpg"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-ShiTuV2_rec.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`.
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

> ❗ 注意：图像特征模型的推理结果为一组向量，需要配合检索模块完成图像的识别。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

图像特征模块可以集成的 PaddleX 产线有<b>通用图像特征产线</b>(coming soon)，只需要替换模型路径即可完成相关产线的图像特征模块的模型更新。在产线集成中，你可以使用服务化部署来部署你得到的模型。

2.<b>模块集成</b>

您产出的权重可以直接集成到图像特征模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。

您也可以利用 PaddleX 高性能推理插件来优化您模型的推理过程，进一步提升效率，详细的流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。
