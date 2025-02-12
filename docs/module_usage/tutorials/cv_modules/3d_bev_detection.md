---
comments: true
---

# 3D多模态融合检测模块使用教程

## 一、概述
3D多模态融合检测模块是计算机视觉和自动驾驶领域关键组成部分，负责在图像或视频中定位和标记出包含特定目标的区域的3D坐标和检测框信息。该模块的性能直接影响到整个视觉或自动驾驶感知系统的准确性和效率。3D多模态融合检测模块通常会输出目标区域的3D边界框（Bounding Boxes），这些3D边界框将作为输入传递给目标识别模块进行后续处理。

## 二、支持模型列表

<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(%)</th>
<th>NDS</th>
<th>介绍</th>
</tr>
<tr>
<td>BEVFusion</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/BEVFusion_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BEVFusion_pretrained.pdparams">训练模型</a></td>
<td>53.9</td>
<td>60.9</td>

<td rowspan="2">BEVFusion是一种在BEV视角下的多模态融合模型，采用两个分支处理不同模态的数据，得到lidar和camera在BEV视角下的特征，camera分支采用LSS这种自底向上的方式来显式的生成图像BEV特征，lidar分支采用经典的点云检测网络，最后对两种模态的BEV特征进行对齐和融合，应用于检测head或分割head

</td>
</tr>
<tr>


</table>


<p><b>注：以上精度指标为<a href="https://www.nuscenes.org/nuscenes">nuscenes</a>验证集 mAP(0.5:0.95), NDS 60.9, 精度类型为 FP32。</b></p></details>


## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成 wheel 包的安装后，几行代码即可完成目标检测模块的推理，可以任意切换该模块下的模型，您也可以将3D多模态融合检测模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例输入](https://paddle-model-ecology.bj.bcebos.com/paddlex/det_3d/demo_det_3d/nuscenes_demo_infer.tar)到本地。

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="3d_bev_detection")
output = pipeline.predict("nuscenes_demo_infer.tar")

for res in output:
    res.print()  ## 打印预测的结构化输出
    res.save_to_json("./output/")  ## 保存结果到json文件
    res.visualize(save_path="./output/", show=True) ## 3d结果可视化，如果运行环境有图形界面设置show=True，否则设置为False
```

<b>注：</b>   
1、3d检测结果可视化需要先安装open3d包，安装命令如下：
```bash
pip install open3d
```
2、如果运行环境没有图形界面，则无法可视化，但不影响结果的保存，可以在支持图形界面的环境下运行脚本，对保存的结果进行可视化:
```bash
python paddlex/inference/models/3d_bev_detection/visualizer_3d.py --save_path="./output/"
```

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/images/pipelines/3d_bev_detection/02.png">


运行后，得到的结果为：
```bash
{"res":
  {
    'input_path': 'samples/LIDAR_TOP/n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984253447765.pcd.bin',
    'sample_id': 'b4ff30109dd14c89b24789dc5713cf8c',
    'input_img_paths': [
      'samples/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984253404844.jpg',
      'samples/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984253412460.jpg',
      'samples/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984253420339.jpg',
      'samples/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984253427893.jpg',
      'samples/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984253437525.jpg',
      'samples/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984253447423.jpg'
    ]
    "boxes_3d": [
        [
            14.5425386428833,
            22.142045974731445,
            -1.2903141975402832,
            1.8441576957702637,
            4.433370113372803,
            1.7367216348648071,
            6.367165565490723,
            0.0036598597653210163,
            -0.013568558730185032
        ]
    ],
    "labels_3d": [
        0
    ],
    "scores_3d": [
        0.9920279383659363
    ]
  }
}
```

运行结果参数含义如下：
- `input_path`：表示输入待预测样本的输入点云数据路径
- `sample_id`：表示输入待预测样本的输入样本的唯一标识符
- `input_img_paths`：表示输入待预测样本的输入图像数据路径
- `boxes_3d`：表示该3D样本的所有预测框信息, 每个预测框信息为一个长度为9的列表, 各元素分别表示：
  - 0: 中心点x坐标
  - 1: 中心点y坐标
  - 2: 中心点z坐标
  - 3: 检测框宽度
  - 4: 检测框长度
  - 5: 检测框高度
  - 6: 旋转角度
  - 7: 坐标系x方向速度
  - 8: 坐标系y方向速度
- `labels_3d`：表示该3D样本的所有预测框对应的预测类别
- `scores_3d`：表示文该3D样本的所有预测框对应的置信度


相关方法、参数等说明如下：

* `create_model`实例化3D检测模型（此处以`BEVFusion`为例），具体说明如下：
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
<td><code>BEVFusion</code></td>
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

* 调用3D检测模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input` 和 `batch_size`，具体说明如下：

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
<td><code>str</code>/<code>list</code></td>
<td>
<ul>
  <li><b>文件路径</b>，如3D标注文件的本地路径：<code>/root/data/anno_file.pkl</code></li>
  <li><b>列表</b>，列表元素需为上述类型数据，如<code>[\"/root/data/anno_file1.pkl\", \"//root/data/anno_file2.pkl\"]</code>
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

* 对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为`json`文件的操作:

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


</table>

关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的目标检测模型。在使用 PaddleX 开发目标检测模型之前，请务必安装 PaddleX的目标检测相关模型训练插件，安装过程可以参考 [PaddleX本地安装教程](../../../installation/installation.md)

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/nuscenes_demo.tar -P ./dataset
tar -xf ./dataset/nuscenes_demo.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/3d_bev_detection/BEVFusion.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/nuscenes_demo
```
执行上述命令后，PaddleX 会对数据集进行校验。命令运行成功后会在 log 中打印出 `Check dataset passed !` 信息，同时相关产出会保存在当前目录的 `./output/check_dataset` 目录下。校验结果文件保存在 `./output/check_dataset_result.json`，校验结果文件具体内容为

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>

<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;num_classes&quot;: 11,
    &quot;train_mate&quot;: [
      {
        &quot;sample_idx&quot;: &quot;f9878012c3f6412184c294c13ba4bac3&quot;,
        &quot;lidar_path&quot;: &quot;./data/nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin&quot;,
        &quot;image_paths&quot; [
          &quot;./data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-05-21-11-06-59-0400__CAM_FRONT_LEFT__1526915243004917.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-05-21-11-06-59-0400__CAM_FRONT_RIGHT__1526915243019956.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-05-21-11-06-59-0400__CAM_BACK_RIGHT__1526915243027813.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_BACK/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-05-21-11-06-59-0400__CAM_BACK_LEFT__1526915243047295.jpg&quot;
        ]
      },
    ],
    &quot;val_mate&quot;: [
      {
        &quot;sample_idx&quot;: &quot;30e55a3ec6184d8cb1944b39ba19d622&quot;,
        &quot;lidar_path&quot;: &quot;./data/nuscenes/samples/LIDAR_TOP/n015-2018-07-11-11-54-16+0800__LIDAR_TOP__1531281439800013.pcd.bin&quot;,
        &quot;image_paths&quot;: [
          &quot;./data/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_LEFT__1531281439754844.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439762460.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_RIGHT__1531281439770339.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_BACK_RIGHT__1531281439777893.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_BACK/n015-2018-07-11-11-54-16+0800__CAM_BACK__1531281439787525.jpg&quot;,
          &quot;./data/nuscenes/samples/CAM_BACK_LEFT/n015-2018-07-11-11-54-16+0800__CAM_BACK_LEFT__1531281439797423.jpg&quot;
        ]
      },
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;/workspace/bevfusion/Paddle3D/data/nuscenes&quot;,
  &quot;show_type&quot;: &quot;txt&quot;,
  &quot;dataset_type&quot;: &quot;NuscenesMMDataset&quot;
}
</code></pre>
<p>上述校验结果中，check_pass 为 true 表示数据集格式符合要求</p>
</details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过<b>修改配置文件</b>或是<b>追加超参数</b>的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>

<p>3D多模态融合检测模块不支持数据格式转换与数据集划分。</p></details>


### 4.2 模型训练
一条命令即可完成模型的训练，以此处3D多模态融合检测模型 `BEVFusion` 的训练为例：

```bash
python main.py -c paddlex/configs/modules/3d_bev_detection/BEVFusion.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/nuscenes_demo \
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`bevf_pp_2x8_1x_nusc.yaml`，训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
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
<li><code>.pdparams</code>、<code>.pdopt</code>、<code>.pdiparams</code>、<code>.pdmodel</code>：模型权重相关文件，包括网络参数、优化器、静态图网络参数、静态图网络结构等；</li>
</ul></details>

## <b>4.3 模型评估</b>
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：

```bash
python main.py -c paddlex/configs/modules/3d_bev_detection/BEVFusion.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/nuscenes_demo \
```

与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`bevf_pp_2x8_1x_nusc.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>。</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json，其记录了</code>评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 mAP, NDS；</p></details>

### <b>4.4 模型推理和模型集成</b>
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理

* 通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例数据] (https://paddle-model-ecology.bj.bcebos.com/paddlex/det_3d/demo_det_3d/nuscenes_demo_infer.tar)到本地。
```bash
python main.py -c paddlex/configs/modules/3d_bev_detection/BEVFusion.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="nuscenes_demo_infer.tar"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`bevf_pp_2x8_1x_nusc.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

3D多模态融合检测模块可以集成的PaddleX产线有3D检测产线，只需要替换模型路径即可完成相关产线的目标检测模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.<b>模块集成</b>

您产出的权重可以直接集成到3D多模态融合检测模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。
