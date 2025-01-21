---
comments: true
---

# 视频检测模块使用教程

## 一、概述
视频检测任务是计算机视觉系统中的关键组成部分，专注于识别和定位视频序列中的物体或事件。视频检测将视频分解为单独的帧序列, 然后分析这些帧以识别检测物体或动作，例如在监控视频中检测行人，或在体育或娱乐视频中识别特定活动，如“跑步”、“跳跃”或“弹吉他”。
视频检测模块的输出包括每个检测到的物体或事件的边界框和类别标签。这些信息可以被其他模块或系统用于进一步分析，例如跟踪检测到的物体的移动、生成警报或编制统计数据以供决策过程使用。因此，视频检测在从安全监控和自动驾驶到体育分析和内容审核的各种应用中都扮演着重要角色。

## 二、支持模型列表


<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Frame-mAP(@ IoU 0.5)</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>YOWO</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/YOWO_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOWO_pretrained.pdparams">训练模型</a></td>
<td>80.94</td>
<td>462.891M</td>
<td rowspan="1">
YOWO是具有两个分支的单阶段网络。一个分支通过2D-CNN提取关键帧（即当前帧）的空间特征，而另一个分支则通过3D-CNN获取由先前帧组成的剪辑的时空特征。为准确汇总这些特征，YOWO使用了一种通道融合和关注机制，最大程度地利用了通道间的依赖性。最后将融合后的特征进行帧级检测。
</td>
</tr>

</table>


<p><b>注：以上精度指标为 <a href="http://www.thumos.info/download.html">UCF101-24</a> test数据集上的测试指标Frame-mAP (@ IoU 0.5)。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p></details>

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)。

完成 wheel 包的安装后，几行代码即可完成视频检测模块的推理，可以任意切换该模块下的模型，您也可以将视频检测的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例视频](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi)到本地。

```python
from paddlex import create_model
model = create_model("YOWO")
output = model.predict("HorseRiding.avi", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_video("./output/")
    res.save_to_json("./output/res.json")
```

关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。

## 四、二次开发
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的视频检测模型。在使用 PaddleX 开发视频检测模型之前，请务必安装 PaddleX 的 视频检测  [PaddleX本地安装教程](../../../installation/installation.md)中的二次开发部分。

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX视频检测任务模块数据标注教程](../../../data_annotations/video_modules/video_detection.md)

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/video_det_examples.tar -P ./dataset
tar -xf ./dataset/video_det_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/video_detection/YOWO.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/video_det_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息。命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>
<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">
{
  "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "..\/..\/dataset\/video_det_examples\/label_map.txt",
    "num_classes": 24,
    "train_samples": 6878,
    "train_sample_paths": [
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/SoccerJuggling\/v_SoccerJuggling_g19_c06\/00296.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/SkateBoarding\/v_SkateBoarding_g17_c04\/00026.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/RopeClimbing\/v_RopeClimbing_g01_c03\/00055.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/HorseRiding\/v_HorseRiding_g11_c05\/00132.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/PoleVault\/v_PoleVault_g13_c03\/00089.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/Basketball\/v_Basketball_g13_c04\/00050.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/PoleVault\/v_PoleVault_g01_c05\/00024.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/RopeClimbing\/v_RopeClimbing_g03_c04\/00118.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/GolfSwing\/v_GolfSwing_g01_c06\/00231.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/TrampolineJumping\/v_TrampolineJumping_g02_c02\/00134.jpg"
    ],
    "val_samples": 3916,
    "val_sample_paths": [
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/IceDancing\/v_IceDancing_g22_c02\/00017.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/TennisSwing\/v_TennisSwing_g04_c02\/00046.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/SoccerJuggling\/v_SoccerJuggling_g08_c03\/00169.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/Fencing\/v_Fencing_g24_c02\/00009.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/Diving\/v_Diving_g16_c02\/00110.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/HorseRiding\/v_HorseRiding_g08_c02\/00079.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/PoleVault\/v_PoleVault_g17_c07\/00008.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/Skiing\/v_Skiing_g20_c06\/00221.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/PoleVault\/v_PoleVault_g17_c07\/00137.jpg",
      "check_dataset\/..\/..\/dataset\/video_det_examples\/rgb-images\/GolfSwing\/v_GolfSwing_g24_c01\/00093.jpg"
    ]
  },
  "analysis": {
    "histogram": "check_dataset\/histogram.png"
  },
  "dataset_path": "video_det_examples",
  "show_type": "video",
  "dataset_type": "VideoDetDataset"
}
</code></pre>
<p>上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.num_classes</code>：该数据集类别数为 24；</li>
<li><code>attributes.train_samples</code>：该数据集训练集样本数量为 6878；</li>
<li><code>attributes.val_samples</code>：该数据集验证集样本数量为 3916；</li>
<li><code>attributes.train_sample_paths</code>：该数据集训练集样本可视化视频相对路径列表；</li>
<li><code>attributes.val_sample_paths</code>：该数据集验证集样本可视化视频相对路径列表；</li>
</ul>
<p>另外，数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/video_detection/01.png"></p></details>

#### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过<b>修改配置文件</b>或是<b>追加超参数</b>的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>

<p><b>（1）数据集格式转换</b></p>
<p>视频检测暂不支持数据转换。</p>
<p><b>（2）数据集划分</b></p>
<p>视频检测暂不支持数据划分。</p>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处视频检测模型 YOWO 的训练为例：

```
python main.py -c paddlex/configs/modules/video_detection/YOWO.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/video_det_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`YOWO.yaml`,训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定第 2 卡 gpu 训练：`-o Global.device=gpu:2`，视频检测只支持单卡训练；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅模型对应任务模块的配置文件说明[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

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
python main.py -c  paddlex/configs/modules/video_detection/YOWO.yaml  \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/video_det_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`YOWO.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>。</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json，其记录了</code>评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 mAP；</p></details>

### <b>4.4 模型推理和模型集成</b>

在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理
通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例视频](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi)到本地。

```bash
python main.py -c paddlex/configs/modules/video_detection/YOWO.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="HorseRiding.avi"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`YOWO.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

视频检测模块可以集成的 PaddleX 产线有[通用视频检测产线](../../../pipeline_usage/tutorials/video_pipelines/video_detection.md)，只需要替换模型路径即可完成相关产线的视频检测模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.<b>模块集成</b>

您产出的权重可以直接集成到视频检测模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。
