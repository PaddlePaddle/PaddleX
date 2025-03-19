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
<td>YOWO</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/YOWO_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/YOWO_pretrained.pdparams">训练模型</a></td>
<td>80.94</td>
<td>462.891M</td>
<td rowspan="1">
YOWO是具有两个分支的单阶段网络。一个分支通过2D-CNN提取关键帧（即当前帧）的空间特征，而另一个分支则通过3D-CNN获取由先前帧组成的剪辑的时空特征。为准确汇总这些特征，YOWO使用了一种通道融合和关注机制，最大程度地利用了通道间的依赖性。最后将融合后的特征进行帧级检测。
</td>
</tr>

</table>

<strong>测试环境说明:</strong>

  <ul>
      <li><b>性能测试环境</b>
          <ul>
              <li><strong>测试数据集：</strong><a href="http://www.thumos.info/download.html">UCF101-24</a> test数据集。</li>
              <li><strong>硬件配置：</strong>
                  <ul>
                      <li>GPU：NVIDIA Tesla T4</li>
                      <li>CPU：Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>其他环境：Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
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

</details>

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)。

完成 wheel 包的安装后，几行代码即可完成视频检测模块的推理，可以任意切换该模块下的模型，您也可以将视频检测的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例视频](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi)到本地。

```python
from paddlex import create_model
model = create_model(model_name="YOWO")
output = model.predict(input="HorseRiding.avi", batch_size=1)
for res in output:
    res.print()
    res.save_to_video(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

<details><summary>👉 <b>运行后，得到的结果为：（点击展开）</b></summary>

```bash
{'res': {'input_path': 'HorseRiding.avi', 'result': [[[[110, 40, 170, 171], 0.8385784886274905, 'HorseRiding']], [[[112, 31, 168, 167], 0.8587647461352432, 'HorseRiding']], [[[106, 28, 164, 165], 0.8579590929730969, 'HorseRiding']], [[[106, 24, 165, 171], 0.8743957465404151, 'HorseRiding']], [[[107, 22, 165, 172], 0.8488322619908999, 'HorseRiding']], [[[112, 22, 173, 171], 0.8446755521458691, 'HorseRiding']], [[[115, 23, 177, 176], 0.8454028365262367, 'HorseRiding']], [[[117, 22, 178, 179], 0.8484261880748285, 'HorseRiding']], [[[117, 22, 181, 181], 0.8319480115446183, 'HorseRiding']], [[[117, 39, 182, 183], 0.820551099084625, 'HorseRiding']], [[[117, 41, 183, 185], 0.8202395831914338, 'HorseRiding']], [[[121, 47, 185, 190], 0.8261058921745246, 'HorseRiding']], [[[123, 46, 188, 196], 0.8307278306829033, 'HorseRiding']], [[[125, 44, 189, 197], 0.8259781361122833, 'HorseRiding']], [[[128, 47, 191, 195], 0.8227593229866699, 'HorseRiding']], [[[127, 44, 192, 193], 0.8205373129456817, 'HorseRiding']], [[[129, 39, 192, 185], 0.8223318812628619, 'HorseRiding']], [[[127, 31, 196, 179], 0.8501208612019866, 'HorseRiding']], [[[128, 22, 193, 171], 0.8315708410681566, 'HorseRiding']], [[[130, 22, 192, 169], 0.8318588228062005, 'HorseRiding']], [[[132, 18, 193, 170], 0.8310494469100611, 'HorseRiding']], [[[132, 18, 194, 172], 0.8302132445350239, 'HorseRiding']], [[[133, 18, 194, 176], 0.8339063714162727, 'HorseRiding']], [[[134, 26, 200, 183], 0.8365876380675275, 'HorseRiding']], [[[133, 16, 198, 182], 0.8395230321418268, 'HorseRiding']], [[[133, 17, 199, 184], 0.8198139782724922, 'HorseRiding']], [[[140, 28, 204, 189], 0.8344166596681291, 'HorseRiding']], [[[139, 27, 204, 187], 0.8412694521771158, 'HorseRiding']], [[[139, 28, 204, 185], 0.8500098862888805, 'HorseRiding']], [[[135, 19, 199, 179], 0.8506627974981384, 'HorseRiding']], [[[132, 15, 201, 178], 0.8495054272547193, 'HorseRiding']], [[[136, 14, 199, 173], 0.8451630721500223, 'HorseRiding']], [[[136, 12, 200, 167], 0.8366456814214907, 'HorseRiding']], [[[133, 8, 200, 168], 0.8457252233401213, 'HorseRiding']], [[[131, 7, 197, 162], 0.8400586356358062, 'HorseRiding']], [[[131, 8, 195, 163], 0.8320492682901985, 'HorseRiding']], [[[129, 4, 194, 159], 0.8298043752822792, 'HorseRiding']], [[[127, 5, 194, 162], 0.8348390851948722, 'HorseRiding']], [[[125, 7, 190, 164], 0.8299688814865505, 'HorseRiding']], [[[125, 6, 191, 164], 0.8303107088154711, 'HorseRiding']], [[[123, 8, 190, 168], 0.8348342187965798, 'HorseRiding']], [[[125, 14, 189, 170], 0.8356523950497134, 'HorseRiding']], [[[127, 18, 191, 171], 0.8392671764931521, 'HorseRiding']], [[[127, 30, 193, 178], 0.8441704160826191, 'HorseRiding']], [[[128, 18, 190, 181], 0.8438125326146775, 'HorseRiding']], [[[128, 12, 189, 186], 0.8390128962093542, 'HorseRiding']], [[[129, 15, 190, 185], 0.8471056476788448, 'HorseRiding']], [[[129, 16, 191, 184], 0.8536121834731034, 'HorseRiding']], [[[129, 16, 192, 185], 0.8488154629800881, 'HorseRiding']], [[[128, 15, 194, 184], 0.8417711698421471, 'HorseRiding']], [[[129, 13, 195, 187], 0.8412510238991473, 'HorseRiding']], [[[129, 14, 191, 187], 0.8404350980083457, 'HorseRiding']], [[[129, 13, 190, 189], 0.8382891279858882, 'HorseRiding']], [[[129, 11, 187, 191], 0.8318282305903217, 'HorseRiding']], [[[128, 8, 188, 195], 0.8043430817880264, 'HorseRiding']], [[[131, 25, 193, 199], 0.826184954516826, 'HorseRiding']], [[[124, 35, 191, 203], 0.8270462809459467, 'HorseRiding']], [[[121, 38, 191, 206], 0.8350931715324705, 'HorseRiding']], [[[124, 41, 195, 212], 0.8331239341053625, 'HorseRiding']], [[[128, 42, 194, 211], 0.8343046153103657, 'HorseRiding']], [[[131, 40, 192, 203], 0.8309784496027532, 'HorseRiding']], [[[130, 32, 195, 202], 0.8316640083647542, 'HorseRiding']], [[[135, 30, 196, 197], 0.8272172409555161, 'HorseRiding']], [[[131, 16, 197, 186], 0.8388410406147955, 'HorseRiding']], [[[134, 15, 202, 184], 0.8485738297037244, 'HorseRiding']], [[[136, 15, 209, 182], 0.8529430205135213, 'HorseRiding']], [[[134, 13, 218, 182], 0.8601191479922718, 'HorseRiding']], [[[144, 10, 213, 183], 0.8591963099263467, 'HorseRiding']], [[[151, 12, 219, 184], 0.8617965108346937, 'HorseRiding']], [[[151, 10, 220, 186], 0.8631923599955371, 'HorseRiding']], [[[145, 10, 216, 186], 0.8800860885204287, 'HorseRiding']], [[[144, 10, 216, 186], 0.8858840451538228, 'HorseRiding']], [[[146, 11, 214, 190], 0.8773644144886106, 'HorseRiding']], [[[145, 24, 214, 193], 0.8605544385867248, 'HorseRiding']], [[[146, 23, 214, 193], 0.8727294882672254, 'HorseRiding']], [[[148, 22, 212, 198], 0.8713131467067079, 'HorseRiding']], [[[146, 29, 213, 197], 0.8579099324651196, 'HorseRiding']], [[[154, 29, 217, 199], 0.8547794072847914, 'HorseRiding']], [[[151, 26, 217, 203], 0.8641733722316758, 'HorseRiding']], [[[146, 24, 212, 199], 0.8613466257602624, 'HorseRiding']], [[[142, 25, 210, 194], 0.8492670944810214, 'HorseRiding']], [[[134, 24, 204, 192], 0.8428117300203049, 'HorseRiding']], [[[136, 25, 204, 189], 0.8486779356971397, 'HorseRiding']], [[[127, 21, 199, 179], 0.8513896296400709, 'HorseRiding']], [[[125, 10, 192, 192], 0.8510201771386576, 'HorseRiding']], [[[124, 8, 191, 192], 0.8493999019508465, 'HorseRiding']], [[[121, 8, 192, 193], 0.8487097098892171, 'HorseRiding']], [[[119, 6, 187, 193], 0.847543279648022, 'HorseRiding']], [[[118, 12, 190, 190], 0.8503535936620565, 'HorseRiding']], [[[122, 22, 189, 194], 0.8427901493276977, 'HorseRiding']], [[[118, 24, 188, 195], 0.8418835400352087, 'HorseRiding']], [[[120, 25, 188, 205], 0.847192725785284, 'HorseRiding']], [[[122, 25, 189, 207], 0.8444105420674433, 'HorseRiding']], [[[120, 23, 189, 208], 0.8470784016639392, 'HorseRiding']], [[[121, 23, 188, 205], 0.843428111269418, 'HorseRiding']], [[[117, 23, 186, 206], 0.8420809714166708, 'HorseRiding']], [[[119, 5, 199, 197], 0.8288265053231356, 'HorseRiding']], [[[121, 8, 192, 195], 0.8197548738023599, 'HorseRiding']]]}}
```

参数含义如下：
- `input_path`：表示输入待预测视频的路径
- `result`：视频帧预测的目标框信息，一个列表。列表中每个元素代表一个图像帧的检出结果，仍是一个列表，每个元素是一个检测框的信息，分别是：
  - <code>[xmin, ymin, xmax, ymax]</code>: 目标框左上角和右下角坐标，均为浮点数
  - float: 目标框的置信度，浮点数
  - string: 目标框的所属类别，字符串

</details>

可视化视频如下：
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/video_detection/HorseRiding_res.jpg">

上述Python脚本中，执行了如下几个步骤：

* `create_model`实例化视频检测模型（`YOWO`），具体说明如下：


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
<td><code>device</code></td>
<td>模型推理设备</td>
<td><code>str</code></td>
<td>支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。</td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理</td>
<td><code>bool</code></td>
<td>无</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code> nms_thresh</code></td>
<td>非极大值抑制（Non-Maximum Suppression, NMS）过程中的IoU阈值参数；如果不指定，将默认使用PaddleX官方模型配置</td>
<td><code>float/None</code></td>
<td>大于0的浮点数/None</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code> score_thresh</code></td>
<td>预测置信度阈值；如果不指定，将默认使用PaddleX官方模型配置</td>
<td><code>float/None</code></td>
<td>大于0的浮点数/None</td>
<td><code>None</code></td>
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
<td><code>nms_thresh</code></td>
<td>非极大值抑制（Non-Maximum Suppression, NMS）过程中的IoU阈值参数</td>
<td><code>float|None</code></td>
<td>
<ul>
  <li><b>float</b>：大于0的浮点数；</li>
  <li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为0.4；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>score_thresh</code></td>
<td>预测置信度阈值</td>
<td><code>float|None</code></td>
<td>
<ul>
  <li><b>float</b>：大于0的浮点数；</li>
  <li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为0.8；</li>
</ul>
</td>
<td><code>None</code></td>
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

</details>

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
