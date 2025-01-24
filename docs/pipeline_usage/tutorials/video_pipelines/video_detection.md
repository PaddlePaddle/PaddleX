---
comments: true
---

# 通用视频检测产线使用教程

## 1. 通用视频检测产线介绍

视频检测是一种识别和定位视频内容中特定对象或事件的技术，广泛应用于安防监控、交通管理和行为分析等领域。该技术能够实时捕捉和分析视频中的动态变化，如人员活动、车辆流动及异常事件等。通过深度学习模型，视频检测能够高效提取视频中的空间和时间特征，实现精准识别和定位。视频检测不仅提升了监控系统的智能化程度，还为提高安全性和运营效率提供了重要支撑。随着技术的发展，视频检测将在更多场景中发挥关键作用。

<img src="https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/images/yowo.jpg">

<b>视频检测</b><b>产线中包含视频检测模块</b>包含的模型如下。

<p><b>视频检测模块（可选）：</b></p>


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

<p><b>注：以上精度指标为 <a href="http://www.thumos.info/download.html">UCF101-24</a> test数据集上的测试指标Frame-mAP (@ IoU 0.5)。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>

## 2. 快速开始

PaddleX 支持在本地使用命令行或 Python 体验产线的效果。

在本地使用通用视频检测产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

### 2.1 本地体验

#### 2.1 命令行方式体验
一行命令即可快速体验视频检测产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline video_detection --input HorseRiding.avi --device gpu:0 --save_path output
```

相关的参数说明可以参考[2.1.2 Python脚本方式集成](#212-python脚本方式集成)中的参数说明。

运行后，会将结果打印到终端上，结果如下：

<details><summary> 👉点击展开</summary>

```bash
{'input_path': 'HorseRiding.avi', 'result': [[[[110, 40, 170, 171], 0.8385784886274905, 'HorseRiding']], [[[112, 31, 168, 167], 0.8587647461352432, 'HorseRiding']], [[[106, 28, 164, 165], 0.8579590929730969, 'HorseRiding']], [[[106, 24, 165, 171], 0.8743957465404151, 'HorseRiding']], [[[107, 22, 165, 172], 0.8488322619908999, 'HorseRiding']], [[[112, 22, 173, 171], 0.8446755521458691, 'HorseRiding']], [[[115, 23, 177, 176], 0.8454028365262367, 'HorseRiding']], [[[117, 22, 178, 179], 0.8484261880748285, 'HorseRiding']], [[[117, 22, 181, 181], 0.8319480115446183, 'HorseRiding']], [[[117, 39, 182, 183], 0.820551099084625, 'HorseRiding']], [[[117, 41, 183, 185], 0.8202395831914338, 'HorseRiding']], [[[121, 47, 185, 190], 0.8261058921745246, 'HorseRiding']], [[[123, 46, 188, 196], 0.8307278306829033, 'HorseRiding']], [[[125, 44, 189, 197], 0.8259781361122833, 'HorseRiding']], [[[128, 47, 191, 195], 0.8227593229866699, 'HorseRiding']], [[[127, 44, 192, 193], 0.8205373129456817, 'HorseRiding']], [[[129, 39, 192, 185], 0.8223318812628619, 'HorseRiding']], [[[127, 31, 196, 179], 0.8501208612019866, 'HorseRiding']], [[[128, 22, 193, 171], 0.8315708410681566, 'HorseRiding']], [[[130, 22, 192, 169], 0.8318588228062005, 'HorseRiding']], [[[132, 18, 193, 170], 0.8310494469100611, 'HorseRiding']], [[[132, 18, 194, 172], 0.8302132445350239, 'HorseRiding']], [[[133, 18, 194, 176], 0.8339063714162727, 'HorseRiding']], [[[134, 26, 200, 183], 0.8365876380675275, 'HorseRiding']], [[[133, 16, 198, 182], 0.8395230321418268, 'HorseRiding']], [[[133, 17, 199, 184], 0.8198139782724922, 'HorseRiding']], [[[140, 28, 204, 189], 0.8344166596681291, 'HorseRiding']], [[[139, 27, 204, 187], 0.8412694521771158, 'HorseRiding']], [[[139, 28, 204, 185], 0.8500098862888805, 'HorseRiding']], [[[135, 19, 199, 179], 0.8506627974981384, 'HorseRiding']], [[[132, 15, 201, 178], 0.8495054272547193, 'HorseRiding']], [[[136, 14, 199, 173], 0.8451630721500223, 'HorseRiding']], [[[136, 12, 200, 167], 0.8366456814214907, 'HorseRiding']], [[[133, 8, 200, 168], 0.8457252233401213, 'HorseRiding']], [[[131, 7, 197, 162], 0.8400586356358062, 'HorseRiding']], [[[131, 8, 195, 163], 0.8320492682901985, 'HorseRiding']], [[[129, 4, 194, 159], 0.8298043752822792, 'HorseRiding']], [[[127, 5, 194, 162], 0.8348390851948722, 'HorseRiding']], [[[125, 7, 190, 164], 0.8299688814865505, 'HorseRiding']], [[[125, 6, 191, 164], 0.8303107088154711, 'HorseRiding']], [[[123, 8, 190, 168], 0.8348342187965798, 'HorseRiding']], [[[125, 14, 189, 170], 0.8356523950497134, 'HorseRiding']], [[[127, 18, 191, 171], 0.8392671764931521, 'HorseRiding']], [[[127, 30, 193, 178], 0.8441704160826191, 'HorseRiding']], [[[128, 18, 190, 181], 0.8438125326146775, 'HorseRiding']], [[[128, 12, 189, 186], 0.8390128962093542, 'HorseRiding']], [[[129, 15, 190, 185], 0.8471056476788448, 'HorseRiding']], [[[129, 16, 191, 184], 0.8536121834731034, 'HorseRiding']], [[[129, 16, 192, 185], 0.8488154629800881, 'HorseRiding']], [[[128, 15, 194, 184], 0.8417711698421471, 'HorseRiding']], [[[129, 13, 195, 187], 0.8412510238991473, 'HorseRiding']], [[[129, 14, 191, 187], 0.8404350980083457, 'HorseRiding']], [[[129, 13, 190, 189], 0.8382891279858882, 'HorseRiding']], [[[129, 11, 187, 191], 0.8318282305903217, 'HorseRiding']], [[[128, 8, 188, 195], 0.8043430817880264, 'HorseRiding']], [[[131, 25, 193, 199], 0.826184954516826, 'HorseRiding']], [[[124, 35, 191, 203], 0.8270462809459467, 'HorseRiding']], [[[121, 38, 191, 206], 0.8350931715324705, 'HorseRiding']], [[[124, 41, 195, 212], 0.8331239341053625, 'HorseRiding']], [[[128, 42, 194, 211], 0.8343046153103657, 'HorseRiding']], [[[131, 40, 192, 203], 0.8309784496027532, 'HorseRiding']], [[[130, 32, 195, 202], 0.8316640083647542, 'HorseRiding']], [[[135, 30, 196, 197], 0.8272172409555161, 'HorseRiding']], [[[131, 16, 197, 186], 0.8388410406147955, 'HorseRiding']], [[[134, 15, 202, 184], 0.8485738297037244, 'HorseRiding']], [[[136, 15, 209, 182], 0.8529430205135213, 'HorseRiding']], [[[134, 13, 218, 182], 0.8601191479922718, 'HorseRiding']], [[[144, 10, 213, 183], 0.8591963099263467, 'HorseRiding']], [[[151, 12, 219, 184], 0.8617965108346937, 'HorseRiding']], [[[151, 10, 220, 186], 0.8631923599955371, 'HorseRiding']], [[[145, 10, 216, 186], 0.8800860885204287, 'HorseRiding']], [[[144, 10, 216, 186], 0.8858840451538228, 'HorseRiding']], [[[146, 11, 214, 190], 0.8773644144886106, 'HorseRiding']], [[[145, 24, 214, 193], 0.8605544385867248, 'HorseRiding']], [[[146, 23, 214, 193], 0.8727294882672254, 'HorseRiding']], [[[148, 22, 212, 198], 0.8713131467067079, 'HorseRiding']], [[[146, 29, 213, 197], 0.8579099324651196, 'HorseRiding']], [[[154, 29, 217, 199], 0.8547794072847914, 'HorseRiding']], [[[151, 26, 217, 203], 0.8641733722316758, 'HorseRiding']], [[[146, 24, 212, 199], 0.8613466257602624, 'HorseRiding']], [[[142, 25, 210, 194], 0.8492670944810214, 'HorseRiding']], [[[134, 24, 204, 192], 0.8428117300203049, 'HorseRiding']], [[[136, 25, 204, 189], 0.8486779356971397, 'HorseRiding']], [[[127, 21, 199, 179], 0.8513896296400709, 'HorseRiding']], [[[125, 10, 192, 192], 0.8510201771386576, 'HorseRiding']], [[[124, 8, 191, 192], 0.8493999019508465, 'HorseRiding']], [[[121, 8, 192, 193], 0.8487097098892171, 'HorseRiding']], [[[119, 6, 187, 193], 0.847543279648022, 'HorseRiding']], [[[118, 12, 190, 190], 0.8503535936620565, 'HorseRiding']], [[[122, 22, 189, 194], 0.8427901493276977, 'HorseRiding']], [[[118, 24, 188, 195], 0.8418835400352087, 'HorseRiding']], [[[120, 25, 188, 205], 0.847192725785284, 'HorseRiding']], [[[122, 25, 189, 207], 0.8444105420674433, 'HorseRiding']], [[[120, 23, 189, 208], 0.8470784016639392, 'HorseRiding']], [[[121, 23, 188, 205], 0.843428111269418, 'HorseRiding']], [[[117, 23, 186, 206], 0.8420809714166708, 'HorseRiding']], [[[119, 5, 199, 197], 0.8288265053231356, 'HorseRiding']], [[[121, 8, 192, 195], 0.8197548738023599, 'HorseRiding']]]}
```


</details>

运行结果参数说明可以参考[2.1.2 Python脚本方式集成](#212-python脚本方式集成)中的结果解释。

可视化结果保存在`save_path`下，可视化结果如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/video_detection/HorseRiding_res.jpg">


#### 2.1.2 Python脚本方式集成

上述命令行是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="video_detection")
output = pipeline.predict(input="HorseRiding.avi")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_video(save_path="./output/") ## 保存结果可视化视频
    res.save_to_json(save_path="./output/") ## 保存预测的结构化输出
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）实例化 `create_pipeline` 实例化产线对象：具体参数说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>config</code></td>
<td>产线具体的配置信息（如果和<code>pipeline</code>同时设置，优先级高于<code>pipeline</code>，且要求产线名和<code>pipeline</code>一致）。</td>
<td><code>dict[str, Any]</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线推理设备。支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理，仅当该产线支持高性能推理时可用。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

（2）调用视频检测产线对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：

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
<td>待预测视频数据，支持多种输入类型，必填</td>
<td><code>Python str|list</code></td>
<td>
<ul>
  <li><b>str</b>：视频文件本地路径：<code>/root/data/video.avi</code>；<b>如URL链接</b>，如视频文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi">示例</a>；<b>如本地目录</b>，该目录下需包含待预测视频，如本地路径：<code>/root/data/</code></li>
  <li><b>List</b>：列表元素需为上述类型数据，如<code>[str, str]</code>，<code>[\"/root/data/video1.mp4\", \"/root/data/video2.avi\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线推理设备</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
  <li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
  <li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
  <li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
  <li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
  <li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
  <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备；</li>
</ul>
</td>
<td><code>None</code></td>
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
</tbody>
</table>

（3）对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为视频、保存为`json`文件的操作:

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
<td>save_to_video</td>
<td>将结果保存为视频格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
</table>

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：

    - `input_path`: `(str)` 待预测图像的输入路径

    - `result`: `(List[List[List]])` 预测结果，每个列表表示一帧图像的预测结果，每帧结果包含以下内容：

        - `[xmin, ymin, xmax, ymax]`: `(list)` 边界框坐标，格式为[xmin, ymin, xmax, ymax]，其中(xmin,ymin)是左上角坐标点，(xmax,ymax)是右下角坐标点
        - `float`: 目标框的置信度，浮点数
        - `str`: 目标框的所属类别，字符串

    - `angle`: `(int)` 文档方向分类的预测结果。启用时取值为[0,90,180,270]；未启用时为-1

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。
- 调用`save_to_video()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。

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
<td rowspan = "1">获取预测的 <code>json</code> 格式的结果</td>
</tr>
<tr>
</tr>
</table>


- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。


此外，您可以获取 video_detection 产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config video_detection --save_path ./my_path
```

若您获取了配置文件，即可对 video_detection 产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。示例如下：

例如，若您的配置文件保存在 `./my_path/video_detection*.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/video_detection.yaml")
output = pipeline.predict(input="HorseRiding.avi")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_video("./output/") ## 保存结果可视化视频
    res.save_to_json("./output/") ## 保存预测的结构化输出
```

<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改 video_detection 产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。同时，CLI 预测也支持传入配置文件，`--pipeline` 指定配置文件的路径即可。

## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 <b>高性能推理</b>：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ <b>服务化部署</b>：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持多种产线服务化部署方案，详细的产线服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/serving.md)。

以下是基础服务化部署的API参考与多语言服务调用示例：

<details><summary>API参考</summary>

<p>对于服务提供的主要操作：</p>
<ul>
<li>HTTP请求方法为POST。</li>
<li>请求体和响应体均为JSON数据（JSON对象）。</li>
<li>当请求处理成功时，响应状态码为<code>200</code>，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。固定为<code>0</code>。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。固定为<code>"Success"</code>。</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>操作结果。</td>
</tr>
</tbody>
</table>
<ul>
<li>当请求处理未成功时，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。与响应状态码相同。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。</td>
</tr>
</tbody>
</table>
<p>服务提供的主要操作如下：</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>对视频进行分类。</p>
<p><code>POST /video-detection</code></p>
<ul>
<li>请求体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>video</code></td>
<td><code>string</code></td>
<td>服务器可访问的视频文件的URL或视频文件内容的Base64编码结果。</td>
<td>是</td>
</tr>
<tr>
<td><code>nmsThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>nms_thresh</code> 参数说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>scoreThresh</code></td>
<td><code>number</code> | <code>null</code></td>
<td>参见产线 <code>predict</code> 方法中的 <code>score_thresh</code> 参数说明。</td>
<td>否</td>
</tr>
</tbody>
</table>
<ul>
<li>请求处理成功时，响应体的<code>result</code>具有如下属性：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>frames</code></td>
<td><code>array</code></td>
<td>每一帧的检测结果。</td>
</tr>
</tbody>
</table>
<p><code>frames</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tr>
<td><code>index</code></td>
<td><code>integer</code></td>
<td>从 0 开始的帧编号</td>
</tr>
<tr>
<td><code>detectedObjects</code></td>
<td><code>array</code></td>
<td>目标的位置、类别等信息。</td>
</tr>
</tbody>
</table>
<p><code>detectedObjects</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>目标位置。数组中元素依次为边界框左上角x坐标、左上角y坐标、右下角x坐标以及右下角y坐标。</td>
</tr>
<tr>
<td><code>categoryName</code></td>
<td><code>string</code></td>
<td>目标类别名称。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>目标得分。</td>
</tr>
</tbody>
</table>
</details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/video-detection&quot; # 服务URL
video_path = &quot;./demo.mp4&quot;

# 对本地视频进行Base64编码
with open(video_path, &quot;rb&quot;) as file:
    video_bytes = file.read()
    video_data = base64.b64encode(video_bytes).decode(&quot;ascii&quot;)

payload = {&quot;video&quot;: video_data}  # Base64编码的文件内容或者视频URL

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()[&quot;result&quot;]
print(&quot;\nFrames:&quot;)
print(result[&quot;frames&quot;])
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果通用视频检测产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升通用视频检测产线的在您的场景中的识别效果。

### 4.1 模型微调
由于通用视频检测产线包含视频检测模块，如果模型产线的效果不及预期，那么您需要参考[视频检测模块开发教程](../../../module_usage/tutorials/video_modules/video_detection.md)中的[二次开发](../../../module_usage/tutorials/video_modules/video_detection.md#四二次开发)章节，使用您的私有数据集对视频检测模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```
......
Pipeline:
  model: YOWO #可修改为微调后模型的本地路径
  device: "gpu"
  batch_size: 1
......
```
随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device` 参数</b>即可完成不同硬件之间的无缝切换。

例如，您使用昇腾 NPU 进行视频检测产线的推理，使用的 CLI 命令为：

```bash
paddlex --pipeline video_detection --input HorseRiding.avi --device npu:0
```
当然，您也可以在 Python 脚本中 `create_pipeline()` 时或者 `predict()` 时指定硬件设备。

若您想在更多种类的硬件上使用通用视频检测产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
