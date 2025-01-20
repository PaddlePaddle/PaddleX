---
comments: true
---

# 通用视频检测产线使用教程

## 1. 通用视频检测产线介绍
视频检测是一种识别和定位视频内容中特定对象或事件的技术，广泛应用于安防监控、交通管理和行为分析等领域。该技术能够实时捕捉和分析视频中的动态变化，如人员活动、车辆流动及异常事件等。通过深度学习模型，尤其是卷积神经网络（CNN）的应用，视频检测能够高效提取视频中的空间和时间特征，实现精准识别和定位。视频检测不仅提升了监控系统的智能化程度，还为提高安全性和运营效率提供了重要支撑。随着技术的发展，视频检测将在更多场景中发挥关键作用。

<img src="https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/images/yowo.jpg">


<details><summary> 👉模型列表详情</summary>

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

## 2. 快速开始

PaddleX 支持在本地使用命令行或 Python 体验产线的效果。

在本地使用通用视频检测产线前，请确保您已经按照PaddleX本地安装教程完成了PaddleX的wheel包安装。

#### 2.1 命令行方式体验
一行命令即可快速体验视频检测产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline video_detection --input HorseRiding.avi --device gpu:0
```
参数说明：

```
--pipeline：产线名称，此处为视频检测产线
--input：待处理的输入视频的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```

在执行上述 Python 脚本时，加载的是默认的视频检测产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

<details><summary> 👉点击展开</summary>

<pre><code>paddlex --get_pipeline_config video_detection
</code></pre>
<p>执行后，视频检测产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为 <code>./my_path</code> ）：</p>
<pre><code>paddlex --get_pipeline_config video_detection --save_path ./my_path
</code></pre>
<p>获取产线配置文件后，可将 <code>--pipeline</code> 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 <code>./video_detection.yaml</code>，只需执行：</p>
<pre><code class="language-bash">paddlex --pipeline ./video_detection.yaml --input HorseRiding.avi --device gpu:0
</code></pre>
<p>其中，<code>--model</code>、<code>--device</code> 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。</p></details>

运行后，得到的结果为：

```
{'input_path': 'HorseRiding.avi', 'result': [[[[110, 40, 170, 171], 0.8385784886274905, 'HorseRiding']], [[[112, 31, 168, 167], 0.8587647461352432, 'HorseRiding']], [[[106, 28, 164, 165], 0.8579590929730969, 'HorseRiding']], [[[106, 24, 165, 171], 0.8743957465404151, 'HorseRiding']], [[[107, 22, 165, 172], 0.8488322619908999, 'HorseRiding']], [[[112, 22, 173, 171], 0.8446755521458691, 'HorseRiding']], [[[115, 23, 177, 176], 0.8454028365262367, 'HorseRiding']], [[[117, 22, 178, 179], 0.8484261880748285, 'HorseRiding']], [[[117, 22, 181, 181], 0.8319480115446183, 'HorseRiding']], [[[117, 39, 182, 183], 0.820551099084625, 'HorseRiding']], [[[117, 41, 183, 185], 0.8202395831914338, 'HorseRiding']], [[[121, 47, 185, 190], 0.8261058921745246, 'HorseRiding']], [[[123, 46, 188, 196], 0.8307278306829033, 'HorseRiding']], [[[125, 44, 189, 197], 0.8259781361122833, 'HorseRiding']], [[[128, 47, 191, 195], 0.8227593229866699, 'HorseRiding']], [[[127, 44, 192, 193], 0.8205373129456817, 'HorseRiding']], [[[129, 39, 192, 185], 0.8223318812628619, 'HorseRiding']], [[[127, 31, 196, 179], 0.8501208612019866, 'HorseRiding']], [[[128, 22, 193, 171], 0.8315708410681566, 'HorseRiding']], [[[130, 22, 192, 169], 0.8318588228062005, 'HorseRiding']], [[[132, 18, 193, 170], 0.8310494469100611, 'HorseRiding']], [[[132, 18, 194, 172], 0.8302132445350239, 'HorseRiding']], [[[133, 18, 194, 176], 0.8339063714162727, 'HorseRiding']], [[[134, 26, 200, 183], 0.8365876380675275, 'HorseRiding']], [[[133, 16, 198, 182], 0.8395230321418268, 'HorseRiding']], [[[133, 17, 199, 184], 0.8198139782724922, 'HorseRiding']], [[[140, 28, 204, 189], 0.8344166596681291, 'HorseRiding']], [[[139, 27, 204, 187], 0.8412694521771158, 'HorseRiding']], [[[139, 28, 204, 185], 0.8500098862888805, 'HorseRiding']], [[[135, 19, 199, 179], 0.8506627974981384, 'HorseRiding']], [[[132, 15, 201, 178], 0.8495054272547193, 'HorseRiding']], [[[136, 14, 199, 173], 0.8451630721500223, 'HorseRiding']], [[[136, 12, 200, 167], 0.8366456814214907, 'HorseRiding']], [[[133, 8, 200, 168], 0.8457252233401213, 'HorseRiding']], [[[131, 7, 197, 162], 0.8400586356358062, 'HorseRiding']], [[[131, 8, 195, 163], 0.8320492682901985, 'HorseRiding']], [[[129, 4, 194, 159], 0.8298043752822792, 'HorseRiding']], [[[127, 5, 194, 162], 0.8348390851948722, 'HorseRiding']], [[[125, 7, 190, 164], 0.8299688814865505, 'HorseRiding']], [[[125, 6, 191, 164], 0.8303107088154711, 'HorseRiding']], [[[123, 8, 190, 168], 0.8348342187965798, 'HorseRiding']], [[[125, 14, 189, 170], 0.8356523950497134, 'HorseRiding']], [[[127, 18, 191, 171], 0.8392671764931521, 'HorseRiding']], [[[127, 30, 193, 178], 0.8441704160826191, 'HorseRiding']], [[[128, 18, 190, 181], 0.8438125326146775, 'HorseRiding']], [[[128, 12, 189, 186], 0.8390128962093542, 'HorseRiding']], [[[129, 15, 190, 185], 0.8471056476788448, 'HorseRiding']], [[[129, 16, 191, 184], 0.8536121834731034, 'HorseRiding']], [[[129, 16, 192, 185], 0.8488154629800881, 'HorseRiding']], [[[128, 15, 194, 184], 0.8417711698421471, 'HorseRiding']], [[[129, 13, 195, 187], 0.8412510238991473, 'HorseRiding']], [[[129, 14, 191, 187], 0.8404350980083457, 'HorseRiding']], [[[129, 13, 190, 189], 0.8382891279858882, 'HorseRiding']], [[[129, 11, 187, 191], 0.8318282305903217, 'HorseRiding']], [[[128, 8, 188, 195], 0.8043430817880264, 'HorseRiding']], [[[131, 25, 193, 199], 0.826184954516826, 'HorseRiding']], [[[124, 35, 191, 203], 0.8270462809459467, 'HorseRiding']], [[[121, 38, 191, 206], 0.8350931715324705, 'HorseRiding']], [[[124, 41, 195, 212], 0.8331239341053625, 'HorseRiding']], [[[128, 42, 194, 211], 0.8343046153103657, 'HorseRiding']], [[[131, 40, 192, 203], 0.8309784496027532, 'HorseRiding']], [[[130, 32, 195, 202], 0.8316640083647542, 'HorseRiding']], [[[135, 30, 196, 197], 0.8272172409555161, 'HorseRiding']], [[[131, 16, 197, 186], 0.8388410406147955, 'HorseRiding']], [[[134, 15, 202, 184], 0.8485738297037244, 'HorseRiding']], [[[136, 15, 209, 182], 0.8529430205135213, 'HorseRiding']], [[[134, 13, 218, 182], 0.8601191479922718, 'HorseRiding']], [[[144, 10, 213, 183], 0.8591963099263467, 'HorseRiding']], [[[151, 12, 219, 184], 0.8617965108346937, 'HorseRiding']], [[[151, 10, 220, 186], 0.8631923599955371, 'HorseRiding']], [[[145, 10, 216, 186], 0.8800860885204287, 'HorseRiding']], [[[144, 10, 216, 186], 0.8858840451538228, 'HorseRiding']], [[[146, 11, 214, 190], 0.8773644144886106, 'HorseRiding']], [[[145, 24, 214, 193], 0.8605544385867248, 'HorseRiding']], [[[146, 23, 214, 193], 0.8727294882672254, 'HorseRiding']], [[[148, 22, 212, 198], 0.8713131467067079, 'HorseRiding']], [[[146, 29, 213, 197], 0.8579099324651196, 'HorseRiding']], [[[154, 29, 217, 199], 0.8547794072847914, 'HorseRiding']], [[[151, 26, 217, 203], 0.8641733722316758, 'HorseRiding']], [[[146, 24, 212, 199], 0.8613466257602624, 'HorseRiding']], [[[142, 25, 210, 194], 0.8492670944810214, 'HorseRiding']], [[[134, 24, 204, 192], 0.8428117300203049, 'HorseRiding']], [[[136, 25, 204, 189], 0.8486779356971397, 'HorseRiding']], [[[127, 21, 199, 179], 0.8513896296400709, 'HorseRiding']], [[[125, 10, 192, 192], 0.8510201771386576, 'HorseRiding']], [[[124, 8, 191, 192], 0.8493999019508465, 'HorseRiding']], [[[121, 8, 192, 193], 0.8487097098892171, 'HorseRiding']], [[[119, 6, 187, 193], 0.847543279648022, 'HorseRiding']], [[[118, 12, 190, 190], 0.8503535936620565, 'HorseRiding']], [[[122, 22, 189, 194], 0.8427901493276977, 'HorseRiding']], [[[118, 24, 188, 195], 0.8418835400352087, 'HorseRiding']], [[[120, 25, 188, 205], 0.847192725785284, 'HorseRiding']], [[[122, 25, 189, 207], 0.8444105420674433, 'HorseRiding']], [[[120, 23, 189, 208], 0.8470784016639392, 'HorseRiding']], [[[121, 23, 188, 205], 0.843428111269418, 'HorseRiding']], [[[117, 23, 186, 206], 0.8420809714166708, 'HorseRiding']], [[[119, 5, 199, 197], 0.8288265053231356, 'HorseRiding']], [[[121, 8, 192, 195], 0.8197548738023599, 'HorseRiding']]]}
```

<img src="https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/images/horse_riding.gif">

可视化视频默认不进行保存，您可以通过 `--save_path` 自定义保存路径，随后所有结果将被保存在指定路径下。

#### 2.2 Python脚本方式集成
几行代码即可完成产线的快速推理，以通用视频检测产线为例：

```
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="video_detection")

output = pipeline.predict("HorseRiding.avi")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_video("./output/") ## 保存结果可视化视频
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
得到的结果与命令行方式相同。

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
<td>无</td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线模型推理设备。支持：“gpu”，“cpu”。</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理，仅当该产线支持高性能推理时可用。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
（2）调用视频检测产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

<table>
<thead>
<tr>
<th>参数类型</th>
<th>参数说明</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>支持直接传入Python变量，如numpy.ndarray表示的视频数据。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入待预测数据文件路径，如视频文件的本地路径：<code>/root/data/video.mp4</code>。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入待预测数据文件URL，如视频文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi">示例</a>。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code>。</td>
</tr>
<tr>
<td>dict</td>
<td>支持传入字典类型，字典的key需与具体任务对应，如视频检测任务对应\"video\"，字典的val支持上述类型数据，例如：<code>{\"video\": \"/root/data1\"}</code>。</td>
</tr>
<tr>
<td>list</td>
<td>支持传入列表，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]，[\"/root/data/video1.mp4\", \"/root/data/video2.mp4\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code>，<code>[{\"video\": \"/root/data1\"}, {\"video\": \"/root/data2/video.mp4\"}]</code>。</td>
</tr>
</tbody>
</table>
（3）调用`predict`方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以batch为单位对数据进行预测，因此预测结果为list形式表示的一组预测结果。

（4）对预测结果进行处理：每个样本的预测结果均为`dict`类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

<table>
<thead>
<tr>
<th>方法</th>
<th>说明</th>
<th>方法参数</th>
</tr>
</thead>
<tbody>
<tr>
<td>print</td>
<td>打印结果到终端</td>
<td><code>- format_json</code>：bool类型，是否对输出内容进行使用json缩进格式化，默认为True；<br/><code>- indent</code>：int类型，json格式化设置，仅当format_json为True时有效，默认为4；<br/><code>- ensure_ascii</code>：bool类型，json格式化设置，仅当format_json为True时有效，默认为False；</td>
</tr>
<tr>
<td>save_to_json</td>
<td>将结果保存为json格式的文件</td>
<td><code>- save_path</code>：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；<br/><code>- indent</code>：int类型，json格式化设置，默认为4；<br/><code>- ensure_ascii</code>：bool类型，json格式化设置，默认为False；</td>
</tr>
<tr>
<td>save_to_video</td>
<td>将结果保存为视频格式的文件</td>
<td><code>- save_path</code>：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；</td>
</tr>
</tbody>
</table>
若您获取了配置文件，即可对视频检测产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 `./my_path/video_detection*.yaml` ，则只需执行：

```
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/video_detection.yaml")
output = pipeline.predict("HorseRiding.avi")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_video("./output/") ## 保存结果可视化视频
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 <b>高性能推理</b>：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ <b>服务化部署</b>：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/service_deploy.md)。

下面是API参考和多语言服务调用示例：

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
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。固定为<code>0</code>。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。固定为<code>"Success"</code>。</td>
</tr>
</tbody>
</table>
<p>响应体还可能有<code>result</code>属性，类型为<code>object</code>，其中存储操作结果信息。</p>
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
<p><code>POST /video-classification</code></p>
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
<td>服务可访问的视频文件的URL或视频文件内容的Base64编码结果。</td>
<td>是</td>
</tr>
<tr>
<td><code>inferenceParams</code></td>
<td><code>object</code></td>
<td>推理参数。</td>
<td>否</td>
</tr>
</tbody>
</table>
<p><code>inferenceParams</code>的属性如下：</p>
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
<td><code>score_threshold</code></td>
<td><code>integer</code></td>
<td>结果中将只保留得分高于该阈值<code>score_threshold</code>的框。</td>
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
<td><code>categories</code></td>
<td><code>array</code></td>
<td>视频类别信息。</td>
</tr>
<tr>
<td><code>video</code></td>
<td><code>string</code></td>
<td>视频检测结果图。视频为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<p><code>categories</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>id</code></td>
<td><code>integer</code></td>
<td>类别ID。</td>
</tr>
<tr>
<td><code>name</code></td>
<td><code>string</code></td>
<td>类别名称。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>类别得分。</td>
</tr>
</tbody>
</table>
<p><code>result</code>示例如下：</p>
<pre><code class="language-json">{
&quot;categories&quot;: [
{
&quot;id&quot;: 5,
&quot;name&quot;: &quot;兔子&quot;,
&quot;score&quot;: 0.93
}
],
&quot;video&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/video-classification&quot; # 服务URL
video_path = &quot;./demo.mp4&quot;
output_video_path = &quot;./out.mp4&quot;

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
with open(output_video_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;video&quot;]))
print(f&quot;Output video saved at {output_video_path}&quot;)
print(&quot;\nCategories:&quot;)
print(result[&quot;categories&quot;])
</code></pre></details>
<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // https://github.com/Huiyicc/cpp-httplib
#include &quot;nlohmann/json.hpp&quot; // https://github.com/nlohmann/json
#include &quot;base64.hpp&quot; // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string videoPath = &quot;./demo.mp4&quot;;
    const std::string outputImagePath = &quot;./out.mp4&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

    // 对本地视频进行Base64编码
    std::ifstream file(videoPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; &quot;Error reading file.&quot; &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj[&quot;video&quot;] = encodedImage;
    std::string body = jsonObj.dump();

    // 调用API
    auto response = client.Post(&quot;/video-classification&quot;, headers, body, &quot;application/json&quot;);
    // 处理接口返回数据
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedImage = result[&quot;video&quot;];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; &quot;Output video saved at &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        }

        auto categories = result[&quot;categories&quot;];
        std::cout &lt;&lt; &quot;\nCategories:&quot; &lt;&lt; std::endl;
        for (const auto&amp; category : categories) {
            std::cout &lt;&lt; category &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; &quot;Failed to send HTTP request.&quot; &lt;&lt; std::endl;
        return 1;
    }

    return 0;
}
</code></pre></details>

<details><summary>Java</summary>

<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = &quot;http://localhost:8080/video-classification&quot;; // 服务URL
        String videoPath = &quot;./demo.mp4&quot;; // 本地视频
        String outputImagePath = &quot;./out.mp4&quot;; // 输出视频

        // 对本地视频进行Base64编码
        File file = new File(videoPath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String videoData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;video&quot;, videoData); // Base64编码的文件内容或者视频URL

        // 创建 OkHttpClient 实例
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get(&quot;application/json; charset=utf-8&quot;);
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // 调用API并处理接口返回数据
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get(&quot;result&quot;);
                String base64Image = result.get(&quot;video&quot;).asText();
                JsonNode categories = result.get(&quot;categories&quot;);

                byte[] videoBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(videoBytes);
                }
                System.out.println(&quot;Output video saved at &quot; + outputImagePath);
                System.out.println(&quot;\nCategories: &quot; + categories.toString());
            } else {
                System.err.println(&quot;Request failed with code: &quot; + response.code());
            }
        }
    }
}
</code></pre></details>

<details><summary>Go</summary>

<pre><code class="language-go">package main

import (
    &quot;bytes&quot;
    &quot;encoding/base64&quot;
    &quot;encoding/json&quot;
    &quot;fmt&quot;
    &quot;io/ioutil&quot;
    &quot;net/http&quot;
)

func main() {
    API_URL := &quot;http://localhost:8080/video-classification&quot;
    videoPath := &quot;./demo.mp4&quot;
    outputImagePath := &quot;./out.mp4&quot;

    // 对本地视频进行Base64编码
    videoBytes, err := ioutil.ReadFile(videoPath)
    if err != nil {
        fmt.Println(&quot;Error reading video file:&quot;, err)
        return
    }
    videoData := base64.StdEncoding.EncodeToString(videoBytes)

    payload := map[string]string{&quot;video&quot;: videoData} // Base64编码的文件内容或者视频URL
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println(&quot;Error marshaling payload:&quot;, err)
        return
    }

    // 调用API
    client := &amp;http.Client{}
    req, err := http.NewRequest(&quot;POST&quot;, API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println(&quot;Error creating request:&quot;, err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println(&quot;Error sending request:&quot;, err)
        return
    }
    defer res.Body.Close()

    // 处理接口返回数据
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println(&quot;Error reading response body:&quot;, err)
        return
    }
    type Response struct {
        Result struct {
            Image      string   `json:&quot;video&quot;`
            Categories []map[string]interface{} `json:&quot;categories&quot;`
        } `json:&quot;result&quot;`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println(&quot;Error unmarshaling response body:&quot;, err)
        return
    }

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println(&quot;Error decoding base64 video data:&quot;, err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing video to file:&quot;, err)
        return
    }
    fmt.Printf(&quot;Image saved at %s.mp4\n&quot;, outputImagePath)
    fmt.Println(&quot;\nCategories:&quot;)
    for _, category := range respData.Result.Categories {
        fmt.Println(category)
    }
}
</code></pre></details>

<details><summary>C#</summary>

<pre><code class="language-csharp">using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = &quot;http://localhost:8080/video-classification&quot;;
    static readonly string videoPath = &quot;./demo.mp4&quot;;
    static readonly string outputImagePath = &quot;./out.mp4&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // 对本地视频进行Base64编码
        byte[] videoBytes = File.ReadAllBytes(videoPath);
        string video_data = Convert.ToBase64String(videoBytes);

        var payload = new JObject{ { &quot;video&quot;, video_data } }; // Base64编码的文件内容或者视频URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        // 调用API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // 处理接口返回数据
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse[&quot;result&quot;][&quot;video&quot;].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($&quot;Output video saved at {outputImagePath}&quot;);
        Console.WriteLine(&quot;\nCategories:&quot;);
        Console.WriteLine(jsonResponse[&quot;result&quot;][&quot;categories&quot;].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/video-classification'
const videoPath = './demo.mp4'
const outputImagePath = &quot;./out.mp4&quot;;

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'video': encodeImageToBase64(videoPath)  // Base64编码的文件内容或者视频URL
  })
};

// 对本地视频进行Base64编码
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// 调用API
axios.request(config)
.then((response) =&gt; {
    // 处理接口返回数据
    const result = response.data[&quot;result&quot;];
    const videoBuffer = Buffer.from(result[&quot;video&quot;], 'base64');
    fs.writeFile(outputImagePath, videoBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output video saved at ${outputImagePath}`);
    });
    console.log(&quot;\nCategories:&quot;);
    console.log(result[&quot;categories&quot;]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>
<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/video-classification&quot;; // 服务URL
$video_path = &quot;./demo.mp4&quot;;
$output_video_path = &quot;./out.mp4&quot;;

// 对本地视频进行Base64编码
$video_data = base64_encode(file_get_contents($video_path));
$payload = array(&quot;video&quot; =&gt; $video_data); // Base64编码的文件内容或者视频URL

// 调用API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// 处理接口返回数据
$result = json_decode($response, true)[&quot;result&quot;];
file_put_contents($output_video_path, base64_decode($result[&quot;video&quot;]));
echo &quot;Output video saved at &quot; . $output_video_path . &quot;\n&quot;;
echo &quot;\nCategories:\n&quot;;
print_r($result[&quot;categories&quot;]);
?&gt;
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

例如，您使用英伟达 GPU 进行视频检测产线的推理，使用的命令为：

```bash
paddlex --pipeline video_detection --input HorseRiding.avi --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需将 `--device` 修改为 npu:0 即可：

```bash
paddlex --pipeline video_detection --input HorseRiding.avi --device npu:0
```
若您想在更多种类的硬件上使用通用视频检测产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
