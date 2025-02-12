# 人体关键点检测产线使用教程

## 1. 人体关键点检测产线介绍

人体关键点检测旨在通过识别和定位人体的特定关节和部位，来实现对人体姿态和动作的分析。该任务不仅需要在图像中检测出人体，还需要精确获取人体的关键点位置，如肩膀、肘部、膝盖等，从而进行姿态估计和行为识别。人体关键点检测广泛应用于运动分析、健康监测、动画制作和人机交互等场景。

PaddleX 的人体关键点检测产线是一个 Top-Down 方案，由行人检测和关键点检测两个模块组成，针对移动端设备优化，可精确流畅地在移动端设备上执行多人姿态估计任务。本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/human_keypoint_detection/01.jpg"/>
<b>人体关键点检测产线中包含了行人检测模块和人体关键点检测模块</b>，有若干模型可供选择，您可以根据下边的 benchmark 数据来选择使用的模型。<b>如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。

<details><summary> 👉模型列表详情</summary>
<b>行人检测模块：</b>
<table>
<tr>
<th>模型</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-YOLOE-L_human</td>
<td>48.0</td>
<td>81.9</td>
<td>33.27 / 9.19</td>
<td>173.72 / 173.72</td>
<td>196.02</td>
<td rowspan="2">基于PP-YOLOE的行人检测模型</td>
</tr>
<tr>
<td>PP-YOLOE-S_human</td>
<td>42.5</td>
<td>77.9</td>
<td>9.94 / 3.42</td>
<td>54.48 / 46.52</td>
<td>28.79</td>
</tr>
</table>
<b>注：以上精度指标为CrowdHuman数据集 mAP(0.5:0.95)。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b>
<b>人体关键点检测模块：</b>
<table>
<tr>
<th>模型</th>
<th>方案</th>
<th>输入尺寸</th>
<th>AP(0.5:0.95)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-TinyPose_128x96</td>
<td>Top-Down</td>
<td>128*96</td>
<td>58.4</td>
<td></td>
<td></td>
<td>4.9</td>
<td rowspan="2">PP-TinyPose 是百度飞桨视觉团队自研的针对移动端设备优化的实时关键点检测模型，可流畅地在移动端设备上执行多人姿态估计任务</td>
</tr>
<tr>
<td>PP-TinyPose_256x192</td>
<td>Top-Down</td>
<td>256*192</td>
<td>68.3</td>
<td></td>
<td></td>
<td>4.9</td>
</tr>
</table>
<b>注：以上精度指标为COCO数据集 AP(0.5:0.95)，所依赖的检测框为ground truth标注得到。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b>
</details>

## 2. 快速开始

PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在本地使用 Python 体验通用图像识别产线的效果。

### 2.1 在线体验

暂不支持在线体验。

### 2.2 本地体验

> ❗ 在本地使用人体关键点检测产线前，请确保您已经按照[PaddleX安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.2.1 命令行方式体验

一行命令即可快速体验人体关键点检测产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/keypoint_detection_001.jpg)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline human_keypoint_detection \
        --input keypoint_detection_001.jpg \
        --det_threshold 0.5 \
        --save_path ./output/ \
        --device gpu:0
```
相关参数和运行结果说明可以参考[2.2.2 Python脚本方式集成](#222-python脚本方式集成)中的参数说明和结果解释。

可视化结果保存至`save_path`，如下所示：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/human_keypoint_detection/01.jpg"/>

#### 2.2.2 Python脚本方式集成
通过上述命令行方式可快速体验查看效果，在项目中往往需要代码集成，您可以通过如下几行代码完成产线的快速推理：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="human_keypoint_detection")

output = pipeline.predict("keypoint_detection_001.jpg", det_threshold=0.5)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）调用 `create_pipeline` 实例化产线对象：具体参数说明如下：

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
<td><code>None</code></td>
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

（2）调用人体关键点检测产线对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：

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
<td>待预测数据，支持多种输入类型，必需参数</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
<li><b>str</b>：如图像文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code></li>
<li><b>List</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>人体检测模型阈值</td>
<td><code>float|None</code></td>
<td>
<ul>
<li><b>float</b>：如<code>0.5</code>， 表示过滤掉所有阈值小于<code>0.5</code>的目标框；</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>0.5</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</table>

（3）对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为`json`文件的操作:

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
<tr>
<td><code>save_to_img()</code></td>
<td>将结果保存为图像格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
</table>
<ul><li><details><summary>👉 调用 <code>print()</code> 方法会将如下结果打印到终端（点击展开）：</summary>

```bash
{'res': {'input_path': 'keypoint_detection_001.jpg', 'boxes': [{'coordinate': [325.65088, 74.46718, 391.5512, 209.46529], 'det_score': 0.9316536784172058, 'keypoints': array([[351.6419    ,  84.80058   ,   0.79337054],
       [353.9377    ,  82.47209   ,   0.7778817 ],
       [349.12946   ,  83.09801   ,   0.7885327 ],
       [359.24466   ,  83.369225  ,   0.80503   ],
       [347.46167   ,  84.1535    ,   0.8710606 ],
       [368.82172   , 101.33514   ,   0.88625187],
       [339.8064    ,  99.65537   ,   0.8432633 ],
       [371.2092    , 123.35563   ,   0.7728337 ],
       [337.78214   , 121.36371   ,   0.9310819 ],
       [368.81366   , 142.71593   ,   0.79723483],
       [337.53455   , 139.85892   ,   0.877297  ],
       [363.0265    , 141.82988   ,   0.7964988 ],
       [345.3075    , 141.98972   ,   0.7532031 ],
       [374.60806   , 171.42578   ,   0.7530604 ],
       [339.11694   , 167.98814   ,   0.7255032 ],
       [382.67047   , 197.82553   ,   0.73685765],
       [336.79745   , 196.5194    ,   0.626142  ]], dtype=float32), 'kpt_score': 0.7961825}, {'coordinate': [271.96713, 69.02892, 336.77832, 217.54662], 'det_score': 0.9304604530334473, 'keypoints': array([[294.48553   ,  84.144104  ,   0.74851245],
       [297.09854   ,  80.97825   ,   0.7341483 ],
       [292.39313   ,  81.7721    ,   0.74603605],
       [302.3231    ,  81.528275  ,   0.7586238 ],
       [290.6292    ,  83.26544   ,   0.7514231 ],
       [313.32928   ,  98.40588   ,   0.83778954],
       [286.23532   , 101.702194  ,   0.91927457],
       [321.99515   , 120.05991   ,   0.90197486],
       [282.39294   , 122.16547   ,   0.74502975],
       [327.164     , 141.25995   ,   0.8172762 ],
       [279.1632    , 133.16023   ,   0.59161717],
       [311.02557   , 142.31526   ,   0.82111686],
       [294.72357   , 143.42067   ,   0.71559554],
       [313.98828   , 174.17151   ,   0.7495116 ],
       [291.76605   , 174.39961   ,   0.7645517 ],
       [321.4924    , 202.4499    ,   0.7817023 ],
       [293.70663   , 204.9227    ,   0.72405976]], dtype=float32), 'kpt_score': 0.77107316}, {'coordinate': [293.55933, 188.65804, 419.47382, 305.4712], 'det_score': 0.9179267883300781, 'keypoints': array([[3.3565637e+02, 2.0941801e+02, 8.1438643e-01],
       [3.3636591e+02, 2.0724442e+02, 7.7529407e-01],
       [3.3486487e+02, 2.0653752e+02, 8.3719862e-01],
       [3.4387805e+02, 2.0405179e+02, 7.9793924e-01],
       [3.4104437e+02, 2.0354083e+02, 6.7090714e-01],
       [3.5167136e+02, 2.1253050e+02, 5.9533423e-01],
       [3.5493774e+02, 2.1316977e+02, 5.1632988e-01],
       [3.2814764e+02, 2.1943013e+02, 5.3697169e-01],
       [3.2577945e+02, 2.2027420e+02, 1.6555195e-01],
       [3.1541614e+02, 2.2199020e+02, 5.2568728e-01],
       [3.1139435e+02, 2.2925937e+02, 2.2075935e-01],
       [3.8441351e+02, 2.4341478e+02, 6.4083064e-01],
       [3.8714008e+02, 2.4532764e+02, 6.4894527e-01],
       [3.5143246e+02, 2.5615021e+02, 7.7424920e-01],
       [3.7133820e+02, 2.7552402e+02, 5.8704698e-01],
       [3.6274625e+02, 2.8303183e+02, 6.1670756e-01],
       [4.0358893e+02, 2.9351334e+02, 4.2383862e-01]], dtype=float32), 'kpt_score': 0.5969399}, {'coordinate': [238.98825, 182.67476, 372.81628, 307.61395], 'det_score': 0.914400041103363, 'keypoints': array([[282.9012    , 208.31485   ,   0.6685285 ],
       [282.95908   , 204.36131   ,   0.66104335],
       [280.90683   , 204.54018   ,   0.7281709 ],
       [274.7831    , 204.04141   ,   0.54541856],
       [270.97324   , 203.04889   ,   0.73486483],
       [269.43472   , 217.63014   ,   0.6707946 ],
       [256.871     , 216.546     ,   0.89603853],
       [277.03226   , 238.2196    ,   0.4412233 ],
       [262.29578   , 241.33434   ,   0.791063  ],
       [292.90753   , 251.69914   ,   0.4993091 ],
       [285.6907    , 252.71925   ,   0.7215052 ],
       [279.36578   , 261.8949    ,   0.6626504 ],
       [270.43402   , 268.07068   ,   0.80625033],
       [311.96924   , 261.36716   ,   0.67315185],
       [309.32407   , 262.97354   ,   0.72746485],
       [345.22446   , 285.02255   ,   0.60142016],
       [334.69235   , 291.57108   ,   0.7674925 ]], dtype=float32), 'kpt_score': 0.6821406}, {'coordinate': [66.23172, 93.531204, 124.48463, 217.99655], 'det_score': 0.9086756110191345, 'keypoints': array([[ 91.04524   , 108.79487   ,   0.8234256 ],
       [ 92.67917   , 106.63517   ,   0.79848343],
       [ 88.41122   , 106.8017    ,   0.8122996 ],
       [ 95.353096  , 106.96488   ,   0.85210425],
       [ 84.35098   , 107.85205   ,   0.971826  ],
       [ 99.92103   , 119.87272   ,   0.853371  ],
       [ 79.69138   , 121.08684   ,   0.8854925 ],
       [103.019554  , 135.00996   ,   0.73513967],
       [ 72.38997   , 136.8782    ,   0.7727014 ],
       [104.561935  , 146.01869   ,   0.8377464 ],
       [ 72.70636   , 151.44576   ,   0.67577386],
       [ 98.69484   , 151.30742   ,   0.8381225 ],
       [ 85.946     , 152.07056   ,   0.7904873 ],
       [106.64397   , 175.77159   ,   0.8179414 ],
       [ 84.6963    , 178.4353    ,   0.8094256 ],
       [111.30463   , 201.2306    ,   0.74394226],
       [ 80.08708   , 204.05814   ,   0.8457697 ]], dtype=float32), 'kpt_score': 0.8155325}, {'coordinate': [160.1294, 78.35935, 212.01868, 153.2241], 'det_score': 0.8295672535896301, 'keypoints': array([[1.89240387e+02, 9.08055573e+01, 7.36447990e-01],
       [1.91318649e+02, 8.84640198e+01, 7.86390483e-01],
       [1.87943207e+02, 8.88532104e+01, 8.23230743e-01],
       [1.95832245e+02, 8.76751480e+01, 6.76276207e-01],
       [1.86741409e+02, 8.96744080e+01, 7.87400603e-01],
       [2.04019852e+02, 9.83068924e+01, 7.34004617e-01],
       [1.85355087e+02, 9.81262970e+01, 6.23330474e-01],
       [2.01501678e+02, 1.12709480e+02, 2.93740422e-01],
       [1.80446320e+02, 1.11967369e+02, 5.50001860e-01],
       [1.95137482e+02, 9.73322601e+01, 4.24658984e-01],
       [1.74287552e+02, 1.21760696e+02, 3.51236403e-01],
       [1.97997589e+02, 1.24219963e+02, 3.45360219e-01],
       [1.83250824e+02, 1.22610085e+02, 4.38733459e-01],
       [1.96233871e+02, 1.22864418e+02, 5.36903977e-01],
       [1.66795364e+02, 1.25634903e+02, 3.78726840e-01],
       [1.80727753e+02, 1.42604034e+02, 2.78717279e-01],
       [1.75880920e+02, 1.41181213e+02, 1.70833692e-01]], dtype=float32), 'kpt_score': 0.5256467}, {'coordinate': [52.482475, 59.36664, 96.47121, 135.45993], 'det_score': 0.7726763486862183, 'keypoints': array([[ 73.98227   ,  74.01257   ,   0.71940714],
       [ 75.44208   ,  71.73432   ,   0.6955297 ],
       [ 72.20365   ,  71.9637    ,   0.6138198 ],
       [ 77.7856    ,  71.665825  ,   0.73568064],
       [ 69.342285  ,  72.25549   ,   0.6311799 ],
       [ 83.1019    ,  77.65522   ,   0.7037722 ],
       [ 64.89729   ,  78.846565  ,   0.56623787],
       [ 85.16928   ,  88.88764   ,   0.5665537 ],
       [ 61.65655   ,  89.35312   ,   0.4463089 ],
       [ 80.01986   ,  91.51777   ,   0.30305162],
       [ 70.90767   ,  89.90153   ,   0.48063472],
       [ 78.70658   ,  97.33488   ,   0.39359188],
       [ 68.3219    ,  97.67902   ,   0.41903985],
       [ 80.69448   , 109.193985  ,   0.14496553],
       [ 65.57641   , 105.08109   ,   0.27744702],
       [ 79.44859   , 122.69015   ,   0.17710638],
       [ 64.03736   , 120.170425  ,   0.46565098]], dtype=float32), 'kpt_score': 0.4905869}, {'coordinate': [7.081953, 80.3705, 46.81927, 161.72012], 'det_score': 0.6587498784065247, 'keypoints': array([[ 29.51531   ,  91.49908   ,   0.75517464],
       [ 31.225754  ,  89.82169   ,   0.7765606 ],
       [ 27.376017  ,  89.71614   ,   0.80448   ],
       [ 33.515877  ,  90.82257   ,   0.7093001 ],
       [ 23.521307  ,  90.84212   ,   0.777707  ],
       [ 37.539314  , 101.381516  ,   0.6913692 ],
       [ 18.340288  , 102.41546   ,   0.7203535 ],
       [ 39.826218  , 113.37301   ,   0.5913918 ],
       [ 16.857304  , 115.10882   ,   0.5492331 ],
       [ 28.826103  , 121.861855  ,   0.39205936],
       [ 22.47133   , 120.69003   ,   0.6120081 ],
       [ 34.177963  , 126.15756   ,   0.5601723 ],
       [ 21.39047   , 125.30078   ,   0.5064371 ],
       [ 27.961575  , 133.33154   ,   0.54826814],
       [ 22.303364  , 129.8608    ,   0.2293001 ],
       [ 31.242027  , 153.047     ,   0.36292207],
       [ 21.80127   , 153.78947   ,   0.30531448]], dtype=float32), 'kpt_score': 0.58188534}, {'coordinate': [126.131096, 30.263107, 168.5759, 134.09885], 'det_score': 0.6441988348960876, 'keypoints': array([[149.89236   ,  43.87846   ,   0.75441885],
       [151.99484   ,  41.95912   ,   0.82070917],
       [148.18002   ,  41.775055  ,   0.8453321 ],
       [155.37967   ,  42.06968   ,   0.83349544],
       [145.38167   ,  41.69159   ,   0.8233239 ],
       [159.26329   ,  53.284737  ,   0.86246717],
       [142.35178   ,  51.206886  ,   0.6940705 ],
       [157.3975    ,  71.31917   ,   0.7624757 ],
       [136.59795   ,  66.40522   ,   0.55612797],
       [142.90988   ,  78.28269   ,   0.779243  ],
       [135.43607   ,  73.9765    ,   0.5737738 ],
       [155.7851    ,  82.44225   ,   0.6966109 ],
       [143.4588    ,  80.91763   ,   0.60589534],
       [153.45274   , 102.84818   ,   0.62720954],
       [131.59738   ,  87.54947   ,   0.4976839 ],
       [155.56401   , 125.58888   ,   0.5414401 ],
       [139.57607   , 122.08866   ,   0.26570275]], dtype=float32), 'kpt_score': 0.67882234}, {'coordinate': [112.50212, 64.127, 150.35353, 125.85529], 'det_score': 0.5013833045959473, 'keypoints': array([[1.35197662e+02, 7.29378281e+01, 5.58694899e-01],
       [1.36285202e+02, 7.16439133e+01, 6.38598502e-01],
       [1.33776855e+02, 7.16437454e+01, 6.36756659e-01],
       [1.37833389e+02, 7.24015121e+01, 4.13749218e-01],
       [1.31340057e+02, 7.30362549e+01, 5.70683837e-01],
       [1.42542435e+02, 8.28875885e+01, 2.30803847e-01],
       [1.29773300e+02, 8.52729874e+01, 4.94463116e-01],
       [1.41332916e+02, 9.43963928e+01, 9.36751068e-02],
       [1.28858521e+02, 9.95147858e+01, 2.72373617e-01],
       [1.44981277e+02, 7.83604965e+01, 8.68032947e-02],
       [1.34379593e+02, 8.23366165e+01, 1.67876005e-01],
       [1.37895874e+02, 1.08476562e+02, 1.58305198e-01],
       [1.30837265e+02, 1.07525513e+02, 1.45044222e-01],
       [1.31290604e+02, 1.02961494e+02, 7.68775940e-02],
       [1.17951675e+02, 1.07433502e+02, 2.09531561e-01],
       [1.29175934e+02, 1.14402641e+02, 1.46551579e-01],
       [1.27901909e+02, 1.16773926e+02, 2.08665460e-01]], dtype=float32), 'kpt_score': 0.3005561}]}}
```
</details></li></ul>

- 输出结果参数含义如下：
    - `input_path`：表示输入图像的路径
    - `boxes`：检测到人体信息，一个字典列表，每个字典包含以下信息：
        - `coordinate`：人体目标框坐标，格式为[xmin, ymin, xmax, ymax]
        - `det_score`：人体目标框置信度
        - `keypoints`：关键点坐标信息，一个numpy数组，形状为[num_keypoints, 3]，其中每个关键点由[x, y, score]组成，score为该关键点的置信度
        - `kpt_score`：关键点整体的置信度，即关键点的平均置信度

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)

* 此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">获取预测的 <code>json</code> 格式的结果</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">获取格式为 <code>dict</code> 的可视化图像</td>
</tr>
</table>

- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。
- `img` 属性返回的预测结果是一个字典类型的数据。键为 `res` ，对应的值是一个用于可视化人体关键点检测结果的 `Image.Image` 对象。

上述Python脚本集成方式默认使用 PaddleX 官方配置文件中的参数设置，若您需要自定义配置文件，可先执行如下命令获取官方配置文件，并保存在 `my_path` 中：

```bash
paddlex --get_pipeline_config human_keypoint_detection --save_path ./my_path
```

若您获取了配置文件，即可对人体关键点检测产线各项配置进行自定义。只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为自定义产线配置文件路径即可。

例如，若您的自定义配置文件保存在 `./my_path/human_keypoint_detection.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/human_keypoint_detection.yaml")
output = pipeline.predict("keypoint_detection_001.jpg")
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

## 3. 开发集成/部署

如果人体关键点检测产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将通用图像识别产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

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
<p>对图像进行人体关键点检测。</p>
<p><code>POST /human-keypoint-detection</code></p>
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
<td><code>image</code></td>
<td><code>string</code></td>
<td>服务器可访问的图像文件的URL或图像文件内容的Base64编码结果。</td>
<td>是</td>
</tr>
<tr>
<td><code>detThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>人体检测模型阈值</td>
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
<td><code>image</code></td>
<td><code>string</code></td>
<td>服务器可访问的图像文件的URL或图像文件内容的Base64编码结果。</td>
</tr>
<tr>
<td><code>persons</code></td>
<td><code>array</code></td>
<td>人体关键点检测结果。</td>
</tr>
</tbody>
</table>
<p><code>persons</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>kpts</code></td>
<td><code>array</code></td>
<td>关键点坐标。</td>
</tr>
<tr>
<td><code>detScore</code></td>
<td><code>number</code></td>
<td>检测得分。</td>
</tr>
<tr>
<td><code>kptScore</code></td>
<td><code>number</code></td>
<td>关键点得分。</td>
</tr>
</tbody>
</table>
</details>
<details><summary>多语言调用服务示例</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/human-keypoint-detection" # 服务URL
image_path = "./demo.jpg"
output_image_path = "./out.jpg"

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data}  # Base64编码的文件内容或者图像URL

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()["result"]
with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected persons:")
print(result["persons"])
</code></pre>
</details>
</details>
<br />

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。


## 4. 二次开发

如果人体关键点检测产线提供的默认模型权重在您的场景中精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升该产线的在您的场景中的识别效果。

### 4.1 模型微调

由于人体关键点检测产线包含两个模块（行人检测模块和人体关键点检测模块），模型产线的效果不及预期可能来自于其中任何一个模块。

您可以对识别效果差的图片进行分析，如果在分析过程中发现有较多的行人目标未被检测出来，那么可能是行人检测模型存在不足，您需要参考[行人检测模块开发教程](../../../module_usage/tutorials/cv_modules/human_detection.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/human_detection.md#四二次开发)章节，使用您的私有数据集对行人检测模型进行微调；如果在已检测到行人出现关键点检测错误，这表明关键点检测模型需要进一步改进，您需要参考[关键点检测模块开发教程](../../../module_usage/tutorials/cv_modules/human_keypoint_detection.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/human_keypoint_detection.md#四二次开发)章节,对关键点检测模型进行微调。

### 4.2 模型应用

当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```yaml
pipeline_name: human_keypoint_detection

SubModules:
  ObjectDetection:
    module_name: object_detection
    model_name: PP-YOLOE-S_human
    model_dir: null #可修改为微调后行人检测模型的本地路径
    batch_size: 1
    threshold: null
    img_size: null
  KeypointDetection:
    module_name: keypoint_detection
    model_name: PP-TinyPose_128x96
    model_dir: #可修改为微调后关键点检测模型的本地路径
    batch_size: 1
    flip: False
    use_udp: null
```
随后， 参考[2.2 本地体验](#22-本地体验)中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持

PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU 和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device`参数</b>即可完成不同硬件之间的无缝切换。

例如，使用昇腾 NPU 进行人体关键点检测产线快速推理：

```bash
paddlex --pipeline human_keypoint_detection \
        --input keypoint_detection_001.jpg \
        --det_threshold 0.5 \
        --save_path ./output/ \
        --device npu:0
```

当然，您也可以在 Python 脚本中 `create_pipeline()` 时或者 `predict()` 时指定硬件设备。

若您想在更多种类的硬件上使用通用图像识别产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
