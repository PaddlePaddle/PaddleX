简体中文 | [English](human_keypoint_detection.en.md)

# 人体关键点检测产线使用教程

## 1. 人体关键点检测产线介绍

人体关键点检测旨在通过识别和定位人体的特定关节和部位，来实现对人体姿态和动作的分析。该任务不仅需要在图像中检测出人体，还需要精确获取人体的关键点位置，如肩膀、肘部、膝盖等，从而进行姿态估计和行为识别。人体关键点检测广泛应用于运动分析、健康监测、动画制作和人机交互等场景。

PaddleX 的人体关键点检测产线是一个 Top-Down 方案，由行人检测和关键点检测两个模块组成，针对移动端设备优化，可精确流畅地在移动端设备上执行多人姿态估计任务。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/human_keypoint_detection/01.jpg">

<b>人体关键点检测产线中包含了行人检测模块和人体关键点检测模块</b>，有若干模型可供选择，您可以根据下边的 benchmark 数据来选择使用的模型。<b>如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。

<summary> 👉模型列表详情</summary>

<b>行人检测模块：</b>

<table>
  <tr>
    <th >模型</th>
    <th >mAP(0.5:0.95)</th>
    <th >mAP(0.5)</th>
    <th >GPU推理耗时（ms）</th>
    <th >CPU推理耗时 (ms)</th>
    <th >模型存储大小（M）</th>
    <th >介绍</th>
  </tr>
  <tr>
    <td>PP-YOLOE-L_human</td>
    <td>48.0</td>
    <td>81.9</td>
    <td>32.8</td>
    <td>777.7</td>
    <td>196.02</td>
    <td rowspan="2">基于PP-YOLOE的行人检测模型</td>
  </tr>
  <tr>
    <td>PP-YOLOE-S_human</td>
    <td>42.5</td>
    <td>77.9</td>
    <td>15.0</td>
    <td>179.3</td>
    <td>28.79</td>
  </tr>
</table>

<b>注：以上精度指标为CrowdHuman数据集 mAP(0.5:0.95)。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b>

<b>人体关键点检测模块：</b>

<table>
  <tr>
    <th >模型</th>
    <th >方案</th>
    <th >输入尺寸</th>
    <th >AP(0.5:0.95)</th>
    <th >GPU推理耗时（ms）</th>
    <th >CPU推理耗时 (ms)</th>
    <th >模型存储大小（M）</th>
    <th >介绍</th>
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

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/human_keypoint_detection/01.jpg">

#### 2.2.2 Python脚本方式集成
通过上述命令行方式可快速体验查看效果，在项目中往往需要代码集成，您可以通过如下几行代码完成产线的快速推理：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="human_keypoint_detection")

output = pipeline.predict("keypoint_detection_001.jpg"， det_threshold=0.5)
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
  <li><b>str</b>：如图像文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code></li>
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

（3）对预测结果进行处理，每个样本的预测结果均为`dict`类型，且支持打印、保存为图片、保存为`json`文件的操作:

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
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
</table>

<ul><li><details><summary>👉 调用 <code>print()</code> 方法会将如下结果打印到终端（点击展开）：</summary>

```bash
{'res': {'input_path': 'keypoint_detection_001.jpg', 'boxes': [{'coordinate': [325.65088, 74.46718, 391.5512, 209.46529], 'det_score': 0.9316536784172058, 'keypoints': array([[3.52227936e+02, 8.88497543e+01, 5.41676700e-01],
       [3.51974579e+02, 8.66366196e+01, 6.21515572e-01],
       [3.52865662e+02, 8.64902344e+01, 5.60755610e-01],
       [3.50862457e+02, 8.75393066e+01, 5.66961825e-01],
       [3.57415802e+02, 8.63235092e+01, 5.54250121e-01],
       [3.39434937e+02, 9.99389191e+01, 6.28665030e-01],
       [3.69297821e+02, 1.01169853e+02, 6.00828469e-01],
       [3.36788544e+02, 1.18020966e+02, 5.31029820e-01],
       [3.71721039e+02, 1.22033646e+02, 6.07613802e-01],
       [3.66371948e+02, 1.40476746e+02, 1.52681962e-01],
       [3.67885651e+02, 1.39969559e+02, 3.90044987e-01],
       [3.47553253e+02, 1.42274353e+02, 5.21435857e-01],
       [3.60833710e+02, 1.42503479e+02, 4.64817137e-01],
       [3.40133362e+02, 1.67570465e+02, 5.92474759e-01],
       [3.74433594e+02, 1.69982712e+02, 5.48423827e-01],
       [3.37616333e+02, 1.92737564e+02, 5.85887253e-01],
       [3.82684723e+02, 1.96479385e+02, 6.19615853e-01]], dtype=float32), 'kpt_score': 0.53462815}, {'coordinate': [271.96713, 69.02892, 336.77832, 217.54662], 'det_score': 0.9304604530334473, 'keypoints': array([[2.96400787e+02, 8.58611679e+01, 7.14319646e-01],
       [2.97301758e+02, 8.28755493e+01, 7.04051554e-01],
       [2.94497406e+02, 8.29398193e+01, 6.89844370e-01],
       [3.00162109e+02, 8.35247955e+01, 5.55446565e-01],
       [2.93188751e+02, 8.33744202e+01, 6.51386738e-01],
       [3.12854675e+02, 9.81457520e+01, 7.32430100e-01],
       [2.86463226e+02, 1.01262375e+02, 6.10454917e-01],
       [3.19350311e+02, 1.18383713e+02, 5.93547344e-01],
       [2.82401520e+02, 1.21164886e+02, 5.71586847e-01],
       [3.23966248e+02, 1.39364532e+02, 5.18607676e-01],
       [2.82263916e+02, 1.44509521e+02, 2.59432912e-01],
       [3.09791840e+02, 1.43603912e+02, 6.89817309e-01],
       [2.94868561e+02, 1.44677597e+02, 5.97069323e-01],
       [3.14365845e+02, 1.74088943e+02, 6.36058152e-01],
       [2.92653442e+02, 1.75070770e+02, 5.97140312e-01],
       [3.19849792e+02, 2.02162598e+02, 6.80035114e-01],
       [2.94255920e+02, 2.03049500e+02, 6.08293772e-01]], dtype=float32), 'kpt_score': 0.6123249}, {'coordinate': [293.55933, 188.65804, 419.47382, 305.4712], 'det_score': 0.9179267883300781, 'keypoints': array([[3.33173096e+02, 2.05474487e+02, 5.18341482e-01],
       [3.36098663e+02, 2.03492996e+02, 5.60671568e-01],
       [3.37248505e+02, 2.03364868e+02, 5.00729203e-01],
       [3.40604095e+02, 2.02393539e+02, 4.98033434e-01],
       [3.43625671e+02, 2.01539536e+02, 6.13991261e-01],
       [3.24516022e+02, 2.18521667e+02, 3.14208776e-01],
       [3.52951965e+02, 2.12051971e+02, 4.42923039e-01],
       [3.14448853e+02, 2.22776672e+02, 2.55664617e-01],
       [3.65774384e+02, 2.25498718e+02, 8.26140717e-02],
       [3.06869843e+02, 2.34243729e+02, 1.35185301e-01],
       [3.08855865e+02, 2.36824249e+02, 1.00039296e-01],
       [3.82195679e+02, 2.42062302e+02, 5.45506418e-01],
       [3.88757233e+02, 2.42933960e+02, 5.79574823e-01],
       [3.50280792e+02, 2.56009766e+02, 7.92343557e-01],
       [3.70955750e+02, 2.74127930e+02, 5.06902397e-01],
       [3.61553833e+02, 2.83896454e+02, 6.03924632e-01],
       [3.94064087e+02, 2.88825836e+02, 3.72752368e-01]], dtype=float32), 'kpt_score': 0.43667096}, {'coordinate': [238.98825, 182.67476, 372.81628, 307.61395], 'det_score': 0.914400041103363, 'keypoints': array([[281.63354   , 197.61014   ,   0.76263565],
       [283.38297   , 195.05458   ,   0.8535259 ],
       [277.73865   , 192.96712   ,   0.7486459 ],
       [282.2258    , 197.37396   ,   0.5293759 ],
       [267.41278   , 193.56656   ,   0.6177453 ],
       [269.7986    , 215.17424   ,   0.6587688 ],
       [259.27332   , 214.76183   ,   0.7745857 ],
       [277.53683   , 237.42062   ,   0.48790172],
       [260.1237    , 240.18477   ,   0.44012186],
       [293.51572   , 250.89894   ,   0.49827316],
       [290.2274    , 252.19504   ,   0.58322966],
       [279.096     , 260.06042   ,   0.6834831 ],
       [269.9528    , 265.9034    ,   0.74632865],
       [313.1393    , 260.79523   ,   0.6337413 ],
       [310.01425   , 262.5001    ,   0.7376388 ],
       [348.17996   , 283.56802   ,   0.6096815 ],
       [334.12622   , 292.06284   ,   0.805234  ]], dtype=float32), 'kpt_score': 0.6571127}, {'coordinate': [66.23172, 93.531204, 124.48463, 217.99655], 'det_score': 0.9086756110191345, 'keypoints': array([[ 91.31993   , 108.413284  ,   0.615049  ],
       [ 92.08924   , 106.03603   ,   0.63400346],
       [ 88.434235  , 105.775925  ,   0.6342117 ],
       [ 94.41964   , 106.27531   ,   0.5885308 ],
       [ 84.07658   , 105.80367   ,   0.6773151 ],
       [100.38561   , 118.80038   ,   0.74734527],
       [ 79.46563   , 119.58027   ,   0.7821885 ],
       [102.27228   , 136.0127    ,   0.5907435 ],
       [ 73.76375   , 135.51848   ,   0.7132327 ],
       [101.763245  , 148.3819    ,   0.35871926],
       [ 72.33199   , 148.83861   ,   0.5405212 ],
       [ 99.1199    , 151.6756    ,   0.83278877],
       [ 86.4599    , 152.03287   ,   0.78065455],
       [106.40269   , 176.46979   ,   0.75466657],
       [ 84.909454  , 178.44617   ,   0.76010597],
       [110.97942   , 201.19633   ,   0.7917331 ],
       [ 79.87372   , 202.87093   ,   0.79150075]], dtype=float32), 'kpt_score': 0.68195945}, {'coordinate': [160.1294, 78.35935, 212.01868, 153.2241], 'det_score': 0.8295672535896301, 'keypoints': array([[1.8115924e+02, 1.0371443e+02, 2.5303254e-01],
       [1.8318883e+02, 9.6959526e+01, 1.5748371e-01],
       [1.8303551e+02, 9.8071350e+01, 2.1673845e-01],
       [1.8769695e+02, 9.1113632e+01, 1.4220884e-01],
       [1.8665227e+02, 9.1424126e+01, 1.2998220e-01],
       [1.9881558e+02, 9.9527107e+01, 2.6830634e-01],
       [1.8619264e+02, 9.9051491e+01, 1.8842754e-01],
       [1.9972902e+02, 1.2386106e+02, 4.7812667e-01],
       [1.8038458e+02, 1.2146417e+02, 1.7550260e-01],
       [1.8155409e+02, 1.3735040e+02, 3.2514629e-01],
       [1.7971712e+02, 1.3371999e+02, 1.1313542e-01],
       [1.9606516e+02, 1.4140919e+02, 2.3604973e-01],
       [1.8650092e+02, 1.4260675e+02, 1.3515554e-01],
       [1.9617020e+02, 1.2273723e+02, 9.6943676e-02],
       [1.6671684e+02, 1.2564886e+02, 1.2711491e-01],
       [1.8317670e+02, 1.3923177e+02, 1.0834377e-01],
       [1.7997801e+02, 1.3850316e+02, 9.3360245e-02]], dtype=float32), 'kpt_score': 0.19088578}, {'coordinate': [52.482475, 59.36664, 96.47121, 135.45993], 'det_score': 0.7726763486862183, 'keypoints': array([[7.38075943e+01, 7.33277664e+01, 1.63257480e-01],
       [7.37732239e+01, 7.15934525e+01, 1.55248597e-01],
       [7.20166702e+01, 7.13588028e+01, 1.96659654e-01],
       [6.95982971e+01, 7.10932083e+01, 1.26999229e-01],
       [6.98693237e+01, 7.16391983e+01, 1.35578454e-01],
       [8.22228088e+01, 7.77278976e+01, 2.35379949e-01],
       [6.47475586e+01, 7.88423233e+01, 2.10787609e-01],
       [8.33889618e+01, 9.05893707e+01, 2.98420697e-01],
       [8.30510330e+01, 9.10888824e+01, 1.13309503e-01],
       [8.09216843e+01, 9.85093231e+01, 1.84670463e-01],
       [7.77405396e+01, 1.01128220e+02, 1.49517819e-01],
       [7.58817215e+01, 1.02311646e+02, 7.63842911e-02],
       [6.97640839e+01, 1.01978600e+02, 9.00617689e-02],
       [8.89746017e+01, 9.87928925e+01, 2.00097740e-01],
       [6.45541153e+01, 1.04130150e+02, 1.00787796e-01],
       [8.81069489e+01, 1.19858818e+02, 1.84717909e-01],
       [7.08108673e+01, 1.38108154e+02, 9.07213315e-02]], dtype=float32), 'kpt_score': 0.15956473}, {'coordinate': [7.081953, 80.3705, 46.81927, 161.72012], 'det_score': 0.6587498784065247, 'keypoints': array([[2.68747215e+01, 9.24635468e+01, 3.17601502e-01],
       [2.71188889e+01, 9.08834305e+01, 2.46545449e-01],
       [2.69905357e+01, 9.03851013e+01, 3.27626228e-01],
       [2.34424419e+01, 8.87997513e+01, 2.75899678e-01],
       [3.25261230e+01, 8.93143845e+01, 3.42958122e-01],
       [1.98818531e+01, 9.67693405e+01, 2.83849210e-01],
       [3.82729301e+01, 9.91002884e+01, 3.19851965e-01],
       [1.63669338e+01, 1.10157967e+02, 1.96907550e-01],
       [4.11151352e+01, 1.10147133e+02, 2.70823181e-01],
       [1.86983719e+01, 1.17376358e+02, 1.15746662e-01],
       [1.98090267e+01, 1.16526924e+02, 1.02475688e-01],
       [2.51145611e+01, 1.23519379e+02, 3.24807376e-01],
       [3.34709854e+01, 1.24678688e+02, 2.65269905e-01],
       [2.82129307e+01, 1.42196121e+02, 2.98054874e-01],
       [2.94088726e+01, 1.42497360e+02, 3.57838601e-01],
       [2.95637341e+01, 1.57201065e+02, 2.14882523e-01],
       [3.03766575e+01, 1.57535706e+02, 2.10423440e-01]], dtype=float32), 'kpt_score': 0.26303306}, {'coordinate': [126.131096, 30.263107, 168.5759, 134.09885], 'det_score': 0.6441988348960876, 'keypoints': array([[148.10135   ,  40.584896  ,   0.44685563],
       [150.00848   ,  38.423157  ,   0.5721373 ],
       [146.84933   ,  38.88104   ,   0.5450204 ],
       [153.57166   ,  38.53051   ,   0.58872294],
       [144.609     ,  38.854267  ,   0.54383296],
       [158.78825   ,  51.609245  ,   0.6847385 ],
       [141.20293   ,  49.92705   ,   0.52605945],
       [157.85371   ,  70.32525   ,   0.7879656 ],
       [136.42497   ,  68.15052   ,   0.4752547 ],
       [144.46915   ,  79.161385  ,   0.5807479 ],
       [135.84734   ,  75.86984   ,   0.32583416],
       [155.16513   ,  78.74157   ,   0.56849873],
       [141.66093   ,  77.88423   ,   0.45576522],
       [152.68689   , 100.64953   ,   0.5331878 ],
       [130.97485   ,  87.03784   ,   0.73304355],
       [153.57033   , 123.742294  ,   0.39946193],
       [132.91501   , 114.52923   ,   0.36085486]], dtype=float32), 'kpt_score': 0.5369401}, {'coordinate': [112.50212, 64.127, 150.35353, 125.85529], 'det_score': 0.5013833045959473, 'keypoints': array([[1.34417511e+02, 7.67317352e+01, 8.11196864e-02],
       [1.33561203e+02, 7.61824722e+01, 5.88811524e-02],
       [1.33302322e+02, 7.54709702e+01, 3.77583615e-02],
       [1.33238602e+02, 7.65276260e+01, 8.76586884e-02],
       [1.27832169e+02, 7.29993439e+01, 5.68802767e-02],
       [1.32234711e+02, 8.55900650e+01, 6.25893995e-02],
       [1.29263702e+02, 8.66081772e+01, 7.35298842e-02],
       [1.17821297e+02, 6.38808823e+01, 1.47604376e-01],
       [1.17373665e+02, 6.40265808e+01, 1.84295043e-01],
       [1.39441742e+02, 9.73589020e+01, 6.45915419e-02],
       [1.24748589e+02, 1.04544739e+02, 5.86665794e-02],
       [1.35098206e+02, 7.81749954e+01, 8.30232948e-02],
       [1.34638489e+02, 7.91068802e+01, 8.16871747e-02],
       [1.36119888e+02, 8.93165436e+01, 1.34096310e-01],
       [1.30067749e+02, 9.34937820e+01, 8.98712277e-02],
       [1.36004654e+02, 1.16780487e+02, 1.60800099e-01],
       [1.28087891e+02, 1.15956802e+02, 1.99588016e-01]], dtype=float32), 'kpt_score': 0.097802415}]}}
```
</details></li></ul>

- 输出结果参数含义如下：
    - `input_path`：表示输入图像的路径
    - `boxes`：检测到人体信息，一个字典列表，每个字典包含以下信息：
        - `coordinate`：人体目标框坐标，格式为[xmin, ymin, xmax, ymax]
        - `det_score`：人体目标框置信度
        - `keypoints`：关键点坐标信息，一个numpy数组，形状为[num_keypoints, 3]，其中每个关键点由[x, y, score]组成，score为该关键点的置信度
        - `kpt_score`：关键点整体的置信度，即关键点的平均置信度

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。
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
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">获取预测的 <code>json</code> 格式的结果</td>
</tr>
<tr>
<td rowspan = "2"><code>img</code></td>
<td rowspan = "2">获取格式为 <code>dict</code> 的可视化图像</td>
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

</details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/ocr&quot; # 服务URL
image_path = &quot;./demo.jpg&quot;
output_image_path = &quot;./out.jpg&quot;

# 对本地图像进行Base64编码
with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}  # Base64编码的文件内容或者图像URL

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
print(&quot;\nDetected texts:&quot;)
print(result[&quot;texts&quot;])
</code></pre>
</details>
</details>


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

若您想在更多种类的硬件上使用通用图像识别产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
