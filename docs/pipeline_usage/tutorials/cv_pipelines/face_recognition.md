---
comments: true
---

# 人脸识别产线使用教程

## 1. 人脸识别产线介绍
人脸识别任务是计算机视觉领域的重要组成部分，旨在通过分析和比较人脸特征，实现对个人身份的自动识别。该任务不仅需要检测图像中的人脸，还需要对人脸图像进行特征提取和匹配，从而在数据库中找到对应的身份信息。人脸识别广泛应用于安全认证、监控系统、社交媒体和智能设备等场景。

人脸识别产线是专注于解决人脸定位和识别任务的端到端串联系统，可以从图像中快速准确地定位人脸区域、提取人脸特征，并与特征库中预先建立的特征做检索比对，从而确认身份信息。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/face_recognition/02.jpg"/>
<b>人脸识别产线中包含了人脸检测模块和人脸特征模块</b>，每个模块中包含了若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。<b>如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。

<p><b>人脸检测模块：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>AP(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>BlazeFace</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/BlazeFace_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace_pretrained.pdparams">训练模型</a></td>
<td>15.4</td>
<td>50.90 / 45.74</td>
<td>71.92 / 71.92</td>
<td>0.447</td>
<td>轻量高效的人脸检测模型</td>
</tr>
<tr>
<tr>
<td>BlazeFace-FPN-SSH</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/BlazeFace-FPN-SSH_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/BlazeFace-FPN-SSH_pretrained.pdparams">训练模型</a></td>
<td>18.7</td>
<td>58.99 / 51.75</td>
<td>87.39 / 87.39</td>
<td>0.606</td>
<td>BlazeFace的改进模型，增加FPN和SSH结构</td>
</tr>
<tr>
<td>PicoDet_LCNet_x2_5_face</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_LCNet_x2_5_face_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_LCNet_x2_5_face_pretrained.pdparams">训练模型</a></td>
<td>31.4</td>
<td>33.91 / 26.53</td>
<td>153.56 / 79.21</td>
<td>28.9</td>
<td>基于PicoDet_LCNet_x2_5的人脸检测模型</td>
</tr>
<tr>
<td>PP-YOLOE_plus-S_face</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-YOLOE_plus-S_face_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_face_pretrained.pdparams">训练模型</a></td>
<td>36.1</td>
<td>21.28 / 11.09</td>
<td>137.26 / 72.09</td>
<td>26.5</td>
<td>基于PP-YOLOE_plus-S的人脸检测模型</td>
</tr>
</tr></tbody>
</table>

<p><b>人脸特征模块：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>输出特征维度</th>
<th>Acc (%)<br/>AgeDB-30/CFP-FP/LFW</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>MobileFaceNet</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/MobileFaceNet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/MobileFaceNet_pretrained.pdparams">训练模型</a></td>
<td>128</td>
<td>96.28/96.71/99.58</td>
<td>3.31 / 0.73</td>
<td>5.93 / 1.30</td>
<td>4.1</td>
<td>基于MobileFaceNet在MS1Mv3数据集上训练的人脸特征提取模型</td>
</tr>
<tr>
<td>ResNet50_face</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ResNet50_face_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_face_pretrained.pdparams">训练模型</a></td>
<td>512</td>
<td>98.12/98.56/99.77</td>
<td>6.12 / 3.11</td>
<td>15.85 / 9.44</td>
<td>87.2</td>
<td>基于ResNet50在MS1Mv3数据集上训练的人脸特征提取模型</td>
</tr>
</tbody>
</table>

<strong>测试环境说明:</strong>

  <ul>
      <li><b>性能测试环境</b>
          <ul>
             <li><strong>测试数据集：
             </strong>
               <ul>
                 <li>人脸检测模型：COCO 格式的 WIDER-FACE 验证集上，以640*640作为输入尺寸评估得到的。</li>
                 <li>人脸特征模型：分别在 AgeDB-30、CFP-FP 和 LFW 数据集。</li>
               </ul>
             </li>
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

## 2. 快速开始
PaddleX 所提供的模型产线均可以快速体验效果，你可以在线体验人脸识别产线的效果，也可以在本地使用命令行或 Python 体验人脸识别产线的效果。

### 2.1 在线体验

暂不支持在线体验

### 2.2 本地体验
> ❗ 在本地使用人脸识别产线前，请确保您已经按照[PaddleX安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。如果您希望选择性安装依赖，请参考安装教程中的相关说明。该产线对应的依赖分组为 `cv`。

#### 2.2.1 命令行方式体验

暂不支持命令行体验

#### 2.2.2 Python脚本方式集成
请下载[测试图像](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/friends1.jpg)进行测试。
在该产线的运行示例中需要预先构建人脸特征库，您可以参考如下指令下载官方提供的demo数据用来后续构建人脸特征库。
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/face_demo_gallery.tar
tar -xf ./face_demo_gallery.tar
```

若您希望用私有数据集建立人脸特征库，可以参考[2.3节 构建特征库的数据组织方式](#23-构建特征库的数据组织方式)。之后通过几行代码即可完成人脸特征库建立和人脸识别产线的快速推理。

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="face_recognition")

index_data = pipeline.build_index(gallery_imgs="face_demo_gallery", gallery_label="face_demo_gallery/gallery.txt")
index_data.save("face_index")

output = pipeline.predict("friends1.jpg", index=index_data)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）调用 `create_pipeline` 实例化人脸识别产线对象。具体参数说明如下：

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
<td>是否启用高性能推理插件。如果为 <code>None</code>，则使用配置文件或 <code>config</code> 中的配置。</td>
<td><code>bool</code> | <code>None</code></td>
<td>无</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>hpi_config</code></td>
<td>高性能推理配置</td>
<td><code>dict</code> | <code>None</code></td>
<td>无</td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

（2）调用人脸识别产线对象的 `build_index` 方法，构建人脸特征库。具体参数说明如下：

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
<tbody>
<tr>
<td><code>gallery_imgs</code></td>
<td>要添加的底库图片，必需参数</td>
<td><code>str</code>|<code>list</code></td>
<td>
<ul>
<li><b>str</b>：图片根目录，数据组织方式参考<a href="#2.3-构建特征库的数据组织方式">2.3节 构建特征库的数据组织方式</a></li>
<li><b>List[numpy.ndarray]</b>：numpy.array列表类型的底库图片数据</li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>gallery_label</code></td>
<td>底库图片的标注信息，必需参数</td>
<td><code>str|list</code></td>
<td>
<ul>
<li><b>str</b>：标注文件的路径，数据组织方式与构建特征库时相同，参考<a href="#2.3-构建特征库的数据组织方式">2.3节 构建特征库的数据组织方式</a></li>
<li><b>List[str]</b>：str列表类型的底库图片标注</li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>metric_type</code></td>
<td>特征度量方式，可选参数</td>
<td><code>str</code></td>
<td>
<ul>
<li><code>"IP"</code>：内积（Inner Product）</li>
<li><code>"L2"</code>：欧几里得距离（Euclidean Distance）</li>
</ul>
</td>
<td><code>"IP"</code></td>
</tr>
<tr>
<td><code>index_type</code></td>
<td>索引类型，可选参数</td>
<td><code>str</code></td>
<td>
<ul>
<li><code>"HNSW32"</code>：检索速度较快且精度较高，但不支持 <code>remove_index()</code> 操作</li>
<li><code>"IVF"</code>：检索速度较快但精度相对较低，支持 <code>append_index()</code> 和 <code>remove_index()</code> 操作</li>
<li><code>"Flat"</code>：检索速度较低精度较高，支持 <code>append_index()</code> 和 <code>remove_index()</code> 操作</li>
</ul>
</td>
<td><code>"HNSW32"</code></td>
</tr>
</tbody>
</table>

- 特征库对象 `index` 支持 `save` 方法，用于将特征库保存到磁盘：

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
<td><code>save_path</code></td>
<td>特征库文件的保存目录，如<code>drink_index</code>。</td>
<td><code>str</code></td>
<td>无</td>
</tr>
</tbody>
</table>

（3）调用人脸识别产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`input`，用于输入待预测数据，支持多种输入方式，具体示例如下：
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
<td><code>index</code></td>
<td>产线推理预测所用的特征库，可选参数。如不传入该参数，则默认使用产线配置文件中指定的索引库。</td>
<td><code>str|paddlex.inference.components.retrieval.faiss.IndexData|None</code></td>
<td>
<ul>
<li><b>str</b>类型表示的目录（该目录下需要包含特征库文件，包括<code>vector.index</code>和<code>index_info.yaml</code>）</li>
<li><code>build_index</code>方法创建的<b>IndexData</b>对象</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
</table>

（4）对预测结果进行处理：每个样本的预测结果均为对应的Result对象，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

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

- 调用 `print()` 方法会将如下结果打印到终端：
```bash
{'res': {'input_path': 'friends1.jpg', 'boxes': [{'labels': ['Chandler', 'Chandler', 'Chandler', 'Chandler', 'Chandler'], 'rec_scores': [0.5884832143783569, 0.5777347087860107, 0.5082703828811646, 0.48792028427124023, 0.4842316806316376], 'det_score': 0.9119220972061157, 'coordinate': [790.40015, 170.34453, 868.47626, 279.54446]}, {'labels': ['Joey', 'Joey', 'Joey', 'Joey', 'Joey'], 'rec_scores': [0.5654032826423645, 0.5601680278778076, 0.5382657051086426, 0.5320160984992981, 0.5209866762161255], 'det_score': 0.9052104353904724, 'coordinate': [1274.6246, 184.58124, 1353.4016, 300.0643]}, {'labels': ['Phoebe', 'Phoebe', 'Phoebe', 'Phoebe', 'Phoebe'], 'rec_scores': [0.6462339162826538, 0.6003466844558716, 0.5999515652656555, 0.583031415939331, 0.5640993118286133], 'det_score': 0.9041699171066284, 'coordinate': [1052.4514, 192.52296, 1129.5226, 292.84177]}, {'labels': ['Ross', 'Ross', 'Ross', 'Ross', 'Ross'], 'rec_scores': [0.5012176036834717, 0.49081552028656006, 0.48970693349838257, 0.4808862805366516, 0.4794950783252716], 'det_score': 0.9031845331192017, 'coordinate': [162.41049, 156.96768, 242.07184, 266.13004]}, {'labels': ['Monica', 'Monica', 'Monica', 'Monica', 'Monica'], 'rec_scores': [0.5704089403152466, 0.5037636756896973, 0.4877302646636963, 0.46702104806900024, 0.4376206696033478], 'det_score': 0.8862134218215942, 'coordinate': [572.18176, 216.25815, 639.2387, 311.08417]}, {'labels': ['Rachel', 'Rachel', 'Rachel', 'Rachel', 'Rachel'], 'rec_scores': [0.6107711791992188, 0.5915063619613647, 0.5776835083961487, 0.569993257522583, 0.5594189167022705], 'det_score': 0.8822972774505615, 'coordinate': [303.12866, 231.94759, 374.5314, 330.2883]}]}}
```

- 输出结果参数含义如下：
    - `input_path`：表示输入图像的路径
    - `boxes`：检测到的人脸信息，一个字典列表，每个字典包含以下信息：
        - `labels`：识别标签列表，按照分数从高到低排序
        - `rec_scores`：识别分数列表，其中元素与`labels`一一对应
        - `det_score`：检测得分
        - `coordinate`：人脸框坐标，格式为[xmin, ymin, xmax, ymax]

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)，上述示例中，可视化结果如下所示：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/face_recognition/02.jpg"/>

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
- `img` 属性返回的预测结果是一个字典类型的数据。键为 `res` ，对应的值是一个用于可视化人脸识别结果的 `Image.Image` 对象。

上述Python脚本集成方式默认使用 PaddleX 官方配置文件中的参数设置，若您需要自定义配置文件，可先执行如下命令获取官方配置文件，并保存在 `my_path` 中：

```bash
paddlex --get_pipeline_config face_recognition --save_path ./my_path
```

若您获取了配置文件，即可对人脸识别产线各项配置进行自定义。只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为自定义产线配置文件路径即可。

例如，若您的自定义配置文件保存在 `./my_path/face_recognition.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/face_recognition.yaml")

output = pipeline.predict("friends1.jpg", index="face_index")
for res in output:
    res.print()
    res.save_to_json("./output/")
    res.save_to_img("./output/")
```

<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改人脸识别产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。

#### 2.2.3 人脸特征库的添加和删除操作

若您希望将更多的人脸图像添加到特征库中，则可以调用 `append_index` 方法；删除人脸图像特征，则可以调用 `remove_index` 方法。

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="face_recognition")

index_data = pipeline.build_index(gallery_imgs="face_demo_gallery", gallery_label="face_demo_gallery/gallery.txt", index_type="IVF", metric_type="IP")
index_data = pipeline.append_index(gallery_imgs="face_demo_gallery", gallery_label="face_demo_gallery/gallery.txt", index=index_data)
index_data = pipeline.remove_index(remove_ids="face_demo_gallery/remove_ids.txt", index=index_data)
index_data.save("face_index")
```

上述方法参数说明如下：
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
<tbody>
<tr>
<td><code>gallery_imgs</code></td>
<td>要添加的底库图片，必需参数</td>
<td><code>str</code>|<code>list</code></td>
<td>
<ul>
<li><b>str</b>：图片根目录，数据组织方式参考<a href="#2.3-构建特征库的数据组织方式">2.3节 构建特征库的数据组织方式</a></li>
<li><b>List[numpy.ndarray]</b>：numpy.array列表类型的底库图片数据</li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>gallery_label</code></td>
<td>底库图片的标注信息，必需参数</td>
<td><code>str|list</code></td>
<td>
<ul>
<li><b>str</b>：标注文件的路径，数据组织方式与构建特征库时相同，参考<a href="#2.3-构建特征库的数据组织方式">2.3节 构建特征库的数据组织方式</a></li>
<li><b>List[str]</b>：str列表类型的底库图片标注</li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>metric_type</code></td>
<td>特征度量方式，可选参数</td>
<td><code>str</code></td>
<td>
<ul>
<li><code>"IP"</code>：内积（Inner Product）</li>
<li><code>"L2"</code>：欧几里得距离（Euclidean Distance）</li>
</ul>
</td>
<td><code>"IP"</code></td>
</tr>
<tr>
<td><code>index_type</code></td>
<td>索引类型，可选参数</td>
<td><code>str</code></td>
<td>
<ul>
<li><code>"HNSW32"</code>：检索速度较快且精度较高，但不支持 <code>remove_index()</code> 操作</li>
<li><code>"IVF"</code>：检索速度较快但精度相对较低，支持 <code>append_index()</code> 和 <code>remove_index()</code> 操作</li>
<li><code>"Flat"</code>：检索速度较低精度较高，支持 <code>append_index()</code> 和 <code>remove_index()</code> 操作</li>
</ul>
</td>
<td><code>"HNSW32"</code></td>
</tr>
<tr>
<td><code>remove_ids</code></td>
<td>待删除的索引序号，</td>
<td><code>str</code>|<code>list</code></td>
<td>
<ul>
<li><b>str</b>：表示的txt文件的路径，内容为待删除的索引id，每行一个“id”；</li>
<li><b>List[int]</b>：表示的待删除的索引序号。仅在 <code>remove_index</code> 中有效。</li></ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>index</code></td>
<td>产线推理预测所用的特征库</td>
<td><code>str|paddlex.inference.components.retrieval.faiss.IndexData</code></td>
<td>
<ul>
<li><b>str</b>类型表示的目录（该目录下需要包含特征库文件，包括<code>vector.index</code>和<code>index_info.yaml</code>）</li>
<li><code>build_index</code>方法创建的<b>IndexData</b>对象</li>
</ul>
</td>
<td>无</td>
</tr>
</tbody>
</table>
<b>注意</b>：<code>HNSW32</code>在windows平台存在兼容性问题，可能导致索引库无法构建、加载。

### 2.3 构建特征库的数据组织方式
PaddleX的人脸识别产线示例需要使用预先构建好的特征库进行人脸特征检索。如果您希望用私有数据构建人脸特征库，则需要按照如下方式组织数据：

```bash
data_root             # 数据集根目录，目录名称可以改变
├── images            # 图像的保存目录，目录名称可以改变
│   ├── ID0           # 身份ID名字，最好是有意义的名字，比如人名
│   │   ├── xxx.jpg   # 图片，此处支持层级嵌套
│   │   ├── xxx.jpg   # 图片，此处支持层级嵌套
│   │       ...
│   ├── ID1           # 身份ID名字，最好是有意义的名字，比如人名
│   │   ├── xxx.jpg   # 图片，此处支持层级嵌套
│   │   ├── xxx.jpg   # 图片，此处支持层级嵌套
│   │       ...
│       ...
└── gallery.txt       # 特征库数据集标注文件，文件名称可以改变。每行给出待检索人脸图像路径和图像标签，使用空格分隔，内容举例：images/Chandler/Chandler00037.jpg Chandler
```
## 3. 开发集成/部署
如果人脸识别产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将人脸识别产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

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
<li><b><code>buildIndex</code></b></li>
</ul>
<p>构建特征向量索引。</p>
<p><code>POST /face-recognition-index-build</code></p>
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
<td><code>imageLabelPairs</code></td>
<td><code>array</code></td>
<td>用于构建索引的图像-标签对。</td>
<td>是</td>
</tr>
</tbody>
</table>
<p><code>imageLabelPairs</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>label</code></td>
<td><code>string</code></td>
<td>标签。</td>
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
<td><code>indexKey</code></td>
<td><code>string</code></td>
<td>索引对应的键，用于标识建立的索引。可用作其他操作的输入。</td>
</tr>
<tr>
<td><code>imageCount</code></td>
<td><code>integer</code></td>
<td>索引的图像数量。</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>addImagesToIndex</code></b></li>
</ul>
<p>将图像（对应的特征向量）加入索引。</p>
<p><code>POST /face-recognition-index-add</code></p>
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
<td><code>imageLabelPairs</code></td>
<td><code>array</code></td>
<td>用于构建索引的图像-标签对。</td>
<td>是</td>
</tr>
<tr>
<td><code>indexKey</code></td>
<td><code>string</code></td>
<td>索引对应的键。由<code>buildIndex</code>操作提供。</td>
<td>是</td>
</tr>
</tbody>
</table>
<p><code>imageLabelPairs</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>label</code></td>
<td><code>string</code></td>
<td>标签。</td>
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
<td><code>imageCount</code></td>
<td><code>integer</code></td>
<td>索引的图像数量。</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>removeImagesFromIndex</code></b></li>
</ul>
<p>从索引中移除图像（对应的特征向量）。</p>
<p><code>POST /face-recognition-index-remove</code></p>
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
<td><code>ids</code></td>
<td><code>array</code></td>
<td>需要从索引中移除的向量的ID。</td>
<td>是</td>
</tr>
<tr>
<td><code>indexKey</code></td>
<td><code>string</code></td>
<td>索引对应的键。由<code>buildIndex</code>操作提供。</td>
<td>是</td>
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
<td><code>imageCount</code></td>
<td><code>integer</code></td>
<td>索引的图像数量。</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>进行图像识别。</p>
<p><code>POST /face-recognition-infer</code></p>
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
<td><code>indexKey</code></td>
<td><code>string</code></td>
<td>索引对应的键。由<code>buildIndex</code>操作提供。</td>
<td>否</td>
</tr>
<tr>
<td><code>detThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>det_threshold</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>recThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>rec_threshold</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>hammingRadius</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>hamming_radius</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>topk</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>topk</code> 参数相关说明。</td>
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
<td><code>faces</code></td>
<td><code>array</code></td>
<td>检测到的人脸的信息。</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code> | <code>null</code></td>
<td>识别结果图。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<p><code>faces</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td>人脸目标位置。数组中元素依次为边界框左上角x坐标、左上角y坐标、右下角x坐标以及右下角y坐标。</td>
</tr>
<tr>
<td><code>recResults</code></td>
<td><code>array</code></td>
<td>识别结果。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>检测得分。</td>
</tr>
</tbody>
</table>
<p><code>recResults</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>label</code></td>
<td><code>string</code></td>
<td>标签。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>识别得分。</td>
</tr>
</tbody>
</table>
</details>
<details><summary>多语言调用服务示例</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import pprint
import sys

import requests

API_BASE_URL = "http://0.0.0.0:8080"

base_image_label_pairs = [
    {"image": "./demo0.jpg", "label": "ID0"},
    {"image": "./demo1.jpg", "label": "ID1"},
    {"image": "./demo2.jpg", "label": "ID2"},
]
image_label_pairs_to_add = [
    {"image": "./demo3.jpg", "label": "ID2"},
]
ids_to_remove = [1]
infer_image_path = "./demo4.jpg"
output_image_path = "./out.jpg"

for pair in base_image_label_pairs:
    with open(pair["image"], "rb") as file:
        image_bytes = file.read()
        image_data = base64.b64encode(image_bytes).decode("ascii")
    pair["image"] = image_data

payload = {"imageLabelPairs": base_image_label_pairs}
resp_index_build = requests.post(f"{API_BASE_URL}/face-recognition-index-build", json=payload)
if resp_index_build.status_code != 200:
    print(f"Request to face-recognition-index-build failed with status code {resp_index_build}.")
    pprint.pp(resp_index_build.json())
    sys.exit(1)
result_index_build = resp_index_build.json()["result"]
print(f"Number of images indexed: {result_index_build['imageCount']}")

for pair in image_label_pairs_to_add:
    with open(pair["image"], "rb") as file:
        image_bytes = file.read()
        image_data = base64.b64encode(image_bytes).decode("ascii")
    pair["image"] = image_data

payload = {"imageLabelPairs": image_label_pairs_to_add, "indexKey": result_index_build["indexKey"]}
resp_index_add = requests.post(f"{API_BASE_URL}/face-recognition-index-add", json=payload)
if resp_index_add.status_code != 200:
    print(f"Request to face-recognition-index-add failed with status code {resp_index_add}.")
    pprint.pp(resp_index_add.json())
    sys.exit(1)
result_index_add = resp_index_add.json()["result"]
print(f"Number of images indexed: {result_index_add['imageCount']}")

payload = {"ids": ids_to_remove, "indexKey": result_index_build["indexKey"]}
resp_index_remove = requests.post(f"{API_BASE_URL}/face-recognition-index-remove", json=payload)
if resp_index_remove.status_code != 200:
    print(f"Request to face-recognition-index-remove failed with status code {resp_index_remove}.")
    pprint.pp(resp_index_remove.json())
    sys.exit(1)
result_index_remove = resp_index_remove.json()["result"]
print(f"Number of images indexed: {result_index_remove['imageCount']}")

with open(infer_image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {"image": image_data, "indexKey": result_index_build["indexKey"]}
resp_infer = requests.post(f"{API_BASE_URL}/face-recognition-infer", json=payload)
if resp_infer.status_code != 200:
    print(f"Request to face-recogntion-infer failed with status code {resp_infer}.")
    pprint.pp(resp_infer.json())
    sys.exit(1)
result_infer = resp_infer.json()["result"]

with open(output_image_path, "wb") as file:
    file.write(base64.b64decode(result_infer["image"]))
print(f"Output image saved at {output_image_path}")
print("\nDetected faces:")
pprint.pp(result_infer["faces"])
</code></pre>
</details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/on_device_deployment.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。


## 4. 二次开发
如果 人脸识别 产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升通用该产线的在您的场景中的识别效果。

### 4.1 模型微调
由于人脸识别产线包含两个模块（人脸检测和人脸特征），模型产线的效果不及预期可能来自于其中任何一个模块。

您可以对识别效果差的图片进行分析，如果在分析过程中发现有较多的人脸未被检测出来，那么可能是人脸检测模型存在不足，您需要参考[人脸检测模块开发教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/face_detection.html)中的<b>二次开发</b>章节，使用您的私有数据集对人脸检测模型进行微调；如果在已检测到的人脸出现匹配错误，这表明人脸特征模块需要进一步改进，您需要参考[人脸特征模块开发教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/face_feature.html)中的<b>二次开发</b>章节，对人脸特征模块进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```yaml

...

SubModules:
  Detection:
    module_name: face_detection
    model_name: PP-YOLOE_plus-S_face
    model_dir: null #可修改为微调后人脸检测模型的本地路径
    batch_size: 1
  Recognition:
    module_name: face_feature
    model_name: ResNet50_face
    model_dir: null #可修改为微调后人脸特征模型的本地路径
    batch_size: 1
```

随后， 参考[2.2 本地体验](#22-本地体验)中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device`参数</b>即可完成不同硬件之间的无缝切换。

例如，使用Python运行人脸识别产线时，将运行设备从英伟达 GPU 更改为昇腾 NPU，仅需将脚本中的 `device` 修改为 npu 即可：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="face_recognition",
    device="npu:0" # gpu:0 --> npu:0
    )
```
若您想在更多种类的硬件上使用人脸识别产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
