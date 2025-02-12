---
comments: true
---

# 3D多模态融合检测产线使用教程

## 1. 3D多模态融合检测产线介绍

3D多模态融合检测产线支持输入多种传感器（激光雷达、环视RGB相机等）数据，通过深度学习等方法对数据进行处理，输出三维空间中物体的位置、形状、朝向、类别等信息。在自动驾驶、机器人导航、工业自动化等领域具有广泛应用。

BEVFusion 是一种多模态 3D 目标检测模型，通过将环视摄像头图像和 LiDAR 点云数据融合到统一的鸟瞰图（Bird's Eye View，BEV）表示中，实现不同传感器的特征对齐并融合，克服了单一传感器的局限性，显著提升了检测精度和鲁棒性，适用于自动驾驶等复杂场景。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/3d_bev_detection/01.png">

<b>3D多模态融合检测产线中包含了3D多模态融合检测模块</b>，模块中包含了一个<b>BEVFusion模型</b>，我们提供了该模型的 benchmark 数据：

<p><b>3D多模态融合检测模块：</b></p>
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
<td rowspan="2">BEVFusion是一种在BEV视角下的多模态融合检测模型，采用两个分支处理不同模态的数据，得到lidar和camera在BEV视角下的特征，camera分支采用LSS这种自底向上的方式来显式的生成图像BEV特征，lidar分支采用经典的点云检测网络，最后对两种模态的BEV特征进行对齐和融合，应用于检测head或分割head。
</td>
</tr>
<tr>
</table>

<p>注：以上精度指标为<a href="https://www.nuscenes.org/nuscenes">nuscenes</a>验证集 mAP(0.5:0.95), NDS 60.9, 精度类型为 FP32。</p></details>

## 2. 快速开始

PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验3D多模态融合检测产线的效果，也可以在本地使用命令行或 Python 体验3D多模态融合检测产线的效果。

### 2.1 在线体验

暂不支持在线体验。

### 2.2 本地体验
> ❗ 在本地使用3D多模态融合检测产线前，请确保您已经按照[PaddleX安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.2.1 命令行方式体验

一行命令即可快速体验3D多模态融合检测产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/det_3d/demo_det_3d/nuscenes_demo_infer.tar)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline 3d_bev_detection \
        --input nuscenes_demo_infer.tar \
        --device gpu:0
```

参数说明：

```
--pipeline：产线名称，此处为3D多模态融合检测产线
--input：输入的包含点云图像文件的.tar压缩文件的本地路径。3D多模态融合检测为为多输入模型，输入依赖点云、图像以及转换矩阵等其他信息。tar解压文件包含samples路径，sweeps路径和nuscnes_infos_val.pkl文件，其中samples包含当前输入的所有图像和点云数据，sweeps包含关联帧点云数据，nuscnes_infos_val.pkl文件包含所有点云和图像在samples和sweeps下的相对路径以及转换矩阵等相关信息。
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```

运行后，会将结果打印在终端上，结果如下：

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
- `scores_3d`：表示该3D样本的所有预测框对应的置信度


#### 2.2.2 Python脚本方式集成
* 上述命令行是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

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


在上述 Python 脚本中，执行了如下几个步骤：

（1）调用 `create_pipeline` 实例化 3D多模态融合检测 产线对象：具体参数说明如下：

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

（2）调用3D多模态融合检测产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`input`，用于输入待预测数据，支持多种输入方式，具体示例如下：

<table>
<thead>
<tr>
<th>参数类型</th>
<th>参数说明</th>
</tr>
</thead>
<tbody>
<tr>
<td>str</td>
<td><b>tar文件路径</b>，例如：<code>/root/data/nuscenes_demo_infer.tar</code></td>
</tr>
<tr>
<td>list</td>
<td><b>列表</b>，列表元素需为上述类型数据，如<code>["/root/data/nuscenes_demo_infer1.tar", "/root/data/nuscenes_demo_infer2.tar"]</td>
</tr>
</tbody>
</table>

<p><b>注：pkl文件可以按照<a href="https://github.com/ADLab-AutoDrive/BEVFusion/blob/main/tools/create_data.py">脚本</a>进行制作。</b></p></details>

（3）调用`predict`方法获取预测结果：`predict` 方法为 `generator`，因此需要通过调用获得预测结果，`predict` 方法以batch为单位对数据进行预测，因此预测结果为list形式表示的一组预测结果。

（4）对预测结果进行处理：每个样本的预测结果均为 `dict` 类型，且支持打印，或保存为json文件，如：

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
</tbody>
</table>

此外，您可以获取3D多模态融合检测产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config 3d_bev_detection --save_path ./my_path
```

若您获取了配置文件，即可对3D多模态融合检测产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。示例如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="./my_path/3d_bev_detection.yaml")

output = pipeline.predict("nuscenes_demo_infer.tar")

for res in output:
    res.print()  ## 打印预测的结构化输出
    res.save_to_json("./output/")  ## 保存结果到json文件
    res.visualize(save_path="./output/", show=True) ## 3d结果可视化，如果运行环境有图形界面设置show=True，否则设置为False
```

<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改3D多模态融合检测产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。同时，CLI 预测也支持传入配置文件，`--pipeline` 指定配置文件的路径即可。

## 3. 开发集成/部署
如果 3D多模态融合检测 产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将通用 3D多模态融合检测 产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#222-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 <b>高性能推理</b>：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ <b>服务化部署</b>：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持多种产线服务化部署方案，详细的产线服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/serving.md)。

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
<p>进行3D多模态融合检测。</p>
<p><code>POST /bev-3d-object-detection</code></p>
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
<td><code>tar</code></td>
<td><code>string</code></td>
<td>服务器可访问的tar文件的URL或路径。</td>
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
<td>长度为9的列表, 0: 中心点x坐标、1: 中心点y坐标、2: 中心点z坐标、3: 检测框宽度、4: 检测框长度、5: 检测框高度、6: 旋转角度、7: 坐标系x方向速度、8: 坐标系y方向速度</td>
</tr>
<tr>
<td><code>categoryId</code></td>
<td><code>integer</code></td>
<td>目标类别ID。</td>
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


<pre><code class="language-python">
import requests

API_URL = &quot;http://localhost:8080/bev-3d-object-detection&quot; # 服务URL
tar_path = &quot;./nuscenes_demo_infer.tar&quot;

payload = {&quot;tar&quot;: tar_path}

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
print(&quot;Detected objects:&quot;)
print(result[&quot;detectedObjects&quot;])
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。

您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果 3D多模态融合检测 产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升 3D多模态融合检测 产线的在您的场景中的识别效果。

### 4.1 模型微调

参考[3D多模态融合检测模块开发教程](../../../module_usage/tutorials/cv_modules/3d_bev_detection.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/3d_bev_detection.md#四二次开发)章节，使用您的私有数据集模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```bash
......
Pipeline:
  device: "gpu:0"
  det_model: "BEVFusion"        #可修改为微调后3D目标检测模型的本地路径
  det_batch_size: 1
  device: gpu
......
```
随后， 参考[2.2 本地体验](#22-本地体验)中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。


##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device`参数</b>即可完成不同硬件之间的无缝切换。

例如，使用Python运行3D多模态融合检测产线时，将运行设备从英伟达 GPU 更改为昇腾 NPU，仅需将脚本中的 `device` 修改为 npu 即可：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="3d_bev_detection",
    device="npu:0" # gpu:0 --> npu:0
    )
```
若您想在更多种类的硬件上使用3D多模态融合检测产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
