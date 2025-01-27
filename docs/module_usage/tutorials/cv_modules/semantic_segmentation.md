---
comments: true
---

# 语义分割模块使用教程

## 一、概述
语义分割是计算机视觉中的一种技术，它通过对图像中每个像素进行分类，将图像划分为不同的语义区域，每个区域对应一个具体的类别。这种技术能够生成精细的分割图，清晰地展现图像中的对象及其边界，为图像分析和理解提供有力支持。

## 二、支持模型列表

<table>
<thead>
<tr>
<th>模型名称</th><th>Model Download Link</th>
<th>mloU（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>OCRNet_HRNet-W48</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W48_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W48_pretrained.pdparams">Trained Model</a></td>
<td>82.15</td>
<td>78.9976</td>
<td>2226.95</td>
<td>249.8 M</td>
</tr>
<tr>
<td>PP-LiteSeg-T</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-T_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-T_pretrained.pdparams">Trained Model</a></td>
<td>73.10</td>
<td>7.6827</td>
<td>138.683</td>
<td>28.5 M</td>
</tr>
</tbody>
</table>

> ❗ 以上列出的是语义分割模块重点支持的<b>2个核心模型</b>，该模块总共支持<b>18个模型</b>，完整的模型列表如下：

<details><summary> 👉模型列表详情</summary>

<table>
<thead>
<tr>
<th>模型名称</th><th>Model Download Link</th>
<th>mloU（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Deeplabv3_Plus-R50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R50_pretrained.pdparams">Trained Model</a></td>
<td>80.36</td>
<td>61.0531</td>
<td>1513.58</td>
<td>94.9 M</td>
</tr>
<tr>
<td>Deeplabv3_Plus-R101</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3_Plus-R101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3_Plus-R101_pretrained.pdparams">Trained Model</a></td>
<td>81.10</td>
<td>100.026</td>
<td>2460.71</td>
<td>162.5 M</td>
</tr>
<tr>
<td>Deeplabv3-R50</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R50_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3-R50_pretrained.pdparams">Trained Model</a></td>
<td>79.90</td>
<td>82.2631</td>
<td>1735.83</td>
<td>138.3 M</td>
</tr>
<tr>
<td>Deeplabv3-R101</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/Deeplabv3-R101_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Deeplabv3-R101_pretrained.pdparams">Trained Model</a></td>
<td>80.85</td>
<td>121.492</td>
<td>2685.51</td>
<td>205.9 M</td>
</tr>
<tr>
<td>OCRNet_HRNet-W18</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W18_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W18_pretrained.pdparams">Trained Model</a></td>
<td>80.67</td>
<td>48.2335</td>
<td>906.385</td>
<td>43.1 M</td>
</tr>
<tr>
<td>OCRNet_HRNet-W48</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/OCRNet_HRNet-W48_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/OCRNet_HRNet-W48_pretrained.pdparams">Trained Model</a></td>
<td>82.15</td>
<td>78.9976</td>
<td>2226.95</td>
<td>249.8 M</td>
</tr>
<tr>
<td>PP-LiteSeg-T</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-T_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-T_pretrained.pdparams">Trained Model</a></td>
<td>73.10</td>
<td>7.6827</td>
<td>138.683</td>
<td>28.5 M</td>
</tr>
<tr>
<td>PP-LiteSeg-B</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/PP-LiteSeg-B_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LiteSeg-B_pretrained.pdparams">Trained Model</a></td>
<td>75.25</td>
<td>10.9935</td>
<td>194.727</td>
<td>47.0 M</td>
</tr>
<tr>
<td>SegFormer-B0 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B0 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B0 (slice)_pretrained.pdparams">Trained Model</a></td>
<td>76.73</td>
<td>11.1946</td>
<td>268.929</td>
<td>13.2 M</td>
</tr>
<tr>
<td>SegFormer-B1 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B1 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B1 (slice)_pretrained.pdparams">Trained Model</a></td>
<td>78.35</td>
<td>17.9998</td>
<td>403.393</td>
<td>48.5 M</td>
</tr>
<tr>
<td>SegFormer-B2 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B2 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B2 (slice)_pretrained.pdparams">Trained Model</a></td>
<td>81.60</td>
<td>48.0371</td>
<td>1248.52</td>
<td>96.9 M</td>
</tr>
<tr>
<td>SegFormer-B3 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B3 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B3 (slice)_pretrained.pdparams">Trained Model</a></td>
<td>82.47</td>
<td>64.341</td>
<td>1666.35</td>
<td>167.3 M</td>
</tr>
<tr>
<td>SegFormer-B4 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B4 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B4 (slice)_pretrained.pdparams">Trained Model</a></td>
<td>82.38</td>
<td>82.4336</td>
<td>1995.42</td>
<td>226.7 M</td>
</tr>
<tr>
<td>SegFormer-B5 (slice)</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SegFormer-B5 (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SegFormer-B5 (slice)_pretrained.pdparams">Trained Model</a></td>
<td>82.58</td>
<td>97.3717</td>
<td>2420.19</td>
<td>229.7 M</td>
</tr>
</tbody>
</table>
<p><b>以上模型精度指标测量自<a href="https://www.cityscapes-dataset.com/">Cityscapes</a>数据集。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p>
<table>
<thead>
<tr>
<th>模型名称</th><th>Model Download Link</th>
<th>mloU（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>SeaFormer_base</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_base(slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_base(slice)_pretrained.pdparams">Trained Model</a></td>
<td>40.92</td>
<td>24.4073</td>
<td>397.574</td>
<td>30.8 M</td>
</tr>
<tr>
<td>SeaFormer_large</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_large (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_large (slice)_pretrained.pdparams">Trained Model</a></td>
<td>43.66</td>
<td>27.8123</td>
<td>550.464</td>
<td>49.8 M</td>
</tr>
<tr>
<td>SeaFormer_small</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_small (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_small (slice)_pretrained.pdparams">Trained Model</a></td>
<td>38.73</td>
<td>19.2295</td>
<td>358.343</td>
<td>14.3 M</td>
</tr>
<tr>
<td>SeaFormer_tiny</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/SeaFormer_tiny (slice)_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SeaFormer_tiny (slice)_pretrained.pdparams">Trained Model</a></td>
<td>34.58</td>
<td>13.9496</td>
<td>330.132</td>
<td>6.1M</td>
</tr>
</tbody>
</table>
<p><b>SeaFormer系列模型的精度指标测量自<a href="https://groups.csail.mit.edu/vision/datasets/ADE20K/">ADE20k</a>数据集。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p></details>

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成 wheel 包的安装后，几行代码即可完成语义分割模块的推理，可以任意切换该模块下的模型，您也可以将语义分割的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_semantic_segmentation_002.png)到本地。

```python
from paddlex import create_model
model = create_model(model_name="PP-LiteSeg-T")
output = model.predict(input="general_semantic_segmentation_002.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

运行后，得到的结果为：
```bash
{'res': "{'input_path': 'general_semantic_segmentation_002.png', 'pred': '...'}"}
```
运行结果参数含义如下：
- `input_path`: 表示输入待预测图像的路径
- `pred`: 语义分割模型实际预测的mask，由于数据过大不便于直接print，所以此处用`...`替换，可以通过`res.save_to_img()`将预测结果保存为图片，通过`res.save_to_json()`将预测结果保存为json文件。

可视化图片如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/semanticseg/general_semantic_segmentation_002_res.png">

相关方法、参数等说明如下：

* `create_model`实例化通用语义分割模型（此处以`PP-LiteSeg-T`为例），具体说明如下：
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
<td><code>无</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>模型存储路径</td>
<td><code>str</code></td>
<td>无</td>
<td>无</td>
</tr>
<tr>
<td><code>target_size</code></td>
<td>模型预测时分辨率</td>
<td><code>int/tuple</code></td>
<td><code>None/-1/int/tuple</code></td>
<td><code>None</code></td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* `target_size`在初始化时指定模型推理时分辨率，默认为`None`。`-1`表示直接使用原图尺寸推理，`None`表示沿用上一层的设置，参数设置的优先级从高到低为：`predict参数传入 > create_model初始化传入 > yaml配置文件设置`。

* 调用通用语义分割模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input` 、 `batch_size` 和 `target_size`，具体说明如下：

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
<td><code>Python Var</code>/<code>str</code>/<code>dict</code>/<code>list</code></td>
<td>
<ul>
  <li><b>Python变量</b>，如<code>numpy.ndarray</code>表示的图像数据</li>
  <li><b>文件路径</b>，如图像文件的本地路径：<code>/root/data/img.jpg</code></li>
  <li><b>URL链接</b>，如图像文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_semantic_segmentation_001.png">示例</a></li>
  <li><b>本地目录</b>，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code></li>
  <li><b>列表</b>，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
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
<tr>
<td><code>target_size</code></td>
<td>推理时图像的尺寸(W, H)</td>
<td><code>int</code>/<code>tuple</code></td>
<td>
<ul>
  <li><b>-1</b>，表示直接原图尺寸推理</li>
  <li><b>None</b>，表示沿用上一层设置, 参数设置优先级从高到低为: <code>predict参数传入 > create_model初始化传入 > yaml配置文件设置</code></li>
  <li><b>int</b>，如512，表示推理时使用<code>(512, 512)</code>分辨率</li>
  <li><b>tuple</b>，如(512, 1024)，表示推理时使用<code>(512, 1024)</code>分辨率</li>
</ul>
</td>
<td>None</td>
</tr>
</table>

* 对预测结果进行处理，每个样本的预测结果均为`dict`类型，且支持打印、保存为图片、保存为`json`文件的操作:

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
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
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
<tr>
<td rowspan = "1"><code>img</code></td>
<td rowspan = "1">获取格式为<code>dict</code>的可视化图像</td>
</tr>

</table>


关于更多 PaddleX 的单模型推理的 API 的使用方法，可以参考[PaddleX单模型Python脚本使用说明](../../instructions/model_python_API.md)。


## 四、二次开发
如果你追求更高精度的现有模型，可以使用PaddleX的二次开发能力，开发更好的语义分割模型。在使用PaddleX开发语义分割模型之前，请务必安装PaddleX的图像分割相关的模型训练能力，安装过程可以参考 [PaddleX本地安装教程](../../../installation/installation.md)中的二次开发部分。

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/seg_optic_examples.tar -P ./dataset
tar -xf ./dataset/seg_optic_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/seg_optic_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在log中打印出 `Check dataset passed !` 信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>

<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;train_sample_paths&quot;: [
      &quot;check_dataset/demo_img/P0005.jpg&quot;,
      &quot;check_dataset/demo_img/P0050.jpg&quot;
    ],
    &quot;train_samples&quot;: 267,
    &quot;val_sample_paths&quot;: [
      &quot;check_dataset/demo_img/N0139.jpg&quot;,
      &quot;check_dataset/demo_img/P0137.jpg&quot;
    ],
    &quot;val_samples&quot;: 76,
    &quot;num_classes&quot;: 2
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/seg_optic_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;SegDataset&quot;
}
</code></pre>
<p>上述校验结果中，check_pass 为 True 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.num_classes</code>：该数据集类别数为 2；</li>
<li><code>attributes.train_samples</code>：该数据集训练集样本数量为 267；</li>
<li><code>attributes.val_samples</code>：该数据集验证集样本数量为 76；</li>
<li><code>attributes.train_sample_paths</code>：该数据集训练集样本可视化图片相对路径列表；</li>
<li><code>attributes.val_sample_paths</code>：该数据集验证集样本可视化图片相对路径列表；</li>
</ul>
<p>数据集校验还对数据集中所有类别的样本数量分布情况进行了分析，并绘制了分布直方图（histogram.png）：</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/semanticseg/01.png"></p></details>

#### 4.1.3 数据集格式转换/数据集划分（可选）（点击展开）
<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>

<p>在您完成数据校验之后，可以通过修改配置文件或是追加超参数的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。</p>
<p><b>（1）数据集格式转换</b></p>
<p>语义分割支持 <code>LabelMe</code> 格式的数据集转换为要求的格式。</p>
<p>数据集校验相关的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: 是否进行数据集格式转换，支持<code>LabelMe</code> 格式的数据集转换，默认为 <code>False</code>;</li>
<li><code>src_dataset_type</code>: 如果进行数据集格式转换，则需设置源数据集格式，默认为 <code>null</code>，支持的源数据集格式是<code>LabelMe</code>；
例如，您想转换 <code>LabelMe</code> 格式的数据集，提供了一个<code>LabelMe</code> 格式的示例数据集，如下：</li>
</ul>
<pre><code class="language-bash">wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/seg_dataset_to_convert.tar -P ./dataset
tar -xf ./dataset/seg_dataset_to_convert.tar -C ./dataset/
</code></pre>
<p>下载后，需要修改 <code>paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml</code>配置如下：</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: LabelMe
  ......
</code></pre>
<p>随后执行命令：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/seg_dataset_to_convert
</code></pre>
<p>当然，以上参数同样支持通过追加命令行参数的方式进行设置，以 <code>LabelMe</code> 格式的数据集为例：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/seg_dataset_to_convert \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
</code></pre>
<p><b>（2）数据集划分</b></p>
<p>数据集划分的参数可以通过修改配置文件中 CheckDataset 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li>CheckDataset:</li>
<li>split:</li>
<li>enable: 是否进行重新划分数据集，为 True 时进行数据集格式转换，默认为 False；</li>
<li>train_percent: 如果重新划分数据集，则需要设置训练集的百分比，类型为0-100之间的任意整数，需要保证与 val_percent 的值之和为100；</li>
</ul>
<p>例如，您想重新划分数据集为 训练集占比90%、验证集占比10%，则需将配置文件修改为：</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
</code></pre>
<p>随后执行命令：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/seg_optic_examples
</code></pre>
<p>数据划分执行之后，原有标注文件会被在原路径下重命名为 xxx.bak。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/seg_optic_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 模型训练
一条命令即可完成模型的训练，此处以移动端语义分割模型PP-LiteSeg-T的训练为例：

```bash
python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/seg_optic_examples
```


需要如下几步：

* 指定模型的.yaml 配置文件路径（此处为 `PP-LiteSeg-T.yam`，训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
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
<li><code>.pdparams</code>、<code>.pdema</code>、<code>.pdopt.pdstate</code>、<code>.pdiparams</code>、<code>.pdmodel</code>：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；</li>
</ul></details>

## 4.3 模型评估
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：

```bash
python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/seg_optic_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-LiteSeg-T.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=``./output/best_model/best_model.pdparams</code>。</p>
<p>在完成模型评估后，通常有以下产出：</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json，其记录了</code>评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 Top1 Acc</p></details>

### 4.4 模型推理和模型集成
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行Python集成。

#### 4.4.1 模型推理
通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_semantic_segmentation_002.png)到本地。

```bash
python main.py -c paddlex/configs/modules/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model" \
    -o Predict.input="general_semantic_segmentation_002.png"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的.yaml 配置文件路径（此处为` PP-LiteSeg-T.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

#### 4.4.2 模型集成
模型可以直接集成到PaddleX产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

图像通用语义分割模块可以集成的PaddleX产线有[通用语义分割](../../../pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.md)，只需要替换模型路径即可完成图像通用语义分割模块的模型更新。在产线集成中，你可以使用高性能部署和服务化部署来部署你得到的模型。

2.<b>模块集成</b>

您产出的权重可以直接集成到语义分割模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。
