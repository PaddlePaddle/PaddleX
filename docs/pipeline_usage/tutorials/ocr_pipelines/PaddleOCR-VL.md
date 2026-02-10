---
comments: true
---

# PaddleOCR-VL介绍

PaddleOCR-VL 是一款先进、高效的文档解析模型，专为文档中的元素识别设计。其核心组件为 PaddleOCR-VL-0.9B，这是一种紧凑而强大的视觉语言模型（VLM），它由 NaViT 风格的动态分辨率视觉编码器与 ERNIE-4.5-0.3B 语言模型组成，能够实现精准的元素识别。该模型支持 109 种语言，并在识别复杂元素（如文本、表格、公式和图表）方面表现出色，同时保持极低的资源消耗。通过在广泛使用的公开基准与内部基准上的全面评测，PaddleOCR-VL 在页级级文档解析与元素级识别均达到 SOTA 表现。它显著优于现有的基于Pipeline方案和文档解析多模态方案以及先进的通用多模态大模型，并具备更快的推理速度。这些优势使其非常适合在真实场景中落地部署。

**2026年1月29日，我们发布了PaddleOCR-VL-1.5。PaddleOCR-VL-1.5不仅以94.5%精度大幅刷新了评测集OmniDocBench v1.5，更创新性地支持了异形框定位，使得PaddleOCR-VL-1.5 在扫描、倾斜、弯折、屏幕拍摄及复杂光照等真实场景中均表现优异。此外，模型还新增了印章识别与文本检测识别能力，关键指标持续领跑。**

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/paddleocr-vl-1.5_metrics.png"/>

## 1. 环境准备

安装 PaddlePaddle 和 PaddleX:

```shell
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install paddlex
```
> 对于 Windows 用户，请使用 WSL 或者 Docker 进行环境搭建。

运行 PaddleOCR-VL 对 GPU 硬件有以下要求：

<table border="1">
<thead>
  <tr>
    <th>推理方式</th>
    <th>GPU Compute Capability</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>PaddlePaddle</td>
    <td>≥ 8.5</td>
  </tr>
  <tr>
    <td>vLLM</td>
    <td>≥ 8 （RTX 3060，RTX 5070，A10，A100, ...） <br />
    7 ≤ GPU Compute Capability < 8 （T4，V100，...）支持运行，但可能出现请求超时、OOM 等异常情况，不推荐使用
    </td>
  </tr>
  <tr>
    <td>SGLang</td>
    <td>8 ≤ GPU Compute Capability < 12</td>
  </tr>
</tbody>
</table>

目前 PaddleOCR-VL 暂不支持 CPU 及 Arm 架构，后续将根据实际需求扩展更多硬件支持，敬请期待！

## 2. 快速开始

PaddleOCR-VL 支持 CLI 命令行方式和 Python API 两种使用方式，其中 CLI 命令行方式更简单，适合快速验证功能，而 Python API 方式更灵活，适合集成到现有项目中。

### 2.1 命令行方式体验

一行命令即可快速体验 PaddleOCR-VL 效果：

```bash
paddlex --pipeline PaddleOCR-VL --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_ocr_vl_demo.png

# 通过 --use_doc_orientation_classify 指定是否使用文档方向分类模型
paddlex --pipeline PaddleOCR-VL --input ./paddleocr_vl_demo.png --use_doc_orientation_classify True

# 通过 --use_doc_unwarping 指定是否使用文本图像矫正模块
paddlex --pipeline PaddleOCR-VL --input ./paddleocr_vl_demo.png --use_doc_unwarping True

# 通过 --use_layout_detection 指定是否使用版面区域检测排序模块
paddlex --pipeline PaddleOCR-VL --input ./paddleocr_vl_demo.png --use_layout_detection False
```

<details><summary><b>命令行支持更多参数设置，点击展开以查看命令行参数的详细说明</b></summary>
<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>待预测数据，必填。
如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)。
</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>指定推理结果文件保存的路径。如果不设置，推理结果将不会保存到本地。</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>版面区域检测排序模型名称。如果不设置，将会使用默认模型。</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>版面区域检测排序模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>版面模型得分阈值。<code>0-1</code> 之间的任意浮点数。如果不设置，将使用初始化的默认值。
</td>
<td><code>float</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>版面检测是否使用后处理NMS。如果不设置，将使用初始化的默认值。</td>
<td><code>bool</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>版面区域检测模型检测框的扩张系数。
任意大于 <code>0</code>  浮点数。如果不设置，将使用初始化的默认值
</td>
<td><code>float</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>版面检测中模型输出的检测框的合并处理模式。
<ul>
<li><b>large</b>，设置为large时，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留外部最大的框，删除重叠的内部框；</li>
<li><b>small</b>，设置为small，表示在模型输出的检测框中，对于互相重叠包含的检测框，只保留内部被包含的小框，删除重叠的外部框；</li>
<li><b>union</b>，不进行框的过滤处理，内外框都保留；</li>
</ul>如果不设置，将使用初始化的参数值。
</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>vl_rec_model_name</code></td>
<td>多模态识别模型名称。如果不设置，将会使用默认模型。</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>vl_rec_model_dir</code></td>
<td>多模态识别模型目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>vl_rec_backend</code></td>
<td>多模态识别模型使用的推理后端。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>vl_rec_server_url</code></td>
<td>如果多模态识别模型使用推理服务，该参数用于指定服务器URL。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>vl_rec_max_concurrency</code></td>
<td>如果多模态识别模型使用推理服务，该参数用于指定最大并发请求数。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>文档方向分类模型的名称。如果不设置，将使用初始化的默认值。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>文档方向分类模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>文本图像矫正模型的名称。如果不设置，将使用初始化的默认值。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>文本图像矫正模型的目录路径。如果不设置，将会下载官方模型。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否加载并使用文档方向分类模块。如果不设置，将使用初始化的默认值，默认初始化为<code>False</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否加载并使用文本图像矫正模块。如果不设置，将使用初始化的默认值，默认初始化为<code>False</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>是否加载并使用版面区域检测排序模块。如果不设置，将使用初始化的默认值，默认初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>是否使用图表解析功能。如果不设置，将使用初始化的默认值，默认初始化为<code>False</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>format_block_content</code></td>
<td>控制是否将 <code>block_content</code> 中的内容格式化为Markdown格式。如果不设置，将使用初始化的默认值，默认初始化为<code>False</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_queues</code></td>
<td>用于控制是否启用内部队列。当设置为 <code>True</code> 时，数据加载（如将 PDF 页面渲染为图像）、版面检测模型处理以及 VLM 推理将分别在独立线程中异步执行，通过队列传递数据，从而提升效率。对于页数较多的 PDF 文档，或是包含大量图像或 PDF 文件的目录，这种方式尤其高效。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>prompt_label</code></td>
<td>VL模型的 prompt 类型设置，当且仅当 <code>use_layout_detection=False</code> 时生效。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>repetition_penalty</code></td>
<td>VL模型采样使用的重复惩罚参数。</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>temperature</code></td>
<td>VL模型采样使用的温度参数。</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>top_p</code></td>
<td>VL模型采样使用的top-p参数。</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>min_pixels</code></td>
<td>VL模型预处理图像时允许的最小像素数。</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>max_pixels</code></td>
<td>VL模型预处理图像时允许的最大像素数。</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>用于推理的设备。支持指定具体卡号：
<ul>
<li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
<li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
<li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
<li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
<li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
<li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
</ul>如果不设置，将使用初始化的默认值，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备。
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>是否启用高性能推理。</td>
<td><code>bool</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>是否启用 Paddle Inference 的 TensorRT 子图引擎。如果模型不支持通过 TensorRT 加速，即使设置了此标志，也不会使用加速。<br/>
对于 CUDA 11.8 版本的飞桨，兼容的 TensorRT 版本为 8.x（x>=6），建议安装 TensorRT 8.6.1.6。<br/>
</td>
<td><code>bool</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>计算精度，如 fp32、fp16。</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>是否启用 MKL-DNN 加速推理。如果 MKL-DNN 不可用或模型不支持通过 MKL-DNN 加速，即使设置了此标志，也不会使用加速。
</td>
<td><code>bool</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td>
MKL-DNN 缓存容量。
</td>
<td><code>int</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>在 CPU 上进行推理时使用的线程数。</td>
<td><code>int</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>PaddleX产线配置文件路径。</td>
<td><code>str</code></td>
<td></td>
</tr>
</tbody>
</table>
</details>
<br />

运行结果会被打印到终端上，默认配置的 PaddleOCR-VL 的运行结果如下：

<details><summary> 👉点击展开</summary>
<pre>
<code>
{'res': {'input_path': 'paddleocr_vl_demo.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True, 'use_chart_recognition': False, 'format_block_content': False}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 6, 'label': 'doc_title', 'score': 0.9636914134025574, 'coordinate': [np.float32(131.31366), np.float32(36.450516), np.float32(1384.522), np.float32(127.984665)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9281806349754333, 'coordinate': [np.float32(585.39465), np.float32(158.438), np.float32(930.2184), np.float32(182.57469)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9840355515480042, 'coordinate': [np.float32(9.023666), np.float32(200.86115), np.float32(361.41583), np.float32(343.8828)]}, {'cls_id': 14, 'label': 'image', 'score': 0.9871416091918945, 'coordinate': [np.float32(775.50574), np.float32(200.66502), np.float32(1503.3807), np.float32(684.9304)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9801855087280273, 'coordinate': [np.float32(9.532196), np.float32(344.90594), np.float32(361.4413), np.float32(440.8244)]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9708921313285828, 'coordinate': [np.float32(28.040405), np.float32(455.87976), np.float32(341.7215), np.float32(520.7117)]}, {'cls_id': 24, 'label': 'vision_footnote', 'score': 0.9002962708473206, 'coordinate': [np.float32(809.0692), np.float32(703.70044), np.float32(1488.3016), np.float32(750.5238)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9825374484062195, 'coordinate': [np.float32(8.896561), np.float32(536.54895), np.float32(361.05237), np.float32(655.8058)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9822263717651367, 'coordinate': [np.float32(8.971573), np.float32(657.4949), np.float32(362.01715), np.float32(774.625)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9767460823059082, 'coordinate': [np.float32(9.407074), np.float32(776.5216), np.float32(361.31067), np.float32(846.82874)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9868153929710388, 'coordinate': [np.float32(8.669495), np.float32(848.2543), np.float32(361.64703), np.float32(1062.8568)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9826608300209045, 'coordinate': [np.float32(8.8025055), np.float32(1063.8615), np.float32(361.46588), np.float32(1182.8524)]}, {'cls_id': 22, 'label': 'text', 'score': 0.982555627822876, 'coordinate': [np.float32(8.820602), np.float32(1184.4663), np.float32(361.66394), np.float32(1302.4507)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9584776759147644, 'coordinate': [np.float32(9.170288), np.float32(1304.2161), np.float32(361.48898), np.float32(1351.7483)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9782056212425232, 'coordinate': [np.float32(389.1618), np.float32(200.38202), np.float32(742.7591), np.float32(295.65146)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9844875931739807, 'coordinate': [np.float32(388.73303), np.float32(297.18463), np.float32(744.00024), np.float32(441.3034)]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9680547714233398, 'coordinate': [np.float32(409.39468), np.float32(455.89386), np.float32(721.7174), np.float32(520.9387)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9741666913032532, 'coordinate': [np.float32(389.71606), np.float32(536.8138), np.float32(742.7112), np.float32(608.00165)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9840384721755981, 'coordinate': [np.float32(389.30988), np.float32(609.39636), np.float32(743.09247), np.float32(750.3231)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9845995306968689, 'coordinate': [np.float32(389.13272), np.float32(751.7772), np.float32(743.058), np.float32(894.8815)]}, {'cls_id': 22, 'label': 'text', 'score': 0.984852135181427, 'coordinate': [np.float32(388.83267), np.float32(896.0371), np.float32(743.58215), np.float32(1038.7345)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9804865717887878, 'coordinate': [np.float32(389.08478), np.float32(1039.9119), np.float32(742.7585), np.float32(1134.4897)]}, {'cls_id': 22, 'label': 'text', 'score': 0.986461341381073, 'coordinate': [np.float32(388.52643), np.float32(1135.8137), np.float32(743.451), np.float32(1352.0085)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9869391918182373, 'coordinate': [np.float32(769.8341), np.float32(775.66235), np.float32(1124.9813), np.float32(1063.207)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9822869896888733, 'coordinate': [np.float32(770.30383), np.float32(1063.938), np.float32(1124.8295), np.float32(1184.2192)]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9689218997955322, 'coordinate': [np.float32(791.3042), np.float32(1199.3169), np.float32(1104.4521), np.float32(1264.6985)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9713128209114075, 'coordinate': [np.float32(770.4253), np.float32(1279.6072), np.float32(1124.6917), np.float32(1351.8672)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9236552119255066, 'coordinate': [np.float32(1153.9058), np.float32(775.5814), np.float32(1334.0654), np.float32(798.1581)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9857938885688782, 'coordinate': [np.float32(1151.5197), np.float32(799.28015), np.float32(1506.3619), np.float32(991.1156)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9820687174797058, 'coordinate': [np.float32(1151.5686), np.float32(991.91095), np.float32(1506.6023), np.float32(1110.8875)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9866049885749817, 'coordinate': [np.float32(1151.6919), np.float32(1112.1301), np.float32(1507.1611), np.float32(1351.9504)]}]}}}
</code></pre></details>

运行结果参数说明可以参考[2.2 Python脚本方式集成](#22-python脚本方式集成)中的结果解释。

<b>注：</b>由于 PaddleOCR-VL 的默认模型较大，推理速度可能较慢，建议实际推理使用[3. 使用推理加速框架提升 VLM 推理性能](#3-使用推理加速框架提升-vlm-推理性能) 方式进行快速推理。

### 2.2 Python脚本方式集成

命令行方式是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成 PaddleOCR-VL 的快速推理，推理代码如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PaddleOCR-VL")

output = pipeline.predict(input="./pp_ocr_vl_demo.png") # 传入图像文件或PDF文件

for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path="output") ## 保存当前图像的结构化json结果
    res.save_to_markdown(save_path="output") ## 保存当前图像的markdown格式的结果
```

如果是 PDF 文件，会将 PDF 的每一页单独处理，每一页的 Markdown 文件也会对应单独的结果。如果您希望对多页的推理结果进行跨页表格合并、重建多级标和合并多页结果等需求，可以通过如下方式实现：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PaddleOCR-VL")

output = pipeline.predict(input="./your_pdf_file.pdf")

pages_res = list(output)

output = pipeline.restructure_pages(pages_res)

# output = pipeline.restructure_pages(pages_res, merge_table=True) # 合并跨页表格
# output = pipeline.restructure_pages(pages_res, merge_table=True, relevel_titles=True) # 合并跨页表格，重建多级标题
# output = pipeline.restructure_pages(pages_res, merge_table=True, relevel_titles=True, concatenate_pages=True) # 合并跨页表格，重建多级标题，合并多页结果为一页

for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path="output") ## 保存当前图像的结构化json结果
    res.save_to_markdown(save_path="output") ## 保存当前图像的markdown格式的结果
```

在上述 Python 脚本中，执行了如下几个步骤：

<details><summary>（1）实例化对象，具体参数说明如下：</summary>

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
<td><code>layout_detection_model_name</code></td>
<td>版面区域检测排序模型名称。如果设置为<code>None</code>，将会使用默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>版面区域检测排序模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>版面模型得分阈值。
<ul>
<li><b>float</b>：<code>0-1</code> 之间的任意浮点数；</li>
<li><b>dict</b>： <code>{0:0.1}</code> key为类别ID，value为该类别的阈值；</li>
<li><b>None</b>：如果设置为<code>None</code>，将使用初始化的默认值。</li>
</ul>
</td>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>版面检测是否使用后处理NMS。如果设置为<code>None</code>，将使用初始化的默认值。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>版面区域检测模型检测框的扩张系数。
<ul>
<li><b>float</b>：任意大于 <code>0</code>  浮点数；</li>
<li><b>Tuple[float,float]</b>：在横纵两个方向各自的扩张系数；</li>
<li><b>dict</b>，dict的key为<b>int</b>类型，代表<code>cls_id</code>, value为<b>tuple</b>类型，如<code>{0: (1.1, 2.0)}</code>，表示将模型输出的第0类别检测框中心不变，宽度扩张1.1倍，高度扩张2.0倍；</li>
<li><b>None</b>：如果设置为<code>None</code>，将使用初始化的默认值。</li>
</ul>
</td>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>版面区域检测的重叠框过滤方式。
<ul>
<li><b>str</b>：<code>large</code>，<code>small</code>，<code>union</code>，分别表示重叠框过滤时选择保留大框，小框还是同时保留；</li>
<li><b>dict</b>： dict的key为<b>int</b>类型，代表<code>cls_id</code>，value为<b>str</b>类型，如<code>{0: "large", 2: "small"}</code>，表示对第0类别检测框使用large模式，对第2类别检测框使用small模式；</li>
<li><b>None</b>：如果设置为<code>None</code>，将使用初始化的默认值。</li>
</ul>
</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>vl_rec_model_name</code></td>
<td>多模态识别模型名称。如果设置为<code>None</code>，将会使用默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>vl_rec_model_dir</code></td>
<td>多模态识别模型目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>vl_rec_backend</code></td>
<td>多模态识别模型使用的推理后端。</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>vl_rec_server_url</code></td>
<td>如果多模态识别模型使用推理服务，该参数用于指定服务器URL。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>vl_rec_max_concurrency</code></td>
<td>如果多模态识别模型使用推理服务，该参数用于指定最大并发请求数。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>文档方向分类模型的名称。如果设置为<code>None</code>，将会使用默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>文档方向分类模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>文本图像矫正模型的名称。如果设置为<code>None</code>，将会使用默认模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>文本图像矫正模型的目录路径。如果设置为<code>None</code>，将会下载官方模型。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否加载并使用文档方向分类模块。如果设置为<code>None</code>，将使用初始化的默认值，默认初始化为<code>False</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否加载并使用文本图像矫正模块。如果设置为<code>None</code>，将使用初始化的默认值，默认初始化为<code>False</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>是否加载并使用版面区域检测排序模块。如果设置为<code>None</code>，将使用初始化的默认值，默认初始化为<code>True</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>是否加载并使用图表解析模块。如果设置为<code>None</code>，将使用初始化的默认值，默认初始化为<code>False</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>format_block_content</code></td>
<td>控制是否将 <code>block_content</code> 中的内容格式化为Markdown格式。如果设置为<code>None</code>，将使用初始化的默认值，默认初始化为<code>False</code>。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>merge_layout_blocks</code></td>
<td>控制是否对跨栏或上下交错分栏的版面检测框进行合并。如果设置为<code>None</code>，将使用初始化的默认值，默认初始化为<code>True</code>。</td>
<td><code>bool|None</code></td>
<td></td>
</tr>
<tr>
<td><code>markdown_ignore_labels</code></td>
<td>需要在Markdown中忽略的版面标签。如果设置为<code>None</code>，将使用初始化的默认值:<code>['number','footnote','header','header_image','footer','footer_image','aside_text']</code></td>
<td><code>list|None</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>用于推理的设备。支持指定具体卡号：
<ul>
<li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
<li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
<li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
<li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
<li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
<li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
<li><b>None</b>：如果设置为<code>None</code>，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备。</li>
</ul>
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>是否启用高性能推理。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>是否启用 Paddle Inference 的 TensorRT 子图引擎。如果模型不支持通过 TensorRT 加速，即使设置了此标志，也不会使用加速。<br/>
对于 CUDA 11.8 版本的飞桨，兼容的 TensorRT 版本为 8.x（x>=6），建议安装 TensorRT 8.6.1.6。<br/>
</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>计算精度，如 fp32、fp16。</td>
<td><code>str</code></td>
<td><code>"fp32"</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>是否启用 MKL-DNN 加速推理。如果 MKL-DNN 不可用或模型不支持通过 MKL-DNN 加速，即使设置了此标志，也不会使用加速。
</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td>
MKL-DNN 缓存容量。
</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>在 CPU 上进行推理时使用的线程数。</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>PaddleX产线配置文件路径。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

</details>

<details><summary>（2）调用 PaddleOCR-VL 对象的 <code>predict()</code> 方法进行推理预测，该方法会返回一个结果列表。另外，PaddleOCR-VL 还提供了 <code>predict_iter()</code> 方法。两者在参数接受和结果返回方面是完全一致的，区别在于 <code>predict_iter()</code> 返回的是一个 <code>generator</code>，能够逐步处理和获取预测结果，适合处理大型数据集或希望节省内存的场景。可以根据实际需求选择使用这两种方法中的任意一种。以下是 <code>predict()</code> 方法的参数及其说明：</summary>

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型，必填。
<ul>
<li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
<li><b>list</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>，<code>["/root/data1", "/root/data2"]。</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否在推理时使用文档方向分类模块。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否在推理时使用文本图像矫正模块。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>是否在推理时使用版面区域检测排序模块。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>是否在推理时使用图表解析模块。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_seal_recognition</code></td>
<td><b>含义：</b>是否使用印章识别功能。<br/>
<b>说明：</b>设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_ocr_for_image_block</code></td>
<td><b>含义：</b>是否对图片中的文字进行识别。<br/>
<b>说明：</b>设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_shape_mode</code></td>
<td><b>含义：</b>用于指定版面检测结果的几何形状表示模式。该参数决定了检测区域（如文本块、图片、表格等）边界的计算方式及展示形态。<br/>
<b>说明：</b>取值说明：
    <ul>
    <li><b>rect (矩形)</b>: 输出水平正向的边界框（包含 x1, y1, x2, y2）。适用于标准的水平排版版面。</li>
    <li><b>quad (四边形)</b>: 输出由四个顶点组成的任意四边形。适用于存在倾斜、透视变形的区域。</li>
    <li><b>poly (多边形)</b>: 输出由多个坐标点组成的闭合轮廓。适用于形状不规则或具有弧度的版面元素，精度最高。</li>
    <li><b>auto (自动)</b>: 系统根据检测目标的复杂程度和置信度，自动选择最合适的形状表达方式。</li>
    </ul>
</td>
<td><code>str</code></td>
<td>"auto"</td>
</tr>
<tr>
<td><code>merge_layout_blocks</code></td>
<td>控制是否对跨栏或上下交错分栏的版面检测框进行合并。如果不设置，将使用初始化的默认值，默认初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>markdown_ignore_labels</code></td>
<td>需要在Markdown中忽略的版面标签。如果不设置，将使用初始化的参数值。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>use_queues</code></td>
<td>用于控制是否启用内部队列。当设置为 <code>True</code> 时，数据加载（如将 PDF 页面渲染为图像）、版面检测模型处理以及 VLM 推理将分别在独立线程中异步执行，通过队列传递数据，从而提升效率。对于页数较多的 PDF 文档，或是包含大量图像或 PDF 文件的目录，这种方式尤其高效。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>prompt_label</code></td>
<td>VL模型的 prompt 类型设置，当且仅当 <code>use_layout_detection=False</code> 时生效。可填写参数为 <code>ocr</code>、<code>formula</code>、<code>table</code> 和 <code>chart</code>。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>format_block_content</code></td>
<td>参数含义与实例化参数基本相同。设置为<code>None</code>表示使用实例化参数，否则该参数优先级更高。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>repetition_penalty</code></td>
<td>VL模型采样使用的重复惩罚参数。</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>temperature</code></td>
<td>VL模型采样使用的温度参数。</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>top_p</code></td>
<td>VL模型采样使用的top-p参数。</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>min_pixels</code></td>
<td>VL模型预处理图像时允许的最小像素数。</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>max_pixels</code></td>
<td>VL模型预处理图像时允许的最大像素数。</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>max_new_tokens</code></td>
<td>VL模型生成的最大token数。</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>merge_layout_blocks</code></td>
<td>控制是否对跨栏或上下交错分栏的版面检测框进行合并。</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>markdown_ignore_labels</code></td>
<td>需要在Markdown中忽略的版面标签。</td>
<td><code>list|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>vlm_extra_args</code></td>
<td><b>含义：</b>VLM额外配置参数。<br/>
<b>说明：</b>目前支持的自定义参数如下：
<ul>
  <li><code>ocr_min_pixels</code>：OCR 最小分辨率</li>
  <li><code>ocr_max_pixels</code>：OCR 最大分辨率</li>
  <li><code>table_min_pixels</code>：表格最小分辨率</li>
  <li><code>table_max_pixels</code>：表格最大分辨率</li>
  <li><code>chart_min_pixels</code>：图表最小分辨率</li>
  <li><code>chart_max_pixels</code>：图表最大分辨率</li>
  <li><code>formula_min_pixels</code>：公式最小分辨率</li>
  <li><code>formula_max_pixels</code>：公式最大分辨率</li>
  <li><code>seal_min_pixels</code>：印章最小分辨率</li>
  <li><code>seal_max_pixels</code>：印章最大分辨率</li>
</ul></td>
<td><code>dict|None</code></td>
<td><code>None</code></td>
</tr>
</table>
</details>

<details><summary>（3）调用 PaddleOCR-VL 对象的 <code>restructure_pages()</code> 方法对推理预测的多页结果列表进行页面重建，该方法会返回一个重建后的多页结果或合并后的单页结果。以下是 <code>restructure_pages()</code> 方法的参数及其说明：</summary>
<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
<tr>
<td><code>res_list</code></td>
<td><b>含义：</b>多页 PDF 推理预测出的结果列表。</td>
<td><code>list|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>merge_tables</code></td>
<td><b>含义：</b>控制是否进行跨页表格合并。</td>
<td><code>Bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>relevel_titles</code></td>
<td><b>含义：</b>控制是否进行多级表格分级</td>
<td><code>Bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>concatenate_pages</code></td>
<td><b>含义：</b>控制是否拼接多页结果为一页</td>
<td><code>Bool</code></td>
<td><code>False</code></td>
</tr>
</details>

<details><summary>（4）对预测结果进行处理：每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为<code>json</code>文件的操作:</summary>

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
<td>是否对输出内容进行使用 <code>JSON</code> 缩进格式化。</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效。</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效。</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">将结果保存为json格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致。</td>
<td>无</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效。</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效。</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>将中间各个模块的可视化图像保存在png格式的图像</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径。</td>
<td>无</td>
</tr>
<tr>
<td rowspan="3"><code>save_to_markdown()</code></td>
<td rowspan="3">将图像或者PDF文件中的每一页分别保存为markdown格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
<tr>
<td><code>pretty</code></td>
<td><code>bool</code></td>
<td>是否美化 <code>markdown</code> 输出结果，将图表等进行居中操作，使 <code>markdown</code> 渲染后更美观。</td>
<td>True</td>
</tr>
<tr>
<td><code>show_formula_number</code></td>
<td><code>bool</code></td>
<td>控制是否在 <code>markdown</code> 中将保留公式编号。设置为 <code>True</code> 时，保留全部公式编号；<code>False</code> 则仅保留公式</td>
<td><code>False</code></td>
</tr>
<tr>
<tr>
<td><code>save_to_html()</code></td>
<td>将文件中的表格保存为html格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径。</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>将文件中的表格保存为xlsx格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径。</td>
<td>无</td>
</tr>
</table>

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：
    - `input_path`: `(str)` 待预测图像或者PDF的输入路径

    - `page_index`: `(Union[int, None])` 如果输入是PDF文件，表示当前是PDF的第几页，从0开始，否则为 `None`

    - `page_count`: `(Union[int, None])` 如果输入是PDF文件，表示当前是PDF的总页数，否则为 `None`

    - `width`: `(int)` 原始输入图像的宽度。

    - `height`: `(int)` 原始输入图像的高度。

    - `model_settings`: `(Dict[str, bool])` 配置 PaddleOCR-VL 所需的模型参数

        - `use_doc_preprocessor`: `(bool)` 控制是否启用文档预处理子产线
        - `use_layout_detection`: `(bool)` 控制是否启用版面检测模块
        - `use_chart_recognition`: `(bool)` 控制是否开启图表识别功能
        - `format_block_content`: `(bool)` 控制是否在`JSON`中保存格式化后的markdown内容
        - `merge_layout_blocks`: `(bool)` 控制是否对多栏布局或上下交错分栏的版面框进行合并
        - `markdown_ignore_labels`: `(List[str])` 需要在Markdown中忽略的版面标签，默认为`['number','footnote','header','header_image','footer','footer_image','aside_text']`

    - `doc_preprocessor_res`: `(Dict[str, Union[List[float], str]])` 文档预处理结果dict，仅当`use_doc_preprocessor=True`时存在
        - `input_path`: `(str)` 文档预处理子接受的图像路径，当输入为`numpy.ndarray`时，保存为`None`，此处为`None`
        - `page_index`: `None`，此处的输入为`numpy.ndarray`，所以值为`None`
        - `model_settings`: `(Dict[str, bool])` 文档预处理子的模型配置参数
          - `use_doc_orientation_classify`: `(bool)` 控制是否启用文档图像方向分类子模块
          - `use_doc_unwarping`: `(bool)` 控制是否启用文本图像扭曲矫正子模块
        - `angle`: `(int)` 文档图像方向分类子模块的预测结果，启用时返回实际角度值

    - `parsing_res_list`: `(List[Dict])` 解析结果的列表，每个元素为一个字典，列表顺序为解析后的阅读顺序。
        - `block_bbox`: `(np.ndarray)` 版面区域的边界框。
        - `block_label`: `(str)` 版面区域的标签，例如`text`, `table`等。
        - `block_content`: `(str)` 内容为版面区域内的内容。
        - `block_id`: `(int)` 版面区域的索引，用于显示版面排序结果。
        - `block_order` `(int)` 版面区域的顺序，用于显示版面阅读顺序,对于非排序部分，默认值为 `None`。

- 调用`save_to_json()` 方法会将上述内容保存到指定的 `save_path` 中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_res.json`，如果指定为文件，则直接保存到该文件中。由于 json 文件不支持保存numpy数组，因此会将其中的 `numpy.array` 类型转换为列表形式。json中的字段内容如下：
    - `input_path`: `(str)` 待预测图像或者PDF的输入路径

    - `page_index`: `(Union[int, None])` 如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`

    - `model_settings`: `(Dict[str, bool])` 配置 PaddleOCR-VL 所需的模型参数

        - `use_doc_preprocessor`: `(bool)` 控制是否启用文档预处理子产线
        - `use_layout_detection`: `(bool)` 控制是否启用版面检测模块
        - `use_chart_recognition`: `(bool)` 控制是否开启图表识别功能
        - `format_block_content`: `(bool)` 控制是否在`JSON`中保存格式化后的markdown内容

    - `doc_preprocessor_res`: `(Dict[str, Union[List[float], str]])` 文档预处理结果dict，仅当`use_doc_preprocessor=True`时存在
        - `input_path`: `(str)` 文档预处理子接受的图像路径，当输入为`numpy.ndarray`时，保存为`None`，此处为`None`
        - `page_index`: `None`，此处的输入为`numpy.ndarray`，所以值为`None`
        - `model_settings`: `(Dict[str, bool])` 文档预处理子的模型配置参数
          - `use_doc_orientation_classify`: `(bool)` 控制是否启用文档图像方向分类子模块
          - `use_doc_unwarping`: `(bool)` 控制是否启用文本图像扭曲矫正子模块
        - `angle`: `(int)` 文档图像方向分类子模块的预测结果，启用时返回实际角度值

    - `parsing_res_list`: `(List[Dict])` 解析结果的列表，每个元素为一个字典，列表顺序为解析后的阅读顺序。
        - `block_bbox`: `(np.ndarray)` 版面区域的边界框。
        - `block_label`: `(str)` 版面区域的标签，例如`text`, `table`等。
        - `block_content`: `(str)` 内容为版面区域内的内容。
        - `block_id`: `(int)` 版面区域的索引，用于显示版面排序结果。
        - `block_order` `(int)` 版面区域的顺序，用于显示版面阅读顺序,对于非排序部分，默认值为 `None`。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的 `save_path` 中，如果指定为目录，则会将版面区域检测可视化图像、全局OCR可视化图像、版面阅读顺序可视化图像等内容保存，如果指定为文件，则直接保存到该文件中。
- 调用`save_to_markdown()` 方法会将转化后的 Markdown 文件保存到指定的 `save_path` 中，保存的文件路径为`save_path/{your_img_basename}.md`，如果输入是 PDF 文件，建议直接指定目录，否责多个 markdown 文件会被覆盖。

此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：
<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>json</code></td>
<td>获取预测的 <code>json</code> 格式的结果</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">获取格式为 <code>dict</code> 的可视化图像</td>
</tr>
<tr>
</tr>
<tr>
<td rowspan="3"><code>markdown</code></td>
<td rowspan="3">获取格式为 <code>dict</code> 的 markdown 结果</td>
</tr>
<tr>
</tr>
<tr>
</tr>
</tbody>
</table>

- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。
- `img` 属性返回的预测结果是一个dict类型的数据。其中，键分别为 `layout_det_res` 和 `layout_order_res`，对应的值是 `Image.Image` 对象：分别用于显示版面区域检测和版面阅读顺序结果的可视化图像。如果没有使用可选模块，则dict中只包含 `layout_det_res`。
- `markdown` 属性返回的预测结果是一个dict类型的数据。其中，键分别为 `markdown_texts` 和`markdown_images`，对应的值分别是 markdown 文本，在 Markdown 中显示的图像（`Image.Image` 对象）。

</details>


## 3. 使用推理加速框架提升 VLM 推理性能

默认配置下的推理性能未经过充分优化，可能无法满足实际生产需求。PaddleX 支持通过 vLLM、SGLang 等推理加速框架提升 VLM 的推理性能，从而加快产线推理速度。使用流程主要分为两个步骤：

1. 启动 VLM 推理服务；
2. 配置 PaddleX 产线，作为客户端调用 VLM 推理服务。

### 3.1 启动 VLM 推理服务

#### 3.1.1 使用 Docker 镜像

PaddleX 针对不同推理加速框架提供了相应的 Docker 镜像，用于快速启动 VLM 推理服务：

* **vLLM**：`ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server`
* **SGLang**：`ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-sglang-server`

以 vLLM 为例，可使用以下命令启动服务：

```bash
docker run \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server
```

服务默认监听 **8080** 端口。

启动容器时可传入参数覆盖默认配置，例如：

```bash
docker run \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server \
    paddlex_genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend vllm
```

若您使用的是  NVIDIA 50 系显卡 (Compute Capability >= 12)，需要在启动服务前安装指定版本的 FlashAttention:

```bash
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server \
    /bin/bash
python -m pip install flash-attn==2.8.3
paddlex_genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8118
```

#### 3.1.2 通过 PaddleX CLI 和启动

由于推理加速框架可能与飞桨框架存在依赖冲突，建议在虚拟环境中安装。示例如下：

```bash
# 创建虚拟环境
python -m venv .venv
# 激活环境
source .venv/bin/activate
# 安装 PaddleX
python -m pip install "paddlex[ocr]"
# 安装 vLLM 服务器插件
paddlex --install genai-vllm-server
# 安装 SGLang 服务器插件
# paddlex --install genai-sglang-server
```

若您使用的是  NVIDIA 50 系显卡 (Compute Capability >= 12)，需要在启动服务前安装指定版本的 FlashAttention:

```bash
python -m pip install flash-attn==2.8.3
```

安装完成后，可通过 `paddlex_genai_server` 命令启动服务：

```bash
paddlex_genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8118
```

该命令支持的参数如下：

| 参数                 | 说明                        |
| ------------------ | ------------------------- |
| `--model_name`     | 模型名称                      |
| `--model_dir`      | 模型目录                      |
| `--host`           | 服务器主机名                    |
| `--port`           | 服务器端口号                    |
| `--backend`        | 后端名称，即使用的推理加速框架名称，可选 `vllm` 或 `sglang` |
| `--backend_config` | 可指定 YAML 文件，包含后端配置        |

### 3.2 客户端使用方法

启动 VLM 推理服务后，客户端即可通过 PaddleX 调用该服务。在使用前，需要安装客户端插件：

```bash
paddlex --install genai-client
```

接着，获取产线配置文件：

```bash
paddlex --get_pipeline_config PaddleOCR-VL
```

配置文件的默认保存路径为 `PaddleOCR-VL.yaml`。将配置文件中的 `VLRecognition.genai_config.backend` 和 `VLRecognition.genai_config.server_url` 字段修改为与此前启动的服务相对应的值，例如：

```yaml
VLRecognition:
  ...
  genai_config:
    backend: vllm-server
    server_url: http://127.0.0.1:8118/v1
```

之后，可以使用修改好的配置文件进行产线调用。例如通过 CLI 调用：

```bash
paddlex --pipeline PaddleOCR-VL.yaml --input paddleocr_vl_demo.png
```

或通过 Python API 调用：

```python
from paddlex import create_pipeline

pipeline = create_pipeline("PaddleOCR-VL.yaml")

for res in pipeline.predict("paddleocr_vl_demo.png"):
    res.print()
```

### 3.3 性能调优

默认配置是在单张 NVIDIA A100 上进行调优的，并假设客户端独占服务，因此可能不适用于其他环境。如果用户在实际使用中遇到性能问题，可以尝试以下优化方法。

#### 3.3.1 服务端参数调整

不同推理加速框架支持的参数不同，可参考各自官方文档了解可用参数及其调整时机：

- [vLLM 官方参数调优指南](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- [SGLang 超参数调整文档](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html)

PaddleX VLM 推理服务支持通过配置文件进行调参。以下示例展示如何调整 vLLM 服务器的 `gpu-memory-utilization` 和 `max-num-seqs` 参数：

1. 创建 YAML 文件 `vllm_config.yaml`，内容如下：

   ```yaml
   gpu-memory-utilization: 0.3
   max-num-seqs: 128
   ```

2. 启动服务时指定配置文件路径：

   ```bash
   paddlex_genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --backend_config vllm_config.yaml
   ```

如果使用支持进程替换（process substitution）的 shell（如 Bash），也可以无需创建配置文件，直接在启动服务时传入配置项：

```bash
paddlex_genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --backend_config <(echo -e 'gpu-memory-utilization: 0.3\nmax-num-seqs: 128')
```

#### 3.3.2 客户端参数调整

PaddleX 会将来自单张或多张输入图像中的子图分组并对服务器发起并发请求，因此并发请求数对性能影响显著。用户可通过修改配置文件中 `VLRecognition.genai_config.max_concurrency` 字段设置最大并发请求数。

当客户端与 VLM 推理服务为 1 对 1 且服务端资源充足时，可适当增加并发数以提升性能；若服务端需支持多个客户端或计算资源有限，则应降低并发数，以避免资源过载导致服务异常。

#### 3.3.3 常用硬件性能调优建议

以下配置均针对客户端与 VLM 推理服务为 1 对 1 的场景。

**NVIDIA RTX 3060**

- **服务端**
    - vLLM：`gpu-memory-utilization: 0.7`
    - FastDeploy：
        - `gpu-memory-utilization: 0.7`
        - `max-concurrency: 2048`


## 4. 服务化部署

若您需要将 PaddleOCR-VL 直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了服务化部署方式，详细说明如下：

### 4.1 安装依赖

执行如下命令，通过 PaddleX CLI 安装 PaddleX 服务化部署插件：

```bash
paddlex --install serving
```

### 4.2 运行服务器

通过 PaddleX CLI 运行服务器：

```bash
paddlex --serve --pipeline PaddleOCR-VL
```

可以看到类似以下展示的信息：

```text
INFO:     Started server process [63108]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

如需调整配置（如模型路径、batch size、部署设备等），可指定 `--pipeline` 为自定义配置文件。

与服务化部署相关的命令行选项如下：

<table>
<thead>
<tr>
<th>名称</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>--pipeline</code></td>
<td>PaddleX 产线注册名或产线配置文件路径。</td>
</tr>
<tr>
<td><code>--device</code></td>
<td>产线部署设备。默认情况下，当 GPU 可用时，将使用 GPU；否则使用 CPU。</td>
</tr>
<tr>
<td><code>--host</code></td>
<td>服务器绑定的主机名或 IP 地址。默认为 <code>0.0.0.0</code>。</td>
</tr>
<tr>
<td><code>--port</code></td>
<td>服务器监听的端口号。默认为 <code>8080</code>。</td>
</tr>
<tr>
<td><code>--use_hpip</code></td>
<td>如果指定，则使用高性能推理。请参考高性能推理文档了解更多信息。</td>
</tr>
<tr>
<td><code>--hpi_config</code></td>
<td>高性能推理配置。请参考高性能推理文档了解更多信息。</td>
</tr>
</tbody>
</table>

### 4.3 客户端调用

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
<p>进行版面解析。</p>
<p><code>POST /layout-parsing</code></p>
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
<td><code>file</code></td>
<td><code>string</code></td>
<td>服务器可访问的图像文件或PDF文件的URL，或上述类型文件内容的Base64编码结果。默认对于超过10页的PDF文件，只有前10页的内容会被处理。<br /> 要解除页数限制，请在产线配置文件中添加以下配置：
<pre><code>Serving:
  extra:
    max_num_input_imgs: null
</code></pre>
</td>
<td>是</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code>｜<code>null</code></td>
<td>文件类型。<code>0</code>表示PDF文件，<code>1</code>表示图像文件。若请求体无此属性，则将根据URL推断文件类型。</td>
<td>否</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_doc_orientation_classify</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>use_doc_unwarping</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useLayoutDetection</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>use_layout_detection</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useChartRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>use_chart_recognition</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useSealRecognition</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>use_seal_recognition</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>useOcrForImageBlock</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>use_ocr_for_image_block</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>object</code> | </code><code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>layout_threshold</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>layout_nms</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>object</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>layout_unclip_ratio</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>object</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>layout_merge_bboxes_mode</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>layoutShapeMode</code></td>
<td><code>string</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>layout_shape_mode</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>promptLabel</code></td>
<td><code>string</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>prompt_label</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>formatBlockContent</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>format_block_content</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>repetitionPenalty</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>repetition_penalty</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>temperature</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>temperature</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>topP</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>top_p</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>minPixels</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>min_pixels</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>maxPixels</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>max_pixels</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>maxNewTokens</code></td>
<td><code>number</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>max_new_tokens</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>mergeLayoutBlocks</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>merge_layout_blocks</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>markdownIgnoreLabels</code></td>
<td><code>array</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>mardown_ignore_labels</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>vlmExtraArgs</code></td>
<td><code>object</code> | <code>null</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>predict</code> 方法的 <code>vlm_extra_args</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>prettifyMarkdown</code></td>
<td><code>boolean</code></td>
<td>是否输出美化后的 Markdown 文本。默认为 <code>true</code>。</td>
<td>否</td>
</tr>
<tr>
<td><code>showFormulaNumber</code></td>
<td><code>boolean</code></td>
<td>输出的 Markdown 文本中是否包含公式编号。默认为 <code>false</code>。</td>
<td>否</td>
</tr>
<tr>
<td><code>restructurePages</code></td>
<td><code>boolean</code></td>
<td>是否重构多页结果。默认为 <code>false</code>。</td>
<td>否</td>
</tr>
<tr>
<td><code>mergeTables</code></td>
<td><code>boolean</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>restructure_pages</code> 方法的 <code>merge_table</code> 参数相关说明。仅当<code>restructurePages</code>为<code>true</code>时生效。</td>
<td>否</td>
</tr>
<tr>
<td><code>relevelTitles</code></td>
<td><code>boolean</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>restructure_pages</code> 方法的 <code>relevel_titles</code> 参数相关说明。仅当<code>restructurePages</code>为<code>true</code>时生效。</td>
<td>否</td>
</tr>
<tr>
<td><code>visualize</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>是否返回可视化结果图以及处理过程中的中间图像等。
<ul style="margin: 0 0 0 1em; padding-left: 0em;">
<li>传入 <code>true</code>：返回图像。</li>
<li>传入 <code>false</code>：不返回图像。</li>
<li>若请求体中未提供该参数或传入 <code>null</code>：遵循配置文件<code>Serving.visualize</code> 的设置。</li>
</ul>
<br/>例如，在配置文件中添加如下字段：<br/>
<pre><code>Serving:
  visualize: False
</code></pre>
将默认不返回图像，通过请求体中的<code>visualize</code>参数可以覆盖默认行为。如果请求体和配置文件中均未设置（或请求体传入<code>null</code>、配置文件中未设置），则默认返回图像。
</td>
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
<td><code>layoutParsingResults</code></td>
<td><code>array</code></td>
<td>版面解析结果。数组长度为1（对于图像输入）或实际处理的文档页数（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中实际处理的每一页的结果。</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>输入数据信息。</td>
</tr>
</tbody>
</table>
<p><code>layoutParsingResults</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>对象的 <code>predict</code> 方法生成结果的 JSON 表示中 <code>res</code> 字段的简化版本，其中去除了 <code>input_path</code> 和 <code>page_index</code> 字段。</td>
</tr>
<tr>
<td><code>markdown</code></td>
<td><code>object</code></td>
<td>Markdown结果。</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>参见预测结果的 <code>img</code> 属性说明。图像为JPEG格式，使用Base64编码。</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> | <code>null</code></td>
<td>输入图像。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<p><code>markdown</code>为一个<code>object</code>，具有如下属性：</p>
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
<td><code>text</code></td>
<td><code>string</code></td>
<td>Markdown文本。</td>
</tr>
<tr>
<td><code>images</code></td>
<td><code>object</code></td>
<td>Markdown图片相对路径和Base64编码图像的键值对。</td>
</tr>
</tbody>
</table>
<ul>
<li><b><code>restructurePages</code></b></li>
</ul>
<p>重构多页结果。</p>
<p><code>POST /restructure-pages</code></p>
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
<td><code>pages</code></td>
<td><code>array</code></td>
<td>页面数组。
</td>
<td>是</td>
</tr>
<tr>
<td><code>mergeTables</code></td>
<td><code>boolean</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>restructure_pages</code> 方法的 <code>merge_tables</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>relevelTitles</code></td>
<td><code>boolean</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>restructure_pages</code> 方法的 <code>relevel_titles</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>concatenatePages</code></td>
<td><code>boolean</code></td>
<td>请参阅PaddleOCR-VL对象中 <code>restructure_pages</code> 方法的 <code>concatenate_pages</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>prettifyMarkdown</code></td>
<td><code>boolean</code></td>
<td>是否输出美化后的 Markdown 文本。默认为 <code>true</code>。</td>
<td>否</td>
</tr>
<tr>
<td><code>showFormulaNumber</code></td>
<td><code>boolean</code></td>
<td>输出的 Markdown 文本中是否包含公式编号。默认为 <code>false</code>。</td>
<td>否</td>
</tr>
</tbody>
</table>
<p><code>pages</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>对应<code>infer</code>操作返回的<code>prunedResult</code>对象。</td>
</tr>
<tr>
<td><code>markdownImages</code></td>
<td><code>object</code>|<code>null</code></td>
<td>对应<code>infer</code>操作返回的<code>markdown</code>对象的<code>images</code>属性。</td>
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
<td><code>layoutParsingResults</code></td>
<td><code>array</code></td>
<td>重构后的版面解析结果。其中每个元素包含的字段请参见对<code>infer</code>操作返回结果的说明（不含可视化结果图和中间图像）。</td>
</tr>
</tbody>
</table>
</details>
<details><summary>多语言调用服务示例</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">
import base64
import requests
import pathlib

BASE_URL = "http://localhost:8080"

image_path = "./demo.jpg"

# 对本地图像进行Base64编码
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "file": image_data, # Base64编码的文件内容或者文件URL
    "fileType": 1, # 文件类型，1表示图像文件
}

response = requests.post(BASE_URL + "/layout-parsing", json=payload)
assert response.status_code == 200, (response.status_code, response.text)

result = response.json()["result"]
pages = []
for i, res in enumerate(result["layoutParsingResults"]):
    pages.append({"prunedResult": res["prunedResult"], "markdownImages": res["markdown"].get("images")})
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        pathlib.Path(img_path).parent.mkdir(exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")

payload = {
    "pages": pages,
    "concatenatePages": True,
}

response = requests.post(BASE_URL + "/restructure-pages", json=payload)
assert response.status_code == 200, (response.status_code, response.text)

result = response.json()["result"]
res = result["layoutParsingResults"][0]
print(res["prunedResult"])
md_dir = pathlib.Path("markdown")
md_dir.mkdir(exist_ok=True)
(md_dir / "doc.md").write_text(res["markdown"]["text"])
for img_path, img in res["markdown"]["images"].items():
    img_path = md_dir / img_path
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(base64.b64decode(img))
print(f"Markdown document saved at {md_dir / 'doc.md'}")
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &lt;filesystem&gt;
#include &lt;fstream&gt;
#include &lt;vector&gt;
#include &lt;string&gt;
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

namespace fs = std::filesystem;

int main() {
    httplib::Client client("localhost", 8080);

    const std::string filePath = "./demo.jpg";

    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return 1;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }

    std::string bufferStr(buffer.data(), static_cast<size_t>(size));
    std::string encodedFile = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["file"] = encodedFile;
    jsonObj["fileType"] = 1;

    auto response = client.Post("/layout-parsing", jsonObj.dump(), "application/json");

    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        if (!result.is_object() || !result.contains("layoutParsingResults")) {
            std::cerr << "Unexpected response format." << std::endl;
            return 1;
        }

        const auto& results = result["layoutParsingResults"];
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& res = results[i];

            if (res.contains("prunedResult")) {
                std::cout << "Layout result [" << i << "]: " << res["prunedResult"].dump() << std::endl;
            }

            if (res.contains("outputImages") && res["outputImages"].is_object()) {
                for (auto& [imgName, imgBase64] : res["outputImages"].items()) {
                    std::string outputPath = imgName + "_" + std::to_string(i) + ".jpg";
                    fs::path pathObj(outputPath);
                    fs::path parentDir = pathObj.parent_path();
                    if (!parentDir.empty() && !fs::exists(parentDir)) {
                        fs::create_directories(parentDir);
                    }

                    std::string decodedImage = base64::from_base64(imgBase64.get<std::string>());

                    std::ofstream outFile(outputPath, std::ios::binary);
                    if (outFile.is_open()) {
                        outFile.write(decodedImage.c_str(), decodedImage.size());
                        outFile.close();
                        std::cout << "Saved image: " << outputPath << std::endl;
                    } else {
                        std::cerr << "Failed to save image: " << outputPath << std::endl;
                    }
                }
            }
        }
    } else {
        std::cerr << "Request failed." << std::endl;
        if (response) {
            std::cerr << "HTTP status: " << response->status << std::endl;
            std::cerr << "Response body: " << response->body << std::endl;
        }
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
import java.nio.file.Paths;
import java.nio.file.Files;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/layout-parsing";
        String imagePath = "./demo.jpg";

        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String base64Image = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("file", base64Image);
        payload.put("fileType", 1);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.get("application/json; charset=utf-8");

        RequestBody body = RequestBody.create(JSON, payload.toString());

        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode root = objectMapper.readTree(responseBody);
                JsonNode result = root.get("result");

                JsonNode layoutParsingResults = result.get("layoutParsingResults");
                for (int i = 0; i < layoutParsingResults.size(); i++) {
                    JsonNode item = layoutParsingResults.get(i);
                    int finalI = i;
                    JsonNode prunedResult = item.get("prunedResult");
                    System.out.println("Pruned Result [" + i + "]: " + prunedResult.toString());

                    JsonNode outputImages = item.get("outputImages");
                    outputImages.fieldNames().forEachRemaining(imgName -> {
                        try {
                            String imgBase64 = outputImages.get(imgName).asText();
                            byte[] imgBytes = Base64.getDecoder().decode(imgBase64);
                            String imgPath = imgName + "_" + finalI + ".jpg";

                            File outputFile = new File(imgPath);
                            File parentDir = outputFile.getParentFile();
                            if (parentDir != null && !parentDir.exists()) {
                                parentDir.mkdirs();
                                System.out.println("Created directory: " + parentDir.getAbsolutePath());
                            }

                            try (FileOutputStream fos = new FileOutputStream(outputFile)) {
                                fos.write(imgBytes);
                                System.out.println("Saved image: " + imgPath);
                            }
                        } catch (IOException e) {
                            System.err.println("Failed to save image: " + e.getMessage());
                        }
                    });
                }
            } else {
                System.err.println("Request failed with HTTP code: " + response.code());
            }
        }
    }
}
</code></pre></details>

<details><summary>Go</summary>

<pre><code class="language-go">package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
    "os"
    "path/filepath"
)

func main() {
    API_URL := "http://localhost:8080/layout-parsing"
    filePath := "./demo.jpg"

    fileBytes, err := ioutil.ReadFile(filePath)
    if err != nil {
        fmt.Printf("Error reading file: %v\n", err)
        return
    }
    fileData := base64.StdEncoding.EncodeToString(fileBytes)

    payload := map[string]interface{}{
        "file":     fileData,
        "fileType": 1,
    }
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Printf("Error marshaling payload: %v\n", err)
        return
    }

    client := &http.Client{}
    req, err := http.NewRequest("POST", API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Printf("Error creating request: %v\n", err)
        return
    }
    req.Header.Set("Content-Type", "application/json")

    res, err := client.Do(req)
    if err != nil {
        fmt.Printf("Error sending request: %v\n", err)
        return
    }
    defer res.Body.Close()

    if res.StatusCode != http.StatusOK {
        fmt.Printf("Unexpected status code: %d\n", res.StatusCode)
        return
    }

    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Printf("Error reading response: %v\n", err)
        return
    }

    type Markdown struct {
        Text   string            `json:"text"`
        Images map[string]string `json:"images"`
    }

    type LayoutResult struct {
        PrunedResult map[string]interface{} `json:"prunedResult"`
        Markdown     Markdown               `json:"markdown"`
        OutputImages map[string]string      `json:"outputImages"`
        InputImage   *string                `json:"inputImage"`
    }

    type Response struct {
        Result struct {
            LayoutParsingResults []LayoutResult `json:"layoutParsingResults"`
            DataInfo             interface{}    `json:"dataInfo"`
        } `json:"result"`
    }

    var respData Response
    if err := json.Unmarshal(body, &respData); err != nil {
        fmt.Printf("Error parsing response: %v\n", err)
        return
    }

    for i, res := range respData.Result.LayoutParsingResults {
        fmt.Printf("Result %d - prunedResult: %+v\n", i, res.PrunedResult)

        mdDir := fmt.Sprintf("markdown_%d", i)
        os.MkdirAll(mdDir, 0755)
        mdFile := filepath.Join(mdDir, "doc.md")
        if err := os.WriteFile(mdFile, []byte(res.Markdown.Text), 0644); err != nil {
            fmt.Printf("Error writing markdown file: %v\n", err)
        } else {
            fmt.Printf("Markdown document saved at %s\n", mdFile)
        }

        for path, imgBase64 := range res.Markdown.Images {
            fullPath := filepath.Join(mdDir, path)
            if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
                fmt.Printf("Error creating directory for markdown image: %v\n", err)
                continue
            }
            imgBytes, err := base64.StdEncoding.DecodeString(imgBase64)
            if err != nil {
                fmt.Printf("Error decoding markdown image: %v\n", err)
                continue
            }
            if err := os.WriteFile(fullPath, imgBytes, 0644); err != nil {
                fmt.Printf("Error saving markdown image: %v\n", err)
            }
        }

        for name, imgBase64 := range res.OutputImages {
            imgBytes, err := base64.StdEncoding.DecodeString(imgBase64)
            if err != nil {
                fmt.Printf("Error decoding output image %s: %v\n", name, err)
                continue
            }
            filename := fmt.Sprintf("%s_%d.jpg", name, i)

            if err := os.MkdirAll(filepath.Dir(filename), 0755); err != nil {
                fmt.Printf("Error creating directory for output image: %v\n", err)
                continue
            }

            if err := os.WriteFile(filename, imgBytes, 0644); err != nil {
                fmt.Printf("Error saving output image %s: %v\n", filename, err)
            } else {
                fmt.Printf("Output image saved at %s\n", filename)
            }
        }
    }
}
</code></pre></details>

<details><summary>C#</summary>

<pre><code class="language-csharp">using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = "http://localhost:8080/layout-parsing";
    static readonly string inputFilePath = "./demo.jpg";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] fileBytes = File.ReadAllBytes(inputFilePath);
        string fileData = Convert.ToBase64String(fileBytes);

        var payload = new JObject
        {
            { "file", fileData },
            { "fileType", 1 }
        };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        JArray layoutParsingResults = (JArray)jsonResponse["result"]["layoutParsingResults"];
        for (int i = 0; i < layoutParsingResults.Count; i++)
        {
            var res = layoutParsingResults[i];
            Console.WriteLine($"[{i}] prunedResult:\n{res["prunedResult"]}");

            JObject outputImages = res["outputImages"] as JObject;
            if (outputImages != null)
            {
                foreach (var img in outputImages)
                {
                    string imgName = img.Key;
                    string base64Img = img.Value?.ToString();
                    if (!string.IsNullOrEmpty(base64Img))
                    {
                        string imgPath = $"{imgName}_{i}.jpg";
                        byte[] imageBytes = Convert.FromBase64String(base64Img);

                        string directory = Path.GetDirectoryName(imgPath);
                        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                        {
                            Directory.CreateDirectory(directory);
                            Console.WriteLine($"Created directory: {directory}");
                        }

                        File.WriteAllBytes(imgPath, imageBytes);
                        Console.WriteLine($"Output image saved at {imgPath}");
                    }
                }
            }
        }
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');
const path = require('path');

const API_URL = 'http://localhost:8080/layout-parsing';
const imagePath = './demo.jpg';
const fileType = 1;

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

const payload = {
  file: encodeImageToBase64(imagePath),
  fileType: fileType
};

axios.post(API_URL, payload)
  .then(response => {
    const results = response.data.result.layoutParsingResults;
    results.forEach((res, index) => {
      console.log(`\n[${index}] prunedResult:`);
      console.log(res.prunedResult);

      const outputImages = res.outputImages;
      if (outputImages) {
        Object.entries(outputImages).forEach(([imgName, base64Img]) => {
          const imgPath = `${imgName}_${index}.jpg`;

          const directory = path.dirname(imgPath);
          if (!fs.existsSync(directory)) {
            fs.mkdirSync(directory, { recursive: true });
            console.log(`Created directory: ${directory}`);
          }

          fs.writeFileSync(imgPath, Buffer.from(base64Img, 'base64'));
          console.log(`Output image saved at ${imgPath}`);
        });
      } else {
        console.log(`[${index}] No outputImages.`);
      }
    });
  })
  .catch(error => {
    console.error('Error during API request:', error.message || error);
  });
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/layout-parsing";
$image_path = "./demo.jpg";

$image_data = base64_encode(file_get_contents($image_path));
$payload = array("file" => $image_data, "fileType" => 1);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)["result"]["layoutParsingResults"];

foreach ($result as $i => $item) {
    echo "[$i] prunedResult:\n";
    print_r($item["prunedResult"]);

    if (!empty($item["outputImages"])) {
        foreach ($item["outputImages"] as $img_name => $img_base64) {
            $output_image_path = "{$img_name}_{$i}.jpg";

            $directory = dirname($output_image_path);
            if (!is_dir($directory)) {
                mkdir($directory, 0777, true);
                echo "Created directory: $directory\n";
            }

            file_put_contents($output_image_path, base64_decode($img_base64));
            echo "Output image saved at $output_image_path\n";
        }
    } else {
        echo "No outputImages found for item $i\n";
    }
}
?&gt;
</code></pre></details>
</details>
<br/>
