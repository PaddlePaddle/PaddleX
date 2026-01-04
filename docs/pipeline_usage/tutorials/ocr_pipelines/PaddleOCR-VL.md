---
comments: true
---

# PaddleOCR-VL介绍

PaddleOCR-VL 是一款先进、高效的文档解析模型，专为文档中的元素识别设计。其核心组件为 PaddleOCR-VL-0.9B，这是一种紧凑而强大的视觉语言模型（VLM），它由 NaViT 风格的动态分辨率视觉编码器与 ERNIE-4.5-0.3B 语言模型组成，能够实现精准的元素识别。该模型支持 109 种语言，并在识别复杂元素（如文本、表格、公式和图表）方面表现出色，同时保持极低的资源消耗。通过在广泛使用的公开基准与内部基准上的全面评测，PaddleOCR-VL 在页级级文档解析与元素级识别均达到 SOTA 表现。它显著优于现有的基于Pipeline方案和文档解析多模态方案以及先进的通用多模态大模型，并具备更快的推理速度。这些优势使其非常适合在真实场景中落地部署。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl/metrics/allmetric.png"/>

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
{'res': {'input_path': 'paddleocr_vl_demo.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True, 'use_chart_recognition': False, 'format_block_content': False}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 6, 'label': 'doc_title', 'score': 0.9636914134025574, 'coordinate': [np.float32(131.31366), np.float32(36.450516), np.float32(1384.522), np.float32(127.984665)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9281806349754333, 'coordinate': [np.float32(585.39465), np.float32(158.438), np.float32(930.2184), np.float32(182.57469)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9840355515480042, 'coordinate': [np.float32(9.023666), np.float32(200.86115), np.float32(361.41583), np.float32(343.8828)]}, {'cls_id': 14, 'label': 'image', 'score': 0.9871416091918945, 'coordinate': [np.float32(775.50574), np.float32(200.66502), np.float32(1503.3807), np.float32(684.9304)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9801855087280273, 'coordinate': [np.float32(9.532196), np.float32(344.90594), np.float32(361.4413), np.float32(440.8244)]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9708921313285828, 'coordinate': [np.float32(28.040405), np.float32(455.87976), np.float32(341.7215), np.float32(520.7117)]}, {'cls_id': 24, 'label': 'vision_footnote', 'score': 0.9002962708473206, 'coordinate': [np.float32(809.0692), np.float32(703.70044), np.float32(1488.3016), np.float32(750.5238)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9825374484062195, 'coordinate': [np.float32(8.896561), np.float32(536.54895), np.float32(361.05237), np.float32(655.8058)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9822263717651367, 'coordinate': [np.float32(8.971573), np.float32(657.4949), np.float32(362.01715), np.float32(774.625)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9767460823059082, 'coordinate': [np.float32(9.407074), np.float32(776.5216), np.float32(361.31067), np.float32(846.82874)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9868153929710388, 'coordinate': [np.float32(8.669495), np.float32(848.2543), np.float32(361.64703), np.float32(1062.8568)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9826608300209045, 'coordinate': [np.float32(8.8025055), np.float32(1063.8615), np.float32(361.46588), np.float32(1182.8524)]}, {'cls_id': 22, 'label': 'text', 'score': 0.982555627822876, 'coordinate': [np.float32(8.820602), np.float32(1184.4663), np.float32(361.66394), np.float32(1302.4507)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9584776759147644, 'coordinate': [np.float32(9.170288), np.float32(1304.2161), np.float32(361.48898), np.float32(1351.7483)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9782056212425232, 'coordinate': [np.float32(389.1618), np.float32(200.38202), np.float32(742.7591), np.float32(295.65146)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9844875931739807, 'coordinate': [np.float32(388.73303), np.float32(297.18463), np.float32(744.00024), np.float32(441.3034)]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9680547714233398, 'coordinate': [np.float32(409.39468), np.float32(455.89386), np.float32(721.7174), np.float32(520.9387)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9741666913032532, 'coordinate': [np.float32(389.71606), np.float32(536.8138), np.float32(742.7112), np.float32(608.00165)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9840384721755981, 'coordinate': [np.float32(389.30988), np.float32(609.39636), np.float32(743.09247), np.float32(750.3231)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9845995306968689, 'coordinate': [np.float32(389.13272), np.float32(751.7772), np.float32(743.058), np.float32(894.8815)]}, {'cls_id': 22, 'label': 'text', 'score': 0.984852135181427, 'coordinate': [np.float32(388.83267), np.float32(896.0371), np.float32(743.58215), np.float32(1038.7345)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9804865717887878, 'coordinate': [np.float32(389.08478), np.float32(1039.9119), np.float32(742.7585), np.float32(1134.4897)]}, {'cls_id': 22, 'label': 'text', 'score': 0.986461341381073, 'coordinate': [np.float32(388.52643), np.float32(1135.8137), np.float32(743.451), np.float32(1352.0085)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9869391918182373, 'coordinate': [np.float32(769.8341), np.float32(775.66235), np.float32(1124.9813), np.float32(1063.207)]}, {'cls_id': 22, 'label': 'text', 'sc
