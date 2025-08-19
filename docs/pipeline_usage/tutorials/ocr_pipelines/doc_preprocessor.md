---
comments: true
---

# 文档图像预处理产线使用教程

## 1. 文档图像预处理产线介绍

文档图像预处理产线集成了文档方向分类和形变矫正两大功能。文档方向分类可自动识别文档的四个方向（0°、90°、180°、270°），确保文档以正确的方向进行后续处理。文本图像矫正模型则用于修正文档拍摄或扫描过程中的几何扭曲，恢复文档的原始形状和比例。适用于数字化文档管理、OCR类任务前处理、以及任何需要提高文档图像质量的场景。通过自动化的方向校正与形变矫正，该模块显著提升了文档处理的准确性和效率，为用户提供更为可靠的图像分析基础。本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/doc_preprocessor/02.jpg">

<b>通用文档图像预处理</b><b>产线中包含可选用的文档图像方向分类模块和文本图像矫正模块</b>，每个模块都包含多个模型，您可以根据下方的基准测试数据选择使用的模型。

### 1.1 模型基准测试数据

> 推理耗时仅包含模型推理耗时，不包含前后处理耗时。

<p><b>文档图像方向分类模块（可选）：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td>
<td>99.06</td>
<td>2.62 / 0.59</td>
<td>3.24 / 1.19</td>
<td>7</td>
<td>基于PP-LCNet_x1_0的文档图像分类模型，含有四个类别，即0度，90度，180度，270度</td>
</tr>
</tbody>
</table>

<p><b>文本图像矫正模块（可选）：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>CER </th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">训练模型</a></td>
<td>0.179</td>
<td>19.05 / 19.05</td>
<td>- / 869.82</td>
<td>30.3</td>
<td>高精度文本图像矫正模型</td>
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
                  <li>文档图像方向分类模型：PaddleX自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。</li>
                  <li> 文本图像矫正模型：<a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet。</a></li>
                </ul>
             </li>
              <li><strong>硬件配置：</strong>
                  <ul>
                      <li>GPU：NVIDIA Tesla T4</li>
                      <li>CPU：Intel Xeon Gold 6271C @ 2.60GHz</li>
                  </ul>
              </li>
              <li><strong>软件环境：</strong>
                  <ul>
                      <li>Ubuntu 20.04 / CUDA 11.8 / cuDNN 8.9 / TensorRT 8.6.1.6</li>
                      <li>paddlepaddle 3.0.0 / paddlex 3.0.3</li>
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

### 1.2 产线基准测试数据

<details>
<summary>点击展开/折叠表格</summary>

<table border="1">
<tr><th>流水线配置</th><th>硬件</th><th>平均推理时间 (ms)</th><th>峰值CPU利用率 (%)</th><th>平均CPU利用率 (%)</th><th>峰值主机内存 (MB)</th><th>平均主机内存 (MB)</th><th>峰值GPU利用率 (%)</th><th>平均GPU利用率 (%)</th><th>峰值设备内存 (MB)</th><th>平均设备内存 (MB)</th></tr>
<tr>
<td rowspan="9">doc_preprocessor-default</td>
<td>Intel 6271C</td>
<td>1127.85</td>
<td>1012.50</td>
<td>789.07</td>
<td>1913.89</td>
<td>1650.30</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>764.23</td>
<td>1003.70</td>
<td>782.77</td>
<td>1916.81</td>
<td>1655.66</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>96.74</td>
<td>174.70</td>
<td>135.91</td>
<td>2023.54</td>
<td>1963.53</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>76.66</td>
<td>148.90</td>
<td>125.73</td>
<td>1748.06</td>
<td>1683.35</td>
<td>11</td>
<td>5.86</td>
<td>828.00</td>
<td>828.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>114.96</td>
<td>168.60</td>
<td>131.05</td>
<td>1837.66</td>
<td>1707.50</td>
<td>26</td>
<td>11.65</td>
<td>694.00</td>
<td>694.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>69.34</td>
<td>140.90</td>
<td>122.80</td>
<td>1992.86</td>
<td>1911.34</td>
<td>11</td>
<td>7.75</td>
<td>890.00</td>
<td>890.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>81.65</td>
<td>146.40</td>
<td>125.37</td>
<td>1896.46</td>
<td>1825.98</td>
<td>14</td>
<td>5.93</td>
<td>586.00</td>
<td>586.00</td>
</tr>
<tr>
<td>M4</td>
<td>368.93</td>
<td>118.20</td>
<td>103.40</td>
<td>1973.23</td>
<td>1756.00</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>122.40</td>
<td>156.90</td>
<td>123.75</td>
<td>1874.29</td>
<td>1727.01</td>
<td>73</td>
<td>23.81</td>
<td>450.00</td>
<td>450.00</td>
</tr>
<tr>
<td rowspan="9">doc_preprocessor-clsonly</td>
<td>Intel 6271C</td>
<td>55.12</td>
<td>1085.60</td>
<td>1057.12</td>
<td>1254.27</td>
<td>1203.36</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>59.48</td>
<td>1051.60</td>
<td>1043.39</td>
<td>1263.88</td>
<td>1206.67</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>47.44</td>
<td>202.70</td>
<td>173.75</td>
<td>1868.11</td>
<td>1810.80</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>43.27</td>
<td>167.80</td>
<td>150.56</td>
<td>1347.92</td>
<td>1325.32</td>
<td>2</td>
<td>1.25</td>
<td>514.00</td>
<td>514.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>53.72</td>
<td>202.70</td>
<td>170.72</td>
<td>1404.16</td>
<td>1352.74</td>
<td>2</td>
<td>1.20</td>
<td>394.00</td>
<td>394.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>41.89</td>
<td>153.80</td>
<td>141.46</td>
<td>1590.48</td>
<td>1540.25</td>
<td>2</td>
<td>1.12</td>
<td>578.00</td>
<td>578.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>44.61</td>
<td>179.80</td>
<td>156.96</td>
<td>1569.85</td>
<td>1514.64</td>
<td>2</td>
<td>0.88</td>
<td>302.00</td>
<td>302.00</td>
</tr>
<tr>
<td>M4</td>
<td>34.12</td>
<td>127.60</td>
<td>123.24</td>
<td>1459.06</td>
<td>1386.33</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>51.81</td>
<td>180.80</td>
<td>154.34</td>
<td>1515.43</td>
<td>1465.99</td>
<td>4</td>
<td>2.89</td>
<td>160.00</td>
<td>160.00</td>
</tr>
<tr>
<td rowspan="9">doc_preprocessor-unwarponly</td>
<td>Intel 6271C</td>
<td>1092.82</td>
<td>1002.10</td>
<td>749.82</td>
<td>1875.45</td>
<td>1680.20</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>747.32</td>
<td>951.30</td>
<td>754.01</td>
<td>1843.54</td>
<td>1645.73</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>95.22</td>
<td>101.20</td>
<td>100.08</td>
<td>1960.48</td>
<td>1863.72</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>78.05</td>
<td>100.90</td>
<td>100.15</td>
<td>1570.25</td>
<td>1446.01</td>
<td>9</td>
<td>5.57</td>
<td>788.00</td>
<td>788.00</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>130.84</td>
<td>100.90</td>
<td>100.25</td>
<td>1561.52</td>
<td>1459.73</td>
<td>24</td>
<td>11.35</td>
<td>656.00</td>
<td>656.00</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>68.62</td>
<td>106.90</td>
<td>100.62</td>
<td>1808.00</td>
<td>1715.35</td>
<td>6</td>
<td>4.83</td>
<td>684.00</td>
<td>684.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>84.19</td>
<td>101.90</td>
<td>100.14</td>
<td>1848.25</td>
<td>1719.24</td>
<td>19</td>
<td>7.73</td>
<td>544.00</td>
<td>544.00</td>
</tr>
<tr>
<td>M4</td>
<td>295.15</td>
<td>102.40</td>
<td>100.76</td>
<td>1957.42</td>
<td>1734.67</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>117.25</td>
<td>101.80</td>
<td>100.36</td>
<td>1674.25</td>
<td>1605.14</td>
<td>48</td>
<td>26.45</td>
<td>412.00</td>
<td>412.00</td>
</tr>
</table>


<table border="1">
<tr><th>Pipeline configuration</th><th>description</th></tr>
<tr>
<td>doc_preprocessor-default</td>
<td>默认配置</td>
</tr>
<tr>
<td>doc_preprocessor-clsonly</td>
<td>默认配置基础上，仅开启文档图像方向分类</td>
</tr>
<tr>
<td>doc_preprocessor-unwarponly</td>
<td>默认配置基础上，仅开启文本图像矫正</td>
</tr>
</table>
</details>


* 测试环境：
    * PaddlePaddle 3.1.0、CUDA 11.8、cuDNN 8.9
    * PaddleX @ develop (f1eb28e23cfa54ce3e9234d2e61fcb87c93cf407)
    * Docker image: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda11.8-cudnn8.9

* 备注：
    * 由于我们没有收集NPU和XPU的设备内存数据，因此表中相应位置的数据标记为N/A。

## 2. 快速开始
PaddleX 支持在本地使用命令行或 Python 体验文档图像预处理产线的效果。

在本地使用文档图像预处理产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。如果您希望选择性安装依赖，请参考安装教程中的相关说明。该产线对应的依赖分组为 `ocr`。

### 2.1 本地体验

#### 2.1.1 命令行方式体验
一行命令即可快速体验文档图像预处理产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/doc_test_rotated.jpg)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline doc_preprocessor \
        --input doc_test_rotated.jpg \
        --use_doc_orientation_classify True \
        --use_doc_unwarping True \
        --save_path ./output \
        --device gpu:0
```

<b>注：</b>PaddleX 支持多个模型托管平台，官方模型默认优先从 HuggingFace 下载。PaddleX 也支持通过环境变量 `PADDLE_PDX_MODEL_SOURCE` 设置优先使用的托管平台，目前支持 `huggingface`、`aistudio`、`bos`、`modelscope`，如优先使用 `bos`：`PADDLE_PDX_MODEL_SOURCE="bos"`；

相关的参数说明可以参考[2.1.2 Python脚本方式集成](#212-python脚本方式集成)中的参数说明。支持同时指定多个设备以进行并行推理，详情请参考 [产线并行推理](../../instructions/parallel_inference.md#指定多个推理设备)。

运行后，会将结果打印到终端上，结果如下：

<pre><code>{'res': {'input_path': 'doc_test_rotated.jpg', 'page_index': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 180}}
</code></pre>

运行结果参数说明可以参考[2.1.2 Python脚本方式集成](#212-python脚本方式集成)中的结果解释。

可视化结果保存在`save_path`下，可视化结果如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/doc_preprocessor/02.jpg"/>


#### 2.1.2 Python脚本方式集成

上述命令行是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="doc_preprocessor")
output = pipeline.predict(
    input="doc_test_rotated.jpg",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/")
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）通过 `create_pipeline()` 实例化 doc_preprocessor 产线对象：具体参数说明如下：

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
<td>产线推理设备。支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。支持同时指定多个设备以进行并行推理，详情请参考产线并行推理文档。</td>
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

（2）调用 doc_preprocessor 产线对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：

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
<td>待预测数据，支持多种输入类型，必填</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
<li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_doc_preprocessor_002.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
<li><b>List</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否使用文档方向分类模块</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>：<code>True</code> 或者 <code>False</code>；</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否使用文档扭曲矫正模块</td>
<td><code>bool|None</code></td>
<td>
<ul>
<li><b>bool</b>：<code>True</code> 或者 <code>False</code>；</li>
<li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>；</li>
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

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：

    - `input_path`: `(str)` 待预测图像的输入路径

    - `page_index`: `(Union[int, None])` 如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`

    - `model_settings`: `(Dict[str, bool])` 配置产线所需的模型参数

        - `use_doc_orientation_classify`: `(bool)` 控制是否启用文档方向分类模块
        - `use_doc_unwarping`: `(bool)` 控制是否启用文档扭曲矫正模块

    - `angle`: `(int)` 文档方向分类的预测结果。启用时取值为[0,90,180,270]；未启用时为-1

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_doc_preprocessor_res_img.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)

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
- `img` 属性返回的预测结果是一个字典类型的数据。其中，键为 `preprocessed_img`，对应的值是 `Image.Image` 对象：用于显示 doc_preprocessor 结果的可视化图像。

此外，您可以获取 doc_preprocessor 产线配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config doc_preprocessor --save_path ./my_path
```

若您获取了配置文件，即可对doc_preprocessor产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。示例如下：

例如，若您的配置文件保存在 `./my_path/doc_preprocessor.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/doc_preprocessor.yaml")
output = pipeline.predict(
    input="doc_test_rotated.jpg"
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
)
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```

<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改 doc_preprocessor 产线初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。同时，CLI 预测也支持传入配置文件，`--pipeline` 指定配置文件的路径即可。

## 3. 开发集成/部署
如果文档图像预处理产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将文档图像预处理产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

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
<p>获取图像文档图像预处理结果。</p>
<p><code>POST /document-preprocessing</code></p>
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
<td><code>integer</code> | <code>null</code></td>
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
<td>请参阅产线对象中 <code>predict</code> 方法的 <code>use_doc_unwarping</code> 参数相关说明。</td>
<td>否</td>
</tr>
<tr>
<td><code>visualize</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>是否返回可视化结果图以及处理过程中的中间图像等。
<ul style="margin: 0 0 0 1em; padding-left: 0em;">
<li>传入 <code>true</code>：返回图像。</li>
<li>传入 <code>false</code>：不返回图像。</li>
<li>若请求体中未提供该参数或传入 <code>null</code>：遵循产线配置文件<code>Serving.visualize</code> 的设置。</li>
</ul>
<br/>例如，在产线配置文件中添加如下字段：<br/>
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
<td><code>docPreprocessingResults</code></td>
<td><code>object</code></td>
<td>文档图像预处理结果。数组长度为1（对于图像输入）或实际处理的文档页数（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中实际处理的每一页的结果。</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>输入数据信息。</td>
</tr>
</tbody>
</table>
<p><code>docPreprocessingResults</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
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
<td><code>outputImage</code></td>
<td><code>string</code></td>
<td>经过预处理的图像。图像为PNG格式，使用Base64编码。</td>
</tr>
<tr>
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>产线对象的 <code>predict</code> 方法生成结果的 JSON 表示中 <code>res</code> 字段的简化版本，其中去除了 <code>input_path</code> 和 <code>page_index</code> 字段。</td>
</tr>
<tr>
<td><code>docPreprocessingImage</code></td>
<td><code>string</code> ｜ <code>null</code></td>
<td>可视化结果图。图像为JPEG格式，使用Base64编码。</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code> ｜ <code>null</code></td>
<td>输入图像。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
</details>
<details><summary>多语言调用服务示例</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/document-preprocessing"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["docPreprocessingResults"]):
    print(res["prunedResult"])
    output_img_path = f"out_{i}.png"
    with open(output_img_path, "wb") as f:
        f.write(base64.b64decode(res["outputImage"]))
    print(f"Output image saved at {output_img_path}")
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &lt;fstream&gt;
#include &lt;vector&gt;
#include &lt;string&gt;
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

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

    auto response = client.Post("/document-preprocessing", jsonObj.dump(), "application/json");

    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        if (!result.is_object() || !result["docPreprocessingResults"].is_array()) {
            std::cerr << "Unexpected response format." << std::endl;
            return 1;
        }

        for (size_t i = 0; i < result["docPreprocessingResults"].size(); ++i) {
            auto res = result["docPreprocessingResults"][i];

            if (res.contains("prunedResult")) {
                std::cout << "Preprocessed result: " << res["prunedResult"].dump() << std::endl;
            }

            if (res.contains("outputImage")) {
                std::string outputImgPath = "out_" + std::to_string(i) + ".png";
                std::string decodedImage = base64::from_base64(res["outputImage"].get<std::string>());

                std::ofstream outFile(outputImgPath, std::ios::binary);
                if (outFile.is_open()) {
                    outFile.write(decodedImage.c_str(), decodedImage.size());
                    outFile.close();
                    std::cout << "Saved image: " << outputImgPath << std::endl;
                } else {
                    std::cerr << "Failed to write image: " << outputImgPath << std::endl;
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

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/document-preprocessing";
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

                JsonNode docPreprocessingResults = result.get("docPreprocessingResults");
                for (int i = 0; i < docPreprocessingResults.size(); i++) {
                    JsonNode item = docPreprocessingResults.get(i);
                    int finalI = i;

                    JsonNode prunedResult = item.get("prunedResult");
                    System.out.println("Pruned Result [" + i + "]: " + prunedResult.toString());

                    String outputImgBase64 = item.get("outputImage").asText();
                    byte[] outputImgBytes = Base64.getDecoder().decode(outputImgBase64);
                    String outputImgPath = "out_" + finalI + ".png";
                    try (FileOutputStream fos = new FileOutputStream(outputImgPath)) {
                        fos.write(outputImgBytes);
                        System.out.println("Saved output image: " + outputImgPath);
                    }

                    JsonNode inputImageNode = item.get("inputImage");
                    if (inputImageNode != null && !inputImageNode.isNull()) {
                        String inputImageBase64 = inputImageNode.asText();
                        byte[] inputImageBytes = Base64.getDecoder().decode(inputImageBase64);
                        String inputImgPath = "inputImage_" + i + ".jpg";
                        try (FileOutputStream fos = new FileOutputStream(inputImgPath)) {
                            fos.write(inputImageBytes);
                            System.out.println("Saved input image to: " + inputImgPath);
                        }
                    }
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
)

func main() {
    API_URL := "http://localhost:8080/document-preprocessing"
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
        fmt.Printf("Error reading response body: %v\n", err)
        return
    }

    type DocPreprocessingResult struct {
        PrunedResult         map[string]interface{} `json:"prunedResult"`
        OutputImage          string                 `json:"outputImage"`
        DocPreprocessingImage *string               `json:"docPreprocessingImage"`
        InputImage           *string                `json:"inputImage"`
    }

    type Response struct {
        Result struct {
            DocPreprocessingResults []DocPreprocessingResult `json:"docPreprocessingResults"`
            DataInfo                interface{}              `json:"dataInfo"`
        } `json:"result"`
    }

    var respData Response
    if err := json.Unmarshal(body, &respData); err != nil {
        fmt.Printf("Error unmarshaling response: %v\n", err)
        return
    }

    for i, res := range respData.Result.DocPreprocessingResults {
        fmt.Printf("Result %d - prunedResult: %+v\n", i, res.PrunedResult)

        imgBytes, err := base64.StdEncoding.DecodeString(res.OutputImage)
        if err != nil {
            fmt.Printf("Error decoding outputImage at index %d: %v\n", i, err)
            continue
        }

        filename := fmt.Sprintf("out_%d.png", i)
        if err := os.WriteFile(filename, imgBytes, 0644); err != nil {
            fmt.Printf("Error saving image %s: %v\n", filename, err)
            continue
        }
        fmt.Printf("Saved output image to %s\n", filename)
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
    static readonly string API_URL = "http://localhost:8080/document-preprocessing";
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

        JArray docPreResults = (JArray)jsonResponse["result"]["docPreprocessingResults"];
        for (int i = 0; i < docPreResults.Count; i++)
        {
            var res = docPreResults[i];
            Console.WriteLine($"[{i}] prunedResult:\n{res["prunedResult"]}");

            string base64Image = res["outputImage"]?.ToString();
            if (!string.IsNullOrEmpty(base64Image))
            {
                string outputPath = $"out_{i}.png";
                byte[] imageBytes = Convert.FromBase64String(base64Image);
                File.WriteAllBytes(outputPath, imageBytes);
                Console.WriteLine($"Output image saved at {outputPath}");
            }
            else
            {
                Console.WriteLine($"outputImage at index {i} is null.");
            }
        }
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');
const path = require('path');

const API_URL = 'http://localhost:8080/document-preprocessing';
const imagePath = './demo.jpg';

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

const payload = {
  file: encodeImageToBase64(imagePath),
  fileType: 1
};

axios.post(API_URL, payload, {
  headers: {
    'Content-Type': 'application/json'
  },
  maxBodyLength: Infinity
})
.then((response) => {
  const results = response.data.result.docPreprocessingResults;

  results.forEach((res, index) => {
    console.log(`\n[${index}] prunedResult:`);
    console.log(res.prunedResult);

    const base64Image = res.outputImage;
    if (base64Image) {
      const outputImagePath = `out_${index}.png`;
      const imageBuffer = Buffer.from(base64Image, 'base64');
      fs.writeFileSync(outputImagePath, imageBuffer);
      console.log(`Output image saved at ${outputImagePath}`);
    } else {
      console.log(`outputImage at index ${index} is null.`);
    }
  });
})
.catch((error) => {
  console.error('API error:', error.message);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/document-preprocessing";
$image_path = "./demo.jpg";
$output_image_path = "./out_0.png";

$image_data = base64_encode(file_get_contents($image_path));
$payload = array("file" => $image_data, "fileType" => 1);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)["result"]["docPreprocessingResults"];

foreach ($result as $i => $item) {
    echo "[$i] prunedResult:\n";
    print_r($item["prunedResult"]);

    if (!empty($item["outputImage"])) {
        $output_image_path = "out_" . $i . ".png";
        file_put_contents($output_image_path, base64_decode($item["outputImage"]));
        echo "Output image saved at $output_image_path\n";
    } else {
        echo "No outputImage found for item $i\n";
    }
}
?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/on_device_deployment.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。


## 4. 二次开发
如果文档图像预处理产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升文档图像预处理产线的在您的场景中的识别效果。

### 4.1 模型微调

由于文档图像预处理产线包含若干模块，模型产线的效果如果不及预期，可能来自于其中任何一个模块。您可以对识别效果差的图片进行分析，进而确定是哪个模块存在问题，并参考以下表格中对应的微调教程链接进行模型微调。


<table>
<thead>
<tr>
<th>情形</th>
<th>微调模块</th>
<th>微调参考链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>整图旋转矫正不准</td>
<td>文档图像方向分类模块</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html">链接</a></td>
</tr>
<tr>
<td>图像扭曲矫正不准</td>
<td>文本图像矫正模块</td>
<td>暂不支持微调</td>
</tr>
</tbody>
</table>

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径填写至产线配置文件中的 `model_dir` 即可：

```yaml
......
  DocOrientationClassify:
    module_name: doc_text_orientation
    model_name: PP-LCNet_x1_0_doc_ori
    model_dir: ./output/best_model/inference  # 替换为微调后的文档图像方向分类模型权重路径
......
```
随后， 参考[2. 快速开始](#2-快速开始)中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device`参数</b>即可完成不同硬件之间的无缝切换。

例如，您使用昇腾 NPU 进行文档图像预处理产线的推理，使用的 CLI 命令为：

```bash
paddlex --pipeline doc_preprocessor \
        --input doc_test_rotated.jpg \
        --use_doc_orientation_classify True \
        --use_doc_unwarping True \
        --save_path ./output \
        --device npu:0
```
当然，您也可以在 Python 脚本中 `create_pipeline()` 时或者 `predict()` 时指定硬件设备。

若您想在更多种类的硬件上使用文档图像预处理产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
