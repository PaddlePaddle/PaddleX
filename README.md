<p align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/logo.png" width="735" height ="200" alt="PaddleX" align="middle" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202-red.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python-3.8%2C%203.9%2C%203.10-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Windows-orange.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Hardware-CPU%2C%20GPU%2C%20XPU%2C%20NPU%2C%20MLU%2C%20DCU-yellow.svg"></a>
</p>

<h4 align="center">
  <a href=#-特性>🌟 特性</a> | <a href=https://aistudio.baidu.com/pipeline/mine>🌐 在线体验</a>｜<a href=#️-快速开始>🚀 快速开始</a> | <a href=#-文档> 📖 文档</a> | <a href=./docs/support_list/pipelines_list.md> 🔥模型产线列表</a>

</h4>

[](#-特性)
<h5 align="center">
  <a href="README.md">🇨🇳 简体中文</a> | <a href="README_en.md">🇬🇧 English</a></a>
</h5>

## 🔍 简介

PaddleX 3.0 是基于飞桨框架构建的一站式全流程开发工具，它集成了众多**开箱即用的预训练模型**，可以实现模型从训练到推理的**全流程开发**，支持国内外**多款主流硬件**，助力AI 开发者进行产业实践。  

|                                                            **通用图像分类**                                                            |                                                            **图像多标签分类**                                                            |                                                            **通用目标检测**                                                            |                                                            **通用实例分割**                                                            |
|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/b302cd7e-e027-4ea6-86d0-8a4dd6d61f39" height="126px" width="180px"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/multilabel_cls.png" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/099e2b00-0bbe-4b20-9c5a-96b69e473bd2" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/09f683b4-27df-4c24-b8a7-84da20fdd182" height="126px" width="180px"> |
|                                                              **通用语义分割**                                                               |                                                            **图像异常检测**                                                            |                                                          **通用OCR**                                                          |                                                          **通用表格识别**                                                          |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/02637f8c-f248-415b-89ab-1276505f198c" height="126px" width="180px"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/image_anomaly_detection.png" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1ef48536-48d4-484b-a6fb-0d6631ba2386" height="126px" width="180px"> |  <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1e798e05-dee7-4b41-9cc4-6708b6014efa" height="126px" width="180px"> |
|                                                              **文本图像智能分析**                                                              |                                                            **时序预测**                                                            |                                                              **时序异常检测**                                                              |                                                         **时序分类**                                                         |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/e3d97f4e-ab46-411c-8155-494c61492b0a" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/6e897bf6-35fe-45e6-a040-e9a1a20cfdf2" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/c54c66cc-da4f-4631-877b-43b0fbb192a6" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0ce925b2-3776-4dde-8ce0-5156d5a2476e" height="126px" width="180px"> |

## 🌟 特性
  🎨 **模型丰富一键调用**：将覆盖文本图像智能分析、OCR、目标检测、时序预测等多个关键领域的 **200+ 飞桨模型**整合为 **19 条模型产线**，通过极简的 Python API 一键调用，快速体验模型效果。同时支持 **20+ 单功能模块**，方便开发者进行模型组合使用。

  🚀 **提高效率降低门槛**：实现基于统一命令和图形界面的模型**全流程开发**，打造大小模型结合、大模型半监督学习和多模型融合的[**8 条特色模型产线**](https://aistudio.baidu.com/intro/paddlex)，大幅度降低迭代模型的成本。  

  🌐 **多种场景灵活部署**：支持**高性能部署**、**服务化部署**和**端侧部署**等多种部署方式，确保不同应用场景下模型的高效运行和快速响应。

  🔧 **主流硬件高效支持**：支持英伟达 GPU、昆仑芯、昇腾和寒武纪等**多种主流硬件**的无缝切换，确保高效运行。

## 📣 近期更新

🔥🔥《PaddleX文档信息个性化抽取新升级》，PP-ChatOCRv3 创新性提供了基于数据融合技术的 OCR 模型二次开发功能，具备更强的模型微调能力。百万级高质量通用 OCR 文本识别数据，按特定比例自动融入垂类模型训练数据，破解产业垂类模型训练导致通用文本识别能力减弱难题。适用自动化办公、金融风控、医疗健康、教育出版、法律党政等产业实际场景。**10月10日（周四）19：00** 直播为您详细解读数据融合技术以及如何利用提示词工程实现更好的信息抽取效果。 [报名链接](https://www.wjx.top/vm/mFhGfwx.aspx?udsid=772552)

🔥🔥 **2024.9.30**，PaddleX 3.0 Beta1 开源版正式发布，提供 **200+ 模型** 通过极简的 Python API 一键调用；实现基于统一命令的模型全流程开发，并开源 **PP-ChatOCRv3** 特色模型产线基础能力；支持 **100+ 模型高性能推理和服务化部署**（持续迭代中），**7 类重点视觉模型端侧部署**；**70+ 模型开发全流程适配昇腾 910B**，**15+ 模型开发全流程适配昆仑芯和寒武纪**

🔥 **2024.6.27**，PaddleX 3.0 Beta 开源版正式发布，支持以低代码的方式在本地端使用多种主流硬件进行产线和模型开发。

🔥 **2024.3.25**，PaddleX 3.0 云端发布，支持在 AI Studio 星河社区 以零代码的方式【创建产线】使用。


 ## 📊 能力支持

PaddleX的各个产线均支持**在线体验**和本地**快速推理**，您可以快速体验各个产线的预训练模型效果，如果您对产线的预训练模型效果满意，可以直接对产线进行[高性能部署](./docs/pipeline_deploy/high_performance_deploy.md)/[服务化部署](./docs/pipeline_deploy/service_deploy.md)/[端侧部署](./docs/pipeline_deploy/lite_deploy.md)，如果不满意，您也可以使用产线的**二次开发**能力，提升效果。完整的产线开发流程请参考[PaddleX产线使用概览](./docs/pipeline_usage/pipeline_develop_guide.md)或各产线使用[教程](#-文档)。



此外，PaddleX 为开发者提供了基于[云端图形化开发界面](https://aistudio.baidu.com/pipeline/mine)的全流程开发工具, 详细请参考[教程《零门槛开发产业级AI模型》](https://aistudio.baidu.com/practical/introduce/546656605663301)


<table >
    <tr>
        <th>模型产线</th>
        <th>在线体验</th>
        <th>快速推理</th>
        <th>高性能部署</th>
        <th>服务化部署</th>
        <th>端侧部署</th>
        <th>二次开发</th>
        <th><a href = "https://aistudio.baidu.com/pipeline/mine">星河零代码产线</a></td>
    </tr>
    <tr>
        <td>通用OCR</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>文档场景信息抽取v3</td>
        <td><a href = "https://aistudio.baidu.com/community/app/182491/webUI?source=appCenter">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>表格识别</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91661?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>通用目标检测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>通用实例分割</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>通用图像分类</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>通用语义分割</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序预测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序异常检测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序分类</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
        <tr>
        <td>小目标检测</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
        <tr>
        <td>图像多标签分类</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>图像异常检测</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>公式识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>印章识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>通用图像识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>行人属性识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>车辆属性识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>人脸识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>

    
</table>

> ❗注：以上功能均基于 GPU/CPU 实现。PaddleX 还可在昆仑、昇腾、寒武纪和海光等主流硬件上进行快速推理和二次开发。下表详细列出了模型产线的支持情况，具体支持的模型列表请参阅[模型列表(MLU)](./docs/support_list/model_list_mlu.md)/[模型列表(NPU)](./docs/support_list/model_list_npu.md)/[模型列表(XPU)](./docs/support_list/model_list_xpu.md)/[模型列表(DCU)](./docs/support_list/model_list_dcu.md)。我们正在适配更多的模型，并在主流硬件上推动高性能和服务化部署的实施。

<details>
  <summary>👉 国产化硬件能力支持</summary>

<table>
  <tr>
    <th>产线名称</th>
    <th>昇腾 910B</th>
    <th>昆仑 R200/R300</th>
    <th>寒武纪 MLU370X8</th>
    <th>海光 Z100</th>
  </tr>
  <tr>
    <td>通用OCR</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>表格识别</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>通用目标检测</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>通用实例分割</td>
    <td>✅</td>
    <td>🚧</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>通用图像分类</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>通用语义分割</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>时序预测</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>时序异常检测</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>时序分类</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
</table>
</details>

## ⏭️ 快速开始

### 🛠️ 安装

> ❗安装 PaddleX 前请先确保您有基础的 **Python 运行环境**。

* **安装 PaddlePaddle**
```bash
# cpu
python -m pip install paddlepaddle==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# gpu，该命令仅适用于 CUDA 版本为 11.8 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，该命令仅适用于 CUDA 版本为 12.3 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```
> ❗ 更多飞桨 Wheel 版本请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation./docs/zh/install/pip/linux-pip.html)。


* **安装PaddleX**

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0b1-py3-none-any.whl
```
  
> ❗ 更多安装方式参考 [PaddleX 安装教程](./docs/installation/installation.md)

### 💻 命令行使用

一行命令即可快速体验产线效果，统一的命令行格式为：

```bash
paddlex --pipeline [产线名称] --input [输入图片] --device [运行设备]
```

只需指定三个参数：
* `pipeline`：产线名称
* `input`：待处理的输入文件（如图片）的本地路径或 URL
* `device`: 使用的 GPU 序号（例如`gpu:0`表示使用第 0 块 GPU），也可选择使用 CPU（`cpu`）


以通用 OCR 产线为例：
```bash
paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0
```
<details>
  <summary><b>👉 点击查看运行结果 </b></summary>

```bash
{'img_path': '/root/.paddlex/predict_input/general_ocr_002.png', 'dt_polys': [[[5, 12], [88, 10], [88, 29], [5, 31]], [[208, 14], [249, 14], [249, 22], [208, 22]], [[695, 15], [824, 15], [824, 60], [695, 60]], [[158, 27], [355, 23], [356, 70], [159, 73]], [[421, 25], [659, 19], [660, 59], [422, 64]], [[337, 104], [460, 102], [460, 127], [337, 129]], [[486, 103], [650, 100], [650, 125], [486, 128]], [[675, 98], [835, 94], [835, 119], [675, 124]], [[64, 114], [192, 110], [192, 131], [64, 134]], [[210, 108], [318, 106], [318, 128], [210, 130]], [[82, 140], [214, 138], [214, 163], [82, 165]], [[226, 136], [328, 136], [328, 161], [226, 161]], [[404, 134], [432, 134], [432, 161], [404, 161]], [[509, 131], [570, 131], [570, 158], [509, 158]], [[730, 138], [771, 138], [771, 154], [730, 154]], [[806, 136], [817, 136], [817, 146], [806, 146]], [[342, 175], [470, 173], [470, 197], [342, 199]], [[486, 173], [616, 171], [616, 196], [486, 198]], [[677, 169], [813, 166], [813, 191], [677, 194]], [[65, 181], [170, 177], [171, 202], [66, 205]], [[96, 208], [171, 205], [172, 230], [97, 232]], [[336, 220], [476, 215], [476, 237], [336, 242]], [[507, 217], [554, 217], [554, 236], [507, 236]], [[87, 229], [204, 227], [204, 251], [87, 254]], [[344, 240], [483, 236], [483, 258], [344, 262]], [[66, 252], [174, 249], [174, 271], [66, 273]], [[75, 279], [264, 272], [265, 297], [76, 303]], [[459, 297], [581, 295], [581, 320], [459, 322]], [[101, 314], [210, 311], [210, 337], [101, 339]], [[68, 344], [165, 340], [166, 365], [69, 368]], [[345, 350], [662, 346], [662, 368], [345, 371]], [[100, 459], [832, 444], [832, 465], [100, 480]]], 'dt_scores': [0.8183103704439653, 0.7609575621092027, 0.8662357274035412, 0.8619508290334809, 0.8495855993183273, 0.8676840017933314, 0.8807986687956436, 0.822308525056085, 0.8686617037621976, 0.8279022169854463, 0.952332847006758, 0.8742692553015098, 0.8477013022907575, 0.8528771493227294, 0.7622965906848765, 0.8492388224448705, 0.8344203789965632, 0.8078477124353284, 0.6300434587457232, 0.8359967356998494, 0.7618617265751318, 0.9481573079350023, 0.8712182945408912, 0.837416955846334, 0.8292475059403851, 0.7860382856406026, 0.7350527486717117, 0.8701022267947695, 0.87172526903969, 0.8779847108088126, 0.7020437651809734, 0.6611684983372949], 'rec_text': ['www.997', '151', 'PASS', '登机牌', 'BOARDING', '舱位 CLASS', '序号SERIALNO.', '座位号SEATNO', '航班 FLIGHT', '日期DATE', 'MU 2379', '03DEC', 'W', '035', 'F', '1', '始发地FROM', '登机口 GATE', '登机时间BDT', '目的地TO', '福州', 'TAIYUAN', 'G11', 'FUZHOU', '身份识别IDNO.', '姓名NAME', 'ZHANGQIWEI', '票号TKTNO.', '张祺伟', '票价FARE', 'ETKT7813699238489/1', '登机口于起飞前10分钟关闭GATESCLOSE1OMINUTESBEFOREDEPARTURETIME'], 'rec_score': [0.9617719054222107, 0.4199012815952301, 0.9652514457702637, 0.9978302121162415, 0.9853208661079407, 0.9445787072181702, 0.9714463949203491, 0.9841841459274292, 0.9564052224159241, 0.9959094524383545, 0.9386572241783142, 0.9825271368026733, 0.9356589317321777, 0.9985442161560059, 0.3965512812137604, 0.15236201882362366, 0.9976775050163269, 0.9547433257102966, 0.9974752068519592, 0.9646636843681335, 0.9907559156417847, 0.9895358681678772, 0.9374122023582458, 0.9909093379974365, 0.9796401262283325, 0.9899340271949768, 0.992210865020752, 0.9478569626808167, 0.9982215762138367, 0.9924325942993164, 0.9941263794898987, 0.96443772315979]}
......
```

可视化结果如下：

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/boardingpass.png)

</details>

其他产线的命令行使用，只需将 `pipeline` 参数调整为相应产线的名称。下面列出了每个产线对应的命令：

<details>
  <summary><b>👉 更多产线的命令行使用</b></summary>

| 产线名称           | 使用命令                                                                                                                                                                                    |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 通用图像分类       | `paddlex --pipeline image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0`                    |
| 通用目标检测       | `paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --device gpu:0`                            |
| 通用实例分割       | `paddlex --pipeline instance_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png --device gpu:0`                  |
| 通用语义分割       | `paddlex --pipeline semantic_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/application/semantic_segmentation/makassaridn-road_demo.png --device gpu:0` |
| 通用图像多标签分类 | `paddlex --pipeline multi_label_image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0`        |
| 小目标检测         | `paddlex --pipeline small_object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg --device gpu:0`                            |
| 图像异常检测       | `paddlex --pipeline anomaly_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png --device gpu:0 `                                              |
| 通用OCR            | `paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0`                                                      |
| 通用表格识别       | `paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --device gpu:0`                                      |
| 通用时序预测       | `paddlex --pipeline ts_fc --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv --device gpu:0`                                                                   |
| 通用时序异常检测   | `paddlex --pipeline ts_ad --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.cs --device gpu:0`                                                                    |
| 通用时序分类       | `paddlex --pipeline ts_cls --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv --device gpu:0`                                                                 |

</details>

### 📝 Python 脚本使用

几行代码即可完成产线的快速推理，统一的 Python 脚本格式如下：
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline=[产线名称])
output = pipeline.predict([输入图片名称])
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```
执行了如下几个步骤：

* `create_pipeline()` 实例化产线对象
* 传入图片并调用产线对象的 `predict` 方法进行推理预测
* 对预测结果进行处理

其他产线的 Python 脚本使用，只需将 `create_pipeline()` 方法的 `pipeline` 参数调整为相应产线的名称。下面列出了每个产线对应的参数名称及详细的使用解释：
<details>
  <summary><b>👉 更多产线的Python脚本使用</b></summary>

| 产线名称           | 对应参数                           | 详细说明                                                                                                                                                         |
|--------------------|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 文档场景信息抽取   | `PP-ChatOCRv3-doc`                 | [文档场景信息抽取v3产线Python脚本使用说明](./docs/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction.md#22-本地体验) |
| 通用图像分类       | `image_classification`             | [通用图像分类产线Python脚本使用说明](./docs/pipeline_usage/tutorials/cv_pipelines/image_classification.md#222-python脚本方式集成)                                |
| 通用目标检测       | `object_detection`                 | [通用目标检测产线Python脚本使用说明](./docs/pipeline_usage/tutorials/cv_pipelines/object_detection.md#222-python脚本方式集成)                                    |
| 通用实例分割       | `instance_segmentation`            | [通用实例分割产线Python脚本使用说明](./docs/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.md#222-python脚本方式集成)                               |
| 通用语义分割       | `semantic_segmentation`            | [通用语义分割产线Python脚本使用说明](./docs/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.md#222-python脚本方式集成)                               |
| 通用图像多标签分类 | `multi_label_image_classification` | [通用图像多标签分类产线Python脚本使用说明](./docs/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.md#22-python脚本方式集成)               |
| 小目标检测         | `small_object_detection`           | [小目标检测产线Python脚本使用说明](./docs/pipeline_usage/tutorials/cv_pipelines/small_object_detection.md#22-python脚本方式集成)                                 |
| 图像异常检测       | `anomaly_detection`                | [图像异常检测产线Python脚本使用说明](./docs/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.md#22-python脚本方式集成)                              |
| 通用OCR            | `OCR`                              | [通用OCR产线Python脚本使用说明](./docs/pipeline_usage/tutorials/ocr_pipelines/OCR.md#222-python脚本方式集成)                                                     |
| 通用表格识别       | `table_recognition`                | [通用表格识别产线Python脚本使用说明](./docs/pipeline_usage/tutorials/ocr_pipelines/table_recognition.md#22-python脚本方式集成)                                   |
| 通用时序预测       | `ts_fc`                            | [通用时序预测产线Python脚本使用说明](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.md#222-python脚本方式集成)                    |
| 通用时序异常检测   | `ts_ad`                            | [通用时序异常检测产线Python脚本使用说明](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.md#222-python脚本方式集成)          |
| 通用时序分类       | `ts_cls`                           | [通用时序分类产线Python脚本使用说明](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.md#222-python脚本方式集成)                 |

</details>


## 📖 文档
<details>
  <summary> <b> ⬇️ 安装 </b></summary>
  
  * [📦 PaddlePaddle 安装教程](./docs/installation/paddlepaddle_install.md)
  * [📦 PaddleX 安装教程](./docs/installation/installation.md) 


</details>

<details open>
<summary> <b> 🔥 产线使用 </b></summary>

* [📑 PaddleX 产线使用概览](./docs/pipeline_usage/pipeline_develop_guide.md)

* <details>
    <summary> <b> 📝 文本图像智能分析 </b></summary>

   * [📄 文档场景信息抽取v3产线使用教程](./docs/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction.md)
  </details>

* <details>
    <summary> <b> 🔍 OCR </b></summary>

    * [📜 通用 OCR 产线使用教程](./docs/pipeline_usage/tutorials/ocr_pipelines/OCR.md)
    * [📊 表格识别产线使用教程](./docs/pipeline_usage/tutorials/ocr_pipelines/table_recognition.md)
  </details>

* <details>
    <summary> <b> 🎥 计算机视觉 </b></summary>

   * [🖼️ 通用图像分类产线使用教程](./docs/pipeline_usage/tutorials/cv_pipelines/image_classification.md)
   * [🎯 通用目标检测产线使用教程](./docs/pipeline_usage/tutorials/cv_pipelines/object_detection.md)
   * [📋 通用实例分割产线使用教程](./docs/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.md)
   * [🗣️ 通用语义分割产线使用教程](./docs/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.md)
   * [🏷️ 图像多标签分类产线使用教程](./docs/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.md)
   * [🔍 小目标检测产线使用教程](./docs/pipeline_usage/tutorials/cv_pipelines/small_object_detection.md)
   * [🖼️ 图像异常检测产线使用教程](./docs/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.md)
  

* <details>
    <summary> <b> ⏱️ 时序分析</b> </summary>

   * [📈 通用时序预测产线使用教程](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.md)
   * [📉 通用时序异常检测产线使用教程](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.md)
   * [🕒 通用时序分类产线使用教程](./docs/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.md)
  </details>



* <details>
    <summary> <b>🔧 相关说明文件</b> </summary>

   * [🖥️ PaddleX 产线命令行使用说明](./docs/pipeline_usage/instructions/pipeline_CLI_usage.md)
   * [📝 PaddleX 产线 Python 脚本使用说明](./docs/pipeline_usage/instructions/pipeline_python_API.md)
  </details>
   
</details>

<details open>
<summary> <b> ⚙️ 单功能模块使用 </b></summary>

* <details>
  <summary> <b> 🔍 OCR </b></summary>

  * [📝 文本检测模块使用教程](./docs/module_usage/tutorials/ocr_modules/text_detection.md)
  * [🔖 印章文本检测模块使用教程](./docs/module_usage/tutorials/ocr_modules/seal_text_detection.md)
  * [🔠 文本识别模块使用教程](./docs/module_usage/tutorials/ocr_modules/text_recognition.md)
  * [🗺️ 版面区域检测模块使用教程](./docs/module_usage/tutorials/ocr_modules/layout_detection.md)
  * [📊 表格结构识别模块使用教程](./docs/module_usage/tutorials/ocr_modules/table_structure_recognition.md)
  * [📄 文档图像方向分类使用教程](./docs/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md)
  * [🔧 文本图像矫正模块使用教程](./docs/module_usage/tutorials/ocr_modules/text_image_unwarping.md)
  * [📐 公式识别模块使用教程](./docs/module_usage/tutorials/ocr_modules/formula_recognition.md)
  
  </details>

* <details>
  <summary> <b> 🖼️ 图像分类 </b></summary>

  * [📂 图像分类模块使用教程](./docs/module_usage/tutorials/cv_modules/image_classification.md)
  * [🏷️ 图像多标签分类模块使用教程](./docs/module_usage/tutorials/cv_modules/ml_classification.md)
  * [👤 行人属性识别模块使用教程](./docs/module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.md)
  * [🚗 车辆属性识别模块使用教程](./docs/module_usage/tutorials/cv_modules/vehicle_attribute_recognition.md)

  </details>

* <details>
  <summary> <b> 🏞️ 图像特征 </b></summary>

    * [🔗 通用图像特征模块使用教程](./docs/module_usage/tutorials/cv_modules/image_feature.md)
  </details>

* <details>
  <summary> <b> 🎯 目标检测 </b></summary>

  * [🎯 目标检测模块使用教程](./docs/module_usage/tutorials/cv_modules/object_detection.md)
  * [📏 小目标检测模块使用教程](./docs/module_usage/tutorials/cv_modules/small_object_detection.md)
  * [🧑‍🤝‍🧑 人脸检测模块使用教程](./docs/module_usage/tutorials/cv_modules/face_detection.md)
  * [🔍 主体检测模块使用教程](./docs/module_usage/tutorials/cv_modules/mainbody_detection.md)
  * [🚶 行人检测模块使用教程](./docs/module_usage/tutorials/cv_modules/human_detection.md)
  * [🚗 车辆检测模块使用教程](./docs/module_usage/tutorials/cv_modules/vehicle_detection.md)

  </details>

* <details>
  <summary> <b> 🖼️ 图像分割 </b></summary>

  * [🗺️ 语义分割模块使用教程](./docs/module_usage/tutorials/cv_modules/semantic_segmentation.md)
  * [🔍 实例分割模块使用教程](./docs/module_usage/tutorials/cv_modules/instance_segmentation.md)
  * [🚨 图像异常检测模块使用教程](./docs/module_usage/tutorials/cv_modules/anomaly_detection.md)
  </details>

* <details>
  <summary> <b> ⏱️ 时序分析 </b></summary>

  * [📈 时序预测模块使用教程](./docs/module_usage/tutorials/time_series_modules/time_series_forecasting.md)
  * [🚨 时序异常检测模块使用教程](./docs/module_usage/tutorials/time_series_modules/time_series_anomaly_detection.md)
  * [🕒 时序分类模块使用教程](./docs/module_usage/tutorials/time_series_modules/time_series_classification.md)
  </details>
    
* <details>
  <summary> <b> 📄 相关说明文件 </b></summary>

  * [📝 PaddleX 单模型 Python 脚本使用说明](./docs/module_usage/instructions/model_python_API.md)
  * [📝 PaddleX 通用模型配置文件参数说明](./docs/module_usage/instructions/config_parameters_common.md)
  * [📝 PaddleX 时序任务模型配置文件参数说明](./docs/module_usage/instructions/config_parameters_time_series.md)
  </details>

</details>

<details>
  <summary> <b> 🏗️ 模型产线部署 </b></summary>

  * [🚀 PaddleX 高性能部署指南](./docs/pipeline_deploy/high_performance_deploy.md)
  * [🖥️ PaddleX 服务化部署指南](./docs/pipeline_deploy/service_deploy.md)
  * [📱 PaddleX 端侧部署指南](./docs/pipeline_deploy/lite_deploy.md)

</details>
<details>
  <summary> <b> 🖥️ 多硬件使用 </b></summary>

  * [⚙️ 多硬件使用指南](./docs/other_devices_support/installation_other_devices.md)
  * [⚙️ DCU Paddle 安装教程](./docs/other_devices_support/paddlepaddle_install_DCU.md)
  * [⚙️ MLU Paddle 安装教程](./docs/other_devices_support/paddlepaddle_install_MLU.md)
  * [⚙️ NPU Paddle 安装教程](./docs//other_devices_support/paddlepaddle_install_NPU.md)
  * [⚙️ XPU Paddle 安装教程](./docs/other_devices_support/paddlepaddle_install_XPU.md)

</details>

<details>
  <summary> <b> 📝 教程&范例 </b></summary>

* [🖼️ 通用图像分类模型产线———垃圾分类教程](./docs/practical_tutorials/image_classification_garbage_tutorial.md)
* [🧩 通用实例分割模型产线———遥感图像实例分割教程](./docs/practical_tutorials/instance_segmentation_remote_sensing_tutorial.md)
* [👥 通用目标检测模型产线———行人跌倒检测教程](./docs/practical_tutorials/object_detection_fall_tutorial.md)
* [👗 通用目标检测模型产线———服装时尚元素检测教程](./docs/practical_tutorials/object_detection_fashion_pedia_tutorial.md)
* [🚗 通用 OCR 模型产线———车牌识别教程](./docs/practical_tutorials/ocr_det_license_tutorial.md)
* [✍️ 通用 OCR 模型产线———手写中文识别教程](./docs/practical_tutorials/ocr_rec_chinese_tutorial.md)
* [🗣️ 通用语义分割模型产线———车道线分割教程](./docs/practical_tutorials/semantic_segmentation_road_tutorial.md)
* [🛠️ 时序异常检测模型产线———设备异常检测应用教程](./docs/practical_tutorials/ts_anomaly_detection.md)
* [🎢 时序分类模型产线———心跳监测时序数据分类应用教程](./docs/practical_tutorials/ts_classification.md)
* [🔋 时序预测模型产线———用电量长期预测应用教程](./docs/practical_tutorials/ts_forecast.md)

  </details>

## 🤔 FAQ

关于我们项目的一些常见问题解答，请参考[FAQ](./docs/FAQ.md)。如果您的问题没有得到解答，请随时在 [Issues](https://github.com/PaddlePaddle/PaddleX/issues) 中提出
## 💬 Discussion

我们非常欢迎并鼓励社区成员在 [Discussions](https://github.com/PaddlePaddle/PaddleX/discussions) 板块中提出问题、分享想法和反馈。无论您是想要报告一个 bug、讨论一个功能请求、寻求帮助还是仅仅想要了解项目的最新动态，这里都是一个绝佳的平台。


## 📄 许可证书

本项目的发布受 [Apache 2.0 license](./LICENSE) 许可认证。

