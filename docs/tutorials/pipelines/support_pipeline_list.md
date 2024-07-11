# PaddleX 模型产线列表

PaddleX 提供了丰富的模型产线，您可以在产线对应的星河社区体验地址里体验，也可以在本地体验，本地体验方式请参考 [PaddleX 模型产线推理预测](./pipeline_inference.md)。

## 基础模型产线

| 产线名称 | 模型列表| 星河社区体验地址 | 产线介绍 |
| :---: | :---: | :---: | :---: |
| 通用图像分类 |[图像分类模型](../models/support_model_list.md#一图像分类)|[体验链接](https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent)|通用图像分类产线汇聚了多个不同量级的图像分类模型。图像分类是计算机视觉领域的基础任务，旨在实现对未知类别的图像进行分类。|
| 通用目标检测 |[目标检测模型](../models/support_model_list.md#二目标检测)|[体验链接](https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent)|通用目标检测产线汇聚了多个不同量级的目标检测模型。目标检测任务是计算机视觉领域的核心任务之一，旨在从图像或视频中找出所有感兴趣的目标（物体），并确定它们的类别和位置。|
| 通用语义分割 |[语义分割模型](../models/support_model_list.md#四语义分割)|[体验链接](https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent)|通用语义分割产线汇聚了多个不同量级的语义分割模型。被广泛用于街景分割、医学图像分割、道路分割等场景。旨在对不同类别的像素或区域进行区分。|
| 通用实例分割 |[实例分割模型](../models/support_model_list.md#https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/tutorials/models/support_model_list.md#三实例分割)|[体验链接](https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent)|通用实例分割产线汇聚了多个不同量级的实例分割模型。实例分割任务是计算机视觉领域的核心任务之一，旨在从图像或视频中找出所有感兴趣的目标（物体），并确定它们的类别和像素边界。|
| 通用OCR |[文本检测模型](../models/support_model_list.md#五文本检测)/[文本识别模型](../models/support_model_list.md#六文本识别)|[体验链接](https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent)|通用 OCR 产线用于解决文字识别任务，提取图片中的文字信息以文本形式输出，PP-OCRv4 是一个端到端 OCR 串联系统，可实现 cpu 上毫秒级的文本内容精准预测，在通用场景上达到开源SOTA。基于该项目，产学研界多方开发者已快速落地多个 OCR 应用，使用场景覆盖通用、制造、金融、交通等各个领域。|
| 通用表格识别 |[文本检测模型](../models/support_model_list.md#五文本检测)/[文本识别模型](../models/support_model_list.md#六文本识别)/版面分析模型/表格识别模型|[体验链接](https://aistudio.baidu.com/community/app/91661?source=appMineRecent)|通用表格识别产线是文本图像分析的子任务之一，旨在从图像中找到表格区域，并预测表格结构和文本内容，将表格恢复成 HTML 格式用于后续编辑或处理。|
| 时序预测 |[时序预测模型](../models/support_model_list.md#九时序预测)|[体验链接](https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent)|时序在每日的生活、工作中随处可见，比如 cpu 负载、上证指数、商场每天的人流量、商品每日的价格等都属于时间序列，总的来说时间序列就是按时间记录的有序数据，而时序预测就是运用历史的数据推测出事物的发展趋势。|
| 时序分类 |[时序分类模型](../models/support_model_list.md#八时序分类)|[体验链接](https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent)|时间序列分类是时间序列分析的一个重要应用场景，目的是利用标记好的训练数据，确定一个时间序列属于预先定义的哪一个类别。常见的应用场景包括：医疗健康监测，工业设备状态监测，交通状况分类等。|
| 时序异常检测 |[时序异常检测模型](../models/support_model_list.md#七时序异常检测)|[体验链接](https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent)|时序异常检测是目前时序数据分析成熟的应用之一，其旨在从正常的时间序列数据中识别出异常的事件或行为模式，在众多领域都发挥着重要作用：量化交易中，用于发现异常交易行为，规避潜在的金融风险；在网络安全领域，用于实时监测网络流量，及时发现并预防网络攻击行为的发生；在自动驾驶汽车领域，异常检测可以持续监测车载传感器数据，及时发现可能导致事故的异常情况；而在大型工业设备维护中，异常检测也能够帮助工程师提前发现设备故障苗头，从而采取预防性维护措施，降低设备停机时间和维修成本。|


## 特色模型产线

| 产线名称 | 模型列表| 星河社区体验地址 | 产线介绍 |
| :---: | :---: | :---: | :---: |
| 大模型半监督学习-图像分类 |[图像分类模型](../models/support_model_list.md#一图像分类)|[体验链接](https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent)|大模型半监督学习-图像分类产线是飞桨特色的图像分类训练产线，通过大小模型联合训练的方式，使用少量有标签数据和大量无标签数据提升模型的精度，大幅度减少人工迭代模型的成本、标注数据的成本。|
| 大模型半监督学习-目标检测 |[目标检测模型](../models/support_model_list.md#二目标检测)|[体验链接](https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent)|大模型半监督学习-目标检测产线是飞桨特色的目标检测训练产线，通过大小模型联合训练的方式，使用少量有标签数据和大量无标注数据提升模型的精度，大幅度减少人工迭代模型的成本、标注数据的成本。|
| 大模型半监督学习-OCR |[文本识别模型](../models/support_model_list.md#六文本识别)|[体验链接](https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent)|大模型半监督学习-OCR 产线是飞桨特色的 OCR 训练产线，由文本检测模型和文本识别模型串联完成。预测图片首先经过文本检测模型获取全部的文本行检测框并进行矫正，之后经文本识别模型得到 OCR 文本结果。在文本识别部分，通过大小模型联合训练的方式，使用少量有标签数据和大量无标签数据提升模型的精度，大幅度减少人工迭代模型的成本、标注数据的成本。|
| 通用场景信息抽取 |[文本检测模型](../models/support_model_list.md#五文本检测)/[文本识别模型](../models/support_model_list.md#六文本识别)|[体验链接](https://aistudio.baidu.com/community/app/91662/webUI?source=appMineRecent)|通用场景信息抽取产线（PP-ChatOCRv2-common）是飞桨特色的复杂文档智能分析解决方案，结合了 LLM 和 OCR 技术，将文心大模型将海量数据和知识相融合，准确率高且应用广泛。|
| 文档场景信息抽取 |[文本检测模型](../models/support_model_list.md#五文本检测)/[文本识别模型](../models/support_model_list.md#六文本识别)/版面分析模型/表格识别模型|[体验链接](https://aistudio.baidu.com/community/app/70303/webUI?source=appMineRecent)|文档场景信息抽取产线（PP-ChatOCRv2-doc）是飞桨特色的复杂文档智能分析解决方案，结合了 LLM 和 OCR 技术，一站式解决生僻字、特殊标点、多页 pdf、表格等常见的复杂文档信息抽取难点问题，结合文心大模型将海量数据和知识相融合，准确率高且应用广泛。|
| 多模型融合时序预测v2 |[时序预测模型](../models/support_model_list.md#九时序预测)|[体验链接](https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent)|多模型融合时序预测v2 产线的特点是针对不同任务场景，能够自适应的选择和集成模型，提升任务的精度。时序在每日的生活、工作中随处可见，时序预测的任务是指根据历史时间序列数据的模式和趋势，对未来的时间序列进行预测的任务。它在许多领域中都有应用，包括金融、天气预报、交通流量预测、销售预测、股票价格预测等。|
| 多模型融合时序异常检测v2 |[时序异常检测模型](../models/support_model_list.md#七时序异常检测)|[体验链接](https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent)|多模型融合时序异常检测产线的特点是针对不同任务场景，能够自适应的选择和集成模型，提升任务的精度。时序异常检测是目前时序数据分析成熟的应用之一，其旨在从正常的时间序列数据中识别出异常的事件或行为模式，在众多领域都发挥着重要作用：量化交易中，用于发现异常交易行为，规避潜在的金融风险；在网络安全领域，用于实时监测网络流量，及时发现并预防网络攻击行为的发生；在自动驾驶汽车领域，异常检测可以持续监测车载传感器数据，及时发现可能导致事故的异常情况；而在大型工业设备维护中，异常检测也能够帮助工程师提前发现设备故障苗头，从而采取预防性维护措施，降低设备停机时间和维修成本。|
