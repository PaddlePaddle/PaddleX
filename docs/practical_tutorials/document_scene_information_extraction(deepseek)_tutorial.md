---
comments: true
---

# PaddleX 3.0 文档场景信息抽取 v3（PP-ChatOCRv3_doc） -- DeepSeek 篇

文档场景信息抽取v3（PP-ChatOCRv3-doc）是飞桨特色的文档和图像智能分析解决方案，结合了 LLM 和 OCR 技术，一站式解决版面分析、生僻字、多页 pdf、表格、印章识别等常见的复杂文档信息抽取难点问题，结合大模型将海量数据和知识相融合，准确率高且应用广泛。近期，飞桨研发团队，对飞桨低代码开发工具PaddleX中文本图像智能产线PP-ChatOCRv3进行升级，一方面实现了基于标准OpenAI接口的大语言模型调用，另一方面针对文本图像信息抽取，丰富了自定义提示词工程的能力。本篇将通过几次文档场景信息抽取任务实验，来介绍该产线在实际场景中与 DeepSeek 大模型结合使用的方法。

<img src="https://github.com/user-attachments/assets/90cb740b-7741-4383-bc4c-663f9d042d02"/>

## 1. 环境准备

在使用之前，您首先需要参考 [PaddleX本地安装教程](../installation/installation.md) 完成 PaddleX 的本地安装。然后准备大语言模型的 api_key，PP-ChatOCRv3 支持调用 [百度云千帆平台](https://console.bce.baidu.com/qianfan/ais/console/onlineService) 提供的大模型推理服务，您可以参考[认证鉴权](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Um2wxbaps) 获取千帆平台的 api_key。

## 2. 快速体验基于 DeepSeek 大模型的文档信息抽取

PaddleX 提供了简单的 Python API，在完成环境准备并下载 [测试文件1](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png) 后，更换以下代码中的 `api_key` 即可快速体验基于 DeepSeek-V3 大模型的文档信息抽取。

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-ChatOCRv3-doc",initial_predictor=False)

visual_predict_res = pipeline.visual_predict(
    input="vehicle_certificate-1.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False)

visual_info_list = []
for res in visual_predict_res:
    visual_info_list.append(res["visual_info"])

chat_result = pipeline.chat(
    key_list=["驾驶室准乘人数"],
    visual_info=visual_info_list,
    chat_bot_config={
      "module_name": "chat_bot",
      "model_name": "deepseek-v3",
      "base_url": "https://qianfan.baidubce.com/v2",
      "api_type": "qianfan",
      "api_key": "api_key" # your api_key
    }
)
print(chat_result)

```
在执行完成上述代码后，可以得到的结果如下：

```
{'chat_res': {'驾驶室准乘人数': '2'}}
```

通过结果可以看出，PP-ChatOCRv3 能够从图像中提取出文本信息，并将提取到的文本信息通过 DeepSeek-V3 大模型进行问题理解和信息抽取，返回需要抽取的问题结果。


## 2. 探究大模型对文本及图像信息抽取的思考方式


DeepSeek-R1 凭借其卓越的文本对话能力和深入的问题思考能力，令人印象深刻。在执行复杂任务或处理用户指令时，该模型除了正常完成对话任务外，还能够展示其解决问题时的思考过程。PP-ChatOCRv3 特色产线已经支持了思考模型结果自适应返回的输出能力，对于支持返回思考过程的模型，PP-ChatOCRv3 能够将思考过程通过额外的 `reasoning_content` 输出字段进行返回。该字段为一个列表字段，包含了
 PP-ChatOCRv3 多次调用大语言模型时的思考结果，通过观察这些思考结果，我们可以深入了解模型是如何一步步地从给定的文本信息中提取出问题的答案，并且利用这些思考结果可以帮助我们对模型的提示词优化提供更多的改进思路。接下来，我们将以一个具体的法律法规文档信息抽取任务为例，使用 `DeepSeek-R1` 模型作为调用大语言模型在 PP-ChatOCRv3 中进行关键信息抽取，并简单探究 DeepSeek-R1 模型的思考过程。

首先，您需要下载 [测试文件2](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/legislation.jpg)，然后更换以下代码中的 `api_key` 并执行：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-ChatOCRv3-doc",initial_predictor=False)

visual_predict_res = pipeline.visual_predict(
    input="legislation.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False)

visual_info_list = []
for res in visual_predict_res:
    visual_info_list.append(res["visual_info"])

chat_result = pipeline.chat(
    key_list=["该规定是何时公布的？"],
    visual_info=visual_info_list,
    chat_bot_config={
      "module_name": "chat_bot",
      "model_name": "deepseek-r1",
      "base_url": "https://qianfan.baidubce.com/v2",
      "api_type": "qianfan",
      "api_key": "api_key" # your api_key
    }
)
print(chat_result)

```
在执行完成上述代码后，可以得到的结果如下：

```
{'chat_res': {'reasoning_content': ['好的，我现在需要处理用户的OCR识别结果，并从中提取指定的关键词列表中的信息。首先，用户提供的OCR文本是关于《勘察设计注册工程师管理规定》的内容，关键词列表是“该规定是何时公布的？”。我的任务是从OCR文本中找到对应的答案，并以JSON格式返回结果。\n\n首先，我需要仔细阅读OCR文本，寻找与公布时间相关的信息。通常在法律法规的开头部分会提到公布日期和施行日期。查看OCR文本的第一段，确实有提到：“（2005年2月4日中华人民共和国建设部令第137号公布自2005年4月1日起施行）”。这里明确提到了公布的日期是2005年2月4日，施行日期是2005年4月1日。\n\n接下来，确认用户的问题是关于“公布”的时间，而不是施行时间。因此，正确的答案应该是2005年2月4日。需要确保没有混淆这两个日期。OCR文本中的结构清晰，公布和施行日期用“公布自”连接，所以分开处理这两个日期没有问题。\n\n然后，检查是否有其他部分可能提到公布时间，但根据上下文，通常这类信息只在开头出现一次，所以这里应该是唯一的来源。确认无误后，将答案格式化为JSON，确保键是用户提供的问题，值是找到的日期。如果找不到相关信息，则返回“未知”，但在这个案例中，信息明确存在。\n\n最后，验证JSON格式是否正确，确保没有语法错误，并且只输出JSON内容，没有其他多余文字。这样用户就能得到准确且格式正确的响应。\n'], '该规定是何时公布的？': '2005年2月4日'}}
```

通过结果可以看出，`DeepSeek-R1` 模型的使用，除了帮助我们完成了'该规定是何时公布的？'这个问题的关系信息抽取任务外，还在 `reasoning_content` 字段返回了其在解决信息抽取问题时所经历的思考过程，例如模型在思考时对该法规的公布时间和实施时间进行了认真区分，对返回的结果进行再次检查等。

## 3. 新模型可快速适配多页 PDF 文件，高效抽取信息。

在实际的应用场景中，除了大量的图片文件外，更多的文档信息抽取任务会涉及到多页 PDF 文件的处理。由于多页 PDF 文件中往往包含大量的文本信息，而将大量的文本信息一次性传递给大语言模型，除了会增加大语言模型的调用成本外，还会降低大语言模型文本信息抽取的准确性。为了解决这一问题，PP-ChatOCRv3 产线中集成了向量检索技术，能够将多页 PDF 文件中的文本信息通过建立向量库的方式进行存储，并通过向量检索技术将文本信息检索到最相关的片段传递给大语言模型，从而大幅降低大语言模型的调用成本并提高文本信息抽取的准确性。在百度云千帆平台，提供了4个向量模型用于建立文本信息的向量库，具体的模型支持列表及其功能特点可参考 [API列表](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu) 中的向量模型部分。接下来我们将使用 `embedding-v1` 模型建立文本信息的向量库，并通过向量检索技术将最相关的片段传递给 `DeepSeek-R1` 大语言模型，从而实现高效抽取多页 PDF 文件中的关键信息。

首先，您需要下载 [测试文件3](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf)，然后更换以下代码中的 `api_key` 并执行：

**注**：由于多页 PDF 文件较大，首次执行时需要较长时间进行文本信息抽取和向量库的建立，代码中已将模型的视觉结果和向量库的建立结果保存到本地，后续可以直接加载使用。

```python
import os
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-ChatOCRv3-doc",initial_predictor=False)

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
visual_predict_res_path = os.path.join(output_dir, "contract.visual")
vector_res_path = os.path.join(output_dir, "contract.vector")
if not os.path.exists(visual_predict_res_path):
    visual_predict_res = pipeline.visual_predict(
      input="contract.pdf",
      use_doc_orientation_classify=False,
      use_doc_unwarping=False)

    visual_info_list = []
    for res in visual_predict_res:
        visual_info_list.append(res["visual_info"])

    vector_info = pipeline.build_vector(visual_info_list, flag_save_bytes_vector=True,retriever_config={
        "module_name": "retriever",
        "model_name": "embedding-v1",
        "base_url": "https://qianfan.baidubce.com/v2",
        "api_type": "qianfan",
        "api_key": "api_key" # your api_key
    })

    pipeline.save_visual_info_list(visual_info_list, visual_predict_res_path)
    vector_info = pipeline.build_vector(visual_info_list)
    pipeline.save_vector(vector_info, vector_res_path)
else:
    visual_info_list = pipeline.load_visual_info_list(visual_predict_res_path)
    vector_info = pipeline.load_vector(vector_res_path)

chat_result = pipeline.chat(
    key_list=["甲方"],
    visual_info=visual_info_list,
    vector_info=vector_info,
    chat_bot_config={
      "module_name": "chat_bot",
      "model_name": "deepseek-r1",
      "base_url": "https://qianfan.baidubce.com/v2",
      "api_type": "openai",
      "api_key": "api_key" # your api_key
    },
    retriever_config={
      "module_name": "retriever",
      "model_name": "embedding-v1",
      "base_url": "https://qianfan.baidubce.com/v2",
      "api_type": "qianfan",
      "api_key": "api_key" # your api_key
    }
)
print(chat_result)

```
在执行完成上述代码后，可以得到的结果如下：

```
{'chat_res': {'reasoning_content': ['好的，我现在需要处理用户的请求，从提供的表格内容中提取关键词列表中的每一项对应的信息。用户给出的关键词是“甲方”，而表格内容是关于郑州数链的品种规格，包括数量、质量指标等。\n\n首先，我要仔细分析表格的结构和内容。表格分为几个部分，首先是数量（吨）的信息，然后是基准质量指标，包括热值、硫分、挥发分、灰分、全水和粒度等参数。所有的数据都是关于产品规格的，没有提到任何涉及合同双方的信息，比如甲方或乙方。\n\n接下来，我需要确认用户的关键词“甲方”是否在表格中存在对应的信息。表格中的内容主要涉及技术规格和数量，没有提到合同中的甲方、乙方或其他相关方。因此，根据用户的指示，如果在表格中没有找到对应的信息，应该将值设为“未知”。\n\n此外，用户要求使用JSON格式返回结果，并且只输出JSON内容，不做其他解释。需要确保JSON格式正确，没有语法错误。因此，最终的输出应该是一个包含“甲方”作为键，值为“未知”的JSON对象。\n\n最后，再次检查表格内容，确认没有遗漏任何可能隐含“甲方”的信息，比如隐藏的字段或注释，但表格的HTML结构中没有这些内容。因此，确定“甲方”的信息确实不存在于表格中，应返回“未知”。\n', "好的，我现在需要处理用户的查询，从给定的表格内容中提取关键词列表中指定的信息。用户的关键词列表是['甲方']，而表格内容是用HTML表格形式呈现的。首先，我需要仔细分析表格的结构和内容，确保准确提取所需的信息。\n\n首先，查看表格的结构。表格中有两行，每行有两列。第一行的第一列是“田方(买方)：郑州数链科技测试企业有限公司”，第二列是“签订时间：2023年12月21日”。第二行的第一列是“乙方(卖方)：股份测试有限公司”，第二列是“签订地点：郑州市郑东新区商务内环2号”。\n\n用户的关键词是“甲方”，我需要确定在表格中是否存在与“甲方”对应的信息。在中文合同中，通常“甲方”指的是合同的一方，可能是买方或卖方，具体取决于合同条款。在表格中，第一行提到的是“田方(买方)”，这里的“田方”可能是“甲方”的笔误或同音词，因为“田”和“甲”在字形上有些相似，容易混淆。而“乙方”则明确标注为卖方。\n\n接下来需要结合上下文和常见合同结构来判断。通常，在合同中，甲方和乙方分别代表不同的角色，如买方和卖方，或者委托方和受托方。在这个例子中，“田方”被标注为买方，而“乙方”是卖方。考虑到用户需要提取的是“甲方”，而表格中没有直接出现“甲方”这个词，但“田方”可能是“甲方”的误写。因此，需要确认是否存在这样的可能性。\n\n可能的解释是，用户在输入时可能将“甲方”误写为“田方”，或者表格中的内容存在排版或识别错误。例如，在OCR识别过程中，“甲”可能被错误识别为“田”。因此，在这种情况下，尽管表格中显示的是“田方”，但根据上下文和常见的合同结构，可以推断这实际上是“甲方”的错误，因此对应的公司名称应为“郑州数链科技测试企业有限公司”。\n\n此外，需要确保没有其他可能的解释。例如，是否存在其他行或列提到“甲方”，但根据提供的表格内容，只有“田方”和“乙方”出现，没有其他相关方。因此，可以合理推断“田方”即为“甲方”，并提取对应的公司名称作为值。\n\n最后，根据用户的要求，如果关键词不存在，则返回“未知”。但在这个案例中，尽管存在可能的误写，但结合上下文可以确定“田方”即为“甲方”，因此应提取对应的公司名称作为结果，而不是返回“未知”。\n"], '甲方': '郑州数链科技测试企业有限公司'}}
```

通过结果可以看出，`DeepSeek-R1` 模型在进行文本信息抽取任务时，仅对全文中的部分信息进行思考调用。有效的降低了对于超长文本进行抽取时大语言模型调用的次数成本和时间成本。结合了向量检索技术的 PP-ChatOCRv3 产线实现了更加快速的文本信息抽取速度和更加低廉的大语言模型调用成本，同时通过实现了对文本信息抽取任务中关键信息的精准定位，为我们在实际的多页 PDF 文件信息抽取场景中提供了更加高效的解决方案。

## 4. 支持自定义提示词工程，拓展大语言模型的功能边界。

在文档信息抽取任务中，除了直接从文本信息中提取出关键信息外，我们还可以通过自定义提示词工程的方式，拓展大语言模型的功能边界。例如，我们可以设计全新的提示词规则，让大语言模型对这些文本信息进行归纳总结，从而帮助我们从大量的文本信息中快速定位到我们需要的关键信息，或者让大语言模型根据文本信息中的内容对用户的问题进行思考判断，给出建议等等。PP-ChatOCRv3 产线已经支持了提示词自定义功能，产线使用的默认提示词可以参考产线的 [配置文件](../../paddlex/configs/pipelines/PP-ChatOCRv3-doc.yaml)，我们可以参考默认配置中的提示词逻辑在chat接口中对提示词进行自定义修改，下面简单介绍其中关于文本内容的相关提示词参数含义：

- `text_task_description`：对话任务的描述，例如“请根据提供的文本内容回答用户的问题”。
- `text_rules_str`：用户设置的细节规则，例如“当返回时间是包含日期信息时，采用“年-月-日的”格式”。
- `text_few_shot_demo_text_content`：用于少样本演示的文本内容，例如“当用户询问关于“该规定是何时公布的？”时，返回2005年2月4日”。
- `text_few_shot_demo_key_value_list`：用于少样本演示的键值对列表，例如[{“该规定是何时公布的？”: "2005年2月4日"}, {“该规定是何时施行的？”: "2005年4月1日"}]

对于一般场景，我们仅需要修改 `text_task_description` 和 `text_rules_str` 两个参数，例如我们希望大语言模型能够根据文本中的内容对用户的，则可以这样设置：

```
text_task_description="你现在的任务是根据提供的文本内容回答用户的问题，并给出你回答问题引用的原文片段"

text_rules_str="对于问题结果，使用“答案：”标注，对于引用原文片段，使用“引用原文：”标注。问题结果中的日期格式为“YYYY-MM-DD”"

```

接下来我们将使用使用前文使用的法律法规测试文件配合 `DeepSeek-R1` 模型进行一个实际的示例演示。您需要更换以下代码中的 `api_key` 并执行：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-ChatOCRv3-doc",initial_predictor=False)

visual_predict_res = pipeline.visual_predict(
    input="legislation.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False)

visual_info_list = []
for res in visual_predict_res:
    visual_info_list.append(res["visual_info"])

chat_result = pipeline.chat(
    key_list=["该规定是何时公布的？"],
    visual_info=visual_info_list,
    text_task_description="你现在的任务是根据提供的文本内容回答用户的问题，并给出你回答问题引用的原文片段",
    text_rules_str="对于问题结果，使用“答案：”标注，对于引用原文片段，使用“引用原文：”标注。问题结果中的日期格式为“YYYY-MM-DD”",
    chat_bot_config={
      "module_name": "chat_bot",
      "model_name": "deepseek-r1",
      "base_url": "https://qianfan.baidubce.com/v2",
      "api_type": "openai",
      "api_key": "api_key" # your api_key
    },
    retriever_config={
      "module_name": "retriever",
      "model_name": "embedding-v1",
      "base_url": "https://qianfan.baidubce.com/v2",
      "api_type": "qianfan",
      "api_key": "api_key" # your api_key
    }
)
print(chat_result)

```
在执行完成上述代码后，可以得到的结果如下：

```
{'chat_res': {'reasoning_content': ['好的，我现在需要处理用户的问题。用户提供的OCR文本是关于《勘察设计注册工程师管理规定》的内容，问题是要找出该规定是何时公布的。首先，我要仔细阅读OCR文本，寻找相关的日期信息。\n\n在OCR文本的开头部分，我看到这样的文字：“勘察设计注册工程师管理规定 （2005年2月4日中华人民共和国建设部令第137号公布自2005年4月1日起施行）”。这里有两个日期，一个是公布的日期，另一个是施行的日期。用户的问题是询问公布的日期，所以需要确认哪个对应公布。\n\n根据中文法律法规的常见格式，通常会在标题后的括号内注明公布日期和施行日期，格式为“（日期公布机关公布，自日期起施行）”。这里的结构是“2005年2月4日中华人民共和国建设部令第137号公布自2005年4月1日起施行”，所以前半部分“2005年2月4日”应该是公布的日期，后半部分“2005年4月1日”是施行日期。\n\n接下来，我需要确认用户的问题是否确实指向公布日期。用户的问题明确问的是“该规定是何时公布的？”，所以答案应该是2005年2月4日。需要将日期格式转换为“YYYY-MM-DD”，即“2005-02-04”。\n\n然后，检查是否有其他可能的日期信息。OCR文本的其他部分主要涉及注册工程师的管理规定，没有提到其他日期。因此，可以确定答案正确。\n\n最后，按照用户的要求，用JSON格式返回结果，问题作为key，答案作为value，并引用对应的原文片段。确保日期格式正确，没有其他多余内容。\n'], '该规定是何时公布的？': {'答案': '2005-02-04', '引用原文': '勘察设计注册工程师管理规定 （2005年2月4日中华人民共和国建设部令第137号公布自2005年4月1日起施行）'}}}
```

通过结果可以看出，我们通过自定义提示词工程的方式，让大语言模型在回答用户问题时，不仅给出了答案，还引用了原文片段，并且将日期格式转换为了“YYYY-MM-DD”的格式。同时，我们也通过使用 DeeepSeek-R1 大模型获取到了模型推理过程中的思考过程，通过返回的思考结果处理可以帮助我们有效的调整提示词策略，也可以帮助用户调整更加具体准确的提问方式，这对于我们理解大语言模型是如何进行文本信息抽取任务是非常有帮助的。
