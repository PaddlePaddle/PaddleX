---
comments: true
---

# PaddleX 3.0 文档场景信息抽取 v3（PP-ChatOCRv3_doc） -- DeepSeek 篇

文档场景信息抽取v3（PP-ChatOCRv3-doc）是飞桨特色的文档和图像智能分析解决方案，结合了 LLM 和 OCR 技术，一站式解决版面分析、生僻字、多页 pdf、表格、印章识别等常见的复杂文档信息抽取难点问题，结合大模型将海量数据和知识相融合，准确率高且应用广泛。近期，飞桨研发团队，对飞桨低代码开发工具PaddleX中文本图像智能产线PP-ChatOCRv3进行升级，一方面实现了基于标准OpenAI接口的大语言模型调用，另一方面针对文本图像信息抽取，丰富了自定义提示词工程的能力。本篇将通过几次文档场景信息抽取任务实验，来介绍该产线在实际场景中与 DeepSeek 大模型结合使用的方法。

<img src="https://github.com/user-attachments/assets/90cb740b-7741-4383-bc4c-663f9d042d02"/>

## 1. 环境准备

在使用之前，您首先需要参考 [PaddleX本地安装教程](../installation/installation.md) 完成 PaddleX 的本地安装。然后准备大语言模型的 api_key，PP-ChatOCRv3 支持调用 [百度云千帆平台](https://console.bce.baidu.com/qianfan/ais/console/onlineService) 提供的大模型推理服务，您可以参考[认证鉴权](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Um2wxbaps) 获取千帆平台的 api_key。更多的 PP-ChatOCRv3 使用细节，可以参考 [PP-ChatOCRv3 文档](../pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v3.md)。

## 2. 快速体验基于 DeepSeek 大模型的文档信息抽取

PaddleX 提供了简单的 Python API，在完成环境准备并下载 [测试文件1](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png) 后，更换以下代码中的 `api_key` 即可快速体验基于 DeepSeek-V3 大模型的文档信息抽取。

```python
from paddlex import create_pipeline

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "deepseek-v3",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

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
    chat_bot_config=chat_bot_config,
)
print(chat_result)
```

在执行完成上述代码后，可以得到的结果如下：

```
{'chat_res': {'驾驶室准乘人数': '2'}}
```

通过结果可以看出，PP-ChatOCRv3 能够从图像中提取出文本信息，并将提取到的文本信息通过 DeepSeek-V3 大模型进行问题理解和信息抽取，返回需要抽取的问题结果。

## 2. 新模型可快速适配多页 PDF 文件，高效抽取信息。

在实际的应用场景中，除了大量的图片文件外，更多的文档信息抽取任务会涉及到多页 PDF 文件的处理。由于多页 PDF 文件中往往包含大量的文本信息，而将大量的文本信息一次性传递给大语言模型，除了会增加大语言模型的调用成本外，还会降低大语言模型文本信息抽取的准确性。为了解决这一问题，PP-ChatOCRv3 产线中集成了向量检索技术，能够将多页 PDF 文件中的文本信息通过建立向量库的方式进行存储，并通过向量检索技术将文本信息检索到最相关的片段传递给大语言模型，从而大幅降低大语言模型的调用成本并提高文本信息抽取的准确性。在百度云千帆平台，提供了4个向量模型用于建立文本信息的向量库，具体的模型支持列表及其功能特点可参考 [API列表](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu) 中的向量模型部分。接下来我们将使用 `embedding-v1` 模型建立文本信息的向量库，并通过向量检索技术将最相关的片段传递给 `DeepSeek-V3` 大语言模型，从而实现高效抽取多页 PDF 文件中的关键信息。

首先，您需要下载 [测试文件2](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf)，然后更换以下代码中的 `api_key` 并执行：

**注**：由于多页 PDF 文件较大，首次执行时需要较长时间进行文本信息抽取和向量库的建立，代码中已将模型的视觉结果和向量库的建立结果保存到本地，后续可以直接加载使用。

```python
import os
import time
from paddlex import create_pipeline

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "deepseek-v3",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

retriever_config = {
    "module_name": "retriever",
    "model_name": "embedding-v1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "qianfan",
    "api_key": "api_key",  # your api_key
}

visual_predict_res_path = os.path.join("output", "contract.visual")
vector_res_path = os.path.join("output", "contract.vector")
start_time = time.time()
pipeline = create_pipeline(pipeline="PP-ChatOCRv3-doc", initial_predictor=False)

if not os.path.exists(visual_predict_res_path):
    visual_predict_res = pipeline.visual_predict(
        input="contract.pdf",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    visual_info_list = []
    for res in visual_predict_res:
        visual_info_list.append(res["visual_info"])

    pipeline.save_visual_info_list(visual_info_list, visual_predict_res_path)
else:
    visual_info_list = pipeline.load_visual_info_list(visual_predict_res_path)
visual_predict_time = time.time()

if not os.path.exists(vector_res_path):
    vector_info = pipeline.build_vector(
        visual_info_list,
        flag_save_bytes_vector=True,
        retriever_config=retriever_config,
    )
    pipeline.save_vector(
        vector_info, vector_res_path, retriever_config=retriever_config
    )
else:
    vector_info = pipeline.load_vector(
        vector_res_path, retriever_config=retriever_config
    )
vector_build_time = time.time()

chat_result = pipeline.chat(
    key_list=["甲方开户行"],
    visual_info=visual_info_list,
    vector_info=vector_info,
    chat_bot_config=chat_bot_config,
    retriever_config=retriever_config,
)
end_time = time.time()
print(chat_result)
print(f"Visual Predict Time: {round((visual_predict_time - start_time), 4)}s")
print(f"Vector Build Time: {round((vector_build_time - visual_predict_time), 4)}s")
print(f"Chat Time: {round((end_time - vector_build_time), 4)}s")
print(f"Total Time: {round((end_time - start_time), 4)}s")

```
在执行完成上述代码后，可以得到的结果如下：

```
{'chat_res': {'甲方开户行': '日照银行股份有限公司开发区支行'}}
Visual Predict Time: 18.6519s
Vector Build Time: 6.1515s
Chat Time: 7.0352s
Total Time: 31.8385s
```

当我们再次执行上述代码时，可以得到的结果如下：

```
{'chat_res': {'甲方开户行': '日照银行股份有限公司开发区支行'}}
Visual Predict Time: 0.0161s
Vector Build Time: 0.0016s
Chat Time: 6.9516s
Total Time: 6.9693s
```

通过对比两次执行结果可以发现，在首次执行时，PP-ChatOCRv3 产线会对多页 PDF 文件中的所有文本信息进行抽取和向量库的建立，耗时较长。而在后续执行时，PP-ChatOCRv3 产线仅需要对向量库进行加载和检索操作，大幅降低了整体的耗时。结合了向量检索技术的 PP-ChatOCRv3 产线有效的降低了对于超长文本进行抽取时大语言模型调用的次数，实现了更加快速的文本信息抽取速度和更加精准的关键信息定位，为我们在实际的多页 PDF 文件信息抽取场景中提供了更加高效的解决方案。

## 3. 探究大模型对文本及图像信息抽取的思考方式


DeepSeek-R1 凭借其卓越的文本对话能力和深入的问题思考能力，令人印象深刻。在执行复杂任务或处理用户指令时，该模型除了正常完成对话任务外，还能够展示其解决问题时的思考过程。PP-ChatOCRv3 特色产线已经支持了思考模型结果自适应返回的输出能力，对于支持返回思考过程的模型，PP-ChatOCRv3 能够将思考过程通过额外的 `reasoning_content` 输出字段进行返回。该字段为一个列表字段，包含了
 PP-ChatOCRv3 多次调用大语言模型时的思考结果，通过观察这些思考结果，我们可以深入了解模型是如何一步步地从给定的文本信息中提取出问题的答案，并且利用这些思考结果可以帮助我们对模型的提示词优化提供更多的改进思路。接下来，我们将以一个具体的法律法规文档信息抽取任务为例，使用 `DeepSeek-R1` 模型作为调用大语言模型在 PP-ChatOCRv3 中进行关键信息抽取，并简单探究 DeepSeek-R1 模型的思考过程。

首先，您需要下载 [测试文件3](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/legislation.jpg)，然后更换以下代码中的 `api_key` 并执行：

```python
from paddlex import create_pipeline

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "deepseek-r1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

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
    chat_bot_config=chat_bot_config,
)
print(chat_result)

```
在执行完成上述代码后，可以得到的结果如下：

```
{'chat_res': {'reasoning_content': ['好的，我现在需要处理用户的查询，从OCR识别结果中提取指定关键词对应的信息。首先，用户提供的OCR文本是关于《勘察设计注册工程师管理规定》的内容，关键词列表是“该规定是何时公布的？”。我的任务是找到这个问题的答案，并以JSON格式返回。\n\n首先，我需要仔细阅读OCR文本，寻找与“公布时间”相关的信息。通常，这类法规文件会在开头部分提到公布日期。快速浏览OCR内容，开头部分确实有提到：“（2005年2月4日中华人民共和国建设部令第137号公布自2005年4月1日起施行）”。这里有两个日期，一个是公布的日期2005年2月4日，另一个是施行日期2005年4月1日。用户的问题是关于公布的日期，所以正确的答案应该是2005年2月4日。\n\n接下来，我需要确认OCR文本中是否有其他可能的日期或相关信息。继续查看后面的章节，比如第一章总则和后续条款，主要涉及注册工程师的管理规定，没有提到其他公布日期。因此，可以确定答案就是开头提到的2005年2月4日。\n\n然后，我需要确保答案格式正确。用户要求返回JSON，且每个关键词对应一个值。如果找不到答案，则设为“未知”。这里显然找到了，所以JSON应该是{"该规定是何时公布的？": "2005年2月4日"}。\n\n最后，检查是否有其他可能的错误，比如OCR识别错误。例如，日期中的数字是否被正确识别。原文中的“2005年2月4日”看起来正确，没有明显的识别错误。因此，答案应该是准确的。\n'], '该规定是何时公布的？': '2005年2月4日'}}
```

通过结果可以看出，`DeepSeek-R1` 模型的使用，除了帮助我们完成了'该规定是何时公布的？'这个问题的关系信息抽取任务外，还在 `reasoning_content` 字段返回了其在解决信息抽取问题时所经历的思考过程，例如模型在思考时对该法规的公布时间和实施时间进行了认真区分，对返回的结果进行再次检查等。

## 4. 支持自定义提示词工程，拓展大语言模型的功能边界。

在文档信息抽取任务中，除了直接从文本信息中提取出关键信息外，我们还可以通过自定义提示词工程的方式，拓展大语言模型的功能边界。例如，我们可以设计全新的提示词规则，让大语言模型对这些文本信息进行归纳总结，从而帮助我们从大量的文本信息中快速定位到我们需要的关键信息，或者让大语言模型根据文本信息中的内容对用户的问题进行思考判断，给出建议等等。PP-ChatOCRv3 产线已经支持了提示词自定义功能，产线使用的默认提示词可以参考产线的 [配置文件](../../paddlex/configs/pipelines/PP-ChatOCRv3-doc.yaml)，我们可以参考默认配置中的提示词逻辑在chat接口中对提示词进行自定义修改，下面简单介绍其中关于文本内容的相关提示词参数含义：

- `text_task_description`：对话任务的描述，例如“请根据提供的文本内容回答用户的问题”。
- `text_rules_str`：用户设置的细节规则，例如“当返回时间是包含日期信息时，采用“YYYY-MM-DD”格式”。
- `text_few_shot_demo_text_content`：用于少样本演示的文本内容，例如“当用户询问关于“该规定是何时公布的？”时，返回2005年2月4日”。
- `text_few_shot_demo_key_value_list`：用于少样本演示的键值对列表，例如[{“该规定是何时公布的？”: "2005年2月4日"}, {“该规定是何时施行的？”: "2005年4月1日"}]。

对于一般场景，我们仅需要修改 `text_task_description` 和 `text_rules_str` 两个参数，例如我们希望大语言模型能够根据文本中的内容对用户的，则可以这样设置：

```

text_task_description="你现在的任务是根据提供的文本内容回答用户的问题，并返回你回答各个问题所引用的原文片段"

text_rules_str="返回的日期格式为“YYYY-MM-DD”"

```

接下来我们将使用使用前文使用的法律法规测试文件配合 `DeepSeek-R1` 模型进行一个实际的示例演示。您需要更换以下代码中的 `api_key` 并执行：

```python
from paddlex import create_pipeline

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "deepseek-r1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

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
    text_task_description="你现在的任务是根据提供的文本内容回答用户的问题，并返回你回答各个问题所引用的原文片段",
    text_rules_str="返回的日期格式为“YYYY-MM-DD”",
    text_output_format='在返回结果时使用JSON格式，包含多个key-value对，key值为我指定的问题，value值为该问题对应的答案。如果认为OCR识别结果中，对于问题key，没有答案，则将value赋值为"未知"。对于问题结果，使用“答案：”标注，对于引用原文片段，使用“引用原文：”标注。请只输出json格式的结果，并做json格式校验后返回',
    chat_bot_config=chat_bot_config,
)
print(chat_result)

```
在执行完成上述代码后，可以得到的结果如下：

```
{'chat_res': {'reasoning_content': ['好的，我现在需要处理用户的问题，根据提供的OCR文本内容回答问题，并按照指定的JSON格式返回结果，同时包含答案和引用的原文片段。首先，我需要仔细阅读用户提供的OCR文本，理解其中的内容，然后针对每个关键词列表中的问题逐一查找答案。\n\n用户的关键词列表中有问题：“该规定是何时公布的？”。我需要从OCR文本中找到相关的日期信息。OCR文本的开头部分提到：“勘察设计注册工程师管理规定 （2005年2月4日中华人民共和国建设部令第137号公布自2005年4月1日起施行）”。这里有两个日期，一个是公布的日期2005年2月4日，另一个是施行的日期2005年4月1日。问题问的是“公布”的时间，所以正确的答案应该是2005年2月4日。\n\n接下来，我需要确认答案的正确性，并找到对应的原文引用。原文中明确写明了公布的日期，因此答案应该是这个日期，并且引用原文中的相关部分。然后，按照用户的要求，将答案和引用部分用JSON格式表示，其中日期格式必须为“YYYY-MM-DD”，所以需要将“2005年2月4日”转换为“2005-02-04”。\n\n同时，用户要求如果问题在文本中没有答案，则返回“未知”。但在这个案例中，答案存在，所以不需要考虑这种情况。最后，确保JSON格式正确，没有语法错误，并且只输出JSON内容，不做其他说明。\n'], '该规定是何时公布的？': {'答案': '2005-02-04', '引用原文': '勘察设计注册工程师管理规定 （2005年2月4日中华人民共和国建设部令第137号公布自2005年4月1日起施行）'}}}
```

通过结果可以看出，我们通过自定义提示词工程的方式，让大语言模型在回答用户问题时，不仅给出了答案，还引用了原文片段，并且将日期格式转换为了“YYYY-MM-DD”的格式。同时，我们也通过使用 DeeepSeek-R1 大模型获取到了模型推理过程中的思考过程，通过返回的思考结果处理可以帮助我们有效的调整提示词策略，也可以帮助用户调整更加具体准确的提问方式，这对于我们理解大语言模型是如何进行文本信息抽取任务是非常有帮助的。
