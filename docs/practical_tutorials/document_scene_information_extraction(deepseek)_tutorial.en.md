---
comments: true
---

# PaddleX 3.0 Document Scene Information Extraction v3 (PP-ChatOCRv3_doc) -- DeepSeek Edition

Document Scene Information Extraction v3 (PP-ChatOCRv3-doc) is a unique solution for document and image intelligent analysis provided by PaddlePaddle, combining Large Language Models (LLMs) and OCR technology to address complex document information extraction challenges such as layout analysis, rare characters, multi-page PDFs, tables, and seal recognition in one stop. By integrating large models, it fuses massive data and knowledge, achieving high accuracy and wide applicability. Recently, the PaddlePaddle development team upgraded the PP-ChatOCRv3 pipeline in PaddleX, a low-code development tool, enabling the invocation of large language models based on the standard OpenAI interface and enriching the capability of custom prompt engineering for text and image information extraction. This section introduces the method of combining this pipeline with the DeepSeek large model in practical scenarios through several document scene information extraction tasks.

<img src="https://github.com/user-attachments/assets/90cb740b-7741-4383-bc4c-663f9d042d02"/>

## 1. Environment Preparation

Before use, you need to complete the local installation of PaddleX by referring to the [PaddleX Local Installation Tutorial](../installation/installation.en.md). Then, prepare the API key for the large language model. PP-ChatOCRv3 supports invoking the large model inference service provided by [Baidu Cloud Qianfan Platform](https://console.bce.baidu.com/qianfan/ais/console/onlineService). You can refer to [Authentication and Authorization](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Um2wxbaps_en) to obtain the API key from the Qianfan platform. For more details on using PP-ChatOCRv3, refer to the [PP-ChatOCRv3 Documentation](../pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v3.en.md).

## 2. Quick Experience of Document Information Extraction Based on the DeepSeek Large Model

PaddleX provides simple Python APIs. After preparing the environment and downloading the [Test File 1](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png), you can quickly experience document information extraction based on the DeepSeek-V3 large model by replacing the `api_key` in the following code.

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

After executing the above code, you can obtain the following result:

```
{'chat_res': {'驾驶室准乘人数': '2'}}
```

The result shows that PP-ChatOCRv3 can extract text information from the image and pass the extracted text information to the DeepSeek-V3 large model for question understanding and information extraction, returning the required extraction result.

## 3. New Model Can Quickly Adapt to Multi-page PDF Files for Efficient Information Extraction

In practical application scenarios, besides a large number of image files, more document information extraction tasks involve multi-page PDF files. Since multi-page PDF files often contain a vast amount of text information, passing all this text information to a large language model at once not only increases the invocation cost but also reduces the accuracy of text information extraction. To address this issue, the PP-ChatOCRv3 pipeline integrates vector retrieval technology, which stores the text information from multi-page PDF files in the form of a vector database and retrieves the most relevant fragments through vector retrieval technology to pass them to the large language model, significantly reducing the invocation cost of the large language model and improving the accuracy of text information extraction. The Baidu Cloud Qianfan platform provides four vector models for establishing vector databases of text information. For the specific model support list and their functional characteristics, refer to the vector model section in the [API List](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu_en). Next, we will use the `embedding-v1` model to establish a vector database of text information and pass the most relevant fragments to the `DeepSeek-V3` large language model through vector retrieval technology, thereby efficiently extracting key information from multi-page PDF files.

First, download the [Test File 2](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract.pdf), then replace the `api_key` in the following code and execute it:

**Note**: Due to the large size of multi-page PDF files, the first execution requires a longer time for text information extraction and vector database establishment. The code saves the visual results of the model and the establishment results of the vector database locally, which can be loaded and used directly subsequently.

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
After executing the above code, the result obtained is as follows:

```
{'chat_res': {'甲方开户行': '日照银行股份有限公司开发区支行'}}
Visual Predict Time: 18.6519s
Vector Build Time: 6.1515s
Chat Time: 7.0352s
Total Time: 31.8385s
```

When we execute the above code again, the result obtained is as follows:

```
{'chat_res': {'甲方开户行': '日照银行股份有限公司开发区支行'}}
Visual Predict Time: 0.0161s
Vector Build Time: 0.0016s
Chat Time: 6.9516s
Total Time: 6.9693s
```

By comparing the results of the two executions, it can be observed that during the first execution, the PP-ChatOCRv3 Pipeline extracts all text information from multi-page PDF files and establishes a vector library, which takes a longer time. During subsequent executions, the PP-ChatOCRv3 Pipeline only needs to load and retrieve the vector library, significantly reducing the overall time consumption. The PP-ChatOCRv3 Pipeline, combined with vector retrieval technology, effectively reduces the number of calls to large language models when extracting ultra-long text, achieving faster text information extraction speed and more accurate key information location. This provides a more efficient solution for us in actual multi-page PDF file information extraction scenarios.

## 4. Exploring the Thinking Mode of Large Models in Text and Image Information Extraction

DeepSeek-R1 impresses with its exceptional text dialogue capabilities and in-depth problem-solving thinking abilities. When executing complex tasks or processing user instructions, in addition to normally completing dialogue tasks, the model can also demonstrate its thinking process during problem-solving. The PP-ChatOCRv3 Pipeline already supports the ability to adaptively return the output of thinking model results. For models that support returning the thinking process, PP-ChatOCRv3 can return the thinking process through an additional `reasoning_content` output field. This field is a list field containing the thinking results of the PP-ChatOCRv3 when calling the large language model multiple times. By observing these thinking results, we can gain insight into how the model gradually extracts the answer to the question from the given text information, and these thinking results can help us provide more improvement ideas for prompt optimization of the model. Next, we will take a specific legal document information extraction task as an example, using the `DeepSeek-R1` model as the large language model called in PP-ChatOCRv3 for key information extraction, and briefly explore the thinking process of the DeepSeek-R1 model.

First, you need to download [Test File 3](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/legislation.jpg), then replace the `api_key` in the following code and execute:

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

The result shows that the use of the `DeepSeek-R1` model not only helps us complete the relationship information extraction task for the question 'When was this regulation announced?' but also returns its thinking process when solving the information extraction problem in the `reasoning_content` field. For example, when thinking, the model carefully distinguishes between the publication date and the implementation date of the regulation and rechecks the returned results.

## 5. Supporting Custom Prompt Engineering to Expand the Functional Boundaries of Large Language Models

In document information extraction tasks, in addition to directly extracting key information from text information, we can also expand the functional boundaries of large language models through custom prompt engineering. For example, we can design new prompt rules to allow large language models to summarize these text information, thereby helping us quickly locate the key information we need from a large amount of text information, or allowing large language models to think and judge user questions based on the content in the text and give suggestions, etc. The PP-ChatOCRv3 Pipeline already supports custom prompt functionality, and the default prompts used by the Pipeline can be referred to in the Pipeline's [configuration file](../../paddlex/configs/pipelines/PP-ChatOCRv3-doc.yaml). We can refer to the prompt logic in the default configuration to customize and modify the prompts in the chat interface. Below is a brief introduction to the meaning of prompt parameters related to text content:

- `text_task_description`: Description of the dialogue task, for example, "请根据提供的文本内容回答用户的问题".
- `text_rules_str`: Detailed rules set by users, for example, "当返回时间是包含日期信息时，采用'YYYY-MM-DD'格式".
- `text_few_shot_demo_text_content`: Text content for few-shot demonstrations, for example, "当用户询问关于“该规定是何时公布的？”时，返回2005年2月4日".
- `text_few_shot_demo_key_value_list`: List of key-value pairs for few-shot demonstrations, for example, [{“该规定是何时公布的？”: "2005年2月4日"}, {“该规定是何时施行的？”: "2005年4月1日"}].

For general scenarios, we only need to modify the `text_task_description` and `text_rules_str` parameters. For example, if we want the large language model to answer users' questions based on the text content and provide quoted passages, we can set it up like this:

```
text_task_description="你现在的任务是根据提供的文本内容回答用户的问题，并返回你回答各个问题所引用的原文片段"

text_rules_str="返回的日期格式为“YYYY-MM-DD”"

```

Next, we will use a previously mentioned legal regulation test file with the `DeepSeek-R1` model for a practical demonstration. You need to replace the `api_key` in the following code and execute it:


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
    text_output_format='在返回结果时使用JSON格式，包含多个key-value对，key值为我指定的问题，value值为该问题对应的答案。如果认为OCR识别结果中，对于问题key，没有答案，则将value赋值为"未知"。对于问题结果，使用“答案：”标注，对于引用原文片段，使用“引用原文：”标注。请只输出json格式的结果，并做json格式校验后返回',
    text_rules_str="返回的日期格式为“YYYY-MM-DD”",
    chat_bot_config=chat_bot_config,
)
print(chat_result)
```

After executing the above code, you can obtain results as follows:

```
{'chat_res': {'reasoning_content': ['好的，我现在需要处理用户的问题，根据提供的OCR文本内容回答问题，并按照指定的JSON格式返回结果，同时包含答案和引用的原文片段。首先，我需要仔细阅读用户提供的OCR文本，理解其中的内容，然后针对每个关键词列表中的问题逐一查找答案。\n\n用户的关键词列表中有问题：“该规定是何时公布的？”。我需要从OCR文本中找到相关的日期信息。OCR文本的开头部分提到：“勘察设计注册工程师管理规定 （2005年2月4日中华人民共和国建设部令第137号公布自2005年4月1日起施行）”。这里有两个日期，一个是公布的日期2005年2月4日，另一个是施行的日期2005年4月1日。问题问的是“公布”的时间，所以正确的答案应该是2005年2月4日。\n\n接下来，我需要确认答案的正确性，并找到对应的原文引用。原文中明确写明了公布的日期，因此答案应该是这个日期，并且引用原文中的相关部分。然后，按照用户的要求，将答案和引用部分用JSON格式表示，其中日期格式必须为“YYYY-MM-DD”，所以需要将“2005年2月4日”转换为“2005-02-04”。\n\n同时，用户要求如果问题在文本中没有答案，则返回“未知”。但在这个案例中，答案存在，所以不需要考虑这种情况。最后，确保JSON格式正确，没有语法错误，并且只输出JSON内容，不做其他说明。\n'], '该规定是何时公布的？': {'答案': '2005-02-04', '引用原文': '勘察设计注册工程师管理规定 （2005年2月4日中华人民共和国建设部令第137号公布自2005年4月1日起施行）'}}}
```

From the results, we can see that by using custom prompt engineering, we have made the large language model not only give answers when answering users' questions but also quote passages, and convert the date format to "YYYY-MM-DD". At the same time, we have also obtained the reasoning process of the model through the use of the DeepSeek-R1 large model. The returned reasoning results can help us effectively adjust the prompt strategy and also help users formulate more specific and accurate questioning methods. This is very helpful for us to understand how large language models perform text information extraction tasks.
