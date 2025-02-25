---
comments: true
---

# PaddleX 3.0 Document Scene Information Extraction v3 (PP-ChatOCRv3_doc) -- DeepSeek Edition

Document Scene Information Extraction v3 (PP-ChatOCRv3-doc) is a unique solution for document and image intelligent analysis provided by PaddlePaddle, combining Large Language Models (LLMs) and OCR technology to address complex document information extraction challenges such as layout analysis, rare characters, multi-page PDFs, tables, and seal recognition in one stop. By integrating large models, it fuses massive data and knowledge, achieving high accuracy and wide applicability. Recently, the PaddlePaddle development team upgraded the PP-ChatOCRv3 pipeline in PaddleX, a low-code development tool, enabling the invocation of large language models based on the standard OpenAI interface and enriching the capability of custom prompt engineering for text and image information extraction. This section introduces the method of combining this pipeline with the DeepSeek large model in practical scenarios through several document scene information extraction tasks.

<img src="https://github.com/user-attachments/assets/90cb740b-7741-4383-bc4c-663f9d042d02"/>

## 1. Environment Preparation

Before use, you need to complete the local installation of PaddleX by referring to the [PaddleX Local Installation Tutorial](../installation/installation.en.md). Then, prepare the API key for the large language model. PP-ChatOCRv3 supports invoking the large model inference service provided by [Baidu Cloud Qianfan Platform](https://console.bce.baidu.com/qianfan/ais/console/onlineService) or [AI Studio](https://ai.baidu.com/ai-doc/AISTUDIO/rm344erns). For the Qianfan Platform, You can refer to [Authentication and Authorization](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Um2wxbaps_en) to obtain the API key from the Qianfan platform.For the AI Studio platform, you can visit [Access Token](https://aistudio.baidu.com/account/accessToken) to obtain your personal accessToken as the API key. For more details on using PP-ChatOCRv3, refer to the [PP-ChatOCRv3 Documentation](../pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v3.en.md).

Note: PP-ChatOCRv3 utilizes both the Chat interface and the Embedding interface of the large model service, and they can be called independently. Currently, the AI Studio Platform does not support the Embedding interface, so for the Embedding interface, you need to use the free service provided by the Qianfan Platform.


## 2. Quick Experience of Document Information Extraction Based on the DeepSeek Large Model

PaddleX provides simple Python APIs. After preparing the environment and downloading the [Test File 1](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png), you can quickly experience document information extraction based on the DeepSeek-R1 large model by replacing the `api_key` in the following code.

```python
from paddlex import create_pipeline

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "deepseek-r1",
    "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    # "base_url": "https://qianfan.baidubce.com/v2", for qianfan platform
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
{'chat_res': {'reasoning_content': ['好，我现在需要处理用户的查询，从给定的HTML表格内容中提取关键词“驾驶室准乘人数”对应的信息。首先，我要仔细阅读用户提供的表格内容，找到相关的部分。\n\n用户给出的表格内容看起来是一个车辆合格证的信息，包含了很多字段。我需要先解析这个HTML表格的结构，找到对应的关键词所在的行。根据用户的关键词列表，我需要关注“驾驶室准乘人数”。\n\n在表格中搜索这个关键词，找到第32项：“32.驾驶室准乘人数（人）2”。这里的“2”应该就是对应的数值。但是需要确认这一行的结构是否有其他干扰信息。查看这一行的HTML结构：\n\n<tr><td>32.驾驶室准乘人数（人）2</td><td colspan="2"></td><td rowspan="6"></td><td colspan="3"></td></tr>\n\n可以看到，第一个<td>标签内包含了“32.驾驶室准乘人数（人）2”，后面的<td>标签都是空的或者跨行列的。因此，数值“2”直接跟在关键词后面，没有其他内容干扰。这时候需要判断这个“2”是否就是正确的值。\n\n接下来需要确认是否存在其他可能的位置提到“驾驶室准乘人数”。例如，是否有其他行可能包含该信息，或者是否有合并单元格导致信息出现在其他位置。但根据当前表格结构，第32项是唯一提到该关键词的地方，且后面紧跟着数值“2”。\n\n需要进一步验证是否正确提取。考虑到在表格中，其他类似字段如“额定载客（人）5”中数值直接跟在后面，因此这里的格式应该是一致的，所以“驾驶室准乘人数”对应的值应该是“2”。\n\n因此，最终结果应该是将“驾驶室准乘人数”的值设为“2”。如果表格中没有该信息，则设为“未知”，但在这个案例中，信息存在且明确，所以正确提取即可。\n'], '驾驶室准乘人数': '2'}}
```

The result shows that PP-ChatOCRv3 can extract text information from the image and pass the extracted text information to the DeepSeek-R1 large model for question understanding and information extraction, returning the required extraction result.

## 3. New Model Can Quickly Adapt to Multi-page PDF Files for Efficient Information Extraction

In practical application scenarios, besides a large number of image files, more document information extraction tasks involve multi-page PDF files. Since multi-page PDF files often contain a vast amount of text information, passing all this text information to a large language model at once not only increases the invocation cost but also reduces the accuracy of text information extraction. To address this issue, the PP-ChatOCRv3 pipeline integrates vector retrieval technology, which stores the text information from multi-page PDF files in the form of a vector database and retrieves the most relevant fragments through vector retrieval technology to pass them to the large language model, significantly reducing the invocation cost of the large language model and improving the accuracy of text information extraction. The Baidu Cloud Qianfan platform provides four vector models for establishing vector databases of text information. For the specific model support list and their functional characteristics, refer to the vector model section in the [API List](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu_en). Next, we will use the `embedding-v1` model to establish a vector database of text information and pass the most relevant fragments to the `DeepSeek-R1` large language model through vector retrieval technology, thereby efficiently extracting key information from multi-page PDF files.

First, download the [Test File 2](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/contract2.pdf), then replace the `api_key` in the following code and execute it:

**Note**: Due to the large size of multi-page PDF files, the free service provided by the Qianfan platform currently experiences a high volume of calls. As a result, there is a limit on the number of tokens per minute. If you test with your own PDF files that have too many pages, you may encounter a TPM (Tokens Per Minute) limit error. This limitation does not apply to other forms of deployed large model services or to Qianfan’s paid users.

```python
import os
import time
from paddlex import create_pipeline

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "deepseek-r1",
    "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
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
        input="contract2.pdf",
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
{'chat_res': {'reasoning_content': ["好的，我现在需要处理用户提供的表格内容，并从中提取关键词列表中指定的信息。关键词列表是['甲方开户行']。首先，我需要仔细分析表格的结构和内容，看看是否有与“甲方开户行”相关的信息。\n\n用户给出的表格内容是用HTML格式编写的，里面有多个表格行和列。首先，我会解析这个HTML表格的结构。表格的第一行显示“郑州数链 datal 品种规格”和“混煤”，这看起来是标题和产品名称。接下来的第二行涉及数量（吨）是7000，还有备注说明最终数量以买卖双方实际结算为准。第三行是“基准质量指标”，下面列出了各项质量指标，如低位热值、干燥基硫分、挥发分、灰分、全水和粒度，以及对应的数值。\n\n现在，我需要检查这些内容中是否存在“甲方开户行”的信息。关键词列表中的每个项都需要逐一核对。表格中的各个单元格主要涉及产品规格、数量、质量指标等，都是关于产品本身的描述，没有提到任何银行账户信息或开户行的相关内容。所有数据行和列都集中在产品属性和交易数量上，没有涉及合同中的甲方或乙方的银行信息。\n\n因此，在确认所有表格内容后，没有找到与“甲方开户行”相关的数据。根据用户的要求，如果某个关键词对应的信息不存在，应该将其值设为“未知”。因此，在生成的JSON结果中，“甲方开户行”对应的值应该是“未知”。\n\n接下来需要确保输出格式正确，使用JSON且不包含多余文字，同时进行格式校验。最终结果应该只包含指定的关键词及其对应的值，这里就是“甲方开户行”：“未知”。\n", "好的，我现在需要处理用户的请求，从给定的表格内容中提取关键词列表中的每一项对应的信息。用户的关键词列表是['甲方开户行']，而表格内容是一个HTML表格，里面包含了一些合同签订的信息。\n\n首先，我要仔细阅读用户提供的表格内容。表格中有两行，每行有两个单元格。第一行左边是“田方(买方)：郑州数链科技测试企业有限公司”，右边是“签订时间：2023年12月21日”。第二行左边是“乙方(卖方)：股份测试有限公司”，右边是“签订地点：郑州市郑东新区商务内环2号”。\n\n用户的关键词是“甲方开户行”，我需要检查表格中是否有这个信息。表格里提到的甲方是买方，即郑州数链科技测试企业有限公司，但表格中并没有提到甲方的开户行信息。签订时间和地点都有，但开户行没有出现。\n\n接下来，我需要确认是否可能在其他地方隐含了开户行信息，比如公司名称中是否包含银行或分行，但郑州数链科技测试企业有限公司的名字里没有银行相关的词汇，所以应该不存在隐含的信息。\n\n根据用户的指示，如果关键词对应的信息不存在，就将value设为“未知”。因此，对于“甲方开户行”，结果应该是“未知”。\n\n最后，我需要确保输出是严格的JSON格式，并且只包含指定的关键词，没有其他多余内容。同时，要校验JSON格式的正确性，避免语法错误。\n", '好的，我现在需要处理用户提供的表格内容，并从中提取指定的关键词列表中的信息。用户给出的关键词是“甲方开户行”，所以我需要仔细查看表格内容，找到对应的信息。\n\n首先，表格的结构是HTML格式的，里面包含了一个表格，有多个行和单元格。每个行中的<td>标签对应该单元格的内容。我需要逐个分析每个单元格的内容，看看是否有与“甲方开户行”相关的信息。\n\n表格内容中的行依次是：\n1. 甲方和乙方的公司名称。\n2. 甲方和乙方的法定代表人。\n3. 授权代表人，但这里都是空的。\n4. 银行账号，甲方和乙方各自的账号。\n5. 开户行，分别列出了甲方和乙方的开户行信息。\n\n在第五行中，甲方的开户行信息是“日照银行股份有限公司开发区支行”，而乙方的则是“招商银行股份有限公司郑州郑东新区支行”。用户需要的是甲方的开户行，因此正确的值应该是“日照银行股份有限公司开发区支行”。\n\n检查关键词列表中的每个项，确认是否存在对应的信息。在这个案例中，关键词是“甲方开户行”，表格中有明确的开户行信息对应甲方，所以应该提取该值。如果没有找到，才会设置为“未知”。这里显然存在该信息，所以最终的JSON结果应该包含正确的开户行名称。\n'], '甲方开户行': '日照银行股份有限公司开发区支行'}}
Visual Predict Time: 14.4605s
Vector Build Time: 4.3939s
Chat Time: 45.7185s
Total Time: 64.5728s
```

When we execute the above code again, the result obtained is as follows:

```
{'chat_res': {'reasoning_content': ["好的，我现在需要处理用户的查询，从给定的表格内容中提取关键词列表中的每一项对应的信息。用户的关键词列表是['甲方开户行']，而表格内容是一个HTML表格。\n\n首先，我需要仔细阅读表格内容，看看是否有任何与“甲方开户行”相关的信息。表格的结构看起来是关于煤炭的规格和质量指标，比如低位热值、硫分、挥发分等。表格中的内容包括品种规格、数量、基准质量指标等，但并没有提到任何银行或开户行的信息。\n\n接下来，我需要确认用户的需求是否有可能在表格中存在其他隐藏的信息，或者是否有其他字段可能间接关联到开户行。例如，是否有公司名称、财务信息或联系信息等部分。但根据提供的HTML代码，表格中的内容主要集中在产品规格和数量上，没有涉及财务或银行账户的信息。\n\n然后，我需要考虑用户可能的误解或表格可能的错误识别。用户可能认为表格中应该包含开户行信息，但实际提供的表格内容并不包含这部分。因此，在这种情况下，正确的做法是将“甲方开户行”对应的值设为“未知”。\n\n最后，确保输出符合用户要求的JSON格式，并且只包含指定的关键词，没有多余的内容。同时，需要验证JSON格式的正确性，避免语法错误。\n", '好的，我现在需要处理用户的任务，从给定的表格内容中提取指定关键词列表中的每一项对应的信息。用户提供的表格内容是用```符号包围的HTML代码，关键词列表是[\'甲方开户行\']。用户要求输出JSON格式，包含每个关键词对应的值，如果找不到则设为“未知”。\n\n首先，我需要解析表格内容。表格中有两行，每行有两个单元格。第一行第一个单元格是“田方(买方)：郑州数链科技测试企业有限公司”，第二个是“签订时间：2023年12月21日”。第二行第一个单元格是“乙方(卖方)：股份测试有限公司”，第二个是“签订地点：郑州市郑东新区商务内环2号”。\n\n接下来，用户的关键词是“甲方开户行”。我需要查找表格中是否有与此相关的信息。表格中提到的甲方是买方，即“郑州数链科技测试企业有限公司”，但表格里并没有提到甲方的开户行信息。签订时间和地点都有，但开户行没有出现。\n\n因此，根据用户的要求，如果找不到对应的值，应该设为“未知”。所以，JSON结果应该是{"甲方开户行": "未知"}。需要确保输出是合法的JSON格式，没有其他多余内容，并且经过校验正确无误。\n', '好的，我现在需要处理用户的请求，从提供的表格内容中提取关键词列表中的信息。关键词是“甲方开户行”，用户希望得到对应的值，如果没有的话就返回“未知”。首先，我需要仔细分析表格的结构和内容。\n\n用户给出的表格是用HTML格式写的，里面有多个行和单元格。看起来每一行有两个单元格，左边是甲方的信息，右边是乙方的。例如，第一行是甲方和乙方的公司名称，第二行是法定代表人的名字，第三行是授权代表人，第四行是银行账号，第五行是开户行。\n\n接下来，我需要找到“甲方开户行”对应的值。根据表格的结构，开户行信息在第五行，左边的单元格是甲方的开户行，右边是乙方的。看一下用户提供的表格内容，第五行的左边单元格是“开户行：日照银行股份有限公司开发区支行”，右边是“招商银行股份有限公司郑州郑东新区支行”。所以，甲方的开户行应该是左边的那个，也就是日照银行的那个。\n\n现在需要确认关键词列表中的每一个项是否都能在表格中找到对应的值。这里的关键词列表只有一个，即“甲方开户行”。根据上面的分析，甲方的开户行确实存在，所以对应的值应该是“日照银行股份有限公司开发区支行”。\n\n需要确保返回的JSON格式正确，并且没有多余的内容。用户特别强调只输出JSON，并且要做格式校验。因此，构造一个JSON对象，key是“甲方开户行”，value是提取到的开户行信息。如果没有找到的话，value就是“未知”，但这里已经找到了，所以没有问题。\n\n最后，检查是否有可能的错误。例如，是否有可能把乙方的开户行误认为是甲方的？需要再次确认表格中的每一行对应的左右单元格。第五行左边确实是甲方的信息，右边是乙方的，所以没问题。因此，最终的结果应该是正确的。\n'], '甲方开户行': '日照银行股份有限公司开发区支行'}}
Visual Predict Time: 0.0105s
Vector Build Time: 0.0007s
Chat Time: 48.3759s
Total Time: 48.387s
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
    "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
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
    "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
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
