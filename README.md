

<div align=center>

<br/><img src="https://github.com/PaddlePaddle/PaddleX/blob/master/images/paddlexlogo.png" width = "450" height = "69" alt="PaddleX" align=center />

</br>

</div>



飞桨全流程开发客户端，集飞桨核心框架、模型库、工具及组件等深度学习开发全流程所需能力于一身，不仅为您提供一键安装的客户端，开源开放的技术内核更方便您根据实际生产需求进行直接调用或二次开发，是提升深度学习项目开发效率的最佳辅助工具。

PaddleX由PaddleX Client可视化前端和PaddleX Core后端技术内核两个部分组成。

PaddleX Client是提升项目开发效率的核心模块，开发者可快速完成深度学习模型全流程开发。而开源开放的后端技术内核PaddleX Core， 为开发者提供统一的任务API，在集成飞桨模型库、工具组件的基础上，提供更高层、简洁的开发方式。开发者可以根据实际业务需求，选择使用可视化前端，或直接调用PaddleX Core后端技术内核完成任务开发。

 PaddleX不仅打通了深度学习开发的全流程、提供可视化开发界面， 还保证了开发者可以直接灵活地使用底层技术模块。

 我们诚挚地邀请您前往 [官网](https://www.paddlepaddle.org.cn/paddle/paddlex)下载试用PaddleX可视化前端，并获得您宝贵的意见或开源项目贡献。



## 目录

* <a href="#1">**产品特性**</a>
* <a href="#2">**PaddleX Client可视化前端**</a>
  1. <a href="#a">下载客户端</a>
  2. <a href="#b">准备数据</a>
  3. <a href="#c">导入我的数据集</a>
  4. <a href="#d">创建项目</a>
  5. <a href="#e">项目开发</a>
* <a href="#3">**PaddleX Core后端技术内核**</a>
* <a href="#4">**FAQ**</a>



## <a name="1">产品特性</a>

1. **全流程打通**

将深度学习开发全流程打通，并提供可视化开发界面， 省去了对各环节API的熟悉过程及重复的代码开发，极大地提升了开发效率。

2. **开源技术内核**

集飞桨产业级CV工具集、迁移学习工具PaddleHub、可视化工具VisualDL、模型压缩工具PaddleSlim等于一身，并提供统一的任务API，可脱离前端单独下载使用，方便被集成与改造，为您的业务实践全程助力。

3. **本地一键安装**

无需额外下载依赖或驱动，提供适配Windows、Mac、Linux系统一键安装的客户端，技术内核可单独通过pip install安装。本地开发、保证数据安全，高度符合产业应用的实际需求。

4. **教程与服务**

从数据集准备到上线部署，为您提供业务开发全流程的文档说明及技术服务。开发者可以通过QQ群、微信群、GitHub社区等多种形式与飞桨团队及同业合作伙伴交流沟通。



## <a name="2">PaddleX Client可视化前端</a>

**<a name="a">第一步：下载客户端</a>**

您需要前往 [官网](https://www.paddlepaddle.org.cn/paddle/paddlex)填写基本信息后下载试用PaddleX可视化前端



**<a name="b">第二步：准备数据**</a>

在开始模型训练前，您需要根据不同的任务类型，将数据标注为相应的格式。目前PaddleX支持【图像分类】、【目标检测】、【语义分割】、【实例分割】四种任务类型。不同类型任务的数据处理方式可查看[数据标注方式]([https://github.com/jiangjiajun/PaddleSolution/tree/master/Docs/3_%E6%A0%87%E6%B3%A8%E8%87%AA%E5%B7%B1%E7%9A%84%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE](https://github.com/jiangjiajun/PaddleSolution/tree/master/Docs/3_标注自己的训练数据))。



**<a name="c">第三步：导入我的数据集</a>**

①数据标注完成后，您需要根据不同的任务，将数据和标注文件，按照客户端提示更名并保存到正确的文件中。

②在客户端新建数据集，选择与数据集匹配的任务类型，并选择数据集对应的路径，将数据集导入。

<img src="https://github.com/PaddlePaddle/PaddleX/blob/master/images/00%E6%95%B0%E6%8D%AE%E9%9B%86%E5%AF%BC%E5%85%A5%E8%AF%B4%E6%98%8E.png" width = "500" height = "350" alt="00数据集导入说明" align=center />

③选定导入数据集后，客户端会自动校验数据及标注文件是否合规，校验成功后，您可根据实际需求，将数据集按比例划分为训练集、验证集、测试集。

④您可在「数据分析」模块按规则预览您标注的数据集，双击单张图片可放大查看。

<img src="https://github.com/PaddlePaddle/PaddleX/blob/master/images/01%E6%95%B0%E6%8D%AE%E5%88%87%E5%88%86%E5%8F%8A%E9%A2%84%E8%A7%88.png" width = "500" height = "300" alt="01数据切分及预览" align=center />



**<a name="d">第四步：创建项目</a>**

① 在完成数据导入后，您可以点击「新建项目」创建一个项目。

② 您可根据实际任务需求选择项目的任务类型，需要注意项目所采用的数据集也带有任务类型属性，两者需要进行匹配。

<img src="https://github.com/PaddlePaddle/PaddleX/blob/master/images/02%E5%88%9B%E5%BB%BA%E9%A1%B9%E7%9B%AE.png" width = "500" height = "300" alt="02创建项目" align=center />



<a name="e">**第五步：项目开发**</a>

① **数据选择**：项目创建完成后，您需要选择已载入客户端并校验后的数据集，并点击下一步，进入参数配置页面。

<img src="https://github.com/PaddlePaddle/PaddleX/blob/master/images/03%E9%80%89%E6%8B%A9%E6%95%B0%E6%8D%AE%E9%9B%86.png" width = "400" height = "200" alt="03选择数据集" align=center />

② **参数配置**：主要分为**模型参数**、**训练参数**、**优化策略**三部分。您可根据实际需求选择模型结构及对应的训练参数、优化策略，使得任务效果最佳。

<img src="https://github.com/PaddlePaddle/PaddleX/blob/master/images/04%E5%8F%82%E6%95%B0%E9%85%8D%E7%BD%AE.png" width = "500" height = "500" alt="04参数配置" align=center />

参数配置完成后，点击启动训练，模型开始训练并进行效果评估。

③ **训练可视化**：

在训练过程中，您可通过VisualDL查看模型训练过程时的参数变化、日志详情，及当前最优的训练集和验证集训练指标。模型在训练过程中通过点击"终止训练"随时终止训练过程。

<img src="https://github.com/PaddlePaddle/PaddleX/blob/master/images/05%E8%AE%AD%E7%BB%83%E5%8F%AF%E8%A7%86%E5%8C%96.png" width = "500" height = "350" alt="05训练可视化" align=center />

<img src="https://github.com/PaddlePaddle/PaddleX/blob/master/images/06VisualDL.png" width = "500" height = "300" alt="06VisualDL" align=center />

模型训练结束后，点击”下一步“，进入模型评估页面。



④ **模型评估**

在模型评估页面，您可将训练后的模型应用在切分时留出的「验证数据集」以测试模型在验证集上的效果。评估方法包括混淆矩阵、精度、召回率等。在这个页面，您也可以直接查看模型在测试数据集上的预测效果。

根据评估结果，您可决定进入模型发布页面，或返回先前步骤调整参数配置重新进行训练。

<img src="https://github.com/PaddlePaddle/PaddleX/blob/master/images/07%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0.jpg" width = "500" height = "550" alt="07模型评估" align=center />

⑤**模型发布**

当模型效果满意后，您可根据实际的生产环境需求，选择将模型发布为需要的版本。

<img src="https://github.com/PaddlePaddle/PaddleX/blob/master/images/08%E6%A8%A1%E5%9E%8B%E5%8F%91%E5%B8%83.png" width = "450" height = "350" alt="08模型发布" align=center />



## <a name="3">PaddleX Core后端技术内核</a>





## <a name="4">FAQ</a>

1. **为什么我的数据集没办法切分？**

   如果您的数据集已经被一个或多个项目引用，数据集将无法切分，您可以额外新建一个数据集，并引用同一批数据，再选择不同的切分比例。

   

2. **任务和项目的区别是什么？**

   一个项目可以包含多条任务，一个项目拥有唯一的数据集，但采用不同的参数配置启动训练会创建多条任务，方便您对比采用不同参数配置的训练效果，并管理多个任务。

   

3. **为什么训练速度这么慢？**

   PaddleX完全采用您本地的硬件进行计算，深度学习任务确实对算力的要求比较高，为了使您能快速体验应用PaddleX进行开发，我们适配了CPU硬件，但强烈建议您使用GPU以提升训练速度和开发体验。

   

4. **我可以在服务器或云平台上部署PaddleX么？**

   PaddleX Client是一个适配本地单机安装的客户端，无法在服务器上直接进行部署，您可以直接使用PaddleX Core后端技术内核，或采用飞桨核心框架进行服务器上的部署。如果您希望使用公有算力，强烈建议您尝试飞桨产品系列中的 [EasyDL](https://ai.baidu.com/easydl/) 或 [AI Studio](https://aistudio.baidu.com/aistudio/index)进行开发。



**如果您有更多问题或建议，欢迎以issue的形式，加入PaddleX官方QQ群（1045148026）直接反馈您的问题和需求**

<div align=center>

<img src="https://github.com/PaddlePaddle/PaddleX/blob/master/images/09qq%E7%BE%A4%E4%BA%8C%E7%BB%B4%E7%A0%81.png" alt="09qq群二维码" align=center />

</div>



