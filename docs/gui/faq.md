## FAQ

1. **为什么训练速度这么慢？**

   PaddleX完全采用您本地的硬件进行计算，深度学习任务确实对算力要求较高，为了使您能快速体验应用PaddleX进行开发，我们适配了CPU硬件，但强烈建议您使用GPU以提升训练速度和开发体验。

   

2. **我可以在服务器或云平台上部署PaddleX么？**

   PaddleX GUI是一个适配本地单机安装的客户端，无法在服务器上直接进行部署，您可以直接使用PaddleX API，或采用飞桨核心框架进行服务器上的部署。如果您希望使用公有算力，强烈建议您尝试飞桨产品系列中的 [EasyDL](https://ai.baidu.com/easydl/) 或 [AI Studio](https://aistudio.baidu.com/aistudio/index)进行开发。

   

3. **PaddleX支持EasyData标注的数据吗？**

   支持，PaddleX可顺畅读取EasyData标注的数据。但当前版本的PaddleX GUI暂时无法支持直接导入EasyData数据格式，您可以参照文档，将[数据集进行转换](https://paddlex.readthedocs.io/zh_CN/latest/appendix/how_to_convert_dataset.html)再导入PaddleX GUI进行后续开发。
   同时，我们也在紧密开发PaddleX GUI可直接导入EasyData数据格式的功能。

   

4. **为什么模型裁剪分析耗时这么长？**

   模型裁剪分析过程是对模型各卷积层的敏感度信息进行分析，根据各参数对模型效果的影响进行不同比例的裁剪。此过程需要重复多次直至FLOPS满足要求，最后再进行精调训练获得最终裁剪后的模型，因此耗时较长。有关模型裁剪的原理，可参见文档[剪裁原理介绍](https://paddlepaddle.github.io/PaddleSlim/algo/algo.html#2-%E5%8D%B7%E7%A7%AF%E6%A0%B8%E5%89%AA%E8%A3%81%E5%8E%9F%E7%90%86)

   

5. **如何调用后端代码？**

   PaddleX 团队为您整理了相关的API接口文档，方便您学习和使用。具体请参见[PaddleX API说明文档](https://paddlex.readthedocs.io/zh_CN/latest/apis/index.html)



**如果您有任何问题或建议，欢迎以issue的形式，或加入PaddleX官方QQ群（1045148026）直接反馈您的问题和需求**

![](/images/QR.jpg)
