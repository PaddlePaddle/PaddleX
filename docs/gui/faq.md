

1. **Why is the training speed so slow?**

   PaddleX makes a calculation using your local hardware entirely and deep learning tasks have a high requirement for computing power. We have adapted the CPU hardware so that you can quickly experience development with PaddleX, but we strongly recommend using a GPU to improve the training speed and development experience.

   

2. **Can I deploy PaddleX on a server or cloud platform?**

   PaddleX GUI is a client that adapts to local stand-alone installation and cannot be directly deployed on the server. You can directly use a PaddleX API or the PaddlePaddle core framework for its deployment on the server. If you wish to use public computing power, it is strongly recommended that you try to use [EasyDL](https://ai.baidu.com/easydl/) or [AI Studio](https://aistudio.baidu.com/aistudio/index) in the PaddlePaddle product series for development.

   

3. **Does PaddleX support EasyData annotated data?**

   Yes, it does. PaddleX can read EasyData annotated data smoothly. However, the current version of PaddleX GUI does not support direct import of the EasyData data format. By referring to the related document, you can [convert datasets](https://paddlex.readthedocs.io/zh_CN/latest/appendix/how_to_convert_dataset.html) and import them to PaddleX GUI for subsequent development. In addition, we are working on the function that PaddleX GUI can directly import the EasyData data format.

   

4. **Why is the model pruning analysis so time-consuming?**

   The model pruning analysis process is to analyze the sensitivity information of convolutional layers of a model and then perform pruning in different proportions according to the impact of parameters on the model effects. This process needs to be repeated several times until the FLOPS meet the requirements. Finally, fine-tuned training is performed to obtain the final pruned model. Therefore, it is time-consuming. For the principle of model pruning, refer to the document [Pruning Principle Introduction](https://paddlepaddle.github.io/PaddleSlim/algo/algo.html#2-%E5%8D%B7%E7%A7%AF%E6%A0%B8%E5%89%AA%E8%A3%81%E5%8E%9F%E7%90%86)

   

5. **How to call backend codes?**

   The PaddleX team has compiled the related API document for your learning and use. For details, refer to the [PaddleX API Description Document](https://paddlex.readthedocs.io/zh_CN/latest/apis/index.html)
   
   
   
6. **How to use PaddleX in an offline environment?**

   PaddleX allows users to train models in a local offline environment, but if you want to use the pre-training models trained on a standard dataset that the PaddleX team has prepared for you, you need to download them in an online environment. You can refer to the complete [document](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/appendix/how_to_offline_run.md) on training models without any networking and see how to quickly you cadownload all pre-training models with one click.

   

7. **Do you have any industry application cases or implemented engineering instances?**

   Yes, we have. PaddleX offers a wealth of industry application cases and complete example projects. Refer to the [PaddleX Industry Casebook] (https://paddlex.readthedocs.io/zh_CN/develop/examples/index.html)

**If you have any questions or suggestions, do not hesitate to provide your feedback in the form of an issue or join PaddleX’s official QQ group (1045148026)**

![](./images/QR.jpg)
