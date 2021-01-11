
PaddleX RESTful
=======================================

PaddleX RESTful是基于PaddleX开发的RESTful API。

对于开发者来说通过如下指令启动PaddleX RESTful服务，开启RESTful服务后可以通过下载Remote版本的GUI或者是web demo连接开启RESTful服务的服务端完成深度学习全流程开发。

同样您还可以根据RESTful API来开发自己的可视化界面。

```
paddlex_restful --start_restful --port 8081 --workspace_dir D:\Workspace
```

**注意：请确保启动RESTful的端口未被防火墙限制**

支持RESTful版本的GUI
---------------------------------------

支持RESTful版本的GUI是针对PaddleX RESTful开发的可视化客户端。开发者可以通过客户端连接开启RESTful服务的服务端，通过GUI远程的实现深度学习全流程：**数据处理** 、 **超参配置** 、 **模型训练及优化** 、 **模型发布**，无需开发一行代码，即可得到高性深度学习推理模型。


支持RESTful版本Web Demo
---------------------------------------

支持RESTful版本Web Demo是针对PaddleX RESTful开发的网页版可视化客户端。开发者可以通过Web Demo连接开启RESTful服务的服务端，远程实现深度学习全流程：**数据处理** 、 **超参配置** 、 **模型训练及优化** 、 **模型发布**，无需开发一行代码，即可得到高性深度学习推理模型。

PaddleX RESTful API 二次开发
---------------------------------------

开发者可以使用PaddleX RESTful API 进行二次开发，按照自己的需求开发可视化界面



.. toctree::
   :maxdepth: 1
   :caption: 文档目录
   
   introduction.md
   restful.md
   quick_start.md
   restful_api.md
   data_struct.md
   tree.md





