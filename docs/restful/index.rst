PaddleX RESTful
=======================================

PaddleX RESTful是基于PaddleX开发的RESTful API。

目前对于开发者来说通过如下指令启动PaddleX RESTful服务，开启RESTful服务后可以通过官网下载Remote版本的GUI连接开启RESTful服务的服务端完成深度学习全流程开发。

同样您还可以根据RESTful API来开发自己的可视化界面。

**paddlex --start_restful --port [端口号] --workspace_dir [工作空间地址]**

PaddleX Remote GUI
---------------------------------------

PaddleX Remote GUI是针对PaddleX RESTful开发的可视化客服端。开发者可以通过客服端连接开启RESTful服务的服务端，通过GUI实现深度学习全流程：**数据处理** 、 **超参配置** 、 **模型训练及优化** 、 **模型发布**，无需开发一行代码，即可得到高性深度学习推理模型。

.. toctree::
   :maxdepth: 2
   :caption: 文档目录

   download.md


PaddleX RESTful API 二次开发
---------------------------------------

开发者可以使用PaddleX RESTful API 进行二次开发，按照自己的需求开发可视化界面，详细请参考以下文档

.. toctree::
   :maxdepth: 2
   :caption: 文档目录

   introduction.md
   quick_start.md
   restful_api.md
   data_struct.md
   tree.md
