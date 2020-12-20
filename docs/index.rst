欢迎使用PaddleX！
=======================================

PaddleX是基于飞桨核心框架、开发套件和工具组件的深度学习全流程开发工具。具备 **全流程打通** 、**融合产业实践** 、**易用易集成** 三大特点。

* 项目官网: http://www.paddlepaddle.org.cn/paddle/paddlex  
* 项目GitHub: https://github.com/PaddlePaddle/PaddleX
* 官方QQ用户群: 1045148026  
* GitHub Issue反馈: http://www.github.com/PaddlePaddle/PaddleX/issues

1. 注：本使用手册在打印为pdf后，可能会存在部分格式的兼容问题；
2. 注：本文档持续在http://paddlex.readthedocs.io/进行更新。


.. toctree::
   :maxdepth: 1
   :caption: 1. 快速了解PaddleX

   quick_start.md
   install.md


.. toctree::
   :maxdepth: 1
   :caption: 2. 数据准备

   data/annotation/index
   data/format/index

.. toctree::
   :maxdepth: 1
   :caption: 3. 模型训练与参数调整

   train/index
   train/prediction.md
   appendix/parameters.md
   train/model_export.md

.. toctree::
   :maxdepth: 1
   :caption: 4. 模型压缩优化

   slim/prune.md
   slim/quant.md

.. toctree::
   :maxdepth: 1
   :caption: 5. 模型多端安全部署

   deploy/export_model.md
   deploy/hub_serving.md
   deploy/server/index
   deploy/nvidia-jetson.md
   deploy/paddlelite/android.md
   deploy/raspberry/index
   deploy/openvino/index

.. toctree::
   :maxdepth: 1
   :caption: 6. 产业案例集

   examples/meter_reader.md
   examples/human_segmentation.md
   examples/remote_sensing.md
   examples/multi-channel_remote_sensing/README.md
   examples/change_detection.md
   examples/industrial_quality_inspection/README.md

.. toctree::
   :maxdepth: 1
   :caption: 7. 可视化客户端使用

   gui/introduce.md
   gui/download.md
   gui/how_to_use.md
   gui/FAQ.md


.. toctree::
   :maxdepth: 1
   :caption: 8. RESTful API使用

   
   restful/introduction.md
   restful/restful.md
   restful/quick_start.md
   restful/restful_api.md
   restful/data_struct.md
   restful/tree.md


.. toctree::
   :maxdepth: 1
   :caption: 9. 附录

   apis/index.rst
   appendix/model_zoo.md
   appendix/metrics.md
   appendix/interpret.md
   appendix/how_to_offline_run.md
   change_log.md
