Welcome to use PaddleX！
=======================================

PaddleX是基于飞桨核心框架、开发套件和工具组件的深度学习全流程开发工具。具备 **全流程打通** 、**融合产业实践** 、**易用易集成** 三大特点。

* Official Website: http://www.paddlepaddle.org.cn/paddle/paddlex
* GitHub: https://github.com/PaddlePaddle/PaddleX
* Official QQ Chat Group: 1045148026
* GitHub Issue: http://www.github.com/PaddlePaddle/PaddleX/issues

1. 注：本使用手册在打印为pdf后，可能会存在部分格式的兼容问题；
2. 注：本文档持续在http://paddlex.readthedocs.io/进行更新。


.. toctree::
   :maxdepth: 1
   :caption: 1. Know PaddleX Quickly

   quick_start.md
   install.md


.. toctree::
   :maxdepth: 1
   :caption: 2. Data Preparation

   data/annotation/index
   data/format/index

.. toctree::
   :maxdepth: 1
   :caption: 3. Model Training and Parameter Adjustment

   train/index
   train/prediction.md
   appendix/parameters.md
   train/model_export.md

.. toctree::
   :maxdepth: 1
   :caption: 4. Model Quantification and Pruning

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
   :caption: 8. 附录

   apis/index.rst
   appendix/model_zoo.md
   appendix/metrics.md
   appendix/interpret.md
   appendix/how_to_offline_run.md
   change_log.md
