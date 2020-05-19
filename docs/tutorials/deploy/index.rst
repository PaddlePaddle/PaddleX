多端部署
==============

本文档指引用户如何采用更高性能地方式来部署使用PaddleX训练的模型。使用本文档模型部署方式，会在模型运算过程中，对模型计算图进行优化，同时减少内存操作，相对比普通的paddlepaddle模型加载和预测方式，预测速度平均可提升1倍，具体各模型性能对比见服务端Python部署的预测性能对比章节。

.. toctree::
   :maxdepth: 2

   deploy_server/index.rst
   deploy_openvino.md
   deploy_lite.md
