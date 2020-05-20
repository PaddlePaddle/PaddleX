多端安全部署
==============

本文档指引用户如何采用更高性能地方式来部署使用PaddleX训练的模型。本文档模型部署采用Paddle Inference高性能部署方式，在模型运算过程中，对模型计算图进行优化，同时减少内存操作，具体各模型性能对比见服务端Python部署的预测性能对比章节。

同时结合产业实践开发者对模型知识产权的保护需求，提供了轻量级模型加密部署的方案，提升深度学习模型部署的安全性。

.. toctree::
   :maxdepth: 2

   deploy_server/index.rst
   deploy_openvino.md
   deploy_lite.md
