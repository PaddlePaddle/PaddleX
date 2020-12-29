# RESTful目录结构  

介绍了PaddleX RESTful的整体目录结构

```
restful
|____system.py	// 处理/system 请求
|____workspace_pb2.py
|____dir.py
|____dataset	// 数据集相关
| |____datasetbase.py	// 数据集基类
| |______init__.py
| |____seg_dataset.py	// 分割数据集
| |____cls_dataset.py	// 分类数据集
| |____dataset.py	// 处理/dataset 请求
| |____operate.py	// 数据集基础操作函数
| |____utils.py		// 数据集基础工具函数
| |____ins_seg_dataset.py	// 示例分割数据集
| |____det_dataset.py	// 检测数据集
|______init__.py
|____model.py	// 处理/model 请求
|____project	// 项目相关
| |____task.py	// 处理/project/task 请求
| |______init__.py
| |____visualize.py	// 数据可视化
| |____operate.py	// 任务基础操作函数
| |____evaluate		// 模型评估
| | |____detection.py	// 检测模型评估
| | |____classification.py	// 分类模型评估
| | |______init__.py
| | |____draw_pred_result.py	// 评估与预测结果可视化
| | |____segmentation.py	// 分割模型评估
| |____train	//模型训练
| | |____detection.py	// 检测模型训练
| | |____params.py	// 模型参数
| | |____classification.py	// 分类模型训练
| | |______init__.py
| | |____params_v2.py	// 模型参数V2版本
| | |____segmentation.py	// 分割模型训练
| |____prune	// 模型剪裁
| | |____detection.py	// 检测模型剪裁
| | |____classification.py	// 分类模型剪裁
| | |______init__.py
| | |____segmentation.py	// 分割模型剪裁
| |____project.py	// 处理/project请求
|____utils.py	// 基础工具函数
|____app.py	// 创建flask app
|____front_demo	// 前端demo
|____workspace.py	// 处理/workspace请求
|____demo.py	// 处理/demo 请求
|____workspace.proto	// workspace 结构化信息
|____frontend_demo // 前端demo
| |____paddlex_restful_demo.html    //前端demo文件

```
