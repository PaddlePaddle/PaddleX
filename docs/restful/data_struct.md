# 数据结构
本文档用于说明PaddleX restful模块自定义的数据结构

## Protobuf结构化数据
```
message Dataset {
    string id = 1;
    string name = 2;
    string desc = 3;
    // 'classification': 分类数据
    // 'segmentation': 分割数据
    // 'detection_voc': 检测数据（仅用于检测)
    // 'detection_coco': 检测数据（用于检测，分割，实例分割)
    string type = 4;
    string path = 5;
    string create_time = 6;
}

message Project {
    string id = 1;
    string name = 2;
    string desc = 3;
    // 'classification'
    // 'segmentation'
    // 'segmentation'
    // 'instance_segmentation'
    // 'remote_segmentation'
    string type = 4;
    string did = 5;
    string path = 6;
    string create_time = 7;
}

message Task {
    string id = 1;
    string name = 2;
    string desc = 3;
    string pid = 4;
    string path = 5;
    string create_time = 6;
    //裁剪任务id
    string parent_id = 7;
}

message PretrainedModel {
    string id = 1;
    string name = 2;
    string model = 3;
    string type = 4;
    // 所属项目id
    string pid = 5;
    string tid = 6;
    string create_time = 7;
    string path = 8;
}

message ExportedModel {
    string id = 1;
    string name = 2;
    string model = 3;
    string type = 4;
    // 所属项目id
    string pid = 5;
    string tid = 6;
    string create_time = 7;
    string path = 8;
    int32 exported_type = 9;
}

message Workspace {
    string version = 1;
    string path = 2;
    map<string, Dataset> datasets = 3;
    map<string, Project> projects = 4;
    map<string, Task> tasks = 5;
    int32 max_dataset_id = 6;
    int32 max_project_id = 7;
    int32 max_task_id = 8;
    string current_time = 9;

    int32 max_pretrained_model_id = 10;
    map<string, PretrainedModel> pretrained_models = 11;

    int32 max_exported_model_id = 12;
    map<string, ExportedModel> exported_models = 13;
}

```

## 状态枚举变量
### DatasetStatus(数据集状态变量)
```
DatasetStatus = Enum('DatasetStatus',  
('XEMPTY', #空数据集  
'XCHECKING', #正在验证数据集  
'XCHECKFAIL', #数据集验证失败  
'XCOPYING',  #正在导入数据集  
'XCOPYDONE', #数据集导入成功  
'XCOPYFAIL', #数据集导入失败  
'XSPLITED' #数据集已经切分  
),start=0)
```

### TaskStatus(任务状态变量)
```
TaskStatus = Enum('TaskStatus',
('XUNINIT',#任务还未初始化
'XINIT',#任务初始化
'XDOWNLOADING',#正在下载预训练模型
'XTRAINING',#任务正在运行中，改状态下任务不能被删除
'XTRAINDONE',#任务完成运行
'XEVALUATED',#任务评估完成
'XEXPORTING',#任务正在导出inference模型
'XEXPORTED',#任务已经导出模型
'XTRAINEXIT', #任务运行中止
'XDOWNLOADFAIL', #任务下载失败
'XTRAINFAIL', #任务运行失败
'XEVALUATING',#任务评估中
'XEVALUATEFAIL', #任务评估失败
'XEXPORTFAIL', #任务导出模型失败
'XPRUNEING', #裁剪分析任务运行中
'XPRUNETRAIN'#裁剪训练任务运行中
),start=0)
```
### ProjectType(项目类型)

```
ProjectType = Enum('ProjectType',
('classification',#分类
'detection',#检测
'segmentation',#分割
'instance_segmentation',#实例分割
'remote_segmentation'#摇杆分割
),start=0)
```

### DownloadStatus(下载状态变量)

```
DownloadStatus = Enum('DownloadStatus',
('XDDOWNLOADING',#下载中
'XDDOWNLOADFAIL',#下载失败
'XDDOWNLOADDONE',下载完成
'XDDECOMPRESSED'解压完成
),start=0)
```


### PruneStatus(裁剪状态变量)

```
PruneStatus = Enum('PruneStatus',
('XSPRUNESTART',#启动裁剪任务
'XSPRUNEING',#正在裁剪模型
'XSPRUNEDONE',#已完成裁剪任务
'XSPRUNEFAIL',#裁剪任务失败
'XSPRUNEEXIT',#裁剪任务已经停止
),start=0)
```

### PredictStatus(预测任务状态变量)

```
PredictStatus = Enum('PredictStatus',
('XPRESTART',#预测开始
'XPREDONE',#预测完成
'XPREFAIL'#预测失败
), start=0)
```
### PretrainedModelStatus(预训练模型状态变量)

```
PretrainedModelStatus = Enum('PretrainedModelStatus',
('XPINIT', #初始化
'XPSAVING', #正在保存
'XPSAVEFAIL',#保存失败
'XPSAVEDONE' #保存完成
),start=0)
```

### ExportedModelType(模型导出状态变量)
```
ExportedModelType = Enum('ExportedModelType',
('XQUANTMOBILE',#量化后的lite模型
'XPRUNEMOBILE', #裁剪后的lite模型
'XTRAINMOBILE',#lite模型
'XQUANTSERVER', #量化后的inference模型
'XPRUNESERVER', #裁剪后的inference模型
'XTRAINSERVER'#inference模型
),start=0)
```
