# 快速开始

## 环境依赖  
- paddlepaddle-gpu/paddlepaddle  
- paddlex  
- pycocotools  


## 服务端启动PaddleX Restful服务
```
 paddlex --start_restful --port [端口号] --workspace_dir [工作空间目录]
```  

## 客服端请求服务端
```
import requests
url = "https://127.0.0.1:5000"
```
- url为实际服务端ip
- 所有的请求，通过ret.status_code是否为200，判断是否正确给Server执行
- 在status_code为200的前提下，如果ret.json()['status']为-1，则表明出错，出错信息在ret.json()['message']里面，如果执行成功， status是1

## 创建一个PaddleX的训练任务  
**下面介绍如何通过api创建一个PaddleX的训练任务，对于每个restful api的详细的输入参数与返回请参考[api 接口文档](./restful_api.md)，对于示例中用到自定的数据结构请参考[数据结构](./data_struct.md)**

### 流程
对于通过restful api创建一个PaddleX的训练任务的主要流程如下
- 1)：创建并导入数据集
- 2)：创建项目并绑定数据集
- 3)：获取参数并创建任务
- 4)：开始训练



### 数据集操作
#### 创建数据集
```
# dataset_type: 支持"detection"/"classification"/"segmentation"/"instance_segmentation"
params = {"name": "我的第一个数据集", "desc": "这里是数据集的描述文字", "dataset_type": "detection"}  
ret = requests.post(url+"/dataset", json=params)  
#获取数据集id
did = ret.json()['id']
```

#### 导入数据集

```
# 导入数据集
params = {'did' : did, 'path' : '/path/to/dataset'}
ret = requests.put(url+"/dataset", json=params)

# 数据集导入是一个异步操作，可以通过不断发送请求，获取导入的状态
params = {"did": did}
ret = requests.get(url+"/dataset", json=params)
#导入状态获取，其中DatasetStatus为自定义枚举标量，用来表示数据集的状态，具体定义请参考数据结构部分
import_status = DatasetStatus(ret.json['dataset_status'])
if import_status == DatasetStatus.XCOPYDONE:
    print("数据集导入成功")
elif import_status == DatasetStatus.XCOPYING:
    print("数据集正在导入中")
elif import_status == DatasetStatus.XCHECKFAIL:
    print("数据集格式校验未通过，请确定数据集格式是否正确)
```

#### 切分数据集
```
# 当数据集导入成功后，可以对数据集进行切分
# 切分数据集按照训练集、验证集、测试集为：6：2：2的形式切分
params = {'did' : did, 'val_split' : 0.2 , 'test_split' : 0.2}
ret = requests.put(url+"/dataset/split', json=params)

#切分数据集后需要获取具体切分情况
params = {'did': did}  
ret = requests.get(url+"/dataset/details', json=params)  
#获取切分情况
dataset_details = ret.json()
```

## 项目操作

### 创建项目
```
# project_type: 支持detection/classification/segmentation/instance_segmentation
params = {'name': '项目名称', 'desc': '项目描述文字', 'project_type' : 'detection'}
ret = requests.post(url+'/project', json=params)
# 获取项目id
pid = ret.json['pid']
```

### 绑定数据集
```
# 修改project中的did属性
# struct支持 project/dataset/task
params = {'struct': 'project', 'id': pid, 'attr_dict': {'did':did}}
ret = requests.put(url+'/workspace', json=params)
```

### 获取训练默认参数
```
params = {"pid", "P0001"}
ret = requests.get(url+"/project/task/parmas", json=params)
#获取默认训练参数
train_params = ret.json()['train']
```



## 任务操作

### 创建任务
```
#将训练参数json化
params_json = json.dumps(train_params)
#创建任务
params = {'pid': 'P0001', 'train':params_json}
ret = requests.post(url+'/task', json=params)
#获取任务id
tid = ret.json()['tid']
```

### 启动训练任务
```
params = {'tid' : tid}
ret = requests.post(url+'/project/task/train', json=params)

#训练任务是一个后台异步操作，可通过如下操作，获取任务训练的最新状态
params = {'tid' : 'T0001', 'type': 'train'}
ret = requests.get(url+'/project/task/metrics', json=params)
ret.json()获取返回值：
{'status': 1, 'train_log': 训练日志}
```
### 停止训练任务
通过如下操作，停止正在训练的任务
```
params = {'tid': tid, 'act': 'stop'}
ret = requests.put(url+'/project/task/train', json=params)
```
