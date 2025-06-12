 [**English**](./VisualDL.en.md)




## VisualDL 介绍
VisualDL是飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、高维数据分布等。可帮助用户更清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型优化。

VisualDL提供丰富的可视化功能，支持标量、图结构、数据样本可视化、直方图、PR曲线及高维数据降维呈现等诸多功能，同时VisualDL提供可视化结果保存服务，通过VDL.service生成链接，保存并分享可视化结果。具体功能使用方式，请参见 [**VisualDL使用指南**](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.2/guides/03_VisualDL/visualdl_usage_cn.html)。项目正处于高速迭代中，敬请期待新组件的加入。

VisualDL支持浏览器种类：Chrome（81和83）、Safari 13、FireFox（77和78）、Edge（Chromium版）。

VisualDL原生支持python的使用， 通过在模型的Python配置中添加几行代码，便可为训练过程提供丰富的可视化支持。

## 目录

* [核心亮点](#核心亮点)

* [安装方式](#安装方式)

* [使用方式](#使用方式)

* [可视化功能概览](#可视化功能概览)

* [开源贡献](#开源贡献)

* [更多细节](#更多细节)

## 一、核心亮点

**简单易用**

API设计简洁直观，使用门槛低，只需几步操作即可实现模型结构的一键可视化，降低学习成本和使用难度，让您快速上手并应用到实际项目中。

**功能丰富**

涵盖标量、数据样本、图结构、直方图、PR曲线及数据降维等多种可视化功能，全面支持模型训练过程中的参数变化趋势分析和高维数据的深入理解，满足不同场景下的多样化需求。

**高兼容性**

全面支持Paddle、ONNX、Caffe等市面上主流的模型结构可视化，轻松应对不同框架构建的模型，为您提供广泛的可视化分析支持，确保不同技术背景的用户都能顺利使用。

**全面支持**

与飞桨服务平台及工具组件全面打通，让您在模型开发、训练、部署等各个环节享受顺畅流程和强大支持。



## 二、安装方式

推荐使用**pip安装**，简单快捷，只需在命令行中输入以下命令：

```shell
python -m pip install visualdl -i https://mirror.baidu.com/pypi/simple
```
如果您需要从源码安装，可以使用以下命令克隆仓库并安装：

```
git clone https://github.com/PaddlePaddle/VisualDL.git
cd VisualDL

python setup.py bdist_wheel
pip install --upgrade dist/visualdl-*.whl
```
**需要注意：** 官方自2020年1月1日起不再维护Python2，为了保障代码可用性，VisualDL现仅支持Python3

## 三、使用方式

VisualDL将训练过程中的数据、参数等信息储存至日志文件中后，启动面板即可查看可视化结果。

### 1. 记录日志

VisualDL的后端提供了Python SDK，可通过LogWriter定制一个日志记录器，接口如下：

```python
class LogWriter(logdir=None,
                comment='',
                max_queue=10,
                flush_secs=120,
                filename_suffix='',
                write_to_disk=True,
                display_name='',
                **kwargs)
```

**接口参数**

| 参数            | 格式    | 含义                                                         |
| --------------- | ------- | ------------------------------------------------------------ |
| logdir          | string  | 日志文件所在的路径，VisualDL将在此路径下建立日志文件并进行记录，如果不填则默认为`runs/${CURRENT_TIME}` |
| comment         | string  | 为日志文件夹名添加后缀，如果制定了logdir则此项无效           |
| max_queue       | int     | 日志记录消息队列的最大容量，默认值为10，达到此容量则立即写入到日志文件，如果设置为0则不缓存   |
| flush_secs      | int     | 日志记录消息队列的最大缓存时间，默认值为120，达到此时间则立即写入到日志文件，如果设置为0则不缓存 |
| filename_suffix | string  | 为默认的日志文件名添加后缀                                   |
| write_to_disk   | boolean | 是否写入到磁盘                                               |
| display_name    | string  | 在面板中`选择数据流`位置显示此参数，如不指定则默认显示日志所在路径（当日志所在路径过长或想隐藏日志所在路径时可指定此参数） |

**示例（设置日志文件并记录标量数据）：**

```python
from visualdl import LogWriter

# 在`./log/scalar_test/train`路径下建立日志文件
with LogWriter(logdir="./log/scalar_test/train") as writer:
    # 使用scalar组件记录一个标量数据
    writer.add_scalar(tag="acc", step=1, value=0.5678)
    writer.add_scalar(tag="acc", step=2, value=0.6878)
    writer.add_scalar(tag="acc", step=3, value=0.9878)
```

### 2. 启动 VisualDL 面板

在上一个示例中，日志已记录三组标量数据。现在，您可以启动 VisualDL 面板来查看日志的可视化结果。启动方式有两种：命令行启动和 Python 脚本启动。

#### a. 在命令行启动

使用命令行启动VisualDL面板，命令格式如下：

```python
visualdl --logdir <dir_1, dir_2, ... , dir_n> --host <host> --port <port> --cache-timeout <cache_timeout> --language <language> --public-path <public_path> --api-only --component_tabs <tab_name1, tab_name2, ...>
```

参数详情：

|      参数       |                                                                                             意义                                                                                             |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| --logdir        | 设定日志所在目录，可以指定多个目录，VisualDL将遍历并且迭代寻找指定目录的子目录，将所有实验结果进行可视化                                                                                     |
| --model         | 设定模型文件路径(非文件夹路径)，VisualDL将在此路径指定的模型文件进行可视化，目前可支持PaddlePaddle、ONNX、Keras、Core ML、Caffe等多种模型结构，详情可查看[graph支持模型种类](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.2/guides/03_VisualDL/visualdl_usage_cn.html#id11) |
| --host          | 设定IP，默认为`127.0.0.1`                                                                                                                                                                    |
| --port          | 设定端口，默认为`8040`                                                                                                                                                                       |
| --cache-timeout | 后端缓存时间，在缓存时间内前端多次请求同一url，返回的数据从缓存中获取，默认为20秒                                                                                                            |
| --language      | VisualDL面板语言，可指定为'en'或'zh'，默认为浏览器使用语言                                                                                                                                   |
| --public-path   | VisualDL面板URL路径，默认是'/app'，即访问地址为'http://&lt;host&gt;:&lt;port&gt;/app'                                                                                                                    |
| --api-only      | 是否只提供API，如果设置此参数，则VisualDL不提供页面展示，只提供API服务，此时API地址为'http://&lt;host&gt;:&lt;port&gt;/&lt;public_path&gt;/api'；若没有设置public_path参数，则默认为'http://&lt;host&gt;:&lt;port&gt;/api' |
| --component_tabs | 设定需要显示的组件，当前支持 `scalar`、`image`、`text`、`embeddings`、`audio`、`histogram`、`hyper_parameters`、`static_graph`、`dynamic_graph`、`pr_curve`、`roc_curve`、`profiler`、`x2paddle`、`fastdeploy_server`、`fastdeploy_client` 共 15 个组件。如果设置了此参数，将只展示所指定的组件。如果没有设置此参数，当指定了 `--logdir` 参数时，将会根据日志文件中拥有的数据类型来自动显示相应的组件。当没有指定 `--logdir` 参数，默认显示 `static_graph`、`x2paddle`、`fastdeploy_server`、`fastdeploy_client` 这四个名称代表的组件。 |


针对上一步生成的日志，启动命令为：

```
visualdl --logdir ./log
```

**示例：**

```
visualdl --logdir ./log --port 8089
```

在浏览器输入：`localhost:8089` 即可查看VisualDL面板。


#### b. 在Python脚本中启动

支持在Python脚本中启动VisualDL面板，接口如下：

```python
visualdl.server.app.run(logdir,
                        host="127.0.0.1",
                        port=8080,
                        cache_timeout=20,
                        language=None,
                        public_path=None,
                        api_only=False,
                        open_browser=False)
```

请注意：除`logdir`外，其他参数均为不定参数，传递时请指明参数名。

接口参数具体如下：

|     参数      |                       格式                       |                                                                                             含义                                                                                             |
| ------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| logdir        | string或list[string_1, string_2, ... , string_n] | 日志文件所在的路径，VisualDL将在此路径下递归搜索日志文件并进行可视化，可指定单个或多个路径                                                                                                   |
| model         | string                                           | 模型文件路径(非文件夹路径)，VisualDL将在此路径指定的模型文件进行可视化                                                                                                   |
| host          | string                                           | 指定启动服务的ip，默认为`127.0.0.1`                                                                                                                                                          |
| port          | int                                              | 启动服务端口，默认为`8040`                                                                                                                                                                   |
| cache_timeout | int                                              | 后端缓存时间，在缓存时间内前端多次请求同一url，返回的数据从缓存中获取，默认为20秒                                                                                                            |
| language      | string                                           | VisualDL面板语言，可指定为'en'或'zh'，默认为浏览器使用语言                                                                                                                                   |
| public_path   | string                                           | VisualDL面板URL路径，默认是'/app'，即访问地址为'http://&lt;host&gt;:&lt;port&gt;/app'                                                                                                                    |
| api_only      | boolean                                          | 是否只提供API，如果设置此参数，则VisualDL不提供页面展示，只提供API服务，此时API地址为'http://&lt;host&gt;:&lt;port&gt;/&lt;public_path&gt;/api'；若没有设置public_path参数，则默认为'http://&lt;host&gt;:&lt;port&gt;/api' |
| open_browser  | boolean                                          | 是否打开浏览器，设置为True则在启动后自动打开浏览器并访问VisualDL面板，若设置api_only，则忽略此参数                                                                                           |
| --component_tabs | string或list[string_1, string_2, ... , string_n] | 设定需要显示的组件，当前支持 `scalar`、`image`、`text`、`embeddings`、`audio`、`histogram`、`hyper_parameters`、`static_graph`、`dynamic_graph`、`pr_curve`、`roc_curve`、`profiler`、`x2paddle`、`fastdeploy_server`、`fastdeploy_client` 共 15 个组件。如果设置了此参数，将只展示所指定的组件。如果没有设置此参数，当指定了 `--logdir` 参数时，将会根据日志文件中拥有的数据类型来自动显示相应的组件。当没有指定 `--logdir` 参数，默认显示 `static_graph`、`x2paddle`、`fastdeploy_server`、`fastdeploy_client` 这四个名称代表的组件。 |

针对上一步生成的日志，我们的启动脚本为：

```python
from visualdl.server import app

app.run(logdir="./log")
```

在使用任意一种方式启动VisualDL面板后，打开浏览器访问VisualDL面板，即可查看日志的可视化结果，如图：

<p align="center">
  <img src="https://user-images.githubusercontent.com/48054808/82786044-67ae9880-9e96-11ea-8a2b-3a0951a6ec19.png" width="60%"/>
</p>



## 四、可视化功能概览

### 1. Scalar（标量）
以图表形式实时展示训练过程参数，如loss、accuracy。让用户通过观察单组或多组训练参数变化，了解训练过程，加速模型调优。具有两大特点：

#### a. 动态展示

在启动VisualDL Board后，LogReader将不断增量的读取日志中数据并供前端调用展示，因此能够在训练中同步观测指标变化，如下图：

<p align="center">
  <img src="https://visualdl.bj.bcebos.com/images/dynamic_display.gif" width="60%"/>
</p>


#### b. 多实验对比

只需在启动VisualDL Board的时将每个实验日志所在路径同时传入即可，每个实验中相同tag的指标将绘制在一张图中同步呈现，如下图：

<p align="center">
  <img src="https://visualdl.bj.bcebos.com/images/multi_experiments.gif" width="100%"/>
</p>


### 2. Image（图像）
实时展示训练过程中的图像数据，用于观察不同训练阶段的图像变化，进而深入了解训练过程及效果。

<p align="center">
<img src="https://user-images.githubusercontent.com/48054808/90356439-24715980-e082-11ea-8896-01c27fc2fc9b.gif" width="85%"/>
</p>

### 3. Audio（音频）
实时查看训练过程中的音频数据，监控语音识别与合成等任务的训练过程。

<p align="center">
<img src="https://user-images.githubusercontent.com/48054808/87659138-b4746880-c78f-11ea-965b-c33804e7c296.png" width="85%"/>
</p>

### 4. Text（文本）
展示文本任务任意阶段的数据输出，对比不同阶段的文本变化，便于深入了解训练过程及效果。

<p align="center">
<img src="https://user-images.githubusercontent.com/28444161/106248340-cdd09400-624b-11eb-8ea9-5a07a239c365.png" width="85%"/>
</p>

### 5. Network Structure（网络结构）

一键可视化模型的网络结构。可查看模型属性、节点信息、节点输入输出等，并支持节点搜索，辅助用户快速分析模型结构与了解数据流向，覆盖动态图与静态图两种格式。

- 动态图

<p align="center">
<img src="https://user-images.githubusercontent.com/22424850/175770313-2509f7e9-041a-4654-9a0f-45f4bd76e1e8.gif" width="85%"/>
</p>

- 静态图

<p align="center">
<img src="https://user-images.githubusercontent.com/22424850/175770315-11e2c3f5-141c-4f05-be86-0e1e2785e11f.gif" width="85%"/>
</p>

### 6. Histogram（直方图）

以直方图形式展示Tensor（weight、bias、gradient等）数据在训练过程中的变化趋势。深入了解模型各层效果，帮助开发者精准调整模型结构。

- Offset模式

<p align="center">
<img src="https://user-images.githubusercontent.com/48054808/86551031-86647c80-bf76-11ea-8ec2-8c86826c8137.png" width="85%"/>
</p>

- Overlay模式

<p align="center">
<img src="https://user-images.githubusercontent.com/48054808/86551033-882e4000-bf76-11ea-8e6a-af954c662ced.png" width="85%"/>
</p>

### 7. PR Curve（精度-召回率曲线）

精度-召回率曲线，帮助开发者权衡模型精度和召回率之间的平衡，设定最佳阈值。

<p align="center">
<img src="https://user-images.githubusercontent.com/48054808/86738774-ee46c000-c067-11ea-90d2-a98aac445cca.png" width="85%"/>
</p>

### 8. High Dimensional（高维数据）

将高维数据进行降维展示，目前支持T-SNE、PCA两种降维方式，用于深入分析高维数据间的关系，方便用户根据数据特征进行算法优化。

<p align="center">
<img src="https://user-images.githubusercontent.com/48054808/82396340-3e4dd100-9a80-11ea-911d-798acdbc9c90.gif" width="85%"/>
</p>

### 9. Hyper Parameters（超参数）

以丰富的视图多角度地可视化超参数与模型关键指标间的关系，便于快速确定最佳超参组合，实现高效调参。

<p align="center">
<img src="https://user-images.githubusercontent.com/28444161/119247155-e9c0c280-bbb9-11eb-8175-58a9c7657a9c.gif" width="85%"/>
</p>

### 10. Performance Analysis（性能分析）

通过多个视图可视化性能分析的数据，辅助用户定位性能瓶颈并进行优化。可参考[使用VisualDL做性能分析](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/profiler/README_CN.md)。

<p align="center">
<img src="https://user-images.githubusercontent.com/22424850/185893177-a049c8d5-2310-4138-8dd5-844cf198e425.gif" width="85%"/>
</p>

### 11. X2Paddle

提供onnx模型转paddle模型的可视化操作界面，帮助用户可视化onnx模型结构并且获取转换后的paddle模型结构和参数文件。

<p align="center">
  <img src="https://user-images.githubusercontent.com/22424850/211203066-f2e43ef5-104f-436a-b44c-cad2b37ad518.gif" width="100%"/>
</p>


### 12. FastDeployServer

基于[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)的Serving可视化部署，提供配置模型库、管理监控服务以及测试服务等功能。详细内容可参考[使用VisualDL进行Serving可视化部署](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/fastdeploy_server/README_CN.md)。

 <p align="center">
  <img src="https://user-images.githubusercontent.com/22424850/211196832-1a05bf80-5aaa-493f-bba2-27e819c18bb9.gif" width="100%"/>
</p>


### 13. FastDeployClient
提供给用户访问fastdeployserver服务的客户端界面，进行一键预测和可视化结果。详细内容可参考[使用VisualDL作为fastdeployserver服务的客户端](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/fastdeploy_client/README_CN.md)。
<p align="center">
  <img src="https://user-images.githubusercontent.com/22424850/211203852-059d5b98-6299-4057-97d8-5209805aa67f.gif" width="100%"/>
</p>


### 14. VDL.service

VisualDL可视化结果保存服务，以链接形式将可视化结果保存下来，方便用户快速、便捷的进行托管与分享。

<p align="center">
<img src="https://user-images.githubusercontent.com/48054808/93729521-72382f00-fbf7-11ea-91ff-6b6ab4b41e32.png" width="85%"/>
</p>


## 五、开源贡献

VisualDL 是由 [PaddlePaddle](https://www.paddlepaddle.org/) 和 [ECharts](https://echarts.apache.org/) 合作推出的开源项目。 Graph 相关功能由 [Netron](https://github.com/lutzroeder/netron) 提供技术支持。 欢迎所有人使用，提意见以及贡献代码。


## 六、更多细节

想了解更多关于VisualDL可视化功能的使用详情介绍，请查看[**VisualDL使用指南**](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README_CN.md)，[**使用VisualDL做性能分析**](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/profiler/README_CN.md)，[**使用VisualDL进行Serving可视化部署**](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/fastdeploy_server/README_CN.md)，[**使用VisualDL作为fastdeployserver服务的客户端**](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/fastdeploy_client/README_CN.md)。
