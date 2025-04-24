# 模型推理 Benchmark

## 目录

- [1. 使用说明](#1.-使用说明)
- [2. 使用示例](#2.-使用示例)
  - [2.1 命令行方式](#2.1-命令行方式)
  - [2.2 Python 脚本方式](#2.2-Python-脚本方式)
- [3. 结果说明](#3.-结果说明)

## 1. 使用说明

Benchmark 功能会统计模型在端到端推理过程中，所有操作的每次迭代的平均执行时间和每个实例的平均执行时间，并给出汇总信息。耗时数据单位为毫秒。

需通过环境变量启用 benchmark 功能，具体如下：

* `PADDLE_PDX_INFER_BENCHMARK`：设置为 `True` 时则开启 benchmark 功能，默认为 `False`；
* `PADDLE_PDX_INFER_BENCHMARK_WARMUP`：测试前的预热次数，默认为 `0`；
* `PADDLE_PDX_INFER_BENCHMARK_ITERS`：测试的循环次数，默认为 `0`；
* `PADDLE_PDX_INFER_BENCHMARK_OUTPUT_DIR`：保存指标的目录，如 `./benchmark`，默认为 `None`，表示不保存 benchmark 指标；
* `PADDLE_PDX_INFER_BENCHMARK_USE_CACHE_FOR_READ`：设置为 `True` 时则对读取输入数据操作应用缓存机制，避免重复I/O开销，并且数据读取及缓存消耗的时间不记录到核心耗时中。默认为 `False`；
* `PADDLE_PDX_INFER_BENCHMARK_USE_NEW_INFER_API`：设置为 `True` 时则使用新的推理API，可以看更细致的分阶段结果。默认为 `False`。

**注意**：

* `PADDLE_PDX_INFER_BENCHMARK_WARMUP` 或 `PADDLE_PDX_INFER_BENCHMARK_ITERS` 需要至少设置一个大于零的值，否则无法使用 benchmark 功能。
* Benchmark 功能目前不适用于模型产线。

## 2. 使用示例

您可以通过以下两种方式之一来使用 benchmark 功能：命令行方式和 Python 脚本方式。

### 2.1 命令行方式

**注意**：

- 输入参数说明请参考 [PaddleX通用模型配置文件参数说明](./config_parameters_common.md)
- 如果 `batch_size` 大于 1，输入数据将被重复 `batch_size` 次以匹配 `batch_size` 的大小。

执行命令：

```bash
PADDLE_PDX_INFER_BENCHMARK=True \
PADDLE_PDX_INFER_BENCHMARK_WARMUP=5 \
PADDLE_PDX_INFER_BENCHMARK_ITERS=10 \
PADDLE_PDX_INFER_BENCHMARK_OUTPUT_DIR=./benchmark \
python main.py \
    -c ./paddlex/configs/modules/object_detection/PicoDet-XS.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir=None \
    -o Predict.batch_size=2 \
    -o Predict.input=./test.png
```

### 2.2 Python 脚本方式

**注意**：

- 输入参数说明请参考 [PaddleX单模型Python脚本使用说明](./model_python_API.md)
- 如果 `batch_size` 大于 1，输入数据将被重复 `batch_size` 次以匹配 `batch_size` 的大小。

创建 `test_infer.py` 脚本：

```python
from paddlex import create_model

model = create_model(model_name="PicoDet-XS", model_dir=None)
output = list(model.predict(input="./test.png", batch_size=2))
```

执行脚本：

```bash
PADDLE_PDX_INFER_BENCHMARK=True \
PADDLE_PDX_INFER_BENCHMARK_WARMUP=5 \
PADDLE_PDX_INFER_BENCHMARK_ITERS=10 \
PADDLE_PDX_INFER_BENCHMARK_OUTPUT_DIR=./benchmark \
python test_infer.py
```

## 3. 结果说明

在开启 benchmark 功能后，将自动打印 benchmark 结果，具体说明如下：

<table border="1">
    <thead>
        <tr>
            <th>字段名</th>
            <th>字段含义</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Iters</td>
            <td>迭代次数，指执行推理的循环次数。</td>
        </tr>
        <tr>
            <td>Batch Size</td>
            <td>批次大小，指每次迭代中处理的实例数量。</td>
        </tr>
        <tr>
            <td>Instances</td>
            <td>实例总数，计算方式为 <code>Iters</code> 乘以 <code>Batch Size</code>。</td>
        </tr>
        <tr>
            <td>Operation</td>
            <td>操作名称，如 <code>Resize</code>、<code>Normalize</code> 等。</td>
        </tr>
        <tr>
            <td>Type</td>
            <td>耗时类型，包括：
            <ul>
            <li>预处理耗时（<code>Preprocessing</code>）</li>
            <li>模型推理耗时（<code>Inference</code>）</li>
            <li>后处理耗时（<code>Postprocessing</code>）</li>
            <li>核心耗时（<code>Core</code>，即预处理耗时+模型推理耗时+后处理耗时）</li>
            <li>其他耗时（<code>Other</code>，例如运行用于编排操作的代码所花费的时间以及由基准测试功能引入的额外开销）</li>
            <li>端到端耗时（<code>End-to-End</code>，即核心耗时+其他耗时）</li>
            </ul>
            </td>
        </tr>
        <tr>
            <td>Avg Time Per Iter (ms)</td>
            <td>每次迭代的平均执行时间，单位为毫秒。</td>
        </tr>
        <tr>
            <td>Avg Time Per Instance (ms)</td>
            <td>每个实例的平均执行时间，单位为毫秒。</td>
        </tr>
    </tbody>
</table>

运行第2节的示例程序所得到的 benchmark 结果如下：

```
                                               Warmup Data
+-------+------------+-----------+----------------+------------------------+----------------------------+
| Iters | Batch Size | Instances |      Type      | Avg Time Per Iter (ms) | Avg Time Per Instance (ms) |
+-------+------------+-----------+----------------+------------------------+----------------------------+
|   5   |     2      |     10    | Preprocessing  |      97.89338876       |        48.94669438         |
|   5   |     2      |     10    |   Inference    |      66.70711380       |        33.35355690         |
|   5   |     2      |     10    | Postprocessing |       0.20138482       |         0.10069241         |
|   5   |     2      |     10    |      Core      |      164.80188738      |        82.40094369         |
|   5   |     2      |     10    |     Other      |       3.41097047       |         1.70548523         |
|   5   |     2      |     10    |   End-to-End   |      168.21285784      |        84.10642892         |
+-------+------------+-----------+----------------+------------------------+----------------------------+
                                           Operation Info
+--------------------+----------------------------------------------------------------------+
|     Operation      |                         Source Code Location                         |
+--------------------+----------------------------------------------------------------------+
|     ReadImage      | /PaddleX/paddlex/inference/models/object_detection/processors.py:34  |
|       Resize       | /PaddleX/paddlex/inference/models/object_detection/processors.py:99  |
|     Normalize      | /PaddleX/paddlex/inference/models/object_detection/processors.py:145 |
|     ToCHWImage     | /PaddleX/paddlex/inference/models/object_detection/processors.py:158 |
|      ToBatch       | /PaddleX/paddlex/inference/models/object_detection/processors.py:216 |
| PaddleCopyToDevice |     /PaddleX/paddlex/inference/models/common/static_infer.py:214     |
|  PaddleModelInfer  |     /PaddleX/paddlex/inference/models/common/static_infer.py:234     |
|  PaddleCopyToHost  |     /PaddleX/paddlex/inference/models/common/static_infer.py:223     |
|   DetPostProcess   | /PaddleX/paddlex/inference/models/object_detection/processors.py:773 |
+--------------------+----------------------------------------------------------------------+
                                                 Detail Data
+-------+------------+-----------+--------------------+------------------------+----------------------------+
| Iters | Batch Size | Instances |     Operation      | Avg Time Per Iter (ms) | Avg Time Per Instance (ms) |
+-------+------------+-----------+--------------------+------------------------+----------------------------+
|   10  |     2      |     20    |     ReadImage      |      76.22221033       |        38.11110517         |
|   10  |     2      |     20    |       Resize       |      12.02824502       |         6.01412251         |
|   10  |     2      |     20    |     Normalize      |       6.14072606       |         3.07036303         |
|   10  |     2      |     20    |     ToCHWImage     |       0.00533939       |         0.00266969         |
|   10  |     2      |     20    |      ToBatch       |       0.93134162       |         0.46567081         |
|   10  |     2      |     20    | PaddleCopyToDevice |       0.92240779       |         0.46120390         |
|   10  |     2      |     20    |  PaddleModelInfer  |       9.66330138       |         4.83165069         |
|   10  |     2      |     20    |  PaddleCopyToHost  |       0.06802108       |         0.03401054         |
|   10  |     2      |     20    |   DetPostProcess   |       0.18665448       |         0.09332724         |
+-------+------------+-----------+--------------------+------------------------+----------------------------+
                                               Summary Data
+-------+------------+-----------+----------------+------------------------+----------------------------+
| Iters | Batch Size | Instances |      Type      | Avg Time Per Iter (ms) | Avg Time Per Instance (ms) |
+-------+------------+-----------+----------------+------------------------+----------------------------+
|   10  |     2      |     20    | Preprocessing  |      95.32786242       |        47.66393121         |
|   10  |     2      |     20    |   Inference    |      10.65373025       |         5.32686512         |
|   10  |     2      |     20    | Postprocessing |       0.18665448       |         0.09332724         |
|   10  |     2      |     20    |      Core      |      106.16824715      |        53.08412358         |
|   10  |     2      |     20    |     Other      |       2.74794563       |         1.37397281         |
|   10  |     2      |     20    |   End-to-End   |      108.91619278      |        54.45809639         |
+-------+------------+-----------+----------------+------------------------+----------------------------+
```

同时，由于设置了`PADDLE_PDX_INFER_BENCHMARK_OUTPUT_DIR=./benchmark`，所以上述结果会保存到到本地： `./benchmark/detail.csv` 和 `./benchmark/summary.csv`：

`detail.csv` 内容如下：

```csv
Iters,Batch Size,Instances,Operation,Avg Time Per Iter (ms),Avg Time Per Instance (ms)
10,2,20,ReadImage,76.22221033,38.11110517
10,2,20,Resize,12.02824502,6.01412251
10,2,20,Normalize,6.14072606,3.07036303
10,2,20,ToCHWImage,0.00533939,0.00266969
10,2,20,ToBatch,0.93134162,0.46567081
10,2,20,PaddleCopyToDevice,0.92240779,0.46120390
10,2,20,PaddleModelInfer,9.66330138,4.83165069
10,2,20,PaddleCopyToHost,0.06802108,0.03401054
10,2,20,DetPostProcess,0.18665448,0.09332724
```

`summary.csv` 内容如下：

```csv
Iters,Batch Size,Instances,Type,Avg Time Per Iter (ms),Avg Time Per Instance (ms)
10,2,20,Preprocessing,95.32786242,47.66393121
10,2,20,Inference,10.65373025,5.32686512
10,2,20,Postprocessing,0.18665448,0.09332724
10,2,20,Core,106.16824715,53.08412358
10,2,20,Other,2.74794563,1.37397281
10,2,20,End-to-End,108.91619278,54.45809639
```
