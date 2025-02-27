# 模型推理 Benchmark

## 目录

- [1. 使用说明](#1.使用说明)
- [2. 使用示例](#2.使用示例)
  - [2.1 命令行方式](#2.1-命令行方式)
  - [2.2 Python 脚本方式](#2.2-Python-脚本方式)
- [3. 结果说明](#3.结果说明)

## 1.使用说明

Benchmark 功能会统计模型在端到端推理过程中，所有操作（`Operation`）和阶段（`Stage`）的每次迭代的平均执行时间（`Avg Time Per Iter (ms)`）和每个样本的平均执行时间（`Avg Time Per Instance (ms)`），单位为毫秒。

需通过环境变量启用 benchmark 功能，具体如下：

* `PADDLE_PDX_INFER_BENCHMARK`：设置为 `True` 时则开启 benchmark 功能，默认为 `False`；
* `PADDLE_PDX_INFER_BENCHMARK_WARMUP`：设置预热，在开始测试前循环迭代 n 次，默认为 `0`；
* `PADDLE_PDX_INFER_BENCHMARK_ITER`：进行测试的循环次数，默认为 `0`；
* `PADDLE_PDX_INFER_BENCHMARK_OUTPUT`：用于设置保存的目录，如 `./benchmark`，默认为 `None`，表示不保存 benchmark 指标；

**注意**：

* `PADDLE_PDX_INFER_BENCHMARK_WARMUP` 或 `PADDLE_PDX_INFER_BENCHMARK_ITER` 需要至少设置一个大于零的值，否则无法使用 benchmark 功能。
* Benchmark 功能目前不适用于模型产线。

## 2.使用示例

您可以通过以下两种方式之一来使用 benchmark 功能：命令行方式和 Python 脚本方式。

### 2.1 命令行方式

**注意**：

- 输入参数说明可参考 [PaddleX通用模型配置文件参数说明](./config_parameters_common.md)
- `Predict.input` 在 benchmark 中只能被设置为输入数据的本地路径。如果 `batch_size` 大于 1，输入数据将被重复 `batch_size` 次以匹配 `batch_size` 的大小。

执行命令：

```bash
PADDLE_PDX_INFER_BENCHMARK=True \
PADDLE_PDX_INFER_BENCHMARK_WARMUP=5 \
PADDLE_PDX_INFER_BENCHMARK_ITER=10 \
PADDLE_PDX_INFER_BENCHMARK_OUTPUT=./benchmark \
python main.py \
    -c ./paddlex/configs/modules/object_detection/PicoDet-XS.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir=None \
    -o Predict.batch_size=2 \
    -o Predict.input=./test.png

# 使用pptrt推理后端
#   -o Predict.kernel_option="{'run_mode': 'trt_fp32'}"
```

### 2.2 Python 脚本方式

**注意**：

- 输入参数说明可参考 [PaddleX单模型Python脚本使用说明](./model_python_API.md)
- `input` 在 benchmark 中只能被设置为输入数据的本地路径。如果 `batch_size` 大于 1，输入数据将被重复 `batch_size` 次以匹配 `batch_size` 的大小。

创建 `test_infer.py` 脚本：

```python
from paddlex import create_model

model = create_model(model_name="PicoDet-XS", model_dir=None)
output = list(model.predict(input="./test.png", batch_size=2))

# 使用pptrt推理后端
# from paddlex import create_model
# from paddlex.inference.utils.pp_option import PaddlePredictorOption

# pp_option = PaddlePredictorOption()
# pp_option.run_mode = "trt_fp32"
# model = create_model(model_name="PicoDet-XS", model_dir=None, pp_option=pp_option)
# output = list(model.predict(input="./test.png", batch_size=2))
```

执行脚本：

```bash
PADDLE_PDX_INFER_BENCHMARK=True \
PADDLE_PDX_INFER_BENCHMARK_WARMUP=5 \
PADDLE_PDX_INFER_BENCHMARK_ITER=10 \
PADDLE_PDX_INFER_BENCHMARK_OUTPUT=./benchmark \
python test_infer.py
```

## 3.结果示例

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
            <td>迭代次数，指执行模型推理的循环次数。</td>
        </tr>
        <tr>
            <td>Batch Size</td>
            <td>批处理大小，指每次迭代中处理的样本数量。</td>
        </tr>
        <tr>
            <td>Instances</td>
            <td>总样本数量，计算方式为 <code>Iters</code> 乘以 <code>Batch Size</code>。</td>
        </tr>
        <tr>
            <td>Operation</td>
            <td>操作名称，如 <code>Resize</code>、<code>Normalize</code> 等。</td>
        </tr>
        <tr>
            <td>Stage</td>
            <td>阶段名称，包括预处理（<code>Preprocessing</code>）、推理（<code>Inference</code>）、后处理（<code>Postprocessing</code>）、其它（<code>Other</code>）以及端到端（<code>End2End</code>）。</td>
        </tr>
        <tr>
            <td>Avg Time Per Iter (ms)</td>
            <td>每次迭代的平均执行时间，单位为毫秒。</td>
        </tr>
        <tr>
            <td>Avg Time Per Instance (ms)</td>
            <td>每个样本的平均执行时间，单位为毫秒。</td>
        </tr>
    </tbody>
</table>

运行第2节的示例程序所得到的 benchmark 结果如下：

```
                                             WarmUp Data
+-------+------------+-----------+-------------+------------------------+----------------------------+
| Iters | Batch Size | Instances |    Stage    | Avg Time Per Iter (ms) | Avg Time Per Instance (ms) |
+-------+------------+-----------+-------------+------------------------+----------------------------+
|   5   |     2      |     10    |  PreProcess |      100.07238388      |        50.03619194         |
|   5   |     2      |     10    |  Inference  |      68.80149841       |        34.40074921         |
|   5   |     2      |     10    | PostProcess |       0.23727417       |         0.11863708         |
|   5   |     2      |     10    |   End2End   |      169.11115646      |        84.55557823         |
+-------+------------+-----------+-------------+------------------------+----------------------------+
                                                 Detail Data
+-------+------------+-----------+--------------------+------------------------+----------------------------+
| Iters | Batch Size | Instances |     Operation      | Avg Time Per Iter (ms) | Avg Time Per Instance (ms) |
+-------+------------+-----------+--------------------+------------------------+----------------------------+
|   10  |     2      |     20    |     ReadImage      |      77.85725594       |        38.92862797         |
|   10  |     2      |     20    |       Resize       |      12.18175888       |         6.09087944         |
|   10  |     2      |     20    |     Normalize      |       6.10103607       |         3.05051804         |
|   10  |     2      |     20    |     ToCHWImage     |       0.00677109       |         0.00338554         |
|   10  |     2      |     20    |      ToBatch       |       0.99833012       |         0.49916506         |
|   10  |     2      |     20    | PaddleCopyToDevice |       3.21643353       |         1.60821676         |
|   10  |     2      |     20    |  PaddleModelInfer  |      10.33163071       |         5.16581535         |
|   10  |     2      |     20    |  PaddleCopyToHost  |       0.07891655       |         0.03945827         |
|   10  |     2      |     20    |   DetPostProcess   |       0.21560192       |         0.10780096         |
+-------+------------+-----------+--------------------+------------------------+----------------------------+
                                             Summary Data
+-------+------------+-----------+-------------+------------------------+----------------------------+
| Iters | Batch Size | Instances |    Stage    | Avg Time Per Iter (ms) | Avg Time Per Instance (ms) |
+-------+------------+-----------+-------------+------------------------+----------------------------+
|   10  |     2      |     20    |  PreProcess |      97.14515209       |        48.57257605         |
|   10  |     2      |     20    |  Inference  |      13.62698078       |         6.81349039         |
|   10  |     2      |     20    | PostProcess |       0.21560192       |         0.10780096         |
|   10  |     2      |     20    |   End2End   |      110.98773479      |        55.49386740         |
+-------+------------+-----------+-------------+------------------------+----------------------------+
```

同时，由于设置了`PADDLE_PDX_INFER_BENCHMARK_OUTPUT=./benchmark`，所以上述结果会保存到到本地： `./benchmark/detail.csv` 和 `./benchmark/summary.csv`：

`detail.csv` 内容如下：

```csv
Iters,Batch Size,Instances,Operation,Avg Time Per Iter (ms),Avg Time Per Instance (ms)
10,2,20,ReadImage,77.85725594,38.92862797
10,2,20,Resize,12.18175888,6.09087944
10,2,20,Normalize,6.10103607,3.05051804
10,2,20,ToCHWImage,0.00677109,0.00338554
10,2,20,ToBatch,0.99833012,0.49916506
10,2,20,PaddleCopyToDevice,3.21643353,1.60821676
10,2,20,PaddleModelInfer,10.33163071,5.16581535
10,2,20,PaddleCopyToHost,0.07891655,0.03945827
10,2,20,DetPostProcess,0.21560192,0.10780096
```

`summary.csv` 内容如下：

```csv
Iters,Batch Size,Instances,Stage,Avg Time Per Iter (ms),Avg Time Per Instance (ms)
10,2,20,PreProcess,97.14515209,48.57257605
10,2,20,Inference,13.62698078,6.81349039
10,2,20,PostProcess,0.21560192,0.10780096
10,2,20,End2End,110.98773479,55.49386740
```
