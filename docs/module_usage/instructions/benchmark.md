# 模型推理 Benchmark

## 目录

- [1. 使用说明](#1.使用说明)
- [2. 使用示例](#2.使用示例)
  - [2.1 命令行方式](#2.1-命令行方式)
  - [2.2 Python 脚本方式](#2.2-Python-脚本方式)
- [3. 结果说明](#3.结果说明)

## 1.使用说明

Benchmark 会统计模型在端到端推理过程中，所有操作（`Operation`）和阶段（`Stage`）的每次迭代的平均执行时间（`Avg Time Per Iter (ms)`）和每个样本的平均执行时间（`Avg Time Per Instance (ms)`），单位为毫秒。

需通过环境变量启用 Benchmark，具体如下：

* `PADDLE_PDX_INFER_BENCHMARK`：设置为 `True` 时则开启 Benchmark，默认为 `False`；
* `PADDLE_PDX_INFER_BENCHMARK_WARMUP`：设置 warm up，在开始测试前循环迭代 n 次，默认为 `0`；
* `PADDLE_PDX_INFER_BENCHMARK_ITER`：进行 Benchmark 测试的循环次数，默认为 `0`；
* `PADDLE_PDX_INFER_BENCHMARK_OUTPUT`：用于设置保存的目录，如 `./benchmark`，默认为 `None`，表示不保存 Benchmark 指标；

**注意**：

* `PADDLE_PDX_INFER_BENCHMARK_WARMUP` 或 `PADDLE_PDX_INFER_BENCHMARK_ITER` 需要至少设置一个大于零的值，否则无法启用 Benchmark。

## 2.使用示例

您可以通过以下两种方式来使用 benchmark：命令行方式和 Python 脚本方式。

### 2.1 命令行方式

**注意**：

- 输入参数说明可参考 [PaddleX通用模型配置文件参数说明](./config_parameters_common.md)
- `Predict.input` 在 Benchmark 只能被设置为输入数据的本地路径。如果 `batch_size` 大于 1，输入数据将被重复 `batch_size` 次以匹配 `batch_size` 的大小。

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
- `input` 在 Benchmark 只能被设置为输入数据的本地路径。如果 `batch_size` 大于 1，输入数据将被重复 `batch_size` 次以匹配 `batch_size` 的大小。

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

在开启 Benchmark 后，将自动打印 Benchmark 结果，具体说明如下：

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
            <td>阶段名称，包括预处理（PreProcess）、推理（Inference）、后处理（PostProcess）、以及端到端（End2End）。</td>
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

运行第2节的示例程序所得到的 Benchmark 结果如下：

```
                                             WarmUp Data
+-------+------------+-----------+-------------+------------------------+----------------------------+
| Iters | Batch Size | Instances |    Stage    | Avg Time Per Iter (ms) | Avg Time Per Instance (ms) |
+-------+------------+-----------+-------------+------------------------+----------------------------+
|   5   |     2      |     10    |  PreProcess |      98.70615005       |        49.35307503         |
|   5   |     2      |     10    |  Inference  |      68.70298386       |        34.35149193         |
|   5   |     2      |     10    | PostProcess |       0.22978783       |         0.11489391         |
|   5   |     2      |     10    |   End2End   |      167.63892174      |        83.81946087         |
+-------+------------+-----------+-------------+------------------------+----------------------------+
                                               Detail Data
+-------+------------+-----------+----------------+------------------------+----------------------------+
| Iters | Batch Size | Instances |   Operation    | Avg Time Per Iter (ms) | Avg Time Per Instance (ms) |
+-------+------------+-----------+----------------+------------------------+----------------------------+
|   10  |     2      |     20    |   ReadImage    |      77.00567245       |        38.50283623         |
|   10  |     2      |     20    |     Resize     |      11.97342873       |         5.98671436         |
|   10  |     2      |     20    |   Normalize    |       6.09791279       |         3.04895639         |
|   10  |     2      |     20    |   ToCHWImage   |       0.00574589       |         0.00287294         |
|   10  |     2      |     20    |    ToBatch     |       0.72050095       |         0.36025047         |
|   10  |     2      |     20    |    Copy2GPU    |       3.15101147       |         1.57550573         |
|   10  |     2      |     20    |     Infer      |       9.58673954       |         4.79336977         |
|   10  |     2      |     20    |    Copy2CPU    |       0.07462502       |         0.03731251         |
|   10  |     2      |     20    | DetPostProcess |       0.22695065       |         0.11347532         |
+-------+------------+-----------+----------------+------------------------+----------------------------+
                                             Summary Data
+-------+------------+-----------+-------------+------------------------+----------------------------+
| Iters | Batch Size | Instances |    Stage    | Avg Time Per Iter (ms) | Avg Time Per Instance (ms) |
+-------+------------+-----------+-------------+------------------------+----------------------------+
|   10  |     2      |     20    |  PreProcess |      95.80326080       |        47.90163040         |
|   10  |     2      |     20    |  Inference  |      12.81237602       |         6.40618801         |
|   10  |     2      |     20    | PostProcess |       0.22695065       |         0.11347532         |
|   10  |     2      |     20    |   End2End   |      108.84258747      |        54.42129374         |
+-------+------------+-----------+-------------+------------------------+----------------------------+
```

同时，由于设置了`PADDLE_PDX_INFER_BENCHMARK_OUTPUT=./benchmark`，所以上述结果会保存到到本地： `./benchmark/detail.csv` 和 `./benchmark/summary.csv`：

`detail.csv` 内容如下：

```csv
Iters,Batch Size,Instances,Operation,Avg Time Per Iter (ms),Avg Time Per Instance (ms)
10,2,20,ReadImage,77.00567245,38.50283623
10,2,20,Resize,11.97342873,5.98671436
10,2,20,Normalize,6.09791279,3.04895639
10,2,20,ToCHWImage,0.00574589,0.00287294
10,2,20,ToBatch,0.72050095,0.36025047
10,2,20,Copy2GPU,3.15101147,1.57550573
10,2,20,Infer,9.58673954,4.79336977
10,2,20,Copy2CPU,0.07462502,0.03731251
10,2,20,DetPostProcess,0.22695065,0.11347532
```

`summary.csv` 内容如下：

```csv
Iters,Batch Size,Instances,Stage,Avg Time Per Iter (ms),Avg Time Per Instance (ms)
10,2,20,PreProcess,95.80326080,47.90163040
10,2,20,Inference,12.81237602,6.40618801
10,2,20,PostProcess,0.22695065,0.11347532
10,2,20,End2End,108.84258747,54.42129374
```
