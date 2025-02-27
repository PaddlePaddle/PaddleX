# Model Inference Benchmark

## Table of Contents

- [1. Usage Instructions](#1-usage-instructions)
- [2. Usage Examples](#2-usage-examples)
  - [2.1 Command Line Method](#21-command-line-method)
  - [2.2 Python Script Method](#22-python-script-method)
- [3. Result Interpretation](#3-result-interpretation)

## 1. Usage Instructions

The Benchmark will measure the average execution time per iteration (`Avg Time Per Iter (ms)`) and the average execution time per instance (`Avg Time Per Instance (ms)`) for all operations (`Operation`) and stages (`Stage`) in the end-to-end inference process, with milliseconds as the unit.

The Benchmark needs to be enabled through environment variables, specifically as follows:

* `PADDLE_PDX_INFER_BENCHMARK`: Set to `True` to enable the Benchmark, default is `False`;
* `PADDLE_PDX_INFER_BENCHMARK_WARMUP`: Sets the warm-up phase, iterating n times before starting the test, default is `0`;
* `PADDLE_PDX_INFER_BENCHMARK_ITER`: The number of iterations for the Benchmark test, default is `0`;
* `PADDLE_PDX_INFER_BENCHMARK_OUTPUT`: Sets the directory for saving results, e.g., `./benchmark`, default is `None`, meaning Benchmark metrics will not be saved;

**Note**:

* Either `PADDLE_PDX_INFER_BENCHMARK_WARMUP` or `PADDLE_PDX_INFER_BENCHMARK_ITER` needs to be set to a value greater than zero to enable the Benchmark.

## 2. Usage Examples

You can use the benchmark in the following two ways: command line method and Python script method.

### 2.1 Command Line Method

**Note**:

- For input parameter descriptions, refer to [PaddleX Common Configuration Parameters for Models](./config_parameters_common_en.md)
- `Predict.input` in Benchmark can only be set to the local path of the input data. If `batch_size` is greater than 1, the input data will be repeated `batch_size` times to match the `batch_size`.

Execution command:

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

# Using the PaddlePaddle Inference backend
#   -o Predict.kernel_option="{'run_mode': 'trt_fp32'}"
```

### 2.2 Python Script Method

**Note**:

- For input parameter descriptions, refer to [PaddleX Single Model Python Script Usage Instructions](./model_python_API_en.md)
- `input` in Benchmark can only be set to the local path of the input data. If `batch_size` is greater than 1, the input data will be repeated `batch_size` times to match the `batch_size`.

Create a `test_infer.py` script:

```python
from paddlex import create_model

model = create_model(model_name="PicoDet-XS", model_dir=None)
output = list(model.predict(input="./test.png", batch_size=2))

# Using the PaddlePaddle Inference backend
# from paddlex import create_model
# from paddlex.inference.utils.pp_option import PaddlePredictorOption

# pp_option = PaddlePredictorOption()
# pp_option.run_mode = "trt_fp32"
# model = create_model(model_name="PicoDet-XS", model_dir=None, pp_option=pp_option)
# output = list(model.predict(input="./test.png", batch_size=2))
```

Execute the script:

```bash
PADDLE_PDX_INFER_BENCHMARK=True \
PADDLE_PDX_INFER_BENCHMARK_WARMUP=5 \
PADDLE_PDX_INFER_BENCHMARK_ITER=10 \
PADDLE_PDX_INFER_BENCHMARK_OUTPUT=./benchmark \
python test_infer.py
```

## 3. Result Example

After enabling the Benchmark, the Benchmark results will be automatically printed, with specific descriptions as follows:

<table border="1">
    <thead>
        <tr>
            <th>Field Name</th>
            <th>Field Meaning</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Iters</td>
            <td>Number of iterations, referring to the number of loops for model inference execution.</td>
        </tr>
        <tr>
            <td>Batch Size</td>
            <td>Batch size, referring to the number of samples processed in each iteration.</td>
        </tr>
        <tr>
            <td>Instances</td>
            <td>Total number of samples, calculated as <code>Iters</code> multiplied by <code>Batch Size</code>.</td>
        </tr>
        <tr>
            <td>Operation</td>
            <td>Operation name, such as <code>Resize</code>, <code>Normalize</code>, etc.</td>
        </tr>
        <tr>
            <td>Stage</td>
            <td>Stage name, including PreProcess, Inference, PostProcess, Others(such as formatting output, packaging results, etc), and End2End.</td>
        </tr>
        <tr>
            <td>Avg Time Per Iter (ms)</td>
            <td>Average execution time per iteration, in milliseconds.</td>
        </tr>
        <tr>
            <td>Avg Time Per Instance (ms)</td>
            <td>Average execution time per sample, in milliseconds.</td>
        </tr>
    </tbody>
</table>

The Benchmark results obtained by running the example programs in Section 2 are as follows:

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

Meanwhile, due to setting `PADDLE_PDX_INFER_BENCHMARK_OUTPUT=./benchmark`, the aforementioned results will be saved locally to: `./benchmark/detail.csv` and `./benchmark/summary.csv`:

The content of `detail.csv` is as follows:

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

The content of `summary.csv` is as follows:

```csv
Iters,Batch Size,Instances,Stage,Avg Time Per Iter (ms),Avg Time Per Instance (ms)
10,2,20,PreProcess,95.80326080,47.90163040
10,2,20,Inference,12.81237602,6.40618801
10,2,20,PostProcess,0.22695065,0.11347532
10,2,20,End2End,108.84258747,54.42129374
```
