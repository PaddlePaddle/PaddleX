# Model Inference Benchmark

## Table of Contents

- [1. Instructions](#1.-Instructions)
- [2. Usage Examples](#2.-Usage-Examples)
  - [2.1 Command Line Method](#2.1-Command-Line-Method)
  - [2.2 Python Script Method](#2.2-Python-Script-Method)
- [3. Explanation of Results](#3.-Explanation-of-Results)

## 1. Instructions

The benchmark feature collects the average execution time per iteration for each operation in the end-to-end model inference process as well as the average execution time per instance, and provides summary information. The time measurements are in milliseconds.

To enable the benchmark feature, you must set the following environment variables:

* `PADDLE_PDX_INFER_BENCHMARK`: When set to `True`, the benchmark feature is enabled (default is `False`).
* `PADDLE_PDX_INFER_BENCHMARK_WARMUP`: The number of warm-up iterations before testing (default is `0`).
* `PADDLE_PDX_INFER_BENCHMARK_ITERS`: The number of iterations for testing (default is `0`).
* `PADDLE_PDX_INFER_BENCHMARK_OUTPUT_DIR`: The directory where the metrics are saved (e.g., `./benchmark`). The default is `None`, meaning the benchmark metrics will not be saved.
* `PADDLE_PDX_INFER_BENCHMARK_USE_CACHE_FOR_READ`: When set to `True`, the caching mechanism is applied to the operation of reading input data to avoid repetitive I/O overhead, and the time consumed by data read and cache is not recorded in the core time (default is `False`).

**Note**:

* At least one of `PADDLE_PDX_INFER_BENCHMARK_WARMUP` or `PADDLE_PDX_INFER_BENCHMARK_ITERS` must be set to a value greater than zero; otherwise, the benchmark feature cannot be used.
* For the pipeline inference benchmark feature, refer to [Pipeline Benchmark](../../pipeline_usage/instructions/benchmark.en.md).

## 2. Usage Examples

You can use the benchmark feature by either the command line method or the Python script method.

### 2.1 Command Line Method

**Note**:

- For a description of the input parameters, please refer to the [PaddleX Common Model Configuration File Parameter Explanation](./config_parameters_common.en.md).
- If `batch_size` is greater than 1, the input data will be duplicated `batch_size` times to match the size of `batch_size`.

Execute the command:

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

### 2.2 Python Script Method

**Note**:

- For a description of the input parameters, please refer to the [PaddleX Single Model Python Usage Instructions](./model_python_API.en.md).
- If `batch_size` is greater than 1, the input data will be duplicated `batch_size` times to match the size of `batch_size`.

Create the script `test_infer.py`:

```python
from paddlex import create_model

model = create_model(model_name="PicoDet-XS", model_dir=None)
output = list(model.predict(input="./test.png", batch_size=2))
```

Run the script:

```bash
PADDLE_PDX_INFER_BENCHMARK=True \
PADDLE_PDX_INFER_BENCHMARK_WARMUP=5 \
PADDLE_PDX_INFER_BENCHMARK_ITERS=10 \
PADDLE_PDX_INFER_BENCHMARK_OUTPUT_DIR=./benchmark \
python test_infer.py
```

## 3. Explanation of Results

After enabling the benchmark feature, the benchmark results will be automatically printed. The details are as follows:

<table border="1">
    <thead>
        <tr>
            <th>Field Name</th>
            <th>Field Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Iters</td>
            <td>Number of iterations, i.e., the number of times inference is executed in a loop.</td>
        </tr>
        <tr>
            <td>Batch Size</td>
            <td>Batch size, i.e., the number of instances processed in each iteration.</td>
        </tr>
        <tr>
            <td>Instances</td>
            <td>Total number of instances, calculated as <code>Iters</code> multiplied by <code>Batch Size</code>.</td>
        </tr>
        <tr>
            <td>Operation</td>
            <td>Name of the operation, such as <code>Resize</code>, <code>Normalize</code>, etc.</td>
        </tr>
        <tr>
            <td>Type</td>
            <td>Type of time consumption, including:
            <ul>
            <li>preprocessing time (<code>Preprocessing</code>)</li>
            <li>model inference time (<code>Inference</code>)</li>
            <li>postprocessing time (<code>Postprocessing</code>)</li>
            <li>core time (<code>Core</code>, i.e., Preprocessing + Inference + Postprocessing)</li>
            <li>other time (<code>Other</code>, e.g., the time taken to run the code that orchestrates the operations, and the extra overhead introduced by the benchmark feature)</li>
            <li>end-to-end time (<code>End-to-End</code>, i.e., Core + Other)</li>
            </ul>
            </td>
        </tr>
        <tr>
            <td>Avg Time Per Iter (ms)</td>
            <td>Average execution time per iteration, in milliseconds.</td>
        </tr>
        <tr>
            <td>Avg Time Per Instance (ms)</td>
            <td>Average execution time per instance, in milliseconds.</td>
        </tr>
    </tbody>
</table>

Below is an example of the benchmark results obtained by running the example program in Section 2:

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

Additionally, since `PADDLE_PDX_INFER_BENCHMARK_OUTPUT_DIR=./benchmark` is set, the above results will be saved locally in `./benchmark/detail.csv` and `./benchmark/summary.csv`.

The contents of `detail.csv` are as follows:

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

The contents of `summary.csv` are as follows:

```csv
Iters,Batch Size,Instances,Type,Avg Time Per Iter (ms),Avg Time Per Instance (ms)
10,2,20,Preprocessing,95.32786242,47.66393121
10,2,20,Inference,10.65373025,5.32686512
10,2,20,Postprocessing,0.18665448,0.09332724
10,2,20,Core,106.16824715,53.08412358
10,2,20,Other,2.74794563,1.37397281
10,2,20,End-to-End,108.91619278,54.45809639
```
