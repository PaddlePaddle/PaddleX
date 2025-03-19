---
comments: true
---

# Automatic Speech Recognition Module Tutorial (Chunk Conformer)

## I. Overview
Chunk Conformer is a streaming speech recognition model optimized for real-time processing with these key features:
- Low-latency chunk-based processing architecture
- Sliding window attention mechanism
- Dynamic batch sizing via `chunk_conformer_batch_sampler`
- Audio chunk preprocessing from `chunk_conformer_reader`

## II. Supported Model List

### Chunk Conformer Models
Model | Training Data | Latency (CPU/GPU) | Size | WER | Download
:-----------: | :-----:| :-------: | :-----: | :-----: |:---------:
Large | 50kh speech | 300ms/50ms | 1.2GB | [Pending] | [chunk_conformer_large](https://paddlespeech.bj.bcebos.com/conformer/chunk_conformer_large-model.tar.gz)
Medium | 50kh speech | 200ms/30ms | 680MB | [Pending] | [chunk_conformer_medium](https://paddlespeech.bj.bcebos.com/conformer/chunk_conformer_medium-model.tar.gz)

## III. Quick Integration
Before quick integration, you need to install the PaddleX wheel package. For the installation method, please refer to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

```python
from paddlex import create_model

# Initialize model with hardware acceleration
model = create_model(
    model_name="chunk_conformer_large",
    device="npu:0"  # Supported devices: npu, mlu, xpu, dcu, gcu
)

# Perform streaming recognition
output = model.predict(
    input="./stream.wav",
    device="npu:0",
    batch_size=4,  # Uses chunk_conformer_batch_sampler
    streaming=True,
    chunk_length=30,  # Seconds per processing window
    overlap=0.5      # Window overlap ratio
)

# Process results
for res in output:
    res.print(json_format=False)
    res.save_to_json(save_path="./output/res.json")
```

### API Reference
#### Model Initialization
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>Model architecture</td>
<td><code>str</code></td>
<td><code>chunk_conformer_large, chunk_conformer_medium</code></td>
<td>Required</td>
</tr>
<tr>
<td><code>device</code></td>
<td>Hardware accelerator</td>
<td><code>str</code></td>
<td><code>npu:0, mlu:0, xpu:0, dcu:0, gcu:0</code></td>
<td><code>cpu</code></td>
</tr>
</table>

#### Prediction Parameters
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Options</th>
<th>Default</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>Input audio path/URL</td>
<td><code>str</code></td>
<td>Local path or web URL</td>
<td>Required</td>
</tr>
<tr>
<td><code>streaming</code></td>
<td>Enable streaming mode</td>
<td><code>bool</code></td>
<td><code>True/False</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>chunk_length</code></td>
<td>Processing window size</td>
<td><code>int</code></td>
<td>10-60 seconds</td>
<td>30</td>
</tr>
<tr>
<td><code>overlap</code></td>
<td>Window overlap ratio</td>
<td><code>float</code></td>
<td>0.0-0.9</td>
<td>0.5</td>
</tr>
</table>

#### Result Handling Methods
<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameters</th>
</tr>
</thead>
<tr>
<td><code>print()</code></td>
<td>Print formatted results</td>
<td><code>json_format</code>: Enable JSON formatting</td>
</tr>
<tr>
<td><code>save_to_json()</code></td>
<td>Save results to file</td>
<td><code>save_path</code>: Output file path</td>
</tr>
</table>
---
comments: true
---

# Automatic Speech Recognition Module Tutorial (Chunk Conformer)

## I. Overview
Chunk Conformer is a streaming speech recognition model optimized for real-time processing with these key features:
- Low-latency chunk-based processing architecture
- Sliding window attention mechanism
- Dynamic batch sizing via `chunk_conformer_batch_sampler`
- Audio chunk preprocessing from `chunk_conformer_reader`

## II. Supported Model List

### Chunk Conformer Models
Model | Training Data | Latency (CPU/GPU) | Size | WER | Download
:-----------: | :-----:| :-------: | :-----: | :-----: |:---------:
Large | 50kh speech | 300ms/50ms | 1.2GB | [Pending] | [chunk_conformer_large](#)
Medium | 50kh speech | 200ms/30ms | 680MB | [Pending] | [chunk_conformer_medium](#)

## III. Quick Integration
```python
from paddlex import create_model
model = create_model(model_name="chunk_conformer_large", device="npu:0")  # Supported devices: npu, mlu, xpu, dcu, gcu
output = model.predict(
    input="./stream.wav",
    device="npu:0",  # Supported devices: npu, mlu, xpu, dcu, gcu
    batch_size=4,  # Uses chunk_conformer_batch_sampler
    streaming=True,
    chunk_length=30,  # Seconds per processing window
    overlap=0.5      # Window overlap ratio
)
```

## IV. Hardware Support
PaddleX Speech Recognition supports multiple AI accelerators. To use specific hardware:

1. First install corresponding PaddlePaddle version:
   - [Ascend NPU Installation Guide](../../other_devices_support/paddlepaddle_install_NPU.en.md)
   - [Cambricon MLU Installation Guide](../../other_devices_support/paddlepaddle_install_MLU.en.md)
   - [Kunlun XPU Installation Guide](../../other_devices_support/paddlepaddle_install_XPU.en.md)
   - [Hygon DCU Installation Guide](../../other_devices_support/paddlepaddle_install_DCU.en.md)
   - [Enflame GCU Installation Guide](../../other_devices_support/paddlepaddle_install_GCU.en.md)

2. Specify device in code:
```python
# Python API
model = create_model(..., device="npu:0")  # npu, mlu, xpu, dcu or gcu

# Command Line
paddlex --module automatic_speech_recognition \
        --model chunk_conformer_large \
        --input ./stream.wav \
        --device npu:0
```

## V. Custom Development
### 5.1 Data Preparation
```bash
wget https://paddlespeech.bj.bcebos.com/PaddleAudio/stream_sample.wav
```

### 5.2 Training Configuration
```python
from paddlex import create_model

model = create_model(
    model_name="chunk_conformer_medium",
    num_classes=5000,  # Phoneme classes
    chunk_size=30,    # Training chunk size (seconds)
    training=True
)

model.set_train_config(
    batch_size=16,
    learning_rate=0.001,
    optimizer="AdamW",
    warmup_steps=1000
)

### 5.3 Evaluation Metrics
```python
# Calculate Word Error Rate (WER)
wer = model.evaluate(
    test_dataset,
    metrics="wer",
    batch_size=8,
    num_workers=4
)

print(f"Word Error Rate: {wer*100:.2f}%")
```

### 5.4 Model Inference
```python
from paddlex.inference.common import ChunkConformerReader

reader = ChunkConformerReader(
    chunk_size=30,
    sample_rate=16000,
    overlap=0.5,
    device="npu:0"  # Supported devices: npu, mlu, xpu, dcu, gcu
)
