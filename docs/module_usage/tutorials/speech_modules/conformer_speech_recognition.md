---
comments: true
---

# 自动语音识别模块使用教程（流式Chunk Conformer）

## 一、概述
Chunk Conformer是专为实时语音处理优化的流式语音识别模型，主要特性包括：
- 基于分块处理的低延迟架构
- 滑动窗口注意力机制
- 通过`chunk_conformer_batch_sampler`实现动态批处理
- 使用`chunk_conformer_reader`进行音频分块预处理

## 二、支持模型列表

### Chunk Conformer模型
模型 | 训练数据 | 延迟（CPU/GPU） | 模型大小 | 词错率 | 下载链接
:-----------: | :-----:| :-------: | :-----: | :-----: |:---------:
Large | 50千小时语音 | 300ms/50ms | 1.2GB | [待更新] | [chunk_conformer_large](#)
Medium | 50千小时语音 | 200ms/30ms | 680MB | [待更新] | [chunk_conformer_medium](#)

### Whisper Model
<table>
  <tr>
    <th >模型</th>
    <th >模型下载链接</th>
    <th >训练数据</th>
    <th >模型大小</th>
    <th >词错率</th>
    <th >介绍</th>
  </tr>
  <tr>
    <td>whisper_large</td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_large.tar">whisper_large</a></td>
    <td >680kh</td>
    <td>5.8G</td>
    <td>2.7 (Librispeech)</td>
    <td rowspan="5">Whisper 是 OpenAI 开发的多语言自动语音识别模型，具备高精度和鲁棒性。它采用端到端架构，能处理嘈杂环境音频，适用于语音助理、实时字幕等多种应用。</td>
  </tr>
  <tr>
    <td>whisper_medium</td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_medium.tar">whisper_medium</a></td>
    <td>680kh</td>
    <td>2.9G</td>
    <td>-</td>
  </tr>
  <tr>
    <td>whisper_small</td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_small.tar">whisper_small</a></td>
    <td>680kh</td>
    <td>923M</td>
    <td>-</td>
  </tr>
  <tr>
    <td>whisper_base</td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_base.tar">whisper_base</a></td>
    <td>680kh</td>
    <td>277M</td>
    <td>-</td>
  </tr>
  <tr>
    <td>whisper_small</td>
    <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/whisper_tiny.tar">whisper_tiny</a></td>
    <td>680kh</td>
    <td>145M</td>
    <td>-</td>
  </tr>
</table>

## 三、快速集成

### Whisper 模型集成示例
```python
from paddlex import create_model
model = create_model(model_name="whisper_large")
output = model.predict(input="./zh.wav", batch_size=1)
for res in output:
    res.print()
    res.save_to_json(save_path="./output/res.json")
```

### Chunk Conformer 集成示例
```python
from paddlex import create_model
model = create_model(model_name="chunk_conformer_large")
output = model.predict(
    input="./stream.wav",
    batch_size=4,  # 使用chunk_conformer_batch_sampler
    streaming=True,
    chunk_length=30,  # 处理窗口时长（秒）
    overlap=0.5      # 窗口重叠比例
)
```

## 四、自定义开发
### 4.1 数据准备
```bash
wget https://paddlespeech.bj.bcebos.com/PaddleAudio/stream_sample.wav
```

### 4.2 训练配置
```python
from paddlex import create_model

model = create_model(
    model_name="chunk_conformer_medium",
    num_classes=5000,  # 音素类别数
    chunk_size=30,    # 训练时块大小（秒）
    training=True
)

model.set_train_config(
    batch_size=16,
    learning_rate=0.001,
    optimizer="AdamW",
    warmup_steps=1000
)
```

### 4.3 评估指标
```python
# 计算词错率（WER）
wer = model.evaluate(
    test_dataset,
    metrics="wer",
    batch_size=8,
    num_workers=4
)

print(f"词错率: {wer*100:.2f}%")
```

### 4.4 模型推理
```python
from paddlex.inference.common import ChunkConformerReader

reader = ChunkConformerReader(
    chunk_size=30,
    sample_rate=16000,
    overlap=0.5
)
