# DocVLM 异步请求取消机制缺失问题

## 问题概述

在 `DocVLMPredictor` 的两阶段推理模式中，当批量子图被发送给远程 VLM 服务处理时，如果其中某个请求失败，其他已提交的请求不会被取消，导致 VLM 服务继续处理这些最终会被丢弃的请求。

## 影响范围

- **文件**: `paddlex/inference/models/doc_vlm/predictor.py`
- **方法**: `_genai_client_process`
- **场景**: 使用远程 VLM 服务（`genai_config.backend` 为 `fastdeploy-server`、`vllm-server`、`sglang-server` 或 `mlx-vlm-server`）

## 问题分析

### 当前实现

```python
def _genai_client_process(self, data, ...):
    futures = []
    for item in data:
        # ... 构建请求 ...
        future = self._genai_client.create_chat_completion(
            [...],
            return_future=True,
            timeout=600,
            **kwargs,
        )
        futures.append(future)

    # 顺序收集结果
    results = []
    for future in futures:
        result = future.result()  # <-- 异常点
        results.append(result.choices[0].message.content)

    return results
```

### 问题说明

1. **第一个循环**：所有请求被并发提交给 VLM 服务，返回 `concurrent.futures.Future` 对象列表
2. **第二个循环**：顺序等待每个 future 的结果
3. **异常场景**：如果第 N 个 `future.result()` 抛出异常：
   - 循环立即中断
   - 第 N+1 到最后的 futures **没有被取消**
   - VLM 服务继续处理这些请求
   - 上层捕获异常后程序继续运行，但这些请求的结果被丢弃

### 调用链路

```
DocVLMPredictor._genai_client_process
  └── GenAIClient.create_chat_completion(return_future=True)
        └── run_async(..., return_future=True)
              └── _AsyncThreadManager.run_async(coro)
                    └── asyncio.run_coroutine_threadsafe(coro, loop)
                          └── 返回 concurrent.futures.Future
```

## 影响

- **资源浪费**：VLM 服务（通常是 GPU 资源）继续处理无用请求
- **延迟累积**：在长期运行的服务中，无用请求可能占用处理队列
- **成本增加**：如果使用按请求计费的服务，会产生额外费用

## 建议修复

```python
def _genai_client_process(self, data, ...):
    futures = []
    try:
        for item in data:
            # ... 图片处理和参数构建 ...
            future = self._genai_client.create_chat_completion(
                [...],
                return_future=True,
                timeout=600,
                **kwargs,
            )
            futures.append(future)

        results = []
        for future in futures:
            result = future.result()
            results.append(result.choices[0].message.content)
        return results
    except Exception:
        # 取消所有未完成的 futures
        for future in futures:
            if not future.done():
                future.cancel()
        raise
```

## 注意事项

1. `concurrent.futures.Future.cancel()` 只能取消**尚未开始执行**的任务
2. 对于已经在执行中的 HTTP 请求，`cancel()` 不会中断它们
3. 真正的请求取消需要 HTTP 客户端和服务端的配合支持
4. 当前修复可以防止队列中等待的请求继续被调度，符合资源清理的最佳实践

## 相关文件

- `paddlex/inference/models/doc_vlm/predictor.py` - 问题所在
- `paddlex/inference/models/common/genai.py` - 异步管理器和 GenAI 客户端实现
- `paddlex/inference/models/base/predictor/base_predictor.py` - 基类定义
