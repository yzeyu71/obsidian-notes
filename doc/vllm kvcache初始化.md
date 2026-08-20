# vLLM V1 ModelRunner V2 的 KV Cache 初始化

本文只描述 `use_v2_model_runner=True` 时的执行路径。此时 GPU worker 使用：

```text
vllm/v1/worker/gpu/model_runner.py
```

不要与 V1 runner 的以下路径混淆：

```text
vllm/v1/worker/gpu_model_runner.py
```

## 一、整体调用链

```text
EngineCore._initialize_kv_caches()
    |
    |-- model_executor.get_kv_cache_specs()
    |-- model_executor.determine_available_memory()
    |-- get_kv_cache_configs(...)
    |-- generate_scheduler_kv_cache_config(...)
    |-- model_executor.initialize_from_config(kv_cache_configs)
    |       |
    |       `-- Executor.collective_rpc("initialize_from_config")
    |               |
    |               `-- WorkerWrapperBase.initialize_from_config()
    |                       |
    |                       `-- GPUWorker.initialize_from_config()
    |                               |
    |                               `-- V2 GPUModelRunner.initialize_kv_cache()
    |
    `-- model_executor.compile_or_warm_up_model()
            `-- collective_rpc("compile_or_warm_up_model")
```

入口位于 `vllm/v1/engine/core.py` 的 `EngineCore._initialize_kv_caches()`。它负责规划 KV cache，而不是直接在 engine-core 进程中分配 GPU tensor。

## 二、EngineCore：规划 KV Cache

### 1. 注册并获取 KV cache spec

```python
register_all_kvcache_specs(vllm_config)
kv_cache_specs = self.model_executor.get_kv_cache_specs()
```

`get_kv_cache_specs()` 通过 executor 的 RPC 在 worker 上查询每个 rank 的 KV cache spec。spec 描述模型各层需要的 cache 类型和布局要求，例如 attention、Mamba、block size、head 数、head size、dtype，以及是否为 non-causal attention。

如果检测到 non-causal attention，EngineCore 会关闭依赖 causal attention 假设的 chunked prefill 和 prefix caching，避免调度语义错误。

### 2. 计算可用显存

```python
has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
if has_kv_cache:
    available_gpu_memory = self.model_executor.determine_available_memory()
else:
    available_gpu_memory = [0] * len(kv_cache_specs)
```

在 GPU worker 中，`determine_available_memory()` 通常会：

1. 执行 `model_runner.profile_run()`，用 dummy batch 测量模型执行的峰值显存。
2. 在启用 CUDA graph 且平台支持时调用 `model_runner.profile_cudagraph_memory()`；但当前 V2 `gpu/model_runner.py` 中该方法明确返回 `0`，尚未实现 CUDA graph 显存估算。
3. 因此，当前 V2 路径不会在这一步创建最小 KV cache 并临时捕获 CUDA graph；可用显存主要依据模型 profiling 结果计算。
4. 根据权重、激活和配置的显存利用率，计算可用于正式 KV cache 的显存。

因此，`profile_cudagraph_memory()` 是 `GPUWorker.determine_available_memory()` 中预留的 profiling 接口；在当前 ModelRunner V2 实现中它是 no-op，不应描述成正式 cache 初始化前已经完成了 CUDA graph 显存预留。

### 3. 生成每个 rank 的 KVCacheConfig

```python
kv_cache_configs = get_kv_cache_configs(
    vllm_config,
    kv_cache_specs,
    available_gpu_memory,
)
```

这里返回的是 `list[KVCacheConfig]`，通常每个 worker rank 对应一个配置。配置中包含 KV cache groups、每组的 spec、block 数，以及用于物理分配的 `kv_cache_tensors` 等信息。

如果 auto-fit 修改了 `max_model_len`，EngineCore 会通过 RPC 调用 worker 的 `update_max_model_len()`，使 worker 和 engine-core 使用相同的最终长度。

### 4. 生成 scheduler 视角的配置

```python
scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
```

EngineCore 随后更新 scheduler 需要的 block size 和 cache capacity，并调用 `vllm_config.validate_block_size()`。

`scheduler_kv_cache_config` 最终传给 Scheduler；而每个 rank 的 `KVCacheConfig` 则传给对应的 worker。

## 三、Executor 和 WorkerWrapperBase：分发配置

### 1. Executor 入口

当前方法名是 `initialize_from_config`，不是 `initialize`：

```python
# vllm/v1/executor/abstract.py

def initialize_from_config(
    self,
    kv_cache_configs: list[KVCacheConfig],
) -> None:
    self.collective_rpc(
        "initialize_from_config",
        args=(kv_cache_configs,),
    )
```

`compile_or_warm_up_model()` 也通过 `collective_rpc` 在所有 worker 上执行，并把各 worker 返回的 `CompilationTimes` 汇总回 engine-core。

### 2. WorkerWrapperBase 按 rank 选择配置

```python
# vllm/v1/worker/worker_base.py

def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
    kv_cache_config = kv_cache_configs[self.global_rank]
    with set_current_vllm_config(self.vllm_config):
        self.worker.initialize_from_config(kv_cache_config)
```

因此，Executor 下发的是配置列表；每个 worker 只使用与自身 `global_rank` 对应的 `KVCacheConfig`。

## 四、GPUWorker：进入 ModelRunner V2

当 `vllm_config.use_v2_model_runner` 为 `True` 时，`GPUWorker.init_device()` 实例化：

```python
# vllm/v1/worker/gpu_worker.py

from vllm.v1.worker.gpu.model_runner import (
    GPUModelRunner as GPUModelRunnerV2,
)

self.model_runner = GPUModelRunnerV2(
    self.vllm_config,
    self.device,
)
```

随后，`GPUWorker.initialize_from_config()` 会先更新本地 block 数、初始化 KV transfer，然后调用：

```python
self.model_runner.initialize_kv_cache(kv_cache_config)
```

如果启用了 routed experts 输出，还会初始化 routed experts capturer；如果配置要求 KV cache zeroing，则在 memory pool 上下文之外构建 zeroing metadata。

## 五、ModelRunner V2：正式初始化 KV Cache

V2 的核心入口是：

```python
# vllm/v1/worker/gpu/model_runner.py

def initialize_kv_cache(
    self,
    kv_cache_config: KVCacheConfig,
) -> None:
```

其实际步骤如下。

### 1. 保存配置并复制必要的 cache group

```python
kv_cache_config = deepcopy(kv_cache_config)
self.kv_cache_config = kv_cache_config
self.may_add_encoder_only_layers_to_kv_cache_config()
```

V2 会先复制配置，避免直接修改调用方对象。对于 encoder-only attention，runner 可能把相应层补充到自己的 cache 配置中；这些层不一定出现在 scheduler 的 cache 配置中。

### 2. 初始化 attention backend、metadata 和 block tables

V2 会调用 `init_attn_backend(...)` 初始化 attention groups 和 kernel block sizes，然后按每个 KV cache group 的 spec 创建 `BlockTables`、输入缓冲区和 attention metadata builders。不同于 V1 runner，V2 使用自己的 `InputBuffers`、request state 和 block table 体系，不是简单构造一个 `InputBatch(block_size=...)`。

### 3. 初始化 Mamba/SSM 和 CUDA graph manager

```python
initialize_mamba_ssu_backend(
    self.vllm_config.mamba_config,
    self.kv_cache_config,
)

cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
    ...,
    use_v2_model_runner=True,
    kv_cache_config=self.kv_cache_config,
    max_num_reqs=self.max_num_reqs,
)

self.cudagraph_manager = ModelCudaGraphManager(...)
```

`use_v2_model_runner=True` 会影响 CUDA graph 模式和 capture size 的解析。若启用了 speculative decoding，speculator 也会在这里绑定 attention 和自己的 CUDA graph manager。

### 4. 分配并绑定 KV cache

```python
self.kv_caches = []
kv_caches_dict = init_kv_cache(
    self.kv_caches,
    self.compilation_config.static_forward_context,
    self.kv_cache_config,
    self.attn_groups,
    self.device,
    self.cache_config.cache_dtype,
    self.kernel_block_sizes,
    self.vllm_config,
)
```

V2 通过 `init_kv_cache(...)` 完成正式的 KV cache 物理分配和绑定，而不是调用 V1 runner 的 `_allocate_kv_cache_tensors()`。

具体布局由 `KVCacheConfig`、KV cache groups、attention backend、kernel block size 和 cache dtype 共同决定。不能将它简化为逐层执行：

```python
torch.empty(tensor_spec.shape, ...)
```

当前 V2 使用的配置字段是 `kv_cache_tensors`，实际 cache 初始化由 `init_kv_cache(...)` 和相关 backend 工具完成。

### 5. 注册 KV connector

初始化完成后，V2 使用返回的 cache 映射创建 KV connector：

```python
self.kv_connector = get_kv_connector(
    self.vllm_config,
    kv_caches_dict,
)
```

这使 KV transfer 能够访问已经绑定到 attention 层的 cache。

## 六、ModelRunner V2 的 warmup 和编译

EngineCore 在完成 KV cache 初始化后调用：

```python
if not envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
    self.model_executor.compile_or_warm_up_model()
```

Executor 通过 RPC 调用每个 worker 的 `compile_or_warm_up_model()`。该方法实际定义在 `GPUWorker`，不是 V2 `GPUModelRunner` 的方法。GPUWorker 会根据 compilation 配置调用 V2 runner 的 `_dummy_run()`，执行 `kernel_warmup(self)`，并在非 eager 模式下调用 V2 runner 的 `capture_model()`。具体行为受 compilation mode、cudagraph mode、capture sizes、speculative decoding 和 multimodal 配置影响。

因此，V2 的 warmup 不能直接套用 V1 runner 中 `self.model_runner._dummy_run(size)` 的完整实现说明。

## 七、运行时新 KV block 的清零

正式初始化时，V2 会根据 KV cache 配置准备 zeroing 所需的 metadata。推理过程中，scheduler 将新分配的 block id 放入 scheduler output，V2 runner 在更新请求状态时处理这些 block，清零新 block，避免复用的显存残留旧数据。

这个过程属于 V2 `execute_model()` 的状态更新路径，不应引用 V1 runner 的属性名或伪代码，例如 `self.kv_block_zeroer is not None`。具体 zeroing 实现由 V2 runner 及其 KV cache/block table 工具负责，并与 attention backend 的 cache layout 保持一致。

## 八、Scheduler 衔接和总结

EngineCore 返回 `scheduler_kv_cache_config`。随后 Scheduler 使用该配置创建 KV cache manager 和 block manager；worker 则持有各自 rank 的完整 `KVCacheConfig` 和已经分配好的物理 cache。

在 ModelRunner V2 前提下，初始化可以概括为：

```text
1. Worker 获取 KV cache spec
2. GPUWorker profile 模型显存
3. EngineCore 根据每 rank 的可用显存生成 KVCacheConfig
4. Executor 通过 collective_rpc 分发配置列表
5. WorkerWrapperBase 按 global_rank 选择配置
6. GPUWorker 调用 V2 GPUModelRunner.initialize_kv_cache
7. V2 初始化 attention/backend/block tables/metadata
8. V2 通过 init_kv_cache 分配并绑定物理 KV cache
9. V2 创建 KV connector 和 CUDA graph 管理状态
10. Executor RPC 执行 warmup/compile
11. Scheduler 使用 scheduler_kv_cache_config 管理逻辑 block
```

关键路径是：

```text
vllm/v1/engine/core.py
    -> vllm/v1/executor/abstract.py
    -> vllm/v1/worker/worker_base.py
    -> vllm/v1/worker/gpu_worker.py
    -> vllm/v1/worker/gpu/model_runner.py
```
