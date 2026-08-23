# vLLM V1 ModelRunner V2 的 KV Cache 初始化

本文只描述 `use_v2_model_runner=True` 时的执行路径。此时 GPU worker 使用：

```text
vllm/v1/worker/gpu/model_runner.py
```

不要与 V1 runner 的以下路径混淆：

```text
vllm/v1/worker/gpu_model_runner.py
```

## KV Cache 的作用

### 1. 为什么需要 KV Cache

Transformer 的自回归解码是一个「逐步生成、逐 token 计算」的过程。第 $t$ 个 token 计算 attention 时，需要和它之前的所有 token（包括 prompt 与已生成的 token）做交互。attention 的 Q/K/V 计算中：

- **Query（Q）** 只依赖当前正在计算的 token，每步都变；
- **Key（K）/ Value（V）** 依赖历史 token 的内容，一旦某个历史 token 计算过就不会再变。

如果不做缓存，每生成一个新 token 都要把整个前缀重新送入模型，重算所有历史 token 的 K 和 V，计算量随序列长度呈 $O(n^2)$ 增长。KV Cache 的思想是：**把已经计算过的历史 token 的 K/V 保存在显存里，每步只计算新 token 的 K/V，再从缓存里读取其余部分**。这样每个 decode 步的计算量降为 $O(n)$。

### 2. KV Cache 是显存消耗的大头

KV Cache 的大小与多个维度成正比：

```text
总字节数 ≈ 层数
           × (K + V)
           × num_kv_heads
           × head_size
           × 最大序列长度（或并发请求的总 token 数）
           × dtype 字节数
```

长上下文、大 batch、大模型场景下，KV Cache 往往超过模型权重本身，成为显存的第一消耗项。因此它不能简单用「分配一大块 tensor」的方式处理，而需要：

- 显存预算（profiling 决定能分多少给 KV Cache）；
- 按 block 的细粒度分配与回收（避免碎片化和浪费）；
- 跨请求复用（prefix cache）。

### 3. vLLM 用「分页」思路管理 KV Cache

vLLM 把 KV Cache 切成固定大小的 **block**（类似操作系统的内存页），每个请求持有一张 **block table**（逻辑 token 位置 → 物理 block 的映射）。好处是：

- 请求按需申请 block，用完即释放；
- 不同的请求可以复用同一批物理 block（prefix cache 命中时直接共享，写入时再 copy-on-write）；
- 滑动窗口、Mamba 等特殊 attention 类型可以在 block 粒度上「跳过 / 回收」不再需要的部分。

### 4. 与本文的关系

本文描述 vLLM V1（`use_v2_model_runner=True`，即 `vllm/v1/worker/gpu/model_runner.py`）中 KV Cache 的完整生命周期：

1. **初始化**：探测显存 → 生成每个 rank 的 `KVCacheConfig` → 下发到 worker → 分配并绑定物理 KV tensor（本文一～六章）。
2. **运行时管理**：调度阶段做 prefix 命中、分配/回收 block、零化与 copy-on-write（本文第九章）。

理解 KV Cache 的上述作用后，再读后面的初始化和调度代码会更清楚每步在解决什么问题。

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

## 二、EngineCore：规划 KV Cache（细化）

入口 `EngineCore._initialize_kv_caches()`（`vllm/v1/engine/core.py`）的真实实现如下，每一步在下方展开：

```python
@instrument(span_name="Prepare model")
def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
    start = time.time()

    # 1) 在 engine-core 进程注册所有内置 KV cache spec 类型
    register_all_kvcache_specs(vllm_config)

    # 2) 通过 executor RPC 从每个 worker 收集该 rank 的 KV cache spec
    kv_cache_specs = self.model_executor.get_kv_cache_specs()

    # 3) non-causal 层会破坏 chunked prefill / prefix caching 的 causal 假设
    if any(getattr(spec, "non_causal", False)
           for worker_specs in kv_cache_specs
           for spec in worker_specs.values()):
        vllm_config.scheduler_config.enable_chunked_prefill = False
        vllm_config.cache_config.enable_prefix_caching = False

    # 4) 计算可用于 KV cache 的显存
    has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
    if has_kv_cache:
        available_gpu_memory = self.model_executor.determine_available_memory()
        self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
    else:
        available_gpu_memory = [0] * len(kv_cache_specs)

    # 5) 生成每个 rank 的 KVCacheConfig（可能触发 auto-fit max_model_len）
    max_model_len_before = vllm_config.model_config.max_model_len
    kv_cache_configs = get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_gpu_memory)
    max_model_len_after = vllm_config.model_config.max_model_len
    if max_model_len_after != max_model_len_before:
        self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

    # 6) 生成 scheduler 视角的配置
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
    vllm_config.validate_block_size()

    # 7) 下发配置给 worker，真正分配 KV cache
    self.model_executor.initialize_from_config(kv_cache_configs)

    # 8) 编译 / warmup（EP 弹性扩容场景跳过）
    if not envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
        self.model_executor.compile_or_warm_up_model()

    # 9) 计时与日志
    ...
    return scheduler_kv_cache_config
```

### 1. 注册 spec 类型：`register_all_kvcache_specs`

`vllm/v1/core/single_type_kv_cache_manager.py:1897` 把每个 spec 类绑定到它的 manager 类，并声明 uniform 基类：

```python
def register_all_kvcache_specs(vllm_config):
    KVCacheSpecRegistry.register(FullAttentionSpec, FullAttentionManager,
                                 uniform_type_base_spec=FullAttentionSpec)
    KVCacheSpecRegistry.register(SlidingWindowSpec, SlidingWindowManager,
                                 uniform_type_base_spec=SlidingWindowSpec)
    KVCacheSpecRegistry.register(SlidingWindowMLASpec, SlidingWindowManager,
                                 uniform_type_base_spec=SlidingWindowMLASpec)
    KVCacheSpecRegistry.register(MambaSpec, MambaManager,
                                 uniform_type_base_spec=MambaSpec)
    KVCacheSpecRegistry.register(ChunkedLocalAttentionSpec,
                                 ChunkedLocalAttentionManager, ...)
    KVCacheSpecRegistry.register(CrossAttentionSpec, CrossAttentionManager, ...)
    # FullAttentionSpec 的子类统一归入 FullAttentionManager
    KVCacheSpecRegistry.register(MLAAttentionSpec, FullAttentionManager,
                                 uniform_type_base_spec=FullAttentionSpec)
    KVCacheSpecRegistry.register(RSWASpec, RSWAManager,
                                 uniform_type_base_spec=FullAttentionSpec)
    KVCacheSpecRegistry.register(HiddenStateCacheSpec, FullAttentionManager,
                                 uniform_type_base_spec=FullAttentionSpec)
    KVCacheSpecRegistry.register(SinkFullAttentionSpec, SinkFullAttentionManager,
                                 uniform_type_base_spec=FullAttentionSpec)
    current_platform.register_custom_kv_cache_specs(vllm_config)
```

作用：在 engine-core 进程内建立「spec 类 → manager 类」查找表，保证后续分组、校验和调度时都能找到对应 manager。

### 2. 收集 spec：`get_kv_cache_specs`

`vllm/v1/executor/abstract.py`：

```python
def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
    return self.collective_rpc("get_kv_cache_spec")
```

V2 runner 的 `get_kv_cache_spec()`（`gpu/model_runner.py`）对 encoder-only 返回 `{}`，否则调用 `get_kv_cache_spec(self.vllm_config)`。每个 worker 返回 `{layer_name: KVCacheSpec}`。

### 3. non-causal 修正

```python
if any(getattr(spec, "non_causal", False)
       for worker_specs in kv_cache_specs for spec in worker_specs.values()):
    vllm_config.scheduler_config.enable_chunked_prefill = False
    vllm_config.cache_config.enable_prefix_caching = False
```

### 4. 显存探测：`determine_available_memory`

`GPUWorker.determine_available_memory()`（`gpu_worker.py:484`）核心：

```python
@torch.inference_mode()
def determine_available_memory(self) -> int:
    if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
        self.model_runner.profile_run()          # 用户显式指定 KV cache 大小
        return reserve_mm_ipc_gpu_memory(kv_cache_memory_bytes, ...)

    with memory_profiling(self.init_snapshot, ...) as profile_result:
        self.model_runner.profile_run()          # dummy batch 测峰值激活

    cudagraph_memory_estimate = 0
    if (current_platform.is_cuda_alike()
            and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE):
        cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()

    self.available_kv_cache_memory_bytes = (
        self.requested_memory - profile_result.non_kv_cache_memory
        - cudagraph_memory_estimate_applied)
    return reserve_mm_ipc_gpu_memory(self.available_kv_cache_memory_bytes, ...)
```

注意 V2 中 `gpu/model_runner.py` 的 `profile_cudagraph_memory()` 当前是 no-op：

```python
def profile_cudagraph_memory(self) -> int:
    # NOTE(woosuk): It is TBD whether we keep this API or not.
    return 0
```

因此 V2 不会在初始化阶段创建最小 KV cache 并临时捕获 CUDA graph 来估算显存。

### 5. 生成每 rank 配置：`get_kv_cache_configs`

`vllm/v1/core/kv_cache_utils.py:2088` 的真实流程：

```python
def get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
    # (a) 合并所有 worker 的 spec（PP 各 stage 层名不同，TP 同 stage 应一致）
    merged_kv_cache_specs = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec

    # (b) 校验所有 spec 都已注册
    KVCacheSpecRegistry.check_kv_cache_spec_registry(merged_kv_cache_specs)

    # (c) 多模块 MTP 时给 SlidingWindowSpec 标记 extra_retained_tokens
    ...

    # (d) 生成全局 KV cache groups（可能原地修改 merged_kv_cache_specs）
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # (e) 把全局 groups 投影到每个 worker（PP 分片视角）
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, ws)
        for ws in kv_cache_specs]

    # (f) num_gpu_blocks_override 时换算 effective memory
    ...

    # (g) auto-fit max_model_len（original_max_model_len == -1 时）
    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(vllm_config, projected_groups_per_worker, available_memory)

    # (h) 逐 worker 检查显存是否足够
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        _check_enough_kv_cache_memory(avail_mem, ...)

    # (i) 逐 worker 生成 KVCacheConfig
    kv_cache_configs = [
        get_kv_cache_config_from_groups(vllm_config, groups, mem)
        for groups, mem in zip(projected_groups_per_worker, available_memory)]

    # (j) 所有 rank 取最小 num_blocks，并按比例缩小 tensor size
    min_num_blocks = min(cfg.num_blocks for cfg in kv_cache_configs)
    for cfg in kv_cache_configs:
        num_blocks_old = cfg.num_blocks
        cfg.num_blocks = min_num_blocks
        for t in cfg.kv_cache_tensors:
            t.size = t.size // num_blocks_old * min_num_blocks

    return kv_cache_configs
```

关键点：合并 spec → 分组 → 按 PP 投影 → auto-fit / 校验 → 每 rank 独立建配置 → 收敛到最小 `num_blocks`，保证所有 worker 一致。

#### 5a. 层分组：`get_kv_cache_groups`

`kv_cache_utils.py:1747` 分支顺序：

```python
def get_kv_cache_groups(vllm_config, kv_cache_spec):
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)      # 强制统一为单一 spec 类型
    if is_kv_cache_type_attention_free(kv_cache_spec):
        return []                                        # 无 attention 模型
    if is_kv_cache_spec_uniform(kv_cache_spec):
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)   # 所有层同 spec
    if (uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec)):
        return _get_kv_cache_groups_uniform_type(uniform_spec)    # 同类型不同 hidden size
    if (grouped_specs := group_and_unify_kv_cache_specs(kv_cache_spec)):
        # DeepSeekV4：full attention + 多种 sliding window，拆成多个 UniformTypeKVCacheSpecs
        ...
    # 其它：抽出 HiddenStateCacheSpec，再按 page size 统一后分组
    ...
    return groups
```

#### 5b. 物理布局：`get_kv_cache_config_from_groups`

`kv_cache_utils.py:1327` 决定 `num_blocks` 与 `kv_cache_tensors`：

```python
def get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory):
    if len(kv_cache_groups) == 0:
        # 无 attention 模型：保留一个 null block
        return KVCacheConfig(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[])

    if len(kv_cache_groups) == 1 and isinstance(
            kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs):
        # 同类型不同 hidden size：每层按各自 page size 分配
        num_blocks = available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        kv_cache_tensors = [
            KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=[layer_name])
            ...]
    elif _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        # DeepSeek V4 默认 packed 布局 / --enable-cross-layers
        num_blocks, kv_cache_tensors = _get_kv_cache_config_packed(...)
    else:
        # 通用多 group：group_size 个内存池，每个池被各 group 的同序号层共享
        group_size = max(len(g.layer_names) for g in kv_cache_groups)
        page_size = get_uniform_page_size([g.kv_cache_spec for g in kv_cache_groups])
        num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)
        for i in range(group_size):
            shared_by = [g.layer_names[i] for g in kv_cache_groups
                         if i < len(g.layer_names)]
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by))

    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=kv_cache_tensors,
                         kv_cache_groups=kv_cache_groups, ...)
```

说明：`num_blocks` 是逻辑 block 数；`kv_cache_tensors` 描述物理 tensor 如何分配、哪些层共享一个 tensor。

### 6. scheduler 配置与 block size

`generate_scheduler_kv_cache_config`（`kv_cache_utils.py:1832`）：

```python
def generate_scheduler_kv_cache_config(kv_cache_configs):
    assert all(cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs)
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values()))
    return cfg
```

`resolve_kv_cache_block_sizes`（`kv_cache_utils.py:607`）计算两个粒度：

```python
def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
    # scheduler_block_size：调度对齐粒度
    #   单 group = cache_config.block_size * dcp
    #   多 group = 各 group 有效 block size 的 LCM
    # hash_block_size：prefix 哈希粒度
    #   单 group = scheduler block size
    #   多 group = GCD（或 prefix_match_unit override）
    ...
```

随后 EngineCore 用 `scheduler_block_size` / `hash_block_size` 创建 Scheduler，Scheduler 内部再据此构建 `KVCacheManager`。

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

当 `vllm_config.use_v2_model_runner` 为 `True` 时，`GPUWorker.init_device()`（`gpu_worker.py`）实例化：

```python
from vllm.v1.worker.gpu.model_runner import (
    GPUModelRunner as GPUModelRunnerV2,
)

self.model_runner = GPUModelRunnerV2(
    self.vllm_config,
    self.device,
)
```

`GPUWorker.initialize_from_config()`（`gpu_worker.py:665`）：

```python
@instrument(span_name="Allocate KV cache")
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    # 同步 block 数到本地 cache_config
    self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    # 先于分配初始化 KV transfer connector
    ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
    with self._maybe_get_memory_pool_context(tag="kv_cache"):
        self.model_runner.initialize_kv_cache(kv_cache_config)
    if self.model_config.enable_return_routed_experts:
        self.model_runner.init_routed_experts_capturer()
    # 在 CuMem 池外构建 zeroing 元数据
    if kv_cache_config.needs_kv_cache_zeroing and hasattr(
            self.model_runner, "_init_kv_zero_meta"):
        self.model_runner._init_kv_zero_meta()
```

## 五、ModelRunner V2：正式初始化 KV Cache

V2 的核心入口是 `gpu/model_runner.py:500` 的 `initialize_kv_cache`，完整实现如下：

```python
def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
    kv_cache_config = deepcopy(kv_cache_config)   # 避免修改调用方对象
    self.kv_cache_config = kv_cache_config

    # (1) encoder-decoder 时 block table 需覆盖 encoder 长度
    block_table_max_model_len = self.max_model_len
    if self.is_encoder_decoder:
        block_table_max_model_len = max(
            block_table_max_model_len,
            self.scheduler_config.max_num_encoder_input_tokens,
            getattr(self.model_config.hf_config, "max_source_positions", 0))

    # (2) 逐 group 计算 block size 与 block table 行宽
    block_sizes, max_num_blocks_per_group = [], []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        block_sizes.append(spec.block_size)
        max_num_blocks = spec.max_num_blocks_per_req(
            self.vllm_config, block_table_max_model_len)
        if isinstance(spec, MambaSpec):
            max_num_blocks = get_block_table_width(max_num_blocks, spec.block_size,
                                                   token_alignment=None)
        else:
            max_num_blocks = get_block_table_width(max_num_blocks, spec.block_size)
        max_num_blocks_per_group.append(max_num_blocks)

    # (3) 初始化 attention groups、backend 与 kernel block size
    self.attn_groups, attn_cg_support, self.kernel_block_sizes = init_attn_backend(
        self.kv_cache_config, self.vllm_config, self.device)

    # (4) 自适应验证（spec decode 相关）
    self.adaptive_verification = maybe_create_adaptive_verification_manager(...)

    # (5) 创建 block table / PCP manager
    self.block_tables = BlockTables(block_sizes=block_sizes,
                                    max_num_blocks_per_group=max_num_blocks_per_group, ...)
    self.pcp_manager = pcp.maybe_build_pcp_manager(...)

    # (6) Mamba/SSM backend
    initialize_mamba_ssu_backend(self.vllm_config.mamba_config, self.kv_cache_config)

    # (7) 解析 CUDA graph 模式
    cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
        ..., use_v2_model_runner=True, kv_cache_config=self.kv_cache_config, ...)
    self.cudagraph_manager = ModelCudaGraphManager(...)

    # (8) speculator 绑定 attention 与自己的 cudagraph manager
    ...

    # (9) 物理分配并绑定
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

    # (10) 创建 KV connector
    self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)
```

`init_kv_cache`（`gpu/attn_utils.py:545`）内部三步：

```python
def init_kv_cache(runner_kv_caches, forward_context, kv_cache_config,
                  attn_groups, device, cache_dtype, kernel_block_sizes, vllm_config):
    shared_kv_cache_layers = get_shared_kv_cache_layers(vllm_config)
    # 1. 按 kv_cache_tensors 分配原始内存
    kv_cache_raw_tensors = _allocate_kv_cache(
        kv_cache_config, shared_kv_cache_layers, device)
    # 2. 按 attention backend 将原始内存 reshape 成各层 KV cache 视图
    kv_caches = _reshape_kv_cache(
        attn_groups=..., kv_cache_raw_tensors=kv_cache_raw_tensors,
        kernel_block_sizes=kernel_block_sizes, cache_dtype=cache_dtype,
        shared_kv_cache_layers=shared_kv_cache_layers, kv_cache_config=kv_cache_config)
    # 3. 绑定到 attention 层
    bind_kv_cache(kv_caches, forward_context, runner_kv_caches, num_attn_module)
    return kv_caches
```

注意：V2 的 `initialize_kv_cache` **没有** V1 中的 `may_add_encoder_only_layers_to_kv_cache_config()` 调用；V2 在 `get_kv_cache_spec()` 阶段对 encoder-only 直接返回空 dict，encoder-only 层在 scheduler 与 runner 两侧的处理方式与 V1 不同。

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

## 九、初始化后的 KV cache 管理与使用

初始化完成后，存在两套协同状态：**engine-core 侧的逻辑管理**（scheduler + `KVCacheManager`）和 **worker 侧的物理 KV tensor**。运行时每个 step 都在两者之间往返 block id 与零化/拷贝指令。

### 9.1 管理器构建：`KVCacheManager` 与 coordinator

Scheduler 在 `__init__`（`vllm/v1/core/sched/scheduler.py:277`）中用 scheduler 视角的 `KVCacheConfig` 构建管理器：

```python
self.kv_cache_manager = KVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=self.max_model_len,
    max_in_flight_tokens=vllm_config.max_in_flight_tokens,
    enable_caching=self.cache_config.enable_prefix_caching,
    scheduler_block_size=self.block_size,
    hash_block_size=hash_block_size,
    watermark=self.scheduler_config.watermark,
    ...)
# 绑定 GPU block pool 给 KV connector（必须在 manager 构建后，block_pool 已就绪）
if self.connector is not None:
    self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)
```

`KVCacheManager.__init__`（`kv_cache_manager.py:118`）内部：

```python
self.coordinator = get_kv_cache_coordinator(kv_cache_config=..., ...)  # 选择 coordinator
self.block_pool = self.coordinator.block_pool
self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
self.empty_kv_cache_blocks = KVCacheBlocks(tuple(() for _ in range(num_groups)))
```

`get_kv_cache_coordinator`（`kv_cache_coordinator.py:923`）按场景三选一：

```python
if not enable_caching:
    return KVCacheCoordinatorNoPrefixCache(...)   # 无 prefix cache
if len(kv_cache_config.kv_cache_groups) == 1:
    return UnitaryKVCacheCoordinator(...)         # 单 group
return HybridKVCacheCoordinator(...)              # 多 group（混合 attention/Mamba）
```

coordinator 内部为每个 KV cache group 实例化一个 `SingleTypeKVCacheManager` 子类（按 `register_all_kvcache_specs` 注册的映射），并共享一个 `BlockPool`。逻辑结构：

```text
KVCacheManager（调度协调层）
    └── KVCacheCoordinator（NoPrefix / Unitary / Hybrid）
            ├── BlockPool（物理 block 池 + null block + 引用计数 + 哈希索引）
            └── 每 group 一个 SingleTypeKVCacheManager
                    ├── FullAttentionManager / SlidingWindowManager /
                    │   ChunkedLocalAttentionManager / MambaManager /
                    │   CrossAttentionManager / RSWAManager / SinkFullAttentionManager
                    ├── req_to_blocks: request_id -> [KVCacheBlock]
                    └── num_cached_block: request_id -> 已缓存 block 数
```

### 9.2 单类型管理器：`SingleTypeKVCacheManager` 基类

`vllm/v1/core/single_type_kv_cache_manager.py:36` 维护每个请求的 block 列表与缓存进度：

```python
# 请求已占用的 block（含 null block 占位）
self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
# 请求已写入哈希索引（可被 prefix 命中）的 block 数
self.num_cached_block: dict[str, int] = {}
# 本轮新分配、需 worker 零化的 block id
self.new_block_ids: list[int] = []
# 部分命中需要 copy-on-write 的 (source, dest) 对
self._pending_cow_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
```

关键方法：

```python
def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks,
                               total_computed_tokens, num_local_computed_tokens, ...):
    # 计算还需分配多少 block：扣除已占 block、扣除窗口外 skip 的 block，
    # 对可回收 spec 应用 admission cap；部分命中额外 +1 用于 CoW
    ...

def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
    # 先处理 partial-hit 的 CoW redirect，再向 block_pool 申请新 block
    ...

def cache_blocks(self, request, num_tokens, retention_interval=None):
    # 把已满的 block 写入 block_pool 的哈希索引，使 prefix cache 可命中
    ...

def free(self, request_id):
    # 逆序归还 block（tail 先回收）
    self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))
```

### 9.3 调度阶段：前缀命中与 block 分配

Scheduler 每步在 `_schedule_request` 里先做 prefix 命中，再分配（`scheduler.py:448/629`）：

```python
computed_blocks, num_new_computed_tokens = \
    self.kv_cache_manager.get_computed_blocks(request)

new_blocks = self.kv_cache_manager.allocate_slots(
    request, num_new_tokens, num_lookahead_tokens=...)
```

`KVCacheManager.get_computed_blocks`（`kv_cache_manager.py:248`）做前缀查找：

```python
def get_computed_blocks(self, request):
    if not self.prefix_cache_lookup_enabled(request):
        return self.empty_kv_cache_blocks, 0, 0
    # 最多命中到 num_tokens - 1（最后一个 token 必须重算以得到 logits）
    max_cache_hit_length = request.num_tokens - 1
    computed_blocks, num_new_computed_tokens, num_uncached = \
        self.coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
    ...
    return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens, boundary
```

`KVCacheManager.allocate_slots`（`kv_cache_manager.py:355`）分阶段：

```python
# 1) 释放窗口外 skip block，并检查自由 block 是否足够（不足返回 None -> 触发抢占）
self.coordinator.remove_skipped_blocks(
    request.request_id,
    max(0, total_computed_tokens - request.num_in_flight_tokens), ...)

num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)
if num_blocks_to_allocate + watermark_blocks > available_blocks:
    return None

# 2) 挂接 prefix 命中块 + 外部(connector)块
self.coordinator.allocate_new_computed_blocks(
    request_id, new_computed_blocks, num_local_computed_tokens, num_external_computed_tokens)

# 3) 为新 token 分配新 block
new_blocks = self.coordinator.allocate_new_blocks(
    request.request_id, num_tokens_need_slot, num_tokens_main_model, num_encoder_tokens)

# 4) 把新 block 写入哈希索引（prefix cache）
self.coordinator.cache_blocks(request, ...)
```

其中 `add_local_computed_blocks` 在启用缓存时调用 `block_pool.touch(blocks)` 增加引用计数；滑动窗口还会用 `null_block` 填充已 skip 的位置。

### 9.4 回收与驱逐：free 与 deferred free

请求结束（`FINISHED_STOPPED` / `FINISHED_ABORTED`）后，scheduler 走 `_free_request_blocks`（`scheduler.py:2429`）：

```python
def _free_request_blocks(self, request):
    if not self.defer_block_free or ...:
        self.kv_cache_manager.free(request)      # 立即归还
    else:
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        self.deferred_frees.append((self.sched_step_seq, blocks))  # 延迟归还
```

延迟归还（async 调度 / KV connector 写竞争场景）在 `update_from_output` 里按 `processed_step_seq` 栅栏排空：

```python
def _drain_deferred_frees(self):
    while self.deferred_frees:
        fence, _ = self.deferred_frees[0]
        if fence > self.processed_step_seq:
            break
        _, blocks = self.deferred_frees.popleft()
        # 逆序归还，tail block 先驱逐
        self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
```

`BlockPool.free_blocks` 递减引用计数；`ref_cnt == 0` 的 block 回到自由队列，可被后续请求复用（复用前 worker 会零化）。

### 9.5 零化与 Copy-on-Write 的下发

Scheduler 在 `schedule()` 结束前取出本轮新 block 与 CoW 对，写入 `SchedulerOutput`（`scheduler.py:1244/1325`）：

```python
new_block_ids_to_zero = self.kv_cache_manager.take_new_block_ids()
...
kv_cache_block_copies = self.kv_cache_manager.take_kv_cache_block_copies()
```

Worker 侧 V2 `update_requests`（`gpu/model_runner.py:1013` 起）消费这两个字段：

```python
def update_requests(self, scheduler_output):
    for req_id, num_computed_tokens, req_new_block_ids in zip(...):
        if req_new_block_ids is not None:
            self.block_tables.append_block_ids(req_index, req_new_block_ids, overwrite=False)

    # 零化新分配 block（防脏数据）
    if scheduler_output.new_block_ids_to_zero:
        assert self.kv_block_zeroer is not None
        self.kv_block_zeroer.zero_block_ids(scheduler_output.new_block_ids_to_zero)

    # 部分 prefix 命中后的 copy-on-write
    if scheduler_output.kv_cache_block_copies:
        copy_kv_cache_blocks_inplace(
            self.kv_caches, self.kv_cache_config.num_blocks,
            scheduler_output.kv_cache_block_copies)
```

### 9.6 运行时全链路小结

```text
每个 step：
  Scheduler.schedule()
    -> KVCacheManager.get_computed_blocks()      # prefix 命中
    -> KVCacheManager.allocate_slots()           # 释放窗口外块 / 申请新块 / 缓存块
    -> SchedulerOutput(block ids, new_block_ids_to_zero, kv_cache_block_copies)
  Executor RPC execute_model()
    -> V2 runner update_requests()
         -> BlockTables.append_block_ids()       # 更新逻辑 -> 物理映射
         -> KVBlockZeroer.zero_block_ids()       # 零化新块
         -> copy_kv_cache_blocks_inplace()       # CoW 部分命中
    -> 模型 forward 读写 KV cache
  Scheduler.update_from_output()
    -> 采样 token / 更新 num_computed_tokens
    -> _free_request_blocks() / _drain_deferred_frees()   # 归还结束请求的 block
```

数据流要点：

- **逻辑 block id** 由 scheduler/manager 管理，经 `SchedulerOutput` 下发给 worker。
- **物理 tensor** 由 V2 runner 持有，`BlockTables` 维护请求块表，`KVBlockZeroer` 保证复用块被清零。
- **前缀复用** 依赖 `block_hashes` 与 `BlockPool` 的哈希索引；滑动窗口/Mamba 等稀疏保留类型在 `remove_skipped_blocks` / `cache_blocks` 里用 null block 或 mask 跳过不可复用的块。
- **引用计数 + 延迟释放** 保证 async 调度与 KV connector 写竞争下不会过早回收正在被读写的 block。
