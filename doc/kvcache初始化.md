以下是 vLLM V1 架构中 `ModelRunnerV2`（`GPUModelRunner`）KV Cache 初始化的完整流程与代码注释。

---

## 一、顶层入口：EngineCore._initialize_kv_caches

整个流程始于 `EngineCore` 初始化时调用的 `_initialize_kv_caches` 方法。该方法承担**顶层规划**职责：探测可用显存并生成 KV Cache 的“施工蓝图”。

```python
# vllm/v1/engine/core.py

def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
    """
    顶层入口：规划 KV Cache 的分配方案。

    返回值:
        KVCacheConfig: 描述 KV Cache 布局的配置对象，包含：
            - num_blocks: 总的块数量
            - kv_cache_groups: 每个 KV Cache 组的规格
            - tensors: 需要分配的张量形状列表
    """
    # Step 1: 探测可用显存
    # 通过 model_executor 向所有 Worker 查询可用显存量
    available_memory = self.model_executor.determine_available_memory()

    # Step 2: 根据可用显存和模型配置，生成 KVCacheConfig
    # get_kv_cache_configs 会综合考虑：
    #   - 模型层数、注意力头数、头维度
    #   - block_size（用户配置，默认 16）
    #   - 可用显存量
    #   - 是否启用 prefix caching 等
    kv_cache_config = get_kv_cache_configs(
        vllm_config,
        kv_cache_specs,      # 每层的 KV Cache 规格
        available_memory
    )
    # 示例: KVCacheConfig(num_blocks=10, ...)

    # Step 3: 将蓝图下发给 Executor，由 Executor 分发给所有 Worker
    self.model_executor.initialize(kv_cache_config)

    return kv_cache_config
```

**关键辅助函数**：`get_kv_cache_configs` 和 `get_kv_cache_config_from_groups` 位于 `vllm/v1/core/kv_cache_utils.py`，负责将模型层的 KV Cache 规格转换为实际的分配配置。

### `EngineCore._initialize_kv_caches` 的作用

它是 vLLM 在启动引擎时，真正“准备 KV cache”的关键函数，定义在 `core.py`。它在 `EngineCore.__init__()` 里被调用，位于模型 executor 创建之后、scheduler 创建之前，作用是：

- 发现模型到底需要哪些 KV cache
- 估算可用显存
- 根据模型结构和配置生成实际的 KV cache layout
- 让 executor 真正初始化这些 cache
- 触发模型编译/预热
- 返回 scheduler 需要的 KV cache 配置

---

### 整体流程

#### 1) 注册所有 KV cache spec
```python
register_all_kvcache_specs(vllm_config)
```

这一步会把模型/attention 相关的 KV cache 规格注册到全局/当前进程里。也就是让框架知道“这个模型有哪些 KV cache 组、每组该怎么描述”。

---

#### 2) 获取模型实际需要的 KV cache specs
```python
kv_cache_specs = self.model_executor.get_kv_cache_specs()
```

这里是从模型 executor 取出各层的 cache specification。
这些 spec 包含信息如：

- 是否是 causal / non-causal attention
- block size
- head 数、dtype、shape 等

这一步决定后续是否真的需要分配 KV cache。

---

#### 3) 处理 non-causal attention 的特殊情况
```python
if any(getattr(spec, "non_causal", False) ...):
    ...
```

如果某些 layer 使用了非因果注意力（例如 Prefix LM / 其他 special attention），那么：

- 不适合 chunked prefill
- 不适合 prefix caching

因为这两种调度策略默认假设 causal attention。函数会在这里关闭对应配置，避免逻辑错误。

这一步很关键：它不是只在“内存层面”做事，而是在“调度语义层面”修正配置。

---

#### 4) 计算当前可用于 KV cache 的显存
```python
has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
if has_kv_cache:
    available_gpu_memory = self.model_executor.determine_available_memory()
```

这里分两种情况：

- 有 KV cache：调用 `determine_available_memory()` 去 profile 模型和当前 GPU 空闲情况，估出“可分配给 KV cache 的显存”
- 没有 KV cache：直接置为 `[0] * len(kv_cache_specs)`

很容易理解：如果模型是 attention-free 之类，不需要 KV cache，就不需要为其分配内存。

---

#### 5) 检查最大长度是否被 auto-fit 调整
```python
max_model_len_before = vllm_config.model_config.max_model_len
kv_cache_configs = get_kv_cache_configs(...)
...
if max_model_len_after != max_model_len_before:
    self.collective_rpc("update_max_model_len", args=(max_model_len_after,))
```

这一步很有意思。
在实际分配 KV cache 时，如果可用内存不足，框架可能会自动缩小 `max_model_len`，也就是“auto-fit”了模型长度。
一旦变了，工作进程（worker）里缓存的旧最大长度必须同步更新，否则不同进程可能有不同理解。

所以这里是把“engine-core 侧决定的最终 max_model_len”广播给 workers。
![](../assets/Pasted%20image%2020260819102843.png)

---

#### 6) 生成 scheduler 视角的配置
```python
scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
...
if kv_cache_groups:
    vllm_config.cache_config.block_size = min(...)
    update_kv_cache_capacity(vllm_config, scheduler_kv_cache_config)
```

这部分把每个 KV cache group 的配置整合成一个“scheduler 可用”的总结对象：

- `num_blocks`: 总共可用多少 cache block
- `kv_cache_groups`: 每个 group 的描述
- `block_size`: scheduler 需要的 block 大小

随后还调用：

- `update_kv_cache_capacity(...)`：更新 cache capacity
- `vllm_config.validate_block_size()`: 校验 block size 合法性

---

#### 7) 真正初始化 cache 和 warmup
```python
self.model_executor.initialize_from_config(kv_cache_configs)
if not envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
    self.model_executor.compile_or_warm_up_model()
```

这里才是真正做“分配和初始化”：

- 初始化各 worker / executor 的 KV cache 状态
- 为模型执行做编译或 warmup
- 这一步是启动耗时的核心之一

---

#### 8) 日志与计时
最后它记录了整个初始化花了多少时间：

- profiling time
- compile time
- encoder compile time

这一般用于评估启动是否异常慢。

---

#### 9) 返回值与 Scheduler 的衔接

它返回的是：

```python
scheduler_kv_cache_config
```

这个对象会在 `__init__` 中继续用来创建 scheduler：

```python
self.scheduler = Scheduler(
    vllm_config=vllm_config,
    kv_cache_config=kv_cache_config,
    ...
)
```

所以它的核心意义是：

- 先确定 cache 规格
- 再让 scheduler 按这个规格做调度
- 最终模型能在正确的 block layout 上跑起来

---

#### 10) 一句话总结

`_initialize_kv_caches` 本质上就是 vLLM 的“启动前缓存规划与初始化器”：它负责把“模型需要的 KV cache 结构 + 可用显存 + 调度要求 + worker 状态”统一收敛成一个完整的 cache config，然后真正准备好执行模型。

---

## 二、中层分发：Executor.initialize

`EngineCore` 将 `KVCacheConfig` 传递给 `ModelExecutor`（如 `MultiProcExecutor`）。

```python
# vllm/v1/executor/multiproc_executor.py

def initialize(self, kv_cache_config: KVCacheConfig) -> None:
    """
    将 KV Cache 配置分发给所有 Worker 进程。

    对于 MultiProcExecutor，每个 Worker 运行在独立的进程中，
    需要通过 RPC 调用其 initialize_kv_cache 方法。
    """
    # 遍历所有 Worker，通过进程间通信调用 Worker.initialize_kv_cache
    for worker in self.workers:
        worker.initialize_kv_cache(kv_cache_config)
        # Worker 内部调用 self.model_runner.initialize_kv_cache
```

---

## 三、核心执行：GPUModelRunner.initialize_kv_cache

`GPUModelRunner`（即 `ModelRunnerV2`）的 `initialize_kv_cache` 方法是初始化的**核心执行阶段**。

```python
# vllm/v1/worker/gpu/model_runner.py

def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
    """
    根据 KVCacheConfig 蓝图，在 GPU 上实际分配 KV Cache 内存。

    这是 ModelRunnerV2 的核心初始化入口，执行以下子步骤：
        1. 分配物理内存 (_allocate_kv_cache_tensors)
        2. 初始化注意力后端 (init_attn_backend)
        3. 初始化 BlockTables
        4. 初始化 InputBatch 数据结构
        5. 初始化 KV 块清零器 (_init_kv_zero_meta)
        6. 初始化 CUDA Graph 管理器
    """
    # ---------- Step 1: 分配物理内存 ----------
    # 根据 kv_cache_config.tensors 在 GPU 上分配实际的 KV Cache 张量
    self._allocate_kv_cache_tensors(kv_cache_config)

    # ---------- Step 2: 初始化注意力后端 ----------
    # 根据配置（FlashAttention、FlashInfer、TRTLLM 等）初始化注意力计算后端
    # init_attn_backend 会设置 forward_context 中的注意力实现
    init_attn_backend(kv_cache_config, self.attn_backends)

    # ---------- Step 3: 初始化 BlockTables ----------
    # 创建 BlockTables 对象，管理逻辑块到物理块的映射
    self.block_tables = BlockTables(
        num_blocks=kv_cache_config.num_blocks,
        block_size=kv_cache_config.block_size,
        device=self.device
    )

    # ---------- Step 4: 初始化 InputBatch ----------
    # 重建 InputBatch 数据结构以匹配新的 KV Cache 参数
    self.input_batch = InputBatch(
        block_size=kv_cache_config.block_size,
        max_num_seqs=self.scheduler_config.max_num_seqs,
        device=self.device
    )

    # ---------- Step 5: 初始化 KV 块清零器 ----------
    # 延迟初始化 KVBlockZeroer，用于运行时对新分配的内存块进行清零
    # 某些注意力内核（如 FP8 KV Cache 的 TRTLLM 内核）会读取整个 KV 页，
    # 未初始化的内存会导致 NaN 值
    self._init_kv_zero_meta()

    # ---------- Step 6: 初始化 CUDA Graph 管理器 ----------
    # 为后续的 CUDA Graph 捕获做准备
    self.cudagraph_manager = CUDAGraphManager(...)
```

### 3.1 物理内存分配：_allocate_kv_cache_tensors

```python
# vllm/v1/worker/gpu/model_runner.py

def _allocate_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> None:
    """
    在 GPU 上分配 KV Cache 的物理内存。

    根据 kv_cache_config.tensors 中描述的形状，为每一层分配实际的张量。
    这些张量构成了存储所有请求 KV 状态的原始内存池。
    """
    # kv_cache_config.tensors 是一个列表，每个元素描述一个需要分配的张量
    # 例如: [("layer_0", (num_blocks, block_size, num_heads, head_dim)), ...]
    for tensor_spec in kv_cache_config.tensors:
        # 使用 torch.empty 在 GPU 上分配内存
        # 注意：此处不进行初始化（未清零），由后续的 KVBlockZeroer 负责
        tensor = torch.empty(
            tensor_spec.shape,
            dtype=self.kv_cache_dtype,  # 如 torch.float16, torch.float8_e4m3fn
            device=self.device
        )
        self.kv_cache[tensor_spec.name] = tensor
    # ... (其余 KV 层张量依此分配)
```

### 3.2 KV 块清零器初始化：_init_kv_zero_meta

```python
# vllm/v1/worker/gpu/model_runner.py

def _init_kv_zero_meta(self) -> None:
    """
    初始化 KVBlockZeroer 实例。

    KVBlockZeroer 使用 Triton 内核高效地将新分配的 KV 块内存清零。
    构造完成后，每个调度步骤可调用 zero_block_ids 清零新分配的块。
    """
    if self.kv_block_zeroer is None:
        # 创建 KVBlockZeroer 实例
        # 它会预计算所有 KV 缓存段的地址，以便后续高效清零
        self.kv_block_zeroer = KVBlockZeroer(
            self.device,
            is_pin_memory_available()
        )
        # 调用 init_meta 预计算段地址
        self.kv_block_zeroer.init_meta(self.kv_cache)
```

---

## 四、前置分支：CUDA Graph 内存预留

在主初始化流程之前，可能存在一个用于 CUDA Graph 内存预留的**前置分支**。这是为了支持 CUDA Graph 加速而进行的性能分析。

```python
# vllm/v1/worker/gpu/model_runner.py

def profile_cudagraph_memory(self) -> None:
    """
    为 CUDA Graph 预留内存的前置流程。

    此方法在主初始化之前被调用，用于测量 CUDA Graph 所需的确切显存量。
    """
    # Step 1: 分配极小的 KV Cache 用于性能分析
    # 通过 num_gpu_blocks_override 强制使用最小配置
    minimal_config = self._create_minimal_kv_cache_config()
    self._init_minimal_kv_cache_for_profiling(minimal_config)

    # Step 2: 执行一次“假的” CUDA Graph 捕获
    # 用最小 KV Cache 运行模型，测量显存占用
    self.capture_model()

    # Step 3: 清理分析状态
    # 释放用于测试的 KV Cache 和 Graph，为正式初始化准备干净环境
    self._teardown_profiling_state()
    # 更新 cache_config.num_gpu_blocks = minimal_config.num_blocks
```

---

## 五、运行时关键：新分配块的内存清零

在模型推理的每个步骤中，当调度器（Scheduler）分配了新的 KV 块时，会触发内存清零操作。

```python
# vllm/v1/worker/gpu/model_runner.py

def execute_model(self, scheduler_output: SchedulerOutput) -> ModelOutput:
    """
    执行模型推理的入口。

    在每个调度步骤中，检查是否有新分配的 KV 块需要清零。
    """
    # ... 前置处理 ...

    # ---------- 新分配块的内存清零 ----------
    # 如果调度器输出了新分配的块 ID，将它们的内存清零
    if scheduler_output.new_block_ids_to_zero:
        # 确保 KVBlockZeroer 已初始化
        assert self.kv_block_zeroer is not None
        # 调用 Triton 内核高效清零这些块
        self.kv_block_zeroer.zero_block_ids(
            scheduler_output.new_block_ids_to_zero
        )

    # ... 继续执行模型前向传播 ...
```

### 5.1 KVBlockZeroer 的实现

```python
# vllm/v1/worker/utils.py

class KVBlockZeroer:
    """
    管理 KV 缓存块的高效清零。

    通过 Triton 内核实现高效的并行清零操作。
    构造一次（在 KV Cache 分配后）预计算段地址，
    然后每个步骤调用 zero_block_ids 清零新分配的块。
    """

    def __init__(self, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.is_pin_memory = is_pin_memory
        self.segments = []  # 预计算的段地址列表

    def init_meta(self, kv_cache: dict[str, torch.Tensor]) -> None:
        """
        一次性预计算 zero_block_ids 所需的段地址。

        对于每个 KV 缓存层，计算其在物理内存中的偏移和跨度，
        以便 zero_block_ids 能直接寻址到每个块。
        """
        # 为每个 KV 缓存张量构建段（segment）
        # 每个段描述了该层中每个块的起始地址和大小
        for name, tensor in kv_cache.items():
            # 计算该层的块跨度（block_stride）
            # 对于 packed KV Cache（如 DeepSeek V4），需要特殊处理
            segment = self._build_segment(tensor)
            self.segments.append(segment)

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """
        清零指定块 ID 列表对应的所有 KV 缓存段。

        使用 Triton 内核并行清零，确保效率。
        """
        if not block_ids:
            return

        # 对每个段，清零 block_ids 中所有块对应的内存区域
        for segment in self.segments:
            # 启动 Triton 内核，并行清零
            self._zero_segment_blocks(segment, block_ids)
```

---

## 六、完整流程时序图

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. EngineCore.__init__                                                 │
│    └── _initialize_kv_caches()                                        │
│        ├── model_executor.determine_available_memory()                │
│        ├── get_kv_cache_configs() → KVCacheConfig                     │
│        └── model_executor.initialize(kv_cache_config)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. MultiProcExecutor.initialize(kv_cache_config)                      │
│    └── for each worker: worker.initialize_kv_cache(kv_cache_config)   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. GPUModelRunner.initialize_kv_cache(kv_cache_config)                │
│    ├── _allocate_kv_cache_tensors()     # 分配物理内存               │
│    ├── init_attn_backend()              # 初始化注意力后端            │
│    ├── BlockTables.__init__()           # 初始化块映射表              │
│    ├── InputBatch.__init__()            # 初始化输入批处理结构        │
│    ├── _init_kv_zero_meta()             # 初始化 KVBlockZeroer        │
│    │   └── KVBlockZeroer.init_meta()    # 预计算段地址                │
│    └── CUDAGraphManager.__init__()      # 初始化 CUDA Graph 管理器    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. [可选前置] profile_cudagraph_memory()                              │
│    ├── _init_minimal_kv_cache_for_profiling()                         │
│    ├── capture_model()                                                │
│    └── _teardown_profiling_state()                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. [运行时] execute_model(scheduler_output)                           │
│    └── if new_block_ids_to_zero:                                      │
│        └── KVBlockZeroer.zero_block_ids(new_block_ids)               │
│            └── Triton kernel: 并行清零新分配的 KV 块                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 七、关键设计要点总结

| 阶段 | 关键组件 | 职责 |
|------|---------|------|
| **顶层规划** | `EngineCore._initialize_kv_caches` | 探测显存、生成 `KVCacheConfig` 蓝图 |
| **中层分发** | `MultiProcExecutor.initialize` | 将蓝图分发给所有 Worker 进程 |
| **物理分配** | `_allocate_kv_cache_tensors` | 在 GPU 上为每层分配 KV Cache 张量 |
| **后端初始化** | `init_attn_backend` | 初始化注意力计算后端（FlashAttention/FlashInfer/TRTLLM） |
| **内存清零准备** | `KVBlockZeroer.init_meta` | 预计算段地址，为运行时清零做准备 |
| **运行时清零** | `KVBlockZeroer.zero_block_ids` | 每步清零新分配的 KV 块，防止 NaN |
| **CUDA Graph** | `profile_cudagraph_memory` | 前置分析，为 CUDA Graph 预留内存 |