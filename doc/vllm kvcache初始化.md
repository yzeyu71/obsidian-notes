# vLLM V1 ModelRunner V2 的 KV Cache 初始化

> 本文基于 **v0.28.0** 源码，以「分享讲解」的形式完整走读 KV Cache 的初始化链路。
> 适合听众：已经会用 vLLM 跑过模型、想理解内部机制、准备读源码或做二次开发的同学。

## 导读：这篇文章讲什么

KV Cache 是 vLLM 里最核心、也最复杂的模块之一：它决定了吞吐、显存占用、前缀复用，也贯穿了「调度器」和「执行器」两个进程。很多读者对着 `KVCacheConfig`、`KVCacheManager`、`BlockPool` 这些概念一头雾水，就是因为不知道**它们是在哪一步、由谁、为了什么被创建出来的**。

这篇文章的回答路径只有一条主线：

> **探测显存 → 规划配置 → 下发到每个 rank → 真正分配物理显存 → warmup → 交给运行时按 block 调度使用**

我们会顺着这条主线，把 vLLM 的代码一层层剥开，看到 `EngineCore`、`Executor`、`Worker`、`GPUModelRunner`、`KVCacheManager` 各自扮演的角色。

### 先读这张表：同一单词的不同含义

源码里有几组**同一个词在不同层级/语境下含义完全不同**的参数，是读代码时最容易踩坑的地方。先建立心智模型，后面每遇到一个都能对号入座。

**① "block size" 家族（5 种含义）**

| 参数名 | 位置 | 单位 | 真实含义 |
|---|---|---|---|
| `spec.block_size` | `KVCacheSpec` | token | 逻辑 block 大小，决定 block table 语义 |
| `scheduler_block_size` | `resolve_kv_cache_block_sizes` | token | 调度对齐粒度（单 group = block_size×dcp；多 group = LCM） |
| `hash_block_size` | 同上 | token | prefix 哈希粒度（单 group = scheduler size；多 group = GCD 或 prefix_match_unit） |
| `kernel_block_size` | `init_kv_cache` / `KVBlockZeroer` | token | attention backend 的 kernel 对齐粒度（虚拟 block 拆分） |
| `block_stride` | `KVCacheTensor`（packed 布局） | **字节** | 每 block 的字节跨度 |

前四个单位都是 token，但语义各不相交；`block_stride` 单位还是字节。读代码时"block_size"几乎每个函数都要重新心算一遍。

**② `page_size`（字节） vs `block_size`（token）**

- `page_size_bytes`（`KVCacheSpec` 属性）单位是**字节**，用于 `available_memory // page_size` 这类字节运算；
- `block_size` 单位是 **token**，用于 `cdiv(max_len, block_size)` 这类 token 运算。

两者词根混用（page/block），一个是字节、一个是 token，相除操作也完全不同。

**③ `num_blocks` 的两种含义**

- `KVCacheConfig.num_blocks` = **全局物理 block pool 的容量**（全 rank 收敛到一致值）；
- `initialize_kv_cache` 里的 `max_num_blocks_per_group` = **单个请求 block table 的行宽**（每请求最多多少块）。

"池容量"与"单请求行宽"都叫 blocks，在第五章的同函数里并存。

**④ `memory` 家族（6 个名字）**

| 名字 | 含义 |
|---|---|
| `kv_cache_memory_bytes` | 用户手动指定的 KV cache 大小 |
| `available_kv_cache_memory_bytes` | worker profiling 算出的可用量 |
| `available_memory` / `available_gpu_memory` | executor 汇总的各 worker 可用量（`list[int]`） |
| `requested_memory` | gpu_memory_utilization × 总显存 |
| `non_kv_cache_memory` | 权重 + 激活占用的非 KV 部分 |
| `cudagraph_memory_estimate(_applied)` | CUDA Graph 预估占用（是否扣除受环境变量门控） |

另外注意 `determine_available_memory` 名字叫 "available memory"，返回值却是**扣除了多模态 IPC 共享内存、并经过 `reserve_mm_ipc_gpu_memory` 处理后的 KV cache 专属可用量**，不是裸显存。

**⑤ 其余容易误读的细节**

- `get_kv_cache_groups(vllm_config, kv_cache_spec)` 与 `_project_kv_cache_groups_to_worker(..., worker_spec)` 的 `kv_cache_spec` / `worker_spec` **实为 `dict[str, KVCacheSpec]`（层名 → spec 的映射）**，不是单个 spec；对比 `SingleTypeKVCacheManager(kv_cache_spec, ...)` 里才是真正的单个 spec；
- `get_num_blocks(vllm_config, num_layers, ...)` 的参数名叫 `num_layers`，调用时传入的**实为 `group_size`（共享内存池数量）**，单 group 时两者相等、多 group 时不等；
- `available_memory` 在 `get_kv_cache_configs` 是 `list[int]`（每 worker 一个），在 `get_kv_cache_config_from_groups` / `get_num_blocks` 是单个 `int`，维度随层级不同；
- `init_kv_cache(runner_kv_caches, ...)` 的 `runner_kv_caches` 是**作为输出被填充的 list**，不是输入；
- `max_num_blocks_per_req(vllm_config, max_len)` 的 `max_len` 对 encoder-decoder 要传**含 encoder 长度**的值，与 `max_model_len`（仅 decoder）语义不同。

---

## 一、先理解问题：KV Cache 是什么、为什么难管理

动手看代码之前，先把「问题」本身想清楚。KV Cache 的难点不在"要不要缓存"，而在于：**它太大了，大到必须被当成一种受管理的资源来对待**。

### 1.1 为什么需要 KV Cache

Transformer 的自回归解码是一个「逐步生成、逐 token 计算」的过程。第 $t$ 个 token 计算 attention 时，需要和它之前的所有 token（prompt 与已生成的 token）做交互。attention 的 Q/K/V 计算中：

- **Query（Q）** 只依赖当前正在计算的 token，每步都变；
- **Key（K）/ Value（V）** 依赖历史 token 的内容，一旦某个历史 token 计算过就不会再变。

如果不做缓存，每生成一个新 token 都要把整个前缀重新送入模型，重算所有历史 token 的 K 和 V，计算量随序列长度呈 $O(n^2)$ 增长。KV Cache 的思想是：**把已经算过的历史 token 的 K/V 保存在显存里，每步只算新 token 的 K/V，再从缓存里读取其余部分**。这样每个 decode 步的计算量降为 $O(n)$。

### 1.2 KV Cache 是显存消耗的大头

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

- **显存预算**：profiling 决定能分多少给 KV Cache；
- **按 block 细粒度分配与回收**：避免碎片化和浪费；
- **跨请求复用**：prefix cache，同样的前缀不重复算。

> **举两个例子感受一下量级**（均设 KV 为 fp16，`head_size = 128`，GQA 8 个 KV head）：
>
> **例 1：Qwen2.5-7B / Llama-3.1-8B（32 层）**
> 每个 token、每层占 $2 \times 8 \times 128 \times 2 = 4\ \text{KiB}$，32 层合计每个 token 128 KiB。
>
> ```text
> 32K 上下文：128 KiB × 32,768 ≈ 4 GiB
> 128K 上下文：128 KiB × 131,072 ≈ 16 GiB
> ```
>
> 该模型权重（8B 参数 × 2 B）约 16 GiB——**单请求 128K 上下文的 KV Cache 就与全部权重一样大**；同时并发 4 个这样的请求，KV Cache 即达权重的 4 倍。
>
> **例 2：Llama-3.1-70B（80 层）**
> 每个 token 占 $2 \times 8 \times 128 \times 2 \times 80 = 320\ \text{KiB}$。
>
> ```text
> 128K 上下文：320 KiB × 131,072 ≈ 40 GiB
> 并发 8 个请求：320 GiB  vs  70B 权重（fp16）约 130 GiB
> ```
>
> 只跑 8 个 128K 请求，KV Cache（320 GiB）就是全部权重（约 130 GiB）的 **2 倍以上**。
>
> 这也是为什么 KV Cache 必须"精打细算"：即便用 GQA 已经把 KV head 从 32~64 压到 8 个，显存消耗依然惊人，任何分配浪费都难以接受。

### 1.3 vLLM 用「分页」思路管理 KV Cache

vLLM 借鉴操作系统的虚拟内存，把 KV Cache 切成固定大小的 **block**（类似内存页），每个请求持有一张 **block table**（逻辑 token 位置 → 物理 block 的映射）。好处是：

- 请求按需申请 block，用完即释放，不会因为「最大长度预留」而浪费；
- 不同请求可以复用同一批物理 block（prefix cache 命中时直接共享，写入时再 copy-on-write）；
- 滑动窗口、Mamba 等特殊 attention 类型可以在 block 粒度上「跳过 / 回收」不再需要的部分。

![](../assets/Pasted%20image%2020260827114959.png)

> **核心要点**：KV Cache 不只是"一个缓存"，而是一套需要预算、分配、复用、回收的资源管理系统。后面的所有代码，本质上都在回答同一组问题——**分多少、怎么分、分给谁、用完怎么还**。

---

## 二、整体调用链：先看全景

搞清楚问题之后，我们先建立全局视角。KV Cache 初始化发生在服务启动阶段，由 engine-core 进程统一驱动，跨进程协作完成。

![](../assets/Pasted%20image%2020260824195538.png)

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

入口位于 `vllm/v1/engine/core.py` 的 `EngineCore._initialize_kv_caches()`。它负责规划 KV cache 配置，并通过 executor 的 collective RPC 下发给所有 worker。

**一眼看懂这张图的关键**：engine-core 进程只负责"规划"，真正动手分配显存的是每个 worker 进程里的 `GPUModelRunner`。中间隔着一层 RPC，配置以 `list[KVCacheConfig]`（每个 rank 一份）的形式传过去。

> **核心要点**：KV Cache 初始化是"一个中心（EngineCore 规划）+ 多个执行者（worker 分配）"的协作。后面三、四、五章分别是这张图的三个层次。

---

## 三、EngineCore：KV Cache 的全局规划

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
        if vllm_config.scheduler_config.enable_chunked_prefill:
            logger.info("Disabling chunked prefill: model has non-causal attention layers.")
            vllm_config.scheduler_config.enable_chunked_prefill = False
        if vllm_config.cache_config.enable_prefix_caching:
            logger.info("Disabling prefix caching: model has non-causal attention layers.")
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
    if scheduler_kv_cache_config.kv_cache_groups:
        vllm_config.cache_config.block_size = min(
            g.kv_cache_spec.block_size
            for g in scheduler_kv_cache_config.kv_cache_groups)
        update_kv_cache_capacity(vllm_config, scheduler_kv_cache_config)
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

这个方法可以概括成三个动作：

1. **收集**（步骤 1-2）：知道模型有哪些 attention 层、各是什么类型的 KV cache；
2. **规划**（步骤 3-6）：确定给 KV Cache 分多少显存、每个 rank 怎么分配、block 多大；
3. **下发**（步骤 7-8）：把配置发下去真正分配，并触发编译与 warmup。

下面逐个展开。

### 3.1 注册 spec 类型：`register_all_kvcache_specs`

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

> 讲解提示：**spec** 描述"某一层需要什么样的 KV cache"（full / sliding window / mamba / cross-attention…），**manager** 描述"运行时怎么管理这种 cache"（分配、缓存、回收）。注册表就是把两者一一对应起来，同时允许不同 spec 复用同一个 manager（比如 MLA 归入 FullAttentionManager）。

### 3.2 收集 spec：`get_kv_cache_specs`

`vllm/v1/executor/abstract.py`：

```python
def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
    return self.collective_rpc("get_kv_cache_spec")
```

V2 runner 的 `get_kv_cache_spec()`（`gpu/model_runner.py`）对 encoder-only 返回 `{}`，否则调用 `get_kv_cache_spec(self.vllm_config)`。每个 worker 返回 `{layer_name: KVCacheSpec}`。

> 讲解提示：为什么是"每 rank 一份 dict"？因为启用了 pipeline parallel 时，不同 stage 上分布着不同层的模型，层名各不相同。这份 dict 就是各 worker 对"我这里有哪些层、各是什么类型"的自报家门。

### 3.3 non-causal 修正：双向注意力不能用前缀缓存

`non-causal`（非因果）模型，例如 BERT 等双向编码器模型，之所以不能开启前缀缓存（`enable_prefix_caching`）和分块预填充（`enable_chunked_prefill`），根本原因在于其**双向注意力机制**与这两个功能所依赖的**因果注意力（Causal Attention）假设**存在根本性冲突 [](https://docs.vllm.ai/en/v0.28.0/api/vllm/model_executor/layers/attention/prefill_prefix_lm_attention/)。

具体原因如下：

#### 核心原因：KV 缓存机制失效

前缀缓存和分块预填充的核心，是缓存并复用之前计算好的 **KV Cache**。这个机制建立在**自回归生成**的假设上：每个新 token 只能"看到"它之前的所有 token（因果注意力）。

然而，`non-causal` 模型（如 BERT）在编码时，允许每个 token **同时关注其左右两边的所有 token** [](https://docs.vllm.ai/en/v0.28.0/api/vllm/model_executor/layers/attention/prefill_prefix_lm_attention/)。这种"双向"的上下文依赖关系，使得 KV Cache 的值会因后续输入的变化而改变，**无法被简单地复用**。

#### 注意力掩码（Attention Mask）不匹配

前缀缓存和分块预填充的调度逻辑，**默认所有注意力计算都使用因果掩码（Causal Mask）** [](https://docs.vllm.ai/en/v0.28.0/api/vllm/model_executor/layers/attention/prefill_prefix_lm_attention/)。而 `non-causal` 模型必须使用**非因果掩码**，允许"看到未来"。如果在 `non-causal` 模型上强制开启这些功能，会导致**注意力计算错误，进而产生数值不一致或模型输出错误的结果**。

因此 engine-core 在收集到带 `non_causal` 标记的 spec 时，会主动关掉这两个功能并打日志：

```python
if any(getattr(spec, "non_causal", False)
       for worker_specs in kv_cache_specs
       for spec in worker_specs.values()):
    if vllm_config.scheduler_config.enable_chunked_prefill:
        logger.info("Disabling chunked prefill: model has non-causal attention layers.")
        vllm_config.scheduler_config.enable_chunked_prefill = False
    if vllm_config.cache_config.enable_prefix_caching:
        logger.info("Disabling prefix caching: model has non-causal attention layers.")
        vllm_config.cache_config.enable_prefix_caching = False
```

### 3.4 显存探测：`determine_available_memory`

`GPUWorker.determine_available_memory()`（`gpu_worker.py:475`）核心：

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

核心思想一句话：**用一个 dummy batch 跑一次前向，量出模型"吃"掉多少显存，剩下的才留给 KV Cache**。`gpu_memory_utilization`（默认 0.9）就是"允许占多少"的上限。


### 3.5 生成每 rank 配置：`get_kv_cache_configs`

> 定位：`get_kv_cache_configs` 是"规划"阶段的核心方法——吃进 3.2 收集的各 worker spec 和 3.4 探测的可用显存，吐出每个 rank 的 `KVCacheConfig`。它要同时解决两个矛盾：**PP 下各 worker 层不同**（需"全局规划、按 worker 投影"）与**各 rank 显存不同**（需"收敛到一致的 block 数"）。

`vllm/v1/core/kv_cache_utils.py:2082` 的真实流程：

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

**逐步拆解（每一步的"为什么"比代码本身更重要）：**

- **(a) 合并 spec —— 建立"整模型视角"**。PP 下每个 worker 只拥有部分层（stage0 有 0-15 层、stage1 有 16-31 层），各自上报的 spec 只是全局的子集；TP 下同 stage 的各 rank 层相同。这里把散落的 spec 合并成一份"全模型层 → spec"映射。如果同一层在不同 worker 上报了不同 spec，直接 assert 失败——同层类型必须一致，这是跨 rank 一致性的第一道防线。
- **(b) 校验注册 —— 防止"运行时找不到 manager"**。`check_kv_cache_spec_registry` 逐个确认 spec 类都在 `register_all_kvcache_specs` 里注册过。注册表是 spec → manager 的映射，漏注册会在分组 / 调度阶段才爆雷，提前到启动期报错更友好。
- **(c) MTP 标记 —— 给滑动窗口"留尾巴"**。使用多模块 MTP（多个 speculative module）时，给每个 `SlidingWindowSpec` 打上 `extra_retained_tokens = num_speculative_tokens - 1`。原因：sliding window 只保留最近窗口的 token，而 MTP 的 store 侧比主模型延迟一拍，若窗口边界正好卡在 MTP 需要的 token 上，缓存里就重建不出窗口，所以要额外保留这些 token。
- **(d) 全局分组 —— 见 3.5.1**。`get_kv_cache_groups` 把"整模型"的层按 attention 类型分成若干个 group（full / sliding window / mamba / cross-attention…）。注意它可能**原地修改** `merged_kv_cache_specs`（例如强制统一混合类型的 page size）。⚠️ 参数名 `kv_cache_spec` **实为 `dict[str, KVCacheSpec]`（层名 → spec 映射）**，不是单个 spec。
- **(e) 按 PP 投影 —— 每个 worker 只看自己的层**。`_project_kv_cache_groups_to_worker` 把全局 group 过滤成只含本 worker 层的子集（`UniformTypeKVCacheSpecs` 里的 per-layer spec 也同步裁剪）。这一步之后，"显存能支撑多少 block"就只和该 worker 自己拥有的层相关。⚠️ 参数 `worker_spec` 同样是 dict，与 (d) 同一命名陷阱。
- **(f) override 换算 —— 用户指定块数时重新校准**。若设置了 `--num-gpu-blocks-override`，实际分配块数与 profiling 出的 `available_memory` 解耦。这里用 `_pool_bytes_per_block` 把 override 换算回 "effective memory"（override × 每 block 字节），让后面的 auto-fit、显存检查、per-worker 配置都基于**同一个有效容量**规划，避免三处对不上。
- **(g) auto-fit max_model_len —— 二分搜索最大可支持上下文**。当用户没指定 `max_model_len`（`original_max_model_len == -1`）时，估算现有显存能装下的最长上下文，取**所有 worker 中能支撑的最小值**（最弱一环），必要时原地改写 `model_config.max_model_len`。这正是 EngineCore 之后比较 `max_model_len_before/after` 并调 `update_max_model_len` 把新值同步给 worker 的原因。
- **(h) 显存足够性检查 —— 启动期就"报错不隔夜"**。逐 worker 计算在 max_model_len 下"至少服务一个请求"需要多少 KV 显存（`_max_memory_usage_bytes_from_groups`），与可用显存比较。不足则直接抛错，并给出"估算最大长度""建议提高 `gpu_memory_utilization` 或降低 `max_model_len`"等可操作提示。
- **(i) 逐 worker 生成配置 —— 见 3.5.2**。每个 worker 用自己的 projected groups + available_memory 调 `get_kv_cache_config_from_groups`，得到各自的 `num_blocks` 与 `kv_cache_tensors`。
- **(j) 收敛到最小 num_blocks —— 多 rank 一致性**。各 rank 显存可能不均衡（物理卡容量不同），但调度器要求**所有 rank 的 block 数完全一致**。于是取全局最小值，并让每个 rank 的 tensor 按 `min / old` 比例缩小，避免多分配的显存闲置。

> **一个具体例子（PP=2、TP=2、纯 full attention 模型）**
>
> ```text
> worker0（stage0，层 0-15）可用显存 20 GiB
> worker1（stage1，层 16-31）可用显存 16 GiB
>
> (a) 合并 → 32 层 full-attention spec
> (d) 全局分组 → 1 个 group（32 层）
> (e) 投影 → worker0 的 group = 层 0-15；worker1 的 group = 层 16-31
> (i) 各自算块数（假设每 block 1 MiB）→ worker0: 20480，worker1: 16384
> (j) 取最小 → 两个 worker 都是 16384，worker0 的 tensor 按 16384/20480 缩小
>
> 结果：所有 rank 一致地拥有 16384 个逻辑 block，调度器无需关心 rank 差异。
> ```

关键点：合并 spec → 分组 → 按 PP 投影 → auto-fit / 校验 → 每 rank 独立建配置 → 收敛到最小 `num_blocks`，保证所有 worker 一致。

> 讲解提示：第 (j) 步很有意思——不同 rank 的显存可能不同（比如多卡不均衡），但调度器要求所有 rank 的 block 数一致，所以**取所有 rank 里最小的那个**，并把 tensor 按比例缩小，避免多分配的显存闲置。

#### 3.5.1 层分组：`get_kv_cache_groups`

`kv_cache_utils.py:1741` 分支顺序：

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

> 讲解提示：**group 是"共享同一份 block table 语义的层的集合"**。大部分模型所有层完全一样，就是一个 group；混合模型（如 DeepSeek V4 的 full + sliding window、Mamba 混合模型）会拆成多个 group。分组的目的是让"同类层"共用内存池、统一 block 大小，从而减少内存碎片。

#### 3.5.2 物理布局：`get_kv_cache_config_from_groups`

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
        num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)  # ⚠️ 参数名是 num_layers，实为 group_size
        for i in range(group_size):
            shared_by = [g.layer_names[i] for g in kv_cache_groups
                         if i < len(g.layer_names)]
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by))

    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=kv_cache_tensors,
                         kv_cache_groups=kv_cache_groups, ...)
```

说明：`num_blocks` 是逻辑 block 数；`kv_cache_tensors` 描述物理 tensor 如何分配、哪些层共享一个 tensor。

### 3.6 scheduler 配置与 block size

`generate_scheduler_kv_cache_config`（`kv_cache_utils.py:1826`）：

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

随后 EngineCore 用该配置同步 `num_gpu_blocks`、取各 group 最小的 `block_size` 写入 `cache_config.block_size`，并调用 `update_kv_cache_capacity` 记录容量日志：

```python
scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
if kv_cache_groups:
    vllm_config.cache_config.block_size = min(
        g.kv_cache_spec.block_size for g in kv_cache_groups)
    update_kv_cache_capacity(vllm_config, scheduler_kv_cache_config)
vllm_config.validate_block_size()
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

> **核心要点**：EngineCore 这一步产出了两类对象——**给 worker 的 `KVCacheConfig`**（每个 rank 一份，描述物理分配）和**给 scheduler 的 `KVCacheConfig`**（统一一份，描述逻辑管理）。物理与逻辑的分工，从这里开始成型。

---

## 四、Executor 与 WorkerWrapperBase：把配置分发到每个 rank

规划完成，接下来是怎么把配置"送"到每个 worker。

### 4.1 Executor 入口

当前方法名是 `initialize_from_config`：

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

### 4.2 WorkerWrapperBase 按 rank 选择配置

```python
# vllm/v1/worker/worker_base.py

def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
    kv_cache_config = kv_cache_configs[self.global_rank]
    with set_current_vllm_config(self.vllm_config):
        self.worker.initialize_from_config(kv_cache_config)
```

因此，Executor 下发的是配置列表；每个 worker 只使用与自身 `global_rank` 对应的 `KVCacheConfig`。

> **核心要点**：配置列表的**顺序**就是 rank 的顺序。`collective_rpc` 保证所有 worker 同时收到，各自"认领"自己的那份。这是一个简单但很重要的约定：任何地方都不需要额外的映射逻辑。

---

## 五、GPUWorker 与 ModelRunner V2：真正的分配与绑定

配置到了 worker，接下来的动作才真正动显存。

### 5.1 实例化 V2 runner

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

### 5.2 `initialize_kv_cache`：核心入口

V2 的核心入口是 `gpu/model_runner.py:495` 的 `initialize_kv_cache`，完整实现如下：

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
        # DCP 时本 rank 一个 block 覆盖 block_size * cp_size 个全局 token
        max_num_blocks = cdiv(
            block_table_max_model_len, spec.block_size * self.dcp_size)
        if isinstance(spec, MambaSpec):
            # Mamba 状态跨 DCP/PCP 复制不切分，行宽额外加 speculative blocks
            max_num_blocks = (
                max_num_blocks if self.cache_config.enable_prefix_caching else 1
            ) + spec.num_speculative_blocks
            max_num_blocks = get_block_table_width(
                max_num_blocks, spec.block_size, token_alignment=None)
        else:
            max_num_blocks = get_block_table_width(max_num_blocks, spec.block_size)
        max_num_blocks_per_group.append(max_num_blocks)

    # (3) 初始化 attention groups、backend 与 kernel block size
    self.attn_groups, attn_cg_support, self.kernel_block_sizes = init_attn_backend(
        self.kv_cache_config, self.vllm_config, self.device)
    attn_cg_support = attn_cg_support.narrow(
        *self.model_state.get_additional_cg_support())

    # (4) 自适应验证（spec decode 相关）；启用后强制 FULL_AND_PIECEWISE 模式
    self.adaptive_verification = maybe_create_adaptive_verification_manager(...)
    if self.adaptive_verification is not None:
        self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE

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

> 讲解提示：这个方法虽然长，但只做了三类事：
> 1. **尺寸计算**（步骤 1-2）：每个 group 的 block 大小、每个请求的 block table 行宽（考虑 encoder 长度、DCP、Mamba）；
> 2. **运行时对象准备**（步骤 3-8）：attention backend、block tables、PCP manager、CUDA graph 模式、speculator；
> 3. **真正分配**（步骤 9-10）：分配物理内存、绑定到各层、创建 KV connector。

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

> **核心要点**：分配不是"allocate 一个 tensor 就完事"，而是 **raw tensor（一大块内存）→ 按层 reshape 成视图 → 绑定到每个 attention 层** 三步。`kv_cache_tensors` 里 `shared_by` 字段决定了哪些层共享同一块内存（这是多 group 模型省显存的关键）。

---

## 六、warmup 与编译：让 GPU 提前"热身"

### 6.1 这一节在解决什么问题

一句话：**warmup 是 vLLM 正式对外服务前的"彩排"**。它用假的（dummy）请求把模型完整跑几遍，让所有"临场才发生的编译 / 调优"在第一个真实请求到来之前完成。

如果跳过这步，第一个真实请求会**边跑边现场编译**：

- 自定义 / Triton kernel 第一次遇到某个形状才开始 JIT 编译；
- torch.compile 的图没有编过；
- CUDA Graph 没有捕获。

结果就是**第一个 token 的延迟暴涨**（vLLM 常见的"冷启动卡顿"）。

EngineCore 在完成 KV cache 初始化后调用：

```python
if not envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
    self.model_executor.compile_or_warm_up_model()
```

Executor 通过 RPC 调用每个 worker 的 `compile_or_warm_up_model()`。注意：**这个方法定义在 `GPUWorker` 上**（`gpu_worker.py:693`），不是 V2 `GPUModelRunner` 的方法。它是 worker 的"对外编排接口"——负责把 runner 提供的 `_dummy_run`、`capture_model`、`warmup_kernels` 按正确顺序串起来。

### 6.2 为什么"热身"有三件不同的事

`compile_or_warm_up_model()` 里其实揉了三类性质不同的"热身"——这是读代码时最容易被绕晕的点：

| 热身 | 解决什么问题 | 底层机制 |
|---|---|---|
| ① shape 级 dummy run（`_dummy_run(size)`） | 让模型按几种典型 token 数量先跑一遍，把编译结果固化下来 | torch.compile（VLLM_COMPILE 模式）按形状编译 |
| ② CUDA Graph 捕获（`capture_model()`） | 把固定形状的一整串 kernel **录制**成图，重放时没有 Python 调度开销，decode 小 batch 延迟大降 | CUDA Graph capture + replay |
| ③ kernel 调优 + Triton JIT（`kernel_warmup` / `warmup_kernels`） | 让 attention、Mamba、sampler 等**非 cudagraph 部分**完成 autotune 和首次编译 | Triton / 自定义 kernel autotune |

背后是两个配置开关决定"走哪条路"：

- `CompilationMode`（NONE / STOCK_TORCH_COMPILE / VLLM_COMPILE）：决定是否用 torch.compile 把模型图编译；
- `cudagraph_mode`（NONE / PIECEWISE / FULL / FULL_AND_PIECEWISE）：决定是否捕获 CUDA Graph。`FULL_AND_PIECEWISE` = decode batch 用全图、prefill / mixed batch 用 piecewise 图，是性能最好的模式（多数配置下为默认）；
- `--enforce-eager` 会把两者都设成 NONE——此时 warmup 只做 ①③。

CUDA Graph 捕获有个硬性要求：**形状必须固定**。所以 vLLM 对一组预设的 batch 大小分别录制（默认 `[1, 2, 4] + 8,16,...,248 + ...`，上限 512 / 1024）。

### 6.3 `GPUWorker.compile_or_warm_up_model` 四阶段流程

按代码顺序拆解（`gpu_worker.py:693-872`）：

**阶段 1：shape 级 dummy run（编译 warmup）**

```python
if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
    # 编译那些不在 cudagraph capture sizes 里、但用户仍希望编译的形状
    warmup_sizes = compile_sizes 减去 cudagraph_capture_sizes
    # 再补上每个 compile_range 的 end，保证每个区间至少编译过一次
for size in sorted(warmup_sizes, reverse=True):
    self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
```

`_dummy_run(size)`（`gpu/model_runner.py:633`）干的事：把 `size` 个 token 摊到若干假请求上，构造一个 `SchedulerOutput`，然后走一遍和真实推理**完全相同的 `execute_model`**。V2 这里还会连带 dummy run speculator（EAGLE / MTP）的 propose。

**阶段 2：kernel 调优**

```python
kernel_warmup(self)
```

必须放在 capture 之前：调优选出的最快 kernel 配置，之后才会被"录进" CUDA Graph。

**阶段 3：CUDA Graph 捕获**

```python
if not self.model_config.enforce_eager:
    cuda_graph_memory_bytes = self.model_runner.capture_model()
```

`capture_model()`（`gpu/model_runner.py:848`）对每个 capture size 调用 `cudagraph_manager.capture(...)`，并测量实际占用显存。这个数值有两个用途：

- 打印"CUDA graph 实际 vs 预估"对比日志；
- 参与"建议 `--kv-cache-memory` 大小"的计算。

**阶段 4（V2 专属）：Triton kernel 补漏**

```python
if self.use_v2_model_runner:
    warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
```

这是 V2 与 V1 最大的不同：它**手工构造真实调度器形状的 SchedulerOutput**（一次 N 请求 prefill → 多个 decode step → 清理），再走 `worker.execute_model` + `sample_tokens`，把 sampler、grammar bitmask、零化等**不在 cudagraph 里的 Triton kernel 全部 JIT 编译一遍**，同时 `kv_block_zeroer.warmup()` 预热零化 kernel。V1 在同样位置只做 sampler / pooler 的 dummy run。


> **核心要点**：四阶段的顺序是——**先 shape 编译，再 kernel 调优，再录 CUDA Graph，最后补漏 Triton kernel**。每一步都是让"运行时必然发生的首次编译"提前到服务开始之前。

---

## 七、Scheduler 衔接：逻辑管理与物理 tensor 的分工

到这里，初始化完成。但 KV Cache 的管理才刚刚开始——初始化产出的配置，要交给调度器去用。

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

> **核心要点**：记住这条边界——
> - **engine-core 侧**：`Scheduler` + `KVCacheManager`，只知道"逻辑 block"，不碰显存；
> - **worker 侧**：V2 `GPUModelRunner` + 物理 KV tensor，只知道"物理 block"，不参与调度。
> 两者每步通过 `SchedulerOutput` 交换 block id 与零化/拷贝指令。

---

## 八、初始化之后：KV Cache 的运行时管理

初始化完成后，存在两套协同状态：**engine-core 侧的逻辑管理**（scheduler + `KVCacheManager`）和 **worker 侧的物理 KV tensor**。运行时每个 step 都在两者之间往返 block id 与零化/拷贝指令。

### 8.1 管理器构建：`KVCacheManager` 与 coordinator

Scheduler 在 `__init__`（`vllm/v1/core/sched/scheduler.py:277`）中用 scheduler 视角的 `KVCacheConfig` 构建管理器：

```python
self.kv_cache_manager = KVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=self.max_model_len,
    max_in_flight_tokens=vllm_config.max_in_flight_tokens,
    enable_caching=self.cache_config.enable_prefix_caching,
    scheduler_block_size=self.block_size,   # 调度对齐粒度（token）
    hash_block_size=hash_block_size,        # prefix 哈希粒度（token）；两者关系见 3.6
    watermark=self.scheduler_config.watermark,
    ...)
# 绑定 GPU block pool 给 KV connector（必须在 manager 构建后，block_pool 已就绪）
if self.connector is not None:
    self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)
```

`KVCacheManager.__init__`（`kv_cache_manager.py:119`）内部：

```python
self.coordinator = get_kv_cache_coordinator(kv_cache_config=..., ...)  # 选择 coordinator
self.block_pool = self.coordinator.block_pool
self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
self.empty_kv_cache_blocks = KVCacheBlocks(tuple(() for _ in range(num_groups)))
```

`get_kv_cache_coordinator`（`kv_cache_coordinator.py:915`）按场景三选一：

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

> 讲解提示：三层结构各司其职——
> `KVCacheManager`（对外统一入口）→ `KVCacheCoordinator`（按场景选策略）→ 每 group 一个 `SingleTypeKVCacheManager`（干具体活）+ 共享的 `BlockPool`（管物理 block 的分配、引用计数、哈希索引）。之前注册表里的"spec → manager"映射，在这里真正生效。

### 8.2 单类型管理器：`SingleTypeKVCacheManager` 基类

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

### 8.3 调度阶段：前缀命中与 block 分配

Scheduler 每步在 `_schedule_request` 里先做 prefix 命中，再分配（`scheduler.py:448/629`）：

```python
computed_blocks, num_new_computed_tokens = \
    self.kv_cache_manager.get_computed_blocks(request)

new_blocks = self.kv_cache_manager.allocate_slots(
    request, num_new_tokens, num_lookahead_tokens=...)
```

`KVCacheManager.get_computed_blocks`（`kv_cache_manager.py:232`）做前缀查找：

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

`KVCacheManager.allocate_slots`（`kv_cache_manager.py:347`）分阶段：

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

### 8.4 回收与驱逐：free 与 deferred free

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

### 8.5 零化与 Copy-on-Write 的下发

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

### 8.6 运行时全链路小结

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

---

## 九、回顾与延伸

### 10.1 一图流回顾

```text
启动阶段：                             运行阶段（每个 step）：
Worker 自报 spec（哪些层要什么 cache）    Scheduler.get_computed_blocks()  ← prefix 命中
GPUWorker 探测可用显存                    Scheduler.allocate_slots()        ← 分配/回收 block
EngineCore 规划 KVCacheConfig（每 rank）  SchedulerOutput(new_block_ids,
  └─ 分组 / auto-fit / 收敛最小 blocks       new_block_ids_to_zero, copies)
Executor 经 collective_rpc 下发           └─> worker update_requests()
WorkerWrapperBase 按 global_rank 认领       ├─ BlockTables.append_block_ids
GPUWorker + V2 Runner.initialize_kv_cache   ├─ KVBlockZeroer.zero_block_ids
  └─ 分配 raw tensor -> reshape -> bind     └─ copy_kv_cache_blocks_inplace（CoW）
GPUWorker compile_or_warm_up_model        forward 读写物理 KV cache
Scheduler 用同一配置构建 KVCacheManager   update_from_output 归还结束请求的 block
```

### 10.2 想深入，可以接着看这些

- **`BlockPool`**（`vllm/v1/core/block_pool.py`）：引用计数、null block、哈希索引、驱逐策略——prefix cache 的物理基础；
- **`KVCacheCoordinator` 三兄弟**（`vllm/v1/core/kv_cache_coordinator.py`）：NoPrefix / Unitary / Hybrid 的策略差异；
- **`KVBlockZeroer`**（`vllm/v1/worker/utils.py`）：零化的 CUDA 实现；
- **DeepSeek V4 的 packed 布局**（`kv_cache_utils.py` 的 `_use_packed_kv_cache_config` / `_get_kv_cache_config_packed`）：多 group 共享内存池的高级玩法；
- **KV transfer（connector）**：P/D 分离、KV offload 场景下 block 池如何在进程间流动。

### 10.3 三个最值得记住的"心智模型"

1. **KV Cache 是一种受管资源**：预算、分配、复用、回收——所有机制都围绕这四个词展开；
2. **规划与执行分离**：engine-core 只做逻辑规划，worker 才碰物理显存，中间靠 `collective_rpc` + "配置列表按 rank 索引"通信；
3. **每步协作**：调度器管"逻辑 block"，runner 管"物理 tensor"，通过 `SchedulerOutput` 交换 block id、零化与拷贝指令。
