
## KV Cache Spec 类型分析
```markdown

### 1. KVCacheSpec 基类

`KVCacheSpec` 描述一个模型层或一组兼容层的 KV cache 格式，核心字段和方法包括：

- `block_size`：一个逻辑 block 包含的 token 数。
- `page_size_bytes`：一个物理 cache page 的字节数。
- `max_memory_usage_bytes()`：该 spec 可能占用的最大显存。
- `max_num_blocks_per_req()`：单个请求需要的 block table 条目数量。

因此，spec 会同时影响：

1. KV cache 显存容量；
2. block table 行宽；
3. scheduler 的 block 分配和 admission；
4. prefix cache 命中与 block 回收；
5. worker 中物理 KV tensor 的布局。

### 2. AttentionSpec

`AttentionSpec` 是 attention 类 cache spec 的基类，包含：

- `num_kv_heads`
- `head_size`
- `head_size_v`
- `dtype`
- `kv_quant_mode`
- `page_size_padded`
- `indexes_kv_by_block_stride`
- `num_head_slots`
- `state_content_bytes`

普通情况下，page 大小可以理解为：

```text
page_size_bytes =
    num_heads
    * storage_block_size
    * state_content_size_bytes
```

其中：

```text
state_content_size_bytes =
    (head_size + head_size_v) * dtype_size
```

如果启用 DCP，attention cache 的 block table 宽度会按照 DCP world size 进行分片计算。

### 3. FullAttentionSpec

`FullAttentionSpec` 表示普通全上下文 attention。

其最大显存使用量大致为：

```text
ceil(max_model_len / block_size) * page_size_bytes
```

它还可以携带：

- `sliding_window`
- `attention_chunk_size`
- `non_causal`

其中 `non_causal` 主要影响调度策略。EngineCore 检测到 non-causal attention 后，会关闭依赖 causal attention 的 chunked prefill 和 prefix caching。

### 4. MLAAttentionSpec

`MLAAttentionSpec` 用于 MLA 模型。

主要特点：

- `storage_block_size = block_size / compress_ratio`
- MLA 通常只保存 latent state，不使用独立的 V，因此 `head_size_v = 0`
- 支持 cache dtype、模型版本和内存对齐配置
- 可通过 `page_size_padded` 适配硬件对齐要求

MLA 在 manager 层通常复用 `FullAttentionManager`，但物理 cache layout 由 MLA spec 和 attention backend 决定。

### 5. HiddenStateCacheSpec

`HiddenStateCacheSpec` 继承自 `MLAAttentionSpec`，用于 speculative decoding 中的 hidden-state cache。

它主要是一个语义标记类型：

- 物理布局沿用 MLA 风格；
- manager 使用 `FullAttentionManager`；
- 不参与普通的 uniform type grouping。

### 6. RSWASpec

`RSWASpec` 表示 Reference Sliding Window Attention。

它的特点是：

- prefill 阶段的前缀保持全局可见；
- decode 阶段只保留最近的窗口；
- prefill 尾部和当前 decode 窗口之间的 gap blocks 可以被回收；
- 物理 page 计算继承 full attention。

其 manager 是 `RSWAManager`。

### 7. ChunkedLocalAttentionSpec

`ChunkedLocalAttentionSpec` 表示按 chunk 工作的局部 attention。

它包含：

```python
attention_chunk_size: int
```

其最大显存使用量不是完整 `max_model_len`，而是按照：

```text
attention_chunk_size + max_in_flight_tokens
```

计算。

运行时，chunk 左侧已经不再需要参与 attention 的 block 会被标记为 skipped 或回收。对应 manager 是 `ChunkedLocalAttentionManager`。

### 8. SlidingWindowSpec

`SlidingWindowSpec` 表示滑动窗口 attention。

它包含：

```python
sliding_window: int
extra_retained_tokens: int
```

最大 resident block 数由以下因素共同决定：

- sliding window 大小；
- in-flight tokens；
- extra retained tokens；
- block size；
- block 边界对齐。

滑动窗口之外的 block 可以被跳过和回收。对应 manager 是 `SlidingWindowManager`。

当前实现中，Sliding Window 不支持 DCP。

### 9. SlidingWindowMLASpec

`SlidingWindowMLASpec` 是 MLA 和 sliding window 的组合：

- 物理布局使用 MLA 的压缩和对齐规则；
- 容量和 block 回收使用 sliding window 规则；
- manager 使用 `SlidingWindowManager`。

### 10. MambaSpec

`MambaSpec` 用于 Mamba、SSM、GDN 等 recurrent state cache。

它通过以下字段描述状态：

```python
shapes: tuple[tuple[int, ...], ...]
dtypes: tuple[torch.dtype, ...]
```

一个 page 的大小是所有 state tensor 大小之和：

```text
page_size_bytes =
    sum(prod(shape) * dtype_size for shape, dtype in zip(shapes, dtypes))
```

Mamba cache mode 会影响容量：

- `all`：按照完整序列保存状态；
- `align`：保存少量对齐状态 block；
- 其他模式：通常保存当前状态和 speculative blocks。

Mamba state 不按 DCP/PCP 分片，因此 block table 计算规则与 attention 不同。对应 manager 是 `MambaManager`。

### 11. EncoderOnlyAttentionSpec

`EncoderOnlyAttentionSpec` 用于 encoder-only attention。

当前实现中：

```python
def max_memory_usage_bytes(...):
    return 0
```

也就是说，它不需要 decoder KV cache。V2 runner 可能会把 encoder-only 层补充到自身 cache 配置中，但它不是普通 decoder KV cache 的持久化分配类型。

### 12. CrossAttentionSpec

`CrossAttentionSpec` 用于 encoder-decoder 模型的 cross-attention。

它缓存的是 encoder states，因此容量根据：

```text
max_num_encoder_input_tokens
```

而不是 decoder 的 `max_model_len` 计算。

Cross-attention cache 通常属于请求私有数据：

- 不适合 prefix cache 共享；
- 不支持普通 prefix cache 命中；
- 对应 manager 是 `CrossAttentionManager`。

### 13. SinkFullAttentionSpec

`SinkFullAttentionSpec` 表示带 attention sink 的 full attention。

它继承 full attention 的物理布局，同时包含：

```python
sink_len: int | None
```

`SinkFullAttentionManager` 会在初始化时预留 sink blocks，使这些 block 不被普通请求回收。

### 14. UniformTypeKVCacheSpecs

`UniformTypeKVCacheSpecs` 不是一种新的 attention cache 格式，而是多个同类 layer spec 的聚合包装：

```python
kv_cache_specs: dict[str, KVCacheSpec]
```

当多个 layer 满足以下条件时，可以被聚合：

- block size 相同；
- uniform type 相同；
- block table 宽度一致；
- cache 行为兼容。

它的 page size 是内部所有 layer page size 之和：

```text
page_size_bytes =
    sum(inner_spec.page_size_bytes)
```

它的最大 block table 宽度要求所有内部 spec 一致。

### 15. KVCacheSpecKind

`KVCacheSpecKind` 是对 spec 的语义分类：

```text
FULL_ATTENTION
MLA_ATTENTION
SLIDING_WINDOW
SLIDING_WINDOW_MLA
MAMBA
CHUNKED_LOCAL_ATTENTION
SINK_FULL_ATTENTION
ENCODER_ONLY_ATTENTION
CROSS_ATTENTION
UNKNOWN
```

`get_kv_cache_spec_kind()` 会优先匹配更具体的子类。例如：

```text
SlidingWindowMLASpec -> SLIDING_WINDOW_MLA
MLAAttentionSpec     -> MLA_ATTENTION
SinkFullAttentionSpec -> SINK_FULL_ATTENTION
FullAttentionSpec    -> FULL_ATTENTION
```

因此，分类顺序很重要，不能先判断父类。

### 16. KVCacheGroupSpec、KVCacheTensor 和 KVCacheConfig

三者职责不同：

```text
多个 layer spec
    -> KVCacheGroupSpec
    -> KVCacheConfig
    -> KVCacheTensor
```

`KVCacheGroupSpec` 表示共享同一个逻辑 block table 的 layer 组：

```python
KVCacheGroupSpec(
    layer_names=[...],
    kv_cache_spec=...,
)
```

`KVCacheTensor` 描述物理分配方式：

- `size`：tensor 的总字节数；
- `shared_by`：共享该 tensor 的 layer；
- `offset`：layer 在 packed block 中的字节偏移；
- `block_stride`：packed layout 中每个 block 的总跨度。

`KVCacheConfig` 是单个 worker rank 的完整配置，包含：

- `num_blocks`
- `kv_cache_groups`
- `kv_cache_tensors`

因此：

- group 决定哪些 layer 共享 block table；
- tensor 决定这些 layer 如何共享或切分物理存储；
- spec 决定 page 大小、容量和调度语义。

### 17. Spec 到 Manager 的映射

当前内置映射关系如下：

| Spec | Manager |
| --- | --- |
| `FullAttentionSpec` | `FullAttentionManager` |
| `MLAAttentionSpec` | `FullAttentionManager` |
| `HiddenStateCacheSpec` | `FullAttentionManager` |
| `RSWASpec` | `RSWAManager` |
| `SlidingWindowSpec` | `SlidingWindowManager` |
| `SlidingWindowMLASpec` | `SlidingWindowManager` |
| `ChunkedLocalAttentionSpec` | `ChunkedLocalAttentionManager` |
| `MambaSpec` | `MambaManager` |
| `CrossAttentionSpec` | `CrossAttentionManager` |
| `SinkFullAttentionSpec` | `SinkFullAttentionManager` |

manager 负责：

- prefix cache 命中；
- block 分配；
- block 回收；
- sliding window 或 chunk 的窗口管理；
- Mamba state 和 speculative block 管理；
- Cross-attention 的请求私有 cache；
- attention sink block 保留。

Spec 会沿着以下链路影响最终运行：

```text
KVCacheSpec
    -> page_size_bytes
    -> max_memory_usage_bytes
    -> KVCacheConfig.num_blocks
    -> KVCacheTensor
    -> init_kv_cache(...)
    -> Scheduler block manager
    -> Worker block table
    -> Attention backend physical layout
```

因此，KV cache spec 不只是初始化阶段的形状描述，它还决定显存预算、block table 宽度、scheduler admission、prefix cache 行为、block 回收策略和最终物理 KV tensor 布局。
```

源码依据：

- `kv_cache_interface.py`
- `kv_cache_spec_registry.py`
- `single_type_kv_cache_manager.py`

其中 `single_type_kv_cache_manager.py` 负责将不同 spec 映射到对应的 manager。