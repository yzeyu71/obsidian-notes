# DeepWiki Q&A with Code Context for Repository: vllm-project/vllm
## Q1
vllm的基本架构
### Answer
## vLLM 基本架构

vLLM V1 采用分层、多进程架构，围绕 `EngineCore` 展开调度与执行的核心循环 vllm:104-113 。

---

### 分层结构

| 层级 | 作用 | 关键组件 |
|---|---|---|
| 外部接口层 | 用户入口 | `LLM` 类、`AsyncLLM`、OpenAI 兼容 API server |
| 配置层 | 参数解析与配置组装 | `EngineArgs`、`VllmConfig`、`ModelConfig` |
| 引擎编排层 | 请求生命周期与进程间通信 | `EngineCore`、`InputProcessor` |
| 调度与内存层 | 资源分配与 KV 缓存管理 | `Scheduler`、`KVCacheManager`、`BlockPool` |
| 执行层 | 硬件上的模型前向计算 | `Executor`、`GPUModelRunner`、`Sampler` |

### 入口 (Entrypoints)

vLLM 提供两种主要入口：

- **`LLM` 类**：用于离线推理，直接与模型交互，无需单独的推理服务器 vllm:14-52 。
- **在线服务**：通过 `vllm serve` 命令启动 API server vllm:54-65 。

### V1 多进程架构

vLLM V1 采用多进程模式来分离关注点、最大化吞吐量，主要进程包括：

1. **API Server 进程**：处理 HTTP 请求，做输入处理（分词、多模态数据加载），并通过 ZMQ 与 engine core 通信 vllm:71-77 。
2. **Engine Core 进程**：运行调度器、管理 KV 缓存、协调 GPU worker 上的模型执行，每个数据并行 (DP) rank 对应一个 engine core 进程 vllm:79-85 。
3. **GPU Worker 进程**：每个 GPU 对应一个 worker 进程，负责加载模型权重、执行前向计算、管理 GPU 内存 vllm:87-93 。
4. **DP Coordinator 进程**（仅在数据并行时存在）：负责 DP rank 间的负载均衡与同步前向传播 vllm:95-101 。

进程数量公式：对于 `N` 个 GPU、`TP` 张量并行、`DP` 数据并行、`A` 个 API server，总进程数为 `A + DP + N`（若 DP > 1 再加 1） vllm:103-117 。

### LLMEngine 与调度

`LLMEngine` 是核心组件，负责接收客户端请求并生成输出，包含输入处理、模型执行（可能跨多机/多卡分布式）、调度和输出处理 vllm:141-157 。`AsyncLLMEngine` 是其异步封装，使用 `asyncio` 后台循环持续处理并发请求，供在线服务使用 vllm:159-169 。

### 请求处理流程

一次请求的完整流转：`AsyncLLM`/`LLM` → `InputProcessor.process_inputs()` 生成 `EngineCoreRequest` → `EngineCore.add_request()` → 循环中 `Scheduler.schedule()` 产出 `SchedulerOutput` → `Executor.execute_model()` 调用 `GPUModelRunner` 执行前向 → `EngineCore.update_from_output()` 更新状态并返回 `EngineCoreOutputs`。

### 关键优化技术

vLLM 的高性能依赖以下核心技术（在 README 中概述）：

- **PagedAttention**：高效管理注意力 KV 内存，消除内存碎片 vllm:28-34 。
- **连续批处理 (Continuous Batching)**、**分块预填充 (Chunked Prefill)**、**前缀缓存 (Prefix Caching)** vllm:32-32 。
- **投机解码 (Speculative Decoding)**：支持 n-gram、suffix、EAGLE、DFlash 等 vllm:37-37 。
- **多硬件平台支持**：NVIDIA GPU (CUDA)、AMD GPU (ROCm)、Intel GPU (XPU)、TPU、CPU（x86/ARM/PowerPC）等，通过 `vllm/platforms` 抽象层实现统一接口 vllm:51-51 。
- **torch.compile 集成**：自动内核生成与图级别优化 vllm:38-38 。

### 配置系统

`VllmConfig` 是全局配置对象，所有组件（引擎、worker、模型类）都通过它统一获取配置，避免逐层传递参数；模型构造函数被统一为 `def __init__(self, *, vllm_config: VllmConfig, prefix: str = "")` 形式 vllm:211-232 。

## Notes

- 更完整的架构图（entrypoints 图、LLMEngine 图、V1 进程架构图）以图片形式存在于 `docs/assets/design/arch_overview/` 中，索引未包含图片内容，建议启动 Devin 会话查看完整渲染文档 `docs/design/arch_overview.md`。
- KV 缓存管理的细节（如 hybrid KV cache、prefix caching 数据结构）属于更深入的子系统，与本次“基本架构”问题相关性较低，仅供参考 vllm:102-119 。

Wiki pages you might want to explore:
- [Overview (vllm-project/vllm)](/wiki/vllm-project/vllm#1)
- [Platform Support (vllm-project/vllm)](/wiki/vllm-project/vllm#10)
### Citations
**File:** docs/design/arch_overview.md (L14-52)
```markdown
### LLM Class

The LLM class provides the primary Python interface for doing offline inference,
which is interacting with a model without using a separate model inference
server.

Here is a sample of `LLM` class usage:

??? code

    ```python
    from vllm import LLM, SamplingParams

    # Define a list of input prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The largest ocean is",
    ]

    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Initialize the LLM engine with the OPT-125M model
    llm = LLM(model="facebook/opt-125m")

    # Generate outputs for the input prompts
    outputs = llm.generate(prompts, sampling_params)

    # Print the generated outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

More API details can be found in the [Offline Inference](../api/README.md#offline-inference) section of the API docs.

The code for the `LLM` class can be found in [vllm/entrypoints/llm.py](../../vllm/entrypoints/llm.py).
```
**File:** docs/design/arch_overview.md (L54-65)
```markdown
### Online Serving

The second primary interface to vLLM is via its online server.
This server can be started using the `vllm serve` command.

```bash
vllm serve <model>
```

The code for the `vllm` CLI can be found in [vllm/entrypoints/cli/main.py](../../vllm/entrypoints/cli/main.py).

More details on the API server can be found in the [Online Serving](../serving/online_serving/README.md) document.
```
**File:** docs/design/arch_overview.md (L71-77)
```markdown
### API Server Process

The API server process handles HTTP requests (e.g., the OpenAI-compatible API), performs input processing (tokenization, multi-modal data loading), and streams results back to clients. It communicates with the engine core process(es) via ZMQ sockets.

By default, there is **1 API server process**, but when data parallelism is used, the API server count automatically scales to match the data parallel size. This can also be manually configured with the `--api-server-count` flag. Each API server connects to **all** engine cores via ZMQ in a many-to-many topology, enabling any API server to route requests to any engine core. Each API server process uses multiple CPU threads for media loading (controlled by `VLLM_MEDIA_LOADING_THREAD_COUNT`, default 8).

The code can be found in [vllm/entrypoints/launchers/api_server](../../vllm/entrypoints/launchers/api_server) and [vllm/v1/utils.py](../../vllm/v1/utils.py).
```
**File:** docs/design/arch_overview.md (L79-85)
```markdown
### Engine Core Process

The engine core process runs the scheduler, manages KV cache, and coordinates model execution across GPU workers. It runs a busy loop that continuously schedules requests and dispatches work to the GPU workers.

There is **1 engine core process per data parallel rank**. For example, with `--data-parallel-size 4`, there are 4 engine core processes.

The code can be found in [vllm/v1/engine/core.py](../../vllm/v1/engine/core.py) and [vllm/v1/engine/utils.py](../../vllm/v1/engine/utils.py).
```
**File:** docs/design/arch_overview.md (L87-93)
```markdown
### GPU Worker Processes

Each GPU is managed by a dedicated worker process. The worker process loads model weights, executes forward passes, and manages GPU memory. Workers communicate with the engine core process that owns them.

There is **1 worker process per GPU**. The total number of GPU worker processes equals `tensor_parallel_size x pipeline_parallel_size` per engine core.

The code can be found in [vllm/v1/executor/multiproc_executor.py](../../vllm/v1/executor/multiproc_executor.py) and [vllm/v1/worker/gpu_worker.py](../../vllm/v1/worker/gpu_worker.py).
```
**File:** docs/design/arch_overview.md (L95-101)
```markdown
### DP Coordinator Process (conditional)

When using data parallelism (`--data-parallel-size > 1`), an additional coordinator process manages load balancing across DP ranks and coordinates synchronized forward passes for MoE models.

There is **1 DP coordinator process** (only when data parallelism is enabled).

The code can be found in [vllm/v1/engine/coordinator.py](../../vllm/v1/engine/coordinator.py).
```
**File:** docs/design/arch_overview.md (L103-117)
```markdown
### Process Count Summary

For a deployment with `N` GPUs, `TP` tensor parallel size, `DP` data parallel size, and `A` API server count:

| Process Type | Count | Notes |
| - | - | - |
| API Server | `A` (default `DP`) | Handles HTTP requests and input processing |
| Engine Core | `DP` (default 1) | Scheduler and KV cache management |
| GPU Worker | `N` (= `DP x PP x TP`) | One per GPU, executes model forward passes |
| DP Coordinator | 1 if `DP > 1`, else 0 | Load balancing across DP ranks |
| **Total** | **`A + DP + N` (+ 1 if DP > 1)** | |

For example, a typical single-node deployment with 4 GPUs (`vllm serve -tp=4`) has:

- 1 API server + 1 engine core + 4 GPU workers = **6 processes**
```
**File:** docs/design/arch_overview.md (L141-157)
```markdown
### LLMEngine

The `LLMEngine` class is the core component of the vLLM engine. It is
responsible for receiving requests from clients and generating outputs from the
model. The `LLMEngine` includes input processing, model execution (possibly
distributed across multiple hosts and/or GPUs), scheduling, and output
processing.

- **Input Processing**: Handles tokenization of input text using the specified
  tokenizer.
- **Scheduling**: Chooses which requests are processed in each step.
- **Model Execution**: Manages the execution of the language model, including
  distributed execution across multiple GPUs.
- **Output Processing**: Processes the outputs generated by the model, decoding the
  token IDs from a language model into human-readable text.

The code for `LLMEngine` can be found in [vllm/engine/llm_engine.py](../../vllm/engine/llm_engine.py).
```
**File:** docs/design/arch_overview.md (L159-169)
```markdown
### AsyncLLMEngine

The `AsyncLLMEngine` class is an asynchronous wrapper for the `LLMEngine` class.
It uses `asyncio` to create a background loop that continuously processes
incoming requests. The `AsyncLLMEngine` is designed for online serving, where it
can handle multiple concurrent requests and stream outputs to clients.

The OpenAI-compatible API server uses the `AsyncLLMEngine`. There is also a demo
API server that serves as a simpler example in [examples/applications/api_server/server.py](../../examples/applications/api_server/server.py).

The code for `AsyncLLMEngine` can be found in [vllm/engine/async_llm_engine.py](../../vllm/engine/async_llm_engine.py).
```
**File:** docs/design/arch_overview.md (L211-232)
```markdown
the `VllmConfig` class, and the model runner can access it directly. We don't
need to change the constructor of the engine, worker, or model class to pass the
new configuration option.

2\. **Uniformity**: The model runner needs a unified interface to create and
initialize the model. vLLM supports more than 50 types of popular open-source
models. Each model has its own initialization logic. If the constructor
signature varies with models, the model runner does not know how to call the
constructor accordingly, without complicated and error-prone inspection logic.
By making the constructor of the model class uniform, the model runner can
easily create and initialize the model without knowing the specific model type.
This is also useful for composing models. Vision-language models often consist
of a vision model and a language model. By making the constructor uniform, we
can easily create a vision model and a language model and compose them into a
vision-language model.

!!! note
    To support this change, all vLLM models' signatures have been updated to:

    ```python
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
    ```
```
**File:** README.md (L28-34)
```markdown
vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
```
**File:** README.md (L37-37)
```markdown
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
```
**File:** README.md (L38-38)
```markdown
- Automatic kernel generation and graph-level transformations using torch.compile
```
**File:** README.md (L51-51)
```markdown
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.
```
**File:** docs/design/prefix_caching.md (L102-119)
```markdown
## Data Structure

The prefix caching in vLLM v1 is implemented in the KV cache manager. The basic building block is the “Block” data class (simplified):

```python
class KVCacheBlock:
    # The block ID (immutable)
    block_id: int
    # The block hash (will be assigned when the block is full,
    # and will be reset when the block is evicted).
    block_hash: BlockHash
    # The number of requests using this block now.
    ref_cnt: int

    # The pointers to form a doubly linked list for the free queue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```
```
## Q2
scheduler在基本架构中是什么角色
### Answer
## Scheduler 的角色

`Scheduler` 是 vLLM V1 引擎编排层中负责**请求调度**与**KV 缓存资源分配**的核心组件，运行在 Engine Core 进程内的忙循环中 vllm:79-85 。

---

### 在架构中的位置

`EngineCore` 初始化时创建 `Scheduler` 实例，将其绑定到 `kv_cache_config`、`structured_output_manager` 等对象上 vllm:144-170 。它实现了抽象接口 `SchedulerInterface` vllm:38-51 。

### 核心职责

1. **调度决策 (`schedule`)**：每次调度对应模型的一次前向传播，产出 `{request_id: num_tokens}` 的字典，决定每个请求本步处理多少 token，支持 chunked prefill、prefix caching、投机解码等场景 vllm:53-83 。
2. **资源分配**：通过 `KVCacheManager.allocate_slots()` 为请求分配 KV 缓存块；分配失败时按调度策略（优先级或 FIFO）抢占低优先级请求 vllm:651-712 。
3. **状态更新 (`update_from_output`)**：模型执行完成后，根据 `ModelRunnerOutput` 更新每个请求的计算进度、投机解码接受/拒绝数、判断请求是否完成 vllm:91-109 ，具体实现处理路由信息、粘性输出等细节 vllm:1797-1833 。
4. **KV Connector 协调**：为 PD 分离/KV 卸载场景构建 `KVConnectorMetadata`，随 `SchedulerOutput` 一并下发 vllm:1319-1332 。
5. **暂停/恢复控制**：`EngineCore.sleep()`、`resume_scheduler()` 等通过 `scheduler.set_pause_state()` 控制调度器暂停状态，用于 sleep 模式或权重更新等场景 vllm:873-879 。

### 请求生命周期中的位置

`EngineCore.add_request()` 将新请求加入调度器的等待队列 vllm:452-492 ；`Scheduler` 内部维护 `waiting`、`running` 等队列结构，并使用可配置的 `SchedulingPolicy`（FCFS 或 priority） vllm:186-199 。整个流转为：`EngineCore.add_request()` → `Scheduler.schedule()` 产出 `SchedulerOutput` → `Executor` 执行模型 → `Scheduler.update_from_output()` 回写状态。

### 补充说明（V1 统一调度器特性）

V1 的统一调度器用简单字典统一处理 prompt 和输出 token，不再区分严格的 prefill/decode 阶段，从而支持 chunked prefill、prefix caching、投机解码 vllm:72-82 。

## Notes

- 治理文档中将 Scheduler 列为 Engine Core 领域下的一个独立 owner 区域（"the core vLLM engine loop scheduling requests to next batch"） vllm:96-99 ，说明它是引擎核心循环中职责边界清晰的模块。
- `Scheduler` 与 `KVCacheManager`/`BlockPool`（KV 缓存分配的底层实现）紧密协作，但这些属于更底层的内存管理细节，未在此详述。

Wiki pages you might want to explore:
- [Overview (vllm-project/vllm)](/wiki/vllm-project/vllm#1)
### Citations
**File:** docs/design/arch_overview.md (L79-85)
```markdown
### Engine Core Process

The engine core process runs the scheduler, manages KV cache, and coordinates model execution across GPU workers. It runs a busy loop that continuously schedules requests and dispatches work to the GPU workers.

There is **1 engine core process per data parallel rank**. For example, with `--data-parallel-size 4`, there are 4 engine core processes.

The code can be found in [vllm/v1/engine/core.py](../../vllm/v1/engine/core.py) and [vllm/v1/engine/utils.py](../../vllm/v1/engine/utils.py).
```
**File:** vllm/v1/engine/core.py (L144-170)
```python
        # Setup KV Caches and update CacheConfig after profiling.
        kv_cache_config = self._initialize_kv_caches(vllm_config)
        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
```
**File:** vllm/v1/engine/core.py (L452-492)
```python
    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        if (
            request.ec_transfer_params is not None
            and self.scheduler.get_ec_connector() is None
        ):
            logger.warning(
                "Got ec_transfer_params, but no ECConnector found. "
                "Disabling ECTransfer for this request."
            )

        self.scheduler.add_request(request)
```
**File:** vllm/v1/engine/core.py (L873-879)
```python
    def resume_scheduler(self) -> None:
        """Resume the scheduler and flush any requests queued while paused."""
        self.scheduler.set_pause_state(PauseState.UNPAUSED)

    def is_scheduler_paused(self) -> bool:
        """Return whether the scheduler is in any pause state."""
        return self.scheduler.pause_state != PauseState.UNPAUSED
```
**File:** vllm/v1/core/sched/interface.py (L38-51)
```python
class SchedulerInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        hash_block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        raise NotImplementedError
```
**File:** vllm/v1/core/sched/interface.py (L53-83)
```python
    @abstractmethod
    def schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
        """Schedule the requests to process in this scheduling step.

        The scheduling decision is made at the iteration level. Each scheduling
        step corresponds to a single forward pass of the model. Therefore, this
        method is called repeatedly by a busy loop in the engine.

        Essentially, the scheduler produces a dictionary of {req_id: num_tokens}
        that specifies how many tokens to process for each request in this
        scheduling step. For example, num_tokens can be as large as the number
        of prompt tokens for new requests, or it can be 1 for the requests that
        are auto-regressively generating new tokens one by one. Otherwise, it
        can be somewhere in between in case of chunked prefills, prefix caching,
        speculative decoding, etc.

        Additionally, the scheduler also returns useful data about each request
        or the batch as a whole. The model runner will use this information in
        preparing inputs to the model.

        Args:
            throttle_prefills: DP prefill balancing. When True (set by the DP
                engine core on non-cadence-aligned steps), new prefill compute is
                deferred to a later step so prefills stay aligned across DP ranks;
                automatically overridden when the rank is saturated.

        Returns:
            A SchedulerOutput object containing information about the scheduled
            requests.
        """
        raise NotImplementedError
```
**File:** vllm/v1/core/sched/interface.py (L91-109)
```python
    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """Update the scheduler state based on the model runner output.

        This method is called after the model runner has processed the scheduled
        requests. The model runner output includes generated token ids, draft
        token ids for next step, etc. The scheduler uses this information to
        update its states, checks the finished requests, and returns the output
        for each request.

        Returns:
            A dict of client index to EngineCoreOutputs object containing the
            outputs for each request originating from that client.
        """
        raise NotImplementedError
```
**File:** vllm/v1/core/sched/scheduler.py (L186-199)
```python
        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []
```
**File:** vllm/v1/core/sched/scheduler.py (L651-712)
```python
            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        # Record the index of the preemption victim to
                        # maintain accurate loop state.
                        victim_index = self.running.index(preempted_req)
                        del self.running[victim_index]
                        # Decrement the loop cursor if the removed request
                        # preceded the current iteration, preventing the
                        # silent omission of the subsequent request.
                        if victim_index < req_index:
                            req_index -= 1

                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            restored = num_scheduled_tokens.pop(preempted_req_id)
                            token_budget += restored
                            input_budget += restored + draft_slots
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(
                        preempted_req,
                        scheduled_timestamp,
                        drop_stale_output=self.requires_kv_delivery,
                    )
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

```
**File:** vllm/v1/core/sched/scheduler.py (L1319-1332)
```python
        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta
```
**File:** vllm/v1/core/sched/scheduler.py (L1797-1833)
```python
        # Persist per-step routed experts into the scheduler-side slot
        # buffer (CPU->CPU fancy-index assign; ~few MB per step).
        # MUST precede the per-request routing reads below: stopped
        # requests may terminate on tokens generated in this very step,
        # whose routing was just D2H'd into model_runner_output.
        routing_data = None
        routing_offsets: dict[str, int] = {}
        if model_runner_output.routed_experts is not None:
            re = model_runner_output.routed_experts
            self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
            routing_data = re.routing_data.astype(
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=False,
            )
            # Build offset map using model runner's request order
            # (input_batch ordering), NOT scheduler dict order.
            offset = 0
            for rid in model_runner_output.req_ids:
                routing_offsets[rid] = offset
                offset += num_scheduled_tokens[rid]

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            output_is_stale = False
            if request is not None:
                request.num_in_flight_tokens -= num_tokens_scheduled
                # Drain any stale share (see _preempt_request) in lockstep.
                if request.num_stale_output_tokens > 0:
                    output_is_stale = True
                    request.num_stale_output_tokens -= num_tokens_scheduled
                    assert request.num_stale_output_tokens >= 0
```
**File:** docs/usage/v1_guide.md (L72-82)
```markdown
!!! note
    vLLM V1’s unified scheduler treats both prompt and output tokens the same
    way by using a simple dictionary (e.g., `{request_id: num_tokens}`) to dynamically
    allocate a fixed token budget per request, enabling features like chunked prefills,
    prefix caching, and speculative decoding without a strict separation between prefill
    and decode phases.

The V1 scheduler supports multiple scheduling policies, including First-Come,
First-Served (FCFS) and priority-based scheduling (where requests are processed
based on assigned priority, with FCFS as a tie-breaker), configurable via the
`--scheduling-policy` argument.
```
**File:** docs/governance/committers.md (L96-99)
```markdown
### Engine Core

- Scheduler: the core vLLM engine loop scheduling requests to next batch
    - @WoosukKwon, @robertgshaw2-redhat, @njhill, @heheda12345
```
## Q3
kvcache初始化流程
## vLLM KV Cache 初始化流程
追踪 EngineCore 启动时如何 profile 显存、计算 KV cache 分组与容量、在各 GPU worker 上实际分配显存，并最终交给 Scheduler 的 KVCacheManager 使用。关键起点在 [1a]，分组算法在 [3a]-[3c]，显存实际分配在 [4c]-[4d]，调度器侧消费在 [5c]。
### 1. EngineCore._initialize_kv_caches 总编排
EngineCore 进程（Engine Core 层）启动阶段的主流程，串联 spec 收集、内存 profiling、配置生成与 worker 初始化。
### 1a. 初始化入口 (`core.py:254`)
EngineCore 构造函数调用此方法，完成 profile→分配→warmup 全流程
```text
def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
```
### 1b. 收集各 worker 的 KV cache 规格 (`core.py:261`)
通过 Executor RPC 到每个 GPU worker 获取其模型层的 KVCacheSpec
```text
kv_cache_specs = self.model_executor.get_kv_cache_specs()
```
### 1c. profiling 可用显存 (`core.py:308`)
对模型做一次前向 profiling，得出每个 worker 可用于 KV cache 的显存字节数
```text
available_gpu_memory = self.model_executor.determine_available_memory()
```
### 1d. 生成每个 worker 的 KVCacheConfig (`core.py:319`)
依据显存与规格合并出分组配置（见 Trace 3）
```text
kv_cache_configs = get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory)
```
### 1e. 生成调度器用的统一配置 (`core.py:332`)
所有 worker 配置在层名以外一致，取任一份供 Scheduler 使用
```text
scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
```
### 1f. 下发配置进行实际显存分配 (`core.py:343`)
触发每个 worker 真正分配 GPU KV cache 显存（见 Trace 4）
```text
self.model_executor.initialize_from_config(kv_cache_configs)
```
### 1g. 编译/预热与 cudagraph 捕获 (`core.py:345`)
KV cache 分配完成后进行模型编译和 cudagraph warmup
```text
self.model_executor.compile_or_warm_up_model()
```
### 2. Executor 抽象层：跨进程 RPC 收集 spec 与显存
Executor 层（vllm/v1/executor/abstract.py）把 EngineCore 的请求转发为对每个 worker 的 collective_rpc 调用，是 Engine Core 与 GPU Worker 进程边界的桥梁。
### 2a. Executor 转发 get_kv_cache_spec (`abstract.py:151`)
return self.collective_rpc("get_kv_cache_spec") 广播到所有 worker
```text
def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
```
### 2b. Executor 转发显存 profiling (`abstract.py:148`)
return self.collective_rpc("determine_available_memory")
```text
def determine_available_memory(self) -> list[int]:  # in bytes
```
### 2c. GPUWorker 侧实现 (`gpu_worker.py:655`)
return self.model_runner.get_kv_cache_spec()，委托给 model runner 解析各 Attention 层
```text
def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
```
### 2d. GPUWorker 显存 profiling 实现 (`gpu_worker.py:481`)
运行一次 profile_run 并计算 gpu_memory_utilization 允许的剩余显存
```text
def determine_available_memory(self) -> int:
```
### 2e. ModelRunner 解析层级 KVCacheSpec (`gpu_model_runner.py:7648`)
遍历 static_forward_context 中的 AttentionLayerBase 得到每层的规格字典
```text
def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
```
### 3. get_kv_cache_configs：跨 worker 分组与容量计算
核心配置层算法（vllm/v1/core/kv_cache_utils.py），把各 worker 的规格与可用内存合并为统一的分组配置，处理混合模型与 PP 分片。
### 3a. 生成各 worker 的 KVCacheConfig (`kv_cache_utils.py:2061`)
合并 spec→生成分组→按内存和 PP 投影裁剪→缩减到最小 block 数
```text
def get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory) -> list[KVCacheConfig]:
```
### 3b. 按统一 page size 分组 (`kv_cache_utils.py:1811`)
把不同类型的层（全量/滑窗/mamba）统一成相同物理页大小的组
```text
groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)
```
### 3c. 投影到单个 worker（PP 分片） (`kv_cache_utils.py:2020`)
流水线并行下每个 worker 只拥有部分层，过滤全局分组到本 worker 的子集
```text
def _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec):
```
### 3d. 生成调度器统一配置 (`kv_cache_utils.py:1836`)
取任一 worker 配置作为调度器视角（各 worker 层名不同但结构一致）
```text
def generate_scheduler_kv_cache_config(kv_cache_configs) -> KVCacheConfig:
```
### 3e. 计算并记录 KV cache 容量 (`kv_cache_utils.py:1871`)
写入 kv_cache_size_tokens / kv_cache_max_concurrency 供日志和后续调度使用
```text
def update_kv_cache_capacity(vllm_config, kv_cache_config) -> None:
```
### 4. initialize_from_config：跨进程下发并实际分配显存
Executor 通过 RPC 把最终 KVCacheConfig 下发到 WorkerWrapper→GPUWorker→GPUModelRunner，最终在 GPU 上真正分配张量并绑定到 Attention 层，是 GPU Worker 进程内的执行路径。
### 4a. Executor 触发 RPC (`abstract.py:120`)
self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
```text
def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
```
### 4b. WorkerWrapperBase 分发本 rank 配置 (`worker_base.py:339`)
kv_cache_config = kv_cache_configs[self.global_rank]; self.worker.initialize_from_config(...)
```text
def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
```
### 4c. GPUWorker 分配 KV cache (`gpu_worker.py:671`)
ensure_kv_transfer_initialized 后调用 model_runner.initialize_kv_cache(kv_cache_config)
```text
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
```
### 4d. ModelRunner 组装 attn backend 并分配张量 (`gpu_model_runner.py:7514`)
调用 initialize_attn_backend 和 initialize_kv_cache_tensors 完成实际显存分配
```text
def initialize_kv_cache(self, kv_cache_config: KVCacheConfig, is_profiling: bool = False) -> None:
```
### 4e. 调用底层显存分配函数 (`gpu_model_runner.py:7463`)
将各层缓存张量的分配委托给 vllm/v1/worker/utils.py 的 allocate_kv_cache
```text
kv_caches = allocate_kv_cache(kv_cache_config, self.device, self.cache_config.get_resolved_kv_cache_layout(), kernel_block_sizes)
```
### 4f. 一次性分配整块显存 buffer (`utils.py:395`)
所有层共享同一块底层分配，再按 layer/block 偏移切片成 view
```text
buf = torch.zeros(sizes.pop(), dtype=torch.int8, device=device)
```
### 4g. 绑定缓存张量到 Attention 层 (`gpu_model_runner.py:7478`)
把分配好的 KV cache 张量注入每个 Attention 模块的 forward context，供前向计算使用
```text
bind_kv_cache(kv_caches, self.compilation_config.static_forward_context, self.kv_caches, num_attn_module)
```
### 5. Scheduler 侧消费 KVCacheConfig 构建 KVCacheManager
Engine Core 层的 Scheduler 组件用同一份 KVCacheConfig 构建自己的内存簿记结构（KVCacheManager），用于调度期的 block 分配与前缀缓存，与 Trace 4 的实际显存分配并行存在（前者是簿记，后者是物理显存）。
### 5a. EngineCore 构造函数中先完成 KV cache 初始化 (`core.py:145`)
得到的 kv_cache_config 随即用于构造 Scheduler
```text
kv_cache_config = self._initialize_kv_caches(vllm_config)
```
### 5b. 构造 Scheduler 并传入配置 (`core.py:156`)
Scheduler 拿到与 worker 一致的 KVCacheConfig 用于调度期簿记
```text
self.scheduler: SchedulerInterface = Scheduler(vllm_config=vllm_config, kv_cache_config=kv_cache_config, ...)
```
### 5c. Scheduler 内部创建 KVCacheManager (`scheduler.py:285`)
调度器持有的簿记结构，负责后续请求的 block 分配、抢占、前缀缓存匹配
```text
self.kv_cache_manager = KVCacheManager(kv_cache_config=kv_cache_config, max_model_len=self.max_model_len, ...)
```


## ModelRunner V2 (MRV2) 的 get_kv_cache_spec 实现
追踪 vLLM GPU Worker 如何在 V1/V2 ModelRunner 间选择，以及 MRV2 (vllm/v1/worker/gpu/) 中 get_kv_cache_spec 的具体实现，并与 V1 (gpu_model_runner.py) 对照。关键跳转：MRV2 委托给 attn_utils 中的模块级函数 [2c]，其核心逻辑与 V1 [3a]-[3d] 几乎一致。
### 1. GPUWorker 根据配置选择 V1 或 V2 ModelRunner
GPU Worker 进程构造阶段，依据 use_v2_model_runner（受 VLLM_USE_V2_MODEL_RUNNER 环境变量或配置默认值控制）决定实例化哪个版本的 ModelRunner，两者共用相同的上层调用接口。
### 1a. V2 开关环境变量定义 (`envs.py:2049`)
未设置时使用配置默认值，决定是否启用 Model Runner V2
```text
"VLLM_USE_V2_MODEL_RUNNER": lambda: maybe_convert_bool(
```
### 1b. GPUWorker 分支判断 (`gpu_worker.py:425`)
Worker 构造函数中根据该标志决定走哪条初始化路径
```text
if self.use_v2_model_runner:
```
### 1c. 导入 MRV2 的 GPUModelRunner (`gpu_worker.py:431`)
V2 实现位于 vllm/v1/worker/gpu/model_runner.py，与 V1 同名但路径不同
```text
from vllm.v1.worker.gpu.model_runner import (  # type: ignore[assignment]
```
### 1d. 实例化 V2 ModelRunner (`gpu_worker.py:436`)
构造 MRV2 实例，后续 KV cache 相关调用均转发至此对象
```text
self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore
```
### 1e. 否则回退导入 V1 ModelRunner (`gpu_worker.py:440`)
V1 实现位于 vllm/v1/worker/gpu_model_runner.py，是原始实现
```text
from vllm.v1.worker.gpu_model_runner import (
```
### 1f. 实例化 V1 ModelRunner (`gpu_worker.py:444`)
默认/回退路径，功能更完整但设计上存在技术债
```text
self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)
```
### 2. MRV2 中 get_kv_cache_spec 的实现路径
GPU Worker 侧接口不区分版本，直接调用 model_runner.get_kv_cache_spec()；MRV2 内部把该方法委托给 attn_utils.py 中独立的模块级函数完成实际的逐层规格收集。
### 2a. GPUWorker 统一入口 (`gpu_worker.py:656`)
不管是 V1 还是 V2 实例，都通过同一个方法名转发
```text
return self.model_runner.get_kv_cache_spec()
```
### 2b. MRV2 ModelRunner 方法 (`model_runner.py:499`)
薄封装，不含业务逻辑，转发给独立工具函数
```text
def get_kv_cache_spec(self):
```
### 2c. 委托给 attn_utils 模块函数 (`model_runner.py:500`)
调用从 gpu.attn_utils 导入的同名函数（非类方法）完成实际逻辑
```text
return get_kv_cache_spec(self.vllm_config)
```
### 2d. 遍历模型中所有 Attention 层 (`attn_utils.py:54`)
从 vllm_config 的 static_forward_context 中提取 AttentionLayerBase 子类实例
```text
attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)
```
### 2e. 逐层询问所需规格 (`attn_utils.py:60`)
跳过不需要 KV cache 的层（如 encoder-only attention）；每层自行决定规格
```text
if spec := attn_module.get_kv_cache_spec(vllm_config):
```
### 2f. 后端定制化规格（MRV2 新增） (`attn_utils.py:62`)
针对 AttentionSpec 额外调用后端的 customize_spec，V1 实现中没有这一步
```text
spec = attn_module.get_attn_backend().customize_spec(spec)
```
### 2g. 汇总为层名到规格的字典 (`attn_utils.py:63`)
最终返回的 dict[str, KVCacheSpec] 会经由 collective_rpc 回传给 EngineCore
```text
kv_cache_spec[layer_name] = spec
```
### 3. 对照：V1 (gpu_model_runner.py) 中的等价实现
V1 ModelRunner 中的同名方法直接作为类方法内联实现，逻辑与 MRV2 几乎一致，但没有独立的 customize_spec 定制化步骤，且代码耦合在同一个大文件中。
### 3a. V1 类方法定义 (`gpu_model_runner.py:7648`)
直接作为 GPUModelRunner 类的实例方法实现，而非委托给独立工具函数
```text
def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
```
### 3b. 同样遍历所有 Attention 层 (`gpu_model_runner.py:7660`)
与 MRV2 [2d] 使用相同的工具函数 get_layers_from_vllm_config
```text
attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
```
### 3c. 逐层询问规格 (`gpu_model_runner.py:7675`)
逻辑与 MRV2 [2e] 一致，但没有后续的 customize_spec 调用
```text
if spec := attn_module.get_kv_cache_spec(self.vllm_config):
```
### 3d. 汇总字典并返回 (`gpu_model_runner.py:7678`)
与 MRV2 [2g] 一样，最终产出 dict[str, KVCacheSpec] 供 GPUWorker 回传
```text
kv_cache_spec[layer_name] = spec
```