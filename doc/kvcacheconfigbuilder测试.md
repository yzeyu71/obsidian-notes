export IMAGE=quay.io/ascend/vllm-ascend:latest
docker run -itd --name vllm-ascend-yzy \
    --privileged \
    --net=host \
    --shm-size=1024g \
    --cap-add=SYS_PTRACE \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp:/tmp \
    -v /root/.cache:/root/.cache \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    ${IMAGE} bash
	
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh


-i https://pypi.tuna.tsinghua.edu.cn/simple
requirements.txt --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi

git fetch origin pull/53558/head:pr-53558
git checkout pr-53558

git fetch origin pull/14889/head:pr-14889
git checkout pr-14889

export ASCEND_RT_VISIBLE_DEVICES=0,1

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --trust-remote-code \
    --port 8001

根据两个 PR 的代码变更和描述，以下是需要执行的测试用例和模型建议。

---

## 一、两个 PR 的核心变更回顾

### PR #53558 (vLLM 主库)
引入可插拔的 `KVCacheConfigBuilder`，解析顺序为：**平台覆盖 > 模型声明 > 默认**。核心代码从 `kv_cache_utils` 迁移到新的 `kv_cache_planning.py` 模块。

### PR #14889 (vLLM-Ascend)
适配主库的 Builder 机制，将 DeepSeekV4 的 KV Cache 规划从 monkey-patch 迁移到正式的 `AscendKVCacheConfigBuilder`。核心变更：
- 新增 `vllm_ascend/worker/kv_cache_config_builder.py`
- `NPUPlatform.get_kv_cache_config_builder_cls()` 注册 Ascend builder
- 移除 DeepSeekV4 的 monkey-patch，迁移到 builder 中
- 保留 `resolve_kv_cache_block_sizes` patch（Ascend DCP 对齐语义与上游不同）

---

## 二、需要执行的测试用例

### 1. 主库单元测试（PR #53558）

| 测试文件 | 说明 |
|----------|------|
| `tests/v1/core/test_kv_cache_config_builder.py` | Builder 加载、解析优先级（默认 builder、模型声明 builder、平台覆盖 builder） |

你已执行过此测试，**5 个用例全部通过**，功能验证通过。
![](../assets/Pasted%20image%2020260825161346.png)

### 2. vLLM-Ascend 单元测试（PR #14889）

PR 描述中明确列出的单元测试：

| 测试文件 | 说明 |
|----------|------|
| `tests/ut/patch/platform/test_prefix_cache_cp_patches.py` | Prefix Cache + CP（Context Parallel）patch 的兼容性 |
| `tests/ut/kv_offload/test_mooncake_connector.py` | Mooncake KV offload 连接器的导入适配 |

**执行命令**：
```bash
cd /vllm-workspace/vllm-ascend
pytest tests/ut/patch/platform/test_prefix_cache_cp_patches.py -v
pytest tests/ut/kv_offload/test_mooncake_connector.py -v
```

### 3. 端到端（E2E）手动回归测试

PR #14889 要求进行以下手动回归验证：

| 测试场景 | 验证目标 |
|----------|----------|
| DeepSeekV4 KV Cache Layout | 验证非 packed shared-tensor layout 正确生成 |
| Mamba / Hybrid 模型 | 验证混合 KV Cache 的 group 规划正确 |
| `resolve_kv_cache_block_sizes` 多 group + DCP | 验证 Ascend DCP 对齐语义 |

### 4. vLLM-Ascend CI 测试体系

vLLM-Ascend 的 CI 测试分为：
- **singlecard**：单卡测试，每个 PR 和每日测试都会运行
- **multicard**：多卡测试

PR 合并前 CI 会自动运行这些测试，但建议你在本地也执行关键的单卡测试。

---

## 三、需要测试的模型

根据 PR 的变更范围和影响，建议测试以下模型类别：

### 1. DeepSeekV4（核心目标）
这是 PR #14889 的核心适配对象。DeepSeekV4 使用特殊的 **非 packed shared-tensor layout** 和 **grouping 逻辑**，必须重点验证。

**建议测试**：
```bash
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0

python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V4 \
    --tensor-parallel-size <num_npus> \
    --max-model-len 8192 \
    --trust-remote-code
```

**验证点**：
- 启动日志中是否出现 `AscendKVCacheConfigBuilder` 的相关日志
- KV Cache 是否能正常分配，无 OOM
- 推理结果是否正确（与 baseline 对比）

### 2. Mamba / Hybrid 模型
PR #14889 明确提到需要回归测试 Mamba/hybrid 模型。hybrid 模型（如 Mamba + Attention 混合）的 KV Cache group 规划逻辑与纯 Attention 模型不同。

**建议测试模型**：
- `state-spaces/mamba-2.8b` 或类似 Mamba 架构模型
- 其他 hybrid 架构模型（如带有 sliding window attention 的模型）

**验证点**：
- Hybrid KV Cache 的 group 分配是否正确
- 多 group 场景下 `resolve_kv_cache_block_sizes` 是否正常工作

### 3. 普通 Transformer 模型（回归验证）
作为 fallback 路径的验证，确保非 DeepSeekV4 模型不受影响。

**建议测试模型**：
- `Qwen/Qwen2.5-7B-Instruct`（或 1.5B 等小模型，便于快速测试）
- `meta-llama/Llama-3.2-3B-Instruct`

**验证点**：这些模型应走默认的 vLLM planning 逻辑，行为与之前完全一致。

### 4. 多卡 + DCP 场景
PR #14889 特别提到需要验证 `resolve_kv_cache_block_sizes` 在 **多 group + DCP（Context Parallel）** 场景下的正确性。

**建议测试**：
```bash
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V4 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code
```

---

## 四、总结

| 测试类型 | 具体内容 | 优先级 |
|----------|----------|--------|
| 主库 UT | `test_kv_cache_config_builder.py`（5 个用例） | ✅ 已完成 |
| Ascend UT | `test_prefix_cache_cp_patches.py` | 高 |
| Ascend UT | `test_mooncake_connector.py` | 高 |
| E2E 推理 | DeepSeekV4 单卡/多卡推理 | **最高** |
| E2E 推理 | Mamba/Hybrid 模型推理 | 高 |
| E2E 推理 | 普通 Transformer 模型（Qwen/Llama）推理 | 中（回归验证） |
| E2E 推理 | 多卡 + DCP 长文本场景 | 中 |

**建议执行顺序**：
1. 先跑完两个 Ascend 单元测试
2. 用 DeepSeekV4 做单卡推理，确认 builder 正常加载
3. 用 Qwen/Llama 做回归验证，确认 fallback 路径正常
4. 如有条件，测试 Mamba 和多卡 DCP 场景



模型启动后，可以通过以下步骤进行测试，以验证其功能正确性和性能。

### 🚀 第一步：快速验证服务是否正常运行

服务启动后，首先通过查询模型列表来确认它已准备就绪。

**1. 检查服务状态**
使用 `curl` 命令查看服务是否正常响应：
```bash
curl http://localhost:8000/v1/models
```
如果服务正常运行，该命令会返回一个包含已加载模型信息的 JSON 对象。

**2. 发送一个简单的推理请求**
用一个简单的提示词（prompt）测试基本的文本生成功能。
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "What is the capital of France?",
    "max_tokens": 50
  }'
```
如果收到包含生成文本的 JSON 响应，则说明模型已成功加载并可以执行推理。

### ✅ 第二步：进行更全面的功能测试

为了更贴近实际使用场景，建议从“离线推理”和“在线服务”两个维度进行测试。

*   **离线推理测试 (Offline Inference)**：使用 `vllm` 的 `run-batch` 命令或编写一个简短的 Python 脚本，批量处理一组固定的提示词，验证输出结果的正确性和一致性。

*   **在线服务测试 (Online Serving)**：通过向服务的 `/v1/completions` 或 `/v1/chat/completions` 端点发送请求来测试。关键在于验证其**OpenAI API 兼容性**，确认请求和响应格式符合预期。

### 📊 第三步：运行性能基准测试 (Benchmark)

vLLM 提供了强大的 `vllm bench` 命令行工具来进行性能评估。运行前，建议先安装必要的依赖：
```bash
pip install vllm[bench]
```

*   **测试吞吐量 (Throughput)**：衡量模型每秒能生成的 token 数量。
    ```bash
    vllm bench throughput \
      --model Qwen/Qwen3-0.6B \
      --input-len 128 \
      --output-len 128 \
      --num-prompts 1000
    ```
    这个命令会模拟处理 1000 个请求，每个请求的输入和输出长度均为 128 个 token。

*   **测试在线服务性能 (Serving Performance)**：在服务已经启动的情况下，模拟并发请求，测量包括首 token 延迟 (TTFT) 在内的关键指标。
    ```bash
    vllm bench serve \
      --base-url http://localhost:8000 \
      --model Qwen/Qwen3-0.6B \
      --num-prompts 200
    ```
    这个命令会向运行中的服务发送 200 个请求。测试结果会包含如下关键指标：
    *   **吞吐量 (Throughput)**：系统每秒处理的 token 总数或请求总数。
    *   **首 token 延迟 (TTFT, Time to First Token)**：从发送请求到收到第一个 token 的时间。
    *   **每输出 token 时间 (TPOT, Time Per Output Token)**：生成每个输出 token 所需的平均时间。
    *   **端到端延迟 (End-to-End Latency)**：完成整个请求的总时间。

### 🧪 第四步：运行项目自带的端到端 (E2E) 测试

`vllm-ascend` 项目本身包含了一些 E2E 测试用例，可以用于验证特定功能。这些测试通常位于 `tests/e2e` 目录下。

你可以使用 `pytest` 来运行它们，例如：
```bash
pytest tests/e2e/singlecard/test_xlite.py -v
```
通过运行这些现有的测试，可以验证你的修改是否破坏了任何已有功能。

### 💎 总结

整个测试流程可以概括为：
1.  **冒烟测试**：用 `curl` 检查服务状态和基本推理能力。
2.  **功能测试**：进行离线批量推理和在线 API 请求，验证输出正确性。
3.  **性能测试**：使用 `vllm bench` 命令获取吞吐量和延迟等关键性能数据。
4.  **回归测试**：运行项目自带的 E2E 测试用例，确保没有引入新问题。

如果在测试中遇到任何错误，随时可以把日志贴出来，我帮你一起分析。