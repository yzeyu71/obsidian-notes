## Summary
This issue documents the end-to-end deployment and training workflow for running **Qwen3-4B** GRPO training with Vime on Ascend NPU (Atlas A3), intended as a reference for the community.
Two environment setup methods are provided:
1. **Use docker** — the published `quay.io/ascend/vime:vime-latest` image
   (recommended for validation).
2. **Use source code** — install Vime and its pinned dependencies from source.

| Item | Configuration |
|---|---|
| Image | `quay.io/ascend/vime:vime-latest` |
| Model | `Qwen/Qwen3-4B` |
| Training backend | Megatron (4 NPUs) |
| Rollout backend | vLLM Ascend (4 NPUs) |
| Weight sync | `native` |
| Algorithm | GRPO |
| Dataset | `zhuzilin/dapo-math-17k` |
| Reward type | `math` |

The reference target is an Atlas A3 host with 16 visible NPUs (8 used: 4 actor + 4 rollout). On an 8-NPU host, set `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` in the test script.

## Preparing the Running Environment

Only `python==3.12` is supported. CANN Toolkit, Kernels, and NNAL/ATB `9.0.0` must be installed on the host.

### Use docker

```bash
export IMAGE=quay.io/ascend/vime:vime-latest
# A2:  export IMAGE=quay.io/ascend/vime:vime-a2-latest

docker run -d --name vime-npu -it --net=host --shm-size=1024g \
    --privileged=true \
    --cap-add=SYS_PTRACE \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp:/tmp \
    -v /data:/data \
    -v /path/to:/path/to \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    $IMAGE

docker exec -it vime-npu bash
```

Device names and driver mount paths vary by host; reuse the mounts from a known
working vLLM Ascend container if the layout differs.

### Use source code
 
Run the steps below in a Python 3.12 environment with CANN 9.0.0. A
`quay.io/ascend/vllm-ascend:nightly-main-a3` container can be used as the base.

```bash
export WORKSPACE=/root
cd "${WORKSPACE}"
```

#### Install vLLM and vLLM Ascend

```bash
export VLLM_COMMIT=9090368b650896bf5fc990c921df7eb4c20355a5

git clone https://github.com/vllm-project/vllm.git "${WORKSPACE}/vllm"
git -C "${WORKSPACE}/vllm" checkout "${VLLM_COMMIT}"
VLLM_TARGET_DEVICE=empty pip install -v -e "${WORKSPACE}/vllm"

git clone https://github.com/vllm-project/vllm-ascend.git "${WORKSPACE}/vllm-ascend"
git -C "${WORKSPACE}/vllm-ascend" submodule update --init --recursive
pip install -v -e "${WORKSPACE}/vllm-ascend"
```
#### Install Megatron-LM, Megatron-Bridge, MindSpeed, and Vime

Clone Vime first — the NPU patches applied to the other components ship inside
it. Vime's Ascend NPU adaptation lives on the **`npu`** branch, so clone that
branch (not `main`):

```bash
git clone --branch npu https://github.com/vllm-project/vime.git "${WORKSPACE}/vime"
export PATCH_DIR="${WORKSPACE}/vime/docker/npu_patch"
```

Then install each component in order — clone and checkout the pinned commit,
**apply its patch, then install**. The filename `meagtron_comm.patch` is
intentionally spelled that way in the repository.

##### 1. Megatron-LM

```bash
export MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
git clone https://github.com/NVIDIA/Megatron-LM.git "${WORKSPACE}/Megatron-LM"
git -C "${WORKSPACE}/Megatron-LM" checkout "${MEGATRON_COMMIT}"

git -C "${WORKSPACE}/Megatron-LM" apply --whitespace=nowarn "${PATCH_DIR}/meagtron_comm.patch"
git -C "${WORKSPACE}/Megatron-LM" apply --whitespace=nowarn "${PATCH_DIR}/megatron.patch"

pip install --no-deps --no-build-isolation -e "${WORKSPACE}/Megatron-LM"
```

##### 2. Megatron-Bridge

Used via `PYTHONPATH` (no editable install); it requires `nvidia-modelopt`.

```bash
export MEGATRON_BRIDGE_COMMIT=3fd3768045422d0aa5c97e90a4e6c659aea9acb9
git clone --branch bridge https://github.com/radixark/Megatron-Bridge.git "${WORKSPACE}/Megatron-Bridge"
git -C "${WORKSPACE}/Megatron-Bridge" checkout "${MEGATRON_BRIDGE_COMMIT}"

git -C "${WORKSPACE}/Megatron-Bridge" apply --whitespace=nowarn "${PATCH_DIR}/megatron-bridge.patch"

pip install --no-build-isolation "nvidia-modelopt[torch]>=0.37.0"
```

##### 3. MindSpeed

```bash
export MINDSPEED_COMMIT=fc63de5c48426dd019c3b3f39e65f5bdf56e4086
git clone https://gitcode.com/Ascend/MindSpeed.git "${WORKSPACE}/MindSpeed"
git -C "${WORKSPACE}/MindSpeed" checkout "${MINDSPEED_COMMIT}"

git -C "${WORKSPACE}/MindSpeed" apply --whitespace=nowarn "${PATCH_DIR}/mindspeed.patch"

pip install --no-deps --no-build-isolation -e "${WORKSPACE}/MindSpeed"
```

##### 4. mbridge

No patch is required for mbridge.

```bash
export MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
git clone https://github.com/ISEEKYAN/mbridge.git "${WORKSPACE}/mbridge"
git -C "${WORKSPACE}/mbridge" checkout "${MBRIDGE_COMMIT}"

pip install --no-deps --no-build-isolation -e "${WORKSPACE}/mbridge"
```

##### 5. Vime

Vime itself needs no patch; install its dependencies and the package (already
cloned above):

```bash
pip install -r "${WORKSPACE}/vime/requirements.txt"
pip install "vllm-router>=0.1.14"
pip install --no-deps --no-build-isolation -e "${WORKSPACE}/vime"
```

Build the matching Ascend `torch_memory_saver` wheel. NPU does not actually use
`torch_memory_saver`, but the code still imports and calls it and will break
without it, and there is currently no published Python 3.12 build — so compile
it from source:

```bash
git clone --branch 2026.6.0 https://github.com/sgl-project/sgl-kernel-npu.git "${WORKSPACE}/sgl-kernel-npu"
cd "${WORKSPACE}/sgl-kernel-npu"
bash build.sh -a kernels
bash build.sh -a memory-saver
pip install --no-deps output/torch_memory_saver-0.0.8-cp312-cp312-linux_aarch64.whl
```

## Prepare model and dataset

```bash
mkdir -p /path/to/models /path/to/datasets

hf download Qwen/Qwen3-4B \
  --local-dir /path/to/models/Qwen3-4B

hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /path/to/datasets/dapo-math-17k
```

## Test script

The validation uses `test_qwen3_4B.sh`:

```bash
export SLIME_SCRIPT_TRAIN_BACKEND=megatron
export PYTHONPATH="/root/Megatron-Bridge/src:/root/Megatron-LM/:$PYTHONPATH"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)  # or any free port
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
# source /usr/local/CANN9.0.0/ascend-toolkit/set_env.sh
# source /usr/local/CANN9.0.0/nnal/atb/set_env.sh

SCRIPT_DIR="/root/vime/scripts/"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"
LOG_FILE="/root/vime/train_qwen3_4b_vllm.log"

python train.py \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 4 \
  ${MODEL_ARGS[@]} \
  \
  --hf-checkpoint /path/to/models/Qwen3-4B/ \
  \
  --prompt-data /path/to/datasets/dapo-math-17k/dapo-math-17k.jsonl \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type math \
  \
  --rollout-backend vllm \
  --vllm-weight-sync-mode native \
  --vllm-gpu-memory-utilization 0.6 \
  --vllm-enable-sleep-mode \
  --vllm-max-model-len 4096 \
  \
  --num-rollout 200 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len 2048 \
  --rollout-temperature 1.0 \
  --global-batch-size 256 \
  --balance-data \
  \
  --advantage-estimator grpo \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --kl-coef 0.00 \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 8192 \
  --load /path/to/models/Qwen3-4B \
  --megatron-to-hf-mode bridge \
  \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --micro-batch-size 1 \
  --use-flash-attn \
  \
  --train-memory-margin-bytes 2147483648 \
  2>&1 | tee -a "$LOG_FILE"
```

## Start training

```bash
docker exec -it vime-npu bash   # docker method only
cd /root/vime

# Source these explicitly if not already initialized by the image.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# For the source install, also export:
# export PYTHONPATH="/root/Megatron-Bridge/src:/root/Megatron-LM:/root/vime:${PYTHONPATH}"

cd /root/vime
bash test_qwen3_4B.sh
```

The full log is written to `/root/vime/train_qwen3_4b_vllm.log`.

Running all 200 rollouts is not required for image validation; completing at least 3 rollouts exercises startup, generation, reward calculation, training, and repeated weight synchronization. The run passes when at least 3 rollouts finish, native weight sync succeeds after each step, rewards/losses stay finite, and no HCCL timeout, Ray worker failure, NPU OOM, or vLLM server error occurs.
