[[issue_qwen3_4b_ascend_container_validation]]

拉起 vime docker
```bash
export IMAGE=quay.io/ascend/vime:vime-latest
# A2:  export IMAGE=quay.io/ascend/vime:vime-a2-latest
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
![[Pasted image 20260609195401.png|697]]

下载模型权重
hf download Qwen/Qwen3-4B --local-dir /path/to/Qwen/Qwen3-4B
补充国内代理
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen3-4B --local-dir /path/to/Qwen/Qwen3-4B
_
cd /root/vime
bash test_qwen3_4B.sh
![[ffbbf131558b569039c2a4b193e9a6d5.png|697]]

