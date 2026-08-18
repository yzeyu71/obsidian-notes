文档：[Buildkite Pipelines | Buildkite Documentation](https://buildkite.com/docs/pipelines)

流水线通过仓库根目录下的 `.buildkite/pipeline.yml` 配置文件定义

这个 Buildkite 项目位于 `vllm` 组织下，可以通过这个链接直接访问： [https://buildkite.com/vllm/vllm-omni](https://buildkite.com/vllm/vllm-omni)

创建buildkite流水线
![[Pasted image 20260611113908.png|697]]

控制使用哪个pipeline.yaml执行
command: "buildkite-agent pipeline upload .buildkite/custom.yml"

执行结果
![[buildkite执行结果-test.png|697]]

创建池子
1、在Agents界面 选择或者创建一个新的cluster
![[Pasted image 20260708101953.png]]
2、创建queue
![[Pasted image 20260624105949.png]]
3、创建agent，点击确认后会生成token，保存token配置
![[Pasted image 20260624110149.png]]

4、设备连接到buildkite 使用A3测试环境为例
安装buildkite-agent
TOKEN="INSERT-YOUR-AGENT-TOKEN-HERE" bash -c "`curl -sL https://raw.githubusercontent.com/buildkite/agent/main/install.sh`"
![[Pasted image 20260624112104.png|697]]
修改配置 设置agent加入的queue
vim ~/.buildkite-agent/buildkite-agent.cfg
![[Pasted image 20260624112604.png]]
启动agent，连接成功
~/.buildkite-agent/bin/buildkite-agent start
![[Pasted image 20260624112722.png]]
queue可以看到新增的agent
![[Pasted image 20260624112709.png]]

设置PR触发buildkite执行
1、在buildkite设置
![[Pasted image 20260706165503.png|689]]
2、在github设置对应的webhook
![[Pasted image 20260706165527.png]]
配置PR触发
![[Pasted image 20260713201420.png]]
![[Pasted image 20260713201448.png]]

控制什么label能触发配置
![[Pasted image 20260713201400.png]]

pipeline如何使用新增的agent
1、确认agent加到哪个cluster中，如CI
![[Pasted image 20260706165221.png]]2、
将pipeline添加到对应的cluster中
![[Pasted image 20260706165329.png]]


![[Pasted image 20260703174602.png]]

获取PR的label
![[Pasted image 20260713112339.png]]

查看缓存空间
![[Pasted image 20260713114011.png|697]]

![[Pasted image 20260721195649.png]]