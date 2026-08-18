### 常用代码仓
git@github.com:yzeyu71/vime.git
git@github.com:vllm-project/vime.git

### 详细操作步骤

以下是标准流程，主要围绕 GitHub 展开：

1. **Fork 项目**：访问项目主页，点击右上角的 **Fork** 按钮，将项目复制到你自己的 GitHub 账号下
    
2. **克隆到本地**：将你 Fork 的项目克隆到电脑上。注意，地址要换成你账号下的项目地址。
    
    bash
    
    git clone https://github.com/你的用户名/项目名称.git
    cd 项目名称
    # (可选)添加原项目地址为上游，用于后续同步更新，名称通常为 upstream
    git remote add upstream https://github.com/原项目所有者/项目名称.git
    
1. **创建新分支**：在本地创建一个新分支，分支名最好能反映修改内容，避免直接在 `main` 或 `master` 分支上修改
    
    bash
    
    git checkout -b feat/你的功能描述
    
    |类型|命名格式|示例|
	|---|---|---|
	|新功能|`feature/描述` 或 `feat/描述`|`feature/user-auth`、`feat/payment-gateway`|
	|修复bug|`bugfix/描述` 或 `fix/描述`|`bugfix/login-error`、`fix/空指针异常`|
	|紧急修复（线上）|`hotfix/描述`|`hotfix/cve-2023-1234`、`hotfix/memory-leak`|
	|文档更新|`docs/描述`|`docs/api-readme`|
	|性能优化|`perf/描述`|`perf/reduce-bundle-size`|
	|重构|`refactor/描述`|`refactor/logger-module`|
	|试验/临时|`wip/描述` (work in progress)|`wip/new-architecture`|
	|版本发布|`release/版本号`|`release/v1.2.0`|
	
1. **编码与提交**：在新分支上完成代码修改，然后提交
    
    bash
    
    git add .
    git commit -s -m "feat: 简洁明了地描述你的更改"
    
2. **同步上游更新**：在推送前，**务必先同步项目的最新代码**，减少后续冲突。
    
    bash
    
    git fetch upstream
    git checkout main
    git merge upstream/main
    git checkout feat/你的功能描述
    git merge main
    
1. **推送到 GitHub**：将本地分支推送到你 GitHub 上的远程仓库
    
    bash
    
    git push origin feat/你的功能描述
    
2. **发起 Pull Request**：
    
    - 打开你 Fork 的 GitHub 仓库页面，通常顶部会出现 **“Compare & pull request”** 按钮
        
    - 若没看到，可切换到你的 `feat/你的功能描述` 分支，点击 **Contribute** 按钮后，再点击 **Open pull request
        
3. **填写 PR 信息**：
    
    - **标题**：用一句话高度概括修改内容
        
    - **描述**：清晰说明背景、修改内容、如何测试等
        
    - **关联 Issue**：使用 `Closes #123` 等关键词，在 PR 合并时能自动关闭关联的 Issue
        
    - **勾选授权**：确保勾选 **“Allow edits from maintainers”**，方便维护者直接帮你微调
        
4. **后续等待与修改**：创建后，等待维护者 Review。若有修改意见，直接在本地你的分支上修改、提交并推送，PR 会自动更新





地将**源仓库（上游）的分支**同步到你的**个人远程仓库（Origin）**里。

整个过程分五步走，只需要用到几个基础的 Git 命令：

### 📝 操作步骤

1. **配置上游 (One-time Setup)**：在你的本地仓库，将**原始项目**的地址添加为一个名为 `upstream` 的新远程仓库，这是所有后续同步的基础[](https://www.cnblogs.com/codedingzhen/p/19699015)[](https://juejin.cn/post/7438845912896192524)。
    
    bash
    
    git remote add upstream https://github.com/【原作者/原组织名】/【项目名】.git
    
2. **获取分支信息**：从上游仓库抓取所有分支的更新信息，这会在本地同步原项目的分支列表，为后续操作做准备[](https://www.cnblogs.com/codedingzhen/p/19699015)[](https://blog.csdn.net/titan__/article/details/149535857)。
    
    bash
    
    git fetch upstream
    
3. **创建并切换到新分支**：基于 `upstream` 的某个目标分支，在你的本地创建一个同名分支，并切换过去。建议使用 `git checkout -b` 命令[](https://www.cnblogs.com/codedingzhen/p/19699015)。
    
    bash
    
    git checkout -b <目标分支名> upstream/<目标分支名>
    
4. **推送至你的 Fork**：将此新分支推送到你 `origin` 仓库，也就是你在 GitHub 上的个人远程仓库中[](https://www.cnblogs.com/codedingzhen/p/19699015)。
    
    bash
    
    git push -u origin <目标分支名>
    
5. **（可选）从 GitHub 上直接创建**：如果上游仓库新增分支，你也可以在 GitHub 上你的项目页面，通过分支选择下拉菜单直接创建分支。
    

### ⚠️ 注意事项

操作时有几个点值得留心：

- **命名冲突**：如果本地或 `origin` 里已存在同名分支，使用 `git push` 时可能会报错。通常的解决方法是先确保这个分支是你需要的，再考虑用 `-f` 参数强制覆盖。
    
- **一次性同步所有分支**：如果想把上游所有的分支都同步过来，可以配合 `git branch -r` 和 `git push --all` 等命令，写一个简单的脚本来批量处理[](https://blog.csdn.net/titan__/article/details/149535857)。
    
- **定期同步分支上的更新**：对于已经同步过来的分支，需要定期执行 `git fetch upstream` 和 `git merge` 命令来获取源仓库最新的代码变更[](https://blog.csdn.net/m0_46437343/article/details/149980860)。
    
- **巧用 GitHub Actions**：如果你想在 GitHub 上自动化同步，也可以利用 GitHub Actions。通过配置 `workflow`，在检测到上游仓库有变动时自动更新。


# 修改并修正提交
git commit --amend --no-edit
# 直接强制推送（用 --force-with-lease 比 -f 更安全）
git push --force-with-lease origin <你的分支名>


    git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"