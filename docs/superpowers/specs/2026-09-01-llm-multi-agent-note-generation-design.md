# LLM Multi-Agent Note Generation Design

## Goal

将 BiliNote 当前由固定 `Planner` 和 `PlanExecutor` 控制的笔记生成流程，演进为一个具备真实 LLM 决策、工具调用、观察、重规划和受限返工能力的 Multi-Agent 系统，同时保持普通用户现有的笔记生成接口、缓存和失败降级行为。

## Problem and boundary

当前下载、字幕、转写、笔记生成和视觉增强虽然被命名为 Agent，但执行顺序和路由主要由程序规则决定。首版不再把确定性媒体处理能力包装成 LLM Agent，而是把它们暴露为受限工具；真正的 Agent 只负责需要语义判断和质量决策的工作。

首版范围是单次笔记生成任务内的四个 LLM 角色：`SupervisorAgent`、`ContentAgent`、`VisualAgent` 和 `ReviewerAgent`。现有 RAG 问答 Tool Calling 循环继续独立运行，不并入生成任务状态。现有视觉扫描、候选帧生成、评分、去重和 Markdown 合成继续作为确定性工具实现。

## Architecture

```text
GenerationRequest and task artifacts
                |
                v
        SupervisorAgent
          /    |     \
         /     |      \
ContentAgent VisualAgent ReviewerAgent
         \     |      /
          \    |     /
       structured observations
                |
          SupervisorAgent
```

`SupervisorAgent` 是唯一拥有全局路由权的 Agent。它读取任务目标、当前状态摘要、已产生的产物、失败记录和剩余预算，输出严格结构化的下一步决策。决策可以是调用一个工具、委派一个专业 Agent、要求返工、降级或完成任务。Supervisor 不直接执行下载、转写或文件写入，而是通过工具注册表调用受限能力。

`ContentAgent` 负责内容理解任务：在已有字幕或转录材料上生成结构化笔记草稿，必要时调用转录查询工具补充指定区间，并根据 Reviewer 的缺陷列表修改草稿。它不决定整个任务是否结束，也不能调用视觉或任意系统工具。

`VisualAgent` 负责视觉证据任务：根据笔记章节和用户目标判断是否值得补充截图，调用现有候选画面和截图质量工具，并在候选不足时请求有限次数的重新搜索或明确降级。底层 FFmpeg、视频读取和质量阈值仍由工具实现。

`ReviewerAgent` 负责质量检查，输出通过/不通过、缺陷分类、证据和建议目标。缺陷必须归类为 `content`、`visual`、`source` 或 `system`，供 Supervisor 选择对应的返工路径；Reviewer 不直接修改笔记。

## State and protocol

运行状态使用结构化对象保存，不依赖 Agent 之间传递自由文本作为控制协议。状态至少包括：任务目标、视频标识和元数据摘要、转录引用、当前 Markdown 草稿、视觉产物引用、评审结果、失败记录、当前迭代次数、内容返工次数、视觉重试次数、工具调用次数、预算和最终状态。

每次模型决策必须解析为 Pydantic 模型，包含 `action`、`agent` 或 `tool`、结构化参数、原因和预期结果。每次工具执行都返回统一的 `Observation`，至少包含成功标志、可供下一次决策使用的数据摘要、产物引用、错误分类和耗时。无效的模型输出、未知工具、参数校验失败和越权调用都转为结构化失败 Observation，并交回 Supervisor；不得执行任意模型生成的 Python、Shell 或文件路径操作。

## Control loop

任务开始时 Supervisor 获得用户目标和已有缓存摘要。它可以先查询视频信息或缓存转录；当字幕不存在、不完整或 Reviewer 明确指出来源不足时，Supervisor 可以选择字幕工具、音频转写工具或指定区间检索。获得足够材料后，Supervisor 委派 ContentAgent 生成草稿。草稿交给 ReviewerAgent 评审；内容缺陷最多触发两次 ContentAgent 返工。若用户请求截图或 Reviewer 判断章节需要视觉证据，Supervisor 委派 VisualAgent；视觉失败最多重试两次，之后保留无截图的有效笔记并记录降级原因。最终 Reviewer 通过，或达到安全预算后由 Supervisor 输出最终结果。

首版硬限制为 Supervisor 最多 12 次决策、ContentAgent 最多 2 次返工、VisualAgent 每个任务最多 2 次重试，并设置工具调用超时和允许工具白名单。达到限制时系统必须返回已有的最佳有效笔记、明确的降级状态和诊断信息，不得无限循环。

## Integration and compatibility

现有 FastAPI 入口、任务轮询响应、`generation_token`、`enhance_token`、本地缓存路径和 `PARTIAL_SUCCESS` 语义保持兼容。Agent Runtime 通过依赖注入复用现有 downloader、transcriber、GPT、视觉增强服务和结果存储。旧固定执行器在迁移期间保留为 fallback；只有显式启用 Agent Runtime 且能力配置满足条件时才走 LLM 闭环，模型调用失败可降级到当前确定性流程，不能阻塞普通笔记生成。

Agent Runtime 不允许让 LLM 直接控制并发、线程、任务状态文件或版本令牌。程序在写入结果前验证当前 `generation_token`，在视觉增强写入前验证 `enhance_token`，继续阻止旧运行覆盖新运行。

## Trace and observability

每次任务写入可序列化的 Agent Trace，记录时间、任务和生成令牌、Agent、动作、工具、结构化输入摘要、Observation 摘要、迭代、耗时、错误分类和最终结果。Trace 不保存 API key，也不默认保存完整字幕或完整 Prompt。Trace 用于调试、回放和面试展示，能够还原“字幕失败→选择转写→内容生成→评审返工→视觉重试/降级→完成”的真实路径。

## Testing and acceptance

单元测试覆盖协议校验、工具白名单、预算限制、路由决策、Observation 记录和 token 隔离。Runtime 测试使用假的 GPT 响应序列验证真实闭环，而不是只断言某个函数被调用。集成测试验证字幕失败时选择转写、内容评审失败时返工、视觉质量不足时重试后降级，以及模型异常时 fallback 不破坏现有结果。现有后端 Agent、视觉和笔记生成回归测试必须继续通过；真实供应商调用和真实视频流程作为独立验收门，不用 mock 测试替代。

## Explicit non-goals

首版不实现跨任务长期记忆、不实现多个 Agent 的自然语言群聊、不允许任意工具发现、不替换现有 RAG 问答 Agent、不把下载器/转写器/Markdown Composer 改造成 LLM Agent，也不承诺多实例持久化队列能力。
