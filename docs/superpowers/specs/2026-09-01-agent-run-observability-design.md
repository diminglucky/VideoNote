# Agent Run 可观测性设计

## 目标

将真实 LLM Multi-Agent 运行时产生的决策、工具调用、专业 Agent 委派、Reviewer 结果、返工、降级和最终状态，以安全、可轮询的方式展示在 BiliNote 产品中。该功能只观察现有运行时，不改变 Supervisor、Content、Visual、Reviewer 的决策逻辑和工具权限。

## 范围与非目标

本阶段包含：

- 后端从每个任务的 JSONL Trace 读取事件并生成限长安全摘要。
- 以 `generation_token` 的不可逆短哈希作为 `generation_id`，避免重试任务混用旧 Trace。
- `/task_status/{task_id}` 在有当前代 Trace 时返回 `agent_run`。
- React 任务状态模型、轮询和笔记页面展示 Agent 时间线。
- 没有 Trace 的旧任务保持现有响应和界面行为。

本阶段不包含：

- SSE、WebSocket 或新的实时通信协议。
- 修改 Agent 的规划、工具白名单、预算或降级策略。
- 暴露 Prompt、API Key、完整转录、完整 Markdown、文件路径或模型原始响应。
- 将 RAG 问答循环合并到视频笔记 Agent Run。

## 架构

```text
JsonlTraceStore
       |
       v
AgentRunProjector -- generation_id filter --> safe agent_run JSON
       |
       v
task_status API -- existing polling --> Zustand Task
                                      |
                                      v
                              AgentRunTimeline
```

`JsonlTraceStore` 为每个生成代写入稳定的 `generation_id`。该 ID 只由 `generation_token` 做 SHA-256 短哈希得到，不在 Trace 或 API 中返回原始 token。`AgentRunProjector` 只读取允许字段，限制事件数量和文本长度，并将未知或损坏行跳过。API 使用请求中的 generation token 计算目标 ID；没有 token 时使用当前状态文件中的 token。旧 Trace 没有 `generation_id` 时不在带 token 的请求中返回，防止历史代污染当前重试。

## API 数据契约

当当前代存在有效 Trace 时，任务状态响应增加：

```json
{
  "agent_run": {
    "mode": "llm_multi_agent",
    "status": "running|completed|degraded|unknown",
    "active_agent": "supervisor|content|visual|reviewer|null",
    "counters": {
      "decisions": 3,
      "tool_calls": 2,
      "llm_calls": 5,
      "content_revisions": 0,
      "visual_retries": 1
    },
    "diagnostics": ["..."],
    "events": [
      {
        "id": "...",
        "timestamp": "...",
        "kind": "decision|tool|observation|final",
        "agent": "supervisor",
        "action": "delegate",
        "tool": null,
        "target_agent": "content",
        "ok": true,
        "summary": "委派 ContentAgent 生成笔记"
      }
    ]
  }
}
```

事件最多返回 60 条，摘要最多 240 个字符，诊断最多 10 条。只返回 `kind`、角色、动作、工具名、目标角色、成功状态、错误类型和限长摘要；忽略事件中的 `data`、Prompt、原始模型消息和 Markdown。事件 ID 由行号和 generation_id 组成，便于前端稳定比较而不泄露路径。

## 前端行为

`AgentRunTimeline` 是一个可折叠的轻量卡片：运行中展示“正在由哪个 Agent 决策”和最近事件；完成后展示状态、预算计数和完整限长事件；降级时突出诊断。它只在 `agentRun` 存在且至少有事件或计数时渲染，普通固定 Workflow 任务不出现空占位。

轮询只在 Agent 摘要发生变化时更新 Zustand，避免无意义渲染。重试时 generation token 变化，旧 Agent Run 被替换；后端拒绝旧代 Trace，前端不会显示旧任务的事件。

## 错误与兼容性

- Trace 文件不存在、为空、JSON 损坏或权限读取失败：返回无 `agent_run` 的普通任务状态，不影响笔记结果。
- Trace 中出现未知字段或未知事件类型：忽略字段，保留可识别的基础事件。
- 当前代没有 `final_state`：状态为 `running`；任务本身结束但 Trace 不完整时状态为 `unknown`，不修改任务状态机。
- 所有 projection 失败都记录后降级为空摘要，不让轮询接口 500。

## 验收标准

1. 带当前 generation_id 的 Trace 可被投影为安全 `agent_run`，完整 Markdown、Prompt、token 和路径不会出现在响应中。
2. 重试任务只显示当前代事件，旧代事件不会混入。
3. 损坏 Trace 不影响 `/task_status` 正常返回。
4. 旧任务和固定 Workflow 任务不出现 Agent 面板。
5. 后端投影测试通过，前端 lint/build 通过。
