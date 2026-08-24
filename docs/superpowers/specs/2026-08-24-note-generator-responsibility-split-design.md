# NoteGenerator 职责拆分设计

## 1. 文档信息

- 日期：2026-08-24
- 状态：已确认，进入实现
- 范围：后端笔记生成流程
- 主要代码：[backend/app/services/note.py](../../../backend/app/services/note.py)
- 关联执行器：[backend/app/agents/executor.py](../../../backend/app/agents/executor.py)

## 2. 摘要

当前 `NoteGenerator` 已经将下载、转写、笔记生成和 Markdown 组合拆分到多个 Agent，并由 `PlanExecutor` 执行计划；但它仍然同时负责运行时组件创建、单次任务状态保存、任务生命周期写入、结果元数据持久化和主流程编排。

本设计采用渐进式拆分，不重写现有 Agent 或执行计划。新增生成请求/上下文模型、运行时工厂、任务生命周期服务和结果存储边界，使 `NoteGenerator` 逐步收窄为流程 Orchestrator，同时保持现有路由调用方式、缓存行为、状态文件格式和返回结果兼容。

## 3. 当前实现事实

### 3.1 已经存在的拆分

`backend/app/services/note.py` 当前已经使用：

- `DownloadAgent`：下载媒体并生成音频元数据；
- `TranscriptAgent`：读取平台字幕、缓存转写或执行转写；
- `NoteWriterAgent`：基于转写和 GPT 生成 Markdown；
- `MarkdownComposerAgent`：处理链接和截图；
- `PlanExecutor`：按照 `ExecutionPlan` 调度上述 Agent。

因此，本次设计不再创建第二套下载、转写或笔记生成实现，也不把 `PlanExecutor` 和 Agent 逻辑搬回其他类。

### 3.2 当前仍然混合的职责

`NoteGenerator.__init__` 当前负责读取转写配置、初始化转写器、组装 `AgentRuntimeServices` 和多个 Agent。`generate()` 当前负责构造执行计划、选择 Downloader、创建 GPT、创建 `AgentRuntimeContext`、调用执行器、更新任务状态和保存元数据。

同一个文件还包含：

- `_init_transcriber()`：转写器创建；
- `_get_gpt()`：Provider 查询、`ModelConfig` 创建和 GPT 工厂调用；
- `_get_downloader()`：平台到 Downloader 的选择；
- `_visual_screenshot_agent()`：截图 Agent 创建；
- `_update_status()` / `_handle_exception()`：任务状态和异常写入；
- `_save_metadata()` / `delete_note()`：任务元数据持久化。

此外，`execution_plan`、`video_path`、`video_img_urls` 被保存为 `NoteGenerator` 实例属性，但它们属于一次生成运行时的中间状态。

### 3.3 兼容性事实

当前路由在处理任务时按请求创建 `NoteGenerator(generation_token=...)`，外部仍通过 `generate()` 传入 URL、平台、模型、格式、视觉理解和输出相关参数。实现必须保留这些入口和返回类型，避免把本次架构调整扩大为 API 变更。

## 4. 目标与非目标

### 4.1 目标

1. 让 `NoteGenerator` 只负责生成流程编排和跨步骤决策。
2. 将每次生成的输入和中间状态放入显式对象，不再依赖生成器实例字段保存任务状态。
3. 将转写器、GPT、Downloader、截图 Agent 的创建集中到运行时工厂。
4. 将任务状态、失败状态和 generation token 的写入集中到任务生命周期服务。
5. 将视频任务元数据保存和删除集中到结果存储边界。
6. 保持现有 Agent、`PlanExecutor`、缓存文件命名、状态文件格式、路由参数和 `NoteResult` 行为兼容。
7. 为新的边界补充可独立测试的单元测试，降低后续增加生成步骤的风险。

### 4.2 非目标

1. 不重写 `PlanExecutor` 的依赖解析和可选步骤语义。
2. 不重写 Downloader、Transcriber、GPT、视觉截图服务或 Agent 内部算法。
3. 不改变 HTTP API、前端轮询协议、状态 JSON 字段或缓存文件格式。
4. 不在本次设计中引入新的任务队列、数据库、依赖注入框架或异步执行模型。
5. 不同时改造前端、浏览器扩展或 Tauri 桌面端。
6. 不为了形式上的“纯调度器”拆出大量只转发一个函数的类。

## 5. 方案选择

### 方案 A：仅移动工厂方法

将 `_init_transcriber()`、`_get_gpt()`、`_get_downloader()` 和 `_visual_screenshot_agent()` 移到一个工厂类，其他逻辑保留在 `NoteGenerator`。

优点是改动最小、回归风险低；缺点是任务状态、异常和数据库写入仍然混在主流程中，无法解决实例状态与持久化边界问题。

### 方案 B：请求/上下文 + 运行时工厂 + 生命周期/结果存储

新增 `GenerationRequest`、`GenerationContext`、`NoteRuntimeFactory`、`TaskLifecycleService` 和 `NoteResultStore`，保留现有 Agent 与 `PlanExecutor`，按阶段迁移职责。

优点是边界清晰、可以逐步提交和验证，并能消除单次任务状态挂在 `NoteGenerator` 上的问题；缺点是初期会增加少量数据类和构造代码。此方案是本设计采用的方案。

### 方案 C：全面重做 Agent runtime 和后台任务入口

同时重写 Agent 服务接口、执行器、后台任务入口和结果模型。

该方案可以获得更彻底的架构统一，但会扩大改动面，容易同时影响缓存命中、视觉增强、generation token 和旧任务恢复，不符合当前“渐进式拆分且行为兼容”的目标。

## 6. 目标架构

```text
note router / background task
            |
            v
      NoteGenerator.generate()
            |
            +--> GenerationRequest
            +--> NoteRuntimeFactory.create(request)
            |       +--> Transcriber
            |       +--> GPT
            |       +--> Downloader
            |       +--> AgentRuntimeServices
            |
            +--> build_note_execution_plan(request)
            +--> PlanExecutor.run(plan, GenerationContext)
            |
            +--> TaskLifecycleService.mark_saving/success/failed()
            +--> NoteResultStore.save_metadata()
            |
            v
        NoteResult
```

### 6.1 `GenerationRequest`

`GenerationRequest` 表示一次生成的不可变输入，至少包含：

- `video_url`
- `platform`
- `quality`
- `task_id`
- `model_name`
- `provider_id`
- `formats`
- `link`
- `screenshot`
- `style`
- `extras`
- `output_path`
- `video_understanding`
- `video_interval`
- `grid_size`
- `defer_screenshots`
- `generation_token`

它负责在入口处统一计算 `wants_link`、`wants_screenshot` 和格式集合，避免主流程中散落重复判断。对外的 `generate()` 参数保持兼容；请求对象只作为内部边界。

### 6.2 `GenerationContext`

`GenerationContext` 表示一次执行过程中的可变状态，复用现有 `AgentRuntimeContext` 能表达的字段，并补充最终生成所需的源链接和诊断信息。它至少包含：

- 请求对应的任务标识和媒体参数；
- Downloader、GPT 等运行时依赖；
- 音频元数据、转写结果、Markdown；
- 视频路径和视频图片 URL；
- 缓存文件路径；
- 诊断信息和最终 `NoteResult`。

本次实现优先复用或扩展 `AgentRuntimeContext`，避免同时维护两套相同的上下文模型。`NoteGenerator` 不再把 `execution_plan`、`video_path`、`video_img_urls` 保存为实例字段。

### 6.3 `NoteRuntimeFactory`

职责是根据 `GenerationRequest` 创建一次执行所需的运行时依赖，并返回 `PlanExecutor` 所需的 Agent 集合和 `GenerationContext` 初始依赖。

它负责：

1. 从 `TranscriberConfigManager` 读取转写器配置并调用现有 `get_transcriber()`；
2. 根据 `provider_id` 查询 Provider，创建 `ModelConfig`，调用现有 `GPTFactory`；
3. 从 `SUPPORT_PLATFORM_MAP` 选择 Downloader，并保留现有不支持平台错误；
4. 创建 `VisualScreenshotAgent`；
5. 组装 `AgentRuntimeServices`，将转写、状态和截图回调注入 Agent。

工厂不得执行下载、转写、模型请求或写数据库；它只创建依赖。

### 6.4 `TaskLifecycleService`

职责是封装 `write_status_record()` 以及异常信息归一化。接口保持小而明确：

```python
mark_parsing(task_id: str) -> None
mark_saving(task_id: str) -> None
mark_success(task_id: str) -> None
mark_failed(task_id: str, exc: BaseException | str) -> None
```

服务持有 `generation_token`、`NOTE_OUTPUT_DIR` 等写入上下文。它必须保留现有 generation token 语义，不能让旧任务覆盖新任务状态。

### 6.5 `NoteResultStore`

职责是封装任务元数据数据库操作：

```python
save_metadata(video_id: str, platform: str, task_id: str) -> None
delete_note(video_id: str, platform: str) -> int
```

保存失败的日志和容错行为必须与当前 `_save_metadata()` 一致。`NoteGenerator` 不直接导入或调用 `insert_video_task()`、`delete_task_by_video()`。

## 7. 生成流程

1. 路由按原有方式调用 `NoteGenerator(generation_token).generate(...)`。
2. `NoteGenerator.generate()` 创建 `GenerationRequest`，规范化格式、截图和链接选项。
3. 生命周期服务写入 `PARSING`。
4. 运行时工厂创建 Downloader、GPT、Transcriber、截图 Agent、Agent 服务和执行器依赖。
5. 根据请求构造 `AgentExecutionContext` 并调用既有 `build_note_execution_plan()`。
6. 创建无任务级实例状态的 `GenerationContext`，交给既有 `PlanExecutor.run()`。
7. 从上下文取得 Markdown、转写结果、音频元数据和视频路径，补充源链接。
8. 生命周期服务写入 `SAVING`，结果存储保存视频任务元数据。
9. 当不是延迟截图模式时，生命周期服务写入 `SUCCESS`。
10. 返回与现有结构一致的 `NoteResult`。
11. 任一步骤失败时，生命周期服务写入 `FAILED`，保持当前记录错误信息并返回 `None` 的兼容行为。

## 8. 错误处理与兼容性规则

### 8.1 组件创建失败

- Provider 不存在时仍抛出 `ProviderError`，错误枚举和消息不变。
- 平台不支持时仍抛出 `NoteError`，错误枚举和消息不变。
- 转写器类型未知时仍在运行时创建阶段失败，并保留当前错误语义。

### 8.2 Agent 执行失败

继续由 `PlanExecutor` 区分必选步骤和可选步骤。可选视觉步骤失败时保留 diagnostics，不改变基础笔记成功语义；必选步骤失败时由主流程捕获并进入 `FAILED`。

### 8.3 状态写入失败

生命周期服务不得吞掉会阻止任务状态一致性的主异常；保存元数据的日志容错行为按当前实现保留。任何 token 检查和状态文件格式不得改变。

### 8.4 并发与实例状态

所有一次生成的中间值必须保存在 `GenerationRequest`、`GenerationContext` 或局部变量中。运行时工厂可以按任务创建依赖，但不得把当前任务的路径、计划或结果写入全局可变状态。

## 9. 分阶段实施

### 阶段 1：建立数据边界

- 添加 `GenerationRequest`，在 `generate()` 入口完成参数归一化。
- 明确 `GenerationContext` 与现有 `AgentRuntimeContext` 的复用关系。
- 把 `execution_plan`、`video_path`、`video_img_urls` 从 `NoteGenerator` 实例字段迁移到局部变量或上下文。
- 保持现有调用路径和行为不变。

### 阶段 2：抽离运行时工厂

- 添加 `NoteRuntimeFactory`。
- 将转写器、GPT、Downloader、截图 Agent 和 `AgentRuntimeServices` 的创建迁移到工厂。
- 为 Provider 缺失、不支持平台和未知转写器类型补工厂单元测试。

### 阶段 3：抽离生命周期和结果存储

- 添加 `TaskLifecycleService`，迁移状态、异常和 generation token 处理。
- 添加 `NoteResultStore`，迁移元数据保存和删除。
- 保持 `NoteGenerator.delete_note()` 的兼容入口，必要时只保留委托，不让路由立即发生破坏性变化。

### 阶段 4：收窄主流程并清理依赖

- 让 `NoteGenerator` 只依赖运行时工厂、执行计划构造器、执行器、生命周期服务和结果存储。
- 删除 `note.py` 中不再需要的基础设施导入和方法。
- 保留公开入口和行为兼容测试。

## 10. 测试与验收

### 10.1 单元测试

必须覆盖：

1. `GenerationRequest` 正确合并 `link`、`screenshot` 和 `_format`。
2. `GenerationContext` 的缓存路径仍使用现有 `<task_id>_audio.json`、`<task_id>_transcript.json` 和 `<task_id>_markdown.md` 命名。
3. 运行时工厂能创建 GPT、转写器、Downloader 和截图 Agent 的组合依赖。
4. Provider 不存在、不支持平台、未知转写器类型的错误类型保持不变。
5. 生命周期服务写入 `PARSING`、`SAVING`、`SUCCESS` 和 `FAILED`，并正确携带 generation token。
6. 结果存储调用正确的数据库 DAO 参数，并保留保存失败日志行为。
7. 同一工厂/生成器实例连续执行两个任务时，两个任务的计划、视频路径、图片 URL 和结果互不污染。

### 10.2 现有聚焦测试

至少运行：

```powershell
cd backend
pytest tests/test_agent_planner.py tests/test_task_serial_executor.py tests/test_note_router_cache_recovery.py -q
```

再运行与 Agent、视觉增强和截图相关的现有测试，确认可选视觉步骤、延迟截图和 Markdown 组合行为未改变。

### 10.3 端到端验收

使用现有后端启动方式执行至少一条真实或项目已有的完整生成流程，并分别确认：

- 普通 URL 生成成功；
- 缓存转写命中时不重复执行不必要的字幕/音频步骤；
- `screenshot` 和 `link` 组合行为保持不变；
- Provider、平台或转写器配置错误会进入 `FAILED`；
- 结果 Markdown、转写结果、音频元数据和数据库任务记录仍可读取；
- generation token 不会被旧任务覆盖。

## 11. 完成标准

本设计对应的实现只有在以下条件都满足后，才可称为该阶段完成：

1. `NoteGenerator` 不再直接创建转写器、GPT、Downloader 或截图 Agent。
2. `NoteGenerator` 不再直接写任务状态或调用视频任务 DAO。
3. 单次生成状态不再通过生成器实例字段保存。
4. 现有 Agent 和 `PlanExecutor` 的职责与行为没有被重复实现或破坏。
5. 聚焦单元测试和回归测试通过。
6. 至少一条完整生成流程完成独立结果读取和状态验证。
7. Git diff 只包含本次设计和实现所需的精确文件，不覆盖用户已有改动。

## 12. 风险与回滚

主要风险是状态回调闭包、延迟截图流程和 generation token 在迁移时被错误绑定到共享对象。实现应按阶段提交，每个阶段保持原有入口可运行；若某阶段回归，只回退该阶段提交，不改动缓存文件和数据库结构。任何发现需要改变 API、状态协议或后台调度模型的需求，都应另立设计，不在本次职责拆分中隐式扩大范围。
