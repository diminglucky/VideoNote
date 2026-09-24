# VideoNote

> 当前版本：`0.2.8`
>
> 使用限制：本项目仅允许个人学习、研究、评估和非商业自用。未经作者明确书面授权，禁止任何形式的商业化使用、商业部署、付费服务、二次售卖或商业集成。

VideoNote 是一个面向真实学习场景的 AI 视频笔记项目。它可以从 Bilibili、YouTube、抖音、快手和本地视频中提取内容，先生成结构化 Markdown 笔记，再根据笔记内容反推需要截图的位置，把真正有用的画面插入到对应章节里。

这个项目基于历史 BiliNote 代码继续演进，但当前产品方向已经重新定位为个人项目 `VideoNote`。仓库里仍保留一些继承下来的目录名，例如 `BillNote_frontend`，后续会继续逐步清理；README、界面方向、截图流程、导出体验和 agent 架构都以 VideoNote 为准。

## 项目目标

VideoNote 不只是“把视频总结成几段话”。它更关注能不能生成一份真正可以复习、整理、导出、导入知识库的笔记。

核心目标：

- 先快速生成可读的 Markdown 笔记，让用户尽早看到内容。
- 根据每个标题和段落的实际内容决定是否需要截图，而不是按视频时长硬凑图片数量。
- 截图要服务于知识点，避免空白页、模糊页、重复页、过多连续图片和无关画面。
- 视频没有高清源时也继续完整生成，不因为清晰度不足直接中断。
- 长任务要有明确进度，失败和降级要能被用户看见。
- 导出时尽量解决 Markdown 图片丢失的问题，让笔记能迁移到其他工具。

## 真实效果截图

下面图片来自本仓库当前 VideoNote 流程生成的真实任务结果，不是历史项目的演示图。

![VideoNote 真实生成截图 1](docs/assets/videonote-real-shot-1.png)

![VideoNote 真实生成截图 2](docs/assets/videonote-real-shot-2.png)

![VideoNote 真实生成截图 3](docs/assets/videonote-real-shot-3.png)

![VideoNote 真实生成截图 4](docs/assets/videonote-real-shot-4.png)

![VideoNote 真实生成截图 5](docs/assets/videonote-real-shot-5.png)

## 主要功能

- 支持视频链接和本地视频文件生成笔记。
- 支持 Bilibili、YouTube、抖音、快手和本地文件。
- 支持平台 cookie、代理、下载器配置，尽量获取更高质量的视频源。
- 支持 Fast Whisper、Groq、Bcut、快手、MLX Whisper 等转写方式。
- 支持多个 LLM provider 和模型配置。
- 生成带目录、章节、时间锚点、来源链接和总结的 Markdown。
- 可选截图增强：先生成笔记，再异步分析文档和视频，把截图插入到合适位置。
- 支持截图质量检测、重复过滤、密度控制和降级提示。
- 支持 Markdown 预览、思维导图、历史记录、重新生成和任务进度展示。
- 支持 Markdown 图片包 ZIP、HTML、DOCX、PDF 等导出方式。
- 支持基于笔记和转写内容的问答索引。
- 提供浏览器扩展入口，可在视频页面快速发起生成。

## 截图策略

VideoNote 不使用固定截图数量。

当前策略是：

1. 先生成基础 Markdown 笔记。
2. 按章节和段落分析笔记中哪些地方需要视觉证据。
3. 把文档位置映射回对应的视频时间窗口。
4. 在候选时间附近截取多个画面。
5. 过滤空白、模糊、重复、低信息密度和无关画面。
6. 控制同一章节内的图片密度，避免连续长段都是截图。
7. 将最终截图异步写回 Markdown，并保留可追踪的视觉报告。

这意味着：信息密度高的章节可以有多张图；纯讲解、过渡、重复画面或价值不大的段落可以没有图。截图数量由内容决定，不由视频时长决定。

## Agent 架构

VideoNote 的 agent 设计追求“职责清晰”和“不过度包装”。顶层编排保持简洁，截图内部流程由视觉增强模块处理。

```text
用户输入
  -> NoteGenerator
  -> LlmNoteOrchestrator
  -> SupervisorAgent
     -> ContentAgent / 闭集工具（字幕、转写、写作）
     -> ReviewerAgent
     -> VisualAgent / 视觉计划
  -> 基础 Markdown
  -> VisualEnhancementService
  -> VisualScreenshotAgent / 截图筛选与写回
  -> RAG 索引 / 最终笔记 / 导出
```

当前主要角色：

- `NoteGenerator`：任务入口，负责参数、状态、缓存、持久化和整体生命周期。
- `LlmNoteOrchestrator`：为单次生成建立 AgentState、预算和 trace，并驱动观察—行动循环。
- `SupervisorAgent`：根据当前状态选择工具、委派角色、评审、返工、降级或完成。
- `ContentAgent`：基于转录和视频元数据生成或修订基础 Markdown。
- `ReviewerAgent`：检查内容完整性、来源一致性和视觉执行质量，未通过时给出结构化问题。
- `VisualAgent`：决定是否需要视觉证据，生成有界时间窗口，并在失败评审后执行一次重规划。
- `VisualEnhancementService`：在基础笔记保存后异步执行截图增强，避免用户一直等不到正文。
- `VisualScreenshotAgent`：负责视觉截图内部流程，包括文档分析、候选时间选择、截帧、评分、去重和插入。
- `MarkdownComposerAgent`：负责最终 Markdown 的结构、链接、截图和版本内容合成。
- `ToolRegistry`：只暴露下载、字幕、转写、写作和视觉增强等受控能力。
- `JsonlTraceStore`：记录脱敏后的决策、Observation、工具调用和最终状态。

下载、转码、转写、Markdown 写盘、截图生成、缓存和数据库持久化仍由确定性服务执行；LLM 只负责受约束的决策，不直接执行任意代码或访问任意路径。

## 技术栈

- 后端：Python 3.11、FastAPI、SQLAlchemy、SQLite
- 前端：React 19、Vite、TypeScript、Tailwind、shadcn/ui
- 桌面端：Tauri
- 浏览器扩展：Vue 3、Vite、MV3
- 视频处理：FFmpeg、yt-dlp
- 转写：Whisper/Groq/Bcut/平台字幕等
- 笔记生成：可配置 LLM provider

## 项目结构

```text
backend/              FastAPI 后端，包含下载、转写、生成、截图、导出和任务状态
BillNote_frontend/    React 前端，包含主页、历史、预览、设置、导出和进度反馈
BillNote_extension/   浏览器扩展，可从视频页面发起 VideoNote 任务
docs/assets/          README 使用的真实项目截图
doc/                  架构设计、重构计划和历史说明
config/               本地配置
note_results/         本地生成结果
```

## 环境要求

- Python 3.11
- Node.js 20+
- pnpm
- FFmpeg
- Windows 推荐使用 Anaconda 环境 `play`
- 至少配置一个可用的 LLM provider 和模型
- 如需下载更高清的视频，建议配置对应平台 cookie

## 快速启动

Windows 下推荐直接运行：

```bat
start-dev.bat
```

默认地址：

- 前端：`http://127.0.0.1:3015`
- 后端：`http://127.0.0.1:8483`
- API 文档：`http://127.0.0.1:8483/docs`

如果端口被占用，先关闭旧的后端和前端终端，再重新启动。

## 手动启动

后端：

```bash
cd backend
pip install -r requirements.txt
python main.py
```

前端：

```bash
cd BillNote_frontend
pnpm install
pnpm dev
```

浏览器扩展：

```bash
cd BillNote_extension
pnpm install
pnpm dev
```

然后在浏览器扩展管理页加载 `BillNote_extension/extension/`。

## 常用配置

常见环境变量：

- `BACKEND_PORT`：后端端口，默认 `8483`
- `FRONTEND_PORT`：前端端口，默认 `3015`
- `APP_BIND_HOST` / `APP_PORT`：Docker 对外监听地址与端口，默认仅绑定 `127.0.0.1:3015`
- `FFMPEG_BIN_PATH`：自定义 FFmpeg 路径
- `TRANSCRIBER_TYPE`：转写方式，例如 `fast-whisper` 或 `groq`
- `WHISPER_MODEL_SIZE`：Whisper 模型大小，例如 `medium`
- `OUT_DIR`：截图输出目录
- `IMAGE_BASE_URL`：Markdown 图片 URL 前缀
- `SCREENSHOT_REVIEW_MODE`：截图视觉复核模式，默认 `off`
- `SCREENSHOT_CANDIDATE_LIMIT`：截图候选数量上限
- `SCREENSHOT_COMFORT_MAX_PER_SECTION`：单章节舒适截图上限，默认 `3`

LLM API key 建议在前端设置页里配置。

## 导出说明

Markdown 本身只保存文字和图片引用。如果图片引用指向本地后端地址，后端关闭、服务器清理、图片目录迁移后，其他工具可能无法显示图片。

推荐方式：

- 需要 Markdown 和图片一起迁移：导出 ZIP，里面包含 `note.md` 和 `images/`。
- 需要单文件阅读或分享：导出 HTML、DOCX 或 PDF。
- 只导出普通 `.md`：适合目标工具可以访问图片地址，或者你已经手动处理图片。

导入 Notion、Obsidian 或其他知识库时，优先使用带图片文件的 ZIP 或自包含格式，避免过一段时间图片失效。

## 测试

后端：

```bash
cd backend
pytest
```

前端：

```bash
cd BillNote_frontend
pnpm build
pnpm lint
```

浏览器扩展：

```bash
cd BillNote_extension
pnpm build
pnpm typecheck
pnpm test
```

当前重点测试覆盖 agent 计划、截图增强、截图密度、部分成功、缓存恢复和视觉质量报告。

## 当前重点

VideoNote 仍在持续重构和打磨，当前优先级：

- 提升真实长视频下的截图命中率。
- 提升基础笔记质量，让截图插入有更可靠的文档依据。
- 缩短端到端生成耗时。
- 改进失败恢复和降级提示。
- 优化重新生成、进度条和后台任务反馈。
- 继续清理历史 BiliNote 命名和无效代码。

## 非商业使用声明

本项目仅允许个人学习、研究、评估和非商业自用。

未经作者明确书面授权，禁止：

- 商业部署
- 商业 SaaS
- 付费 API 服务
- 基于本项目的付费课程、培训、交付或咨询
- 付费托管、运营或笔记生成服务
- 转售、付费分发或付费打包
- 集成到任何商业产品、商业插件或商业平台中

本仓库包含来自开源 BiliNote 项目的历史代码和结构，原始 MIT 版权声明保留在 `LICENSE` 和 `NOTICE` 中。本仓库新增的 VideoNote 定制、产品化调整、README、界面设计和重构内容受以上非商业限制约束。
