import { useState, useEffect, useMemo, memo, FC } from 'react'
import ReactMarkdown from 'react-markdown'
import { Button } from '@/components/ui/button.tsx'
import { Copy, Download, ArrowRight, Play, ExternalLink, FileText, Sparkles, Loader2 } from 'lucide-react'
import { toast } from 'react-hot-toast'
import Error from '@/components/Lottie/error.tsx'
import Loading from '@/components/Lottie/Loading.tsx'
import Idle from '@/components/Lottie/Idle.tsx'
import StepBar from '@/pages/HomePage/components/StepBar.tsx'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { atomDark as codeStyle } from 'react-syntax-highlighter/dist/esm/styles/prism'
import Zoom from 'react-medium-image-zoom'
import 'react-medium-image-zoom/dist/styles.css'
import gfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeSlug from 'rehype-slug'
import JSZip from 'jszip'
import 'katex/dist/katex.min.css'
import 'github-markdown-css/github-markdown-light.css'
import { ScrollArea } from '@/components/ui/scroll-area.tsx'
import { useTaskStore } from '@/store/taskStore'
import { noteStyles } from '@/constant/note.ts'
import { MarkdownHeader } from '@/pages/HomePage/components/MarkdownHeader.tsx'
import TranscriptViewer from '@/pages/HomePage/components/transcriptViewer.tsx'
import MarkmapEditor from '@/pages/HomePage/components/MarkmapComponent.tsx'
import ChatPanel from '@/pages/HomePage/components/ChatPanel.tsx'
import VideoBanner from '@/pages/HomePage/components/VideoBanner.tsx'
import AgentRunTimeline from '@/pages/HomePage/components/AgentRunTimeline.tsx'
import {
  isRunningTaskStatus,
  taskStatusMessage,
  taskSteps,
} from '@/models/taskStateMachine'

interface VersionNote {
  ver_id: string
  content: string
  style: string
  model_name: string
  created_at?: string
}

interface MarkdownViewerProps {
  status: 'idle' | 'loading' | 'success' | 'failed'
}

const remarkPlugins = [gfm, remarkMath]
const rehypePlugins = [rehypeKatex, rehypeSlug]
const markdownImagePattern = /[\uFFFC\uFFFD\uFEFF]?[ \t]*![ \t]*\[([^\]]*)\][ \t]*\(([^)\r\n]+)\)/g

const sanitizeFileName = (name: string) =>
  (name || 'note').replace(/[<>:"/\\|?*]/g, '_').replace(/\s+/g, ' ').trim().slice(0, 120) || 'note'

const cleanMarkdownImageSrc = (rawSrc: string) => {
  const trimmed = rawSrc.trim()
  const withoutAngleBrackets = trimmed.startsWith('<') && trimmed.endsWith('>')
    ? trimmed.slice(1, -1)
    : trimmed
  const titleMatch = withoutAngleBrackets.match(/^(.+?)(?:\s+["'][^"']*["'])$/)
  return (titleMatch?.[1] || withoutAngleBrackets).trim()
}

const normalizeBrokenImageSyntax = (markdown: string) =>
  markdown
    .replace(/[\uFFFC\uFFFD\uFEFF]+[ \t]*![ \t]*\[/g, '![')
    .replace(/!\s+\[/g, '![')
    .replace(/\]\s+\(/g, '](')
    .replace(/^\s*!\[\]\((images\/[^)\r\n]+)\)\s*$/gm, '![]($1)')
    .replace(/!\[([^\]]*)\]\(([^)\r\n]+)\)(?=!\[)/g, '![$1]($2)\n\n')

const tocHeadingPattern = /^(#{1,3})\s*目录\s*$/
const markdownHeadingPattern = /^(#{1,6})\s+(.+?)\s*$/
const fencedBlockPattern = /^\s*(```|~~~)/

const stripMarkdownForText = (value: string) =>
  value
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/<[^>]+>/g, '')
    .replace(/\\/g, '')
    .trim()

const normalizeContentMarkerText = (value: string) =>
  value.replace(/\*?Content-\[((?:\d{2}:)?\d{2}:\d{2})\]\*?/g, '*Content-[$1]*')

const cleanHeadingTitle = (rawTitle: string) => normalizeContentMarkerText(stripMarkdownForText(rawTitle))

const isSummaryHeading = (title: string) => {
  const compact = title.replace(/\s+/g, '')
  return compact === 'AI总结' || compact === '总结' || compact.startsWith('AI总结')
}

const githubStyleAnchor = (title: string, used: Map<string, number>) => {
  const base = stripMarkdownForText(title)
    .toLowerCase()
    .replace(/[\*_~`]/g, '')
    .replace(/[!"#$%&'()*+,./:;<=>?@\[\\\]^`{|}，。！？、；：“”‘’（）【】《》]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-{3,}/g, '--')
    .trim() || 'section'
  const count = used.get(base) || 0
  used.set(base, count + 1)
  return count > 0 ? `${base}-${count}` : base
}

const normalizeAnchorSearchText = (value: string) =>
  decodeURIComponent(value || '')
    .replace(/content-\[((?:\d{2}:)?\d{2}:\d{2})\]/gi, '$1')
    .replace(/原片(?:\s*[（(]?\s*(?:\d{2}:)?\d{2}:\d{2}\s*[）)]?)?/g, '')
    .replace(/[^\p{L}\p{N}]+/gu, '')
    .toLowerCase()

const extractMarkdownTocHeadings = (markdown: string) => {
  const headings: string[] = []
  let inFence = false
  for (const line of markdown.split(/\r?\n/)) {
    if (fencedBlockPattern.test(line)) {
      inFence = !inFence
      continue
    }
    if (inFence) continue
    const match = line.match(markdownHeadingPattern)
    if (!match || match[1].length !== 2) continue
    const title = cleanHeadingTitle(match[2])
    if (!title || title === '目录' || isSummaryHeading(title) || title.includes('原片截图')) continue
    headings.push(title)
  }
  return headings
}

const findTocBlock = (lines: string[]) => {
  let inFence = false
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    if (fencedBlockPattern.test(line)) {
      inFence = !inFence
      continue
    }
    if (inFence || !tocHeadingPattern.test(line.trim())) continue

    let end = index + 1
    while (end < lines.length) {
      const nextLine = lines[end]
      if (markdownHeadingPattern.test(nextLine) && !tocHeadingPattern.test(nextLine.trim())) break
      end += 1
    }
    while (end > index + 1 && ['', '---', '***', '___'].includes(lines[end - 1].trim())) {
      end -= 1
    }
    return [index, end] as const
  }
  return null
}

const tocInsertIndex = (lines: string[]) => {
  let inFence = false
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    if (fencedBlockPattern.test(line)) {
      inFence = !inFence
      continue
    }
    if (inFence) continue
    const match = line.match(markdownHeadingPattern)
    if (match && match[1].length === 2) return index
  }
  return 0
}

const normalizeMarkdownToc = (markdown: string, ensureToc = false) => {
  const headings = extractMarkdownTocHeadings(markdown)
  if (!headings.length) return markdown

  const lines = markdown.split(/\r?\n/)
  const tocBlock = findTocBlock(lines)
  if (!tocBlock && !ensureToc) return markdown
  const used = new Map<string, number>()
  const tocLines = [
    '## 目录',
    '',
    ...headings.map(title => `- [${title}](#${githubStyleAnchor(title, used)})`),
    '',
  ]

  if (tocBlock) {
    const [start, end] = tocBlock
    return [...lines.slice(0, start), ...tocLines, ...lines.slice(end)].join('\n').trimEnd() + '\n'
  }

  const insertAt = tocInsertIndex(lines)
  return [...lines.slice(0, insertAt), ...tocLines, ...lines.slice(insertAt)].join('\n').trimEnd() + '\n'
}

const normalizeNestedTocOriginLinks = (markdown: string) =>
  markdown
    .replace(
      /^(\s*[-*+]\s*)\[([^\]\n]+?)\s+\[原片 @ (\d{2}:\d{2})\]\((https?:\/\/[^)\s]+)\)\*?\]\((#[^)]+)\)/gm,
      (_match, prefix: string, title: string, time: string, _url: string, anchor: string) =>
        `${prefix}[${title.trim()} *Content-[${time}]*](${anchor})`,
    )
    .replace(
      /^(\s*[-*+]\s*)\[([^\]\n]+?)\]\((#[^)]+)\)\s+[·路]\s+\[原片 @ (\d{2}:\d{2})\]\((https?:\/\/[^)\s]+)\)/gm,
      (_match, prefix: string, title: string, anchor: string, time: string) =>
        `${prefix}[${title.trim()} *Content-[${time}]*](${anchor})`,
    )

const normalizeMarkdownForDisplay = (markdown: string) =>
  normalizeNestedTocOriginLinks(normalizeMarkdownToc(normalizeBrokenImageSyntax(markdown)))

const asCount = (value: unknown) => {
  const count = Number(value)
  return Number.isFinite(count) && count > 0 ? count : 0
}

const visualReportSummary = (visualReport: any) => {
  if (!visualReport || typeof visualReport !== 'object') return ''
  const success = asCount(visualReport.successful_slots)
  const planned = asCount(visualReport.planned_slots)
  const skipped = asCount(visualReport.skipped_slots)
  const duplicate = asCount(visualReport.duplicate_slots)
  const failed = asCount(visualReport.failed_slots)
  if (!success && !planned && !skipped && !duplicate && !failed) return ''

  const parts = success > 0 ? [`截图 ${success} 张`] : []
  if (planned) parts.push(success > 0 ? `计划 ${planned}` : `计划 ${planned} 个截图位置`)
  if (skipped) parts.push(`跳过 ${skipped}`)
  if (duplicate) parts.push(`去重 ${duplicate}`)
  if (failed) parts.push(`失败 ${failed}`)
  return parts.join(' · ')
}

const dataUriToBlob = (dataUri: string) => {
  const match = dataUri.match(/^data:([^;,]+)?(;base64)?,(.*)$/)
  if (!match) return null
  const mimeType = match[1] || 'application/octet-stream'
  const isBase64 = Boolean(match[2])
  const payload = isBase64 ? atob(match[3]) : decodeURIComponent(match[3])
  const bytes = new Uint8Array(payload.length)
  for (let i = 0; i < payload.length; i += 1) {
    bytes[i] = payload.charCodeAt(i)
  }
  return new Blob([bytes], { type: mimeType })
}

const blobToDataUri = (blob: Blob) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result || ''))
    reader.onerror = () => reject(reader.error || new Error('图片编码失败'))
    reader.readAsDataURL(blob)
  })

const getImageExtension = (src: string, blob?: Blob | null) => {
  const mimeExtMap: Record<string, string> = {
    'image/png': 'png',
    'image/jpeg': 'jpg',
    'image/jpg': 'jpg',
    'image/gif': 'gif',
    'image/webp': 'webp',
    'image/bmp': 'bmp',
    'image/svg+xml': 'svg',
  }
  if (blob?.type && mimeExtMap[blob.type]) return mimeExtMap[blob.type]

  const cleanSrc = src.split(/[?#]/)[0]
  const extMatch = cleanSrc.match(/\.([a-z0-9]+)$/i)
  return extMatch?.[1]?.toLowerCase() || 'jpg'
}

const getImageBaseName = (src: string, index: number) => {
  try {
    const url = src.startsWith('data:') ? null : new URL(src, window.location.href)
    const fileName = url?.pathname.split('/').filter(Boolean).pop()
    if (fileName) {
      return sanitizeFileName(fileName.replace(/\.[a-z0-9]+$/i, ''))
    }
  } catch {
    // Ignore malformed paths and fall through to a generated name.
  }
  return `image_${String(index + 1).padStart(3, '0')}`
}

const resolveImageUrl = (src: string, baseURL: string) => {
  if (src.startsWith('data:')) return src
  if (/^https?:\/\//i.test(src)) return src
  if (src.startsWith('/')) return `${baseURL}${src}`
  return new URL(src, window.location.href).toString()
}

const downloadBlob = (blob: Blob, fileName: string) => {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

const normalizeExportedMarkdownImages = (markdown: string) =>
  markdown
    .replace(/^[\uFFFC\uFFFD\uFEFF\\\s]+(!\[[^\]]*\]\([^)]+\))\s*$/gm, '$1')
    .replace(/\\(!\[[^\]]*\]\([^)]+\))/g, '$1')
    .replace(/([^\n])(!\[[^\]]*\]\([^)]+\))/g, '$1\n\n$2')
    .replace(/(!\[[^\]]*\]\([^)]+\))([^\n])/g, '$1\n\n$2')
    .replace(/\n{3,}/g, '\n\n')

/**
 * 构建 ReactMarkdown components 对象，baseURL 用于修正图片路径。
 * 使用函数 + useMemo 避免每次渲染都创建新的函数实例。
 */
function createMarkdownComponents(baseURL: string) {
  return {
    h1: ({ children, ...props }: any) => (
      <h1
        className="text-primary my-6 scroll-m-20 text-3xl font-extrabold tracking-tight lg:text-4xl"
        {...props}
      >
        {children}
      </h1>
    ),
    h2: ({ children, ...props }: any) => (
      <h2
        className="text-primary mt-10 mb-4 scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0"
        {...props}
      >
        {children}
      </h2>
    ),
    h3: ({ children, ...props }: any) => (
      <h3
        className="text-primary mt-8 mb-4 scroll-m-20 text-xl font-semibold tracking-tight"
        {...props}
      >
        {children}
      </h3>
    ),
    h4: ({ children, ...props }: any) => (
      <h4
        className="text-primary mt-6 mb-2 scroll-m-20 text-lg font-semibold tracking-tight"
        {...props}
      >
        {children}
      </h4>
    ),
    p: ({ children, ...props }: any) => (
      <p className="leading-7 [&:not(:first-child)]:mt-6" {...props}>
        {children}
      </p>
    ),
    a: ({ href, children, ...props }: any) => {
      const isOriginLink =
        typeof children[0] === 'string' &&
        (children[0] as string).startsWith('原片 @')

      if (isOriginLink) {
        const timeMatch = (children[0] as string).match(/原片 @ (\d{2}:\d{2})/)
        const timeText = timeMatch ? timeMatch[1] : '原片'

        return (
          <span className="origin-link my-2 inline-flex">
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-full bg-neutral-100 px-3 py-1 text-sm font-medium text-neutral-800 transition-colors hover:bg-neutral-200"
              {...props}
            >
              <Play className="h-3.5 w-3.5" />
              <span>原片（{timeText}）</span>
            </a>
          </span>
        )
      }

      // 处理笔记内部锚点链接（如目录跳转）
      if (href?.startsWith('#')) {
        const handleAnchorClick = (e: React.MouseEvent) => {
          e.preventDefault()
          const id = decodeURIComponent(href.slice(1))

          // 1. 优先精确匹配 id
          let target = document.getElementById(id)

          // 2. 精确失败时按 heading 文本模糊匹配
          // LLM 生成的目录锚点可能和 heading 实际文本不完全一致
          //（例如 heading 带 *Content-[00:00]* 后缀，目录链接里没有）
          if (!target) {
            const search = normalizeAnchorSearchText(id)
            const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6')
            for (const h of headings) {
              const text = normalizeAnchorSearchText(h.textContent || '')
              if (text && search && (text.includes(search) || search.includes(text))) {
                target = h
                break
              }
            }
          }

          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' })
          } else {
            toast.error('未找到对应章节')
          }
        }

        return (
          <a
            href={href}
            onClick={handleAnchorClick}
            className="text-primary hover:text-primary/80 inline-flex items-center gap-0.5 font-medium underline underline-offset-4"
            {...props}
          >
            {children}
          </a>
        )
      }

      return (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary hover:text-primary/80 inline-flex items-center gap-0.5 font-medium underline underline-offset-4"
          {...props}
        >
          {children}
          {href?.startsWith('http') && (
            <ExternalLink className="ml-0.5 inline-block h-3 w-3" />
          )}
        </a>
      )
    },
    img: ({ node, ...props }: any) => {
      let src = props.src
      if (src.startsWith('/')) {
        src = baseURL + src
      }
      props.src = src

      return (
        <div className="my-8 flex justify-center">
          <Zoom>
            <img
              {...props}
              className="max-w-full cursor-zoom-in rounded-lg object-cover shadow-md transition-all hover:shadow-lg"
              style={{ maxHeight: '500px' }}
            />
          </Zoom>
        </div>
      )
    },
    strong: ({ children, ...props }: any) => (
      <strong className="text-primary font-bold" {...props}>
        {children}
      </strong>
    ),
    li: ({ children, ...props }: any) => {
      const rawText = String(children)
      const isFakeHeading = /^(\*\*.+\*\*)$/.test(rawText.trim())

      if (isFakeHeading) {
        return (
          <div className="text-primary my-4 text-lg font-bold">{children}</div>
        )
      }

      return (
        <li className="my-1" {...props}>
          {children}
        </li>
      )
    },
    ul: ({ children, ...props }: any) => (
      <ul className="my-6 ml-6 list-disc [&>li]:mt-2" {...props}>
        {children}
      </ul>
    ),
    ol: ({ children, ...props }: any) => (
      <ol className="my-6 ml-6 list-decimal [&>li]:mt-2" {...props}>
        {children}
      </ol>
    ),
    blockquote: ({ children, ...props }: any) => (
      <blockquote
        className="border-primary/20 text-muted-foreground mt-6 border-l-4 pl-4 italic"
        {...props}
      >
        {children}
      </blockquote>
    ),
    code: ({ inline, className, children, ...props }: any) => {
      const match = /language-(\w+)/.exec(className || '')
      const codeContent = String(children).replace(/\n$/, '')

      if (!inline && match) {
        return (
          <div className="group bg-muted relative my-6 overflow-hidden rounded-lg border shadow-sm">
            <div className="bg-muted text-muted-foreground flex items-center justify-between px-4 py-1.5 text-sm font-medium">
              <div>{match[1].toUpperCase()}</div>
              <button
                onClick={() => {
                  navigator.clipboard.writeText(codeContent)
                  toast.success('代码已复制')
                }}
                className="bg-background/80 hover:bg-background flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors"
              >
                <Copy className="h-3.5 w-3.5" />
                复制
              </button>
            </div>
            <SyntaxHighlighter
              style={codeStyle}
              language={match[1]}
              PreTag="div"
              className="!bg-muted !m-0 !p-0"
              customStyle={{
                margin: 0,
                padding: '1rem',
                background: 'transparent',
                fontSize: '0.9rem',
              }}
              {...props}
            >
              {codeContent}
            </SyntaxHighlighter>
          </div>
        )
      }

      return (
        <code
          className="bg-muted relative rounded px-[0.3rem] py-[0.2rem] font-mono text-sm"
          {...props}
        >
          {children}
        </code>
      )
    },
    table: ({ children, ...props }: any) => (
      <div className="my-6 w-full overflow-y-auto">
        <table className="w-full border-collapse text-sm" {...props}>
          {children}
        </table>
      </div>
    ),
    th: ({ children, ...props }: any) => (
      <th
        className="border-muted-foreground/20 border px-4 py-2 text-left font-medium [&[align=center]]:text-center [&[align=right]]:text-right"
        {...props}
      >
        {children}
      </th>
    ),
    td: ({ children, ...props }: any) => (
      <td
        className="border-muted-foreground/20 border px-4 py-2 text-left [&[align=center]]:text-center [&[align=right]]:text-right"
        {...props}
      >
        {children}
      </td>
    ),
    hr: ({ ...props }: any) => (
      <hr className="border-muted-foreground/20 my-8" {...props} />
    ),
  }
}

const MarkdownViewer: FC<MarkdownViewerProps> = memo(({ status }) => {
  const [currentVerId, setCurrentVerId] = useState<string>('')
  const [selectedContent, setSelectedContent] = useState<string>('')
  const [modelName, setModelName] = useState<string>('')
  const [style, setStyle] = useState<string>('')
  const [createTime, setCreateTime] = useState<string>('')
  const [isExporting, setIsExporting] = useState(false)
  // 确保baseURL没有尾部斜杠
  const baseURL = (String(import.meta.env.VITE_API_BASE_URL || '').replace('/api','') || '').replace(/\/$/, '')
  const getCurrentTask = useTaskStore.getState().getCurrentTask
  const currentTask = useTaskStore(state => state.getCurrentTask())
  const taskStatus = currentTask?.status || 'PENDING'
  const taskMessage = currentTask?.message
  const retryTask = useTaskStore.getState().retryTask
  const markdownVersions = useMemo(
    () => currentTask?.markdown || [],
    [currentTask?.markdown],
  )
  const displayContent = useMemo(
    () => normalizeMarkdownForDisplay(selectedContent),
    [selectedContent],
  )
  const currentFormData = currentTask?.formData
  const currentCreatedAt = currentTask?.createdAt
  const [showTranscribe, setShowTranscribe] = useState(false)
  const [showChat, setShowChat] = useState<false | 'half' | 'full'>(false)
  const [viewMode, setViewMode] = useState<'map' | 'preview'>('preview')
  const isRetrySubmitting = Boolean(currentTask?.isRetrySubmitting)
  const isTaskRunning = Boolean(currentTask && (isRunningTaskStatus(taskStatus) || isRetrySubmitting))
  const runningMessage = taskStatusMessage(taskStatus, taskMessage)
  const progressStatus = isRetrySubmitting ? 'PENDING' : taskStatus
  const progressMessage = isRetrySubmitting ? '正在提交重新生成请求，旧笔记会先保留' : runningMessage
  const screenshotSummary = visualReportSummary(currentTask?.visualReport)
  const screenshotRequested = Boolean(
    currentTask?.formData?.screenshot ||
    currentTask?.formData?.format?.includes('screenshot'),
  )
  const progressSteps = useMemo(
    () =>
      screenshotRequested
        ? taskSteps
        : taskSteps.filter(step => step.key !== 'FORMATTING' && step.key !== 'ENHANCING'),
    [screenshotRequested],
  )

  // 缓存 ReactMarkdown components，仅在 baseURL 变化时重建
  const markdownComponents = useMemo(() => createMarkdownComponents(baseURL), [baseURL])

  useEffect(() => {
    if (!currentTask) return

    const latestVersion = [...markdownVersions].sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    )[0]
    if (latestVersion) {
      setCurrentVerId(latestVersion.ver_id)
    } else {
      setCurrentVerId('')
      setSelectedContent('')
    }
  }, [currentTask, markdownVersions, taskStatus])

  useEffect(() => {
    if (!currentTask) return

    const currentVer = markdownVersions.find(v => v.ver_id === currentVerId)
    if (currentVer) {
      setModelName(currentVer.model_name)
      setStyle(currentVer.style)
      setCreateTime(currentVer.created_at || '')
      setSelectedContent(currentVer.content)
    } else {
      setModelName(currentFormData?.model_name || '')
      setStyle(currentFormData?.style || '')
      setCreateTime(currentCreatedAt || '')
    }
  }, [currentTask, currentVerId, markdownVersions, currentFormData, currentCreatedAt])
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(normalizeMarkdownForDisplay(selectedContent))
      toast.success('已复制到剪贴板')
    } catch (e) {
      toast.error('复制失败')
    }
  }
  const handleDownload = async () => {
    const task = getCurrentTask()
    const name = sanitizeFileName(task?.audioMeta.title || 'note')
    const images: Array<{ src: string; alt: string }> = []
    const normalizedContent = normalizeMarkdownForDisplay(selectedContent)

    const markdownWithPlaceholders = normalizedContent.replace(
      markdownImagePattern,
      (match, altText: string, rawSrc: string) => {
        const src = cleanMarkdownImageSrc(rawSrc)
        if (!src) return match
        const index = images.push({ src, alt: altText || '' }) - 1
        return `![${altText || ''}](__VIDEONOTE_IMAGE_${index}__)`
      },
    )

    if (images.length === 0) {
      const blob = new Blob([normalizedContent], { type: 'text/markdown;charset=utf-8' })
      downloadBlob(blob, `${name}.md`)
      toast.success('Markdown 已导出')
      return
    }

    setIsExporting(true)
    const toastId = `export-${task?.id || name}`
    toast.loading('正在打包 Markdown 和图片...', { id: toastId })

    try {
      const zip = new JSZip()
      const imageFolder = zip.folder('images')
      const localImagePaths: string[] = []
      const inlineImagePaths: string[] = []

      await Promise.all(
        images.map(async (image, index) => {
          const imageUrl = resolveImageUrl(image.src, baseURL)
          try {
            let blob: Blob | null = null

            if (imageUrl.startsWith('data:')) {
              blob = dataUriToBlob(imageUrl)
            } else {
              const response = await fetch(imageUrl)
              if (!response.ok) {
                throw new Error(`图片下载失败：${image.src}`)
              }
              blob = await response.blob()
            }

            if (!blob) {
              throw new Error(`图片解析失败：${image.src}`)
            }

            const extension = getImageExtension(imageUrl, blob)
            const baseName = getImageBaseName(imageUrl, index)
            const fileName = `${baseName}.${extension}`
            imageFolder?.file(fileName, blob)
            localImagePaths[index] = `images/${fileName}`
            inlineImagePaths[index] = await blobToDataUri(blob)
          } catch (error) {
            console.warn('图片打包失败，保留原始链接：', image.src, error)
            localImagePaths[index] = imageUrl
            inlineImagePaths[index] = imageUrl
          }
        }),
      )

      const markdownWithImages = localImagePaths.reduce(
        (markdown, imageData, index) =>
          markdown.replace(`__VIDEONOTE_IMAGE_${index}__`, imageData || images[index].src),
        markdownWithPlaceholders,
      )
      const markdownWithInlineImages = inlineImagePaths.reduce(
        (markdown, imageData, index) =>
          markdown.replace(`__VIDEONOTE_IMAGE_${index}__`, imageData || images[index].src),
        markdownWithPlaceholders,
      )
      zip.file('note.md', normalizeExportedMarkdownImages(markdownWithImages))
      zip.file('note-inline.md', normalizeExportedMarkdownImages(markdownWithInlineImages))
      zip.file(
        'README.txt',
        [
          'VideoNote Markdown 导出说明',
          '',
          '1. 推荐：先解压整个 zip，再打开 note.md。note.md 依赖同级 images 文件夹。',
          '2. 如果你的 Markdown 软件没有显示图片，请打开 note-inline.md，它把图片直接写进 Markdown 文件。',
          '3. 不要只从压缩包里单独拖出 note.md，否则 images 文件夹不会跟着一起出来。',
          '',
        ].join('\n'),
      )
      const zipBlob = await zip.generateAsync({ type: 'blob' })
      downloadBlob(zipBlob, `${name}.zip`)

      toast.success(`已导出 Markdown 图片包（${images.length} 张图，含内嵌版）`, { id: toastId })
    } catch (error) {
      console.error('导出失败:', error)
      toast.error(error instanceof Error ? error.message : '导出失败，请稍后重试', { id: toastId })
    } finally {
      setIsExporting(false)
    }
  }

  if (status === 'loading') {
    return (
      <div className="flex h-full w-full items-center justify-center bg-neutral-50 p-8 text-neutral-600">
        <div className="w-full max-w-2xl rounded-lg border border-neutral-200 bg-white p-8 shadow-sm">
          <div className="mb-6 flex items-start gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-neutral-950 text-white">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <p className="text-lg font-semibold text-neutral-950">正在生成 VideoNote</p>
              <p className="mt-1 text-sm text-neutral-500">
                {taskMessage || '正在解析视频内容，长视频会需要更久一些'}
              </p>
            </div>
          </div>
          <StepBar steps={progressSteps} currentStep={taskStatus} />
          <AgentRunTimeline agentRun={currentTask?.agentRun} />
          <div className="mt-6 flex items-center gap-2 text-sm text-neutral-500">
            <Loading className="h-5 w-5" />
            <span>
              {screenshotRequested
                ? '笔记正文会优先生成，关键截图随后异步补齐。'
                : '笔记正文会直接生成并保存。'}
            </span>
          </div>
        </div>
      </div>
    )
  }

  if (status === 'idle') {
    return (
      <div className="flex h-full w-full items-center justify-center bg-neutral-50 p-8">
        <div className="max-w-xl text-center">
          <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-lg border border-neutral-200 bg-white shadow-sm">
            <FileText className="h-7 w-7 text-neutral-700" />
          </div>
          <p className="text-xl font-semibold text-neutral-950">准备生成第一篇视频笔记</p>
          <p className="mx-auto mt-3 max-w-md text-sm leading-6 text-neutral-500">
            在左侧粘贴视频链接或选择本地视频。VideoNote 会先整理成 Markdown，再为关键段落补上有用截图。
          </p>
          <div className="mt-6 grid grid-cols-3 gap-2 text-xs text-neutral-600">
            <div className="rounded-md border border-neutral-200 bg-white px-3 py-2">转写</div>
            <div className="rounded-md border border-neutral-200 bg-white px-3 py-2">总结</div>
            <div className="rounded-md border border-neutral-200 bg-white px-3 py-2">配图</div>
          </div>
        </div>
      </div>
    )
  }

  if (status === 'failed' && markdownVersions.length === 0) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-4 bg-neutral-50 p-8">
        <Error />
        <div className="text-center">
          <p className="text-lg font-bold text-red-500">笔记生成失败</p>
          <p className="mt-2 mb-2 text-xs text-red-400">请检查后台或稍后再试</p>

          <Button onClick={() => retryTask(currentTask.id)} size="lg">
            重试
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-white">
      <MarkdownHeader
        currentTask={currentTask}
        isMultiVersion={markdownVersions.length > 1}
        currentVerId={currentVerId}
        setCurrentVerId={setCurrentVerId}
        modelName={modelName}
        style={style}
        noteStyles={noteStyles}
        onCopy={handleCopy}
        onDownload={handleDownload}
        isExporting={isExporting}
        createAt={createTime}
        showTranscribe={showTranscribe}
        setShowTranscribe={setShowTranscribe}
        showChat={showChat}
        setShowChat={setShowChat}
        viewMode={viewMode}
        setViewMode={setViewMode}
        isTaskRunning={Boolean(isTaskRunning)}
        taskStatus={progressStatus}
        visualSummary={screenshotSummary}
      />

      {viewMode === 'map' ? (
        <div className="flex w-full flex-1 overflow-hidden bg-white">
          <div className={'w-full'}>
            <MarkmapEditor
              value={selectedContent}
              onChange={() => {}}
              height="100%" // 根据需求可以设定百分比或固定高度
              title={currentTask?.audioMeta?.title || '思维导图'}
            />
          </div>
        </div>
      ) : (
        <div className="flex flex-1 overflow-hidden bg-white">
          {selectedContent && selectedContent !== 'loading' && selectedContent !== 'empty' ? (
            <>
              {showChat === 'full' && currentTask ? (
                <div className="h-full w-full">
                  <ChatPanel taskId={currentTask.id} mode="full" onModeChange={setShowChat} />
                </div>
              ) : (
                <>
                  <ScrollArea className="min-w-0 flex-1">
                    {isTaskRunning && (
                      <div className="sticky top-0 z-10 border-b border-amber-200 bg-amber-50/95 px-5 py-3 text-amber-900 shadow-sm backdrop-blur">
                        <div className="mb-2 flex items-center gap-2 text-sm">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          <span>{progressMessage}</span>
                        </div>
                        {screenshotSummary && (
                          <div className="mb-2 text-xs text-amber-800">{screenshotSummary}</div>
                        )}
                        <StepBar steps={progressSteps} currentStep={progressStatus} compact />
                      </div>
                    )}
                    <AgentRunTimeline agentRun={currentTask?.agentRun} />
                    {status === 'failed' && markdownVersions.length > 0 && (
                      <div className="sticky top-0 z-10 border-b border-red-200 bg-red-50/95 px-5 py-3 text-red-800 shadow-sm backdrop-blur">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-sm font-medium">重新生成失败，当前显示的是上一个可用版本</div>
                            <div className="mt-1 text-xs leading-5 text-red-700">
                              {taskMessage || '请检查后端日志或调整生成配置后再重试。'}
                            </div>
                          </div>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            className="h-8 border-red-200 bg-white text-red-700 hover:bg-red-100"
                            onClick={() => currentTask && retryTask(currentTask.id)}
                          >
                            重试
                          </Button>
                        </div>
                      </div>
                    )}
                    <div className="px-5 pt-5">
                      <VideoBanner
                        audioMeta={currentTask?.audioMeta}
                        videoUrl={currentTask?.formData?.video_url}
                        fallbackMarkdown={displayContent}
                      />
                    </div>
                    <div className="markdown-body mx-auto w-full max-w-5xl px-5 pb-10">
                      <ReactMarkdown
                        remarkPlugins={remarkPlugins}
                        rehypePlugins={rehypePlugins}
                        components={markdownComponents}
                      >
                        {displayContent.replace(/^>\s*来源链接：[^\n]*\n*/m, '')}
                      </ReactMarkdown>
                    </div>
                  </ScrollArea>
                  {showTranscribe && (
                    <div className="ml-2 w-2/4">
                      <TranscriptViewer />
                    </div>
                  )}
                  {/* 侧边问答模式：markdown + ChatPanel 各占一半 */}
                  {showChat === 'half' && currentTask && (
                    <div className="ml-2 h-full w-1/2 shrink-0">
                      <ChatPanel taskId={currentTask.id} mode="half" onModeChange={setShowChat} />
                    </div>
                  )}
                </>
              )}
            </>
          ) : (
            <div className="flex h-full w-full items-center justify-center bg-neutral-50">
              <div className="w-[320px] flex-col justify-items-center text-center">
                <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-lg border border-neutral-200 bg-white shadow-sm">
                  <ArrowRight className="h-8 w-8 text-neutral-700" />
                </div>
                <p className="mb-2 font-medium text-neutral-800">从左侧开始创建笔记</p>
                <p className="text-xs leading-5 text-neutral-500">支持主流视频平台和本地视频</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
})

MarkdownViewer.displayName = 'MarkdownViewer'

export default MarkdownViewer
