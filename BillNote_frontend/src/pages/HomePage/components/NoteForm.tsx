/* NoteForm.tsx ---------------------------------------------------- */
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form.tsx'
import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { useForm, useWatch } from 'react-hook-form'
import type { FieldErrors } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'

import {
  Check,
  FileVideo,
  Info,
  Layers,
  Loader2,
  Plus,
  Settings2,
  Sparkles,
  Wand2,
} from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert.tsx'
import { taskApi } from '@/services/taskApi'
import { uploadFile } from '@/services/upload.ts'
import { useTaskStore } from '@/store/taskStore'
import { useModelStore } from '@/store/modelStore'
import { isRunningTaskStatus } from '@/models/taskStateMachine'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip.tsx'
import { Checkbox } from '@/components/ui/checkbox.tsx'
import { Button } from '@/components/ui/button.tsx'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select.tsx'
import { Input } from '@/components/ui/input.tsx'
import { Textarea } from '@/components/ui/textarea.tsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx'
import { noteStyles, noteFormats, videoPlatforms } from '@/constant/note.ts'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'

/* -------------------- 校验 Schema -------------------- */
const formSchema = z
  .object({
    video_url: z.string().optional(),
    platform: z.string().nonempty('请选择平台'),
    quality: z.enum(['fast', 'medium', 'slow']),
    screenshot: z.boolean().optional(),
    link: z.boolean().optional(),
    model_name: z.string().nonempty('请选择模型'),
    model_ref: z.string().optional(),
    format: z.array(z.string()).default([]),
    style: z.string().nonempty('请选择笔记生成风格'),
    extras: z.string().optional(),
    video_understanding: z.boolean().optional(),
    video_interval: z.coerce.number().min(1).max(30).default(6).optional(),
    grid_size: z
      .tuple([z.coerce.number().min(1).max(4), z.coerce.number().min(1).max(4)])
      .default([2, 2])
      .optional(),
  })
  .superRefine(({ video_url, platform }, ctx) => {
    if (platform === 'local') {
      if (!video_url) {
        ctx.addIssue({ code: 'custom', message: '本地视频路径不能为空', path: ['video_url'] })
      }
    }
    else {
      if (!video_url) {
        ctx.addIssue({ code: 'custom', message: '视频链接不能为空', path: ['video_url'] })
      }
      else {
        try {
          const url = new URL(video_url)
          if (!['http:', 'https:'].includes(url.protocol))
            throw new Error()
        }
        catch {
          ctx.addIssue({ code: 'custom', message: '请输入正确的视频链接', path: ['video_url'] })
        }
      }
    }
  })

export type NoteFormValues = z.infer<typeof formSchema>

/* -------------------- 可复用子组件 -------------------- */
const PanelSection = ({
  title,
  tip,
  description,
  children,
}: {
  title: string
  tip?: string
  description?: string
  children: ReactNode
}) => (
  <section className="rounded-md bg-white">
    <div className="mb-3 flex items-start justify-between gap-3 border-b border-neutral-100 pb-2.5">
      <div className="min-w-0">
        <h2 className="text-sm font-semibold text-neutral-950">{title}</h2>
        {description && <p className="mt-0.5 text-xs leading-5 text-neutral-500">{description}</p>}
      </div>
      {tip && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="hover:text-primary h-4 w-4 cursor-pointer text-neutral-400" />
            </TooltipTrigger>
            <TooltipContent className="text-xs">{tip}</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
    {children}
  </section>
)

const SummaryChip = ({ label, value }: { label: string; value: string }) => (
  <span className="inline-flex min-w-0 max-w-full items-center gap-1 rounded-md border border-neutral-200 bg-neutral-50 px-2 py-1 text-[11px] leading-none text-neutral-500">
    <span className="shrink-0">{label}</span>
    <span className="min-w-0 truncate font-medium text-neutral-900">{value || '-'}</span>
  </span>
)

const ChoiceButton = ({
  active,
  title,
  description,
  onClick,
}: {
  active: boolean
  title: string
  description?: string
  onClick: () => void
}) => (
  <button
    type="button"
    onClick={onClick}
    className={`flex min-h-16 w-full items-start gap-3 rounded-md border p-3 text-left transition ${
      active
        ? 'border-neutral-900 bg-neutral-100 text-neutral-950 shadow-sm'
        : 'border-neutral-200 bg-neutral-50 text-neutral-800 hover:border-neutral-300 hover:bg-white'
    }`}
  >
    <span
      className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border ${
        active ? 'border-neutral-900 bg-neutral-950 text-white' : 'border-neutral-300 bg-white text-transparent'
      }`}
    >
      <Check className="h-3 w-3" />
    </span>
    <span className="min-w-0">
      <span className="block text-sm font-medium">{title}</span>
      {description && (
        <span className={`mt-1 block text-xs leading-5 ${active ? 'text-neutral-600' : 'text-neutral-500'}`}>
          {description}
        </span>
      )}
    </span>
  </button>
)

const CheckboxGroup = ({
  value = [],
  onChange,
  disabledMap,
}: {
  value?: string[]
  onChange: (v: string[]) => void
  disabledMap: Record<string, boolean>
}) => (
  <div className="grid gap-2">
    {noteFormats.map(({ label, value: v }) => (
      <label
        key={v}
        className="flex min-h-10 items-center gap-2 rounded-md border border-neutral-200 bg-neutral-50 px-3 text-sm text-neutral-700 transition hover:border-neutral-300 hover:bg-white has-[[data-state=checked]]:border-neutral-950 has-[[data-state=checked]]:bg-white"
      >
        <Checkbox
          checked={value.includes(v)}
          disabled={disabledMap[v]}
          onCheckedChange={checked =>
            onChange(checked ? [...value, v] : value.filter(x => x !== v))
          }
        />
        <span>{label}</span>
      </label>
    ))}
  </div>
)

const qualityOptions = [
  { value: 'fast', title: '快速', description: '优先速度，适合普通口播和初稿' },
  { value: 'medium', title: '均衡', description: '推荐配置，速度和质量更稳' },
  { value: 'slow', title: '精细', description: '更充分处理长视频和复杂内容' },
] as const

/* -------------------- 主组件 -------------------- */
const NoteForm = () => {
  const navigate = useNavigate()
  const [isUploading, setIsUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [activeTab, setActiveTab] = useState('video')
  /* ---- 全局状态 ---- */
  const { addPendingTask, currentTaskId, setCurrentTask, getCurrentTask, retryTask } =
    useTaskStore()
  const { loadEnabledModels, modelList } = useModelStore()

  /* ---- 表单 ---- */
  const form = useForm<NoteFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      platform: 'bilibili',
      quality: 'medium',
      model_name: modelList[0]?.model_name || '',
      model_ref: modelList[0] ? `${modelList[0].provider_id}::${modelList[0].model_name}` : '',
      style: 'minimal',
      video_interval: 6,
      grid_size: [2, 2],
      format: [],
    },
  })
  const currentTask = getCurrentTask()
  const isRetrySubmitting = Boolean(currentTask?.isRetrySubmitting)

  /* ---- 派生状态（只 watch 一次，提高性能） ---- */
  const platform = useWatch({ control: form.control, name: 'platform' }) as string
  const videoUnderstandingEnabled = useWatch({ control: form.control, name: 'video_understanding' })
  const selectedModel = useWatch({ control: form.control, name: 'model_name' })
  const selectedModelRef = useWatch({ control: form.control, name: 'model_ref' })
  const selectedStyle = useWatch({ control: form.control, name: 'style' })
  const selectedQuality = useWatch({ control: form.control, name: 'quality' })
  const selectedFormats = useWatch({ control: form.control, name: 'format' }) || []
  const selectedModelLabel = useMemo(() => {
    const selected = modelList.find(
      item => `${item.provider_id}::${item.model_name}` === selectedModelRef,
    )
    return selected ? `${selected.model_name} · ${selected.provider_id}` : (selectedModel || '未选择')
  }, [modelList, selectedModel, selectedModelRef])
  const editing = currentTask && currentTask.id
  const platformLabel = useMemo(
    () => videoPlatforms.find(item => item.value === platform)?.label || '未选择',
    [platform],
  )
  const styleLabel = useMemo(
    () => noteStyles.find(item => item.value === selectedStyle)?.label || '未选择',
    [selectedStyle],
  )
  const qualityLabel = useMemo(
    () => qualityOptions.find(item => item.value === selectedQuality)?.title || '均衡',
    [selectedQuality],
  )
  const outputLabel = selectedFormats.length > 0 ? `${selectedFormats.length} 项` : '默认'

  const goModelAdd = () => {
    navigate('/settings/model')
  }
  /* ---- 副作用 ---- */
  useEffect(() => {
    loadEnabledModels()
  }, [loadEnabledModels])

  useEffect(() => {
    if (!form.getValues('model_ref') && modelList[0]?.model_name) {
      form.setValue('model_name', modelList[0].model_name)
      form.setValue('model_ref', `${modelList[0].provider_id}::${modelList[0].model_name}`)
    }
  }, [form, modelList])

  useEffect(() => {
    if (!currentTaskId) return
    const task = getCurrentTask()
    if (!task) return
    const { formData } = task

    form.reset({
      platform: formData.platform || 'bilibili',
      video_url: formData.video_url || '',
      model_name: formData.model_name || modelList[0]?.model_name || '',
      model_ref: formData.provider_id && formData.model_name
        ? `${formData.provider_id}::${formData.model_name}`
        : (modelList[0] ? `${modelList[0].provider_id}::${modelList[0].model_name}` : ''),
      style: formData.style || 'minimal',
      quality: formData.quality || 'medium',
      extras: formData.extras || '',
      screenshot: formData.screenshot ?? false,
      link: formData.link ?? false,
      video_understanding: formData.video_understanding ?? false,
      video_interval: formData.video_interval ?? 6,
      grid_size: formData.grid_size ?? [2, 2],
      format: formData.format ?? [],
    })
  }, [
    currentTaskId,
    form,
    getCurrentTask,
  ])

  /* ---- 帮助函数 ---- */
  const isGenerating = () => isRunningTaskStatus(getCurrentTask()?.status)
  const generating = isSubmitting || isRetrySubmitting || isGenerating()
  const handleFileUpload = async (file: File, cb: (url: string) => void) => {
    const formData = new FormData()
    formData.append('file', file)
    setIsUploading(true)
    setUploadSuccess(false)

    try {
      const data = await uploadFile(formData)
      cb(data.url)
      setUploadSuccess(true)
    } catch (err) {
      console.error('上传失败:', err)
      // message.error('上传失败，请重试')
    } finally {
      setIsUploading(false)
    }
  }

  const onSubmit = async (values: NoteFormValues) => {
    if (isSubmitting || isGenerating()) return
    const selectedModelConfig = modelList.find(
      m => `${m.provider_id}::${m.model_name}` === values.model_ref,
    ) || modelList.find(m => m.model_name === values.model_name)
    if (!selectedModelConfig) {
      toast.error('请先选择可用模型')
      return
    }
    const payload: NoteFormValues = {
      ...values,
      provider_id: selectedModelConfig.provider_id,
      task_id: currentTaskId || '',
    }
    setIsSubmitting(true)
    if (currentTaskId) {
      toast.loading('已提交重新生成请求，旧笔记会先保留', { id: `retry-${currentTaskId}` })
    }
    if (currentTaskId) {
      try {
        await retryTask(currentTaskId, payload)
        toast.success('重新生成任务已开始', { id: `retry-${currentTaskId}` })
      } finally {
        setIsSubmitting(false)
      }
      return
    }

    // message.success('已提交任务')
    try {
      const data = await taskApi.generate(payload)
      addPendingTask(data.task_id, values.platform, payload, data.generation_token)
      toast.success('笔记生成任务已提交！')
    } catch (e: any) {
      // 就绪门禁：本地转写模型还没下载好。后端返回 reason='transcriber_model_not_ready'，
      // 引导用户去「设置 → 音频转写配置」下载，而不是留一个静默失败的任务。
      if (e?.data?.reason === 'transcriber_model_not_ready') {
        const downloading = e?.data?.downloading
        toast.error(
          downloading
            ? '转写模型正在下载中，请稍候再提交'
            : '转写模型尚未下载，请先去「音频转写配置」页下载',
        )
        if (!downloading) navigate('/settings/transcriber')
        return
      }
      // 其余错误：axios 拦截器已经弹过 toast，这里只兜底不让 promise 变成未处理 rejection
      console.error('提交任务失败：', e)
    } finally {
      setIsSubmitting(false)
    }
  }
  const onInvalid = (errors: FieldErrors<NoteFormValues>) => {
    console.warn('表单校验失败：', errors)
    const errorFields = Object.keys(errors)
    if (errorFields.some(name => ['platform', 'video_url'].includes(name))) {
      setActiveTab('video')
    } else if (errorFields.some(name => ['model_name', 'style', 'quality'].includes(name))) {
      setActiveTab('model')
    } else if (errorFields.some(name => ['video_understanding', 'video_interval', 'grid_size'].includes(name))) {
      setActiveTab('vision')
    } else {
      setActiveTab('output')
    }
    toast.error('请先完善当前表单信息')
  }
  const handleCreateNew = () => {
    // 🔁 这里清空当前任务状态
    // 比如调用 resetCurrentTask() 或者 navigate 到一个新页面
    setCurrentTask(null)
  }
  const FormButton = () => {
    const label = isSubmitting
      ? '正在提交...'
      : generating
        ? '正在生成...'
        : editing
          ? '重新生成'
          : '生成笔记'

    return (
      <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2">
        <Button
          type="submit"
          className="h-9 rounded-md bg-neutral-950 px-3 text-sm text-white shadow-sm hover:bg-neutral-800"
          disabled={generating}
        >
          {generating && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          {label}
        </Button>

        {editing && (
          <Button
            type="button"
            variant="outline"
            className="h-9 rounded-md px-3 text-sm"
            onClick={handleCreateNew}
          >
            <Plus className="mr-2 h-4 w-4" />
            新建
          </Button>
        )}
      </div>
    )
  }

  /* -------------------- 渲染 -------------------- */
  return (
    <div className="h-full w-full">
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit, onInvalid)}>
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="rounded-lg border border-neutral-200 bg-white shadow-sm"
          >
            <div className="border-b border-neutral-100 p-3 pb-2.5">
              <div className="mb-2.5 flex items-start gap-2.5">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-neutral-950 text-white">
                  <Wand2 className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex min-w-0 items-center justify-between gap-2">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-semibold text-neutral-950">
                        {editing ? '调整当前笔记' : '创建视频笔记'}
                      </div>
                      <p className="mt-0.5 truncate text-xs text-neutral-500">
                        先生成正文，再异步补齐关键画面。
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mb-2.5 flex flex-wrap gap-1.5">
                <SummaryChip label="来源" value={platformLabel} />
                <SummaryChip label="模型" value={selectedModelLabel} />
                <SummaryChip label="风格" value={styleLabel} />
                <SummaryChip label="输出" value={`${qualityLabel} / ${outputLabel}`} />
              </div>

              <FormButton />
            </div>

            <TabsList className="grid h-auto w-full grid-cols-4 rounded-none border-b border-neutral-100 bg-neutral-50 p-1">
              <TabsTrigger value="video" className="h-8 gap-1 text-xs data-[state=active]:bg-white">
                <FileVideo className="h-3.5 w-3.5" />
                视频
              </TabsTrigger>
              <TabsTrigger value="model" className="h-8 gap-1 text-xs data-[state=active]:bg-white">
                <Settings2 className="h-3.5 w-3.5" />
                模型
              </TabsTrigger>
              <TabsTrigger value="vision" className="h-8 gap-1 text-xs data-[state=active]:bg-white">
                <Sparkles className="h-3.5 w-3.5" />
                视觉
              </TabsTrigger>
              <TabsTrigger value="output" className="h-8 gap-1 text-xs data-[state=active]:bg-white">
                <Layers className="h-3.5 w-3.5" />
                输出
              </TabsTrigger>
            </TabsList>

            <TabsContent value="video" className="m-0 p-3">
              <PanelSection
                title="视频来源"
                tip="支持 B 站、YouTube、抖音、快手和本地视频"
                description="选择平台后粘贴链接，或切换到本地视频上传。"
              >
                <div className="grid gap-3">
                  <FormField
                    control={form.control}
                    name="platform"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>平台</FormLabel>
                        <Select
                          disabled={!!editing}
                          value={field.value}
                          onValueChange={field.onChange}
                          defaultValue={field.value}
                        >
                          <FormControl>
                            <SelectTrigger className="h-10 w-full">
                              <SelectValue />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {videoPlatforms?.map(p => (
                              <SelectItem key={p.value} value={p.value}>
                                <div className="flex items-center justify-center gap-2">
                                  <div className="h-4 w-4">{p.logo()}</div>
                                  <span>{p.label}</span>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="video_url"
                    render={({ field }) => (
                      <FormItem className="min-w-0">
                        <FormLabel>{platform === 'local' ? '视频路径' : '视频链接'}</FormLabel>
                        <Input
                          disabled={!!editing}
                          className="h-10"
                          placeholder={platform === 'local' ? '请输入本地视频路径' : '粘贴视频链接'}
                          {...field}
                        />
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="video_url"
                    render={({ field }) => (
                      <FormItem>
                        {platform === 'local' && (
                          <div
                            className="flex h-32 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-neutral-300 bg-neutral-50 text-center transition hover:border-neutral-400 hover:bg-white"
                            onDragOver={e => {
                              e.preventDefault()
                              e.stopPropagation()
                            }}
                            onDrop={e => {
                              e.preventDefault()
                              const file = e.dataTransfer.files?.[0]
                              if (file) handleFileUpload(file, field.onChange)
                            }}
                            onClick={() => {
                              const input = document.createElement('input')
                              input.type = 'file'
                              input.accept = 'video/*'
                              input.onchange = e => {
                                const file = (e.target as HTMLInputElement).files?.[0]
                                if (file) handleFileUpload(file, field.onChange)
                              }
                              input.click()
                            }}
                          >
                            {isUploading ? (
                              <p className="text-sm text-blue-600">上传中，请稍候...</p>
                            ) : uploadSuccess ? (
                              <p className="text-sm text-emerald-600">上传成功</p>
                            ) : (
                              <p className="text-sm text-neutral-500">
                                拖拽视频到这里
                                <br />
                                <span className="text-xs text-neutral-400">或点击选择文件</span>
                              </p>
                            )}
                          </div>
                        )}
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              </PanelSection>
            </TabsContent>

            <TabsContent value="model" className="m-0 p-3">
              <PanelSection
                title="模型与风格"
                description="选择负责写作的模型，并决定笔记的表达密度。"
              >
                <div className="grid gap-4">
                  {modelList.length > 0 ? (
                    <FormField
                      control={form.control}
                      name="model_ref"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>模型</FormLabel>
                          <Select
                            onOpenChange={() => {
                              loadEnabledModels()
                            }}
                            value={field.value}
                            onValueChange={nextRef => {
                              const selected = modelList.find(
                                item => `${item.provider_id}::${item.model_name}` === nextRef,
                              )
                              field.onChange(nextRef)
                              if (selected) form.setValue('model_name', selected.model_name)
                            }}
                            defaultValue={field.value}
                          >
                            <FormControl>
                              <SelectTrigger className="w-full min-w-0 truncate">
                                <SelectValue />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              {modelList.map(m => (
                                <SelectItem
                                  key={`${m.provider_id}::${m.model_name}::${m.id}`}
                                  value={`${m.provider_id}::${m.model_name}`}
                                >
                                  {m.model_name} · {m.provider_id}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  ) : (
                    <FormItem>
                      <FormLabel>模型</FormLabel>
                      <Button type="button" variant="outline" className="w-full" onClick={goModelAdd}>
                        请先添加模型
                      </Button>
                      <FormMessage />
                    </FormItem>
                  )}

                  <FormField
                    control={form.control}
                    name="style"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>笔记风格</FormLabel>
                        <Select
                          value={field.value}
                          onValueChange={field.onChange}
                          defaultValue={field.value}
                        >
                          <FormControl>
                            <SelectTrigger className="w-full min-w-0 truncate">
                              <SelectValue />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {noteStyles.map(({ label, value }) => (
                              <SelectItem key={value} value={value}>
                                {label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="quality"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>生成档位</FormLabel>
                        <div className="grid gap-2">
                          {qualityOptions.map(option => (
                            <ChoiceButton
                              key={option.value}
                              active={field.value === option.value}
                              title={option.title}
                              description={option.description}
                              onClick={() => field.onChange(option.value)}
                            />
                          ))}
                        </div>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              </PanelSection>
            </TabsContent>

            <TabsContent value="vision" className="m-0 p-3">
              <PanelSection
                title="视觉增强"
                tip="将视频截图发给多模态模型辅助分析"
                description="适合图表、代码演示、PPT 课程；普通口播可以关闭。"
              >
                <div className="flex flex-col gap-3">
                  <FormField
                    control={form.control}
                    name="video_understanding"
                    render={() => (
                      <FormItem>
                        <div className="flex items-center justify-between gap-3 rounded-md border border-neutral-200 bg-neutral-50 px-3 py-3">
                          <div>
                            <FormLabel>启用视频理解</FormLabel>
                            <p className="mt-0.5 text-xs text-neutral-500">适合课程、代码演示、图表密集视频</p>
                          </div>
                          <Checkbox
                            checked={videoUnderstandingEnabled}
                            onCheckedChange={v => form.setValue('video_understanding', Boolean(v))}
                          />
                        </div>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <div className="grid grid-cols-2 gap-3">
                    <FormField
                      control={form.control}
                      name="video_interval"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>采样间隔（秒）</FormLabel>
                          <Input
                            disabled={!videoUnderstandingEnabled}
                            min={1}
                            max={30}
                            type="number"
                            {...field}
                          />
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="grid_size"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>拼图尺寸</FormLabel>
                          <div className="flex items-center gap-2">
                            <Input
                              disabled={!videoUnderstandingEnabled}
                              type="number"
                              min={1}
                              max={4}
                              value={field.value?.[0] || 3}
                              onChange={e => field.onChange([Math.max(1, Math.min(+e.target.value, 4)), field.value?.[1] || 3])}
                              className="w-16"
                            />
                            <span className="text-xs text-neutral-400">x</span>
                            <Input
                              disabled={!videoUnderstandingEnabled}
                              type="number"
                              min={1}
                              max={4}
                              value={field.value?.[1] || 3}
                              onChange={e => field.onChange([field.value?.[0] || 3, Math.max(1, Math.min(+e.target.value, 4))])}
                              className="w-16"
                            />
                          </div>
                          <p className="text-muted-foreground text-xs">建议 2 x 2 或 3 x 3</p>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>
                  <Alert variant="warning" className="text-sm">
                    <AlertDescription>
                      <strong>提示：</strong>视频理解功能必须使用多模态模型。
                    </AlertDescription>
                  </Alert>
                </div>
              </PanelSection>
            </TabsContent>

            <TabsContent value="output" className="m-0 p-3">
              <PanelSection
                title="输出结构"
                tip="选择要包含的笔记元素"
                description="控制笔记里是否带目录、跳转、截图和摘要，并补充个性化要求。"
              >
                <div className="space-y-4">
                  <FormField
                    control={form.control}
                    name="format"
                    render={({ field }) => (
                      <FormItem>
                        <CheckboxGroup
                          value={field.value}
                          onChange={field.onChange}
                          disabledMap={{
                            link: platform === 'local',
                            screenshot: false,
                          }}
                        />
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="extras"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>补充要求</FormLabel>
                        <Textarea
                          className="min-h-28 resize-none border-neutral-200 bg-neutral-50 focus-visible:bg-white"
                          placeholder="例如：突出实操步骤，保留关键公式，避免泛泛总结..."
                          {...field}
                        />
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              </PanelSection>
            </TabsContent>
          </Tabs>
        </form>
      </Form>
    </div>
  )
}

export default NoteForm
