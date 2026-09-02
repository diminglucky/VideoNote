import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { taskApi } from '@/services/taskApi'
import { v4 as uuidv4 } from 'uuid'
import toast from 'react-hot-toast'
import { get, set, del } from 'idb-keyval'
import type { TaskStatus } from '@/models/taskStateMachine'
import { resolveCoverUrl } from '@/utils/coverImage'
import type { AgentRun } from '@/services/taskApi'


export interface AudioMeta {
  cover_url: string
  duration: number
  file_path: string
  platform: string
  raw_info: any
  title: string
  video_id: string
}

export interface Segment {
  start: number
  end: number
  text: string
}

export interface Transcript {
  full_text: string
  language: string
  raw: any
  segments: Segment[]
}
export interface Markdown {
  ver_id: string
  content: string
  style: string
  model_name: string
  created_at: string
  generationToken?: string
}

export interface Task {
  id: string
  generationToken?: string
  isRetrySubmitting?: boolean
  platform: string
  markdown: Markdown[]
  transcript: Transcript
  visualReport?: any
  agentRun?: AgentRun
  status: TaskStatus
  message?: string
  audioMeta: AudioMeta
  createdAt: string
  formData: {
    video_url: string
    link: undefined | boolean
    screenshot: undefined | boolean
    platform: string
    quality: string
    model_name: string
    provider_id: string
    format?: string[]
    style?: string
    extras?: string
    video_understanding?: boolean
    video_interval?: number
    grid_size?: number[]
  }
}

type TaskUpdate = Partial<Omit<Task, 'id' | 'createdAt' | 'markdown'>> & {
  markdown?: Markdown[] | string
}

const emptyAudioMeta = (): AudioMeta => ({
  cover_url: '',
  duration: 0,
  file_path: '',
  platform: '',
  raw_info: null,
  title: '',
  video_id: '',
})

interface TaskStore {
  tasks: Task[]
  currentTaskId: string | null
  addPendingTask: (taskId: string, platform: string, formData?: any, generationToken?: string) => void
  updateTaskContent: (id: string, data: TaskUpdate) => void
  removeTask: (id: string) => void
  clearTasks: () => void
  setCurrentTask: (taskId: string | null) => void
  getCurrentTask: () => Task | null
  retryTask: (id: string, payload?: any) => Promise<void>
}

export const latestMarkdownContent = (markdown: Markdown[] | string | undefined): string => {
  if (Array.isArray(markdown)) return markdown[0]?.content || ''
  return typeof markdown === 'string' ? markdown : ''
}

const markdownVersionFromString = (
  task: Pick<Task, 'id' | 'formData'>,
  content: string,
  generationToken?: string,
): Markdown => ({
  ver_id: `${task.id}-${uuidv4()}`,
  content,
  style: task.formData?.style || '',
  model_name: task.formData?.model_name || '',
  created_at: new Date().toISOString(),
  generationToken,
})

const normalizeMarkdownVersions = (
  task: Pick<Task, 'id' | 'formData'>,
  markdown: Markdown[] | string | undefined,
): Markdown[] => {
  if (Array.isArray(markdown)) return markdown
  if (typeof markdown === 'string' && markdown.trim()) {
    return [markdownVersionFromString(task, markdown)]
  }
  return []
}

const upsertMarkdownVersion = (
  task: Pick<Task, 'id' | 'formData' | 'generationToken'>,
  versions: Markdown[],
  content: string,
  generationToken?: string,
): Markdown[] => {
  const token = generationToken || task.generationToken

  if (token) {
    const existingIndex = versions.findIndex(version => version.generationToken === token)
    if (existingIndex >= 0) {
      return versions.map((version, index) =>
        index === existingIndex
          ? {
              ...version,
              content,
              style: task.formData?.style || version.style,
              model_name: task.formData?.model_name || version.model_name,
            }
          : version,
      )
    }
    return [markdownVersionFromString(task, content, token), ...versions]
  }

  const latestContent = latestMarkdownContent(versions)
  if (content === latestContent) return versions

  return [markdownVersionFromString(task, content, token), ...versions]
}

const normalizeAudioMeta = (
  next?: Partial<AudioMeta> | null,
  previous?: AudioMeta,
): AudioMeta => {
  const merged = {
    ...emptyAudioMeta(),
    ...(previous || {}),
    ...(next || {}),
    raw_info: {
      ...(previous?.raw_info || {}),
      ...(next?.raw_info || {}),
    },
  }

  merged.cover_url = resolveCoverUrl(merged) || previous?.cover_url || ''
  return merged
}

const mergeTaskUpdate = (task: Task, data: TaskUpdate): TaskUpdate => {
  if (!data.audioMeta) return data
  return {
    ...data,
    audioMeta: normalizeAudioMeta(data.audioMeta, task.audioMeta),
  }
}

export const useTaskStore = create<TaskStore>()(
  persist(
    (set, get) => ({
      tasks: [],
      currentTaskId: null,

      addPendingTask: (taskId: string, platform: string, formData: any, generationToken?: string) =>

        set(state => ({
          tasks: [
            {
              formData: formData,
              id: taskId,
              generationToken,
              status: 'PENDING',
              markdown: [],
              visualReport: null,
              platform: platform,
              transcript: {
                full_text: '',
                language: '',
                raw: null,
                segments: [],
              },
              createdAt: new Date().toISOString(),
              audioMeta: {
                ...emptyAudioMeta(),
              },
            },
            ...state.tasks,
          ],
          currentTaskId: taskId, // 默认设置为当前任务
        })),

      updateTaskContent: (id, data) =>
          set(state => ({
            tasks: state.tasks.map(task => {
              if (task.id !== id) return task
              const nextData = mergeTaskUpdate(task, data)

              if (typeof nextData.markdown === 'string') {
                const prev = normalizeMarkdownVersions(task, task.markdown)
                const nextMarkdown = upsertMarkdownVersion(
                  task,
                  prev,
                  nextData.markdown,
                  nextData.generationToken || task.generationToken,
                )
                if (nextMarkdown === prev) {
                  const { markdown: _markdown, ...rest } = nextData
                  return { ...task, ...rest }
                }
                return {
                  ...task,
                  ...nextData,
                  markdown: nextMarkdown,
                }
              }

              if (Array.isArray(nextData.markdown)) {
                return { ...task, ...nextData, markdown: nextData.markdown }
              }

              return { ...task, ...nextData, markdown: normalizeMarkdownVersions(task, task.markdown) }
            }),
          })),


      getCurrentTask: () => {
        const currentTaskId = get().currentTaskId
        return get().tasks.find(task => task.id === currentTaskId) || null
      },
      retryTask: async (id: string, payload?: any) => {

        if (!id){
          toast.error('任务不存在')
          return
        }
        const task = get().tasks.find(task => task.id === id)
        if (!task) return

        const newFormData = payload || task.formData
        const previousGenerationToken = task.generationToken
        set(state => ({
          tasks: state.tasks.map(t =>
              t.id === id
                  ? {
                    ...t,
                    formData: newFormData,
                    status: 'PENDING',
                    generationToken: undefined,
                    isRetrySubmitting: true,
                    message: '正在提交重新生成请求...',
                    visualReport: null,
                    agentRun: undefined,
                  }
                  : t
          ),
        }))
        try {
          const response = await taskApi.generate({
            ...newFormData,
            task_id: id,
          })
          const generationToken = response?.generation_token
          if (!generationToken) {
            toast.error('后端未返回 generation_token，请重启后端并确认已运行最新代码')
            throw new Error('Regeneration response is missing generation_token')
          }
          set(state => ({
            tasks: state.tasks.map(t =>
                t.id === id
                    ? {
                      ...t,
                      formData: newFormData,
                      status: 'PENDING',
                      message: '重新生成任务已提交，等待后端开始处理...',
                      generationToken,
                      isRetrySubmitting: false,
                    }
                    : t
            ),
          }))
        } catch (e: any) {
          const errorMessage = e?.msg || e?.message || '重新生成请求提交失败，请检查后端状态后再试'
          set(state => ({
            tasks: state.tasks.map(t =>
                t.id === id
                    ? {
                      ...t,
                      status: 'FAILED',
                      message: errorMessage,
                      generationToken: previousGenerationToken,
                      isRetrySubmitting: false,
                    }
                    : t
            ),
          }))
          // 就绪门禁：转写模型未下载好。不要把任务标成 PENDING（会一直转），
          // 给提示让用户先去下载。
          if (e?.data?.reason === 'transcriber_model_not_ready') {
            toast.error(
              e?.data?.downloading
                ? '转写模型正在下载中，请稍候再重试'
                : '转写模型尚未下载，请先去「设置 → 音频转写配置」页下载',
            )
            return
          }
          console.error('重试任务失败：', e)
          return
        }
      },


      removeTask: async id => {
        const task = get().tasks.find(t => t.id === id)

        // 更新 Zustand 状态
        set(state => ({
          tasks: state.tasks.filter(task => task.id !== id),
          currentTaskId: state.currentTaskId === id ? null : state.currentTaskId,
        }))

        // 调用后端删除接口（如果找到了任务）
        if (task) {
          await taskApi.delete({
            video_id: task.audioMeta.video_id,
            platform: task.platform,
          })
        }
      },

      clearTasks: () => set({ tasks: [], currentTaskId: null }),

      setCurrentTask: taskId => set({ currentTaskId: taskId }),
    }),
    {
      name: 'task-storage',
      version: 1,
      storage: createJSONStorage(() => ({
        getItem: async (name: string): Promise<string | null> => {
          const value = await get(name)
          return value ?? null
        },
        setItem: async (name: string, value: string): Promise<void> => {
          await set(name, value)
        },
      removeItem: async (name: string): Promise<void> => {
          await del(name)
        },
      })),
      migrate: (persisted: any) => {
        if (!persisted?.state?.tasks) return persisted
        return {
          ...persisted,
          state: {
            ...persisted.state,
            tasks: persisted.state.tasks.map((task: Task & { markdown?: Markdown[] | string }) => ({
              ...task,
              markdown: normalizeMarkdownVersions(task, task.markdown),
              audioMeta: normalizeAudioMeta(task.audioMeta),
            })),
          },
        }
      },
    }
  )
)
