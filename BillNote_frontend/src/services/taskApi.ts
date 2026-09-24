import request from '@/utils/request'
import toast from 'react-hot-toast'
import type { TaskStatus } from '@/models/taskStateMachine'

export interface GenerateNoteResponse {
  task_id: string
  generation_token: string
}

export interface AgentRunEvent {
  id: string
  timestamp?: string | null
  kind: 'decision' | 'tool' | 'observation' | 'final' | string
  agent?: string | null
  action?: string | null
  tool?: string | null
  target_agent?: string | null
  ok?: boolean | null
  error_type?: string | null
  summary?: string
}

export interface AgentRun {
  mode: 'llm_multi_agent' | string
  status: 'running' | 'completed' | 'degraded' | 'unknown' | string
  active_agent?: string | null
  counters: {
    decisions: number
    tool_calls: number
    llm_calls: number
    content_revisions: number
    visual_retries: number
  }
  diagnostics: string[]
  events: AgentRunEvent[]
}

export interface TaskStatusResponse {
  status: TaskStatus
  message?: string
  task_id: string
  generation_token?: string
  agent_run?: AgentRun
  result?: {
    markdown?: string
    transcript?: any
    audio_meta?: any
    visual_report?: any
  }
}

export interface GenerateNotePayload {
  video_url: string
  platform: string
  quality: string
  model_name: string
  provider_id: string
  task_id?: string
  link?: boolean
  screenshot?: boolean
  format: Array<string>
  style: string
  extras?: string
  video_understanding?: boolean
  video_interval?: number
  grid_size: Array<number>
}

export const taskApi = {
  async generate(data: GenerateNotePayload): Promise<GenerateNoteResponse> {
    const response = await request.post('/generate_note', data, { timeout: 60000 }) as GenerateNoteResponse | null
    if (!response?.task_id || !response?.generation_token) {
      toast.error('后端未返回 generation_token，请重启后端并确认已运行最新代码')
      throw new Error('Generate note response is missing task_id or generation_token')
    }
    return response
  },

  async status(taskId: string, generationToken?: string): Promise<TaskStatusResponse> {
    const query = generationToken ? `?generation_token=${encodeURIComponent(generationToken)}` : ''
    return await request.get(`/task_status/${taskId}${query}`, { suppressToast: true }) as TaskStatusResponse
  },

  async delete(data: { video_id: string; platform: string; task_id?: string }) {
    const response = await request.post('/delete_task', data)
    toast.success('任务已成功删除')
    return response
  },
}
