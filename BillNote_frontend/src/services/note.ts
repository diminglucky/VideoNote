import { taskApi } from '@/services/taskApi'
import type { GenerateNotePayload, GenerateNoteResponse } from '@/services/taskApi'

export type { GenerateNoteResponse }

export const generateNote = (data: GenerateNotePayload): Promise<GenerateNoteResponse> =>
  taskApi.generate(data)

export const delete_task = ({
  task_id,
  video_id,
  platform,
}: {
  task_id?: string
  video_id: string
  platform: string
}) =>
  taskApi.delete({ task_id, video_id, platform })

export const get_task_status = (task_id: string, generation_token?: string) =>
  taskApi.status(task_id, generation_token)
