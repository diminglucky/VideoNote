import { FC, useEffect, useState } from 'react'
import HomeLayout from '@/layouts/HomeLayout.tsx'
import NoteForm from '@/pages/HomePage/components/NoteForm.tsx'
import MarkdownViewer from '@/pages/HomePage/components/MarkdownViewer.tsx'
import { useTaskStore } from '@/store/taskStore'
import History from '@/pages/HomePage/components/History.tsx'
import { isContentReadyTaskStatus, isFailedTaskStatus, isRunningTaskStatus } from '@/models/taskStateMachine'
import { taskApi } from '@/services/taskApi'
type ViewStatus = 'idle' | 'loading' | 'success' | 'failed'
export const HomePage: FC = () => {
  const tasks = useTaskStore(state => state.tasks)
  const currentTaskId = useTaskStore(state => state.currentTaskId)
  const updateTaskContent = useTaskStore(state => state.updateTaskContent)

  const currentTask = tasks.find(t => t.id === currentTaskId)

  const [status, setStatus] = useState<ViewStatus>('idle')
  const hasRenderableContent = Boolean(
    currentTask && currentTask.markdown.length > 0,
  )

  useEffect(() => {
    if (!currentTask) {
      setStatus('idle')
    } else if (isRunningTaskStatus(currentTask.status)) {
      setStatus(hasRenderableContent ? 'success' : 'loading')
    } else if (isFailedTaskStatus(currentTask.status)) {
      setStatus('failed')
    } else if (
      isContentReadyTaskStatus(currentTask.status) ||
      hasRenderableContent
    ) {
      setStatus('success')
    } else {
      // PENDING、PARSING、DOWNLOADING、TRANSCRIBING、SUMMARIZING 等所有进行中状态
      setStatus('loading')
    }
  }, [currentTask, currentTask?.status, hasRenderableContent])

  useEffect(() => {
    if (!currentTask || !isFailedTaskStatus(currentTask.status)) return

    let cancelled = false
    taskApi.status(currentTask.id, currentTask.generationToken)
      .then(res => {
        if (cancelled || !res.status || isFailedTaskStatus(res.status)) return
        updateTaskContent(currentTask.id, {
          status: res.status,
          message: res.message,
          generationToken: res.generation_token || currentTask.generationToken,
          markdown: res.result?.markdown,
          transcript: res.result?.transcript,
          audioMeta: res.result?.audio_meta,
          visualReport: res.result?.visual_report,
          agentRun: res.agent_run,
        })
      })
      .catch(error => {
        console.warn('Failed to reconcile current task status:', error)
      })

    return () => {
      cancelled = true
    }
  }, [currentTask?.id, currentTask?.generationToken, currentTask?.status, updateTaskContent])

  return (
    <HomeLayout
      NoteForm={<NoteForm />}
      Preview={<MarkdownViewer status={status} />}
      History={<History />}
    />
  )
}
