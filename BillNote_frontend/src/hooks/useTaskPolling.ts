import { useEffect, useRef } from 'react'
import toast from 'react-hot-toast'

import { latestMarkdownContent, useTaskStore } from '@/store/taskStore'
import { taskApi } from '@/services/taskApi'
import {
  isFailedTaskStatus,
  isPartialSuccessTaskStatus,
  isRunningTaskStatus,
  isSuccessTaskStatus,
} from '@/models/taskStateMachine'

type StoreTask = ReturnType<typeof useTaskStore.getState>['tasks'][number]

export const useTaskPolling = (interval = 3000, enabled = true) => {
  const tasks = useTaskStore(state => state.tasks)
  const currentTaskId = useTaskStore(state => state.currentTaskId)
  const updateTaskContent = useTaskStore(state => state.updateTaskContent)

  const tasksRef = useRef(tasks)
  const currentTaskIdRef = useRef(currentTaskId)

  useEffect(() => {
    tasksRef.current = tasks
  }, [tasks])

  useEffect(() => {
    currentTaskIdRef.current = currentTaskId
  }, [currentTaskId])

  useEffect(() => {
    if (!enabled) return

    const syncTask = async (task: StoreTask) => {
      try {
        const res = await taskApi.status(task.id, task.generationToken)
        const { status, message } = res
        const latestTask = tasksRef.current.find(item => item.id === task.id)
        if (!latestTask || latestTask.isRetrySubmitting) return
        if (latestTask.generationToken && res.generation_token !== latestTask.generationToken) return

        if (!status) return

        const nextGenerationToken = res.generation_token || latestTask.generationToken
        if (res.result) {
          const { markdown, transcript, audio_meta, visual_report } = res.result
          const agentRunChanged =
            JSON.stringify(res.agent_run ?? null) !== JSON.stringify(latestTask.agentRun ?? null)
          if (isSuccessTaskStatus(status) && !isSuccessTaskStatus(latestTask.status)) {
            toast.success('笔记生成成功')
          }
          if (isPartialSuccessTaskStatus(status) && latestTask.status !== status) {
            toast(message || '笔记已完成，截图未完全完成', { icon: '!' })
          }
          const latestMarkdown = latestMarkdownContent(latestTask.markdown)
          const visualReportChanged =
            JSON.stringify(visual_report ?? null) !== JSON.stringify(latestTask.visualReport ?? null)
          const audioMetaChanged =
            JSON.stringify(audio_meta ?? null) !== JSON.stringify(latestTask.audioMeta ?? null)
          if (
            status !== latestTask.status ||
            message !== latestTask.message ||
            markdown !== latestMarkdown ||
            visualReportChanged ||
            audioMetaChanged ||
            agentRunChanged
          ) {
            updateTaskContent(task.id, {
              status,
              message,
              generationToken: nextGenerationToken,
              markdown,
              transcript,
              audioMeta: audio_meta,
              visualReport: visual_report,
              agentRun: res.agent_run,
            })
          }
          return
        }

        const agentRunChanged =
          JSON.stringify(res.agent_run ?? null) !== JSON.stringify(latestTask.agentRun ?? null)
        if (status !== latestTask.status || message !== latestTask.message || agentRunChanged) {
          updateTaskContent(task.id, {
            status,
            message,
            generationToken: nextGenerationToken,
            agentRun: res.agent_run,
          })
          if (isFailedTaskStatus(status)) {
            console.warn(`任务 ${task.id} 失败`)
          }
        }
      } catch (e) {
        console.error('任务轮询失败：', e)
      }
    }

    const pollTasks = async (includeCurrentTask = false) => {
      const currentId = currentTaskIdRef.current
      const pendingTasks = tasksRef.current.filter(
        task =>
          !task.isRetrySubmitting &&
          (isRunningTaskStatus(task.status) || (task.id === currentId && isFailedTaskStatus(task.status))),
      )

      if (includeCurrentTask && currentId) {
        const currentTask = tasksRef.current.find(task => task.id === currentId)
        if (
          currentTask &&
          !currentTask.isRetrySubmitting &&
          !pendingTasks.some(task => task.id === currentTask.id)
        ) {
          pendingTasks.push(currentTask)
        }
      }

      for (const task of pendingTasks) {
        await syncTask(task)
      }
    }

    void pollTasks(true)
    const timer = setInterval(() => {
      void pollTasks(false)
    }, interval)

    return () => clearInterval(timer)
  }, [enabled, interval, currentTaskId, updateTaskContent])
}
