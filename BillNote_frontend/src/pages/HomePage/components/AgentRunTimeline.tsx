import { CheckCircle2, CircleAlert, Clock3, GitBranch, Loader2, Wrench } from 'lucide-react'
import type { AgentRun, AgentRunEvent } from '@/services/taskApi'

interface AgentRunTimelineProps {
  agentRun?: AgentRun
}

const agentLabels: Record<string, string> = {
  supervisor: 'SupervisorAgent',
  content: 'ContentAgent',
  visual: 'VisualAgent',
  reviewer: 'ReviewerAgent',
}

const kindLabels: Record<string, string> = {
  decision: '决策',
  tool: '工具',
  observation: 'Observation',
  final: '最终状态',
}

const statusLabels: Record<string, string> = {
  running: '运行中',
  completed: '已完成',
  degraded: '已降级',
  unknown: '状态未知',
}

const hasCounters = (run: AgentRun) =>
  Object.values(run.counters || {}).some(value => Number(value) > 0)

const eventLabel = (event: AgentRunEvent) => {
  if (event.action) return event.action
  if (event.tool) return event.tool
  return kindLabels[event.kind] || event.kind
}

const eventIcon = (event: AgentRunEvent) => {
  if (event.kind === 'final') {
    return event.ok === false ? (
      <CircleAlert className="h-3.5 w-3.5 text-red-500" />
    ) : (
      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
    )
  }
  if (event.kind === 'tool') return <Wrench className="h-3.5 w-3.5 text-blue-500" />
  if (event.kind === 'decision') return <GitBranch className="h-3.5 w-3.5 text-violet-500" />
  return <Clock3 className="h-3.5 w-3.5 text-neutral-400" />
}

const formatTime = (timestamp?: string | null) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return Number.isNaN(date.getTime())
    ? ''
    : date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

const AgentRunTimeline = ({ agentRun }: AgentRunTimelineProps) => {
  if (!agentRun || (!agentRun.events?.length && !hasCounters(agentRun))) return null

  const events = (agentRun.events || []).slice(-12).reverse()
  const status = statusLabels[agentRun.status] || agentRun.status
  const activeAgent = agentRun.active_agent
    ? agentLabels[agentRun.active_agent] || agentRun.active_agent
    : '等待下一步决策'
  const statusClass =
    agentRun.status === 'degraded'
      ? 'text-amber-700'
      : agentRun.status === 'completed'
        ? 'text-emerald-700'
        : agentRun.status === 'running'
          ? 'text-blue-700'
          : 'text-neutral-600'

  return (
    <details className="mx-5 my-4 overflow-hidden rounded-lg border border-violet-200 bg-violet-50/50 text-sm">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3">
        <div className="flex min-w-0 items-center gap-2">
          {agentRun.status === 'running' ? (
            <Loader2 className="h-4 w-4 shrink-0 animate-spin text-blue-600" />
          ) : (
            <GitBranch className="h-4 w-4 shrink-0 text-violet-600" />
          )}
          <span className="font-medium text-neutral-900">LLM Multi-Agent 运行轨迹</span>
          <span className={`truncate text-xs ${statusClass}`}>{status}</span>
        </div>
        <span className="shrink-0 text-xs text-neutral-500">当前：{activeAgent}</span>
      </summary>

      <div className="border-t border-violet-100 bg-white/70 px-4 py-3">
        <div className="mb-3 grid grid-cols-2 gap-2 text-xs text-neutral-600 sm:grid-cols-5">
          <span>决策 {agentRun.counters.decisions}</span>
          <span>工具 {agentRun.counters.tool_calls}</span>
          <span>LLM {agentRun.counters.llm_calls}</span>
          <span>内容返工 {agentRun.counters.content_revisions}</span>
          <span>视觉重试 {agentRun.counters.visual_retries}</span>
        </div>

        {agentRun.diagnostics?.length > 0 && (
          <div className="mb-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-800">
            {agentRun.diagnostics.slice(-3).map((diagnostic, index) => (
              <div key={`${diagnostic}-${index}`}>{diagnostic}</div>
            ))}
          </div>
        )}

        <div className="space-y-2">
          {events.map(event => (
            <div key={event.id} className="flex items-start gap-2 rounded-md border border-neutral-100 bg-white px-3 py-2">
              <div className="mt-0.5 shrink-0">{eventIcon(event)}</div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
                  <span className="font-medium text-neutral-800">
                    {event.agent ? agentLabels[event.agent] || event.agent : 'Agent'}
                  </span>
                  <span className="text-neutral-400">{kindLabels[event.kind] || event.kind}</span>
                  <span className="rounded bg-neutral-100 px-1.5 py-0.5 text-neutral-600">{eventLabel(event)}</span>
                  {event.timestamp && <span className="text-neutral-400">{formatTime(event.timestamp)}</span>}
                </div>
                {(event.error_type || event.summary) && (
                  <div className="mt-1 truncate text-xs text-neutral-600">
                    {event.error_type ? `${event.error_type}：` : ''}{event.summary}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </details>
  )
}

export default AgentRunTimeline
