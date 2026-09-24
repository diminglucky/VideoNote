import { useEffect, useState } from 'react'
import { useModelStore } from '@/store/modelStore'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { CheckCircle2, Loader2 } from 'lucide-react'
import toast from 'react-hot-toast'
import { testConnection } from '@/services/model.ts'

interface ModelSelectorProps {
  providerId: string
  disabled?: boolean
  disabledMessage?: string
  onBeforeLoad?: () => Promise<void>
  onModelSaved?: () => void | Promise<void>
}

export function ModelSelector({
  providerId,
  disabled = false,
  disabledMessage,
  onBeforeLoad,
  onModelSaved,
}: ModelSelectorProps) {
  const { models, loading, selectedModel, loadModels, setSelectedModel, addNewModel } =
    useModelStore()
  const [search, setSearch] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [testing, setTesting] = useState(false)

  const filteredModels = models.filter(model => {
    const keywords = search.trim().toLowerCase().split(/\s+/).filter(Boolean)
    const target = model.id.toLowerCase()
    return keywords.every(kw => target.includes(kw))
  })

  useEffect(() => {
    setSelectedModel('')
  }, [providerId, setSelectedModel])

  const handleLoadModels = async () => {
    if (disabled) {
      toast.error(disabledMessage || '请先完成供应商配置')
      return
    }
    if (onBeforeLoad) {
      try {
        await onBeforeLoad()
      } catch (error: any) {
        toast.error(error?.data?.msg || error?.msg || error?.message || '保存供应商配置失败')
        return
      }
    }
    const loadedModels = await loadModels(providerId, { silent: true })
    if (loadedModels.length > 0) {
      toast.success('模型列表加载成功')
    } else {
      toast.error('未获取到模型列表，请检查供应商配置')
    }
  }

  const handleSubmit = async () => {
    if (disabled) {
      toast.error(disabledMessage || '请先完成供应商配置')
      return
    }
    if (!selectedModel) {
      toast.error('请选择一个模型')
      return
    }
    try {
      setSubmitting(true)
      if (onBeforeLoad) {
        await onBeforeLoad()
      }
      await addNewModel(providerId, selectedModel)
      await onModelSaved?.()
      toast.success('保存模型成功')
    } catch (error) {
      toast.error('保存失败')
    } finally {
      setSubmitting(false)
    }
  }

  const handleTest = async () => {
    if (disabled) {
      toast.error(disabledMessage || '请先完成供应商配置')
      return
    }
    if (!selectedModel) {
      toast.error('请先选择一个模型')
      return
    }
    try {
      setTesting(true)
      if (onBeforeLoad) {
        await onBeforeLoad()
      }
      await testConnection(
        { id: providerId, model: selectedModel },
        { silent: true },
      )
      toast.success('模型连通性测试成功')
    } catch (error: any) {
      toast.error(error?.data?.msg || error?.msg || error?.message || '模型连通性测试失败')
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="grid gap-2">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-neutral-950">选择模型</div>
          <div className="text-xs text-neutral-500">刷新后选择需要启用的模型。</div>
        </div>
        <Button
          variant="outline"
          size="sm"
          type="button"
          onClick={handleLoadModels}
          disabled={loading || disabled}
        >
          {loading ? '加载中...' : '刷新模型'}
        </Button>
      </div>

      <Select value={selectedModel} onValueChange={setSelectedModel}>
        <SelectTrigger className="h-9 w-full">
          <SelectValue placeholder="请选择模型" />
        </SelectTrigger>
        <SelectContent>
          <div className="p-2">
            <Input
              placeholder="搜索模型..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="h-8"
            />
          </div>
          {filteredModels.length > 0 ? (
            filteredModels.map((model, index) => (
              <SelectItem key={`${model.id}-${index}`} value={model.id}>
                {model.id}
              </SelectItem>
            ))
          ) : (
            <div className="px-3 py-2 text-sm text-neutral-500">暂无可选模型</div>
          )}
        </SelectContent>
      </Select>

      {disabledMessage && (
        <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs leading-5 text-amber-800">
          {disabledMessage}
        </div>
      )}

      <div className="grid grid-cols-2 gap-2">
        <Button
          type="button"
          variant="outline"
          onClick={handleTest}
          disabled={disabled || testing || !selectedModel}
          className="h-9"
        >
          {testing ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <CheckCircle2 className="h-4 w-4" />
          )}
          测试
        </Button>
        <Button
          type="button"
          onClick={handleSubmit}
          disabled={disabled || submitting || !selectedModel}
          className="h-9 rounded-md bg-neutral-950 text-white hover:bg-neutral-800"
        >
          {submitting ? '保存中...' : '保存'}
        </Button>
      </div>
    </div>
  )
}
