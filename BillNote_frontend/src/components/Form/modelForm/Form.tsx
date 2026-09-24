import { zodResolver } from '@hookform/resolvers/zod'
import { KeyRound, Server, Trash2, TriangleAlert } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { useForm, useWatch } from 'react-hook-form'
import toast from 'react-hot-toast'
import { useNavigate, useParams } from 'react-router-dom'
import { z } from 'zod'

import { ModelSelector } from '@/components/Form/modelForm/ModelSelector.tsx'
import { Alert, AlertDescription } from '@/components/ui/alert.tsx'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'
import { deleteModelById } from '@/services/model.ts'
import { useModelStore } from '@/store/modelStore'
import { useProviderStore } from '@/store/providerStore'

const ProviderSchema = z.object({
  name: z.string().min(2, '名称不能少于 2 个字符'),
  apiKey: z.string().optional(),
  baseUrl: z.string().url('必须是合法 URL'),
  type: z.string(),
})

type ProviderFormValues = z.infer<typeof ProviderSchema>

type EnabledModel = {
  id: number | string
  model_name: string
}

const FieldRow = ({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) => (
  <div className="grid gap-2 md:grid-cols-[88px_minmax(0,1fr)] md:items-start">
    <FormLabel className="pt-2 text-sm font-medium text-neutral-700 md:text-right">{label}</FormLabel>
    <div className="min-w-0">{children}</div>
  </div>
)

const Panel = ({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children: ReactNode
}) => (
  <section className="flex min-h-0 shrink-0 flex-col rounded-lg border border-neutral-200 bg-white shadow-sm">
    <div className="shrink-0 border-b border-neutral-100 px-4 py-2.5">
      <h2 className="text-sm font-semibold text-neutral-950">{title}</h2>
      <p className="mt-0.5 text-xs leading-5 text-neutral-500">{description}</p>
    </div>
    <div className="min-h-0 flex-1 p-3.5">{children}</div>
  </section>
)

const ProviderForm = ({ isCreate = false }: { isCreate?: boolean }) => {
  const { id: routeId } = useParams()
  const navigate = useNavigate()
  const isEditMode = !isCreate
  const savedProviderId = isEditMode ? routeId : undefined

  const loadProviderById = useProviderStore(state => state.loadProviderById)
  const updateProvider = useProviderStore(state => state.updateProvider)
  const addNewProvider = useProviderStore(state => state.addNewProvider)
  const loadModelsById = useModelStore(state => state.loadModelsById)

  const [loading, setLoading] = useState(true)
  const [isBuiltIn, setIsBuiltIn] = useState(false)
  const [enabledModels, setEnabledModels] = useState<EnabledModel[]>([])

  const providerForm = useForm<ProviderFormValues>({
    resolver: zodResolver(ProviderSchema),
    defaultValues: {
      name: '',
      apiKey: '',
      baseUrl: '',
      type: 'custom',
    },
  })

  const apiKey = useWatch({ control: providerForm.control, name: 'apiKey' })
  const baseUrl = useWatch({ control: providerForm.control, name: 'baseUrl' })
  const providerName = useWatch({ control: providerForm.control, name: 'name' })

  const refreshEnabledModels = useCallback(
    async (providerId = savedProviderId) => {
      if (!providerId) {
        setEnabledModels([])
        return
      }
      const models = await loadModelsById(providerId)
      setEnabledModels((models || []) as EnabledModel[])
    },
    [loadModelsById, savedProviderId],
  )

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      if (isEditMode && routeId) {
        const data = await loadProviderById(routeId)
        providerForm.reset(data)
        setIsBuiltIn(data.type === 'built-in')
        await refreshEnabledModels(routeId)
      } else {
        providerForm.reset({
          name: '',
          apiKey: '',
          baseUrl: '',
          type: 'custom',
        })
        setIsBuiltIn(false)
        setEnabledModels([])
      }
      setLoading(false)
    }

    load()
  }, [isEditMode, loadProviderById, providerForm, refreshEnabledModels, routeId])

  const modelDisabledMessage = useMemo(() => {
    if (!savedProviderId) return '保存供应商后可以添加模型。'
    if (!apiKey || !baseUrl) return '填写 API Key 和 API 地址后，再刷新模型列表。'
    return ''
  }, [apiKey, baseUrl, savedProviderId])

  const persistProviderDraft = useCallback(async () => {
    if (!savedProviderId) {
      throw new Error('请先保存供应商信息')
    }
    const values = providerForm.getValues()
    await updateProvider({ ...values, id: savedProviderId })
    providerForm.reset(values)
  }, [providerForm, savedProviderId, updateProvider])

  const handleDelete = async (modelId: number | string) => {
    if (!window.confirm('确定要删除这个模型吗？')) return

    try {
      await deleteModelById(Number(modelId))
      await refreshEnabledModels()
      toast.success('删除成功')
    } catch (error) {
      toast.error('删除异常')
    }
  }

  const onProviderSubmit = async (values: ProviderFormValues) => {
    try {
      if (isEditMode && savedProviderId) {
        await updateProvider({ ...values, id: savedProviderId })
        providerForm.reset(values)
        toast.success('更新供应商成功')
        return
      }

      const created = (await addNewProvider({ ...values })) as any
      const nextId = typeof created === 'string' ? created : created?.id
      toast.success('新增供应商成功')
      if (nextId) navigate(`/settings/model/${nextId}`, { replace: true })
    } catch (error: any) {
      toast.error(error?.data?.msg || error?.msg || error?.message || '保存供应商失败')
    }
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-neutral-500">
        正在加载供应商配置...
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden p-4">
      <div className="mb-3 flex shrink-0 flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-neutral-400">
            <Server className="h-3.5 w-3.5" />
            Model Provider
          </div>
          <h1 className="truncate text-xl font-semibold text-neutral-950">
            {isEditMode ? providerName || '编辑模型供应商' : '新增模型供应商'}
          </h1>
        </div>
        <span className="rounded-md border border-neutral-200 bg-neutral-50 px-2.5 py-1 text-xs text-neutral-600">
          {isBuiltIn ? '内置供应商' : '自定义供应商'}
        </span>
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
        <Form {...providerForm}>
          <form onSubmit={providerForm.handleSubmit(onProviderSubmit)} className="min-h-0">
            <Panel
              title="供应商配置"
              description="保存基础配置后，再测试连通性并刷新远程模型列表。"
            >
              <div className="flex h-full min-h-0 flex-col">
                <div className="grid gap-2.5">
                  {!isBuiltIn && (
                    <Alert className="border-amber-200 bg-amber-50 py-1.5 text-amber-900">
                      <TriangleAlert className="h-4 w-4" />
                      <AlertDescription>
                        自定义供应商需要兼容 OpenAI SDK 格式，否则模型列表和对话请求可能失败。
                      </AlertDescription>
                    </Alert>
                  )}

                  <FormField
                    control={providerForm.control}
                    name="name"
                    render={({ field }) => (
                      <FormItem>
                        <FieldRow label="名称">
                          <FormControl>
                            <Input {...field} disabled={isBuiltIn} className="h-9" />
                          </FormControl>
                          <FormMessage className="mt-1" />
                        </FieldRow>
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={providerForm.control}
                    name="apiKey"
                    render={({ field }) => (
                      <FormItem>
                        <FieldRow label="API Key">
                          <FormControl>
                            <div className="relative">
                              <KeyRound className="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-neutral-400" />
                              <Input
                                {...field}
                                type="password"
                                className="h-9 pl-9"
                                placeholder="填入供应商密钥"
                              />
                            </div>
                          </FormControl>
                          <FormMessage className="mt-1" />
                        </FieldRow>
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={providerForm.control}
                    name="baseUrl"
                    render={({ field }) => (
                      <FormItem>
                        <FieldRow label="API 地址">
                          <FormControl>
                            <Input
                              {...field}
                              className="h-9"
                              placeholder="https://api.openai.com/v1"
                            />
                          </FormControl>
                          <FormMessage className="mt-1" />
                        </FieldRow>
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={providerForm.control}
                    name="type"
                    render={({ field }) => (
                      <FormItem>
                        <FieldRow label="类型">
                          <FormControl>
                            <Input {...field} disabled className="h-9 bg-neutral-50" />
                          </FormControl>
                          <FormMessage className="mt-1" />
                        </FieldRow>
                      </FormItem>
                    )}
                  />
                </div>

                {!isEditMode && (
                  <div className="mt-auto flex justify-end border-t border-neutral-100 pt-4">
                    <Button
                      type="submit"
                      className="h-9 rounded-md bg-neutral-950 px-4 text-white hover:bg-neutral-800"
                    >
                      保存创建
                    </Button>
                  </div>
                )}
              </div>
            </Panel>
          </form>
        </Form>

        <Panel
          title="模型管理"
          description="只启用你实际会在笔记生成中使用的模型。"
        >
          <div className="flex h-full min-h-0 flex-col gap-3">
            <ModelSelector
              providerId={savedProviderId || ''}
              disabled={Boolean(modelDisabledMessage)}
              disabledMessage={modelDisabledMessage}
              onBeforeLoad={persistProviderDraft}
              onModelSaved={() => refreshEnabledModels()}
            />

            <div className="flex min-h-0 flex-1 flex-col rounded-lg border border-neutral-200 bg-neutral-50 p-3">
              <div className="shrink-0">
                <h3 className="text-sm font-semibold text-neutral-950">已启用模型</h3>
                <p className="mt-1 text-xs text-neutral-500">
                  {enabledModels.length} 个模型可用于生成笔记
                </p>
              </div>

              <div className="mt-3 flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
                {enabledModels.length > 0 ? (
                  enabledModels.map(model => (
                    <div
                      key={model.id}
                      className="flex items-center justify-between gap-2 rounded-md border border-neutral-200 bg-white px-3 py-2"
                    >
                      <span className="min-w-0 truncate text-sm font-medium text-neutral-800">
                        {model.model_name}
                      </span>
                      <button
                        type="button"
                        onClick={() => handleDelete(model.id)}
                        className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-neutral-400 transition hover:bg-red-50 hover:text-red-600"
                        aria-label={`删除 ${model.model_name}`}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))
                ) : (
                  <div className="flex min-h-24 flex-1 items-center justify-center rounded-md border border-dashed border-neutral-300 bg-white px-3 py-4 text-center text-sm text-neutral-500">
                    暂未启用模型
                  </div>
                )}
              </div>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  )
}

export default ProviderForm
