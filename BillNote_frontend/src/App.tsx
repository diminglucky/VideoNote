import './App.css'

import { lazy, Suspense, useEffect } from 'react'
import { BrowserRouter, HashRouter, Navigate, Route, Routes } from 'react-router-dom'

import BackendHealthIndicator from '@/components/BackendHealth/BackendHealthIndicator'
import BackendInitDialog from '@/components/BackendInitDialog'
import StartupBanner from '@/components/SystemDiagnostic/StartupBanner'
import { ONBOARDING_STORAGE_KEY } from '@/constants/onboarding'
import { useCheckBackend } from '@/hooks/useCheckBackend'
import { useTaskPolling } from '@/hooks/useTaskPolling'
import Index from '@/pages/Index'
import { systemCheck } from '@/services/system'
import { HomePage } from './pages/HomePage/Home'

const Onboarding = lazy(() => import('@/pages/Onboarding'))
const SettingPage = lazy(() => import('./pages/SettingPage/index'))
const Model = lazy(() => import('@/pages/SettingPage/Model'))
const ProviderForm = lazy(() => import('@/components/Form/modelForm/Form'))
const AboutPage = lazy(() => import('@/pages/SettingPage/about'))
const Monitor = lazy(() => import('@/pages/SettingPage/Monitor'))
const Downloader = lazy(() => import('@/pages/SettingPage/Downloader'))
const DownloaderForm = lazy(() => import('@/components/Form/DownloaderForm/Form'))
const TranscriberPage = lazy(() => import('@/pages/SettingPage/transcriber'))
const NotFoundPage = lazy(() => import('@/pages/NotFoundPage'))

function OnboardingGuard({ children }: { children: React.ReactNode }) {
  const isTauri = typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window
  if (!isTauri) return <>{children}</>
  if (localStorage.getItem(ONBOARDING_STORAGE_KEY) !== '1') {
    return <Navigate to="/onboarding" replace />
  }
  return <>{children}</>
}

function App() {
  const { loading, initialized, failed, lastError, retry } = useCheckBackend()
  useTaskPolling(3000, initialized)

  useEffect(() => {
    if (initialized) {
      systemCheck()
    }
  }, [initialized])

  if (!initialized) {
    return (
      <>
        <StartupBanner />
        <BackendInitDialog open={loading} failed={failed} lastError={lastError} onRetry={retry} />
      </>
    )
  }

  const isTauri = typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window
  const Router = isTauri ? HashRouter : BrowserRouter

  return (
    <>
      <StartupBanner />
      <BackendHealthIndicator />
      <Router>
        <Suspense fallback={<div className="flex h-screen items-center justify-center">加载中...</div>}>
          <Routes>
            <Route path="/onboarding" element={<Onboarding />} />
            <Route
              path="/"
              element={(
                <OnboardingGuard>
                  <Index />
                </OnboardingGuard>
              )}
            >
              <Route index element={<HomePage />} />
              <Route path="settings" element={<SettingPage />}>
                <Route index element={<Navigate to="model" replace />} />
                <Route path="model" element={<Model />}>
                  <Route path="new" element={<ProviderForm isCreate />} />
                  <Route path=":id" element={<ProviderForm />} />
                </Route>
                <Route path="download" element={<Downloader />}>
                  <Route path=":id" element={<DownloaderForm />} />
                </Route>
                <Route path="transcriber" element={<TranscriberPage />} />
                <Route path="monitor" element={<Monitor />} />
                <Route path="about" element={<AboutPage />} />
                <Route path="*" element={<NotFoundPage />} />
              </Route>
              <Route path="*" element={<NotFoundPage />} />
            </Route>
          </Routes>
        </Suspense>
      </Router>
    </>
  )
}

export default App
