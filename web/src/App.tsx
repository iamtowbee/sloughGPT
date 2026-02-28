import React, { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Link, useLocation, useNavigate } from 'react-router-dom'
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Spinner,
  Badge,
  Avatar,
  AvatarContent
} from '@base-ui/react'
import { useStore } from './store'
import { api } from './utils/api'
import { useGlobalShortcuts } from './utils/keyboard'
import { ThemeProvider, ThemeToggle } from './utils/ThemeProvider'
import ErrorBoundary from './components/ErrorBoundary'
import Home from './components/Home'
import Chat from './components/Chat'
import Datasets from './components/Datasets'
import Models from './components/Models'
import Training from './components/Training'
import Monitoring from './components/Monitoring'
import Settings from './components/Settings'
import ApiDocs from './components/ApiDocs'
import SearchModal, { SearchButton } from './components/Search'

interface NavItem {
  path: string
  label: string
  icon: string
  shortcut?: string
}

const navItems: NavItem[] = [
  { path: '/', label: 'Home', icon: '🏠', shortcut: '1' },
  { path: '/chat', label: 'Chat', icon: '💬', shortcut: '2' },
  { path: '/datasets', label: 'Datasets', icon: '📊', shortcut: '3' },
  { path: '/models', label: 'Models', icon: '🤖', shortcut: '4' },
  { path: '/training', label: 'Training', icon: '🧠', shortcut: '5' },
  { path: '/monitoring', label: 'Monitoring', icon: '📈', shortcut: '6' }
]

const Sidebar: React.FC<{ onSearchClick: () => void }> = ({ onSearchClick }) => {
  const location = useLocation()
  
  useGlobalShortcuts()
  
  return (
    <aside className="w-64 bg-slate-900 text-white h-screen fixed left-0 top-0 flex flex-col">
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center gap-3">
          <Avatar className="h-10 w-10 bg-blue-500">
            <AvatarContent className="text-lg">🦁</AvatarContent>
          </Avatar>
          <div>
            <h1 className="font-bold text-lg">SloughGPT</h1>
            <p className="text-xs text-slate-400">Enterprise AI</p>
          </div>
        </div>
      </div>
      
      <nav className="flex-1 p-4 overflow-y-auto">
        <ul className="space-y-1">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path
            return (
              <li key={item.path}>
                <Link
                  to={item.path}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  }`}
                  title={item.shortcut ? `${item.label} (${item.shortcut})` : item.label}
                >
                  <span className="text-xl">{item.icon}</span>
                  <span className="font-medium flex-1">{item.label}</span>
                  {item.shortcut && (
                    <kbd className="text-xs bg-slate-700 px-1.5 py-0.5 rounded">
                      {item.shortcut}
                    </kbd>
                  )}
                </Link>
              </li>
            )
          })}
        </ul>
        
        <div className="mt-6 pt-6 border-t border-slate-700 space-y-1">
          <button
            onClick={onSearchClick}
            className="flex items-center gap-3 px-4 py-3 rounded-lg text-slate-300 hover:bg-slate-800 hover:text-white transition-colors w-full"
          >
            <span className="text-xl">🔍</span>
            <span className="font-medium flex-1">Search</span>
            <kbd className="text-xs bg-slate-700 px-1.5 py-0.5 rounded">⌘K</kbd>
          </button>
          
          <Link
            to="/settings"
            className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
              location.pathname === '/settings'
                ? 'bg-blue-600 text-white'
                : 'text-slate-300 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <span className="text-xl">⚙️</span>
            <span className="font-medium">Settings</span>
          </Link>
          
          <Link
            to="/api-docs"
            className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
              location.pathname === '/api-docs'
                ? 'bg-blue-600 text-white'
                : 'text-slate-300 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <span className="text-xl">📚</span>
            <span className="font-medium">API Docs</span>
          </Link>
        </div>
      </nav>
      
      <div className="p-4 border-t border-slate-700">
        <div className="flex items-center justify-between text-sm text-slate-400">
          <span>v2.0.0</span>
          <Badge variant="secondary" size="sm">Online</Badge>
        </div>
        <p className="text-xs text-slate-500 mt-1">
          Press ? for shortcuts
        </p>
      </div>
    </aside>
  )
}

const Header: React.FC<{ onSearchClick: () => void }> = ({ onSearchClick }) => {
  const { activeTab, isLoading, error, clearError, isConnected, setConnected } = useStore()
  const navigate = useNavigate()
  
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await api.healthCheck()
        setConnected(!response.error)
      } catch {
        setConnected(false)
      }
    }
    
    checkConnection()
    const interval = setInterval(checkConnection, 30000)
    return () => clearInterval(interval)
  }, [setConnected])
  
  return (
    <header className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 h-16 px-6 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <h2 className="text-xl font-semibold text-slate-800 dark:text-white">
          {navItems.find(item => item.path === activeTab)?.label || 'Dashboard'}
        </h2>
        
        {isLoading && (
          <Spinner className="h-5 w-5 text-blue-500" size="5" />
        )}
      </div>
      
      <div className="flex items-center gap-4">
        <SearchButton onClick={onSearchClick} />
        
        <ThemeToggle />
        
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm text-slate-500 dark:text-slate-400">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        
        {error && (
          <div className="flex items-center gap-2 px-3 py-1.5 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg">
            <span className="text-sm">{error}</span>
            <button onClick={clearError} className="hover:text-red-800">
              ✕
            </button>
          </div>
        )}
        
        <Button 
          variant="outline" 
          size="sm"
          onClick={() => navigate('/settings')}
        >
          ⚙️
        </Button>
        
        <Avatar className="h-8 w-8 bg-slate-600">
          <AvatarContent className="text-sm">👤</AvatarContent>
        </Avatar>
      </div>
    </header>
  )
}

const AppContent: React.FC = () => {
  const [isSearchOpen, setIsSearchOpen] = useState(false)
  const { setModels, setDatasets, setConnected } = useStore()
  
  useEffect(() => {
    // Keyboard shortcut for search
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setIsSearchOpen(true)
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])
  
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const modelsRes = await api.getModels()
        if (modelsRes.data) {
          setModels(modelsRes.data.models)
        }
        
        const datasetsRes = await api.listDatasets()
        if (datasetsRes.data) {
          setDatasets(datasetsRes.data.datasets)
        }
        
        const healthRes = await api.healthCheck()
        setConnected(!healthRes.error)
      } catch (error) {
        console.error('Error loading initial data:', error)
        setConnected(false)
      }
    }
    
    loadInitialData()
  }, [setModels, setDatasets, setConnected])
  
  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
      <Sidebar onSearchClick={() => setIsSearchOpen(true)} />
      
      <div className="ml-64">
        <Header onSearchClick={() => setIsSearchOpen(true)} />
        
        <main className="p-6">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/models" element={<Models />} />
            <Route path="/training" element={<Training />} />
            <Route path="/monitoring" element={<Monitoring />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/api-docs" element={<ApiDocs />} />
          </Routes>
        </main>
      </div>
      
      <SearchModal isOpen={isSearchOpen} onClose={() => setIsSearchOpen(false)} />
    </div>
  )
}

const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <BrowserRouter>
          <AppContent />
        </BrowserRouter>
      </ThemeProvider>
    </ErrorBoundary>
  )
}

export default App
