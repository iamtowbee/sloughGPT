import React, { useState, useEffect } from 'react'
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  Spinner
} from '@base-ui/react'
import { useStore, Model } from '../store'
import { api } from '../utils/api'

interface ModelsProps {}

export const Models: React.FC<ModelsProps> = () => {
  const [models, setModels] = useState<Model[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedProvider, setSelectedProvider] = useState<string>('all')
  
  const { models: storeModels, setModels: updateStoreModels } = useStore()

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    setIsLoading(true)
    try {
      const response = await api.getModels()
      if (response.data) {
        setModels(response.data.models)
        updateStoreModels(response.data.models)
      }
    } catch (error) {
      console.error('Error loading models:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleModelInfo = (model: Model) => {
    const pricing = model.pricing ? 
      `\nPricing:\n  Prompt: $${model.pricing.prompt}/1K tokens\n  Completion: $${model.pricing.completion}/1K tokens` : 
      ''

    alert(`Model: ${model.name}

Provider: ${model.provider}
Status: ${model.status}
Context Length: ${model.context_length?.toLocaleString() || 'N/A'} tokens${pricing}

Description:
${model.description || 'No description available'}`)
  }

  const handleModelTest = async (modelId: string) => {
    const testPrompt = 'Hello, how are you?'
    
    try {
      const response = await api.generate(
        testPrompt,
        modelId,
        50,
        0.7
      )
      
      if (response.data) {
        alert(`Model: ${modelId}\n\nPrompt: ${testPrompt}\n\nGenerated: ${response.data.text}\nTokens: ${response.data.tokens_generated}\nTime: ${response.data.processing_time_ms.toFixed(2)}ms`)
      }
    } catch (error) {
      console.error('Error testing model:', error)
      alert('Error testing model')
    }
  }

  const providers = ['all', ...new Set(models.map(m => m.provider))]

  const filteredModels = selectedProvider === 'all' 
    ? models 
    : models.filter(m => m.provider === selectedProvider)

  const getProviderColor = (provider: string) => {
    const colors: Record<string, string> = {
      'OpenAI': 'bg-green-100 text-green-800',
      'Anthropic': 'bg-purple-100 text-purple-800',
      'Meta': 'bg-blue-100 text-blue-800',
      'Mistral': 'bg-indigo-100 text-indigo-800',
      'Local': 'bg-orange-100 text-orange-800',
      'BigScience': 'bg-pink-100 text-pink-800',
      'Stability AI': 'bg-red-100 text-red-800'
    }
    return colors[provider] || 'bg-gray-100 text-gray-800'
  }

  const renderModelCard = (model: Model) => (
    <Card key={model.id} className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <span className="font-medium">{model.name}</span>
          <div className="flex gap-1">
            <span className={`px-2 py-0.5 rounded-full text-xs ${getProviderColor(model.provider)}`}>
              {model.provider}
            </span>
            <span className={`px-2 py-0.5 rounded-full text-xs ${
              model.status === 'available' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              {model.status}
            </span>
          </div>
        </CardTitle>
        <CardDescription className="text-sm">
          {model.description || 'No description'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between text-sm text-slate-500 mb-3">
          <span>Context: {model.context_length?.toLocaleString() || 'N/A'} tokens</span>
        </div>
        
        {model.pricing && (
          <div className="text-xs text-slate-500 mb-3">
            <span className="font-medium">Pricing:</span> ${model.pricing.prompt}/1K prompt + ${model.pricing.completion}/1K completion
          </div>
        )}
        
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleModelInfo(model)}
            className="flex-1"
          >
            ℹ️ Info
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleModelTest(model.id)}
            className="flex-1"
            disabled={model.status !== 'available'}
          >
            🚀 Test
          </Button>
        </div>
      </CardContent>
    </Card>
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="text-xl font-semibold">Models</span>
            <span className="text-sm text-slate-500">
              {models.length} models available
            </span>
          </CardTitle>
          <CardDescription>
            Browse and manage available AI models
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Provider Filter */}
      <div className="flex gap-2 flex-wrap">
        {providers.map(provider => (
          <Button
            key={provider}
            variant={selectedProvider === provider ? 'primary' : 'outline'}
            size="sm"
            onClick={() => setSelectedProvider(provider)}
          >
            {provider === 'all' ? 'All' : provider}
          </Button>
        ))}
      </div>

      {/* Model List */}
      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <Spinner className="h-8 w-8 text-blue-500" size="8" />
        </div>
      ) : filteredModels.length === 0 ? (
        <Card className="h-64 flex items-center justify-center text-slate-400">
          <div className="text-center">
            <div className="text-4xl mb-4">🤖</div>
            <p>No models found</p>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredModels.map(renderModelCard)}
        </div>
      )}
    </div>
  )
}

export default Models
