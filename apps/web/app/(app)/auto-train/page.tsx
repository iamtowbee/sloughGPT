'use client'

import { useState, useRef, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Select } from '@/components/ui/select'
import { inferenceHealthLabel, useApiHealth } from '@/hooks/useApiHealth'
import { 
  TrainingMessages, 
  TrainingMessage, 
  TrainingHeader, 
  TrainingConfig,
  PersonalityEvolutionChart,
  PersonalityDataPoint
} from '@/components/chat/index-training'

export default function AutoTrainPage() {
  const [running, setRunning] = useState(false)
  const [messages, setMessages] = useState<TrainingMessage[]>([])
  const [teacherModel, setTeacherModel] = useState('gpt2')
  const [temperature, setTemperature] = useState('0.8')
  const [learningRate, setLearningRate] = useState('0.01')
  const [modelPath, setModelPath] = useState('models/auto-training/baby.pt')
  const [stepCount, setStepCount] = useState(0)
  const [currentLoss, setCurrentLoss] = useState<number | null>(null)
  const [lossHistory, setLossHistory] = useState<number[]>([])
  const [startTime, setStartTime] = useState<number | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [stepsPerSec, setStepsPerSec] = useState(0)
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected')
  const [lastSaveTime, setLastSaveTime] = useState<number | null>(null)
  const [showLog, setShowLog] = useState(false)
  const [trainingLog, setTrainingLog] = useState<string[]>([])
  const [epoch, setEpoch] = useState(0)
  const { state } = useApiHealth()
  const chatEndRef = useRef<HTMLDivElement>(null)
  const esRef = useRef<EventSource | null>(null)
  const messageIdRef = useRef(0)
  
  const [evalResult, setEvalResult] = useState<string | null>(null)
  const [personalityData, setPersonalityData] = useState<PersonalityDataPoint[]>([])
  
  const config: TrainingConfig = {
    teacherModel,
    temperature,
    learningRate,
    modelPath,
    isRunning: running,
    stepCount: messages.length,
    currentLoss: currentLoss,
    hasMessages: messages.length > 0,
  }

  // Timer effect
  useEffect(() => {
    if (!running || !startTime) {
      setElapsed(0)
      setStepsPerSec(0)
      return
    }
    const interval = setInterval(() => {
      const secs = Math.floor((Date.now() - startTime) / 1000)
      setElapsed(secs)
      if (secs > 0) {
        setStepsPerSec(stepCount / secs)
      }
    }, 1000)
    return () => clearInterval(interval)
  }, [running, startTime, stepCount])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && !running) {
        e.preventDefault()
        startAutoTrain()
      }
      if (e.key === 'Escape' && running) {
        stopAutoTrain()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [running])
  
  const apiStatus = inferenceHealthLabel(state)

  const startAutoTrain = async () => {
    console.log('Starting auto-train...')
    setRunning(true)
    setMessages([])
    setStepCount(0)
    setCurrentLoss(null)
    setLossHistory([])
    setPersonalityData([])
    setStartTime(Date.now())
    setElapsed(0)
    setStepsPerSec(0)
    setConnectionStatus('connecting')
    setEvalResult(null)

    try {
      const response = await fetch('http://localhost:8000/auto-train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          teacher_model: teacherModel,
          temperature: parseFloat(temperature),
          learning_rate: parseFloat(learningRate),
          baby_model_path: modelPath,
        }),
      })

      console.log('Start response:', response.status)
      if (!response.ok) {
        setRunning(false)
        return
      }

      const es = new EventSource('http://localhost:8000/auto-train/stream')
      esRef.current = es
      setConnectionStatus('connected')

      es.onopen = () => {
        setConnectionStatus('connected')
      }

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          
          if (data.error) {
            console.error('Server error:', data.error)
            es.close()
            setRunning(false)
            return
          }
          
          if (data.teacher) {
            setMessages(prev => [...prev, {
              id: `msg-${++messageIdRef.current}`,
              role: 'teacher',
              content: data.teacher,
              step: data.step,
            }])
          }
          if (data.student || data.baby_response) {
            setMessages(prev => [...prev, {
              id: `msg-${++messageIdRef.current}`,
              role: 'baby',
              content: data.student || data.baby_response,
              step: data.step,
            }])
          }
          if (data.teacher_feedback || data.correction) {
            setMessages(prev => [...prev, {
              id: `msg-${++messageIdRef.current}`,
              role: 'correction',
              content: data.teacher_feedback || data.correction,
              step: data.step,
            }])
          }
          if (data.training_done) {
            const turnNum = Math.floor((data.step - 1) / 3) + 1
            setStepCount(turnNum)
            setCurrentLoss(data.loss)
            if (data.loss !== null) {
              setLossHistory(prev => [...prev.slice(-49), data.loss])
            }
            if (data.personality) {
              setPersonalityData(prev => [...prev, {
                step: data.step,
                formality: data.personality.formality,
                detail_level: data.personality.detail_level,
                certainty: data.personality.certainty,
              }])
            }
          }
          if (data.done) {
            es.close()
            setRunning(false)
          }
          if (data.checkpoint_saved) {
            setLastSaveTime(Date.now())
            setTrainingLog(prev => [...prev, `Step ${data.step}: Checkpoint saved`])
          }
          
          // Auto-scroll
          setTimeout(() => {
            chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
          }, 50)
        } catch (e) {
          console.error('Parse error:', e)
        }
      }

      es.onerror = (err) => {
        console.log('SSE connection closed')
        setConnectionStatus('disconnected')
      }
    } catch (e) {
      console.error('Failed to start:', e)
      setRunning(false)
    }
  }

  const stopAutoTrain = () => {
    if (esRef.current) {
      esRef.current.close()
    }
    fetch('http://localhost:8000/auto-train/stop', { method: 'POST' })
    setRunning(false)
    // Trigger evaluation after training stops
    setTimeout(async () => {
      try {
        const res = await fetch('http://localhost:8000/auto-train/eval?prompt=Hello', { method: 'POST' })
        const data = await res.json()
        if (data.baby_output) {
          setEvalResult(data.baby_output)
        }
      } catch (e) {
        console.error('Eval failed:', e)
      }
    }, 1000)
  }

  return (
    <div className="sl-page mx-auto w-full max-w-4xl px-3 sm:px-4 py-3 sm:py-4">
      <TrainingHeader 
        config={config}
        onStart={startAutoTrain}
        onStop={stopAutoTrain}
        onReset={() => {
          if (esRef.current) esRef.current.close()
          setRunning(false)
          setMessages([])
          setStepCount(0)
          setCurrentLoss(null)
          setLossHistory([])
        }}
      />

      {/* Live Training Stats - show when running */}
      {running && (
        <div className="mt-3 sm:mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3">
          <Card className="py-2 sm:py-3">
            <div className="text-center">
              <div className="text-[10px] sm:text-xs text-muted-foreground">Time</div>
              <div className="text-lg sm:text-xl font-bold text-purple-600">
                {elapsed > 0 ? `${Math.floor(elapsed / 60)}:${(elapsed % 60).toString().padStart(2, '0')}` : '0:00'}
              </div>
            </div>
          </Card>
          <Card className="py-2 sm:py-3">
            <div className="text-center">
              <div className="text-[10px] sm:text-xs text-muted-foreground">Turns</div>
              <div className="text-lg sm:text-xl font-bold text-blue-600">{stepCount}</div>
              <div className="text-[10px] sm:text-xs text-muted-foreground">{stepsPerSec.toFixed(2)}/s</div>
            </div>
          </Card>
          <Card className="py-2 sm:py-3">
            <div className="text-center">
              <div className="text-[10px] sm:text-xs text-muted-foreground">Loss</div>
              <div className="text-lg sm:text-xl font-bold text-orange-600">
                {currentLoss !== null ? currentLoss.toFixed(2) : '--'}
              </div>
              {lossHistory.length > 1 ? (
                <div className="text-[10px] sm:text-xs text-muted-foreground">
                  {lossHistory[lossHistory.length - 1] < lossHistory[0] ? '↓' : '↑'}
                  {Math.abs((lossHistory[lossHistory.length - 1] - lossHistory[0]) / lossHistory[0] * 100).toFixed(0)}%
                </div>
              ) : null}
            </div>
          </Card>
          <Card className="py-2 sm:py-3">
            <div className="text-center">
              <div className="text-[10px] sm:text-xs text-muted-foreground">Msgs</div>
              <div className="text-lg sm:text-xl font-bold text-green-600">{messages.length}</div>
            </div>
          </Card>
        </div>
      )}

      {/* Personality Evolution Chart */}
      {personalityData.length > 0 && (
        <div className="mt-3 sm:mt-4">
          <PersonalityEvolutionChart data={personalityData} />
        </div>
      )}

      <Card className="mt-3 sm:mt-4">
        <CardHeader className="pb-1 sm:pb-2">
          <div className="flex justify-between items-center">
            <CardTitle className="text-sm sm:text-base flex items-center gap-2">
              <span>Training Flow</span>
              {running && (
                <div className="flex items-center gap-1.5">
                  <span className="relative flex h-1.5 w-1.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-green-500"></span>
                  </span>
                  <span className="text-green-500 text-xs font-medium">Active</span>
                </div>
              )}
            </CardTitle>
            <span className="text-[10px] sm:text-xs text-muted-foreground">
              {messages.length > 0 ? `${Math.floor(messages.length / 3)} turns` : 'No turns yet'}
            </span>
          </div>
          {/* Progress bar */}
          {running && (
            <div className="mt-1.5 h-1 bg-muted rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-blue-500 to-green-500 transition-all duration-500"
                style={{ width: `${Math.min((stepCount % 20) * 5, 100)}%` }}
              />
            </div>
          )}
        </CardHeader>
<CardContent className="max-h-[300px] sm:max-h-[400px] overflow-y-auto p-0">
          {messages.length === 0 ? (
            <div className="text-center py-6 sm:py-10 space-y-2">
              <div className="text-sm sm:text-base font-medium">Ready to Train</div>
              <div className="text-xs text-muted-foreground max-w-[200px] sm:max-w-xs mx-auto">
                Teacher asksBaby respondsTeacher corrects
              </div>
              <div className="text-[10px] sm:text-xs text-muted-foreground mt-2 sm:mt-3">
                Press <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">Ctrl</kbd> + <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">Enter</kbd> to start
              </div>
            </div>
          ) : (
            <TrainingMessages messages={messages} />
          )}
          <div ref={chatEndRef} />
        </CardContent>
      </Card>

      {/* Training complete summary */}
      {!running && messages.length > 0 && (
        <div className="mt-3 sm:mt-4 p-3 sm:p-4 bg-muted rounded-lg">
          <div className="flex justify-between items-center">
            <div>
              <div className="text-sm font-medium">Training Complete</div>
              <div className="text-[10px] sm:text-xs text-muted-foreground">
                {Math.floor(messages.length / 3)} turns • {messages.length} msgs
              </div>
            </div>
            {currentLoss !== null && (
              <div className="text-right">
                <div className="text-[10px] sm:text-xs text-muted-foreground">Final Loss</div>
                <div className="text-base sm:text-lg font-bold text-orange-600">{currentLoss.toFixed(2)}</div>
              </div>
            )}
          </div>
          {evalResult && (
            <div className="mt-2 pt-2 border-t">
              <div className="text-[10px] sm:text-xs text-muted-foreground">Baby Model Output</div>
              <div className="text-xs sm:text-sm font-mono bg-background p-1.5 sm:p-2 rounded mt-1 break-all">{evalResult}</div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}