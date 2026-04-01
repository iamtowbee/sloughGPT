'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

import { api } from '@/lib/api'
import { useAuthStore } from '@/lib/auth'

export default function LoginPage() {
  const router = useRouter()
  const login = useAuthStore((state) => state.login)
  const [isLogin, setIsLogin] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [form, setForm] = useState({
    username: '',
    email: '',
    password: '',
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      if (isLogin) {
        const res = await api.login(form.username, form.password)
        if (res.token) {
          login(res.user, res.token)
          router.push('/')
        } else {
          setError('Login failed')
        }
      } else {
        const res = await api.register(form.username, form.email, form.password)
        if (res.token) {
          login(res.user, res.token)
          router.push('/')
        } else {
          setError('Registration failed')
        }
      }
    } catch {
      setError('An error occurred. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center sl-shell-main p-6">
      <div className="w-full max-w-md sl-card-solid p-8 border border-border shadow-xl">
        <div className="text-center mb-8">
          <div className="h-14 w-14 rounded-xl bg-primary/20 ring-1 ring-primary/30 flex items-center justify-center text-primary text-lg font-semibold font-mono mx-auto mb-4">
            S
          </div>
          <h1 className="text-2xl font-semibold text-foreground tracking-tight">SloughGPT</h1>
          <p className="text-muted-foreground mt-2 text-sm">
            {isLogin ? 'Sign in to your account' : 'Create your account'}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="sl-label">Username</label>
            <input
              type="text"
              value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
              required
              className="sl-input"
            />
          </div>

          {!isLogin && (
            <div>
              <label className="sl-label">Email</label>
              <input
                type="email"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                required={!isLogin}
                className="sl-input"
              />
            </div>
          )}

          <div>
            <label className="sl-label">Password</label>
            <input
              type="password"
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              required
              className="sl-input"
            />
          </div>

          {error && <p className="text-destructive text-sm">{error}</p>}

          <button type="submit" disabled={loading} className="w-full sl-btn-primary py-2.5 rounded-lg">
            {loading ? 'Loading...' : isLogin ? 'Sign In' : 'Create Account'}
          </button>
        </form>

        <p className="text-center mt-6 text-muted-foreground text-sm">
          {isLogin ? "Don't have an account? " : 'Already have an account? '}
          <button
            type="button"
            onClick={() => setIsLogin(!isLogin)}
            className="text-primary font-medium hover:underline"
          >
            {isLogin ? 'Sign up' : 'Sign in'}
          </button>
        </p>

        <p className="text-center mt-4 text-muted-foreground text-xs font-mono">
          Default admin: <span className="text-foreground">admin</span> / <span className="text-foreground">admin123</span>
        </p>
      </div>
    </div>
  )
}
