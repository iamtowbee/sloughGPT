export class HttpError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown
  ) {
    super(message)
    this.name = 'HttpError'
  }
}

export interface RequestConfig {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH'
  headers?: Record<string, string>
  body?: unknown
  timeout?: number
  retries?: number
  retryDelay?: number
}

export interface RequestOptions extends RequestConfig {
  baseURL?: string
}

const DEFAULT_TIMEOUT = 30000
const DEFAULT_RETRIES = 3
const DEFAULT_RETRY_DELAY = 1000

class HttpClient {
  private baseURL: string
  private defaultHeaders: Record<string, string>

  constructor(baseURL: string = '', headers: Record<string, string> = {}) {
    this.baseURL = baseURL
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      ...headers,
    }
  }

  setAuthToken(token: string) {
    this.defaultHeaders['Authorization'] = `Bearer ${token}`
  }

  clearAuthToken() {
    delete this.defaultHeaders['Authorization']
  }

  private async withTimeout<T>(promise: Promise<T>, timeout: number): Promise<T> {
    const timeoutPromise = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`Request timeout after ${timeout}ms`)), timeout)
    )
    return Promise.race([promise, timeoutPromise])
  }

  private async withRetry<T>(
    fn: () => Promise<T>,
    retries: number,
    retryDelay: number
  ): Promise<T> {
    let lastError: Error
    
    for (let i = 0; i <= retries; i++) {
      try {
        return await fn()
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err))
        
        if (i < retries && this.isRetryableError(lastError)) {
          await this.sleep(retryDelay * (i + 1))
          continue
        }
        
        throw lastError
      }
    }
    
    throw lastError!
  }

  private isRetryableError(error: Error): boolean {
    if (error.message.includes('timeout')) return true
    if (error instanceof HttpError) {
      return error.status >= 500 || error.status === 429
    }
    return false
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  private buildUrl(endpoint: string, params?: Record<string, string>): string {
    const url = new URL(endpoint, this.baseURL)
    
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        url.searchParams.append(key, value)
      })
    }
    
    return url.toString()
  }

  async request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
    const {
      method = 'GET',
      headers = {},
      body,
      timeout = DEFAULT_TIMEOUT,
      retries = DEFAULT_RETRIES,
      retryDelay = DEFAULT_RETRY_DELAY,
      baseURL = this.baseURL,
    } = options

    const url = this.buildUrl(endpoint, undefined)
    const requestHeaders = { ...this.defaultHeaders, ...headers }

    const makeRequest = async (): Promise<T> => {
      const response = await this.withTimeout(
        fetch(url, {
          method,
          headers: requestHeaders,
          body: body ? JSON.stringify(body) : undefined,
        }),
        timeout
      )

      if (!response.ok) {
        let data: unknown
        try {
          data = await response.json()
        } catch {
          data = undefined
        }
        
        throw new HttpError(
          `HTTP ${response.status}: ${response.statusText}`,
          response.status,
          data
        )
      }

      const contentType = response.headers.get('content-type')
      if (contentType?.includes('application/json')) {
        return response.json() as Promise<T>
      }
      
      return response.text() as unknown as T
    }

    return this.withRetry(makeRequest, retries, retryDelay)
  }

  async get<T>(endpoint: string, params?: Record<string, string>, options?: RequestConfig): Promise<T> {
    const url = this.buildUrl(endpoint, params)
    return this.request<T>(url, { ...options, method: 'GET' })
  }

  async post<T>(endpoint: string, body?: unknown, options?: RequestConfig): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'POST', body })
  }

  async put<T>(endpoint: string, body?: unknown, options?: RequestConfig): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'PUT', body })
  }

  async delete<T>(endpoint: string, options?: RequestConfig): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'DELETE' })
  }

  async patch<T>(endpoint: string, body?: unknown, options?: RequestConfig): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'PATCH', body })
  }

  async upload<T>(
    endpoint: string,
    file: File | Blob,
    fieldName: string = 'file',
    options?: RequestConfig
  ): Promise<T> {
    const formData = new FormData()
    formData.append(fieldName, file)

    const response = await fetch(this.buildUrl(endpoint, undefined), {
      method: 'POST',
      headers: { ...this.defaultHeaders, ...options?.headers },
      body: formData,
    })

    if (!response.ok) {
      throw new HttpError(`Upload failed: ${response.status}`, response.status)
    }

    return response.json() as Promise<T>
  }
}

export function createHttpClient(baseURL: string, authToken?: string): HttpClient {
  const client = new HttpClient(baseURL)
  
  if (authToken) {
    client.setAuthToken(authToken)
  }
  
  return client
}

export { HttpClient as default }