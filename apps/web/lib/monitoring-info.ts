/** Map GET /info JSON to Monitoring UI state (real host + CUDA fields from API). */

export interface SystemInfo {
  platform: string
  python: string
  cpu_cores: number
  cpu_percent: number | null
  memory_total: number
  memory_used: number
  /** Host RAM; from used/total bytes when available so % matches the GB line. */
  memory_percent: number | null
  gpu_available: boolean
  gpu_name?: string
  gpu_memory?: number
  gpu_used?: number
  gpu_percent: number | null
  /** API process resident set size (bytes), when psutil reports it. */
  process_rss_bytes?: number | null
  /** True when API returned a ``host`` block (psutil-backed). */
  host_metrics_available: boolean
}

type InfoJson = {
  pytorch_version?: string
  cuda_available?: boolean
  host?: {
    platform?: string
    platform_release?: string
    cpu_count_logical?: number
    cpu_percent?: number
    memory_total_bytes?: number
    memory_used_bytes?: number
    memory_percent?: number
    process_rss_bytes?: number | null
  }
  cuda?: {
    device?: string
    memory_total?: number
    memory_total_bytes?: number
    memory_used_bytes?: number
    memory_percent?: number
  }
}

/** Match the same bytes we show as GB — psutil `memory_percent` can disagree with `used`/`total`. */
function memoryPercentFromBytes(used: number, total: number): number | null {
  if (!(total > 0) || used < 0) return null
  return Math.min(100, Math.max(0, (used / total) * 100))
}

export function mapInfoToSystemInfo(data: InfoJson): SystemInfo {
  const pytorch = typeof data.pytorch_version === 'string' ? data.pytorch_version : 'N/A'
  const host = data.host
  const cuda = data.cuda

  let gpuMemory: number | undefined
  let gpuUsed: number | undefined
  let gpuPercent: number | null = null

  if (cuda) {
    if (typeof cuda.memory_total_bytes === 'number') {
      gpuMemory = cuda.memory_total_bytes
    } else if (typeof cuda.memory_total === 'number') {
      gpuMemory = Math.round(cuda.memory_total * 1e9)
    }
    if (typeof cuda.memory_used_bytes === 'number') {
      gpuUsed = cuda.memory_used_bytes
    }
    if (typeof cuda.memory_percent === 'number') {
      gpuPercent = cuda.memory_percent
    }
  }

  if (host && typeof host.cpu_percent === 'number') {
    const rss =
      typeof host.process_rss_bytes === 'number' && host.process_rss_bytes >= 0
        ? host.process_rss_bytes
        : null
    const memTotal = typeof host.memory_total_bytes === 'number' ? host.memory_total_bytes : 0
    const memUsed = typeof host.memory_used_bytes === 'number' ? host.memory_used_bytes : 0
    const memPctFromBytes = memoryPercentFromBytes(memUsed, memTotal)
    return {
      platform: [host.platform, host.platform_release].filter(Boolean).join(' '),
      python: pytorch,
      cpu_cores: typeof host.cpu_count_logical === 'number' ? host.cpu_count_logical : 0,
      cpu_percent: host.cpu_percent,
      memory_total: memTotal,
      memory_used: memUsed,
      memory_percent:
        memPctFromBytes != null
          ? memPctFromBytes
          : typeof host.memory_percent === 'number'
            ? host.memory_percent
            : null,
      gpu_available: Boolean(data.cuda_available),
      gpu_name: typeof cuda?.device === 'string' ? cuda.device : undefined,
      gpu_memory: gpuMemory,
      gpu_used: gpuUsed,
      gpu_percent: gpuPercent,
      process_rss_bytes: rss,
      host_metrics_available: true,
    }
  }

  return {
    platform: 'Unknown',
    python: pytorch,
    cpu_cores: typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 0 : 0,
    cpu_percent: null,
    memory_total: 0,
    memory_used: 0,
    memory_percent: null,
    gpu_available: Boolean(data.cuda_available),
    gpu_name: typeof cuda?.device === 'string' ? cuda.device : undefined,
    gpu_memory: gpuMemory,
    gpu_used: gpuUsed,
    gpu_percent: gpuPercent,
    process_rss_bytes: null,
    host_metrics_available: false,
  }
}
