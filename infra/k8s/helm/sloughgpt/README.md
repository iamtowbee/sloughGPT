# SloughGPT Helm Chart

A Helm chart for deploying SloughGPT - Enterprise AI Framework with Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- NVIDIA GPU operator (for GPU support)
- Ingress controller (e.g., nginx-ingress)

## Installing the Chart

```bash
# Add the repository
helm repo add sloughgpt https://iamtowbee.github.io/sloughGPT
helm repo update

# Install the chart
helm install sloughgpt sloughgpt/sloughgpt --namespace sloughgpt --create-namespace
```

If you have the **monorepo cloned** and prefer not to add a Helm repo, install from the chart directory (from repo root):

```bash
helm install sloughgpt ./infra/k8s/helm/sloughgpt/ -n sloughgpt --create-namespace
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of API replicas | `2` |
| `image.repository` | Docker image repository | `ghcr.io/iamtowbee/sloughgpt` |
| `image.tag` | Docker image tag | `latest` |
| `service.type` | Service type | `ClusterIP` |
| `ingress.enabled` | Enable ingress | `true` |
| `autoscaling.enabled` | Enable HPA | `true` |
| `autoscaling.minReplicas` | Minimum replicas | `2` |
| `autoscaling.maxReplicas` | Maximum replicas | `10` |
| `persistence.enabled` | Enable PVC | `true` |
| `persistence.size` | PVC size | `10Gi` |
| `config.device` | Device (cuda/cpu) | `cuda` |
| `config.maxBatchSize` | Max batch size | `32` |
| `monitoring.enabled` | Enable monitoring | `true` |

## Quick Start

```bash
# Install with custom values
helm install sloughgpt sloughgpt/sloughgpt \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=sloughgpt.example.com \
  --set config.device=cuda

# Check status
kubectl get pods -n sloughgpt

# View logs
kubectl logs -n sloughgpt -l app.kubernetes.io/component=api
```

## Production Usage

From the **repository root** (chart path is `infra/k8s/helm/sloughgpt`):

```bash
# Create namespace
kubectl create namespace sloughgpt

# Install with values file
helm install sloughgpt ./infra/k8s/helm/sloughgpt/ -n sloughgpt -f values-production.yaml

# Upgrade
helm upgrade sloughgpt ./infra/k8s/helm/sloughgpt/ -n sloughgpt
```

## GPU Support

For GPU support, ensure the NVIDIA GPU operator is installed:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/master/statefulsets/nvidia-gpu-operator.yaml
```

Then configure the chart:

```yaml
# values.yaml
resources:
  api:
    limits:
      nvidia.com/gpu: 1
  model:
    limits:
      nvidia.com/gpu: 1
```

## Monitoring

The chart includes Prometheus monitoring by default:

- ServiceMonitor for API metrics
- PodMonitor for detailed metrics
- Alert rules for high latency, error rates, GPU memory

Install Prometheus Operator first:

```bash
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml
```

## Uninstall

```bash
helm uninstall sloughgpt -n sloughgpt
kubectl delete namespace sloughgpt
```
