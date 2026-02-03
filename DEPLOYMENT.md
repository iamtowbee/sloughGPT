# SloughGPT Production Deployment Guide

## 🚀 Production Deployment Instructions

This guide covers complete production deployment of SloughGPT with monitoring, scaling, and high availability.

## 📋 Prerequisites

### Required Tools
- **Docker** - Container runtime
- **Kubernetes** - Container orchestration
- **Helm** - Package manager (optional)
- **kubectl** - Kubernetes CLI
- **Domain name** - For SSL and ingress

### Cloud Providers
- **AWS EKS** - Elastic Kubernetes Service
- **Google GKE** - Google Kubernetes Engine  
- **Azure AKS** - Azure Kubernetes Service
- **Digital Ocean** - DigitalOcean Kubernetes
- **Local cluster** - Minikube, k3s, or Docker Desktop

## 🔧 Quick Start

### 1. Clone and Configure
```bash
git clone https://github.com/your-org/sloughgpt.git
cd sloughgpt

# Copy configuration template
cp k8s/config.yaml.template k8s/config.yaml
# Edit configuration with your values
```

### 2. Deploy to Production
```bash
# Deploy all services
./deploy-production.sh production latest us-west-2 sloughgpt

# Or with custom registry
REGISTRY=your-registry.com ./deploy-production.sh production latest us-west-2
```

### 3. Verify Deployment
```bash
# Check all pods
kubectl get pods -n sloughgpt

# Check services
kubectl get services -n sloughgpt

# Check ingress
kubectl get ingress -n sloughgpt
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 Internet / Users                │
└─────────────────────┬───────────────────────────┘
                    │ Ingress Controller (NGINX)
                    │ SSL Termination
                    │ Load Balancing
┌───────────────────┴───────────────────────────┐
│           Kubernetes Cluster                │
│                                          │
│ ┌─────────────┬─────────────┬─────────────┐ │
│ │   API Pods  │ Database    │ Monitoring  │ │
│ │ (x3 replicas)│  (Postgres) │ (Prometheus) │ │
│ │             │             │   + Grafana │ │
│ └─────────────┴─────────────┴─────────────┘ │
│                                          │
│ ┌─────────────┬─────────────┬─────────────┐ │
│ │   Cache     │ Learning    │ Training    │ │
│ │  (Redis)    │   Service   │   Jobs      │ │
│ └─────────────┴─────────────┴─────────────┘ │
└─────────────────────────────────────────────────┘
```

## 📊 Components

### Core Services
- **sloughgpt-api** - Main API server (3+ replicas)
- **postgres** - PostgreSQL database (1 replica)
- **redis** - Redis cache (1 replica)

### Monitoring Stack
- **prometheus** - Metrics collection
- **grafana** - Visualization dashboard
- **nginx-ingress** - Load balancing and SSL

### Optional Services
- **sloughgpt-learning** - Data learning pipeline
- **sloughgpt-training** - Model training jobs

## 🔐 Security Configuration

### SSL/TLS
- Automatic SSL with cert-manager
- Let's Encrypt certificates
- HTTPS only in production

### Secrets Management
- Kubernetes secrets for sensitive data
- Environment variable injection
- RBAC for access control

### Network Security
- Network policies enabled
- Ingress/egress controls
- Web Application Firewall (WAF)

## 📈 Monitoring & Alerting

### Prometheus Metrics
```bash
# Key endpoints
http://api.sloughgpt.com/metrics
http://prometheus.sloughgpt.com:9090
```

### Grafana Dashboards
- **API Performance** - Response times, error rates
- **Resource Usage** - CPU, memory, storage
- **Cost Tracking** - Token usage, expenses
- **User Analytics** - Active users, requests

### Alerting Rules
- High error rate (>5%)
- High response time (>2s)
- Resource thresholds (>80%)
- Cost budget alerts

## 📈 Auto-Scaling

### Horizontal Pod Autoscaler
```yaml
minReplicas: 3
maxReplicas: 20
targetCPUUtilization: 70%
targetMemoryUtilization: 80%
```

### Scaling Triggers
- CPU utilization >70%
- Memory utilization >80%
- Request queue length
- Custom metrics

## 🔧 Configuration Management

### ConfigMaps
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sloughgpt-config
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  DATABASE_URL: "postgresql://..."
  REDIS_URL: "redis://..."
  MODEL_CACHE_SIZE: "1000"
  MAX_CONCURRENT_REQUESTS: "100"
```

### Secrets
```bash
# Create secrets
kubectl create secret generic sloughgpt-secrets \
  --from-literal=DATABASE_PASSWORD="your-password" \
  --from-literal=JWT_SECRET="your-jwt-secret" \
  --from-literal=API_SECRET="your-api-secret"
```

## 🗂️ Data Management

### Database Backups
```bash
# Daily automated backups
kubectl create cronjob database-backup \
  --schedule="0 2 * * *" \
  --image=postgres:15 \
  --dry-run=client -o yaml | kubectl apply -f -
```

### Persistent Storage
- **PostgreSQL**: 20GB SSD storage
- **Redis**: 5GB SSD storage  
- **Prometheus**: 10GB SSD storage
- **Grafana**: 5GB SSD storage

## 🚀 Performance Optimization

### Caching Strategy
- **Redis** for session data
- **Model responses** caching
- **Database query** caching
- **Static assets** CDN

### Database Optimization
- Connection pooling
- Read replicas (optional)
- Query optimization
- Index management

## 🔍 Troubleshooting

### Common Issues

#### Pod Not Starting
```bash
# Check pod events
kubectl describe pod -n sloughgpt <pod-name>

# Check logs
kubectl logs -f -n sloughgpt deployment/sloughgpt-api

# Check resource limits
kubectl top pods -n sloughgpt
```

#### Database Connection Issues
```bash
# Check database pod
kubectl get pods -n sloughgpt -l app=postgres

# Test database connection
kubectl exec -it -n sloughgpt deployment/postgres -- psql -U sloughgpt -d sloughgpt

# Check database logs
kubectl logs -f -n sloughgpt deployment/postgres
```

#### High Resource Usage
```bash
# Check resource usage
kubectl top pods -n sloughgpt
kubectl top nodes

# Check HPA status
kubectl get hpa -n sloughgpt

# Check resource limits
kubectl describe pod -n sloughgpt <pod-name>
```

### Health Checks
```bash
# API health
curl https://api.sloughgpt.com/health

# Readiness probe
curl https://api.sloughgpt.com/ready

# Database health
kubectl exec -n sloughgpt deployment/postgres -- pg_isready

# Redis health
kubectl exec -n sloughgpt deployment/redis -- redis-cli ping
```

## 🔄 CI/CD Integration

### GitHub Actions Workflow
```yaml
name: Deploy to Production
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to K8s
        run: |
          ./deploy-production.sh ${{ github.sha }}
        env:
          KUBECONFIG: ${{ secrets.KUBECONFIG }}
          REGISTRY: your-registry.com
```

### Rollback Strategy
```bash
# View deployment history
kubectl rollout history deployment/sloughgpt-api -n sloughgpt

# Rollback to previous version
kubectl rollout undo deployment/sloughgpt-api -n sloughgpt

# Rollback to specific version
kubectl rollout undo deployment/sloughgpt-api --to-revision=2 -n sloughgpt
```

## 📋 Maintenance

### Updates and Patches
```bash
# Update to new version
kubectl set image deployment/sloughgpt-api \
  sloughgpt-api=your-registry.com/sloughgpt/api:v1.2.0 \
  -n sloughgpt

# Rolling restart
kubectl rollout restart deployment/sloughgpt-api -n sloughgpt

# Zero-downtime updates
kubectl rollout status deployment/sloughgpt-api -n sloughgpt
```

### Scaling Operations
```bash
# Manual scaling
kubectl scale deployment sloughgpt-api --replicas=10 -n sloughgpt

# Auto-scaling status
kubectl get hpa -n sloughgpt

# Event-driven scaling
kubectl autoscale deployment sloughgpt-api \
  --cpu-percent=70 \
  --min=3 \
  --max=50 \
  -n sloughgpt
```

## 🌐 Domain Configuration

### DNS Records
```
A Record: api.sloughgpt.com -> Load Balancer IP
A Record: prometheus.sloughgpt.com -> Load Balancer IP  
A Record: grafana.sloughgpt.com -> Load Balancer IP
CNAME: *.sloughgpt.com -> Load Balancer Hostname
```

### SSL Certificates
- Automatic with cert-manager
- Wildcard certificates: *.sloughgpt.com
- Auto-renewal enabled

## 📊 Cost Management

### Resource Planning
- **API Pods**: 3x 1CPU, 2GB RAM = $150/month
- **Database**: 1x 2CPU, 4GB RAM, 20GB SSD = $80/month  
- **Cache**: 1x 0.5CPU, 1GB RAM, 5GB SSD = $30/month
- **Monitoring**: 2x 1CPU, 2GB RAM, 15GB SSD = $60/month
- **Load Balancer**: Standard LB = $25/month
- **Total**: ~$345/month (plus data transfer)

### Cost Optimization
- Use spot instances for non-critical workloads
- Implement auto-scaling to reduce idle resources
- Optimize storage with lifecycle policies
- Monitor and right-size resources

## 🎯 Next Steps

1. **Configure monitoring alerts** for your organization
2. **Set up backup strategies** for data protection  
3. **Implement disaster recovery** procedures
4. **Configure log aggregation** with centralized systems
5. **Set up CI/CD pipelines** for automated deployments

## 🆘 Support

### Documentation
- [API Documentation](https://api.sloughgpt.com/docs)
- [Monitoring Guide](https://docs.sloughgpt.com/monitoring)
- [Troubleshooting Guide](https://docs.sloughgpt.com/troubleshooting)

### Community
- [GitHub Issues](https://github.com/your-org/sloughgpt/issues)
- [Discord Community](https://discord.gg/sloughgpt)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/sloughgpt)

---

**🎉 Your SloughGPT production deployment is now ready!**

This setup provides enterprise-grade reliability, scalability, and monitoring for your SloughGPT deployment.