#!/usr/bin/env python3
"""
Automated Deployment System for SloGPT
Containerization and cloud deployment automation.
"""

import os
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml


class DeploymentManager:
    """Manages automated deployment of SloGPT models."""
    
    def __init__(self, output_dir: str = "deployment"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.deployment_configs = {}
        self.deployment_logs = []
        
    def create_docker_configuration(self, model_path: str, model_name: str) -> Dict:
        """Create Docker configuration for model deployment."""
        print(f"🐳 Creating Docker configuration for {model_name}")
        
        config = {
            'version': '3.8',
            'services': {
                'slogpt-api': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile'
                    },
                    'ports': ['8000:8000'],
                    'environment': [
                        'MODEL_PATH=/app/models',
                        f'MODEL_NAME={model_name}',
                        'PYTHONPATH=/app'
                    ],
                    'volumes': [
                        f'{Path(model_path).parent}:/app/models'
                    ],
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    }
                }
            }
        }
        
        docker_compose_path = self.output_dir / f"docker-compose_{model_name}.yml"
        with open(docker_compose_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"   ✅ Docker Compose configuration saved: {docker_compose_path}")
        
        return {
            'docker_compose_path': str(docker_compose_path),
            'service_name': f'slogpt-{model_name}',
            'ports': ['8000:8000']
        }
    
    def create_dockerfile(self, model_name: str, requirements: List[str] = None) -> str:
        """Create optimized Dockerfile for SloGPT deployment."""
        
        if requirements is None:
            requirements = [
                'torch>=1.9.0',
                'fastapi>=0.68.0',
                'uvicorn>=0.15.0',
                'pydantic>=1.8.0',
                'numpy>=1.21.0',
                'transformers>=4.20.0'
            ]
        
        dockerfile_content = f"""# Optimized Dockerfile for SloGPT {model_name}
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "simple_api_server:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        # Save requirements.txt
        requirements_path = self.output_dir / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
        
        # Save Dockerfile
        dockerfile_path = self.output_dir / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        print(f"   ✅ Dockerfile created: {dockerfile_path}")
        print(f"   ✅ Requirements created: {requirements_path}")
        
        return str(dockerfile_path)
    
    def create_kubernetes_manifest(self, model_name: str, model_config: Dict) -> Dict:
        """Create Kubernetes deployment manifest."""
        print(f"☸️ Creating Kubernetes manifest for {model_name}")
        
        # Default resource requirements based on model size
        model_size = model_config.get('parameters', 1000000)  # Default assumption
        if model_size < 1000000:  # Small model
            cpu_request = '500m'
            cpu_limit = '1000m'
            memory_request = '1Gi'
            memory_limit = '2Gi'
        elif model_size < 10000000:  # Medium model
            cpu_request = '1000m'
            cpu_limit = '2000m'
            memory_request = '2Gi'
            memory_limit = '4Gi'
        else:  # Large model
            cpu_request = '2000m'
            cpu_limit = '4000m'
            memory_request = '4Gi'
            memory_limit = '8Gi'
        
        manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f'slogpt-{model_name}',
                'labels': {
                    'app': f'slogpt-{model_name}',
                    'version': 'v1.0.0'
                }
            },
            'spec': {
                'replicas': 2,
                'selector': {
                    'matchLabels': {
                        'app': f'slogpt-{model_name}'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': f'slogpt-{model_name}'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': f'slogpt-{model_name}',
                            'image': f'slogpt-{model_name}:latest',
                            'ports': [{
                                'containerPort': 8000,
                                'protocol': 'TCP'
                            }],
                            'env': [
                                {
                                    'name': 'MODEL_NAME',
                                    'value': model_name
                                },
                                {
                                    'name': 'PYTHONPATH',
                                    'value': '/app'
                                }
                            ],
                            'resources': {
                                'requests': {
                                    'cpu': cpu_request,
                                    'memory': memory_request
                                },
                                'limits': {
                                    'cpu': cpu_limit,
                                    'memory': memory_limit
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Service manifest
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f'slogpt-{model_name}-service',
                'labels': {
                    'app': f'slogpt-{model_name}'
                }
            },
            'spec': {
                'selector': {
                    'app': f'slogpt-{model_name}'
                },
                'ports': [{
                    'protocol': 'TCP',
                    'port': 80,
                    'targetPort': 8000
                }],
                'type': 'LoadBalancer'
            }
        }
        
        # Save manifests
        k8s_dir = self.output_dir / "kubernetes"
        k8s_dir.mkdir(exist_ok=True)
        
        deployment_path = k8s_dir / f"deployment-{model_name}.yaml"
        service_path = k8s_dir / f"service-{model_name}.yaml"
        
        with open(deployment_path, 'w') as f:
            yaml.dump(manifest, f)
        
        with open(service_path, 'w') as f:
            yaml.dump(service_manifest, f)
        
        print(f"   ✅ Kubernetes deployment manifest: {deployment_path}")
        print(f"   ✅ Kubernetes service manifest: {service_path}")
        
        return {
            'deployment_path': str(deployment_path),
            'service_path': str(service_path),
            'cpu_request': cpu_request,
            'cpu_limit': cpu_limit,
            'memory_request': memory_request,
            'memory_limit': memory_limit
        }
    
    def create_ci_cd_pipeline(self, model_name: str, repository_url: str) -> Dict:
        """Create CI/CD pipeline configuration."""
        print(f"🔄 Creating CI/CD pipeline for {model_name}")
        
        # GitHub Actions workflow
        github_workflow = {
            'name': 'SloGPT CI/CD Pipeline',
            'on': {
                'push': {
                    'branches': ['main', 'develop']
                },
                'pull_request': {
                    'branches': ['main']
                }
            },
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v3'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '3.9'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install -r requirements.txt'
                        },
                        {
                            'name': 'Run tests',
                            'run': 'python3 -m pytest tests/ -v'
                        },
                        {
                            'name': 'Run benchmark tests',
                            'run': 'python3 benchmark_system.py --dataset-only'
                        }
                    ]
                },
                'build-and-deploy': {
                    'needs': 'test',
                    'runs-on': 'ubuntu-latest',
                    'if': 'github.ref == "refs/heads/main"',
                    'steps': [
                        {
                            'name': 'Build Docker image',
                            'run': f'docker build -t slogpt-{model_name}:${{ github.sha }} .'
                        },
                        {
                            'name': 'Deploy to staging',
                            'run': f'''
                                echo "Deploying to staging..."
                                # Add deployment commands here
                                kubectl apply -f kubernetes/deployment-{model_name}.yaml
                                kubectl apply -f kubernetes/service-{model_name}.yaml
                            '''
                        }
                    ]
                }
            }
        }
        
        # Save workflow
        workflows_dir = self.output_dir / ".github/workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_path = workflows_dir / "ci-cd.yml"
        with open(workflow_path, 'w') as f:
            yaml.dump(github_workflow, f)
        
        print(f"   ✅ CI/CD workflow saved: {workflow_path}")
        
        return {
            'workflow_path': str(workflow_path),
            'triggers': ['push', 'pull_request'],
            'environments': ['staging', 'production']
        }
    
    def create_monitoring_config(self, model_name: str) -> Dict:
        """Create monitoring and logging configuration."""
        print(f"📊 Creating monitoring configuration for {model_name}")
        
        # Prometheus monitoring config
        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'scrape_configs': [{
                'job_name': f'slogpt-{model_name}',
                'static_configs': [{
                    'targets': ['localhost:8000'],
                    'metrics_path': '/metrics',
                    'scrape_interval': '5s'
                }]
            }]
        }
        
        # Grafana dashboard config
        grafana_dashboard = {
            'dashboard': {
                'title': f'SloGPT {model_name} Dashboard',
                'tags': ['slogpt', 'ml', 'monitoring'],
                'timezone': 'browser',
                'panels': [
                    {
                        'title': 'Request Rate',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': f'rate(slogpt_{model_name}_requests_total[5m])'
                            }
                        ]
                    },
                    {
                        'title': 'Response Time',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': f'histogram_quantile(0.95, slopigt_{model_name}_response_time_seconds_bucket[5m])'
                            }
                        ]
                    },
                    {
                        'title': 'Memory Usage',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': f'slogpt_{model_name}_memory_usage_bytes'
                            }
                        ]
                    },
                    {
                        'title': 'GPU Utilization',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': f'slogpt_{model_name}_gpu_utilization_percent'
                            }
                        ]
                    }
                ]
            }
        }
        
        # Save monitoring configs
        monitoring_dir = self.output_dir / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        prometheus_path = monitoring_dir / "prometheus.yml"
        grafana_path = monitoring_dir / "grafana-dashboard.json"
        
        with open(prometheus_path, 'w') as f:
            yaml.dump(prometheus_config, f)
        
        with open(grafana_path, 'w') as f:
            json.dump(grafana_dashboard, f, indent=2)
        
        print(f"   ✅ Prometheus config: {prometheus_path}")
        print(f"   ✅ Grafana dashboard: {grafana_path}")
        
        return {
            'prometheus_config': str(prometheus_path),
            'grafana_dashboard': str(grafana_path)
        }
    
    def generate_deployment_package(self, model_path: str, model_name: str, repository_url: str = None) -> Dict:
        """Generate complete deployment package."""
        print(f"📦 Generating deployment package for {model_name}")
        print("=" * 50)
        
        package_info = {
            'model_name': model_name,
            'model_path': model_path,
            'generated_at': time.time(),
            'components': {}
        }
        
        # 1. Docker configuration
        print("🐳 Phase 1: Docker Configuration")
        dockerfile_path = self.create_dockerfile(model_name)
        docker_config = self.create_docker_configuration(model_path, model_name)
        package_info['components']['docker'] = {
            'dockerfile_path': dockerfile_path,
            'docker_compose': docker_config
        }
        
        # 2. Kubernetes manifests
        print("\n☸️ Phase 2: Kubernetes Configuration")
        
        # Load model config for resource estimation
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                model_config = checkpoint.get('config', {})
            else:
                model_config = {'parameters': 1000000}  # Default
        except:
            model_config = {'parameters': 1000000}
        
        k8s_config = self.create_kubernetes_manifest(model_name, model_config)
        package_info['components']['kubernetes'] = k8s_config
        
        # 3. CI/CD pipeline
        print("\n🔄 Phase 3: CI/CD Pipeline")
        if repository_url:
            cicd_config = self.create_ci_cd_pipeline(model_name, repository_url)
            package_info['components']['ci_cd'] = cicd_config
        
        # 4. Monitoring configuration
        print("\n📊 Phase 4: Monitoring Configuration")
        monitoring_config = self.create_monitoring_config(model_name)
        package_info['components']['monitoring'] = monitoring_config
        
        # 5. Deployment scripts
        print("\n🚀 Phase 5: Deployment Scripts")
        scripts_config = self._create_deployment_scripts(model_name, docker_config, k8s_config)
        package_info['components']['scripts'] = scripts_config
        
        # 6. Documentation
        print("\n📚 Phase 6: Documentation")
        docs_config = self._create_deployment_documentation(model_name, package_info)
        package_info['components']['documentation'] = docs_config
        
        # Save complete package info
        package_json_path = self.output_dir / f"deployment-package-{model_name}.json"
        with open(package_json_path, 'w') as f:
            json.dump(package_info, f, indent=2)
        
        print(f"\n✅ Deployment package generated successfully!")
        print(f"📁 Package info: {package_json_path}")
        print(f"📦 All components saved in: {self.output_dir}")
        
        return package_info
    
    def _create_deployment_scripts(self, model_name: str, docker_config: Dict, k8s_config: Dict) -> Dict:
        """Create deployment scripts."""
        scripts_dir = self.output_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Docker deployment script
        docker_script = f"""#!/bin/bash
# Docker deployment script for {model_name}

set -e

echo "🐳 Starting Docker deployment for {model_name}..."

# Build Docker image
echo "📦 Building Docker image..."
docker build -t slogpt-{model_name}:latest .

# Run container
echo "🚀 Starting container..."
docker run -d \\
    --name slogpt-{model_name} \\
    -p 8000:8000 \\
    --restart unless-stopped \\
    slogpt-{model_name}:latest

echo "✅ Container started successfully!"
echo "🌐 API available at: http://localhost:8000"
echo "📊 Logs: docker logs -f slogpt-{model_name}"

# Health check
echo "⏳ Waiting for service to be healthy..."
sleep 10

if curl -f http://localhost:8000/health; then
    echo "✅ Service is healthy!"
else
    echo "❌ Service health check failed!"
    exit 1
fi
"""
        
        # Kubernetes deployment script
        k8s_script = f"""#!/bin/bash
# Kubernetes deployment script for {model_name}

set -e

echo "☸️ Starting Kubernetes deployment for {model_name}..."

# Apply configurations
echo "📦 Applying deployment manifest..."
kubectl apply -f kubernetes/deployment-{model_name}.yaml

echo "🌐 Applying service manifest..."
kubectl apply -f kubernetes/service-{model_name}.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/slogpt-{model_name}

# Wait for service
echo "🌐 Waiting for service external IP..."
kubectl get service slogpt-{model_name}-service --watch

# Get service URL
SERVICE_IP=$(kubectl get service slogpt-{model_name}-service -o jsonpath='{{.status.loadBalancer.ingress[0].ip}}')
if [ -n "$SERVICE_IP" ]; then
    echo "✅ Service deployed successfully!"
    echo "🌐 API available at: http://$SERVICE_IP"
else
    echo "⚠️ Service not externally accessible (using NodePort)"
fi

echo "📊 Pod status:"
kubectl get pods -l app=slogpt-{model_name}
"""
        
        # Save scripts
        docker_script_path = scripts_dir / f"deploy-docker-{model_name}.sh"
        k8s_script_path = scripts_dir / f"deploy-k8s-{model_name}.sh"
        
        with open(docker_script_path, 'w') as f:
            f.write(docker_script)
        
        with open(k8s_script_path, 'w') as f:
            f.write(k8s_script)
        
        # Make scripts executable
        os.chmod(docker_script_path, 0o755)
        os.chmod(k8s_script_path, 0o755)
        
        return {
            'docker_script': str(docker_script_path),
            'kubernetes_script': str(k8s_script_path)
        }
    
    def _create_deployment_documentation(self, model_name: str, package_info: Dict) -> Dict:
        """Create deployment documentation."""
        docs_dir = self.output_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        readme_content = f"""# SloGPT {model_name} Deployment Guide

This guide covers deploying the SloGPT {model_name} model to production environments.

## 🚀 Quick Start

### Docker Deployment
```bash
# Using Docker Compose
docker-compose -f docker-compose_{model_name}.yml up -d

# Using standalone Docker
bash scripts/deploy-docker-{model_name}.sh
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
bash scripts/deploy-k8s-{model_name}.sh

# Or apply manifests directly
kubectl apply -f kubernetes/deployment-{model_name}.yaml
kubectl apply -f kubernetes/service-{model_name}.yaml
```

## 📊 Model Information

- **Model Name**: {model_name}
- **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(package_info['generated_at']))}
- **Components**: {len(package_info['components'])}

## 🔧 Configuration

### Environment Variables
- `MODEL_NAME`: Model identifier
- `MODEL_PATH`: Path to model files
- `PYTHONPATH`: Application path

### Ports
- **API Server**: 8000
- **Health Check**: 8000/health

### Resources
{package_info.get('components', {}).get('kubernetes', {}).get('cpu_request', 'N/A')}

## 📈 Monitoring

The deployment includes comprehensive monitoring setup:

- **Prometheus**: Metrics collection at `/metrics`
- **Grafana**: Visualization dashboard
- **Health Checks**: Automatic health monitoring

## 🛠️ Troubleshooting

### Common Issues

1. **Container won't start**
   ```bash
   docker logs slogpt-{model_name}
   ```

2. **Service not accessible**
   ```bash
   kubectl get pods -l app=slogpt-{model_name}
   kubectl describe service slogpt-{model_name}-service
   ```

3. **High memory usage**
   - Reduce model batch size
   - Increase memory limits in Kubernetes manifest

## 🔄 CI/CD Integration

The deployment package includes automated CI/CD pipeline:
- **GitHub Actions**: `.github/workflows/ci-cd.yml`
- **Triggers**: Push to main branch
- **Environments**: Staging and Production

## 📞 Support

For issues and questions:
1. Check container logs
2. Review monitoring dashboard
3. Consult troubleshooting section

Generated by SloGPT Deployment System
{time.strftime('%Y-%m-%d', time.gmtime(package_info['generated_at']))}
"""
        
        # Save README
        readme_path = docs_dir / f"README-{model_name}.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        return {
            'readme_path': str(readme_path),
            'documentation_generated': True
        }


def main():
    """Command line interface for deployment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SloGPT Automated Deployment")
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--name', required=True, help='Model name for deployment')
    parser.add_argument('--output', default='deployment', help='Output directory')
    parser.add_argument('--repo', help='Repository URL for CI/CD')
    parser.add_argument('--docker-only', action='store_true', help='Only create Docker configuration')
    parser.add_argument('--k8s-only', action='store_true', help='Only create Kubernetes manifests')
    
    args = parser.parse_args()
    
    # Create deployment manager
    deployer = DeploymentManager(args.output)
    
    if args.docker_only:
        # Only Docker configuration
        dockerfile_path = deployer.create_dockerfile(args.name)
        docker_config = deployer.create_docker_configuration(args.model, args.name)
        print(f"\n✅ Docker configuration completed!")
        print(f"📁 Dockerfile: {dockerfile_path}")
        print(f"📁 Docker Compose: {docker_config['docker_compose_path']}")
        
    elif args.k8s_only:
        # Only Kubernetes configuration
        # Load model config for resource estimation
        try:
            import torch
            checkpoint = torch.load(args.model, map_location='cpu')
            if isinstance(checkpoint, dict):
                model_config = checkpoint.get('config', {})
            else:
                model_config = {'parameters': 1000000}
        except:
            model_config = {'parameters': 1000000}
        
        k8s_config = deployer.create_kubernetes_manifest(args.name, model_config)
        print(f"\n✅ Kubernetes configuration completed!")
        print(f"📁 Deployment: {k8s_config['deployment_path']}")
        print(f"📁 Service: {k8s_config['service_path']}")
        
    else:
        # Complete deployment package
        package_info = deployer.generate_deployment_package(args.model, args.name, args.repo)
        
        print(f"\n🎉 Deployment package completed successfully!")
        print(f"📦 All files generated in: {args.output}")
        print(f"📊 Package info: deployment-package-{args.name}.json")
    
    return 0


if __name__ == '__main__':
    exit(main())