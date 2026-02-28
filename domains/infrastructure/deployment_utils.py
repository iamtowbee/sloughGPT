"""
Deployment System - Ported from recovered deployment_system.py
Containerization and cloud deployment automation.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml


class DeploymentConfig:
    """Deployment configuration."""
    
    def __init__(
        self,
        model_name: str = "slogpt",
        port: int = 8000,
        gpu_enabled: bool = True,
        workers: int = 1,
    ):
        self.model_name = model_name
        self.port = port
        self.gpu_enabled = gpu_enabled
        self.workers = workers


class DeploymentManager:
    """Manages automated deployment of SloughGPT models."""
    
    def __init__(self, output_dir: str = "deployment"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
        self.deployment_logs: List[str] = []
    
    def create_docker_compose(self, config: DeploymentConfig) -> Path:
        """Create Docker Compose configuration."""
        compose = {
            'version': '3.8',
            'services': {
                f'{config.model_name}-api': {
                    'build': {'context': '.', 'dockerfile': 'Dockerfile'},
                    'ports': [f'{config.port}:{config.port}'],
                    'environment': [
                        f'MODEL_NAME={config.model_name}',
                        'PYTHONPATH=/app',
                    ],
                    'deploy': {
                        'resources': {
                            'reservations': {
'devices': [{
                        'driver': 'nvidia',
                        'count': 1,
                        'capabilities': ['gpu']
                    }] if config.gpu_enabled else []
                            }
                        }
                    } if config.gpu_enabled else {},
                    'restart': 'unless-stopped',
                }
            }
        }
        
        path = self.output_dir / f"docker-compose-{config.model_name}.yml"
        with open(path, 'w') as f:
            yaml.dump(compose, f)
        
        return path
    
    def create_dockerfile(self, config: DeploymentConfig) -> Path:
        """Create Dockerfile."""
        dockerfile = f'''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE {config.port}

CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "{config.port}"]
'''
        
        path = self.output_dir / f"Dockerfile-{config.model_name}"
        with open(path, 'w') as f:
            f.write(dockerfile)
        
        return path
    
    def build_docker(self, config: DeploymentConfig, tag: Optional[str] = None) -> bool:
        """Build Docker image."""
        tag = tag or f"{config.model_name}:latest"
        
        try:
            result = subprocess.run(
                ["docker", "build", "-t", tag, "."],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            self.deployment_logs.append("Docker not found")
            return False
    
    def deploy_docker_compose(self, config: DeploymentConfig, detach: bool = True) -> bool:
        """Deploy using Docker Compose."""
        compose_file = self.create_docker_compose(config)
        
        cmd = ["docker-compose", "-f", str(compose_file), "up"]
        if detach:
            cmd.append("-d")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            self.deployment_logs.append("docker-compose not found")
            return False
    
    def get_deployment_status(self, config: DeploymentConfig) -> Dict:
        """Get deployment status."""
        return {
            "model_name": config.model_name,
            "port": config.port,
            "gpu_enabled": config.gpu_enabled,
            "logs": self.deployment_logs[-10:],
        }


__all__ = ["DeploymentManager", "DeploymentConfig"]
