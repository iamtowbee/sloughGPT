"""Deployment management module for production infrastructure."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import os
import subprocess
from pathlib import Path


@dataclass
class DeploymentConfig:
    environment: str
    region: str
    project_name: str
    cluster_name: str
    namespace: str
    replicas: int
    resources: Dict[str, str]
    ingress_host: str
    ssl_enabled: bool = True


@dataclass
class ServiceConfig:
    name: str
    image: str
    port: int
    replicas: int
    resources: Dict[str, str]
    env_vars: Dict[str, str]
    health_check: Optional[Dict[str, Any]] = None


@dataclass
class DeploymentStatus:
    service: str
    replicas: int
    ready_replicas: int
    status: str
    created_at: datetime
    last_updated: datetime


class DeploymentManager:
    def __init__(self):
        self.deployments: Dict[str, DeploymentConfig] = {}
        self.services: Dict[str, ServiceConfig] = {}
        self.deployment_status: Dict[str, DeploymentStatus] = {}
        self.kubeconfig_path = os.path.expanduser("~/.kube/config")
    
    def create_deployment_config(self, environment: str, region: str, project_name: str,
                               cluster_name: str, namespace: str, replicas: int,
                               resources: Dict[str, str], ingress_host: str) -> str:
        """Create deployment configuration."""
        config_id = f"{environment}-{project_name}"
        
        config = DeploymentConfig(
            environment=environment,
            region=region,
            project_name=project_name,
            cluster_name=cluster_name,
            namespace=namespace,
            replicas=replicas,
            resources=resources,
            ingress_host=ingress_host,
            ssl_enabled=True
        )
        
        self.deployments[config_id] = config
        return config_id
    
    def add_service(self, deployment_id: str, service_config: ServiceConfig) -> None:
        """Add service to deployment."""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        self.services[service_config.name] = service_config
    
    def generate_kubernetes_manifests(self, deployment_id: str) -> Dict[str, str]:
        """Generate Kubernetes manifests for deployment."""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        config = self.deployments[deployment_id]
        manifests = {}
        
        # Namespace
        manifests["namespace"] = self._generate_namespace_yaml(config)
        
        # ConfigMaps
        manifests["configmap"] = self._generate_configmap_yaml(config)
        
        # Secrets
        manifests["secret"] = self._generate_secret_yaml(config)
        
        # Services
        for service_name, service_config in self.services.items():
            manifests[f"service-{service_name}"] = self._generate_service_yaml(service_config)
        
        # Deployments
        for service_name, service_config in self.services.items():
            manifests[f"deployment-{service_name}"] = self._generate_deployment_yaml(service_config, config)
        
        # Ingress
        manifests["ingress"] = self._generate_ingress_yaml(config)
        
        # HPA
        for service_name, service_config in self.services.items():
            manifests[f"hpa-{service_name}"] = self._generate_hpa_yaml(service_name, service_config, config)
        
        return manifests
    
    def _generate_namespace_yaml(self, config: DeploymentConfig) -> str:
        """Generate namespace YAML."""
        return f"""apiVersion: v1
kind: Namespace
metadata:
  name: {config.namespace}
  labels:
    name: {config.namespace}
    environment: {config.environment}"""
    
    def _generate_configmap_yaml(self, config: DeploymentConfig) -> str:
        """Generate ConfigMap YAML."""
        return f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: {config.project_name}-config
  namespace: {config.namespace}
data:
  ENVIRONMENT: "{config.environment}"
  REGION: "{config.region}"
  PROJECT_NAME: "{config.project_name}"
  LOG_LEVEL: "info\""""
    
    def _generate_secret_yaml(self, config: DeploymentConfig) -> str:
        """Generate Secret YAML."""
        return f"""apiVersion: v1
kind: Secret
metadata:
  name: {config.project_name}-secrets
  namespace: {config.namespace}
type: Opaque
data:
  # Base64 encoded values
  JWT_SECRET: "eW91ci1qd3Qtc2VjcmV0LWtleS1jaGFuZ2UtaW4tcHJvZHVjdGlvbg=="
  DATABASE_URL: "cG9zdGdyZXNxbDovL3VzZXI6cGFzc0BkYi5leGFtcGxlLmNvbS9zbG91Z2hncHQ="
  REDIS_URL: "cmVkaXM6Ly9yZWRpcy5leGFtcGxlLmNvbS82Mzc5\""""
    
    def _generate_service_yaml(self, service_config: ServiceConfig) -> str:
        """Generate Service YAML."""
        return f"""apiVersion: v1
kind: Service
metadata:
  name: {service_config.name}-service
  namespace: default
  labels:
    app: {service_config.name}
spec:
  selector:
    app: {service_config.name}
  ports:
  - port: {service_config.port}
    targetPort: {service_config.port}
    protocol: TCP
  type: ClusterIP"""
    
    def _generate_deployment_yaml(self, service_config: ServiceConfig, config: DeploymentConfig) -> str:
        """Generate Deployment YAML."""
        env_vars = ""
        for key, value in service_config.env_vars.items():
            env_vars += f'        - name: {key}\n          value: "{value}"\n'
        
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {service_config.name}
  namespace: default
  labels:
    app: {service_config.name}
spec:
  replicas: {service_config.replicas}
  selector:
    matchLabels:
      app: {service_config.name}
  template:
    metadata:
      labels:
        app: {service_config.name}
    spec:
      containers:
      - name: {service_config.name}
        image: {service_config.image}
        ports:
        - containerPort: {service_config.port}
        resources:
          requests:
            memory: "{service_config.resources.get('memory_request', '256Mi')}"
            cpu: "{service_config.resources.get('cpu_request', '250m')}"
          limits:
            memory: "{service_config.resources.get('memory_limit', '512Mi')}"
            cpu: "{service_config.resources.get('cpu_limit', '500m')}"
{env_vars}        envFrom:
        - configMapRef:
            name: {config.project_name}-config
        - secretRef:
            name: {config.project_name}-secrets"""
    
    def _generate_ingress_yaml(self, config: DeploymentConfig) -> str:
        """Generate Ingress YAML."""
        tls_config = ""
        if config.ssl_enabled:
            tls_config = f"""
  tls:
  - hosts:
    - {config.ingress_host}
    secretName: {config.project_name}-tls"""
        
        return f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {config.project_name}-ingress
  namespace: default
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  rules:
  - host: {config.ingress_host}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8000
      - path: /monitoring
        pathType: Prefix
        backend:
          service:
            name: monitoring-service
            port:
              number: 9090{tls_config}"""
    
    def _generate_hpa_yaml(self, service_name: str, service_config: ServiceConfig, config: DeploymentConfig) -> str:
        """Generate Horizontal Pod Autoscaler YAML."""
        return f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {service_name}-hpa
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {service_name}
  minReplicas: {max(1, service_config.replicas // 2)}
  maxReplicas: {service_config.replicas * 2}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80"""
    
    def deploy_infrastructure(self, deployment_id: str) -> Dict[str, Any]:
        """Deploy infrastructure to Kubernetes."""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        manifests = self.generate_kubernetes_manifests(deployment_id)
        
        # Apply manifests to Kubernetes
        results = {}
        
        for manifest_name, manifest_content in manifests.items():
            try:
                # Write manifest to temporary file
                temp_file = f"/tmp/{manifest_name}.yaml"
                with open(temp_file, 'w') as f:
                    f.write(manifest_content)
                
                # Apply with kubectl
                result = subprocess.run(
                    ["kubectl", "apply", "-f", temp_file],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    results[manifest_name] = {"status": "success", "output": result.stdout}
                else:
                    results[manifest_name] = {"status": "error", "error": result.stderr}
                
                # Clean up temp file
                os.remove(temp_file)
                
            except Exception as e:
                results[manifest_name] = {"status": "error", "error": str(e)}
        
        return results