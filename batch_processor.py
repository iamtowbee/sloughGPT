#!/usr/bin/env python3
"""
Batch Processing and Automation Tools for SloGPT

Process multiple datasets, schedule training jobs, and automate workflows.
"""

import os
import json
import time
import yaml
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue


class BatchProcessor:
    """Process multiple datasets in parallel."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.results = []
        self.progress_queue = queue.Queue()
    
    def process_datasets(self, datasets: List[Dict], operation: str) -> List[Dict]:
        """Process multiple datasets with specified operation."""
        print(f"🔄 Processing {len(datasets)} datasets with operation: {operation}")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_dataset = {
                executor.submit(self._process_single_dataset, dataset, operation): dataset
                for dataset in datasets
            }
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    result["dataset"] = dataset["name"]
                    self.results.append(result)
                    completed += 1
                    print(f"✅ Completed {completed}/{len(datasets)}: {dataset['name']}")
                except Exception as e:
                    error_result = {
                        "dataset": dataset["name"],
                        "status": "error",
                        "error": str(e)
                    }
                    self.results.append(error_result)
                    print(f"❌ Failed {dataset['name']}: {e}")
        
        return self.results
    
    def _process_single_dataset(self, dataset: Dict, operation: str) -> Dict:
        """Process a single dataset."""
        name = dataset["name"]
        source = dataset.get("source")
        
        if operation == "create":
            return self._create_dataset(dataset)
        elif operation == "prepare":
            return self._prepare_dataset(name)
        elif operation == "validate":
            return self._validate_dataset(name)
        elif operation == "train":
            return self._train_dataset(dataset)
        else:
            return {"status": "error", "error": f"Unknown operation: {operation}"}
    
    def _create_dataset(self, dataset: Dict) -> Dict:
        """Create a dataset."""
        name = dataset["name"]
        source = dataset.get("source")
        text = dataset.get("text")
        
        if text:
            cmd = f'python3 create_dataset_fixed.py "{name}" "{text}"'
        elif source and Path(source).is_file():
            cmd = f'python3 create_dataset_fixed.py "{name}" --file "{source}"'
        elif source and Path(source).is_dir():
            cmd = f'python3 create_dataset_fixed.py "{name}" --folder "{source}"'
        else:
            return {"status": "error", "error": "Invalid source specification"}
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return {
            "status": "success" if result.returncode == 0 else "error",
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }
    
    def _prepare_dataset(self, name: str) -> Dict:
        """Prepare a dataset."""
        cmd = f'python3 universal_prepare.py --name "{name}" --source "./datasets/{name}" --recursive'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return {
            "status": "success" if result.returncode == 0 else "error",
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }
    
    def _validate_dataset(self, name: str) -> Dict:
        """Validate a dataset."""
        cmd = f'python3 advanced_dataset_features.py validate "{name}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return {
            "status": "success" if result.returncode == 0 else "error",
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }
    
    def _train_dataset(self, dataset: Dict) -> Dict:
        """Train on a dataset."""
        name = dataset["name"]
        config = dataset.get("config", {})
        
        # Build training command
        cmd_parts = ["python3", "train_simple.py", name]
        
        # Add configuration options
        for key, value in config.items():
            if key == "from_model":
                cmd_parts.extend(["--from", value])
            elif key == "mixed":
                cmd_parts.extend(["--mixed", value])
            elif key == "device":
                cmd_parts.extend(["--device", value])
            else:
                cmd_parts.extend([f"--{key}", str(value)])
        
        cmd = " ".join(cmd_parts)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
            "config": config
        }


class WorkflowScheduler:
    """Schedule and automate training workflows."""
    
    def __init__(self):
        self.scheduled_jobs = []
        self.running_jobs = {}
        self.completed_jobs = []
        self.job_queue = queue.Queue()
    
    def schedule_job(self, job_config: Dict) -> str:
        """Schedule a new job."""
        job_id = f"job_{int(time.time())}_{len(self.scheduled_jobs)}"
        
        job = {
            "id": job_id,
            "config": job_config,
            "status": "scheduled",
            "created_at": datetime.now().isoformat(),
            "scheduled_at": job_config.get("scheduled_at"),
            "dependencies": job_config.get("dependencies", [])
        }
        
        self.scheduled_jobs.append(job)
        print(f"📅 Scheduled job: {job_id}")
        
        return job_id
    
    def run_job(self, job_id: str) -> Dict:
        """Run a scheduled job."""
        # Find job
        job = next((j for j in self.scheduled_jobs if j["id"] == job_id), None)
        if not job:
            return {"status": "error", "error": f"Job {job_id} not found"}
        
        # Check dependencies
        for dep_id in job["dependencies"]:
            if dep_id not in [j["id"] for j in self.completed_jobs]:
                return {"status": "waiting", "error": f"Waiting for dependency: {dep_id}"}
        
        # Update job status
        job["status"] = "running"
        job["started_at"] = datetime.now().isoformat()
        self.running_jobs[job_id] = job
        
        print(f"🚀 Starting job: {job_id}")
        
        # Execute job
        try:
            result = self._execute_job(job["config"])
            job["status"] = "completed"
            job["completed_at"] = datetime.now().isoformat()
            job["result"] = result
            
            # Move from running to completed
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            self.completed_jobs.append(job)
            
            print(f"✅ Completed job: {job_id}")
            return {"status": "success", "result": result}
            
        except Exception as e:
            job["status"] = "failed"
            job["failed_at"] = datetime.now().isoformat()
            job["error"] = str(e)
            
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            
            print(f"❌ Failed job: {job_id} - {e}")
            return {"status": "error", "error": str(e)}
    
    def _execute_job(self, job_config: Dict) -> Dict:
        """Execute a job configuration."""
        job_type = job_config.get("type", "train")
        
        if job_type == "batch_process":
            batch_config = job_config.get("batch_config", {})
            datasets = batch_config.get("datasets", [])
            operation = batch_config.get("operation", "create")
            
            processor = BatchProcessor()
            return processor.process_datasets(datasets, operation)
        
        elif job_type == "train":
            dataset_config = {
                "name": job_config.get("dataset"),
                "config": job_config.get("training_config", {})
            }
            
            processor = BatchProcessor()
            return processor._train_dataset(dataset_config)
        
        elif job_type == "pipeline":
            return self._execute_pipeline(job_config)
        
        else:
            raise ValueError(f"Unknown job type: {job_type}")
    
    def _execute_pipeline(self, pipeline_config: Dict) -> Dict:
        """Execute a multi-step pipeline."""
        steps = pipeline_config.get("steps", [])
        results = []
        
        for i, step in enumerate(steps):
            print(f"🔄 Pipeline step {i+1}/{len(steps)}: {step.get('name', 'Step ' + str(i+1))}")
            
            step_result = self._execute_job(step)
            step_result["step"] = step.get("name", f"step_{i+1}")
            results.append(step_result)
            
            if step_result.get("status") == "error":
                print(f"❌ Pipeline failed at step: {step.get('name')}")
                break
        
        return {
            "status": "success" if all(r.get("status") == "success" for r in results) else "error",
            "results": results
        }
    
    def list_jobs(self, status_filter: str = None) -> List[Dict]:
        """List jobs with optional status filter."""
        all_jobs = self.scheduled_jobs + list(self.running_jobs.values()) + self.completed_jobs
        
        if status_filter:
            return [job for job in all_jobs if job.get("status") == status_filter]
        
        return all_jobs
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id in self.running_jobs:
            job = self.running_jobs[job_id]
            job["status"] = "cancelled"
            job["cancelled_at"] = datetime.now().isoformat()
            
            del self.running_jobs[job_id]
            print(f"⏹️ Cancelled job: {job_id}")
            return True
        
        return False


class AutomationManager:
    """High-level automation for common workflows."""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """Load automation templates."""
        return {
            "research_pipeline": {
                "name": "Research Dataset Pipeline",
                "description": "Create, validate, and train research datasets",
                "steps": [
                    {
                        "name": "create_datasets",
                        "type": "batch_process",
                        "batch_config": {
                            "datasets": [
                                {"name": "papers", "source": "./data/research_papers"},
                                {"name": "articles", "source": "./data/research_articles"}
                            ],
                            "operation": "create"
                        }
                    },
                    {
                        "name": "validate_datasets",
                        "type": "batch_process", 
                        "batch_config": {
                            "datasets": [
                                {"name": "papers"},
                                {"name": "articles"}
                            ],
                            "operation": "validate"
                        }
                    },
                    {
                        "name": "mix_datasets",
                        "type": "mix",
                        "batch_config": {
                            "ratios": {"papers": 0.6, "articles": 0.4},
                            "output": "research_mixed"
                        }
                    },
                    {
                        "name": "train_model",
                        "type": "train",
                        "dataset": "research_mixed",
                        "training_config": {
                            "batch_size": 32,
                            "learning_rate": 3e-4,
                            "max_iters": 5000
                        }
                    }
                ]
            },
            
            "production_training": {
                "name": "Production Training Pipeline",
                "description": "Automated production model training",
                "steps": [
                    {
                        "name": "data_validation",
                        "type": "batch_process",
                        "batch_config": {
                            "datasets": [
                                {"name": "production_data"},
                                {"name": "validation_data"}
                            ],
                            "operation": "validate"
                        }
                    },
                    {
                        "name": "model_training",
                        "type": "train",
                        "dataset": "production_data",
                        "training_config": {
                            "from_model": "gpt2",
                            "batch_size": 64,
                            "learning_rate": 1e-5,
                            "device": "auto"
                        }
                    },
                    {
                        "name": "evaluation",
                        "type": "evaluate",
                        "dataset": "validation_data",
                        "model": "production_data"
                    }
                ]
            }
        }
    
    def run_template(self, template_name: str, variables: Dict = None) -> str:
        """Run an automation template."""
        template = self.templates.get(template_name)
        if not template:
            print(f"❌ Template not found: {template_name}")
            print("Available templates:")
            for name in self.templates.keys():
                print(f"  - {name}")
            return None
        
        print(f"🚀 Running template: {template['name']}")
        print(f"📝 {template['description']}")
        
        # Substitute variables in template
        template_config = self._substitute_variables(template, variables or {})
        
        # Create and run pipeline
        scheduler = WorkflowScheduler()
        job_id = scheduler.schedule_job({
            "type": "pipeline",
            "steps": template_config["steps"]
        })
        
        result = scheduler.run_job(job_id)
        return job_id
    
    def _substitute_variables(self, template: Dict, variables: Dict) -> Dict:
        """Substitute variables in template."""
        template_str = json.dumps(template)
        
        for var, value in variables.items():
            template_str = template_str.replace(f"${{{var}}}", str(value))
        
        return json.loads(template_str)
    
    def list_templates(self) -> None:
        """List available automation templates."""
        print("📋 Available Automation Templates:")
        print("=" * 50)
        
        for name, template in self.templates.items():
            print(f"\n🎯 {name}")
            print(f"   {template['description']}")
            print(f"   Steps: {len(template['steps'])}")


def main():
    """Main CLI for batch processing and automation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch processing and automation for SloGPT")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Process multiple datasets')
    batch_parser.add_argument('--config', required=True, help='Batch configuration file')
    batch_parser.add_argument('--operation', default='create', choices=['create', 'prepare', 'validate', 'train'])
    batch_parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    # Scheduler commands
    schedule_parser = subparsers.add_parser('schedule', help='Schedule jobs')
    schedule_parser.add_argument('--config', required=True, help='Job configuration file')
    schedule_parser.add_argument('--at', help='Schedule time (YYYY-MM-DD HH:MM)')
    
    run_parser = subparsers.add_parser('run', help='Run scheduled job')
    run_parser.add_argument('job_id', help='Job ID to run')
    
    list_jobs_parser = subparsers.add_parser('list-jobs', help='List jobs')
    list_jobs_parser.add_argument('--status', help='Filter by status')
    
    cancel_parser = subparsers.add_parser('cancel', help='Cancel running job')
    cancel_parser.add_argument('job_id', help='Job ID to cancel')
    
    # Automation templates
    template_parser = subparsers.add_parser('template', help='Run automation template')
    template_parser.add_argument('name', help='Template name')
    template_parser.add_argument('--var', action='append', help='Variables (format: key=value)')
    
    list_templates_parser = subparsers.add_parser('list-templates', help='List automation templates')
    
    args = parser.parse_args()
    
    if args.command == 'batch':
        print(f"🔄 Processing batch job from {args.config}")
        
        # Load batch configuration
        with open(args.config, 'r') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        processor = BatchProcessor(args.workers)
        results = processor.process_datasets(config.get("datasets", []), args.operation)
        
        # Save results
        output_file = f"batch_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Results saved to: {output_file}")
    
    elif args.command == 'schedule':
        print(f"📅 Scheduling job from {args.config}")
        
        with open(args.config, 'r') as f:
            job_config = json.load(f)
        
        scheduler = WorkflowScheduler()
        job_id = scheduler.schedule_job(job_config)
        print(f"✅ Job scheduled: {job_id}")
    
    elif args.command == 'run':
        scheduler = WorkflowScheduler()
        result = scheduler.run_job(args.job_id)
        print(f"📊 Job result: {result}")
    
    elif args.command == 'list-jobs':
        scheduler = WorkflowScheduler()
        jobs = scheduler.list_jobs(args.status)
        
        print(f"📋 Jobs ({'Status: ' + args.status if args.status else 'All'})")
        for job in jobs:
            status_icon = {"scheduled": "📅", "running": "🚀", "completed": "✅", "failed": "❌", "cancelled": "⏹️"}
            icon = status_icon.get(job.get("status"), "❓")
            print(f"  {icon} {job['id']} - {job.get('status', 'unknown')}")
    
    elif args.command == 'cancel':
        scheduler = WorkflowScheduler()
        success = scheduler.cancel_job(args.job_id)
        
        if success:
            print(f"✅ Cancelled job: {args.job_id}")
        else:
            print(f"❌ Job not found: {args.job_id}")
    
    elif args.command == 'template':
        # Parse variables
        variables = {}
        if args.var:
            for var in args.var:
                if '=' in var:
                    key, value = var.split('=', 1)
                    variables[key.strip()] = value.strip()
        
        automation = AutomationManager()
        job_id = automation.run_template(args.name, variables)
        print(f"🚀 Template job: {job_id}")
    
    elif args.command == 'list-templates':
        automation = AutomationManager()
        automation.list_templates()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()