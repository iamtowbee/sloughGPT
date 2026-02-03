#!/usr/bin/env python3
"""
Distributed Training Support for SloGPT

Multi-GPU and cluster training capabilities for large-scale model training.
"""

import os
import json
import time
import subprocess
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional
import socket
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class DistributedTrainer:
    """Manages distributed training across multiple GPUs or nodes."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.backend = config.get('backend', 'nccl')
        self.device = None
        
        self.setup_distributed()
    
    def setup_distributed(self):
        """Initialize distributed training."""
        if not self.config.get('use_distributed', False):
            print("🚀 Running in single-process mode")
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            return
        
        # Check if distributed training is available
        if not dist.is_available():
            print("⚠️ Distributed training not available")
            print("   Falling back to single GPU/CPU training")
            self.config['use_distributed'] = False
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            return
        
        # Initialize process group
        try:
            init_method = self.config.get('init_method', 'env://')
            world_size = self.config.get('world_size', 1)
            rank = self.config.get('rank', 0)
            
            # Set environment variables for distributed training
            if 'RANK' not in os.environ and rank > 0:
                os.environ['RANK'] = str(rank)
            if 'WORLD_SIZE' not in os.environ and world_size > 1:
                os.environ['WORLD_SIZE'] = str(world_size)
            if 'LOCAL_RANK' not in os.environ:
                local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
                os.environ['LOCAL_RANK'] = str(local_rank)
            
            dist.init_process_group(
                backend=self.backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank
            )
            
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get('LOCAL_RANK', self.rank))
            
            # Setup device
            if torch.cuda.is_available():
                self.device = f"cuda:{self.local_rank}"
                torch.cuda.set_device(self.local_rank)
            else:
                self.device = f"cpu"
            
            if self.rank == 0:
                print(f"🔗 Distributed setup complete")
                print(f"   World size: {self.world_size}")
                print(f"   Global rank: {self.rank}")
                print(f"   Local rank: {self.local_rank}")
                print(f"   Backend: {self.backend}")
                print(f"   Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Distributed setup failed: {e}")
            print("   Falling back to single GPU/CPU training")
            self.config['use_distributed'] = False
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def create_distributed_config(self, base_config: Dict, num_gpus: int = 1, num_nodes: int = 1, node_rank: int = 0) -> Dict:
        """Create distributed training configuration."""
        distributed_config = {
            'distributed': True,
            'backend': self.config.get('backend', 'nccl'),
            'world_size': num_gpus * num_nodes,
            'rank': node_rank * num_gpus + self.local_rank,
            'local_rank': self.local_rank,
            'init_method': self.config.get('init_method', 'env'),
            'num_gpus': num_gpus,
            'num_nodes': num_nodes,
            'node_rank': node_rank,
            
            'training': {
                **base_config
            }
        }
        
        # Adjust batch size for distributed training
        original_batch_size = base_config.get('batch_size', 32)
        distributed_config['training']['batch_size'] = original_batch_size // num_gpus
        
        # Use gradient accumulation if needed
        if original_batch_size % num_gpus != 0:
            distributed_config['training']['gradient_accumulation_steps'] = num_gpus
            distributed_config['training']['accumulated_batch_size'] = original_batch_size
        else:
            distributed_config['training']['gradient_accumulation_steps'] = 1
        
        return distributed_config
    
    def wrap_model(self, model):
        """Wrap model for distributed training."""
        if not self.config.get('use_distributed', False):
            return model
        
        if not self.config.get('wrap_model', True):
            return model
        
        if self.config.get('use_ddp', True) and dist.is_initialized():
            # Move model to the correct device first
            model = model.to(self.device)
            
            # Wrap with DistributedDataParallel
            model = DDP(model, device_ids=[self.local_rank] if torch.cuda.is_available() else None)
            
            if self.rank == 0:
                print(f"🔗 Model wrapped with DDP")
                print(f"   Device IDs: {[self.local_rank] if torch.cuda.is_available() else None}")
        
        return model
    
    def save_checkpoint_distributed(self, model, optimizer, epoch: int, loss: float, additional_info: Dict = None):
        """Save checkpoint in distributed training."""
        # Only save on rank 0
        if self.rank != 0:
            return
        
        checkpoint_dir = Path(self.config.get('output_dir', 'out-distributed'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model state dict (handle DDP wrapped models)
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'device': self.device
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save latest checkpoint
        latest_path = checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        print(f"💾 Latest checkpoint: {latest_path}")
    
    def load_checkpoint_distributed(self, checkpoint_path: str):
        """Load checkpoint for distributed training."""
        if self.config.get('use_distributed', False):
            return None
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return None
        
        if self.local_rank == 0:
            print(f"📥 Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        # Validate compatibility
        if 'world_size' in checkpoint and checkpoint['world_size'] != self.world_size:
            print(f"⚠️ World size mismatch: checkpoint={checkpoint['world_size']}, current={self.world_size}")
            return None
        
        if 'rank' in checkpoint and checkpoint['rank'] != self.rank:
            print(f"⚠️ Rank mismatch: checkpoint={checkpoint['rank']}, current={self.rank}")
            return None
        
        return checkpoint
    
    def cleanup_distributed(self):
        """Clean up distributed training resources."""
        if self.config.get('use_distributed', False):
            return
        
        dist.destroy_process_group()
        print("🧹 Distributed training cleanup complete")


class ClusterManager:
    """Manages multi-node cluster training."""
    
    def __init__(self, master_config: Dict):
        self.master_config = master_config
        self.slave_nodes = []
        self.task_queue = queue.Queue()
        
    def setup_master(self):
        """Setup training master node."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.master_config['host'], self.master_config['port']))
        self.socket.listen(5)
        
        print(f"🖥 Master server listening on {self.master_config['host']}:{self.master_config['port']}")
        
        # Accept connections from slave nodes
        for i in range(self.master_config['num_slaves']):
            def handle_slave(conn, addr):
                print(f"🔗 Connected to slave {addr}")
                slave_data = {
                    'connection': conn,
                    'address': addr,
                    'status': 'ready'
                }
                self.slave_nodes.append(slave_data)
        
        # Start accepting connections
        threading.Thread(target=self._accept_connections, daemon=True).start()
    
    def _accept_connections(self):
        """Accept connections from slave nodes."""
        while True:
            conn, addr = self.socket.accept()
            self._handle_slave(conn, addr)
    
    def _handle_slave(self, conn, addr):
        """Handle communication with a slave node."""
        try:
            threading.Thread(target=self._slave_communication, args=(conn, addr), daemon=True).start()
        except Exception as e:
            print(f"❌ Error handling slave {addr}: {e}")
    
    def _slave_communication(self, conn, addr):
        """Handle ongoing communication with a slave node."""
        slave_data = next((s for s in self.slave_nodes if s['address'] == addr), None)
        if slave_data:
            slave_data['connection'] = conn
            slave_data['status'] = 'connected'
        
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                
                # Handle different message types
                if data.startswith('STATUS:'):
                    status = data[7:].strip()
                    slave_data['status'] = status
                elif data.startswith('METRICS:'):
                    metrics = json.loads(data[9:].strip())
                    print(f"📊 Metrics from {addr}: {metrics}")
                elif data.startswith('CHECKPOINT:'):
                    # Handle checkpoint transfer
                    pass
                elif data.startswith('TRAINING:'):
                    # Handle training commands
                    pass
                elif data.startswith('DONE:'):
                    # Training completed
                    print(f"✅ Training completed on {addr}")
                    slave_data['status'] = 'completed'
                elif data.startswith('ERROR:'):
                    error_msg = data[6:].strip()
                    print(f"❌ Error on {addr}: {error_msg}")
                    slave_data['status'] = 'error'
                else:
                    # Unknown message
                    print(f"🔔 Unknown message from {addr}: {data[:50]}...")
        
        except Exception as e:
            print(f"❌ Communication error with {addr}: {e}")
        finally:
            conn.close()
            if slave_data:
                slave_data['status'] = 'disconnected'
    
    def distribute_training(self, config: Dict):
        """Distribute training job across cluster."""
        available_slaves = [s for s in self.slave_nodes if s['status'] == 'ready']
        
        if not available_slaves:
            print("⚠️ No ready slave nodes available")
            return False
        
        print(f"🚀 Distributing training to {len(available_slaves)} slave nodes")
        
        # Distribute configuration and data
        for i, slave in enumerate(available_slaves):
            node_config = {
                'node_id': i,
                'total_nodes': len(available_slaves),
                'node_rank': i,
                'distributed_config': config,
                'master_config': self.master_config
            }
            
            # Send configuration
            try:
                data = f"CONFIG:{json.dumps(node_config)}"
                slave['connection'].send(data.encode())
                print(f"📤 Sent configuration to slave {i}")
            except Exception as e:
                print(f"❌ Error sending config to slave {i}: {e}")
        
        # Wait for all slaves to be ready
        time.sleep(2)
        ready_slaves = [s for s in self.slave_nodes if s['status'] == 'connected']
        
        if len(ready_slaves) != len(available_slaves):
            print(f"⚠️ Only {len(ready_slaves)}/{len(available_slaves)} slaves ready")
            return False
        
        print(f"🚀 All {len(ready_slaves)} slaves ready - starting distributed training")
        
        # Start training on all slaves
        for i, slave in enumerate(ready_slaves):
            try:
                job_id = f'cluster_job_{int(time.time())}'
                start_data = f"TRAINING:{json.dumps({'job_id': job_id})}"
                slave['connection'].send(start_data.encode())
                print(f"🚀 Started training on slave {i}")
            except Exception as e:
                print(f"❌ Error starting training on slave {i}: {e}")
        
        return True
    
    def collect_results(self):
        """Collect training results from all slave nodes."""
        print("📊 Collecting training results from cluster...")
        
        all_results = []
        
        for i, slave in enumerate(self.slave_nodes):
            try:
                if slave['status'] == 'completed':
                    # Get final results
                    data = f"GET_RESULTS:"
                    slave['connection'].send(data.encode())
                    
                    # Wait for response
                    response = slave['connection'].recv(4096)
                    if response:
                        results = json.loads(response.decode())
                        all_results.extend(results)
                        print(f"📊 Collected results from slave {i}")
            
            except Exception as e:
                print(f"❌ Error collecting from slave {i}: {e}")
        
        print(f"✅ Cluster training complete. Collected results from {len(all_results)} nodes")
        
        return all_results


def create_distributed_config(master_config: Dict) -> Dict:
    """Create distributed training configuration."""
    return {
        'master': {
            'host': master_config.get('host', 'localhost'),
            'port': master_config.get('port', 29500),
            'num_slaves': master_config.get('num_slaves', 2),
            'data_dir': master_config.get('data_dir', '/shared/data')
        },
        'training': {
            'sync_frequency': master_config.get('sync_frequency', 100),
            'heartbeat_timeout': master_config.get('heartbeat_timeout', 30),
            'checkpoint_dir': master_config.get('checkpoint_dir', './cluster_checkpoints')
        }
    }


def launch_distributed_training(config: Dict) -> Dict:
    """Launch distributed training job."""
    # Validate configuration
    if not config.get('distributed', False):
        return {"error": "Distributed training not enabled"}
    
    # Choose between multi-GPU and cluster mode
    if config.get('use_cluster', False):
        return _launch_cluster_training(config)
    else:
        return _launch_multi_gpu_training(config)


def _launch_multi_gpu_training(config: Dict) -> Dict:
    """Launch multi-GPU distributed training."""
    print("🚀 Launching multi-GPU distributed training")
    
    # Check available GPUs
    if not torch.cuda.is_available():
        return {"error": "CUDA not available for multi-GPU training"}
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        return {"error": f"Need at least 2 GPUs for multi-GPU training (found {num_gpus})"}
    
    distributed_trainer = DistributedTrainer(config)
    
    # Create distributed configuration
    num_nodes = config.get('num_nodes', 1)
    distributed_config = distributed_trainer.create_distributed_config(config, num_gpus, num_nodes)
    
    print(f"📊 Multi-GPU Configuration:")
    print(f"   GPUs: {num_gpus}")
    print(f"   Nodes: {num_nodes}")
    print(f"   Total processes: {distributed_config['world_size']}")
    
    return {"status": "launched", "config": distributed_config}


def _launch_cluster_training(config: Dict) -> Dict:
    """Launch cluster-based distributed training."""
    print("🚀 Launching cluster-based distributed training")
    
    # Create cluster configuration
    cluster_config = create_distributed_config(config)
    
    if 'error' in cluster_config:
        return cluster_config
    
    master_manager = ClusterManager(cluster_config['master'])
    
    # Wait for master to be ready
    time.sleep(2)
    
    # Start distributed training
    success = master_manager.distribute_training(config)
    
    if success:
        return {"status": "launched", "config": cluster_config}
    else:
        return {"status": "failed", "error": "Cluster setup failed"}


def main():
    """Command line interface for distributed training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SloGPT Distributed Training")
    subparsers = parser.add_subparsers(dest='mode', help='Training modes')
    
    # Single-GPU training (default)
    single_parser = subparsers.add_parser('single', help='Single GPU training')
    single_parser.add_argument('--dataset', required=True, help='Dataset name')
    single_parser.add_argument('--config', help='Training config file')
    single_parser.add_argument('--backend', default='nccl', choices=['nccl', 'gloo'], help='Backend for distributed training')
    single_parser.add_argument('--world_size', type=int, help='Total world size (for debugging)')
    
    # Multi-GPU training
    multi_parser = subparsers.add_parser('multi-gpu', help='Multi-GPU training')
    multi_parser.add_argument('--dataset', required=True, help='Dataset name')
    multi_parser.add_argument('--config', help='Training config file')
    multi_parser.add_argument('--num_gpus', type=int, help='Number of GPUs to use')
    
    # Cluster training
    cluster_parser = subparsers.add_parser('cluster', help='Cluster training')
    cluster_parser.add_argument('--config', required=True, help='Cluster configuration file')
    cluster_parser.add_argument('--master', action='store_true', help='Run as master node')
    cluster_parser.add_argument('--slave', action='store_true', help='Run as slave node')
    cluster_parser.add_argument('--host', default='localhost', help='Master node host')
    cluster_parser.add_argument('--port', type=int, default=29500, help='Master node port')
    cluster_parser.add_argument('--num_slaves', type=int, default=2, help='Number of slave nodes')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Single GPU training with optional distributed support
        config = {
            'dataset': args.dataset,
            'config': args.config,
            'use_distributed': False,
            'backend': args.backend,
            'world_size': args.world_size or 1
        }
        
        if args.config:
            # Load from config file
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        
        trainer = DistributedTrainer(config)
        
        print("🚀 Starting single GPU training")
        print(f"   Dataset: {config['dataset']}")
        print(f"   Backend: {config['backend']}")
        print(f"   Distributed: {config['use_distributed']}")
        
        return _launch_multi_gpu_training(config)
    
    elif args.mode == 'multi-gpu':
        # Multi-GPU distributed training
        config = {
            'dataset': args.dataset,
            'config': args.config,
            'use_distributed': True,
            'use_ddp': True,
            'backend': args.backend,
            'num_gpus': args.num_gpus
        }
        
        if args.config:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        
        return _launch_multi_gpu_training(config)
    
    elif args.mode == 'cluster':
        # Cluster-based distributed training
        if args.master:
            # Run as master node
            cluster_config = create_distributed_config({
                'host': args.host,
                'port': args.port,
                'num_slaves': args.num_slaves
            })
            
            master_manager = ClusterManager(cluster_config['master'])
            master_manager.setup_master()
            
            print("🖥 Master node running. Waiting for slave connections...")
            
            # Keep master running
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down master node")
                master_manager.cleanup_distributed()
        
        elif args.slave:
            # Run as slave node
            config = {
                'use_distributed': True,
                'backend': args.backend,
                'master_host': args.host,
                'master_port': args.port
            }
            
            trainer = DistributedTrainer(config)
            trainer.setup_distributed()
            
            print(f"🔗 Slave node connecting to master at {args.host}:{args.port}")
            
            # Connect to master and wait for configuration
            try:
                # This would be more complex in a real implementation
                print(f"📝 Slave node running for master {args.host}:{args.port}")
                time.sleep(30)  # In real implementation, would connect to master here
            except KeyboardInterrupt:
                print("\n🛑 Shutting down slave node")
                trainer.cleanup_distributed()
        
        else:
            # Load cluster config and launch as master
            with open(args.config, 'r') as f:
                cluster_config = json.load(f)
            
            return _launch_cluster_training(cluster_config)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()