import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict
import torch

from .config import settings

logger = logging.getLogger(__name__)

class BatchPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class BatchRequest:
    """Individual request within a batch"""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    future: asyncio.Future
    priority: BatchPriority = BatchPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Batch:
    """Collection of requests to process together"""
    batch_id: str
    requests: List[BatchRequest]
    created_at: float = field(default_factory=time.time)
    max_wait_time: float = settings.BATCH_TIMEOUT
    
    def is_full(self, max_size: int) -> bool:
        """Check if batch is at maximum capacity"""
        return len(self.requests) >= max_size
    
    def is_expired(self) -> bool:
        """Check if batch has exceeded maximum wait time"""
        return time.time() - self.created_at > self.max_wait_time
    
    def add_request(self, request: BatchRequest) -> bool:
        """Add request to batch if not full"""
        if not self.is_full(settings.BATCH_SIZE):
            self.requests.append(request)
            return True
        return False
    
    def get_priority_score(self) -> float:
        """Calculate batch priority score for processing order"""
        if not self.requests:
            return 0.0
        
        # Weight by priority and wait time
        priority_weight = sum(req.priority.value for req in self.requests) / len(self.requests)
        wait_weight = min((time.time() - self.created_at) / self.max_wait_time, 1.0)
        
        return priority_weight * 0.7 + wait_weight * 0.3

class BatchProcessor:
    """Advanced batch processor with priority queuing and dynamic batching"""
    
    def __init__(self, model_manager, max_batch_size: int = None, 
                 max_wait_time: float = None):
        self.model_manager = model_manager
        self.max_batch_size = max_batch_size or settings.BATCH_SIZE
        self.max_wait_time = max_wait_time or settings.BATCH_TIMEOUT
        
        # Batch queues by priority
        self.pending_requests: Dict[BatchPriority, List[BatchRequest]] = {
            priority: [] for priority in BatchPriority
        }
        
        # Active batches
        self.active_batches: List[Batch] = []
        
        # Processing state
        self.is_running = False
        self.processor_task = None
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time": 0.0,
            "priority_processed": defaultdict(int),
            "timeout_processed": 0,
            "full_batches": 0
        }
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the batch processor"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processor_task = asyncio.create_task(self._batch_processor_loop())
        logger.info("Batch processor started")
    
    async def stop(self):
        """Stop the batch processor"""
        self.is_running = False
        
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining requests
        await self._process_remaining_requests()
        logger.info("Batch processor stopped")
    
    async def add_request(self, prompt: str, max_tokens: int, temperature: float, 
                         top_p: float, priority: BatchPriority = BatchPriority.NORMAL,
                         metadata: Dict[str, Any] = None) -> str:
        """Add request to batch queue"""
        
        request_id = str(uuid.uuid4())
        request = BatchRequest(
            request_id=request_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            future=asyncio.Future(),
            priority=priority,
            metadata=metadata or {}
        )
        
        async with self._lock:
            self.pending_requests[priority].append(request)
            self.stats["total_requests"] += 1
        
        logger.debug(f"Added request {request_id} with priority {priority.name}")
        return request_id
    
    async def _batch_processor_loop(self):
        """Main batch processing loop"""
        while self.is_running:
            try:
                # Collect and process batches
                await self._collect_batches()
                await self._process_ready_batches()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)
    
    async def _collect_batches(self):
        """Collect requests into batches based on priority and timing"""
        async with self._lock:
            current_time = time.time()
            
            # Process priorities in order (highest first)
            for priority in sorted(BatchPriority, key=lambda p: p.value, reverse=True):
                requests = self.pending_requests[priority]
                
                while requests:
                    # Check if we should create a new batch
                    if (len(requests) >= self.max_batch_size or 
                        (requests and current_time - requests[0].created_at > self.max_wait_time)):
                        
                        # Create new batch
                        batch_size = min(len(requests), self.max_batch_size)
                        batch_requests = requests[:batch_size]
                        
                        batch = Batch(
                            batch_id=str(uuid.uuid4()),
                            requests=batch_requests,
                            max_wait_time=self.max_wait_time
                        )
                        
                        self.active_batches.append(batch)
                        
                        # Remove requests from queue
                        self.pending_requests[priority] = requests[batch_size:]
                        requests = self.pending_requests[priority]
                        
                        logger.debug(f"Created batch {batch.batch_id} with {len(batch_requests)} requests")
                    else:
                        break
    
    async def _process_ready_batches(self):
        """Process batches that are ready"""
        if not self.active_batches:
            return
        
        # Sort batches by priority score
        ready_batches = [
            batch for batch in self.active_batches 
            if batch.is_full(self.max_batch_size) or batch.is_expired()
        ]
        
        if not ready_batches:
            return
        
        # Sort by priority (highest first)
        ready_batches.sort(key=lambda b: b.get_priority_score(), reverse=True)
        
        # Process batches
        for batch in ready_batches:
            try:
                await self._process_batch(batch)
                self.active_batches.remove(batch)
                
                # Update statistics
                self.stats["total_batches"] += 1
                self.stats["avg_batch_size"] = (
                    (self.stats["avg_batch_size"] * (self.stats["total_batches"] - 1) + len(batch.requests)) /
                    self.stats["total_batches"]
                )
                
                if batch.is_full(self.max_batch_size):
                    self.stats["full_batches"] += 1
                if batch.is_expired():
                    self.stats["timeout_processed"] += 1
                
                # Update priority statistics
                for request in batch.requests:
                    self.stats["priority_processed"][request.priority.name] += 1
                    wait_time = time.time() - request.created_at
                    self.stats["avg_wait_time"] = (
                        (self.stats["avg_wait_time"] * (self.stats["total_requests"] - 1) + wait_time) /
                        self.stats["total_requests"]
                    )
                
            except Exception as e:
                logger.error(f"Error processing batch {batch.batch_id}: {e}")
                # Fail all requests in batch
                for request in batch.requests:
                    if not request.future.done():
                        request.future.set_exception(e)
                
                self.active_batches.remove(batch)
    
    async def _process_batch(self, batch: Batch):
        """Process a single batch of requests"""
        if not batch.requests:
            return
        
        try:
            # Prepare batch data
            prompts = [req.prompt for req in batch.requests]
            max_tokens_list = [req.max_tokens for req in batch.requests]
            
            # Tokenize all prompts (you'll need to implement proper tokenizer)
            input_ids_batch = []
            for prompt in prompts:
                input_ids = self.model_manager._encode(prompt)
                input_ids_batch.append(input_ids)
            
            # Pad to same length
            max_len = max(len(ids) for ids in input_ids_batch)
            padded_batch = []
            attention_masks = []
            
            for ids in input_ids_batch:
                padding = [0] * (max_len - len(ids))
                padded = ids + padding
                attention_mask = [1] * len(ids) + padding
                
                padded_batch.append(padded)
                attention_masks.append(attention_mask)
            
            # Generate in batch
            with torch.no_grad():
                input_tensor = torch.tensor(padded_batch).to(self.model_manager.device)
                attention_tensor = torch.tensor(attention_masks).to(self.model_manager.device)
                
                # Use the first request's parameters for the batch
                # In production, you might want to group by similar parameters
                first_request = batch.requests[0]
                
                output_batch = self.model_manager.model.generate(
                    input_tensor,
                    attention_mask=attention_tensor,
                    max_new_tokens=max(max_tokens_list),
                    temperature=first_request.temperature,
                    top_p=first_request.top_p,
                    do_sample=True,
                    pad_token_id=0  # Add proper padding token
                )
            
            # Process results
            for i, request in enumerate(batch.requests):
                try:
                    original_length = len(input_ids_batch[i])
                    generated_ids = output_batch[i][original_length:]
                    
                    # Remove padding tokens
                    generated_ids = generated_ids[generated_ids != 0]
                    
                    # Decode response
                    result = self.model_manager._decode(generated_ids)
                    
                    # Set result for request
                    if not request.future.done():
                        request.future.set_result(result)
                        
                except Exception as e:
                    logger.error(f"Error processing request {request.request_id}: {e}")
                    if not request.future.done():
                        request.future.set_exception(e)
            
            logger.debug(f"Processed batch {batch.batch_id} with {len(batch.requests)} requests")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fail all requests in batch
            for request in batch.requests:
                if not request.future.done():
                    request.future.set_exception(e)
    
    async def _process_remaining_requests(self):
        """Process any remaining requests when shutting down"""
        async with self._lock:
            all_requests = []
            for priority_requests in self.pending_requests.values():
                all_requests.extend(priority_requests)
            
            if all_requests:
                logger.info(f"Processing {len(all_requests)} remaining requests during shutdown")
                
                # Create final batch
                final_batch = Batch(
                    batch_id="shutdown",
                    requests=all_requests
                )
                
                await self._process_batch(final_batch)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        async with self._lock:
            return {
                "is_running": self.is_running,
                "pending_requests": {
                    priority.name: len(requests) 
                    for priority, requests in self.pending_requests.items()
                },
                "active_batches": len(self.active_batches),
                "statistics": self.stats.copy(),
                "configuration": {
                    "max_batch_size": self.max_batch_size,
                    "max_wait_time": self.max_wait_time
                }
            }
    
    async def reset_stats(self):
        """Reset batch processing statistics"""
        async with self._lock:
            self.stats = {
                "total_requests": 0,
                "total_batches": 0,
                "avg_batch_size": 0.0,
                "avg_wait_time": 0.0,
                "priority_processed": defaultdict(int),
                "timeout_processed": 0,
                "full_batches": 0
            }