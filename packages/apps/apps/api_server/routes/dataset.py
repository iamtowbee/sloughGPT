from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import time
import os
import json
from pathlib import Path

from ..core.cache_manager import CacheManager
from ..core.config import settings
from ..core.async_dataset_manager import AsyncDatasetManager, DatasetInfo
from ..dependencies import get_cache_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Global async dataset manager
async_dataset_manager = AsyncDatasetManager()

async def get_dataset_manager() -> AsyncDatasetManager:
    """Get async dataset manager dependency"""
    return async_dataset_manager

class DatasetStatus(BaseModel):
    name: str = Field(..., description="Dataset name")
    status: str = Field(..., description="Dataset status")
    size: int = Field(..., description="Dataset size in bytes")
    samples: int = Field(..., description="Number of samples")
    last_modified: float = Field(..., description="Last modification timestamp")

class DatasetUpdateRequest(BaseModel):
    dataset_name: str = Field(..., description="Dataset name")
    text: str = Field(..., min_length=1, max_length=10000, description="Text to add")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

class DatasetUpdateResponse(BaseModel):
    success: bool = Field(..., description="Update success status")
    samples_added: int = Field(..., description="Number of samples added")
    dataset_size: int = Field(..., description="New dataset size")
    timestamp: float = Field(..., description="Update timestamp")

@router.get("/status", response_model=List[DatasetStatus])
async def get_dataset_status(
    dataset_manager: AsyncDatasetManager = Depends(get_dataset_manager)
):
    """Get status of all datasets with async I/O"""
    
    try:
        await dataset_manager.initialize()
        datasets = await dataset_manager.list_datasets()
        
        # Convert to response format
        return [
            DatasetStatus(
                name=info.name,
                status="loaded",
                size=info.size,
                samples=info.samples,
                last_modified=info.last_modified
            )
            for info in datasets
        ]
        
    except Exception as e:
        logger.error(f"Failed to get dataset status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dataset status")

@router.post("/update_text", response_model=DatasetUpdateResponse)
async def update_dataset_text(
    request: DatasetUpdateRequest,
    background_tasks: BackgroundTasks,
    cache_manager: CacheManager = Depends(get_cache_manager),
    dataset_manager: AsyncDatasetManager = Depends(get_dataset_manager)
):
    """Update dataset with new text using async I/O"""
    
    try:
        await dataset_manager.initialize()
        
        # Create new sample
        new_sample = {
            "text": request.text,
            "metadata": request.metadata or {},
            "timestamp": time.time()
        }
        
        # Add sample to dataset
        success = await dataset_manager.add_sample(request.dataset_name, new_sample)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add sample to dataset")
        
        # Get updated dataset info
        dataset_info = await dataset_manager.get_dataset_info(request.dataset_name)
        dataset_size = dataset_info.size if dataset_info else 0
        
        # Clear related cache entries
        if cache_manager:
            background_tasks.add_task(clear_dataset_cache, request.dataset_name, cache_manager)
        
        return DatasetUpdateResponse(
            success=True,
            samples_added=1,
            dataset_size=dataset_size,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Failed to update dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to update dataset")

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    cache_manager: CacheManager = Depends(get_cache_manager),
    dataset_manager: AsyncDatasetManager = Depends(get_dataset_manager)
):
    """Upload a dataset file with async processing"""
    
    try:
        # Validate file type
        if not file.filename.endswith(('.json', '.jsonl', '.txt')):
            raise HTTPException(status_code=400, detail="Only JSON, JSONL, and TXT files are supported")
        
        await dataset_manager.initialize()
        
        # Read file content
        content = await file.read()
        
        # Process and validate dataset
        samples_processed = 0
        samples = []
        
        if file.filename.endswith('.json'):
            try:
                data = json.loads(content.decode('utf-8'))
                if isinstance(data, list):
                    samples = data
                    samples_processed = len(data)
                elif isinstance(data, dict) and 'samples' in data:
                    samples = data['samples']
                    samples_processed = len(samples)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")
        
        elif file.filename.endswith('.jsonl'):
            # Process JSONL format
            lines = content.decode('utf-8').strip().split('\n')
            for line in lines:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                        samples_processed += 1
                    except Exception as e:
                        logger.warning(f"Skipping invalid JSONL line: {e}")
        
        elif file.filename.endswith('.txt'):
            # Process text format (each line as a sample)
            lines = content.decode('utf-8').strip().split('\n')
            for line in lines:
                if line.strip():
                    samples.append({"text": line.strip()})
                    samples_processed += 1
        
        # Write to dataset using async manager
        dataset_name = Path(file.filename).stem
        if samples:
            await dataset_manager.write_dataset(dataset_name, samples)
        
        # Clear cache
        if cache_manager and background_tasks:
            background_tasks.add_task(clear_dataset_cache, dataset_name, cache_manager)
        
        logger.info(f"Dataset uploaded: {file.filename} ({samples_processed} samples)")
        
        return {
            "message": "Dataset uploaded successfully",
            "filename": file.filename,
            "size": len(content),
            "samples_processed": samples_processed,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload dataset")

@router.delete("/{dataset_name}")
async def delete_dataset(
    dataset_name: str,
    cache_manager: CacheManager = Depends(get_cache_manager),
    dataset_manager: AsyncDatasetManager = Depends(get_dataset_manager)
):
    """Delete a dataset using async I/O"""
    
    try:
        await dataset_manager.initialize()
        
        success = await dataset_manager.delete_dataset(dataset_name)
        
        if not success:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Clear cache
        if cache_manager:
            await clear_dataset_cache(dataset_name, cache_manager)
        
        logger.info(f"Dataset deleted: {dataset_name}")
        
        return {
            "message": "Dataset deleted successfully",
            "dataset_name": dataset_name,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete dataset")

@router.get("/{dataset_name}/samples")
async def get_dataset_samples(
    dataset_name: str,
    limit: int = 100,
    offset: int = 0,
    dataset_manager: AsyncDatasetManager = Depends(get_dataset_manager)
):
    """Get samples from a dataset with async I/O and pagination"""
    
    try:
        await dataset_manager.initialize()
        
        samples = await dataset_manager.read_dataset(dataset_name, limit=limit, offset=offset)
        dataset_info = await dataset_manager.get_dataset_info(dataset_name)
        
        # Get total count (approximate if dataset is large)
        total = 0
        if dataset_info:
            total = dataset_info.samples
        
        return {
            "dataset_name": dataset_name,
            "samples": samples,
            "offset": offset,
            "limit": limit,
            "total": total
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        logger.error(f"Failed to get dataset samples: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dataset samples")

@router.get("/{dataset_name}/stats")
async def get_dataset_stats(
    dataset_name: str,
    dataset_manager: AsyncDatasetManager = Depends(get_dataset_manager)
):
    """Get detailed dataset statistics"""
    
    try:
        await dataset_manager.initialize()
        
        stats = await dataset_manager.get_dataset_stats(dataset_name)
        
        return {
            "dataset_name": dataset_name,
            "stats": stats,
            "timestamp": time.time()
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        logger.error(f"Failed to get dataset stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dataset statistics")

async def clear_dataset_cache(dataset_name: str, cache_manager: CacheManager):
    """Clear cache entries for a dataset"""
    try:
        # Clear dataset-related cache entries
        # You'd implement specific cache clearing logic here
        await cache_manager.delete(f"dataset:{dataset_name}")
        logger.info(f"Cleared cache for dataset: {dataset_name}")
    except Exception as e:
        logger.warning(f"Failed to clear cache for dataset {dataset_name}: {e}")