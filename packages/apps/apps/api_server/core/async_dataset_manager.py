import asyncio
import aiofiles
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
import aiofiles.os
from contextlib import asynccontextmanager
import weakref
from dataclasses import dataclass

from .config import settings

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    name: str
    path: Path
    size: int
    samples: int
    last_modified: float
    file_type: str

class AsyncFileManager:
    """Async file manager with connection pooling and caching"""
    
    def __init__(self, max_open_files: int = 50):
        self.max_open_files = max_open_files
        self._open_files: Dict[str, aiofiles.threadpool.AsyncFileIO] = {}
        self._file_refs: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        
    @asynccontextmanager
    async def open_file(self, file_path: Path, mode: str = 'r') -> AsyncGenerator[aiofiles.threadpool.AsyncFileIO, None]:
        """Open file with connection pooling"""
        file_key = str(file_path)
        
        async with self._lock:
            # Check if file is already open
            if file_key in self._open_files:
                self._file_refs[file_key] += 1
                file_handle = self._open_files[file_key]
            else:
                # Close oldest files if at limit
                if len(self._open_files) >= self.max_open_files:
                    await self._close_oldest_file()
                
                # Open new file
                file_handle = await aiofiles.open(file_path, mode)
                self._open_files[file_key] = file_handle
                self._file_refs[file_key] = 1
        
        try:
            yield file_handle
        finally:
            async with self._lock:
                self._file_refs[file_key] -= 1
                if self._file_refs[file_key] <= 0:
                    await self._close_file(file_key)
    
    async def _close_oldest_file(self):
        """Close the least recently used file"""
        if not self._open_files:
            return
        
        oldest_key = next(iter(self._open_files))
        await self._close_file(oldest_key)
    
    async def _close_file(self, file_key: str):
        """Close a specific file"""
        if file_key in self._open_files:
            try:
                await self._open_files[file_key].close()
            except Exception as e:
                logger.warning(f"Error closing file {file_key}: {e}")
            finally:
                del self._open_files[file_key]
                del self._file_refs[file_key]
    
    async def close_all(self):
        """Close all open files"""
        async with self._lock:
            for file_key in list(self._open_files.keys()):
                await self._close_file(file_key)

class AsyncDatasetManager:
    """Async dataset manager with optimized I/O operations"""
    
    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or Path(settings.DATASET_PATH)
        self.file_manager = AsyncFileManager()
        self._dataset_cache: Dict[str, DatasetInfo] = {}
        self._cache_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize dataset manager"""
        await self.dataset_path.mkdir(parents=True, exist_ok=True)
        await self._refresh_dataset_cache()
        
    async def list_datasets(self) -> List[DatasetInfo]:
        """List all datasets with metadata"""
        async with self._cache_lock:
            return list(self._dataset_cache.values())
    
    async def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get specific dataset information"""
        async with self._cache_lock:
            return self._dataset_cache.get(dataset_name)
    
    async def read_dataset(self, dataset_name: str, limit: Optional[int] = None, 
                          offset: int = 0) -> List[Dict[str, Any]]:
        """Read dataset with async I/O and pagination"""
        dataset_file = self.dataset_path / f"{dataset_name}.json"
        
        if not await aiofiles.os.path.exists(dataset_file):
            raise FileNotFoundError(f"Dataset {dataset_name} not found")
        
        try:
            async with self.file_manager.open_file(dataset_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                
                # Handle different dataset formats
                if isinstance(data, list):
                    samples = data
                elif isinstance(data, dict) and 'samples' in data:
                    samples = data['samples']
                else:
                    samples = []
                
                # Apply pagination
                if offset > 0:
                    samples = samples[offset:]
                if limit is not None:
                    samples = samples[:limit]
                
                return samples
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dataset {dataset_name}: {e}")
            raise ValueError(f"Invalid JSON format in dataset {dataset_name}")
        except Exception as e:
            logger.error(f"Error reading dataset {dataset_name}: {e}")
            raise
    
    async def write_dataset(self, dataset_name: str, data: List[Dict[str, Any]], 
                           append: bool = False) -> int:
        """Write dataset with async I/O"""
        dataset_file = self.dataset_path / f"{dataset_name}.json"
        
        try:
            if append and await aiofiles.os.path.exists(dataset_file):
                # Read existing data and append
                existing_data = await self.read_dataset(dataset_name)
                combined_data = existing_data + data
            else:
                combined_data = data
            
            # Write with atomic operation
            temp_file = dataset_file.with_suffix('.tmp')
            
            async with self.file_manager.open_file(temp_file, 'w') as f:
                content = json.dumps(combined_data, indent=2, ensure_ascii=False)
                await f.write(content)
            
            # Atomic rename
            await aiofiles.os.rename(temp_file, dataset_file)
            
            # Update cache
            await self._update_dataset_cache(dataset_name)
            
            logger.info(f"Dataset {dataset_name} written with {len(combined_data)} samples")
            return len(combined_data)
            
        except Exception as e:
            logger.error(f"Error writing dataset {dataset_name}: {e}")
            # Clean up temp file if it exists
            temp_file = dataset_file.with_suffix('.tmp')
            if await aiofiles.os.path.exists(temp_file):
                await aiofiles.os.remove(temp_file)
            raise
    
    async def add_sample(self, dataset_name: str, sample: Dict[str, Any]) -> bool:
        """Add single sample to dataset"""
        try:
            await self.write_dataset(dataset_name, [sample], append=True)
            return True
        except Exception as e:
            logger.error(f"Error adding sample to {dataset_name}: {e}")
            return False
    
    async def delete_dataset(self, dataset_name: str) -> bool:
        """Delete dataset file"""
        dataset_file = self.dataset_path / f"{dataset_name}.json"
        
        try:
            if await aiofiles.os.path.exists(dataset_file):
                await aiofiles.os.remove(dataset_file)
                
                # Remove from cache
                async with self._cache_lock:
                    self._dataset_cache.pop(dataset_name, None)
                
                logger.info(f"Dataset {dataset_name} deleted")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting dataset {dataset_name}: {e}")
            return False
    
    async def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get detailed dataset statistics"""
        try:
            samples = await self.read_dataset(dataset_name, limit=1000)  # Sample for stats
            
            if not samples:
                return {"total_samples": 0, "avg_text_length": 0}
            
            # Calculate statistics
            text_lengths = []
            for sample in samples:
                if 'text' in sample:
                    text_lengths.append(len(sample['text']))
            
            avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
            
            return {
                "total_samples": len(samples),
                "avg_text_length": avg_length,
                "min_text_length": min(text_lengths) if text_lengths else 0,
                "max_text_length": max(text_lengths) if text_lengths else 0,
                "sample_keys": list(samples[0].keys()) if samples else []
            }
            
        except Exception as e:
            logger.error(f"Error getting stats for {dataset_name}: {e}")
            return {"error": str(e)}
    
    async def _refresh_dataset_cache(self):
        """Refresh dataset information cache"""
        async with self._cache_lock:
            self._dataset_cache.clear()
            
            if not await aiofiles.os.path.exists(self.dataset_path):
                return
            
            async for file_path in self._async_glob(self.dataset_path, "*.json"):
                try:
                    stat = await aiofiles.os.stat(file_path)
                    dataset_name = file_path.stem
                    
                    # Quick sample count
                    samples = 0
                    try:
                        sample_data = await self.read_dataset(dataset_name, limit=10)
                        samples = len(sample_data)
                    except Exception:
                        pass
                    
                    info = DatasetInfo(
                        name=dataset_name,
                        path=file_path,
                        size=stat.st_size,
                        samples=samples,
                        last_modified=stat.st_mtime,
                        file_type=file_path.suffix
                    )
                    
                    self._dataset_cache[dataset_name] = info
                    
                except Exception as e:
                    logger.warning(f"Error caching dataset {file_path}: {e}")
    
    async def _update_dataset_cache(self, dataset_name: str):
        """Update specific dataset cache entry"""
        dataset_file = self.dataset_path / f"{dataset_name}.json"
        
        if await aiofiles.os.path.exists(dataset_file):
            try:
                stat = await aiofiles.os.stat(dataset_file)
                samples = await self.read_dataset(dataset_name, limit=10)
                
                info = DatasetInfo(
                    name=dataset_name,
                    path=dataset_file,
                    size=stat.st_size,
                    samples=len(samples),
                    last_modified=stat.st_mtime,
                    file_type=dataset_file.suffix
                )
                
                async with self._cache_lock:
                    self._dataset_cache[dataset_name] = info
                    
            except Exception as e:
                logger.warning(f"Error updating cache for {dataset_name}: {e}")
    
    async def _async_glob(self, path: Path, pattern: str) -> AsyncGenerator[Path, None]:
        """Async glob implementation"""
        loop = asyncio.get_event_loop()
        
        def sync_glob():
            return path.glob(pattern)
        
        files = await loop.run_in_executor(None, sync_glob)
        for file_path in files:
            if file_path.is_file():
                yield file_path
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.file_manager.close_all()
        logger.info("Async dataset manager cleaned up")