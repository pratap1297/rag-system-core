"""
Processing Pipeline for RAG System
Manages the complete document processing workflow
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from core.logging_system import get_logger
from core.exceptions import ProcessingError, ConfigurationError
from core.error_handler import with_error_handling
from core.monitoring import get_performance_monitor

from .document_processor import DocumentProcessor, ProcessingResult
from .chunking_strategies import ChunkingStrategy, FixedSizeChunker, SentenceChunker

class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # Input settings
    input_directory: Optional[str] = None
    file_patterns: List[str] = None
    recursive: bool = True
    
    # Processing settings
    chunking_strategy: str = "fixed_size"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_concurrent: int = 5
    
    # Output settings
    output_directory: Optional[str] = None
    save_metadata: bool = True
    save_chunks: bool = True
    
    # Quality settings
    min_chunk_size: int = 50
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    def __post_init__(self):
        if self.file_patterns is None:
            self.file_patterns = ['*.pdf', '*.docx', '*.doc', '*.txt']

@dataclass
class PipelineResult:
    """Pipeline execution result"""
    pipeline_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime]
    total_files: int
    processed_files: int
    successful_files: int
    failed_files: int
    total_chunks: int
    processing_results: List[ProcessingResult]
    errors: List[str]
    
    @property
    def duration(self) -> float:
        """Get pipeline duration in seconds"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100

class ProcessingPipeline:
    """Document processing pipeline"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("processing_pipeline")
        self.monitor = get_performance_monitor()
        
        # Initialize document processor
        self.document_processor = DocumentProcessor(config)
        
        # Pipeline state
        self.current_pipeline_id = None
        self.is_running = False
        self.should_cancel = False
        
        # Callbacks
        self.progress_callback: Optional[Callable] = None
        self.completion_callback: Optional[Callable] = None
        
        self.logger.info("Processing pipeline initialized")
    
    @with_error_handling("processing_pipeline", "run_pipeline")
    def run_pipeline(self, pipeline_config: Union[PipelineConfig, Dict[str, Any]]) -> PipelineResult:
        """Run the processing pipeline"""
        
        # Convert dict to PipelineConfig if needed
        if isinstance(pipeline_config, dict):
            pipeline_config = PipelineConfig(**pipeline_config)
        
        # Generate pipeline ID
        pipeline_id = self._generate_pipeline_id()
        self.current_pipeline_id = pipeline_id
        
        self.logger.info(f"Starting pipeline {pipeline_id}")
        
        # Initialize result
        result = PipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.PENDING,
            start_time=datetime.now(),
            end_time=None,
            total_files=0,
            processed_files=0,
            successful_files=0,
            failed_files=0,
            total_chunks=0,
            processing_results=[],
            errors=[]
        )
        
        try:
            self.is_running = True
            self.should_cancel = False
            result.status = PipelineStatus.RUNNING
            
            # Step 1: Discover files
            self.logger.info("Discovering files...")
            file_paths = self._discover_files(pipeline_config)
            result.total_files = len(file_paths)
            
            if not file_paths:
                raise ProcessingError("No files found to process")
            
            self.logger.info(f"Found {len(file_paths)} files to process")
            
            # Step 2: Validate files
            self.logger.info("Validating files...")
            valid_files = self._validate_files(file_paths, pipeline_config)
            
            if not valid_files:
                raise ProcessingError("No valid files found")
            
            self.logger.info(f"Validated {len(valid_files)} files")
            
            # Step 3: Create chunking strategy
            chunking_strategy = self._create_chunking_strategy(pipeline_config)
            
            # Step 4: Process files
            self.logger.info("Processing files...")
            processing_results = self._process_files(valid_files, chunking_strategy, pipeline_config)
            
            # Step 5: Collect results
            result.processing_results = processing_results
            result.processed_files = len(processing_results)
            result.successful_files = sum(1 for r in processing_results if r.success)
            result.failed_files = result.processed_files - result.successful_files
            result.total_chunks = sum(len(r.chunks) for r in processing_results if r.success)
            
            # Collect errors
            for proc_result in processing_results:
                if proc_result.errors:
                    result.errors.extend(proc_result.errors)
            
            # Step 6: Save results if configured
            if pipeline_config.output_directory:
                self._save_results(processing_results, pipeline_config)
            
            # Mark as completed
            result.status = PipelineStatus.COMPLETED if result.successful_files > 0 else PipelineStatus.FAILED
            result.end_time = datetime.now()
            
            self.logger.info(f"Pipeline {pipeline_id} completed: "
                           f"{result.successful_files}/{result.total_files} successful, "
                           f"{result.total_chunks} chunks created, "
                           f"duration: {result.duration:.2f}s")
            
            # Call completion callback
            if self.completion_callback:
                try:
                    self.completion_callback(result)
                except Exception as e:
                    self.logger.warning(f"Completion callback failed: {e}")
            
            return result
            
        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.end_time = datetime.now()
            result.errors.append(str(e))
            
            self.logger.error(f"Pipeline {pipeline_id} failed: {e}")
            raise ProcessingError(f"Pipeline execution failed: {str(e)}")
            
        finally:
            self.is_running = False
            self.current_pipeline_id = None
    
    async def run_pipeline_async(self, pipeline_config: Union[PipelineConfig, Dict[str, Any]]) -> PipelineResult:
        """Run the processing pipeline asynchronously"""
        
        # Convert dict to PipelineConfig if needed
        if isinstance(pipeline_config, dict):
            pipeline_config = PipelineConfig(**pipeline_config)
        
        # Generate pipeline ID
        pipeline_id = self._generate_pipeline_id()
        self.current_pipeline_id = pipeline_id
        
        self.logger.info(f"Starting async pipeline {pipeline_id}")
        
        # Initialize result
        result = PipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.PENDING,
            start_time=datetime.now(),
            end_time=None,
            total_files=0,
            processed_files=0,
            successful_files=0,
            failed_files=0,
            total_chunks=0,
            processing_results=[],
            errors=[]
        )
        
        try:
            self.is_running = True
            self.should_cancel = False
            result.status = PipelineStatus.RUNNING
            
            # Step 1: Discover files
            file_paths = await self._discover_files_async(pipeline_config)
            result.total_files = len(file_paths)
            
            if not file_paths:
                raise ProcessingError("No files found to process")
            
            # Step 2: Validate files
            valid_files = await self._validate_files_async(file_paths, pipeline_config)
            
            if not valid_files:
                raise ProcessingError("No valid files found")
            
            # Step 3: Create chunking strategy
            chunking_strategy = self._create_chunking_strategy(pipeline_config)
            
            # Step 4: Process files asynchronously
            processing_results = await self.document_processor.process_documents_async(
                valid_files,
                chunking_strategy,
                pipeline_config.max_concurrent
            )
            
            # Step 5: Collect results
            result.processing_results = processing_results
            result.processed_files = len(processing_results)
            result.successful_files = sum(1 for r in processing_results if r.success)
            result.failed_files = result.processed_files - result.successful_files
            result.total_chunks = sum(len(r.chunks) for r in processing_results if r.success)
            
            # Collect errors
            for proc_result in processing_results:
                if proc_result.errors:
                    result.errors.extend(proc_result.errors)
            
            # Step 6: Save results if configured
            if pipeline_config.output_directory:
                await self._save_results_async(processing_results, pipeline_config)
            
            # Mark as completed
            result.status = PipelineStatus.COMPLETED if result.successful_files > 0 else PipelineStatus.FAILED
            result.end_time = datetime.now()
            
            self.logger.info(f"Async pipeline {pipeline_id} completed: "
                           f"{result.successful_files}/{result.total_files} successful")
            
            return result
            
        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.end_time = datetime.now()
            result.errors.append(str(e))
            
            self.logger.error(f"Async pipeline {pipeline_id} failed: {e}")
            raise ProcessingError(f"Async pipeline execution failed: {str(e)}")
            
        finally:
            self.is_running = False
            self.current_pipeline_id = None
    
    def cancel_pipeline(self):
        """Cancel the currently running pipeline"""
        
        if self.is_running:
            self.should_cancel = True
            self.logger.info(f"Cancellation requested for pipeline {self.current_pipeline_id}")
        else:
            self.logger.warning("No pipeline is currently running")
    
    def set_progress_callback(self, callback: Callable):
        """Set progress callback function"""
        self.progress_callback = callback
    
    def set_completion_callback(self, callback: Callable):
        """Set completion callback function"""
        self.completion_callback = callback
    
    def _discover_files(self, config: PipelineConfig) -> List[str]:
        """Discover files to process"""
        
        if not config.input_directory:
            raise ConfigurationError("Input directory not specified")
        
        input_path = Path(config.input_directory)
        
        if not input_path.exists():
            raise ConfigurationError(f"Input directory does not exist: {input_path}")
        
        files = []
        
        for pattern in config.file_patterns:
            if config.recursive:
                pattern_files = list(input_path.rglob(pattern))
            else:
                pattern_files = list(input_path.glob(pattern))
            
            files.extend([str(f) for f in pattern_files if f.is_file()])
        
        # Remove duplicates and sort
        files = sorted(list(set(files)))
        
        return files
    
    async def _discover_files_async(self, config: PipelineConfig) -> List[str]:
        """Discover files asynchronously"""
        
        # For now, just run synchronous version in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._discover_files, config)
    
    def _validate_files(self, file_paths: List[str], config: PipelineConfig) -> List[str]:
        """Validate files for processing"""
        
        valid_files = []
        
        for file_path in file_paths:
            if self.should_cancel:
                break
            
            try:
                is_valid, message = self.document_processor.validate_document(file_path)
                
                if is_valid:
                    valid_files.append(file_path)
                else:
                    self.logger.warning(f"Invalid file {file_path}: {message}")
                    
            except Exception as e:
                self.logger.warning(f"Validation error for {file_path}: {e}")
        
        return valid_files
    
    async def _validate_files_async(self, file_paths: List[str], config: PipelineConfig) -> List[str]:
        """Validate files asynchronously"""
        
        # For now, just run synchronous version in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._validate_files, file_paths, config)
    
    def _create_chunking_strategy(self, config: PipelineConfig) -> ChunkingStrategy:
        """Create chunking strategy from config"""
        
        strategy_name = config.chunking_strategy.lower()
        
        if strategy_name == "fixed_size":
            return FixedSizeChunker(
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap
            )
        elif strategy_name == "sentence":
            return SentenceChunker(
                max_sentences=config.chunk_size // 100  # Rough estimate
            )
        else:
            self.logger.warning(f"Unknown chunking strategy: {strategy_name}, using fixed_size")
            return FixedSizeChunker(
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap
            )
    
    def _process_files(self, file_paths: List[str], chunking_strategy: ChunkingStrategy,
                      config: PipelineConfig) -> List[ProcessingResult]:
        """Process files synchronously"""
        
        results = []
        
        for i, file_path in enumerate(file_paths):
            if self.should_cancel:
                self.logger.info("Processing cancelled by user")
                break
            
            try:
                # Report progress
                if self.progress_callback:
                    try:
                        self.progress_callback(i + 1, len(file_paths), file_path)
                    except Exception as e:
                        self.logger.warning(f"Progress callback failed: {e}")
                
                # Process document
                result = self.document_processor.process_document(file_path, chunking_strategy)
                results.append(result)
                
                self.logger.info(f"Processed {i+1}/{len(file_paths)}: {Path(file_path).name}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                
                # Create failed result
                failed_result = ProcessingResult(
                    document_id=f"failed_{i}",
                    file_path=file_path,
                    metadata=None,
                    chunks=[],
                    processing_time=0.0,
                    success=False,
                    errors=[str(e)]
                )
                results.append(failed_result)
        
        return results
    
    def _save_results(self, results: List[ProcessingResult], config: PipelineConfig):
        """Save processing results to disk"""
        
        if not config.output_directory:
            return
        
        output_path = Path(config.output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary = {
            'pipeline_id': self.current_pipeline_id,
            'timestamp': datetime.now().isoformat(),
            'total_files': len(results),
            'successful_files': sum(1 for r in results if r.success),
            'total_chunks': sum(len(r.chunks) for r in results if r.success),
            'files': []
        }
        
        for result in results:
            file_info = {
                'document_id': result.document_id,
                'file_path': result.file_path,
                'success': result.success,
                'chunk_count': len(result.chunks),
                'processing_time': result.processing_time,
                'errors': result.errors
            }
            summary['files'].append(file_info)
        
        # Save summary as JSON
        import json
        summary_path = output_path / f"pipeline_summary_{self.current_pipeline_id}.json"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Pipeline results saved to {output_path}")
    
    async def _save_results_async(self, results: List[ProcessingResult], config: PipelineConfig):
        """Save results asynchronously"""
        
        # For now, just run synchronous version in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_results, results, config)
    
    def _generate_pipeline_id(self) -> str:
        """Generate unique pipeline ID"""
        
        import uuid
        return f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        
        return {
            'is_running': self.is_running,
            'current_pipeline_id': self.current_pipeline_id,
            'should_cancel': self.should_cancel
        } 