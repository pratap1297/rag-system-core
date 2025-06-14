"""
Document Processor for RAG System
Orchestrates document processing pipeline
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
from enum import Enum

from ..core.logging_system import get_logger
from ..core.exceptions import FileProcessingError, ProcessingError
from ..core.error_handler import with_error_handling
from ..core.monitoring import get_performance_monitor

from .text_extractor import TextExtractor, ExtractionResult
from .chunking_strategies import ChunkingStrategy, FixedSizeChunker, TextChunk
from .metadata_extractor import MetadataExtractor, DocumentMetadata

class ProcessingStatus(Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessingResult:
    """Result of document processing"""
    document_id: str
    file_path: str
    metadata: DocumentMetadata
    chunks: List[TextChunk]
    processing_time: float
    success: bool
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class DocumentProcessor:
    """Main document processor"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("document_processor")
        self.monitor = get_performance_monitor()
        
        # Initialize components
        self.text_extractor = TextExtractor(config)
        self.metadata_extractor = MetadataExtractor(config)
        
        # Default chunking strategy
        chunking_config = self.config.get('chunking', {})
        self.default_chunker = FixedSizeChunker(
            chunk_size=chunking_config.get('chunk_size', 1000),
            overlap=chunking_config.get('overlap', 200)
        )
        
        self.logger.info("Document processor initialized")
    
    @with_error_handling("document_processor", "process_document")
    def process_document(self, file_path: str, 
                        chunking_strategy: Optional[ChunkingStrategy] = None,
                        document_id: Optional[str] = None) -> ProcessingResult:
        """Process a single document"""
        
        start_time = time.time()
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileProcessingError(f"File not found: {file_path}")
            
            # Generate document ID if not provided
            if not document_id:
                document_id = self._generate_document_id(file_path)
            
            self.logger.info(f"Processing document: {file_path} (ID: {document_id})")
            
            errors = []
            
            # Step 1: Extract text
            try:
                extraction_result = self.text_extractor.extract_text(str(file_path))
                text_content = extraction_result.text
                
                self.logger.info(f"Text extracted: {len(text_content)} characters")
                
            except Exception as e:
                error_msg = f"Text extraction failed: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                
                # Return failed result
                return ProcessingResult(
                    document_id=document_id,
                    file_path=str(file_path),
                    metadata=None,
                    chunks=[],
                    processing_time=0.0,
                    success=False,
                    errors=errors
                )
            
            # Step 2: Extract metadata
            try:
                metadata = self.metadata_extractor.extract_metadata(
                    str(file_path),
                    text_content,
                    {
                        'extraction_method': extraction_result.extraction_method,
                        'confidence': extraction_result.confidence,
                        'page_count': extraction_result.page_count,
                        'metadata': extraction_result.metadata
                    }
                )
                
                self.logger.info(f"Metadata extracted for {file_path.name}")
                
            except Exception as e:
                error_msg = f"Metadata extraction failed: {str(e)}"
                errors.append(error_msg)
                self.logger.warning(error_msg)
                
                # Create minimal metadata
                metadata = self._create_minimal_metadata(file_path, text_content)
            
            # Step 3: Chunk text
            try:
                chunker = chunking_strategy or self.default_chunker
                
                # Add document ID to metadata for chunking
                chunk_metadata = {
                    'document_id': document_id,
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'processing_date': datetime.now().isoformat()
                }
                
                chunks = chunker.chunk_text(text_content, chunk_metadata)
                
                self.logger.info(f"Text chunked into {len(chunks)} segments")
                
            except Exception as e:
                error_msg = f"Text chunking failed: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                chunks = []
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = ProcessingResult(
                document_id=document_id,
                file_path=str(file_path),
                metadata=metadata,
                chunks=chunks,
                processing_time=processing_time,
                success=len(chunks) > 0,
                errors=errors
            )
            
            # Record metrics
            if result.success:
                self.monitor.record_metric("document_processing_success", 1.0)
                self.monitor.record_metric("chunks_created", len(chunks))
            else:
                self.monitor.record_metric("document_processing_failure", 1.0)
            
            self.logger.info(f"Document processing completed: {file_path.name}, "
                           f"success={result.success}, chunks={len(chunks)}, "
                           f"time={processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.monitor.record_metric("document_processing_error", 1.0)
            
            raise ProcessingError(
                f"Document processing failed for {file_path}: {str(e)}",
                component="document_processor",
                operation="process_document",
                original_error=e
            )
    
    async def process_documents_async(self, file_paths: List[str],
                                    chunking_strategy: Optional[ChunkingStrategy] = None,
                                    max_concurrent: int = 5) -> List[ProcessingResult]:
        """Process multiple documents asynchronously"""
        
        self.logger.info(f"Starting async processing of {len(file_paths)} documents")
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(file_path: str) -> ProcessingResult:
            async with semaphore:
                # Run synchronous processing in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, 
                    self.process_document, 
                    file_path, 
                    chunking_strategy
                )
        
        # Process all documents concurrently
        tasks = [process_single(file_path) for file_path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to process {file_paths[i]}: {result}")
                
                # Create failed result
                failed_result = ProcessingResult(
                    document_id=self._generate_document_id(Path(file_paths[i])),
                    file_path=file_paths[i],
                    metadata=None,
                    chunks=[],
                    processing_time=0.0,
                    success=False,
                    errors=[str(result)]
                )
                processed_results.append(failed_result)
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.success)
        self.logger.info(f"Async processing completed: {successful}/{len(file_paths)} successful")
        
        return processed_results
    
    def process_batch(self, file_paths: List[str],
                     chunking_strategy: Optional[ChunkingStrategy] = None) -> List[ProcessingResult]:
        """Process multiple documents in batch"""
        
        self.logger.info(f"Starting batch processing of {len(file_paths)} documents")
        
        results = []
        successful = 0
        
        for i, file_path in enumerate(file_paths):
            try:
                self.logger.info(f"Processing document {i+1}/{len(file_paths)}: {file_path}")
                
                result = self.process_document(file_path, chunking_strategy)
                results.append(result)
                
                if result.success:
                    successful += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                
                # Create failed result
                failed_result = ProcessingResult(
                    document_id=self._generate_document_id(Path(file_path)),
                    file_path=file_path,
                    metadata=None,
                    chunks=[],
                    processing_time=0.0,
                    success=False,
                    errors=[str(e)]
                )
                results.append(failed_result)
        
        self.logger.info(f"Batch processing completed: {successful}/{len(file_paths)} successful")
        return results
    
    def validate_document(self, file_path: str) -> Tuple[bool, str]:
        """Validate if document can be processed"""
        
        try:
            # Check if file exists and is readable
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False, "File does not exist"
            
            if not file_path.is_file():
                return False, "Path is not a file"
            
            # Check file size
            file_size = file_path.stat().st_size
            max_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 100MB default
            
            if file_size == 0:
                return False, "File is empty"
            
            if file_size > max_size:
                return False, f"File too large: {file_size} bytes (max: {max_size})"
            
            # Validate with text extractor
            is_valid, message = self.text_extractor.validate_file(str(file_path))
            if not is_valid:
                return False, f"Text extractor validation failed: {message}"
            
            return True, "Document is valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        
        # This would typically pull from monitoring system
        # For now, return basic stats
        
        return {
            'total_processed': 0,  # Would be tracked in monitoring
            'successful': 0,
            'failed': 0,
            'average_processing_time': 0.0,
            'total_chunks_created': 0,
            'supported_formats': self.text_extractor.get_supported_formats()
        }
    
    def _generate_document_id(self, file_path: Path) -> str:
        """Generate unique document ID"""
        
        import hashlib
        
        # Create ID from file path and modification time
        file_info = f"{file_path.absolute()}_{file_path.stat().st_mtime}"
        
        return hashlib.md5(file_info.encode()).hexdigest()
    
    def _create_minimal_metadata(self, file_path: Path, text_content: str) -> DocumentMetadata:
        """Create minimal metadata when extraction fails"""
        
        from .metadata_extractor import DocumentMetadata
        from datetime import datetime
        
        file_stats = file_path.stat()
        
        return DocumentMetadata(
            file_path=str(file_path),
            file_name=file_path.name,
            file_size=file_stats.st_size,
            file_type=file_path.suffix.lower(),
            mime_type='unknown',
            created_date=datetime.fromtimestamp(file_stats.st_ctime),
            modified_date=datetime.fromtimestamp(file_stats.st_mtime),
            file_hash='',
            text_length=len(text_content) if text_content else 0,
            processing_date=datetime.now(),
            extraction_method='minimal',
            extraction_confidence=0.5
        ) 