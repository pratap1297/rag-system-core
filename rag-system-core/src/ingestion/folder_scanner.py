"""
Phase 5.1: Folder Scanner Module for RAG System
Enterprise-grade folder monitoring and document ingestion with intelligent change detection
"""

import asyncio
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
import logging

# Core imports with fallback
try:
    from ..core.logging_system import get_logger
    from ..core.exceptions import ProcessingError, ConfigurationError
    from ..core.error_handler import with_error_handling
    from ..core.monitoring import get_performance_monitor
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)
    
    class ProcessingError(Exception):
        pass
    
    class ConfigurationError(Exception):
        pass
    
    def with_error_handling(module, operation):
        def decorator(func):
            return func
        return decorator
    
    def get_performance_monitor():
        return None


class FileStatus(Enum):
    """File processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    DELETED = "deleted"


class Priority(Enum):
    """File processing priority"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ChangeType(Enum):
    """Type of file system change"""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileMetadata:
    """Comprehensive file metadata"""
    file_path: str
    relative_path: str
    filename: str
    extension: str
    size: int
    modified_time: float
    created_time: float
    content_hash: Optional[str] = None
    
    # Organizational metadata
    site_id: Optional[str] = None
    category: Optional[str] = None
    department: Optional[str] = None
    document_type: Optional[str] = None
    
    # Processing metadata
    priority: Priority = Priority.NORMAL
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileState:
    """Complete file state tracking"""
    metadata: FileMetadata
    status: FileStatus = FileStatus.PENDING
    processing_attempts: int = 0
    last_attempt: Optional[datetime] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    ingestion_id: Optional[str] = None
    
    # Change tracking
    change_type: Optional[ChangeType] = None
    detected_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None


@dataclass
class ScannerConfig:
    """Folder scanner configuration"""
    monitored_directories: List[str] = field(default_factory=list)
    scan_interval: int = 60  # seconds
    max_depth: int = 10
    enable_content_hashing: bool = True
    supported_extensions: Set[str] = field(default_factory=lambda: {
        '.pdf', '.txt', '.docx', '.doc', '.md', '.json', '.csv', '.xlsx', '.pptx'
    })
    
    # File filtering
    max_file_size_mb: int = 100
    min_file_size_bytes: int = 1
    exclude_patterns: List[str] = field(default_factory=lambda: [
        '.*', '__pycache__', '*.tmp', '*.log', '*.bak'
    ])
    
    # Processing configuration
    max_concurrent_files: int = 5
    retry_attempts: int = 3
    retry_delay: int = 60  # seconds
    processing_timeout: int = 300  # seconds
    
    # Metadata extraction rules
    path_metadata_rules: Dict[str, Dict[str, str]] = field(default_factory=dict)
    auto_categorization: bool = True
    
    # Performance settings
    enable_parallel_scanning: bool = True
    scan_batch_size: int = 100
    memory_limit_mb: int = 500


@dataclass
class ScannerStats:
    """Scanner performance statistics"""
    total_files_tracked: int = 0
    files_pending: int = 0
    files_processing: int = 0
    files_successful: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    
    total_scans: int = 0
    last_scan_time: Optional[datetime] = None
    last_scan_duration: float = 0.0
    avg_scan_duration: float = 0.0
    
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    
    # Error tracking
    scan_errors: int = 0
    processing_errors: int = 0
    last_error: Optional[str] = None
    
    # Performance metrics
    files_per_second: float = 0.0
    memory_usage_mb: float = 0.0


class FolderScanner:
    """
    Enterprise Folder Scanner Module - Phase 5.1
    
    Provides comprehensive folder monitoring with:
    - Intelligent change detection
    - Metadata extraction and enrichment
    - Priority-based processing queues
    - Error handling and retry logic
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: ScannerConfig, processing_callback: Optional[Callable] = None):
        self.config = config
        self.processing_callback = processing_callback
        self.logger = get_logger(__name__)
        self.monitor = get_performance_monitor()
        
        # State management
        self.file_states: Dict[str, FileState] = {}
        self.processing_queue: List[str] = []
        self.is_running = False
        self.is_scanning = False
        
        # Threading
        self._lock = threading.RLock()
        self._scan_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=config.max_concurrent_files)
        
        # Statistics
        self.stats = ScannerStats()
        
        # Validation
        self._validate_config()
        
        self.logger.info(f"Folder scanner initialized with {len(config.monitored_directories)} directories")
    
    def _validate_config(self):
        """Validate scanner configuration"""
        if not self.config.monitored_directories:
            raise ConfigurationError("No monitored directories specified")
        
        for directory in self.config.monitored_directories:
            path = Path(directory)
            if not path.exists():
                self.logger.warning(f"Monitored directory does not exist: {directory}")
            elif not path.is_dir():
                raise ConfigurationError(f"Path is not a directory: {directory}")
        
        if self.config.scan_interval < 10:
            self.logger.warning("Scan interval is very low, may impact performance")
        
        if self.config.max_concurrent_files > 20:
            self.logger.warning("High concurrent file limit may impact system performance")
    
    @with_error_handling("folder_scanner", "start_monitoring")
    def start_monitoring(self) -> bool:
        """Start folder monitoring"""
        if self.is_running:
            self.logger.warning("Scanner is already running")
            return False
        
        self.is_running = True
        
        # Start scanning thread
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        
        # Start processing thread
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        
        self.logger.info("Folder monitoring started")
        return True
    
    @with_error_handling("folder_scanner", "stop_monitoring")
    def stop_monitoring(self) -> bool:
        """Stop folder monitoring"""
        if not self.is_running:
            return False
        
        self.is_running = False
        
        # Wait for threads to finish
        if self._scan_thread and self._scan_thread.is_alive():
            self._scan_thread.join(timeout=10)
        
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=10)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        self.logger.info("Folder monitoring stopped")
        return True
    
    def _scan_loop(self):
        """Main scanning loop"""
        self.logger.info("Scanner loop started")
        
        while self.is_running:
            try:
                self._perform_scan()
                
                # Sleep in small intervals for responsive shutdown
                for _ in range(self.config.scan_interval):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.stats.scan_errors += 1
                self.stats.last_error = str(e)
                self.logger.error(f"Error in scan loop: {e}")
                time.sleep(5)  # Brief pause before retry
        
        self.logger.info("Scanner loop stopped")
    
    def _process_loop(self):
        """Main processing loop"""
        self.logger.info("Processing loop started")
        
        while self.is_running:
            try:
                self._process_queue()
                time.sleep(1)  # Check queue every second
            except Exception as e:
                self.stats.processing_errors += 1
                self.stats.last_error = str(e)
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(5)
        
        self.logger.info("Processing loop stopped")
    
    @with_error_handling("folder_scanner", "perform_scan")
    def _perform_scan(self):
        """Perform a complete scan of all monitored directories"""
        if self.is_scanning:
            self.logger.debug("Scan already in progress, skipping")
            return
        
        self.is_scanning = True
        scan_start = time.time()
        
        try:
            self.logger.debug("Starting directory scan")
            
            # Track current scan files
            current_files: Set[str] = set()
            changes_detected = 0
            
            # Scan each monitored directory
            for directory in self.config.monitored_directories:
                if not self.is_running:
                    break
                
                directory_changes = self._scan_directory(directory, current_files)
                changes_detected += directory_changes
            
            # Detect deleted files
            deleted_files = self._detect_deleted_files(current_files)
            changes_detected += len(deleted_files)
            
            # Update statistics
            scan_duration = time.time() - scan_start
            self.stats.total_scans += 1
            self.stats.last_scan_time = datetime.now()
            self.stats.last_scan_duration = scan_duration
            
            # Update average scan duration
            if self.stats.total_scans > 1:
                self.stats.avg_scan_duration = (
                    (self.stats.avg_scan_duration * (self.stats.total_scans - 1) + scan_duration) 
                    / self.stats.total_scans
                )
            else:
                self.stats.avg_scan_duration = scan_duration
            
            if changes_detected > 0:
                self.logger.info(f"Scan completed: {changes_detected} changes detected in {scan_duration:.2f}s")
            else:
                self.logger.debug(f"Scan completed: no changes detected in {scan_duration:.2f}s")
            
        finally:
            self.is_scanning = False
    
    def _scan_directory(self, directory: str, current_files: Set[str]) -> int:
        """Scan a single directory for changes"""
        changes_detected = 0
        directory_path = Path(directory)
        
        if not directory_path.exists():
            self.logger.warning(f"Directory no longer exists: {directory}")
            return 0
        
        try:
            # Use parallel scanning if enabled
            if self.config.enable_parallel_scanning:
                changes_detected = self._scan_directory_parallel(directory_path, current_files)
            else:
                changes_detected = self._scan_directory_sequential(directory_path, current_files)
                
        except Exception as e:
            self.logger.error(f"Error scanning directory {directory}: {e}")
            self.stats.scan_errors += 1
        
        return changes_detected
    
    def _scan_directory_sequential(self, directory_path: Path, current_files: Set[str]) -> int:
        """Sequential directory scanning"""
        changes_detected = 0
        
        for root, dirs, files in os.walk(directory_path):
            # Check depth limit
            depth = len(Path(root).relative_to(directory_path).parts)
            if depth > self.config.max_depth:
                dirs.clear()  # Don't recurse deeper
                continue
            
            # Filter directories
            dirs[:] = [d for d in dirs if not self._should_exclude_path(Path(root) / d)]
            
            # Process files in batches
            for i in range(0, len(files), self.config.scan_batch_size):
                if not self.is_running:
                    break
                
                batch = files[i:i + self.config.scan_batch_size]
                for filename in batch:
                    file_path = Path(root) / filename
                    
                    if self._should_process_file(file_path):
                        current_files.add(str(file_path))
                        if self._detect_file_changes(file_path):
                            changes_detected += 1
        
        return changes_detected
    
    def _scan_directory_parallel(self, directory_path: Path, current_files: Set[str]) -> int:
        """Parallel directory scanning using thread pool"""
        changes_detected = 0
        file_batches = []
        
        # Collect files in batches
        for root, dirs, files in os.walk(directory_path):
            depth = len(Path(root).relative_to(directory_path).parts)
            if depth > self.config.max_depth:
                dirs.clear()
                continue
            
            dirs[:] = [d for d in dirs if not self._should_exclude_path(Path(root) / d)]
            
            for i in range(0, len(files), self.config.scan_batch_size):
                batch_files = []
                for filename in files[i:i + self.config.scan_batch_size]:
                    file_path = Path(root) / filename
                    if self._should_process_file(file_path):
                        batch_files.append(file_path)
                
                if batch_files:
                    file_batches.append(batch_files)
        
        # Process batches in parallel
        if file_batches:
            futures = []
            for batch in file_batches:
                future = self._executor.submit(self._process_file_batch, batch, current_files)
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    batch_changes = future.result(timeout=30)
                    changes_detected += batch_changes
                except Exception as e:
                    self.logger.error(f"Error processing file batch: {e}")
        
        return changes_detected
    
    def _process_file_batch(self, file_batch: List[Path], current_files: Set[str]) -> int:
        """Process a batch of files"""
        changes_detected = 0
        
        for file_path in file_batch:
            if not self.is_running:
                break
            
            current_files.add(str(file_path))
            if self._detect_file_changes(file_path):
                changes_detected += 1
        
        return changes_detected
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Determine if a file should be processed"""
        # Check if file exists and is a regular file
        if not file_path.is_file():
            return False
        
        # Check extension
        if file_path.suffix.lower() not in self.config.supported_extensions:
            return False
        
        # Check size limits
        try:
            file_size = file_path.stat().st_size
            if file_size < self.config.min_file_size_bytes:
                return False
            if file_size > self.config.max_file_size_mb * 1024 * 1024:
                return False
        except OSError:
            return False
        
        # Check exclude patterns
        if self._should_exclude_path(file_path):
            return False
        
        return True
    
    def _should_exclude_path(self, path: Path) -> bool:
        """Check if path matches exclude patterns"""
        path_str = str(path)
        
        for pattern in self.config.exclude_patterns:
            if pattern.startswith('*') and path_str.endswith(pattern[1:]):
                return True
            elif pattern.endswith('*') and path_str.startswith(pattern[:-1]):
                return True
            elif pattern in path_str:
                return True
        
        return False
    
    def _detect_file_changes(self, file_path: Path) -> bool:
        """Detect if a file has changed"""
        file_path_str = str(file_path)
        
        try:
            # Get current file info
            stat = file_path.stat()
            current_size = stat.st_size
            current_mtime = stat.st_mtime
            current_ctime = stat.st_ctime
            
            # Check if file is already tracked
            with self._lock:
                if file_path_str in self.file_states:
                    existing_state = self.file_states[file_path_str]
                    existing_metadata = existing_state.metadata
                    
                    # Check for modifications
                    if (current_mtime != existing_metadata.modified_time or 
                        current_size != existing_metadata.size):
                        
                        # Update metadata
                        updated_metadata = self._extract_file_metadata(file_path)
                        existing_state.metadata = updated_metadata
                        existing_state.change_type = ChangeType.MODIFIED
                        existing_state.detected_at = datetime.now()
                        existing_state.status = FileStatus.PENDING
                        existing_state.error_message = None
                        
                        # Add to processing queue
                        if file_path_str not in self.processing_queue:
                            self.processing_queue.append(file_path_str)
                        
                        self.logger.debug(f"File modified: {file_path.name}")
                        return True
                else:
                    # New file detected
                    metadata = self._extract_file_metadata(file_path)
                    file_state = FileState(
                        metadata=metadata,
                        change_type=ChangeType.CREATED,
                        status=FileStatus.PENDING
                    )
                    
                    self.file_states[file_path_str] = file_state
                    
                    # Add to processing queue
                    if file_path_str not in self.processing_queue:
                        self.processing_queue.append(file_path_str)
                    
                    self.logger.debug(f"New file detected: {file_path.name}")
                    return True
        
        except OSError as e:
            self.logger.error(f"Error accessing file {file_path}: {e}")
        
        return False
    
    def _extract_file_metadata(self, file_path: Path) -> FileMetadata:
        """Extract comprehensive metadata from file"""
        stat = file_path.stat()
        
        # Basic metadata
        metadata = FileMetadata(
            file_path=str(file_path),
            relative_path=self._get_relative_path(file_path),
            filename=file_path.name,
            extension=file_path.suffix.lower(),
            size=stat.st_size,
            modified_time=stat.st_mtime,
            created_time=stat.st_ctime
        )
        
        # Content hash if enabled
        if self.config.enable_content_hashing:
            metadata.content_hash = self._calculate_file_hash(file_path)
        
        # Extract organizational metadata from path
        self._extract_path_metadata(file_path, metadata)
        
        # Auto-categorization
        if self.config.auto_categorization:
            metadata.document_type = self._categorize_document(file_path)
        
        # Set priority based on file characteristics
        metadata.priority = self._determine_priority(file_path, metadata)
        
        return metadata
    
    def _get_relative_path(self, file_path: Path) -> str:
        """Get relative path from monitored directory"""
        for directory in self.config.monitored_directories:
            dir_path = Path(directory)
            try:
                if file_path.is_relative_to(dir_path):
                    return str(file_path.relative_to(dir_path))
            except ValueError:
                continue
        
        return str(file_path)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _extract_path_metadata(self, file_path: Path, metadata: FileMetadata):
        """Extract metadata from file path using configured rules"""
        path_parts = file_path.parts
        
        # Apply path metadata rules
        for rule_name, rule_config in self.config.path_metadata_rules.items():
            pattern = rule_config.get('pattern', '')
            field = rule_config.get('field', '')
            
            if pattern and field:
                for part in path_parts:
                    if pattern.lower() in part.lower():
                        if field == 'site_id':
                            metadata.site_id = part
                        elif field == 'category':
                            metadata.category = part
                        elif field == 'department':
                            metadata.department = part
                        break
        
        # Default site extraction from path
        if not metadata.site_id and len(path_parts) > 1:
            # Try to extract site from directory structure
            for part in path_parts[:-1]:  # Exclude filename
                if any(keyword in part.lower() for keyword in ['site', 'location', 'facility']):
                    metadata.site_id = part
                    break
    
    def _categorize_document(self, file_path: Path) -> str:
        """Auto-categorize document based on filename and extension"""
        filename_lower = file_path.name.lower()
        extension = file_path.suffix.lower()
        
        # Category mapping based on filename patterns
        category_patterns = {
            'manual': ['manual', 'guide', 'handbook', 'instruction'],
            'report': ['report', 'analysis', 'summary', 'findings'],
            'procedure': ['procedure', 'process', 'workflow', 'sop'],
            'policy': ['policy', 'regulation', 'compliance', 'standard'],
            'incident': ['incident', 'accident', 'event', 'issue'],
            'maintenance': ['maintenance', 'repair', 'service', 'inspection'],
            'safety': ['safety', 'hazard', 'risk', 'emergency'],
            'training': ['training', 'education', 'course', 'certification']
        }
        
        for category, patterns in category_patterns.items():
            if any(pattern in filename_lower for pattern in patterns):
                return category
        
        # Fallback based on extension
        extension_categories = {
            '.pdf': 'document',
            '.docx': 'document',
            '.doc': 'document',
            '.txt': 'text',
            '.md': 'documentation',
            '.json': 'data',
            '.csv': 'data',
            '.xlsx': 'spreadsheet'
        }
        
        return extension_categories.get(extension, 'unknown')
    
    def _determine_priority(self, file_path: Path, metadata: FileMetadata) -> Priority:
        """Determine processing priority for file"""
        filename_lower = file_path.name.lower()
        
        # High priority patterns
        high_priority_patterns = [
            'urgent', 'critical', 'emergency', 'incident', 'alert', 'immediate'
        ]
        
        # Critical priority patterns
        critical_priority_patterns = [
            'safety', 'hazard', 'accident', 'failure', 'outage'
        ]
        
        for pattern in critical_priority_patterns:
            if pattern in filename_lower:
                return Priority.CRITICAL
        
        for pattern in high_priority_patterns:
            if pattern in filename_lower:
                return Priority.HIGH
        
        # Large files get lower priority
        if metadata.size > 50 * 1024 * 1024:  # 50MB
            return Priority.LOW
        
        return Priority.NORMAL
    
    def _detect_deleted_files(self, current_files: Set[str]) -> List[str]:
        """Detect files that have been deleted"""
        deleted_files = []
        
        with self._lock:
            tracked_files = set(self.file_states.keys())
            deleted_files_set = tracked_files - current_files
            
            for file_path in deleted_files_set:
                if file_path in self.file_states:
                    file_state = self.file_states[file_path]
                    file_state.change_type = ChangeType.DELETED
                    file_state.status = FileStatus.DELETED
                    file_state.detected_at = datetime.now()
                    
                    # Remove from processing queue
                    if file_path in self.processing_queue:
                        self.processing_queue.remove(file_path)
                    
                    deleted_files.append(file_path)
                    self.logger.debug(f"File deleted: {Path(file_path).name}")
        
        return deleted_files
    
    def _process_queue(self):
        """Process files in the processing queue"""
        if not self.processing_queue:
            return
        
        with self._lock:
            # Sort queue by priority
            self.processing_queue.sort(
                key=lambda fp: self.file_states[fp].metadata.priority.value,
                reverse=True
            )
            
            # Process files up to concurrent limit
            processing_count = sum(
                1 for state in self.file_states.values() 
                if state.status == FileStatus.PROCESSING
            )
            
            available_slots = self.config.max_concurrent_files - processing_count
            
            if available_slots > 0:
                files_to_process = self.processing_queue[:available_slots]
                
                for file_path in files_to_process:
                    if file_path in self.file_states:
                        self._submit_file_processing(file_path)
                        self.processing_queue.remove(file_path)
    
    def _submit_file_processing(self, file_path: str):
        """Submit file for processing"""
        if file_path not in self.file_states:
            return
        
        file_state = self.file_states[file_path]
        file_state.status = FileStatus.PROCESSING
        file_state.last_attempt = datetime.now()
        file_state.processing_attempts += 1
        
        # Submit to thread pool
        future = self._executor.submit(self._process_file, file_path)
        future.add_done_callback(lambda f: self._handle_processing_result(file_path, f))
        
        self.logger.debug(f"Submitted for processing: {Path(file_path).name}")
    
    def _process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file"""
        start_time = time.time()
        
        try:
            if not self.processing_callback:
                # No processing callback configured
                return {
                    'status': 'skipped',
                    'message': 'No processing callback configured'
                }
            
            # Get file state
            file_state = self.file_states.get(file_path)
            if not file_state:
                return {
                    'status': 'failed',
                    'error': 'File state not found'
                }
            
            # Prepare processing context
            processing_context = {
                'file_path': file_path,
                'metadata': asdict(file_state.metadata),
                'change_type': file_state.change_type.value if file_state.change_type else None,
                'attempt': file_state.processing_attempts
            }
            
            # Call processing callback
            result = self.processing_callback(processing_context)
            
            # Record processing time
            processing_time = time.time() - start_time
            
            return {
                'status': result.get('status', 'success'),
                'processing_time': processing_time,
                'ingestion_id': result.get('ingestion_id'),
                'message': result.get('message', 'Processing completed')
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing file {file_path}: {e}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'processing_time': processing_time
            }
    
    def _handle_processing_result(self, file_path: str, future):
        """Handle processing result"""
        try:
            result = future.result()
            
            with self._lock:
                if file_path in self.file_states:
                    file_state = self.file_states[file_path]
                    
                    # Update state based on result
                    if result['status'] == 'success':
                        file_state.status = FileStatus.SUCCESS
                        file_state.processed_at = datetime.now()
                        file_state.ingestion_id = result.get('ingestion_id')
                        file_state.error_message = None
                        self.stats.files_successful += 1
                        
                    elif result['status'] == 'skipped':
                        file_state.status = FileStatus.SKIPPED
                        file_state.processed_at = datetime.now()
                        file_state.error_message = result.get('message')
                        self.stats.files_skipped += 1
                        
                    else:  # failed
                        file_state.status = FileStatus.FAILED
                        file_state.error_message = result.get('error', 'Unknown error')
                        self.stats.files_failed += 1
                        
                        # Retry logic
                        if file_state.processing_attempts < self.config.retry_attempts:
                            # Schedule retry
                            retry_time = datetime.now() + timedelta(seconds=self.config.retry_delay)
                            self.logger.info(f"Scheduling retry for {Path(file_path).name} at {retry_time}")
                            # Add back to queue after delay (simplified - in production use proper scheduler)
                            threading.Timer(
                                self.config.retry_delay,
                                lambda: self.processing_queue.append(file_path)
                            ).start()
                    
                    # Update processing time
                    if 'processing_time' in result:
                        file_state.processing_time = result['processing_time']
                        self.stats.total_processing_time += result['processing_time']
                        
                        # Update average processing time
                        processed_files = (self.stats.files_successful + 
                                         self.stats.files_failed + 
                                         self.stats.files_skipped)
                        if processed_files > 0:
                            self.stats.avg_processing_time = (
                                self.stats.total_processing_time / processed_files
                            )
                    
                    self.logger.debug(f"Processing completed for {Path(file_path).name}: {result['status']}")
        
        except Exception as e:
            self.logger.error(f"Error handling processing result for {file_path}: {e}")
            
            with self._lock:
                if file_path in self.file_states:
                    file_state = self.file_states[file_path]
                    file_state.status = FileStatus.FAILED
                    file_state.error_message = f"Result handling error: {str(e)}"
                    self.stats.files_failed += 1
    
    # Public API methods
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive scanner status"""
        with self._lock:
            # Update current statistics
            self.stats.total_files_tracked = len(self.file_states)
            self.stats.files_pending = sum(
                1 for state in self.file_states.values() 
                if state.status == FileStatus.PENDING
            )
            self.stats.files_processing = sum(
                1 for state in self.file_states.values() 
                if state.status == FileStatus.PROCESSING
            )
            
            # Calculate performance metrics
            if self.stats.last_scan_duration > 0:
                self.stats.files_per_second = (
                    self.stats.total_files_tracked / self.stats.last_scan_duration
                )
            
            return {
                'is_running': self.is_running,
                'is_scanning': self.is_scanning,
                'monitored_directories': self.config.monitored_directories,
                'statistics': asdict(self.stats),
                'configuration': {
                    'scan_interval': self.config.scan_interval,
                    'max_concurrent_files': self.config.max_concurrent_files,
                    'supported_extensions': list(self.config.supported_extensions),
                    'max_file_size_mb': self.config.max_file_size_mb
                },
                'queue_size': len(self.processing_queue)
            }
    
    def get_file_states(self, status_filter: Optional[FileStatus] = None) -> Dict[str, Dict[str, Any]]:
        """Get file states with optional filtering"""
        with self._lock:
            if status_filter:
                filtered_states = {
                    path: state for path, state in self.file_states.items()
                    if state.status == status_filter
                }
            else:
                filtered_states = self.file_states.copy()
            
            return {
                path: {
                    'metadata': asdict(state.metadata),
                    'status': state.status.value,
                    'processing_attempts': state.processing_attempts,
                    'last_attempt': state.last_attempt.isoformat() if state.last_attempt else None,
                    'error_message': state.error_message,
                    'processing_time': state.processing_time,
                    'ingestion_id': state.ingestion_id,
                    'change_type': state.change_type.value if state.change_type else None,
                    'detected_at': state.detected_at.isoformat(),
                    'processed_at': state.processed_at.isoformat() if state.processed_at else None
                }
                for path, state in filtered_states.items()
            }
    
    @with_error_handling("folder_scanner", "force_scan")
    def force_scan(self) -> Dict[str, Any]:
        """Force an immediate scan of all directories"""
        if self.is_scanning:
            return {
                'success': False,
                'message': 'Scan already in progress'
            }
        
        try:
            # Temporarily set is_running to True for force scan
            original_running_state = self.is_running
            self.is_running = True
            
            try:
                self._perform_scan()
            finally:
                # Restore original running state
                self.is_running = original_running_state
            
            return {
                'success': True,
                'message': 'Forced scan completed',
                'statistics': asdict(self.stats),
                'files_tracked': len(self.file_states),
                'queue_size': len(self.processing_queue)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def add_directory(self, directory: str) -> bool:
        """Add a new directory to monitor"""
        directory_path = Path(directory)
        
        if not directory_path.exists():
            self.logger.error(f"Directory does not exist: {directory}")
            return False
        
        if not directory_path.is_dir():
            self.logger.error(f"Path is not a directory: {directory}")
            return False
        
        if directory in self.config.monitored_directories:
            self.logger.warning(f"Directory already monitored: {directory}")
            return False
        
        self.config.monitored_directories.append(directory)
        self.logger.info(f"Added directory to monitoring: {directory}")
        return True
    
    def remove_directory(self, directory: str) -> bool:
        """Remove a directory from monitoring"""
        if directory not in self.config.monitored_directories:
            return False
        
        self.config.monitored_directories.remove(directory)
        
        # Remove file states for files in this directory
        with self._lock:
            files_to_remove = [
                path for path in self.file_states.keys()
                if path.startswith(directory)
            ]
            
            for file_path in files_to_remove:
                del self.file_states[file_path]
                if file_path in self.processing_queue:
                    self.processing_queue.remove(file_path)
        
        self.logger.info(f"Removed directory from monitoring: {directory}")
        return True
    
    def retry_failed_files(self) -> int:
        """Retry processing of all failed files"""
        retry_count = 0
        
        with self._lock:
            for file_path, file_state in self.file_states.items():
                if file_state.status == FileStatus.FAILED:
                    file_state.status = FileStatus.PENDING
                    file_state.error_message = None
                    
                    if file_path not in self.processing_queue:
                        self.processing_queue.append(file_path)
                        retry_count += 1
        
        self.logger.info(f"Queued {retry_count} failed files for retry")
        return retry_count
    
    def clear_processed_files(self) -> int:
        """Clear successfully processed files from tracking"""
        cleared_count = 0
        
        with self._lock:
            files_to_remove = [
                path for path, state in self.file_states.items()
                if state.status in [FileStatus.SUCCESS, FileStatus.DELETED]
            ]
            
            for file_path in files_to_remove:
                del self.file_states[file_path]
                cleared_count += 1
        
        self.logger.info(f"Cleared {cleared_count} processed files from tracking")
        return cleared_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        with self._lock:
            return asdict(self.stats)
    
    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()


# Factory function for easy instantiation
def create_folder_scanner(config_dict: Dict[str, Any], 
                         processing_callback: Optional[Callable] = None) -> FolderScanner:
    """Create a folder scanner from configuration dictionary"""
    
    # Convert dictionary to ScannerConfig
    config = ScannerConfig(
        monitored_directories=config_dict.get('monitored_directories', []),
        scan_interval=config_dict.get('scan_interval', 60),
        max_depth=config_dict.get('max_depth', 10),
        enable_content_hashing=config_dict.get('enable_content_hashing', True),
        supported_extensions=set(config_dict.get('supported_extensions', [
            '.pdf', '.txt', '.docx', '.doc', '.md', '.json', '.csv', '.xlsx', '.pptx'
        ])),
        max_file_size_mb=config_dict.get('max_file_size_mb', 100),
        min_file_size_bytes=config_dict.get('min_file_size_bytes', 1),
        exclude_patterns=config_dict.get('exclude_patterns', [
            '.*', '__pycache__', '*.tmp', '*.log', '*.bak'
        ]),
        max_concurrent_files=config_dict.get('max_concurrent_files', 5),
        retry_attempts=config_dict.get('retry_attempts', 3),
        retry_delay=config_dict.get('retry_delay', 60),
        processing_timeout=config_dict.get('processing_timeout', 300),
        path_metadata_rules=config_dict.get('path_metadata_rules', {}),
        auto_categorization=config_dict.get('auto_categorization', True),
        enable_parallel_scanning=config_dict.get('enable_parallel_scanning', True),
        scan_batch_size=config_dict.get('scan_batch_size', 100),
        memory_limit_mb=config_dict.get('memory_limit_mb', 500)
    )
    
    return FolderScanner(config, processing_callback)