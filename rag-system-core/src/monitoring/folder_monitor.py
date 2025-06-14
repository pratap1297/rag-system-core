"""
Folder Monitoring Service for RAG System
Monitors specified folders for file changes and automatically ingests new/modified files
"""
import os
import time
import threading
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import json

@dataclass
class FileState:
    """Represents the state of a monitored file"""
    path: str
    size: int
    mtime: float
    hash: str
    doc_path: str
    last_ingested: Optional[str] = None
    ingestion_status: str = "pending"  # pending, success, failed
    error_message: Optional[str] = None

@dataclass
class FolderMonitorStats:
    """Statistics for folder monitoring"""
    monitored_folders: List[str]
    total_files_tracked: int
    files_ingested: int
    files_failed: int
    last_scan_time: Optional[str]
    is_running: bool
    check_interval: int

class FolderMonitor:
    """Monitors folders for file changes and triggers ingestion"""
    
    def __init__(self, container=None, config_manager=None):
        self.container = container
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread = None
        self.file_states: Dict[str, FileState] = {}
        self.monitored_folders: List[str] = []
        
        # Configuration
        self.check_interval = 60  # seconds
        self.supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.json', '.csv'}
        self.max_file_size_mb = 100
        self.auto_ingest = True
        self.recursive = True
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load configuration
        self._load_config()
        
        # Statistics
        self.stats = {
            'files_scanned': 0,
            'files_ingested': 0,
            'files_failed': 0,
            'last_scan_time': None,
            'scan_count': 0
        }
        
        self.logger.info("Folder monitor initialized")
    
    def _load_config(self):
        """Load configuration from config manager"""
        if not self.config_manager:
            return
        
        try:
            config = self.config_manager.get_config()
            folder_config = getattr(config, 'folder_monitoring', None)
            
            if folder_config:
                self.check_interval = getattr(folder_config, 'check_interval_seconds', 60)
                self.supported_extensions = set(getattr(folder_config, 'supported_extensions', ['.txt', '.md', '.pdf', '.docx', '.json', '.csv']))
                self.max_file_size_mb = getattr(folder_config, 'max_file_size_mb', 100)
                self.auto_ingest = getattr(folder_config, 'auto_ingest', True)
                self.recursive = getattr(folder_config, 'recursive', True)
                self.monitored_folders = list(getattr(folder_config, 'monitored_folders', []))
                
                self.logger.info(f"Loaded folder monitoring config: {len(self.monitored_folders)} folders, {self.check_interval}s interval")
        except Exception as e:
            self.logger.error(f"Failed to load folder monitoring config: {e}")
    
    def add_folder(self, folder_path: str) -> Dict[str, Any]:
        """Add a folder to monitoring"""
        folder_path = os.path.abspath(folder_path)
        
        if not os.path.exists(folder_path):
            return {"success": False, "error": f"Folder does not exist: {folder_path}"}
        
        if not os.path.isdir(folder_path):
            return {"success": False, "error": f"Path is not a directory: {folder_path}"}
        
        with self._lock:
            if folder_path not in self.monitored_folders:
                self.monitored_folders.append(folder_path)
                self._save_monitored_folders()
                
                # Initial scan of the new folder
                self._scan_folder(folder_path)
                
                self.logger.info(f"Added folder to monitoring: {folder_path}")
                return {
                    "success": True, 
                    "message": f"Folder added to monitoring: {folder_path}",
                    "files_found": len([f for f in self.file_states.keys() if f.startswith(folder_path)])
                }
            else:
                return {"success": False, "error": f"Folder already being monitored: {folder_path}"}
    
    def remove_folder(self, folder_path: str) -> Dict[str, Any]:
        """Remove a folder from monitoring"""
        folder_path = os.path.abspath(folder_path)
        
        with self._lock:
            if folder_path in self.monitored_folders:
                self.monitored_folders.remove(folder_path)
                self._save_monitored_folders()
                
                # Remove file states for this folder
                files_to_remove = [f for f in self.file_states.keys() if f.startswith(folder_path)]
                for file_path in files_to_remove:
                    del self.file_states[file_path]
                
                self.logger.info(f"Removed folder from monitoring: {folder_path}")
                return {
                    "success": True, 
                    "message": f"Folder removed from monitoring: {folder_path}",
                    "files_removed": len(files_to_remove)
                }
            else:
                return {"success": False, "error": f"Folder not being monitored: {folder_path}"}
    
    def get_monitored_folders(self) -> List[str]:
        """Get list of monitored folders"""
        return self.monitored_folders.copy()
    
    def _save_monitored_folders(self):
        """Save monitored folders to config"""
        if not self.config_manager:
            return
        
        try:
            # Update config
            config = self.config_manager.get_config()
            if hasattr(config, 'folder_monitoring'):
                config.folder_monitoring.monitored_folders = self.monitored_folders
                self.config_manager.save_config()
                self.logger.info("Saved monitored folders to config")
        except Exception as e:
            self.logger.error(f"Failed to save monitored folders: {e}")
    
    def start_monitoring(self) -> Dict[str, Any]:
        """Start folder monitoring"""
        if self.is_running:
            return {"success": False, "error": "Monitoring is already running"}
        
        if not self.monitored_folders:
            return {"success": False, "error": "No folders configured for monitoring"}
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"Started folder monitoring for {len(self.monitored_folders)} folders")
        return {
            "success": True, 
            "message": f"Started monitoring {len(self.monitored_folders)} folders",
            "folders": self.monitored_folders,
            "interval": self.check_interval
        }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop folder monitoring"""
        if not self.is_running:
            return {"success": False, "error": "Monitoring is not running"}
        
        self.is_running = False
        
        # Wait for thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Stopped folder monitoring")
        return {"success": True, "message": "Folder monitoring stopped"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        with self._lock:
            return {
                "is_running": self.is_running,
                "monitored_folders": self.monitored_folders.copy(),
                "total_files_tracked": len(self.file_states),
                "files_ingested": len([f for f in self.file_states.values() if f.ingestion_status == "success"]),
                "files_failed": len([f for f in self.file_states.values() if f.ingestion_status == "failed"]),
                "files_pending": len([f for f in self.file_states.values() if f.ingestion_status == "pending"]),
                "last_scan_time": self.stats.get('last_scan_time'),
                "check_interval": self.check_interval,
                "scan_count": self.stats.get('scan_count', 0),
                "auto_ingest": self.auto_ingest
            }
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info("Folder monitoring loop started")
        
        while self.is_running:
            try:
                self._scan_all_folders()
                self.stats['scan_count'] = self.stats.get('scan_count', 0) + 1
                self.stats['last_scan_time'] = datetime.now().isoformat()
                
                # Sleep in small intervals to allow quick shutdown
                for _ in range(self.check_interval):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying
        
        self.logger.info("Folder monitoring loop stopped")
    
    def _scan_all_folders(self):
        """Scan all monitored folders for changes"""
        changes_detected = []
        
        for folder_path in self.monitored_folders:
            if os.path.exists(folder_path):
                folder_changes = self._scan_folder(folder_path)
                changes_detected.extend(folder_changes)
        
        if changes_detected:
            self.logger.info(f"Detected {len(changes_detected)} file changes")
            
            # Process changes if auto-ingest is enabled
            if self.auto_ingest:
                self._process_changes(changes_detected)
        
        # Also process any existing pending files if auto-ingest is enabled
        if self.auto_ingest:
            self._process_pending_files()
    
    def _scan_folder(self, folder_path: str) -> List[Tuple[str, str]]:
        """Scan a single folder for changes"""
        changes = []
        current_files = {}
        
        # Scan files
        if self.recursive:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if self._should_monitor_file(file_path):
                        current_files[file_path] = self._get_file_info(file_path)
        else:
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and self._should_monitor_file(file_path):
                    current_files[file_path] = self._get_file_info(file_path)
        
        # Check for new or modified files
        for file_path, file_info in current_files.items():
            if file_info is None:
                continue
                
            old_state = self.file_states.get(file_path)
            
            if not old_state:
                # New file
                changes.append(('new', file_path))
                self._create_file_state(file_path, file_info)
            elif (file_info['mtime'] != old_state.mtime or 
                  file_info['size'] != old_state.size):
                # Potentially modified file - check hash
                new_hash = self._get_file_hash(file_path)
                if new_hash != old_state.hash:
                    changes.append(('modified', file_path))
                    self._update_file_state(file_path, file_info, new_hash)
        
        # Check for deleted files
        for file_path, file_state in list(self.file_states.items()):
            if file_path.startswith(folder_path) and file_path not in current_files:
                if os.path.exists(file_path):
                    continue  # File still exists, might be in different scan
                changes.append(('deleted', file_path))
                del self.file_states[file_path]
        
        return changes
    
    def _should_monitor_file(self, file_path: str) -> bool:
        """Check if file should be monitored"""
        try:
            # Check extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self.supported_extensions:
                return False
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size_mb * 1024 * 1024:
                return False
            
            # Check if file is accessible
            if not os.access(file_path, os.R_OK):
                return False
            
            return True
        except Exception:
            return False
    
    def _get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file information"""
        try:
            stat = os.stat(file_path)
            return {
                'size': stat.st_size,
                'mtime': stat.st_mtime
            }
        except Exception as e:
            self.logger.error(f"Failed to get file info for {file_path}: {e}")
            return None
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate file hash"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def _create_file_state(self, file_path: str, file_info: Dict[str, Any]):
        """Create new file state"""
        doc_path = self._generate_doc_path(file_path)
        file_hash = self._get_file_hash(file_path)
        
        self.file_states[file_path] = FileState(
            path=file_path,
            size=file_info['size'],
            mtime=file_info['mtime'],
            hash=file_hash,
            doc_path=doc_path,
            ingestion_status="pending"
        )
    
    def _update_file_state(self, file_path: str, file_info: Dict[str, Any], new_hash: str):
        """Update existing file state"""
        if file_path in self.file_states:
            state = self.file_states[file_path]
            state.size = file_info['size']
            state.mtime = file_info['mtime']
            state.hash = new_hash
            state.ingestion_status = "pending"
            state.error_message = None
    
    def _generate_doc_path(self, file_path: str) -> str:
        """Generate document path for file"""
        # Create a relative path from the monitored folder
        for folder in self.monitored_folders:
            if file_path.startswith(folder):
                rel_path = os.path.relpath(file_path, folder)
                folder_name = os.path.basename(folder)
                return f"folder_monitor/{folder_name}/{rel_path}"
        
        # Fallback to filename
        return f"folder_monitor/{os.path.basename(file_path)}"
    
    def _process_changes(self, changes: List[Tuple[str, str]]):
        """Process detected file changes"""
        if not self.container:
            self.logger.warning("No container available for ingestion")
            return
        
        try:
            ingestion_engine = self.container.get('ingestion_engine')
            if not ingestion_engine:
                self.logger.error("Ingestion engine not available")
                return
            
            for change_type, file_path in changes:
                if change_type in ['new', 'modified']:
                    self._ingest_file(file_path, ingestion_engine)
                elif change_type == 'deleted':
                    self._handle_deleted_file(file_path, ingestion_engine)
                    
        except Exception as e:
            self.logger.error(f"Failed to process changes: {e}")
    
    def _process_pending_files(self):
        """Process any files that are currently pending ingestion"""
        if not self.container:
            return
        
        try:
            ingestion_engine = self.container.get('ingestion_engine')
            if not ingestion_engine:
                return
            
            # Find all pending files
            pending_files = [
                file_path for file_path, file_state in self.file_states.items()
                if file_state.ingestion_status == "pending"
            ]
            
            if pending_files:
                self.logger.info(f"Processing {len(pending_files)} pending files")
                for file_path in pending_files:
                    self._ingest_file(file_path, ingestion_engine)
                    
        except Exception as e:
            self.logger.error(f"Failed to process pending files: {e}")
    
    def _ingest_file(self, file_path: str, ingestion_engine):
        """Ingest a single file"""
        try:
            file_state = self.file_states.get(file_path)
            if not file_state:
                return
            
            # Prepare metadata
            metadata = {
                "source": "folder_monitor",
                "original_path": file_path,
                "doc_path": file_state.doc_path,
                "monitored_folder": self._get_parent_folder(file_path),
                "file_size": file_state.size,
                "file_hash": file_state.hash,
                "ingestion_time": datetime.now().isoformat()
            }
            
            # Ingest file
            result = ingestion_engine.ingest_file(file_path, metadata)
            
            if result.get('status') == 'success':
                file_state.ingestion_status = "success"
                file_state.last_ingested = datetime.now().isoformat()
                file_state.error_message = None
                self.stats['files_ingested'] = self.stats.get('files_ingested', 0) + 1
                self.logger.info(f"Successfully ingested: {os.path.basename(file_path)}")
            else:
                file_state.ingestion_status = "failed"
                file_state.error_message = result.get('error', 'Unknown error')
                self.stats['files_failed'] = self.stats.get('files_failed', 0) + 1
                self.logger.error(f"Failed to ingest {file_path}: {file_state.error_message}")
                
        except Exception as e:
            if file_path in self.file_states:
                self.file_states[file_path].ingestion_status = "failed"
                self.file_states[file_path].error_message = str(e)
            self.stats['files_failed'] = self.stats.get('files_failed', 0) + 1
            self.logger.error(f"Exception ingesting {file_path}: {e}")
    
    def _handle_deleted_file(self, file_path: str, ingestion_engine):
        """Handle deleted file"""
        try:
            # Get the file state to retrieve doc_path
            file_state = self.file_states.get(file_path)
            doc_path = None
            
            if file_state:
                doc_path = file_state.doc_path
                self.logger.info(f"File deleted: {os.path.basename(file_path)} (doc_path: {doc_path})")
            else:
                # Generate doc_path if file_state is not available
                doc_path = self._generate_doc_path(file_path)
                self.logger.info(f"File deleted: {os.path.basename(file_path)} (generated doc_path: {doc_path})")
            
            # Delete vectors from the vector store
            result = ingestion_engine.delete_file(file_path, doc_path)
            
            if result.get('status') == 'success':
                vectors_deleted = result.get('vectors_deleted', 0)
                if vectors_deleted > 0:
                    self.logger.info(f"Successfully deleted {vectors_deleted} vectors for deleted file: {os.path.basename(file_path)}")
                else:
                    self.logger.info(f"No vectors found to delete for file: {os.path.basename(file_path)}")
            else:
                error_msg = result.get('error', 'Unknown error')
                self.logger.error(f"Failed to delete vectors for file {file_path}: {error_msg}")
            
            # Remove from file_states tracking
            if file_path in self.file_states:
                del self.file_states[file_path]
                self.logger.info(f"Removed {os.path.basename(file_path)} from tracking")
                
        except Exception as e:
            self.logger.error(f"Error handling deleted file {file_path}: {e}")
    
    def _get_parent_folder(self, file_path: str) -> str:
        """Get the monitored parent folder for a file"""
        for folder in self.monitored_folders:
            if file_path.startswith(folder):
                return folder
        return ""
    
    def get_file_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all file states"""
        return {path: asdict(state) for path, state in self.file_states.items()}
    
    def force_scan(self) -> Dict[str, Any]:
        """Force an immediate scan of all folders"""
        try:
            changes = []
            for folder_path in self.monitored_folders:
                if os.path.exists(folder_path):
                    folder_changes = self._scan_folder(folder_path)
                    changes.extend(folder_changes)
            
            if self.auto_ingest and changes:
                self._process_changes(changes)
            
            # Also process any existing pending files if auto-ingest is enabled
            if self.auto_ingest:
                self._process_pending_files()
            
            self.stats['last_scan_time'] = datetime.now().isoformat()
            
            return {
                "success": True,
                "message": f"Scan completed",
                "changes_detected": len(changes),
                "files_tracked": len(self.file_states)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# Global instance
folder_monitor = None

def initialize_folder_monitor(container, config_manager):
    """Initialize the global folder monitor"""
    global folder_monitor
    folder_monitor = FolderMonitor(container, config_manager)
    return folder_monitor 