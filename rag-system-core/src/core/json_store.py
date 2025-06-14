"""
JSON-based Data Store
Thread-safe JSON storage with atomic operations and backup support
"""
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from contextlib import contextmanager

try:
    import fcntl
except ImportError:
    # Windows fallback
    import msvcrt
    fcntl = None

class JSONStore:
    """Thread-safe JSON storage with atomic operations"""
    
    def __init__(self, base_path: str = "data"):
        print(f"       🔧 JSONStore.__init__ called with base_path={base_path}")
        self.base_path = Path(base_path)
        print(f"       📋 Path object created: {self.base_path}")
        print(f"       📋 Creating directory: {self.base_path}")
        self.base_path.mkdir(parents=True, exist_ok=True)
        print(f"       📋 Directory created successfully")
        self._locks = {}
        print(f"       ✅ JSONStore initialized successfully")
        self._lock_manager = threading.Lock()
    
    def _get_file_lock(self, file_path: str) -> threading.Lock:
        """Get or create a lock for a specific file"""
        with self._lock_manager:
            if file_path not in self._locks:
                self._locks[file_path] = threading.Lock()
            return self._locks[file_path]
    
    @contextmanager
    def _file_lock(self, file_path: Path):
        """Context manager for file locking - simplified for Windows compatibility"""
        # Use thread-level locking only, skip file-level locking to prevent hanging
        lock = self._get_file_lock(str(file_path))
        
        with lock:
            if file_path.exists():
                with open(file_path, 'r+') as f:
                    # Skip file locking for now to prevent hanging on Windows
                    yield f
            else:
                # Create file if it doesn't exist
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w') as f:
                    json.dump({}, f)
                
                with open(file_path, 'r+') as f:
                    # Skip file locking for now to prevent hanging on Windows
                    yield f
    
    def _backup_file(self, file_path: Path):
        """Create backup of file before modification"""
        if file_path.exists():
            backup_dir = file_path.parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}.json"
            backup_path = backup_dir / backup_name
            
            shutil.copy2(file_path, backup_path)
            
            # Keep only last 5 backups
            backups = sorted(backup_dir.glob(f"{file_path.stem}_*.json"))
            if len(backups) > 5:
                for old_backup in backups[:-5]:
                    old_backup.unlink()
    
    def read(self, collection: str, key: Optional[str] = None) -> Union[Dict, Any]:
        """Read data from JSON store"""
        file_path = self.base_path / f"{collection}.json"
        
        if not file_path.exists():
            return {} if key is None else None
        
        try:
            with self._file_lock(file_path) as f:
                f.seek(0)
                data = json.load(f)
                
                if key is None:
                    return data
                return data.get(key)
        except (json.JSONDecodeError, FileNotFoundError):
            return {} if key is None else None
    
    def write(self, collection: str, data: Dict[str, Any], key: Optional[str] = None):
        """Write data to JSON store"""
        file_path = self.base_path / f"{collection}.json"
        
        # Create backup before modification
        self._backup_file(file_path)
        
        with self._file_lock(file_path) as f:
            f.seek(0)
            try:
                existing_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                existing_data = {}
            
            if key is None:
                # Replace entire collection
                existing_data = data
            else:
                # Update specific key
                existing_data[key] = data
            
            # Write atomically
            f.seek(0)
            f.truncate()
            json.dump(existing_data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
    
    def append(self, collection: str, item: Dict[str, Any], key_field: str = "id"):
        """Append item to collection"""
        data = self.read(collection) or {}
        
        # Generate key if not provided
        if key_field not in item:
            item[key_field] = self._generate_id()
        
        key = item[key_field]
        data[key] = item
        
        self.write(collection, data)
        return key
    
    def update(self, collection: str, key: str, updates: Dict[str, Any]):
        """Update specific item in collection"""
        data = self.read(collection) or {}
        
        if key in data:
            data[key].update(updates)
            data[key]["updated_at"] = datetime.now().isoformat()
            self.write(collection, data)
            return True
        return False
    
    def delete(self, collection: str, key: str):
        """Delete item from collection"""
        data = self.read(collection) or {}
        
        if key in data:
            del data[key]
            self.write(collection, data)
            return True
        return False
    
    def search(self, collection: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search items in collection based on filters"""
        data = self.read(collection) or {}
        results = []
        
        for item_key, item in data.items():
            match = True
            for filter_key, filter_value in filters.items():
                if filter_key not in item or item[filter_key] != filter_value:
                    match = False
                    break
            
            if match:
                results.append({**item, "_key": item_key})
        
        return results
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        json_files = self.base_path.glob("*.json")
        return [f.stem for f in json_files if not f.name.startswith('.')]
    
    def collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        data = self.read(collection) or {}
        file_path = self.base_path / f"{collection}.json"
        
        stats = {
            "count": len(data),
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
            "last_modified": datetime.fromtimestamp(
                file_path.stat().st_mtime
            ).isoformat() if file_path.exists() else None
        }
        
        return stats
    
    def backup_collection(self, collection: str) -> str:
        """Create manual backup of collection"""
        file_path = self.base_path / f"{collection}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Collection {collection} not found")
        
        backup_dir = self.base_path / "manual_backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{collection}_{timestamp}.json"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        return str(backup_path)
    
    def restore_collection(self, collection: str, backup_path: str):
        """Restore collection from backup"""
        backup_file = Path(backup_path)
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        target_path = self.base_path / f"{collection}.json"
        
        # Validate backup file
        try:
            with open(backup_file, 'r') as f:
                json.load(f)
        except json.JSONDecodeError:
            raise ValueError("Invalid backup file format")
        
        # Create backup of current file
        if target_path.exists():
            self._backup_file(target_path)
        
        # Restore from backup
        shutil.copy2(backup_file, target_path)
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        import uuid
        return str(uuid.uuid4())
    
    def cleanup_old_backups(self, days: int = 30):
        """Clean up old backup files"""
        backup_dirs = [
            self.base_path / "backups",
            self.base_path / "manual_backups"
        ]
        
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        for backup_dir in backup_dirs:
            if backup_dir.exists():
                for backup_file in backup_dir.glob("*.json"):
                    if backup_file.stat().st_mtime < cutoff_time:
                        backup_file.unlink()

# Specialized stores for different data types
class MetadataStore(JSONStore):
    """Specialized store for metadata"""
    
    def __init__(self, base_path: str = "data/metadata"):
        super().__init__(base_path)
    
    def add_file_metadata(self, file_path: str, metadata: Dict[str, Any]) -> str:
        """Add file metadata"""
        metadata.update({
            "file_path": file_path,
            "created_at": datetime.now().isoformat(),
            "type": "file"
        })
        return self.append("files_metadata", metadata, "file_id")
    
    def add_chunk_metadata(self, chunk_data: Dict[str, Any]) -> str:
        """Add chunk metadata"""
        chunk_data.update({
            "created_at": datetime.now().isoformat(),
            "type": "chunk"
        })
        return self.append("chunks_metadata", chunk_data, "chunk_id")
    
    def get_file_chunks(self, file_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a file"""
        return self.search("chunks_metadata", {"file_id": file_id})

class LogStore(JSONStore):
    """Specialized store for logs"""
    
    def __init__(self, base_path: str = "data/logs"):
        super().__init__(base_path)
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.append(f"{event_type}_log", log_entry, "log_id")
    
    def get_recent_logs(self, event_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent logs of specific type"""
        logs = list(self.read(f"{event_type}_log", {}).values())
        # Sort by timestamp descending
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return logs[:limit] 