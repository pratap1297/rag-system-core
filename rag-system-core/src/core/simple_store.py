"""
Simple JSON Store
Basic JSON storage without file locking to avoid Windows hanging issues
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

class SimpleJSONStore:
    """Simple JSON storage without file locking"""
    
    def __init__(self, base_path: str = "data"):
        print(f"         🔧 SimpleJSONStore.__init__ called with base_path={base_path}")
        self.base_path = Path(base_path)
        print(f"         📋 Path object created: {self.base_path}")
        # Defer directory creation until first use to avoid hanging
        print(f"         ✅ SimpleJSONStore initialized (directory creation deferred)")
    
    def _ensure_directory(self):
        """Ensure directory exists (called on first use)"""
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)
    
    def read(self, collection: str, key: Optional[str] = None) -> Union[Dict, Any]:
        """Read data from JSON store"""
        self._ensure_directory()
        file_path = self.base_path / f"{collection}.json"
        
        if not file_path.exists():
            return {} if key is None else None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                if key is None:
                    return data
                return data.get(key)
        except (json.JSONDecodeError, FileNotFoundError):
            return {} if key is None else None
    
    def write(self, collection: str, data: Dict[str, Any], key: Optional[str] = None):
        """Write data to JSON store"""
        self._ensure_directory()
        file_path = self.base_path / f"{collection}.json"
        
        try:
            existing_data = self.read(collection) or {}
        except:
            existing_data = {}
        
        if key is None:
            # Replace entire collection
            existing_data = data
        else:
            # Update specific key
            existing_data[key] = data
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=2, default=str)
    
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
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        import uuid
        return str(uuid.uuid4())

# Specialized stores
class SimpleMetadataStore(SimpleJSONStore):
    """Simple metadata store"""
    
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

class SimpleLogStore(SimpleJSONStore):
    """Simple log store"""
    
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