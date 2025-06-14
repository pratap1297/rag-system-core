"""
In-Memory Store
Completely in-memory storage that doesn't touch the filesystem
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid

class MemoryStore:
    """Completely in-memory storage"""
    
    def __init__(self, base_path: str = "data"):
        print(f"         🔧 MemoryStore.__init__ called with base_path={base_path}")
        self.base_path = base_path  # Store for compatibility but don't use
        self.collections = {}  # In-memory storage
        print(f"         ✅ MemoryStore initialized (completely in-memory)")
    
    def read(self, collection: str, key: Optional[str] = None) -> Union[Dict, Any]:
        """Read data from memory store"""
        if collection not in self.collections:
            return {} if key is None else None
        
        data = self.collections[collection]
        if key is None:
            return data
        return data.get(key)
    
    def write(self, collection: str, data: Dict[str, Any], key: Optional[str] = None):
        """Write data to memory store"""
        if collection not in self.collections:
            self.collections[collection] = {}
        
        if key is None:
            # Replace entire collection
            self.collections[collection] = data
        else:
            # Update specific key
            self.collections[collection][key] = data
    
    def append(self, collection: str, item: Dict[str, Any], key_field: str = "id"):
        """Append item to collection"""
        if collection not in self.collections:
            self.collections[collection] = {}
        
        # Generate key if not provided
        if key_field not in item:
            item[key_field] = self._generate_id()
        
        key = item[key_field]
        self.collections[collection][key] = item
        return key
    
    def update(self, collection: str, key: str, updates: Dict[str, Any]):
        """Update specific item in collection"""
        if collection not in self.collections:
            return False
        
        data = self.collections[collection]
        if key in data:
            data[key].update(updates)
            data[key]["updated_at"] = datetime.now().isoformat()
            return True
        return False
    
    def delete(self, collection: str, key: str):
        """Delete item from collection"""
        if collection not in self.collections:
            return False
        
        data = self.collections[collection]
        if key in data:
            del data[key]
            return True
        return False
    
    def search(self, collection: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search items in collection based on filters"""
        if collection not in self.collections:
            return []
        
        data = self.collections[collection]
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
        return list(self.collections.keys())
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        return str(uuid.uuid4())

# Specialized stores
class MemoryMetadataStore(MemoryStore):
    """In-memory metadata store"""
    
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metadata store statistics"""
        try:
            files_count = len(self.collections.get("files_metadata", {}))
            chunks_count = len(self.collections.get("chunks_metadata", {}))
            
            return {
                'total_files': files_count,
                'total_chunks': chunks_count,
                'collections': len(self.collections),
                'collections_info': {
                    name: {'count': len(data)} 
                    for name, data in self.collections.items()
                }
            }
        except Exception as e:
            return {
                'total_files': 0,
                'total_chunks': 0,
                'collections': 0,
                'collections_info': {},
                'error': str(e)
            }

    def get_all_files(self) -> List[Dict[str, Any]]:
        """Get all files metadata"""
        files_data = self.collections.get("files_metadata", {})
        return [
            {
                **metadata,
                'filename': metadata.get('file_path', 'unknown').split('/')[-1],
                'chunk_count': len(self.get_file_chunks(metadata.get('file_id', '')))
            }
            for metadata in files_data.values()
        ]
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks metadata"""
        chunks_data = self.collections.get("chunks_metadata", {})
        return [
            {
                **metadata,
                'filename': metadata.get('filename', 'unknown')
            }
            for metadata in chunks_data.values()
        ]
class MemoryLogStore(MemoryStore):
    """In-memory log store"""
    
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
        collection_name = f"{event_type}_log"
        if collection_name not in self.collections:
            return []
        
        logs = list(self.collections[collection_name].values())
        # Sort by timestamp descending
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return logs[:limit] 