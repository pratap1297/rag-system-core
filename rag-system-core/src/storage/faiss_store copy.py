"""
FAISS Vector Store
High-performance vector similarity search using FAISS
"""
import faiss
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

from ..core.error_handling import FAISSError, StorageError

class FAISSStore:
    """FAISS-based vector store for similarity search"""
    
    def __init__(self, index_path: str = "data/vectors/index.faiss", dimension: int = 384):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index = None
        self.metadata_path = self.index_path.parent / "vector_metadata.pkl"
        self.id_to_metadata = {}
        self.index_to_id = {}  # Maps FAISS index position to our vector ID
        self.next_id = 0
        
        # Create directory if it doesn't exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load index
        self._initialize_index()
        
        logging.info(f"FAISS store initialized with dimension {dimension}")
    
    def _initialize_index(self):
        """Initialize or load existing FAISS index"""
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self._load_metadata()
                logging.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logging.warning(f"Failed to load existing index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        # Use IndexFlatIP for cosine similarity (after L2 normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_to_metadata = {}
        self.index_to_id = {}
        self.next_id = 0
        logging.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def _load_metadata(self):
        """Load vector metadata"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.id_to_metadata = data.get('id_to_metadata', {})
                    self.index_to_id = data.get('index_to_id', {})
                    self.next_id = data.get('next_id', 0)
            except Exception as e:
                logging.warning(f"Failed to load metadata: {e}")
                self.id_to_metadata = {}
                self.index_to_id = {}
                self.next_id = 0
    
    def _save_metadata(self):
        """Save vector metadata"""
        try:
            data = {
                'id_to_metadata': self.id_to_metadata,
                'index_to_id': self.index_to_id,
                'next_id': self.next_id,
                'saved_at': datetime.now().isoformat()
            }
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            raise StorageError(f"Failed to save metadata: {e}")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[int]:
        """Add vectors to the index with metadata"""
        if len(vectors) != len(metadata):
            raise FAISSError("Number of vectors must match number of metadata entries")
        
        if not vectors:
            return []
        
        try:
            # Convert to numpy array and normalize
            vector_array = np.array(vectors, dtype=np.float32)
            if vector_array.shape[1] != self.dimension:
                raise FAISSError(f"Vector dimension {vector_array.shape[1]} doesn't match index dimension {self.dimension}")
            
            normalized_vectors = self._normalize_vectors(vector_array)
            
            # Get current index size before adding
            current_index_size = self.index.ntotal
            
            # Add to index
            self.index.add(normalized_vectors)
            
            # Store metadata and maintain index mapping
            vector_ids = []
            for i, meta in enumerate(metadata):
                vector_id = self.next_id
                faiss_index = current_index_size + i
                
                self.id_to_metadata[vector_id] = {
                    **meta,
                    'added_at': datetime.now().isoformat(),
                    'vector_id': vector_id
                }
                self.index_to_id[faiss_index] = vector_id
                vector_ids.append(vector_id)
                self.next_id += 1
            
            # Save everything
            self.save_index()
            
            logging.info(f"Added {len(vectors)} vectors to FAISS index")
            return vector_ids
            
        except Exception as e:
            raise FAISSError(f"Failed to add vectors: {e}")
    
    def search(self, query_vector: List[float], k: int = 5, 
               filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query vector
            query_array = np.array([query_vector], dtype=np.float32)
            if query_array.shape[1] != self.dimension:
                raise FAISSError(f"Query vector dimension {query_array.shape[1]} doesn't match index dimension {self.dimension}")
            
            normalized_query = self._normalize_vectors(query_array)
            
            # Search in FAISS
            search_k = min(k * 2, self.index.ntotal)  # Get more results for filtering
            scores, indices = self.index.search(normalized_query, search_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                # Get vector ID from index mapping
                vector_id = self.index_to_id.get(idx)
                if vector_id is None:
                    continue
                
                # Get metadata using vector ID
                if vector_id in self.id_to_metadata:
                    metadata = self.id_to_metadata[vector_id].copy()
                    
                    # Skip deleted vectors
                    if metadata.get('deleted', False):
                        continue
                    
                    metadata['similarity_score'] = float(score)
                    
                    # Apply filters if provided
                    if filter_metadata:
                        if not self._matches_filter(metadata, filter_metadata):
                            continue
                    
                    results.append(metadata)
                    
                    if len(results) >= k:
                        break
            
            return results
            
        except Exception as e:
            raise FAISSError(f"Search failed: {e}")
    
    def search_with_metadata(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors with enhanced metadata format for query engine"""
        if self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query vector
            query_array = np.array([query_vector], dtype=np.float32)
            if query_array.shape[1] != self.dimension:
                raise FAISSError(f"Query vector dimension {query_array.shape[1]} doesn't match index dimension {self.dimension}")
            
            normalized_query = self._normalize_vectors(query_array)
            
            # Search in FAISS
            scores, indices = self.index.search(normalized_query, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                # Get vector ID from index mapping
                vector_id = self.index_to_id.get(idx)
                if vector_id is None:
                    # Create fallback metadata for orphaned vectors
                    result = {
                        'faiss_index': int(idx),
                        'similarity_score': float(score),
                        'score': float(score),
                        'vector_id': f'orphan_{idx}',
                        'doc_id': 'unknown',
                        'content': 'Content not available (orphaned vector)',
                        'filename': 'unknown',
                        'chunk_id': f'chunk_{idx}',
                        'text': 'Content not available (orphaned vector)',
                        'metadata': {}
                    }
                    results.append(result)
                    continue
                
                # Get metadata using vector ID
                if vector_id in self.id_to_metadata:
                    metadata = self.id_to_metadata[vector_id].copy()
                    
                    # Skip deleted vectors
                    if metadata.get('deleted', False):
                        continue
                    
                    # Format result for query engine compatibility
                    result = {
                        'faiss_index': int(idx),
                        'similarity_score': float(score),
                        'score': float(score),  # For compatibility
                        'vector_id': str(vector_id),
                        'doc_id': metadata.get('doc_id', 'unknown'),
                        'content': metadata.get('content', metadata.get('text', '')),
                        'text': metadata.get('content', metadata.get('text', '')),  # For compatibility
                        'filename': metadata.get('filename', metadata.get('file_path', 'unknown')),
                        'chunk_id': metadata.get('chunk_id', f'chunk_{vector_id}'),
                        'chunk_index': metadata.get('chunk_index', 0),
                        'page_number': metadata.get('page_number'),
                        'metadata': metadata
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            raise FAISSError(f"Search with metadata failed: {e}")
    
    def _matches_filter(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def get_vector_metadata(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific vector"""
        return self.id_to_metadata.get(vector_id)
    
    def update_metadata(self, vector_id: int, updates: Dict[str, Any]):
        """Update metadata for a specific vector"""
        if vector_id in self.id_to_metadata:
            self.id_to_metadata[vector_id].update(updates)
            self.id_to_metadata[vector_id]['updated_at'] = datetime.now().isoformat()
            self._save_metadata()
        else:
            raise FAISSError(f"Vector ID {vector_id} not found")
    
    def delete_vectors(self, vector_ids: List[int]):
        """Delete vectors from the index (marks as deleted in metadata)"""
        for vector_id in vector_ids:
            if vector_id in self.id_to_metadata:
                self.id_to_metadata[vector_id]['deleted'] = True
                self.id_to_metadata[vector_id]['deleted_at'] = datetime.now().isoformat()
        
        self._save_metadata()
        logging.info(f"Marked {len(vector_ids)} vectors as deleted")
    
    def rebuild_index(self):
        """Rebuild index excluding deleted vectors"""
        if not self.id_to_metadata:
            return
        
        try:
            # Get active vectors
            active_vectors = []
            active_metadata = []
            new_id_mapping = {}
            
            for old_id, metadata in self.id_to_metadata.items():
                if not metadata.get('deleted', False):
                    # We can't retrieve the original vector from FAISS easily
                    # This is a limitation - in a production system, you might want to
                    # store vectors separately or use a different approach
                    logging.warning("Rebuild index requires re-adding vectors from source")
                    return
            
            # Create new index
            self._create_new_index()
            
            # Re-add active vectors
            if active_vectors:
                self.add_vectors(active_vectors, active_metadata)
            
            logging.info("Index rebuilt successfully")
            
        except Exception as e:
            raise FAISSError(f"Failed to rebuild index: {e}")
    
    def save_index(self):
        """Save the FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            self._save_metadata()
            
        except Exception as e:
            raise StorageError(f"Failed to save index: {e}")
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the index"""
        active_count = sum(1 for meta in self.id_to_metadata.values() 
                          if not meta.get('deleted', False))
        
        return {
            'ntotal': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'active_vectors': active_count,
            'deleted_vectors': len(self.id_to_metadata) - active_count,
            'index_path': str(self.index_path),
            'metadata_path': str(self.metadata_path)
        }
    
    def backup_index(self, backup_path: str):
        """Create a backup of the index"""
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Backup FAISS index
        if self.index_path.exists():
            backup_index_path = backup_path / f"index_{timestamp}.faiss"
            faiss.write_index(self.index, str(backup_index_path))
        
        # Backup metadata
        if self.metadata_path.exists():
            backup_metadata_path = backup_path / f"metadata_{timestamp}.pkl"
            with open(self.metadata_path, 'rb') as src, open(backup_metadata_path, 'wb') as dst:
                dst.write(src.read())
        
        logging.info(f"Index backed up to {backup_path}")
        return str(backup_path)
    
    def restore_index(self, backup_path: str):
        """Restore index from backup"""
        backup_path = Path(backup_path)
        
        # Find latest backup files
        index_backups = list(backup_path.glob("index_*.faiss"))
        metadata_backups = list(backup_path.glob("metadata_*.pkl"))
        
        if not index_backups or not metadata_backups:
            raise FAISSError("No valid backup files found")
        
        # Get latest backups
        latest_index = max(index_backups, key=lambda x: x.stat().st_mtime)
        latest_metadata = max(metadata_backups, key=lambda x: x.stat().st_mtime)
        
        # Restore
        self.index = faiss.read_index(str(latest_index))
        
        with open(latest_metadata, 'rb') as f:
            data = pickle.load(f)
            self.id_to_metadata = data.get('id_to_metadata', {})
            self.index_to_id = data.get('index_to_id', {})
            self.next_id = data.get('next_id', 0)
        
        # Save restored index
        self.save_index()
        
        logging.info(f"Index restored from {backup_path}")
    
    def clear_index(self):
        """Clear all vectors from the index"""
        self._create_new_index()
        self.save_index()
        logging.info("Index cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS store"""
        try:
            active_count = sum(1 for meta in self.id_to_metadata.values() 
                              if not meta.get('deleted', False))
            
            return {
                'vector_count': self.index.ntotal if self.index else 0,
                'dimension': self.dimension,
                'active_vectors': active_count,
                'deleted_vectors': len(self.id_to_metadata) - active_count,
                'metadata_count': len(self.id_to_metadata),
                'index_type': type(self.index).__name__ if self.index else None,
                'next_id': self.next_id,
                'index_size_mb': self.index_path.stat().st_size / (1024 * 1024) if self.index_path.exists() else 0,
                'is_trained': self.index.is_trained if self.index else False,
                'status': 'healthy' if self.index and self.index.ntotal >= 0 else 'error'
            }
        except Exception as e:
            logging.error(f"Error getting FAISS stats: {e}")
            return {
                'vector_count': 0,
                'dimension': self.dimension,
                'active_vectors': 0,
                'deleted_vectors': 0,
                'metadata_count': 0,
                'index_type': None,
                'next_id': 0,
                'index_size_mb': 0,
                'is_trained': False,
                'status': 'error',
                'error': str(e)
            } 