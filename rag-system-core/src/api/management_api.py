"""
Management API for RAG System
Provides endpoints for viewing, cleaning, and managing vectors, chunks, and documents
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel

# Response models
class VectorInfo(BaseModel):
    vector_id: int
    doc_id: str
    text_preview: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None

class DocumentInfo(BaseModel):
    doc_id: str
    title: Optional[str] = None
    filename: Optional[str] = None
    chunk_count: int
    total_text_length: int
    metadata: Dict[str, Any]
    created_at: Optional[str] = None

class CleanupResult(BaseModel):
    action: str
    affected_count: int
    details: List[str]

class UpdateRequest(BaseModel):
    vector_ids: Optional[List[int]] = None
    doc_ids: Optional[List[str]] = None
    updates: Dict[str, Any]

def create_management_router(container) -> APIRouter:
    """Create management API router"""
    router = APIRouter(prefix="/manage", tags=["management"])
    
    @router.get("/vectors", response_model=List[VectorInfo])
    async def list_vectors(
        limit: int = Query(50, description="Maximum number of vectors to return"),
        offset: int = Query(0, description="Number of vectors to skip"),
        doc_id_filter: Optional[str] = Query(None, description="Filter by document ID pattern"),
        text_search: Optional[str] = Query(None, description="Search in text content")
    ):
        """List all vectors with optional filtering"""
        try:
            faiss_store = container.get('faiss_store')
            
            # Get all vector metadata
            all_vectors = []
            for vector_id, metadata in faiss_store.id_to_metadata.items():
                if metadata.get('deleted', False):
                    continue
                
                # Apply doc_id filter
                if doc_id_filter and doc_id_filter.lower() not in metadata.get('doc_id', '').lower():
                    continue
                
                # Apply text search
                if text_search and text_search.lower() not in metadata.get('text', '').lower():
                    continue
                
                vector_info = VectorInfo(
                    vector_id=vector_id,
                    doc_id=metadata.get('doc_id', 'unknown'),
                    text_preview=metadata.get('text', '')[:200],
                    metadata=metadata
                )
                all_vectors.append(vector_info)
            
            # Apply pagination
            start = offset
            end = offset + limit
            return all_vectors[start:end]
            
        except Exception as e:
            logging.error(f"Error listing vectors: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list vectors: {str(e)}")
    
    @router.get("/documents", response_model=List[DocumentInfo])
    async def list_documents(
        limit: int = Query(50, description="Maximum number of documents to return"),
        title_filter: Optional[str] = Query(None, description="Filter by title pattern")
    ):
        """List all documents grouped by doc_id"""
        try:
            faiss_store = container.get('faiss_store')
            
            # Group vectors by doc_id
            documents = {}
            for vector_id, metadata in faiss_store.id_to_metadata.items():
                # Safely handle None metadata
                if metadata is None:
                    metadata = {}
                
                if metadata.get('deleted', False):
                    continue
                
                doc_id = metadata.get('doc_id', 'unknown')
                
                if doc_id not in documents:
                    documents[doc_id] = {
                        'doc_id': doc_id,
                        'title': metadata.get('title') if metadata else None,
                        'filename': metadata.get('filename') if metadata else None,
                        'chunks': [],
                        'total_text_length': 0,
                        'metadata': metadata if metadata else {},
                        'created_at': metadata.get('added_at') if metadata else None
                    }
                
                documents[doc_id]['chunks'].append({
                    'vector_id': vector_id,
                    'text': metadata.get('text', '') if metadata else '',
                    'chunk_index': metadata.get('chunk_index', 0) if metadata else 0
                })
                documents[doc_id]['total_text_length'] += len(metadata.get('text', '') if metadata else '')
            
            # Convert to response format
            doc_list = []
            for doc_id, doc_data in documents.items():
                # Apply title filter safely
                title = doc_data.get('title') or ''
                if title_filter and title_filter.lower() not in title.lower():
                    continue
                
                doc_info = DocumentInfo(
                    doc_id=doc_id,
                    title=doc_data.get('title'),
                    filename=doc_data.get('filename'),
                    chunk_count=len(doc_data['chunks']),
                    total_text_length=doc_data['total_text_length'],
                    metadata=doc_data['metadata'],
                    created_at=doc_data.get('created_at')
                )
                doc_list.append(doc_info)
            
            # Apply limit
            return doc_list[:limit]
            
        except Exception as e:
            logging.error(f"Error listing documents: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")
    
    @router.get("/vector/{vector_id}")
    async def get_vector_details(vector_id: int):
        """Get detailed information about a specific vector"""
        try:
            faiss_store = container.get('faiss_store')
            
            metadata = faiss_store.get_vector_metadata(vector_id)
            if not metadata:
                raise HTTPException(status_code=404, detail="Vector not found")
            
            return {
                'vector_id': vector_id,
                'metadata': metadata,
                'full_text': metadata.get('text', ''),
                'doc_id': metadata.get('doc_id', 'unknown'),
                'chunk_index': metadata.get('chunk_index', 0),
                'created_at': metadata.get('added_at'),
                'is_deleted': metadata.get('deleted', False)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error getting vector details: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get vector details: {str(e)}")
    
    @router.get("/document/{doc_id}")
    async def get_document_details(doc_id: str):
        """Get detailed information about a specific document"""
        try:
            faiss_store = container.get('faiss_store')
            
            # Find all chunks for this document
            chunks = []
            for vector_id, metadata in faiss_store.id_to_metadata.items():
                # Safely handle None metadata
                if metadata is None:
                    metadata = {}
                
                if metadata.get('doc_id') == doc_id and not metadata.get('deleted', False):
                    chunks.append({
                        'vector_id': vector_id,
                        'chunk_index': metadata.get('chunk_index', 0),
                        'text': metadata.get('text', ''),
                        'metadata': metadata
                    })
            
            if not chunks:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Sort chunks by index
            chunks.sort(key=lambda x: x['chunk_index'])
            
            # Get document metadata from first chunk
            doc_metadata = chunks[0]['metadata']
            
            return {
                'doc_id': doc_id,
                'title': doc_metadata.get('title'),
                'filename': doc_metadata.get('filename'),
                'description': doc_metadata.get('description'),
                'chunk_count': len(chunks),
                'total_text_length': sum(len(chunk['text']) for chunk in chunks),
                'chunks': chunks,
                'metadata': doc_metadata,
                'created_at': doc_metadata.get('added_at')
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error getting document details: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get document details: {str(e)}")
    
    @router.delete("/vectors")
    async def delete_vectors(vector_ids: List[int] = Body(..., description="List of vector IDs to delete")):
        """Delete specific vectors"""
        try:
            faiss_store = container.get('faiss_store')
            
            # Validate vector IDs exist
            valid_ids = []
            for vector_id in vector_ids:
                if vector_id in faiss_store.id_to_metadata:
                    valid_ids.append(vector_id)
            
            if not valid_ids:
                raise HTTPException(status_code=404, detail="No valid vector IDs found")
            
            # Delete vectors
            faiss_store.delete_vectors(valid_ids)
            
            return CleanupResult(
                action="delete_vectors",
                affected_count=len(valid_ids),
                details=[f"Deleted vector {vid}" for vid in valid_ids]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error deleting vectors: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete vectors: {str(e)}")
    
    @router.delete("/documents")
    async def delete_documents(doc_ids: List[str] = Body(..., description="List of document IDs to delete")):
        """Delete entire documents (all their chunks)"""
        try:
            faiss_store = container.get('faiss_store')
            
            deleted_count = 0
            deleted_vectors = []
            
            for doc_id in doc_ids:
                # Find all vectors for this document
                vector_ids = []
                for vector_id, metadata in faiss_store.id_to_metadata.items():
                    # Safely handle None metadata
                    if metadata is None:
                        metadata = {}
                    
                    if metadata.get('doc_id') == doc_id and not metadata.get('deleted', False):
                        vector_ids.append(vector_id)
                
                if vector_ids:
                    faiss_store.delete_vectors(vector_ids)
                    deleted_count += len(vector_ids)
                    deleted_vectors.extend(vector_ids)
                    logging.info(f"Deleted document {doc_id} with {len(vector_ids)} vectors")
                else:
                    logging.warning(f"Document {doc_id} not found or already deleted")
            
            return CleanupResult(
                action="delete_documents",
                affected_count=deleted_count,
                details=[f"Deleted document {doc_id}" for doc_id in doc_ids]
            )
            
        except Exception as e:
            logging.error(f"Error deleting documents: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete documents: {str(e)}")
    
    @router.post("/cleanup/unknown")
    async def cleanup_unknown_documents():
        """Clean up all documents with 'unknown' or generic IDs"""
        try:
            faiss_store = container.get('faiss_store')
            
            # Find vectors with unknown or generic doc_ids
            unknown_patterns = [
                r'^doc_unknown_\d+$',
                r'^doc_\d+_[a-f0-9]+$',  # timestamp_hash pattern
                r'^unknown$',
                r'^doc_$'
            ]
            
            vectors_to_delete = []
            for vector_id, metadata in faiss_store.id_to_metadata.items():
                # Safely handle None metadata
                if metadata is None:
                    metadata = {}
                
                if metadata.get('deleted', False):
                    continue
                
                doc_id = metadata.get('doc_id', '')
                for pattern in unknown_patterns:
                    if re.match(pattern, doc_id):
                        vectors_to_delete.append(vector_id)
                        break
            
            if vectors_to_delete:
                faiss_store.delete_vectors(vectors_to_delete)
            
            return CleanupResult(
                action="cleanup_unknown",
                affected_count=len(vectors_to_delete),
                details=[f"Cleaned up {len(vectors_to_delete)} unknown documents"]
            )
            
        except Exception as e:
            logging.error(f"Error cleaning up unknown documents: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to cleanup unknown documents: {str(e)}")
    
    @router.post("/cleanup/duplicates")
    async def cleanup_duplicate_documents():
        """Remove duplicate documents based on text similarity"""
        try:
            faiss_store = container.get('faiss_store')
            
            # Group documents by similar text content
            documents = {}
            for vector_id, metadata in faiss_store.id_to_metadata.items():
                # Safely handle None metadata
                if metadata is None:
                    metadata = {}
                
                if metadata.get('deleted', False):
                    continue
                
                text = metadata.get('text', '')
                text_hash = hash(text.strip().lower())
                
                if text_hash not in documents:
                    documents[text_hash] = []
                documents[text_hash].append(vector_id)
            
            # Find duplicates (groups with more than one vector)
            duplicates_to_delete = []
            for text_hash, vector_ids in documents.items():
                if len(vector_ids) > 1:
                    # Keep the first one, delete the rest
                    duplicates_to_delete.extend(vector_ids[1:])
            
            if duplicates_to_delete:
                faiss_store.delete_vectors(duplicates_to_delete)
            
            return CleanupResult(
                action="cleanup_duplicates",
                affected_count=len(duplicates_to_delete),
                details=[f"Removed {len(duplicates_to_delete)} duplicate vectors"]
            )
            
        except Exception as e:
            logging.error(f"Error cleaning up duplicates: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to cleanup duplicates: {str(e)}")
    
    @router.put("/update")
    async def update_metadata(request: UpdateRequest):
        """Update metadata for vectors or documents"""
        try:
            faiss_store = container.get('faiss_store')
            updated_count = 0
            
            # Update specific vectors
            if request.vector_ids:
                for vector_id in request.vector_ids:
                    if vector_id in faiss_store.id_to_metadata:
                        faiss_store.update_metadata(vector_id, request.updates)
                        updated_count += 1
            
            # Update by document IDs
            if request.doc_ids:
                for vector_id, metadata in faiss_store.id_to_metadata.items():
                    # Safely handle None metadata
                    if metadata is None:
                        metadata = {}
                    
                    if metadata.get('doc_id') in request.doc_ids:
                        faiss_store.update_metadata(vector_id, request.updates)
                        updated_count += 1
            
            return {
                "updated_count": updated_count,
                "updates_applied": request.updates
            }
            
        except Exception as e:
            logging.error(f"Error updating metadata: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to update metadata: {str(e)}")
    
    @router.post("/reindex/doc_ids")
    async def reindex_document_ids():
        """Regenerate document IDs using improved naming logic"""
        try:
            faiss_store = container.get('faiss_store')
            
            def generate_better_doc_id(metadata, chunk_index):
                """Generate improved document ID"""
                title = metadata.get('title', '').strip()
                filename = metadata.get('filename', '').strip()
                description = metadata.get('description', '').strip()
                
                if title:
                    doc_name = title.replace(' ', '_').replace('/', '_').replace('\\', '_')[:50]
                elif filename:
                    import os
                    doc_name = os.path.splitext(filename)[0].replace(' ', '_').replace('/', '_').replace('\\', '_')[:50]
                elif description:
                    words = description.split()[:5]
                    doc_name = '_'.join(words).replace('/', '_').replace('\\', '_')[:50]
                else:
                    import hashlib
                    import time
                    content_hash = hashlib.md5(str(metadata).encode()).hexdigest()[:8]
                    timestamp = str(int(time.time()))[-6:]
                    doc_name = f"doc_{timestamp}_{content_hash}"
                
                return f"doc_{doc_name}_{chunk_index}"
            
            updated_count = 0
            for vector_id, metadata in faiss_store.id_to_metadata.items():
                # Safely handle None metadata
                if metadata is None:
                    metadata = {}
                
                if metadata.get('deleted', False):
                    continue
                
                old_doc_id = metadata.get('doc_id', '')
                chunk_index = metadata.get('chunk_index', 0)
                new_doc_id = generate_better_doc_id(metadata, chunk_index)
                
                if old_doc_id != new_doc_id:
                    faiss_store.update_metadata(vector_id, {'doc_id': new_doc_id})
                    updated_count += 1
            
            return CleanupResult(
                action="reindex_doc_ids",
                affected_count=updated_count,
                details=[f"Updated {updated_count} document IDs with improved naming"]
            )
            
        except Exception as e:
            logging.error(f"Error reindexing document IDs: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to reindex document IDs: {str(e)}")
    
    @router.get("/stats/detailed")
    async def get_detailed_stats():
        """Get detailed statistics about the vector store"""
        try:
            faiss_store = container.get('faiss_store')
            
            # Basic stats
            total_vectors = len(faiss_store.id_to_metadata)
            active_vectors = sum(1 for meta in faiss_store.id_to_metadata.values() 
                               if meta is not None and not meta.get('deleted', False))
            deleted_vectors = total_vectors - active_vectors
            
            # Document stats
            documents = {}
            unknown_docs = 0
            
            for vector_id, metadata in faiss_store.id_to_metadata.items():
                # Safely handle None metadata
                if metadata is None:
                    metadata = {}
                
                if metadata.get('deleted', False):
                    continue
                
                doc_id = metadata.get('doc_id', 'unknown')
                if 'unknown' in doc_id.lower():
                    unknown_docs += 1
                
                if doc_id not in documents:
                    documents[doc_id] = 0
                documents[doc_id] += 1
            
            # Text length stats
            text_lengths = []
            for metadata in faiss_store.id_to_metadata.values():
                # Safely handle None metadata
                if metadata is None:
                    metadata = {}
                
                if not metadata.get('deleted', False):
                    text_lengths.append(len(metadata.get('text', '')))
            
            avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
            
            return {
                'total_vectors': total_vectors,
                'active_vectors': active_vectors,
                'deleted_vectors': deleted_vectors,
                'total_documents': len(documents),
                'unknown_documents': unknown_docs,
                'avg_chunks_per_document': active_vectors / len(documents) if documents else 0,
                'avg_text_length_per_chunk': avg_text_length,
                'largest_document_chunks': max(documents.values()) if documents else 0,
                'smallest_document_chunks': min(documents.values()) if documents else 0,
                'documents_by_chunk_count': dict(sorted(documents.items(), key=lambda x: x[1], reverse=True)[:10])
            }
            
        except Exception as e:
            logging.error(f"Error getting detailed stats: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get detailed stats: {str(e)}")
    
    return router 