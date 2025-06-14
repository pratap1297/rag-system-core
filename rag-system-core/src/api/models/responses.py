"""
API Response Models
Pydantic models for API responses
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class SourceInfo(BaseModel):
    """Information about a source document"""
    text: str = Field(..., description="Excerpt from the source")
    similarity_score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(..., description="Source metadata")
    source_type: str = Field(..., description="Type of source")

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    query: str = Field(..., description="The original query")
    response: str = Field(..., description="Generated response")
    sources: List[SourceInfo] = Field(..., description="Source documents used")
    total_sources: int = Field(..., description="Total number of sources found")
    timestamp: str = Field(..., description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the company policy on remote work?",
                "response": "According to the company policy, remote work is allowed...",
                "sources": [
                    {
                        "text": "Remote work policy states that employees may work...",
                        "similarity_score": 0.95,
                        "metadata": {"department": "HR", "document": "policy.pdf"},
                        "source_type": "policy"
                    }
                ],
                "total_sources": 3,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }

class UploadResponse(BaseModel):
    """Response model for file upload"""
    status: str = Field(..., description="Upload status")
    file_id: Optional[str] = Field(None, description="Generated file ID")
    file_path: Optional[str] = Field(None, description="File path")
    chunks_created: Optional[int] = Field(None, description="Number of chunks created")
    vectors_stored: Optional[int] = Field(None, description="Number of vectors stored")
    reason: Optional[str] = Field(None, description="Reason for skipping (if applicable)")
    is_update: Optional[bool] = Field(None, description="Whether this was an update operation")
    old_vectors_deleted: Optional[int] = Field(None, description="Number of old vectors deleted during update")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "file_id": "file_abc123",
                "file_path": "/uploads/document.pdf",
                "chunks_created": 15,
                "vectors_stored": 15
            }
        }

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Overall system status")
    timestamp: Optional[str] = Field(None, description="Health check timestamp")
    components: Dict[str, Any] = Field(..., description="Component health status")
    issues: List[str] = Field(..., description="List of issues found")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "components": {
                    "config": {"status": "healthy"},
                    "faiss_store": {"status": "healthy", "index_size": 1000},
                    "embedder": {"status": "healthy", "dimension": 384}
                },
                "issues": []
            }
        }

class StatsResponse(BaseModel):
    """Response model for system statistics"""
    total_files: int = Field(..., description="Total number of files ingested")
    total_chunks: int = Field(..., description="Total number of chunks")
    total_vectors: int = Field(..., description="Total number of vectors")
    collections: int = Field(..., description="Number of collections")
    active_vectors: Optional[int] = Field(None, description="Number of active vectors")
    
    class Config:
        schema_extra = {
            "example": {
                "total_files": 50,
                "total_chunks": 750,
                "total_vectors": 750,
                "collections": 3,
                "active_vectors": 750
            }
        } 