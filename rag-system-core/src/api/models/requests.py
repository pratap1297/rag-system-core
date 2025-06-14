"""
API Request Models
Pydantic models for API requests
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="The user's query", min_length=1, max_length=1000)
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters for search")
    top_k: Optional[int] = Field(5, description="Number of results to return", ge=1, le=20)
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the company policy on remote work?",
                "filters": {"source_type": "policy"},
                "top_k": 5
            }
        }

class UploadRequest(BaseModel):
    """Request model for file upload"""
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the file")
    
    class Config:
        schema_extra = {
            "example": {
                "metadata": {
                    "department": "HR",
                    "category": "policy",
                    "tags": ["remote-work", "policy"]
                }
            }
        }

class ConfigUpdateRequest(BaseModel):
    """Request model for configuration updates"""
    component: str = Field(..., description="Configuration component to update")
    updates: Dict[str, Any] = Field(..., description="Configuration updates")
    
    class Config:
        schema_extra = {
            "example": {
                "component": "retrieval",
                "updates": {
                    "top_k": 10,
                    "similarity_threshold": 0.8
                }
            }
        } 