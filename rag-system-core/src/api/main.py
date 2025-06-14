"""
FastAPI Application
Main API application for the RAG system
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from .models.requests import QueryRequest, UploadRequest
from .models.responses import QueryResponse, UploadResponse, HealthResponse
from ..core.error_handling import RAGSystemError
# Global heartbeat monitor - will be set by main.py
heartbeat_monitor = None
from .management_api import create_management_router

# Global thread pool for CPU-intensive tasks
try:
    from ..core.constants import DEFAULT_THREAD_POOL_SIZE
    thread_pool = ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_SIZE)
except ImportError:
    # Fallback if constants not available
    thread_pool = ThreadPoolExecutor(max_workers=4)

def create_api_app(container, monitoring=None, heartbeat_monitor_instance=None) -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Set the global heartbeat monitor
    global heartbeat_monitor
    if heartbeat_monitor_instance:
        heartbeat_monitor = heartbeat_monitor_instance
        logging.info(f"✅ Heartbeat monitor set in API: {type(heartbeat_monitor)}")
    else:
        logging.warning("⚠️ No heartbeat monitor instance provided to API")
    
    # Get configuration
    config_manager = container.get('config_manager')
    config = config_manager.get_config()
    
    # Create FastAPI app
    app = FastAPI(
        title="RAG System API",
        description="Enterprise RAG System with FastAPI, FAISS, and LangGraph",
        version="1.0.0",
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None
    )
    
    # Store heartbeat monitor in app state for reliable access
    app.state.heartbeat_monitor = heartbeat_monitor
    
    # Add CORS middleware
    cors_origins = getattr(config.api, 'cors_origins', [])
    if not cors_origins:
        # Default CORS for development
        cors_origins = ["*"] if config.debug else ["http://localhost:3000", "http://localhost:8080"]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Dependency to get services
    def get_query_engine():
        return container.get('query_engine')
    
    def get_ingestion_engine():
        return container.get('ingestion_engine')
    
    def get_config():
        return config
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        try:
            from datetime import datetime
            # Simple health check without external API calls
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'api': {'status': 'healthy'},
                    'container': {'status': 'healthy', 'services': len(container.list_services())}
                },
                'issues': []
            }
            return HealthResponse(**health_status)
        except Exception as e:
            from datetime import datetime
            return HealthResponse(
                status="error",
                timestamp=datetime.now().isoformat(),
                components={},
                issues=[str(e)]
            )
    
    async def _process_query_async(query_text: str, max_results: int = 3) -> Dict[str, Any]:
        """Process query asynchronously with timeout"""
        def _process_query():
            try:
                # Get components directly from container
                embedder = container.get('embedder')
                faiss_store = container.get('faiss_store')
                llm_client = container.get('llm_client')
                metadata_store = container.get('metadata_store')
                
                # Generate query embedding with timeout
                query_embedding = embedder.embed_text(query_text)
                
                # Search FAISS index
                search_results = faiss_store.search(query_embedding, k=max_results)
                
                # Retrieve context and sources
                context_texts = []
                sources = []
                
                for result in search_results:
                    # Extract text and metadata from FAISS result
                    text = result.get('text', '')
                    score = result.get('similarity_score', 0.0)
                    doc_id = result.get('doc_id', 'unknown')
                    
                    if text:
                        context_texts.append(text)
                        sources.append({
                            "doc_id": doc_id,
                            "text": text[:200],
                            "score": float(score),
                            "metadata": result
                        })
                
                # Generate LLM response with timeout
                if context_texts:
                    context = "\n\n".join(context_texts)
                    prompt = f"""Based on the following context, answer the question: {query_text}

Context:
{context}

Answer:"""
                    
                    response = llm_client.generate(prompt, max_tokens=500)
                else:
                    response = "I couldn't find relevant information to answer your question."
                
                return {
                    "response": response,
                    "sources": sources,
                    "query": query_text,
                    "context_used": len(context_texts)
                }
                
            except Exception as e:
                logging.error(f"Query processing error: {e}")
                raise e
        
        # Run with timeout
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(thread_pool, _process_query),
                timeout=30.0  # 30 second timeout
            )
            return result
        except asyncio.TimeoutError:
            logging.error("Query processing timed out")
            raise HTTPException(status_code=408, detail="Query processing timed out")
        except Exception as e:
            logging.error(f"Query processing failed: {e}")
            raise HTTPException(status_code=500, detail="Query processing failed")
    
    # Query endpoint
    @app.post("/query")
    async def query(request: dict):
        """Process a query and return response with sources"""
        query_text = request.get("query", "")
        max_results = request.get("max_results", 3)
        
        if not query_text:
            raise HTTPException(status_code=400, detail="Query is required")
        
        return await _process_query_async(query_text, max_results)
    
    async def _process_text_ingestion_async(text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process text ingestion asynchronously with timeout"""
        def _process_text():
            try:
                # Get components directly from container
                embedder = container.get('embedder')
                chunker = container.get('chunker')
                faiss_store = container.get('faiss_store')
                metadata_store = container.get('metadata_store')
                
                # Check for existing documents and delete old vectors
                old_vectors_deleted = 0
                doc_path = metadata.get('doc_path')
                if doc_path:
                    # Search for existing vectors with this doc_path
                    existing_vectors = []
                    for vector_id, vector_metadata in faiss_store.id_to_metadata.items():
                        if (not vector_metadata.get('deleted', False) and 
                            vector_metadata.get('doc_path') == doc_path):
                            existing_vectors.append(vector_id)
                    
                    if existing_vectors:
                        logging.info(f"Found {len(existing_vectors)} existing vectors for doc_path: {doc_path}")
                        faiss_store.delete_vectors(existing_vectors)
                        old_vectors_deleted = len(existing_vectors)
                        logging.info(f"Deleted {old_vectors_deleted} old vectors for text update")
                
                # Process the text
                chunks = chunker.chunk_text(text)
                
                if not chunks:
                    return {
                        "status": "error",
                        "message": "No chunks generated from text"
                    }
                
                # Generate embeddings
                chunk_texts = [chunk.get('text', str(chunk)) for chunk in chunks]
                embeddings = embedder.embed_texts(chunk_texts)
                
                # Store in FAISS
                chunk_metadata_list = []
                
                # Generate a better document identifier
                def generate_doc_id(metadata, chunk_index):
                    """Generate a meaningful document ID that includes doc_path"""
                    doc_path = metadata.get('doc_path', '')
                    
                    if doc_path:
                        # Use doc_path as the base for the ID
                        # Remove leading slash and replace special chars
                        doc_id_base = doc_path.strip('/').replace('/', '_').replace(' ', '_')
                        return f"{doc_id_base}_chunk_{chunk_index}"
                    
                    # Fallback to existing logic if no doc_path
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
                    
                    return f"{doc_name}_chunk_{chunk_index}"
                
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk.get('text', str(chunk))
                    # Create flat metadata structure - no nesting
                    chunk_meta = {
                        'text': chunk_text,
                        'chunk_index': i,
                        'doc_id': generate_doc_id(metadata, i),
                        'doc_path': metadata.get('doc_path'),  # Ensure doc_path is at top level
                        'filename': metadata.get('filename'),
                        'title': metadata.get('title'),
                        'description': metadata.get('description'),
                        'source_type': metadata.get('source_type', 'text'),
                        'timestamp': metadata.get('timestamp', time.time()),
                        'operation': metadata.get('operation', 'ingest'),
                        'source': metadata.get('source', 'api')
                    }
                    # Don't nest metadata within metadata
                    chunk_metadata_list.append(chunk_meta)
                
                vector_ids = faiss_store.add_vectors(embeddings, chunk_metadata_list)
                
                # Store metadata
                file_id = metadata_store.add_file_metadata("text_input", metadata)
                for i, (chunk, vector_id) in enumerate(zip(chunks, vector_ids)):
                    chunk_text = chunk.get('text', str(chunk))
                    chunk_metadata = {
                        "file_id": file_id,
                        "chunk_index": i,
                        "text": chunk_text,
                        "vector_id": vector_id,
                        "doc_id": generate_doc_id(metadata, i)
                    }
                    metadata_store.add_chunk_metadata(chunk_metadata)
                
                return {
                    "status": "success",
                    "file_id": file_id,
                    "chunks_created": len(chunks),
                    "embeddings_generated": len(embeddings),
                    "is_update": old_vectors_deleted > 0,
                    "old_vectors_deleted": old_vectors_deleted
                }
                
            except Exception as e:
                logging.error(f"Text ingestion error: {e}")
                raise e
        
        # Run with timeout
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(thread_pool, _process_text),
                timeout=120.0  # 2 minute timeout for ingestion
            )
            return result
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Text ingestion timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text ingestion failed: {str(e)}")
    
    # Text ingestion endpoint
    @app.post("/ingest")
    async def ingest_text(request: dict):
        """Ingest text directly"""
        text = request.get("text", "")
        metadata = request.get("metadata", {})
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        return await _process_text_ingestion_async(text, metadata)
    
    async def _process_file_upload_async(file_content: bytes, filename: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process file upload asynchronously with timeout"""
        def _process_file():
            try:
                import tempfile
                import os
                
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Get ingestion engine
                    ingestion_engine = container.get('ingestion_engine')
                    
                    # Process file
                    result = ingestion_engine.ingest_file(tmp_file_path, metadata)
                    return result
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                        
            except Exception as e:
                logging.error(f"File upload processing error: {e}")
                raise e
        
        # Run with timeout
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(thread_pool, _process_file),
                timeout=300.0  # 5 minute timeout for file processing
            )
            return result
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="File processing timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

    # File upload endpoint
    @app.post("/upload", response_model=UploadResponse)
    async def upload_file(
        file: UploadFile = File(...),
        metadata: Optional[str] = None
    ):
        """Upload and process a file"""
        try:
            # Read file content
            file_content = await file.read()
            
            # Parse metadata if provided
            file_metadata = {}
            if metadata:
                import json
                try:
                    file_metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    file_metadata = {"description": metadata}
            
            # Add file info to metadata
            file_metadata.update({
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size": len(file_content)
            })
            
            # Debug: Log the metadata being passed
            logging.info(f"Upload metadata being passed to ingestion: {file_metadata}")
            
            # Process file
            result = await _process_file_upload_async(file_content, file.filename, file_metadata)
            
            return UploadResponse(
                status="success" if result.get("status") == "success" else "error",
                file_id=result.get("file_id"),
                file_path=result.get("file_path"),
                chunks_created=result.get("chunks_created", 0),
                vectors_stored=result.get("vectors_stored"),
                reason=result.get("reason"),
                is_update=result.get("is_update"),
                old_vectors_deleted=result.get("old_vectors_deleted")
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"File upload error: {e}")
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    # Detailed health check endpoint
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with component testing"""
        try:
            from datetime import datetime
            
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {},
                'issues': []
            }
            
            # Test components with timeout
            try:
                # Test embedder
                embedder = container.get('embedder')
                test_embedding = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        thread_pool, 
                        lambda: embedder.embed_text("test")
                    ),
                    timeout=10.0
                )
                health_status['components']['embedder'] = {
                    'status': 'healthy',
                    'dimension': len(test_embedding)
                }
            except Exception as e:
                health_status['components']['embedder'] = {'status': 'error', 'error': str(e)}
                health_status['issues'].append(f"Embedder error: {e}")
            
            # Test FAISS store
            try:
                faiss_store = container.get('faiss_store')
                stats = faiss_store.get_stats()
                health_status['components']['faiss_store'] = {
                    'status': 'healthy',
                    'vector_count': stats.get('vector_count', 0)
                }
            except Exception as e:
                health_status['components']['faiss_store'] = {'status': 'error', 'error': str(e)}
                health_status['issues'].append(f"FAISS store error: {e}")
            
            # Test LLM client
            try:
                llm_client = container.get('llm_client')
                test_response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        thread_pool,
                        lambda: llm_client.generate("Hello", max_tokens=5)
                    ),
                    timeout=15.0
                )
                health_status['components']['llm_client'] = {
                    'status': 'healthy',
                    'test_response_length': len(test_response) if test_response else 0
                }
            except Exception as e:
                health_status['components']['llm_client'] = {'status': 'error', 'error': str(e)}
                health_status['issues'].append(f"LLM client error: {e}")
            
            # Set overall status
            if health_status['issues']:
                health_status['status'] = 'degraded' if len(health_status['issues']) < 3 else 'unhealthy'
            
            return health_status
            
        except Exception as e:
            from datetime import datetime
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'components': {},
                'issues': [str(e)]
            }

    @app.get("/stats")
    async def get_stats():
        """Get system statistics"""
        try:
            # Get stats with timeout
            def _get_stats():
                faiss_store = container.get('faiss_store')
                metadata_store = container.get('metadata_store')
                embedder = container.get('embedder')
                
                faiss_stats = faiss_store.get_stats()
                metadata_stats = metadata_store.get_stats()
                
                # Get unique documents
                unique_docs = set()
                for vector_id, metadata in faiss_store.id_to_metadata.items():
                    if not metadata.get('deleted', False):
                        doc_id = metadata.get('doc_id', 'unknown')
                        unique_docs.add(doc_id)
                
                # Enhanced stats
                enhanced_stats = {
                    'faiss_store': faiss_stats,
                    'metadata_store': metadata_stats,
                    'timestamp': time.time(),
                    'total_vectors': faiss_stats.get('active_vectors', 0),
                    'total_documents': len(unique_docs),
                    'total_chunks': faiss_stats.get('active_vectors', 0),
                    'embedding_model': getattr(embedder, 'model_name', getattr(embedder, 'model', 'sentence-transformers')),
                    'vector_dimensions': faiss_stats.get('dimension', 384),
                    'index_type': faiss_stats.get('index_type', 'FAISS'),
                    'documents': sorted(list(unique_docs))
                }
                
                return enhanced_stats
            
            loop = asyncio.get_event_loop()
            stats = await asyncio.wait_for(
                loop.run_in_executor(thread_pool, _get_stats),
                timeout=10.0
            )
            return stats
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Stats request timed out")
        except Exception as e:
            logging.error(f"Stats error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

    @app.get("/documents")
    async def get_documents():
        """Get list of all documents in the vector store"""
        try:
            def _get_documents():
                faiss_store = container.get('faiss_store')
                
                # Get unique documents
                unique_docs = set()
                doc_details = {}
                
                for vector_id, metadata in faiss_store.id_to_metadata.items():
                    # Safely handle None metadata
                    if metadata is None:
                        metadata = {}
                    
                    if not metadata.get('deleted', False):
                        doc_id = metadata.get('doc_id', 'unknown')
                        unique_docs.add(doc_id)
                        
                        # Collect document details with safe extraction
                        if doc_id not in doc_details:
                            doc_details[doc_id] = {
                                'doc_id': doc_id,
                                'chunks': 0,
                                'doc_path': metadata.get('doc_path', '') if metadata else '',
                                'filename': metadata.get('filename', '') if metadata else '',
                                'upload_timestamp': metadata.get('upload_timestamp', '') if metadata else '',
                                'source': metadata.get('source', '') if metadata else ''
                            }
                        doc_details[doc_id]['chunks'] += 1
                
                return {
                    "documents": sorted(list(unique_docs)),
                    "total_documents": len(unique_docs),
                    "document_details": list(doc_details.values())
                }
            
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(thread_pool, _get_documents),
                timeout=10.0
            )
            return result
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Documents request timed out")
        except Exception as e:
            logging.error(f"Documents error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

    @app.get("/config")
    async def get_config_info(config=Depends(get_config)):
        """Get configuration information"""
        return {
            'environment': config.environment,
            'debug': config.debug,
            'api': {
                'host': config.api.host,
                'port': config.api.port
            },
            'embedding': {
                'provider': config.embedding.provider,
                'model': config.embedding.model
            },
            'llm': {
                'provider': config.llm.provider,
                'model': config.llm.model
            }
        }

    # ========== COMPREHENSIVE HEARTBEAT ENDPOINTS ==========
    
    @app.get("/heartbeat")
    async def get_heartbeat():
        """Get comprehensive system heartbeat"""
        try:
            if heartbeat_monitor:
                health = await heartbeat_monitor.comprehensive_health_check()
                return health.to_dict()
            else:
                raise HTTPException(status_code=503, detail="Heartbeat monitor not initialized")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health/summary")
    async def get_health_summary():
        """Get health summary (no auth required for monitoring tools)"""
        try:
            if heartbeat_monitor:
                summary = heartbeat_monitor.get_health_summary()
                return summary
            else:
                return {"status": "unknown", "message": "Heartbeat monitor not initialized"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health/components")
    async def get_component_health():
        """Get detailed component health status"""
        try:
            if heartbeat_monitor:
                if not heartbeat_monitor.last_health_check:
                    health = await heartbeat_monitor.comprehensive_health_check()
                else:
                    health = heartbeat_monitor.last_health_check
                
                return {
                    "components": [comp.to_dict() for comp in health.components],
                    "timestamp": health.timestamp
                }
            else:
                raise HTTPException(status_code=503, detail="Heartbeat monitor not initialized")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health/history")
    async def get_health_history(limit: int = 24):
        """Get health check history"""
        try:
            if heartbeat_monitor:
                history = heartbeat_monitor.health_history
                
                # Return recent history
                recent_history = history[-limit:] if len(history) > limit else history
                
                return {
                    "history": recent_history,
                    "total_checks": len(history),
                    "returned_checks": len(recent_history)
                }
            else:
                return {"history": [], "total_checks": 0, "returned_checks": 0}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/health/check")
    async def trigger_health_check():
        """Manually trigger health check"""
        try:
            if heartbeat_monitor:
                health = await heartbeat_monitor.comprehensive_health_check()
                return {
                    "message": "Health check completed",
                    "overall_status": health.overall_status.value,
                    "timestamp": health.timestamp,
                    "component_count": len(health.components)
                }
            else:
                raise HTTPException(status_code=503, detail="Heartbeat monitor not initialized")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/heartbeat/start")
    async def start_heartbeat():
        """Start heartbeat monitoring"""
        try:
            # Try to get heartbeat monitor from multiple sources
            monitor = heartbeat_monitor or getattr(app.state, 'heartbeat_monitor', None)
            if monitor:
                # Check if monitor has the start_monitoring method
                if hasattr(monitor, 'start_monitoring'):
                    monitor.start_monitoring()
                    return {
                        "message": "Heartbeat monitoring started",
                        "status": "active",
                        "timestamp": time.time()
                    }
                else:
                    raise HTTPException(status_code=500, detail="Heartbeat monitor does not support start_monitoring")
            else:
                raise HTTPException(status_code=503, detail="Heartbeat monitor not initialized")
        except Exception as e:
            logging.error(f"Error starting heartbeat: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start heartbeat: {str(e)}")

    @app.post("/heartbeat/stop")
    async def stop_heartbeat():
        """Stop heartbeat monitoring"""
        try:
            # Try to get heartbeat monitor from multiple sources
            monitor = heartbeat_monitor or getattr(app.state, 'heartbeat_monitor', None)
            if monitor:
                # Check if monitor has the stop_monitoring method
                if hasattr(monitor, 'stop_monitoring'):
                    monitor.stop_monitoring()
                    return {
                        "message": "Heartbeat monitoring stopped",
                        "status": "inactive",
                        "timestamp": time.time()
                    }
                else:
                    raise HTTPException(status_code=500, detail="Heartbeat monitor does not support stop_monitoring")
            else:
                raise HTTPException(status_code=503, detail="Heartbeat monitor not initialized")
        except Exception as e:
            logging.error(f"Error stopping heartbeat: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to stop heartbeat: {str(e)}")

    @app.get("/heartbeat/status")
    async def get_heartbeat_status():
        """Get heartbeat monitoring status"""
        try:
            # Try to get heartbeat monitor from multiple sources
            monitor = heartbeat_monitor or getattr(app.state, 'heartbeat_monitor', None)
            logging.info(f"🔍 Heartbeat status check - global: {heartbeat_monitor}, app.state: {getattr(app.state, 'heartbeat_monitor', None)}")
            if monitor:
                is_running = getattr(monitor, 'is_running', False)
                return {
                    "enabled": is_running,
                    "status": "active" if is_running else "inactive",
                    "interval_seconds": getattr(monitor, 'interval', 30),
                    "last_check": getattr(monitor, 'last_check_time', None),
                    "total_checks": len(getattr(monitor, 'health_history', [])),
                    "timestamp": time.time()
                }
            else:
                return {
                    "enabled": False,
                    "status": "not_initialized",
                    "timestamp": time.time()
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/heartbeat/logs")
    async def get_heartbeat_logs(limit: int = 50):
        """Get recent heartbeat logs"""
        try:
            if heartbeat_monitor:
                history = getattr(heartbeat_monitor, 'health_history', [])
                recent_logs = history[-limit:] if len(history) > limit else history
                
                # Format logs for display
                formatted_logs = []
                for log_entry in recent_logs:
                    if isinstance(log_entry, dict):
                        formatted_logs.append({
                            "timestamp": log_entry.get("timestamp", "Unknown"),
                            "status": log_entry.get("overall_status", "Unknown"),
                            "components": len(log_entry.get("components", [])),
                            "message": f"Health check completed - {log_entry.get('overall_status', 'Unknown')}"
                        })
                    else:
                        formatted_logs.append({
                            "timestamp": getattr(log_entry, 'timestamp', 'Unknown'),
                            "status": getattr(log_entry, 'overall_status', 'Unknown'),
                            "components": len(getattr(log_entry, 'components', [])),
                            "message": f"Health check completed - {getattr(log_entry, 'overall_status', 'Unknown')}"
                        })
                
                return {
                    "logs": formatted_logs,
                    "total_logs": len(history),
                    "returned_logs": len(formatted_logs),
                    "timestamp": time.time()
                }
            else:
                return {
                    "logs": [],
                    "total_logs": 0,
                    "returned_logs": 0,
                    "message": "Heartbeat monitor not initialized"
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Folder Monitoring Endpoints
    @app.get("/folder-monitor/status")
    async def get_folder_monitor_status():
        """Get folder monitoring status"""
        try:
            # Import folder monitor
            from ..monitoring.folder_monitor import folder_monitor, initialize_folder_monitor
            
            # If folder monitor is not initialized, try to initialize it
            if not folder_monitor:
                try:
                    config_manager = container.get('config_manager')
                    if config_manager:
                        global_folder_monitor = initialize_folder_monitor(container, config_manager)
                        # Update the global variable
                        import src.monitoring.folder_monitor as fm_module
                        fm_module.folder_monitor = global_folder_monitor
                        logging.info("✅ Folder monitor initialized on-demand")
                except Exception as init_e:
                    logging.error(f"Failed to initialize folder monitor on-demand: {init_e}")
            
            # Try again after initialization
            from ..monitoring.folder_monitor import folder_monitor
            
            if folder_monitor:
                status = folder_monitor.get_status()
                return {
                    "success": True,
                    "status": status,
                    "timestamp": time.time()
                }
            else:
                return {
                    "success": False,
                    "error": "Folder monitor not initialized",
                    "timestamp": time.time()
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/folder-monitor/add")
    async def add_monitored_folder(request: dict):
        """Add a folder to monitoring"""
        try:
            folder_path = request.get('folder_path')
            if not folder_path:
                raise HTTPException(status_code=400, detail="folder_path is required")
            
            from ..monitoring.folder_monitor import folder_monitor, initialize_folder_monitor
            
            # If folder monitor is not initialized, try to initialize it
            if not folder_monitor:
                try:
                    config_manager = container.get('config_manager')
                    if config_manager:
                        global_folder_monitor = initialize_folder_monitor(container, config_manager)
                        # Update the global variable
                        import src.monitoring.folder_monitor as fm_module
                        fm_module.folder_monitor = global_folder_monitor
                        logging.info("✅ Folder monitor initialized on-demand for add operation")
                except Exception as init_e:
                    logging.error(f"Failed to initialize folder monitor on-demand: {init_e}")
                    raise HTTPException(status_code=503, detail=f"Folder monitor initialization failed: {init_e}")
            
            # Try again after initialization
            from ..monitoring.folder_monitor import folder_monitor
            
            if not folder_monitor:
                raise HTTPException(status_code=503, detail="Folder monitor not initialized")
            
            result = folder_monitor.add_folder(folder_path)
            
            if result.get('success'):
                # Automatically trigger a scan after adding folder for immediate feedback
                try:
                    scan_result = folder_monitor.force_scan()
                    scan_info = {
                        "immediate_scan": True,
                        "changes_detected": scan_result.get('changes_detected', 0),
                        "files_tracked": scan_result.get('files_tracked', 0)
                    }
                except Exception as scan_e:
                    logging.warning(f"Failed to trigger immediate scan after adding folder: {scan_e}")
                    scan_info = {"immediate_scan": False, "scan_error": str(scan_e)}
                
                return {
                    "success": True,
                    "message": result.get('message'),
                    "files_found": result.get('files_found', 0),
                    "folder_path": folder_path,
                    "timestamp": time.time(),
                    **scan_info
                }
            else:
                raise HTTPException(status_code=400, detail=result.get('error', 'Failed to add folder'))
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/folder-monitor/remove")
    async def remove_monitored_folder(request: dict):
        """Remove a folder from monitoring"""
        try:
            folder_path = request.get('folder_path')
            if not folder_path:
                raise HTTPException(status_code=400, detail="folder_path is required")
            
            from ..monitoring.folder_monitor import folder_monitor
            
            if not folder_monitor:
                raise HTTPException(status_code=503, detail="Folder monitor not initialized")
            
            result = folder_monitor.remove_folder(folder_path)
            
            if result.get('success'):
                return {
                    "success": True,
                    "message": result.get('message'),
                    "files_removed": result.get('files_removed', 0),
                    "folder_path": folder_path,
                    "timestamp": time.time()
                }
            else:
                raise HTTPException(status_code=400, detail=result.get('error', 'Failed to remove folder'))
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/folder-monitor/folders")
    async def get_monitored_folders():
        """Get list of monitored folders"""
        try:
            from ..monitoring.folder_monitor import folder_monitor
            
            if folder_monitor:
                folders = folder_monitor.get_monitored_folders()
                return {
                    "success": True,
                    "folders": folders,
                    "count": len(folders),
                    "timestamp": time.time()
                }
            else:
                return {
                    "success": False,
                    "folders": [],
                    "count": 0,
                    "error": "Folder monitor not initialized",
                    "timestamp": time.time()
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/folder-monitor/start")
    async def start_folder_monitoring():
        """Start folder monitoring"""
        try:
            from ..monitoring.folder_monitor import folder_monitor
            
            if not folder_monitor:
                raise HTTPException(status_code=503, detail="Folder monitor not initialized")
            
            result = folder_monitor.start_monitoring()
            
            if result.get('success'):
                return {
                    "success": True,
                    "message": result.get('message'),
                    "folders": result.get('folders', []),
                    "interval": result.get('interval', 60),
                    "timestamp": time.time()
                }
            else:
                raise HTTPException(status_code=400, detail=result.get('error', 'Failed to start monitoring'))
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/folder-monitor/stop")
    async def stop_folder_monitoring():
        """Stop folder monitoring"""
        try:
            from ..monitoring.folder_monitor import folder_monitor
            
            if not folder_monitor:
                raise HTTPException(status_code=503, detail="Folder monitor not initialized")
            
            result = folder_monitor.stop_monitoring()
            
            if result.get('success'):
                return {
                    "success": True,
                    "message": result.get('message'),
                    "timestamp": time.time()
                }
            else:
                raise HTTPException(status_code=400, detail=result.get('error', 'Failed to stop monitoring'))
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/folder-monitor/scan")
    async def force_folder_scan():
        """Force an immediate scan of all monitored folders"""
        try:
            from ..monitoring.folder_monitor import folder_monitor
            
            if not folder_monitor:
                raise HTTPException(status_code=503, detail="Folder monitor not initialized")
            
            result = folder_monitor.force_scan()
            
            if result.get('success'):
                return {
                    "success": True,
                    "message": result.get('message'),
                    "changes_detected": result.get('changes_detected', 0),
                    "files_tracked": result.get('files_tracked', 0),
                    "timestamp": time.time()
                }
            else:
                raise HTTPException(status_code=500, detail=result.get('error', 'Scan failed'))
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/folder-monitor/files")
    async def get_monitored_files():
        """Get status of all monitored files"""
        try:
            from ..monitoring.folder_monitor import folder_monitor
            
            if folder_monitor:
                file_states = folder_monitor.get_file_states()
                return {
                    "success": True,
                    "files": file_states,
                    "count": len(file_states),
                    "timestamp": time.time()
                }
            else:
                return {
                    "success": False,
                    "files": {},
                    "count": 0,
                    "error": "Folder monitor not initialized",
                    "timestamp": time.time()
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/documents/{doc_path:path}")
    async def delete_document(doc_path: str):
        """Delete a specific document and its vectors from the system"""
        try:
            faiss_store = container.get('faiss_store')
            metadata_store = container.get('metadata_store')
            
            # Find vectors associated with this document
            vectors_to_delete = []
            for vector_id, metadata in faiss_store.id_to_metadata.items():
                if (not metadata.get('deleted', False) and 
                    metadata.get('doc_path') == doc_path):
                    vectors_to_delete.append(vector_id)
            
            if not vectors_to_delete:
                return {
                    "status": "warning",
                    "message": f"No vectors found for document: {doc_path}",
                    "vectors_deleted": 0,
                    "doc_path": doc_path
                }
            
            # Delete the vectors
            deleted_count = faiss_store.delete_vectors(vectors_to_delete)
            
            return {
                "status": "success",
                "message": f"Document deleted successfully: {doc_path}",
                "vectors_deleted": deleted_count,
                "doc_path": doc_path,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logging.error(f"Error deleting document {doc_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

    @app.post("/clear")
    async def clear_vector_store():
        """Clear all vectors and documents from the system"""
        try:
            faiss_store = container.get('faiss_store')
            metadata_store = container.get('metadata_store')
            
            # Get stats before clearing
            stats_before = faiss_store.get_stats()
            vectors_before = stats_before.get('active_vectors', 0)
            
            # Get document count before clearing
            documents_before = len(set(
                metadata.get('doc_id', 'unknown') 
                for metadata in faiss_store.id_to_metadata.values()
                if not metadata.get('deleted', False)
            ))
            
            # Get chunk count before clearing
            chunks_before = len([
                metadata for metadata in faiss_store.id_to_metadata.values()
                if not metadata.get('deleted', False)
            ])
            
            # Clear the FAISS store
            faiss_store.clear_index()
            
            # Clear metadata store if it has a clear method
            try:
                if hasattr(metadata_store, 'clear_all_data'):
                    metadata_store.clear_all_data()
                elif hasattr(metadata_store, 'clear'):
                    metadata_store.clear()
            except Exception as e:
                logging.warning(f"Could not clear metadata store: {e}")
            
            return {
                "status": "success",
                "message": "Vector store cleared successfully",
                "vectors_deleted": vectors_before,
                "documents_deleted": documents_before,
                "chunks_deleted": chunks_before,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logging.error(f"Error clearing vector store: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to clear vector store: {str(e)}")

    @app.get("/health/performance")
    async def get_performance_metrics():
        """Get detailed performance metrics"""
        try:
            if heartbeat_monitor:
                # Get current performance metrics
                metrics = await heartbeat_monitor._get_performance_metrics()
                
                # Add additional metrics if available
                try:
                    faiss_store = container.get('faiss_store')
                    stats = faiss_store.get_stats()
                    metrics.update({
                        'vector_store_metrics': stats
                    })
                except Exception:
                    pass
                
                return metrics
            else:
                raise HTTPException(status_code=503, detail="Heartbeat monitor not initialized")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.exception_handler(RAGSystemError)
    async def rag_error_handler(request, exc: RAGSystemError):
        """Handle RAG system specific errors"""
        return JSONResponse(
            status_code=500,
            content={
                "error": "RAG System Error",
                "message": str(exc),
                "type": exc.__class__.__name__
            }
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request, exc: Exception):
        """Handle general exceptions"""
        logging.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred"
            }
        )

    # Add management API router
    management_router = create_management_router(container)
    app.include_router(management_router)
    
    # Add ServiceNow API router
    try:
        from .routes.servicenow import router as servicenow_router
        app.include_router(servicenow_router, prefix="/api")
        logging.info("✅ ServiceNow API routes registered")
    except Exception as e:
        logging.warning(f"⚠️ ServiceNow API routes not available: {e}")

    # Startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Initialize resources on startup"""
        logging.info("🚀 RAG System API starting up...")
        
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown"""
        logging.info("🛑 RAG System API shutting down...")
        # Shutdown thread pool gracefully
        thread_pool.shutdown(wait=True)
        logging.info("✅ Thread pool shutdown complete")

    logging.info("FastAPI application created")
    return app