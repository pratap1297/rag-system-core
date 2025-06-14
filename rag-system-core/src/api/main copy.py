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
thread_pool = ThreadPoolExecutor(max_workers=4)

def create_api_app(container, monitoring=None) -> FastAPI:
    """Create and configure FastAPI application"""
    
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
    
    # Add CORS middleware
    if config.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.cors_origins,
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
            raise HTTPException(status_code=408, detail="Query processing timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    
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
                    """Generate a meaningful document ID"""
                    # Try different metadata fields in order of preference
                    title = metadata.get('title', '').strip()
                    filename = metadata.get('filename', '').strip()
                    description = metadata.get('description', '').strip()
                    
                    if title:
                        # Use title, sanitized for ID
                        doc_name = title.replace(' ', '_').replace('/', '_').replace('\\', '_')[:50]
                    elif filename:
                        # Use filename without extension
                        import os
                        doc_name = os.path.splitext(filename)[0].replace(' ', '_').replace('/', '_').replace('\\', '_')[:50]
                    elif description:
                        # Use first few words of description
                        words = description.split()[:5]
                        doc_name = '_'.join(words).replace('/', '_').replace('\\', '_')[:50]
                    else:
                        # Generate from content hash or timestamp
                        import hashlib
                        import time
                        content_hash = hashlib.md5(str(metadata).encode()).hexdigest()[:8]
                        timestamp = str(int(time.time()))[-6:]  # Last 6 digits of timestamp
                        doc_name = f"doc_{timestamp}_{content_hash}"
                    
                    return f"doc_{doc_name}_{chunk_index}"
                
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk.get('text', str(chunk))
                    chunk_meta = {
                        'text': chunk_text,
                        'chunk_index': i,
                        'doc_id': generate_doc_id(metadata, i),
                        **metadata  # Include original metadata
                    }
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
                message=f"File processed successfully: {result.get('chunks_created', 0)} chunks created",
                file_id=result.get("file_id"),
                chunks_created=result.get("chunks_created", 0)
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
                
                faiss_stats = faiss_store.get_stats()
                metadata_stats = metadata_store.get_stats()
                
                return {
                    'faiss_store': faiss_stats,
                    'metadata_store': metadata_stats,
                    'timestamp': time.time()
                }
            
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

    logging.info("FastAPI application created")
    return app 