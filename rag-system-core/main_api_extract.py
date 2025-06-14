from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
import shutil
import tempfile
from typing import List, Dict, Any, Optional, Union
import numpy as np
import faiss
import json
import os
from datetime import datetime
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(src_dir))

from src.document_processing.text_extractor import TextExtractor
from src.document_processing.text_chunker import TextChunker
from src.document_processing.vector_embedder import VectorEmbedder, create_vector_embedder
from src.document_processing.advanced_search import AdvancedSearch

app = FastAPI(title="Document Processing API")

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
extractor = TextExtractor()
chunker = TextChunker(chunk_size=1000, overlap=200)
embedder = create_vector_embedder(provider="sentence-transformers")
advanced_search = AdvancedSearch(embedder)
lemmatizer = WordNetLemmatizer()

# Initialize FAISS with multiple index types
DIMENSION = 768  # Default dimension for sentence-transformers

# HNSW index for fast approximate search
M = 16  # Number of connections per node in HNSW
hnsw_index = faiss.IndexHNSWFlat(DIMENSION, M)
hnsw_index.hnsw.efConstruction = 40
hnsw_index.hnsw.efSearch = 16

# IVF index for large-scale datasets
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(DIMENSION)
ivf_index = faiss.IndexIVFFlat(quantizer, DIMENSION, nlist)
ivf_index.train(np.random.rand(1000, DIMENSION).astype('float32'))  # Train with random data

# Initialize GPU if available
if torch.cuda.is_available():
    res = faiss.StandardGpuResources()
    hnsw_index = faiss.index_cpu_to_gpu(res, 0, hnsw_index)
    ivf_index = faiss.index_cpu_to_gpu(res, 0, ivf_index)

chunk_store = {}  # Store chunk text and metadata
current_id = 0

class SearchQuery(BaseModel):
    query: str
    k: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None
    use_hybrid_search: bool = True
    min_score: float = 0.0
    use_query_expansion: bool = True
    use_semantic_routing: bool = True
    index_type: str = "hnsw"  # "hnsw" or "ivf"
    facets: Optional[Dict[str, List[str]]] = None

class ProcessOptions(BaseModel):
    chunk_size: int = 1000
    overlap: int = 200
    use_advanced_chunking: bool = True
    use_semantic_clustering: bool = True
    n_clusters: int = 5

def expand_query(query: str) -> List[str]:
    """Expand query using WordNet synonyms and lemmatization"""
    expanded_queries = [query]
    
    # Tokenize and lemmatize
    tokens = word_tokenize(query.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Get synonyms for each token
    for token in lemmatized_tokens:
        synsets = wordnet.synsets(token)
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name() != token:
                    # Create new query with synonym
                    new_query = query.replace(token, lemma.name())
                    expanded_queries.append(new_query)
    
    return expanded_queries

def semantic_route(query: str, k: int = 5) -> List[int]:
    """Route query to most relevant clusters using semantic similarity"""
    query_embedding = embedder.embed_text(query)
    
    # Get cluster centroids
    centroids = []
    for i in range(nlist):
        if ivf_index.ntotal > 0:
            cluster_vectors = ivf_index.get_cluster_vectors(i)
            if len(cluster_vectors) > 0:
                centroids.append(np.mean(cluster_vectors, axis=0))
    
    if not centroids:
        return []
    
    # Calculate similarity with centroids
    similarities = cosine_similarity([query_embedding], centroids)[0]
    top_clusters = np.argsort(similarities)[-k:]
    
    return top_clusters.tolist()

def hybrid_search(query: str, k: int = 5, filter_metadata: Optional[Dict] = None,
                 use_query_expansion: bool = True, use_semantic_routing: bool = True,
                 index_type: str = "hnsw", facets: Optional[Dict[str, List[str]]] = None) -> List[Dict]:
    """Enhanced hybrid search with query expansion and semantic routing"""
    # Query expansion
    if use_query_expansion:
        queries = expand_query(query)
    else:
        queries = [query]
    
    all_results = []
    
    for expanded_query in queries:
        # Vector search
        query_embedding = embedder.embed_text(expanded_query)
        
        # Select index based on type
        if index_type == "ivf" and use_semantic_routing:
            # Get relevant clusters
            relevant_clusters = semantic_route(expanded_query)
            if relevant_clusters:
                # Search only in relevant clusters
                distances, indices = ivf_index.search(
                    np.array([query_embedding], dtype=np.float32),
                    k * 2,
                    nprobe=len(relevant_clusters)
                )
            else:
                distances, indices = ivf_index.search(
                    np.array([query_embedding], dtype=np.float32),
                    k * 2
                )
        else:
            distances, indices = hnsw_index.search(
                np.array([query_embedding], dtype=np.float32),
                k * 2
            )
        
        # Keyword matching
        query_terms = set(re.findall(r'\w+', expanded_query.lower()))
        
        # Process results
        for distance, idx in zip(distances[0], indices[0]):
            if str(idx) in chunk_store:
                chunk_data = chunk_store[str(idx)]
                
                # Apply metadata filtering
                if filter_metadata:
                    if not all(chunk_data.get('metadata', {}).get(k) == v 
                              for k, v in filter_metadata.items()):
                        continue
                
                # Apply faceted search
                if facets:
                    if not all(chunk_data.get('metadata', {}).get(facet) in values
                              for facet, values in facets.items()):
                        continue
                
                # Calculate scores
                chunk_terms = set(re.findall(r'\w+', chunk_data['text'].lower()))
                keyword_score = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0
                vector_score = 1 / (1 + distance)
                combined_score = 0.7 * vector_score + 0.3 * keyword_score
                
                all_results.append({
                    "chunk_id": int(idx),
                    "text": chunk_data["text"],
                    "filename": chunk_data["filename"],
                    "timestamp": chunk_data["timestamp"],
                    "metadata": chunk_data["metadata"],
                    "vector_score": float(vector_score),
                    "keyword_score": float(keyword_score),
                    "combined_score": float(combined_score),
                    "query": expanded_query
                })
    
    # Deduplicate and sort results
    seen_ids = set()
    unique_results = []
    for result in sorted(all_results, key=lambda x: x['combined_score'], reverse=True):
        if result['chunk_id'] not in seen_ids:
            seen_ids.add(result['chunk_id'])
            unique_results.append(result)
    
    return unique_results[:k]

def save_index():
    """Save FAISS index and chunk store to disk"""
    os.makedirs('vector_store', exist_ok=True)
    if torch.cuda.is_available():
        hnsw_index_cpu = faiss.index_gpu_to_cpu(hnsw_index)
        ivf_index_cpu = faiss.index_gpu_to_cpu(ivf_index)
        faiss.write_index(hnsw_index_cpu, 'vector_store/hnsw.index')
        faiss.write_index(ivf_index_cpu, 'vector_store/ivf.index')
    else:
        faiss.write_index(hnsw_index, 'vector_store/hnsw.index')
        faiss.write_index(ivf_index, 'vector_store/ivf.index')
    with open('vector_store/chunk_store.json', 'w') as f:
        json.dump(chunk_store, f)

def load_index():
    """Load FAISS index and chunk store from disk"""
    global hnsw_index, ivf_index, chunk_store, current_id
    try:
        if os.path.exists('vector_store/hnsw.index'):
            hnsw_index_cpu = faiss.read_index('vector_store/hnsw.index')
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                hnsw_index = faiss.index_cpu_to_gpu(res, 0, hnsw_index_cpu)
            else:
                hnsw_index = hnsw_index_cpu
        if os.path.exists('vector_store/ivf.index'):
            ivf_index_cpu = faiss.read_index('vector_store/ivf.index')
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                ivf_index = faiss.index_cpu_to_gpu(res, 0, ivf_index_cpu)
            else:
                ivf_index = ivf_index_cpu
        if os.path.exists('vector_store/chunk_store.json'):
            with open('vector_store/chunk_store.json', 'r') as f:
                chunk_store = json.load(f)
            current_id = max(map(int, chunk_store.keys())) + 1 if chunk_store else 0
    except Exception as e:
        print(f"Error loading index: {e}")

# Load existing index if available
load_index()

@app.post("/extract")
async def extract_document(file: UploadFile = File(...)):
    # Save uploaded file to a temp location
    try:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        result = extractor.extract_text(tmp_path)
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        return JSONResponse({
            "filename": file.filename,
            "extraction_method": result.extraction_method,
            "confidence": result.confidence,
            "text_preview": result.text[:500],
            "text_length": len(result.text),
            "metadata": result.metadata,
            "page_count": getattr(result, 'page_count', None),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    options: ProcessOptions = ProcessOptions()
):
    global current_id
    try:
        # 1. Save and extract text
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # 2. Extract text
        extraction_result = extractor.extract_text(tmp_path)
        
        # 3. Chunk text with advanced options
        if options.use_advanced_chunking:
            # Use semantic-aware chunking
            chunks = chunker.chunk_text(
                extraction_result.text,
                chunk_size=options.chunk_size,
                overlap=options.overlap
            )
        else:
            chunks = chunker.chunk_text(
                extraction_result.text,
                chunk_size=options.chunk_size,
                overlap=options.overlap
            )
        
        # 4. Generate embeddings and store in FAISS
        chunk_ids = []
        embeddings = []
        
        # Batch process embeddings
        for chunk in chunks:
            embedding = embedder.embed_text(chunk)
            embeddings.append(embedding)
        
        # Add batch to both indices
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            hnsw_index.add(embeddings_array)
            ivf_index.add(embeddings_array)
            
            # Store chunks and metadata
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_store[str(current_id)] = {
                    "text": chunk,
                    "filename": file.filename,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": extraction_result.metadata,
                    "embedding": embedding.tolist()
                }
                chunk_ids.append(current_id)
                current_id += 1
        
        # 5. Perform semantic clustering if requested
        cluster_info = None
        if options.use_semantic_clustering and len(embeddings) > 0:
            kmeans = KMeans(n_clusters=min(options.n_clusters, len(embeddings)))
            clusters = kmeans.fit_predict(embeddings_array)
            
            # Store cluster assignments
            for chunk_id, cluster_id in zip(chunk_ids, clusters):
                if str(chunk_id) in chunk_store:
                    chunk_store[str(chunk_id)]["cluster_id"] = int(cluster_id)
            
            cluster_info = {
                "n_clusters": len(set(clusters)),
                "cluster_sizes": [int(np.sum(clusters == i)) for i in range(len(set(clusters)))]
            }
        
        # Save updated indices and store
        save_index()
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        
        return JSONResponse({
            "filename": file.filename,
            "extraction_method": extraction_result.extraction_method,
            "confidence": extraction_result.confidence,
            "text_length": len(extraction_result.text),
            "chunk_count": len(chunks),
            "chunk_ids": chunk_ids,
            "metadata": extraction_result.metadata,
            "clustering": cluster_info
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/search")
async def search_similar(query: SearchQuery):
    try:
        # Get expanded queries if enabled
        if query.use_query_expansion:
            queries = advanced_search.expand_query(query.query)
        else:
            queries = [query.query]
        
        all_results = []
        
        for expanded_query in queries:
            # Generate embedding for query
            query_embedding = embedder.embed_text(expanded_query)
            
            # Select index based on type
            if query.index_type == "ivf" and query.use_semantic_routing:
                # Get relevant clusters
                centroids = []
                for i in range(nlist):
                    if ivf_index.ntotal > 0:
                        cluster_vectors = ivf_index.get_cluster_vectors(i)
                        if len(cluster_vectors) > 0:
                            centroids.append(np.mean(cluster_vectors, axis=0))
                
                relevant_clusters = advanced_search.semantic_route(expanded_query, centroids)
                if relevant_clusters:
                    # Search only in relevant clusters
                    distances, indices = ivf_index.search(
                        np.array([query_embedding], dtype=np.float32),
                        query.k * 2,
                        nprobe=len(relevant_clusters)
                    )
                else:
                    distances, indices = ivf_index.search(
                        np.array([query_embedding], dtype=np.float32),
                        query.k * 2
                    )
            else:
                distances, indices = hnsw_index.search(
                    np.array([query_embedding], dtype=np.float32),
                    query.k * 2
                )
            
            # Process results
            for distance, idx in zip(distances[0], indices[0]):
                if str(idx) in chunk_store:
                    chunk_data = chunk_store[str(idx)]
                    
                    # Apply metadata filtering
                    if query.filter_metadata:
                        if not all(chunk_data.get('metadata', {}).get(k) == v 
                                  for k, v in query.filter_metadata.items()):
                            continue
                    
                    # Apply faceted search
                    if query.facets:
                        if not all(chunk_data.get('metadata', {}).get(facet) in values
                                  for facet, values in query.facets.items()):
                            continue
                    
                    # Calculate scores
                    scores = advanced_search.calculate_scores(
                        expanded_query,
                        chunk_data["text"],
                        distance
                    )
                    
                    all_results.append({
                        "chunk_id": int(idx),
                        "text": chunk_data["text"],
                        "filename": chunk_data["filename"],
                        "timestamp": chunk_data["timestamp"],
                        "metadata": chunk_data["metadata"],
                        "cluster_id": chunk_data.get("cluster_id"),
                        **scores,
                        "query": expanded_query
                    })
        
        # Deduplicate and sort results
        seen_ids = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x['combined_score'], reverse=True):
            if result['chunk_id'] not in seen_ids:
                seen_ids.add(result['chunk_id'])
                unique_results.append(result)
        
        # Filter by minimum score
        results = [r for r in unique_results if r['combined_score'] >= query.min_score]
        
        return JSONResponse({
            "query": query.query,
            "expanded_queries": queries if query.use_query_expansion else [query.query],
            "results": results[:query.k]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/cluster")
async def cluster_documents(
    n_clusters: int = 5,
    method: str = "kmeans",  # "kmeans" or "hierarchical"
    min_cluster_size: int = 2
):
    """Cluster documents using K-means or hierarchical clustering"""
    try:
        # Get all embeddings
        embeddings = []
        chunk_ids = []
        for chunk_id, data in chunk_store.items():
            if 'embedding' in data:
                embeddings.append(data['embedding'])
                chunk_ids.append(int(chunk_id))
        
        if not embeddings:
            return JSONResponse({"message": "No documents to cluster"})
        
        embeddings_array = np.array(embeddings)
        
        if method == "hierarchical":
            # Hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            clusters = clustering.fit_predict(embeddings_array)
        else:
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(embeddings_array)
        
        # Organize results by cluster
        cluster_results = {i: [] for i in range(n_clusters)}
        for chunk_id, cluster_id in zip(chunk_ids, clusters):
            if str(chunk_id) in chunk_store:
                data = chunk_store[str(chunk_id)]
                cluster_results[cluster_id].append({
                    "chunk_id": chunk_id,
                    "text": data["text"],
                    "filename": data["filename"],
                    "metadata": data["metadata"]
                })
        
        # Filter out small clusters
        cluster_results = {
            k: v for k, v in cluster_results.items()
            if len(v) >= min_cluster_size
        }
        
        return JSONResponse({
            "n_clusters": len(cluster_results),
            "clustering_method": method,
            "clusters": cluster_results
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

@app.get("/facets")
async def get_facets():
    """Get available facets and their values from the document store"""
    try:
        facets = defaultdict(set)
        for data in chunk_store.values():
            for key, value in data.get('metadata', {}).items():
                facets[key].add(str(value))
        
        return JSONResponse({
            "facets": {k: list(v) for k, v in facets.items()}
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get facets: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "extractor": "initialized",
            "chunker": "initialized",
            "embedder": "initialized",
            "advanced_search": "initialized",
            "faiss": {
                "hnsw_index_size": hnsw_index.ntotal,
                "ivf_index_size": ivf_index.ntotal,
                "dimension": DIMENSION,
                "gpu_enabled": torch.cuda.is_available()
            }
        }
    } 