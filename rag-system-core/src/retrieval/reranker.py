"""
Reranker Module
Cross-encoder based reranking for improved retrieval relevance
"""
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("sentence-transformers not available. Reranking will be disabled.")

try:
    from ..core.error_handling import RetrievalError
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.error_handling import RetrievalError

class Reranker:
    """Cross-encoder based reranker for improving retrieval relevance"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2", 
                 device: str = "cpu", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.enabled = CROSS_ENCODER_AVAILABLE
        
        if self.enabled:
            self._initialize_model()
        else:
            logging.warning("Reranker disabled due to missing dependencies")
    
    def _initialize_model(self):
        """Initialize the cross-encoder model"""
        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
            logging.info(f"Reranker initialized with model: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize reranker: {e}")
            self.enabled = False
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: The search query
            documents: List of document dictionaries with 'text' or 'content' field
            top_k: Number of top documents to return (None = return all)
            
        Returns:
            List of reranked documents with updated scores
        """
        if not self.enabled or not documents:
            return documents[:top_k] if top_k else documents
        
        try:
            # Extract text content from documents
            doc_texts = []
            for doc in documents:
                text = doc.get('text', doc.get('content', ''))
                if not text:
                    # Fallback to any text field
                    text = str(doc.get('metadata', {}).get('content', ''))
                doc_texts.append(text)
            
            # Create query-document pairs
            pairs = [(query, text) for text in doc_texts]
            
            # Get relevance scores from cross-encoder
            scores = self._predict_scores(pairs)
            
            # Add rerank scores to documents and sort
            reranked_docs = []
            for doc, score in zip(documents, scores):
                doc_copy = doc.copy()
                doc_copy['rerank_score'] = float(score)
                doc_copy['original_score'] = doc.get('similarity_score', doc.get('score', 0))
                reranked_docs.append(doc_copy)
            
            # Sort by rerank score (descending)
            reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Return top_k results
            result = reranked_docs[:top_k] if top_k else reranked_docs
            
            logging.info(f"Reranked {len(documents)} documents, returning top {len(result)}")
            return result
            
        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            # Fallback to original ranking
            return documents[:top_k] if top_k else documents
    
    def _predict_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Predict relevance scores for query-document pairs"""
        if not self.model:
            return [0.0] * len(pairs)
        
        try:
            # Process in batches to avoid memory issues
            all_scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i + self.batch_size]
                batch_scores = self.model.predict(batch_pairs)
                all_scores.extend(batch_scores.tolist())
            
            return all_scores
            
        except Exception as e:
            logging.error(f"Score prediction failed: {e}")
            return [0.0] * len(pairs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'batch_size': self.batch_size,
            'enabled': self.enabled,
            'available': CROSS_ENCODER_AVAILABLE
        }
    
    def is_enabled(self) -> bool:
        """Check if reranking is enabled and available"""
        return self.enabled and self.model is not None

class FallbackReranker:
    """Fallback reranker that uses similarity scores when cross-encoder is unavailable"""
    
    def __init__(self):
        self.enabled = True
        logging.info("Using fallback reranker (similarity-based)")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fallback reranking using existing similarity scores"""
        if not documents:
            return documents
        
        # Sort by existing similarity score
        sorted_docs = sorted(
            documents, 
            key=lambda x: x.get('similarity_score', x.get('score', 0)), 
            reverse=True
        )
        
        # Add rerank_score field for consistency
        for doc in sorted_docs:
            doc['rerank_score'] = doc.get('similarity_score', doc.get('score', 0))
            doc['original_score'] = doc['rerank_score']
        
        result = sorted_docs[:top_k] if top_k else sorted_docs
        logging.info(f"Fallback reranked {len(documents)} documents, returning top {len(result)}")
        return result
    
    def is_enabled(self) -> bool:
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_name': 'fallback_similarity',
            'device': 'cpu',
            'enabled': True,
            'available': True
        }

def create_reranker(config_manager) -> Reranker:
    """Factory function to create reranker based on configuration"""
    config = config_manager.get_config()
    
    if not config.retrieval.enable_reranking:
        logging.info("Reranking disabled in configuration")
        return FallbackReranker()
    
    try:
        return Reranker(
            model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
            device="cpu",  # Can be configured later
            batch_size=32
        )
    except Exception as e:
        logging.warning(f"Failed to create reranker, using fallback: {e}")
        return FallbackReranker() 