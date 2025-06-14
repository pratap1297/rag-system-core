"""
Text Embedder
Generate embeddings using multiple providers (sentence-transformers, Cohere)
"""
import logging
import numpy as np
import os
from typing import List, Union, Optional
from abc import ABC, abstractmethod

try:
    from ..core.error_handling import EmbeddingError
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
    from error_handling import EmbeddingError

class BaseEmbedder(ABC):
    """Base class for embedding providers"""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        pass

class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence Transformers embedder"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = "cpu", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logging.info(f"Loaded SentenceTransformer model: {self.model_name}")
        except ImportError:
            raise EmbeddingError("sentence-transformers package not installed")
        except Exception as e:
            raise EmbeddingError(f"Failed to load SentenceTransformer model: {e}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        try:
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                all_embeddings.extend(batch_embeddings.tolist())
            return all_embeddings
        except Exception as e:
            raise EmbeddingError(f"Failed to generate SentenceTransformer embeddings: {e}")
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

class CohereEmbedder(BaseEmbedder):
    """Cohere embedder"""
    
    def __init__(self, model_name: str = "embed-english-v3.0", 
                 api_key: Optional[str] = None, batch_size: int = 96):
        self.model_name = model_name
        self.api_key = api_key or os.getenv('COHERE_API_KEY')
        self.batch_size = batch_size
        self.client = None
        self._dimension = None
        self._load_client()
    
    def _load_client(self):
        """Load the Cohere client"""
        if not self.api_key:
            raise EmbeddingError("Cohere API key not provided. Set COHERE_API_KEY environment variable.")
        
        try:
            import cohere
            self.client = cohere.Client(self.api_key)
            logging.info(f"Loaded Cohere client with model: {self.model_name}")
            
            # Get dimension by testing with a sample text
            test_response = self.client.embed(
                texts=["test"],
                model=self.model_name,
                input_type="search_document"
            )
            self._dimension = len(test_response.embeddings[0])
            logging.info(f"Cohere embedding dimension: {self._dimension}")
            
        except ImportError:
            raise EmbeddingError("cohere package not installed")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize Cohere client: {e}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        try:
            import time
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                start_time = time.time()
                response = self.client.embed(
                    texts=batch,
                    model=self.model_name,
                    input_type="search_document"
                )
                elapsed = time.time() - start_time
                logging.debug(f"Cohere embedding batch ({len(batch)} texts) took {elapsed:.2f} seconds")
                all_embeddings.extend(response.embeddings)
            return all_embeddings
        except Exception as e:
            logging.error(f"Cohere embedding error: {e}")
            raise EmbeddingError(f"Failed to generate Cohere embeddings: {e}")
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension

class Embedder:
    """Multi-provider text embedder"""
    
    def __init__(self, provider: str = "cohere", model_name: Optional[str] = None, 
                 api_key: Optional[str] = None, device: str = "cpu", batch_size: int = 32):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.device = device
        self.batch_size = batch_size
        self.embedder = None
        
        self._initialize_embedder()
        logging.info(f"Embedder initialized with provider: {provider}")
    
    def _initialize_embedder(self):
        """Initialize the appropriate embedder"""
        if self.provider == "cohere":
            model = self.model_name or "embed-english-v3.0"
            self.embedder = CohereEmbedder(
                model_name=model,
                api_key=self.api_key,
                batch_size=self.batch_size
            )
        elif self.provider == "sentence-transformers":
            model = self.model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self.embedder = SentenceTransformerEmbedder(
                model_name=model,
                device=self.device,
                batch_size=self.batch_size
            )
        else:
            raise EmbeddingError(f"Unsupported embedding provider: {self.provider}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return self.embedder.embed_texts(texts)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedder.get_dimension()
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        embeddings = self.embed_texts([text1, text2])
        
        # Cosine similarity
        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2)) 