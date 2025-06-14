"""
System Constants
Defines embedding dimensions and other system-wide constants
"""

# Embedding dimensions by provider and model
EMBEDDING_DIMENSIONS = {
    'sentence-transformers': {
        'all-MiniLM-L6-v2': 384,
        'all-mpnet-base-v2': 768,
        'all-distilroberta-v1': 768,
    },
    'cohere': {
        'embed-english-v3.0': 1024,
        'embed-multilingual-v3.0': 1024,
        'embed-english-light-v3.0': 384,
    },
    'openai': {
        'text-embedding-ada-002': 1536,
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
    }
}

# Default embedding configurations
DEFAULT_EMBEDDING_CONFIG = {
    'sentence-transformers': {
        'model': 'sentence-transformers/all-MiniLM-L6-v2',
        'dimension': 384
    },
    'cohere': {
        'model': 'embed-english-v3.0',
        'dimension': 1024
    },
    'openai': {
        'model': 'text-embedding-ada-002',
        'dimension': 1536
    }
}

# System limits
MAX_FILE_SIZE_MB = 100
MAX_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 100
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# API limits
MAX_QUERY_LENGTH = 1000
MAX_RESULTS_PER_QUERY = 50
DEFAULT_TOP_K = 5

# Thread pool settings
DEFAULT_THREAD_POOL_SIZE = 4
MAX_THREAD_POOL_SIZE = 16

def get_embedding_dimension(provider: str, model_name: str) -> int:
    """Get embedding dimension for a specific provider and model"""
    if provider in EMBEDDING_DIMENSIONS:
        # Extract model name from full path if needed
        model_key = model_name.split('/')[-1] if '/' in model_name else model_name
        
        if model_key in EMBEDDING_DIMENSIONS[provider]:
            return EMBEDDING_DIMENSIONS[provider][model_key]
        
        # Return default for provider if specific model not found
        if provider in DEFAULT_EMBEDDING_CONFIG:
            return DEFAULT_EMBEDDING_CONFIG[provider]['dimension']
    
    # Fallback to sentence-transformers default
    return 384

def get_default_model_for_provider(provider: str) -> tuple:
    """Get default model name and dimension for a provider"""
    if provider in DEFAULT_EMBEDDING_CONFIG:
        config = DEFAULT_EMBEDDING_CONFIG[provider]
        return config['model'], config['dimension']
    
    # Fallback
    return 'sentence-transformers/all-MiniLM-L6-v2', 384 