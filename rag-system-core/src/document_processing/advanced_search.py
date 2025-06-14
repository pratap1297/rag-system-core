import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

class AdvancedSearch:
    def __init__(self, embedder):
        self.embedder = embedder
        self.lemmatizer = WordNetLemmatizer()

    def expand_query(self, query: str) -> List[str]:
        """Expand query using WordNet synonyms and lemmatization"""
        expanded_queries = [query]
        
        # Tokenize and lemmatize
        tokens = word_tokenize(query.lower())
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
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

    def semantic_route(self, query: str, centroids: List[np.ndarray], k: int = 5) -> List[int]:
        """Route query to most relevant clusters using semantic similarity"""
        query_embedding = self.embedder.encode(query)
        
        if not centroids:
            return []
        
        # Calculate similarity with centroids
        similarities = cosine_similarity([query_embedding], centroids)[0]
        top_clusters = np.argsort(similarities)[-k:]
        
        return top_clusters.tolist()

    def calculate_scores(self, query: str, chunk_text: str, distance: float) -> Dict[str, float]:
        """Calculate various relevance scores for a chunk"""
        # Vector similarity score
        vector_score = 1 / (1 + distance)
        
        # Keyword matching score
        query_terms = set(re.findall(r'\w+', query.lower()))
        chunk_terms = set(re.findall(r'\w+', chunk_text.lower()))
        keyword_score = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0
        
        # Combined score (70% vector, 30% keyword)
        combined_score = 0.7 * vector_score + 0.3 * keyword_score
        
        return {
            "vector_score": float(vector_score),
            "keyword_score": float(keyword_score),
            "combined_score": float(combined_score)
        } 