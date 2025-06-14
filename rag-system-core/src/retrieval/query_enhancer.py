"""
Query Enhancer
Advanced query processing, expansion, and reformulation for better retrieval
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Query enhancement will use fallback.")

try:
    from ..core.error_handling import QueryError
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.error_handling import QueryError

class QueryType(Enum):
    """Types of queries for different processing strategies"""
    FACTUAL = "factual"           # What is X? Define Y
    PROCEDURAL = "procedural"     # How to do X? Steps for Y
    COMPARATIVE = "comparative"   # Compare X and Y, Difference between
    CAUSAL = "causal"            # Why does X? What causes Y?
    TEMPORAL = "temporal"        # When did X? Timeline of Y
    LOCATION = "location"        # Where is X? Location of Y
    GENERAL = "general"          # General questions

@dataclass
class QueryIntent:
    """Represents the detected intent of a query"""
    query_type: QueryType
    confidence: float
    keywords: List[str]
    entities: List[str]
    context_hints: List[str]

@dataclass
class EnhancedQuery:
    """Represents an enhanced query with multiple variants"""
    original_query: str
    intent: QueryIntent
    expanded_queries: List[str]
    reformulated_queries: List[str]
    keywords: List[str]
    semantic_variants: List[str]
    confidence_scores: Dict[str, float]

class QueryEnhancer:
    """Advanced query enhancement and expansion system"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 max_expansions: int = 3,
                 max_reformulations: int = 2):
        self.model_name = model_name
        self.max_expansions = max_expansions
        self.max_reformulations = max_reformulations
        self.model = None
        self.enabled = SENTENCE_TRANSFORMERS_AVAILABLE
        
        # Query patterns for intent detection
        self.query_patterns = {
            QueryType.FACTUAL: [
                r'\b(what is|define|definition of|meaning of|explain)\b',
                r'\b(describe|tell me about)\b'
            ],
            QueryType.PROCEDURAL: [
                r'\b(how to|how do|how can|steps to|process of)\b',
                r'\b(guide|tutorial|instructions)\b'
            ],
            QueryType.COMPARATIVE: [
                r'\b(compare|comparison|difference|versus|vs|better than)\b',
                r'\b(similarities|differences between)\b'
            ],
            QueryType.CAUSAL: [
                r'\b(why|because|cause|reason|due to)\b',
                r'\b(what causes|what leads to)\b'
            ],
            QueryType.TEMPORAL: [
                r'\b(when|timeline|history|chronology)\b',
                r'\b(before|after|during|since)\b'
            ],
            QueryType.LOCATION: [
                r'\b(where|location|place|geography)\b',
                r'\b(in which|at what)\b'
            ]
        }
        
        # Expansion templates
        self.expansion_templates = {
            QueryType.FACTUAL: [
                "What is {topic}?",
                "Define {topic}",
                "Explain {topic} in detail"
            ],
            QueryType.PROCEDURAL: [
                "How to {action}?",
                "Steps for {action}",
                "Guide to {action}"
            ],
            QueryType.COMPARATIVE: [
                "Compare {topic1} and {topic2}",
                "Difference between {topic1} and {topic2}",
                "{topic1} versus {topic2}"
            ]
        }
        
        # Common synonyms and related terms
        self.synonym_map = {
            'machine learning': ['ML', 'artificial intelligence', 'AI', 'automated learning'],
            'deep learning': ['neural networks', 'DL', 'deep neural networks'],
            'artificial intelligence': ['AI', 'machine intelligence', 'cognitive computing'],
            'natural language processing': ['NLP', 'text processing', 'language understanding'],
            'computer vision': ['image recognition', 'visual computing', 'image processing'],
            'algorithm': ['method', 'procedure', 'technique', 'approach'],
            'data': ['information', 'dataset', 'records', 'facts'],
            'model': ['system', 'framework', 'architecture', 'structure'],
            'training': ['learning', 'education', 'instruction', 'preparation'],
            'prediction': ['forecast', 'estimation', 'projection', 'inference']
        }
        
        if self.enabled:
            self._initialize_model()
        else:
            logging.warning("Query enhancement disabled - using fallback processing")
    
    def _initialize_model(self):
        """Initialize the sentence transformer model for semantic processing"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logging.info(f"Query enhancer initialized with model: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize query enhancer: {e}")
            self.enabled = False
    
    def enhance_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> EnhancedQuery:
        """
        Enhance a query with expansion, reformulation, and intent detection
        
        Args:
            query: Original user query
            context: Optional context information
            
        Returns:
            EnhancedQuery object with all enhancements
        """
        if not query.strip():
            raise QueryError("Empty query provided")
        
        try:
            # Clean and normalize query
            cleaned_query = self._clean_query(query)
            
            # Detect query intent
            intent = self._detect_intent(cleaned_query)
            
            # Extract keywords
            keywords = self._extract_keywords(cleaned_query)
            
            # Generate query expansions
            expanded_queries = self._expand_query(cleaned_query, intent, keywords)
            
            # Generate reformulations
            reformulated_queries = self._reformulate_query(cleaned_query, intent)
            
            # Generate semantic variants if model available
            semantic_variants = []
            if self.enabled:
                semantic_variants = self._generate_semantic_variants(cleaned_query)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                cleaned_query, expanded_queries, reformulated_queries, semantic_variants
            )
            
            enhanced_query = EnhancedQuery(
                original_query=query,
                intent=intent,
                expanded_queries=expanded_queries,
                reformulated_queries=reformulated_queries,
                keywords=keywords,
                semantic_variants=semantic_variants,
                confidence_scores=confidence_scores
            )
            
            logging.info(f"Query enhanced: {len(expanded_queries)} expansions, {len(reformulated_queries)} reformulations")
            return enhanced_query
            
        except Exception as e:
            logging.error(f"Query enhancement failed: {e}")
            # Return basic enhancement as fallback
            return self._create_fallback_enhancement(query)
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect the intent and type of the query"""
        query_lower = query.lower()
        
        # Check patterns for each query type
        detected_types = []
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_types.append(query_type)
                    break
        
        # Determine primary type
        if detected_types:
            primary_type = detected_types[0]  # Take first match
            confidence = 0.8 if len(detected_types) == 1 else 0.6
        else:
            primary_type = QueryType.GENERAL
            confidence = 0.5
        
        # Extract entities and keywords
        keywords = self._extract_keywords(query)
        entities = self._extract_entities(query)
        context_hints = self._extract_context_hints(query)
        
        return QueryIntent(
            query_type=primary_type,
            confidence=confidence,
            keywords=keywords,
            entities=entities,
            context_hints=context_hints
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query"""
        # Remove stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Extract phrases (2-3 word combinations)
        phrases = []
        for i in range(len(words) - 1):
            if words[i] not in stop_words and words[i+1] not in stop_words:
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 5:  # Avoid very short phrases
                    phrases.append(phrase)
        
        return keywords + phrases
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from the query"""
        # Simple entity extraction - could be enhanced with NER models
        entities = []
        
        # Look for capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(capitalized_words)
        
        # Look for technical terms
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms like AI, ML, NLP
            r'\b\w+\-\w+\b',   # Hyphenated terms
            r'\b\w+\.\w+\b'    # Dotted terms
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_context_hints(self, query: str) -> List[str]:
        """Extract context hints that might help with retrieval"""
        hints = []
        
        # Domain indicators
        domain_keywords = {
            'machine learning': ['algorithm', 'model', 'training', 'prediction'],
            'programming': ['code', 'function', 'variable', 'syntax'],
            'data science': ['data', 'analysis', 'statistics', 'visualization'],
            'artificial intelligence': ['AI', 'neural', 'cognitive', 'intelligent']
        }
        
        query_lower = query.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                hints.append(domain)
        
        return hints
    
    def _expand_query(self, query: str, intent: QueryIntent, keywords: List[str]) -> List[str]:
        """Generate expanded versions of the query"""
        expansions = []
        
        # Synonym-based expansion
        for keyword in keywords[:3]:  # Limit to top 3 keywords
            if keyword in self.synonym_map:
                for synonym in self.synonym_map[keyword][:2]:  # Max 2 synonyms per keyword
                    expanded = query.replace(keyword, synonym)
                    if expanded != query and expanded not in expansions:
                        expansions.append(expanded)
        
        # Template-based expansion
        if intent.query_type in self.expansion_templates:
            templates = self.expansion_templates[intent.query_type]
            for template in templates[:2]:  # Max 2 templates
                if '{topic}' in template and keywords:
                    expanded = template.format(topic=keywords[0])
                    if expanded not in expansions:
                        expansions.append(expanded)
        
        # Keyword combination expansion
        if len(keywords) >= 2:
            combined = f"{keywords[0]} {keywords[1]}"
            if combined not in query:
                expansions.append(f"What is {combined}?")
        
        return expansions[:self.max_expansions]
    
    def _reformulate_query(self, query: str, intent: QueryIntent) -> List[str]:
        """Generate reformulated versions of the query"""
        reformulations = []
        
        # Question type reformulations
        if intent.query_type == QueryType.FACTUAL:
            if not query.lower().startswith(('what', 'define')):
                reformulations.append(f"What is {query.rstrip('?')}?")
                reformulations.append(f"Define {query.rstrip('?')}")
        
        elif intent.query_type == QueryType.PROCEDURAL:
            if not query.lower().startswith('how'):
                reformulations.append(f"How to {query.rstrip('?')}?")
                reformulations.append(f"Steps for {query.rstrip('?')}")
        
        elif intent.query_type == QueryType.COMPARATIVE:
            if 'compare' not in query.lower():
                keywords = intent.keywords[:2]
                if len(keywords) >= 2:
                    reformulations.append(f"Compare {keywords[0]} and {keywords[1]}")
        
        # Generic reformulations
        if query.endswith('?'):
            # Convert question to statement
            statement = query.rstrip('?')
            reformulations.append(f"Information about {statement}")
        else:
            # Convert statement to question
            reformulations.append(f"What is {query}?")
        
        return reformulations[:self.max_reformulations]
    
    def _generate_semantic_variants(self, query: str) -> List[str]:
        """Generate semantic variants using embeddings"""
        if not self.enabled or not self.model:
            return []
        
        variants = []
        
        try:
            # Simple semantic variants - could be enhanced with paraphrasing models
            # For now, generate variations based on sentence structure
            
            # Convert questions to statements and vice versa
            if query.endswith('?'):
                statement = query.rstrip('?')
                variants.append(f"Information about {statement}")
                variants.append(f"Details on {statement}")
            else:
                variants.append(f"What is {query}?")
                variants.append(f"Tell me about {query}")
            
            # Add context-aware variants
            variants.append(f"Explain {query}")
            variants.append(f"Overview of {query}")
            
        except Exception as e:
            logging.warning(f"Failed to generate semantic variants: {e}")
        
        return variants[:3]  # Limit to 3 variants
    
    def _calculate_confidence_scores(self, original: str, expansions: List[str], 
                                   reformulations: List[str], variants: List[str]) -> Dict[str, float]:
        """Calculate confidence scores for different query enhancements"""
        scores = {
            'original': 1.0,  # Original query always has highest confidence
        }
        
        # Score expansions based on keyword overlap
        for i, expansion in enumerate(expansions):
            overlap = self._calculate_overlap(original, expansion)
            scores[f'expansion_{i}'] = 0.8 + (overlap * 0.2)
        
        # Score reformulations
        for i, reformulation in enumerate(reformulations):
            scores[f'reformulation_{i}'] = 0.7
        
        # Score semantic variants
        for i, variant in enumerate(variants):
            scores[f'variant_{i}'] = 0.6
        
        return scores
    
    def _calculate_overlap(self, query1: str, query2: str) -> float:
        """Calculate word overlap between two queries"""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query"""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Fix common typos and normalize
        query = query.replace('  ', ' ')
        
        return query
    
    def _create_fallback_enhancement(self, query: str) -> EnhancedQuery:
        """Create a basic enhancement when full processing fails"""
        intent = QueryIntent(
            query_type=QueryType.GENERAL,
            confidence=0.5,
            keywords=self._extract_keywords(query),
            entities=[],
            context_hints=[]
        )
        
        return EnhancedQuery(
            original_query=query,
            intent=intent,
            expanded_queries=[],
            reformulated_queries=[],
            keywords=intent.keywords,
            semantic_variants=[],
            confidence_scores={'original': 1.0}
        )
    
    def get_all_query_variants(self, enhanced_query: EnhancedQuery) -> List[Tuple[str, float]]:
        """Get all query variants with their confidence scores"""
        variants = [(enhanced_query.original_query, 1.0)]
        
        # Add expansions
        for i, expansion in enumerate(enhanced_query.expanded_queries):
            score = enhanced_query.confidence_scores.get(f'expansion_{i}', 0.8)
            variants.append((expansion, score))
        
        # Add reformulations
        for i, reformulation in enumerate(enhanced_query.reformulated_queries):
            score = enhanced_query.confidence_scores.get(f'reformulation_{i}', 0.7)
            variants.append((reformulation, score))
        
        # Add semantic variants
        for i, variant in enumerate(enhanced_query.semantic_variants):
            score = enhanced_query.confidence_scores.get(f'variant_{i}', 0.6)
            variants.append((variant, score))
        
        # Sort by confidence score
        variants.sort(key=lambda x: x[1], reverse=True)
        
        return variants
    
    def get_enhancer_info(self) -> Dict[str, Any]:
        """Get information about the query enhancer configuration"""
        return {
            'enhancer_type': 'advanced',
            'model_name': self.model_name,
            'max_expansions': self.max_expansions,
            'max_reformulations': self.max_reformulations,
            'enabled': self.enabled,
            'model_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'supported_query_types': [qt.value for qt in QueryType],
            'synonym_domains': list(self.synonym_map.keys())
        }

def create_query_enhancer(config_manager) -> QueryEnhancer:
    """Factory function to create query enhancer based on configuration"""
    # Could read from config in the future
    return QueryEnhancer(
        model_name="all-MiniLM-L6-v2",
        max_expansions=3,
        max_reformulations=2
    ) 