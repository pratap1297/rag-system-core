"""
Query Engine
Main engine for processing user queries and generating responses
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.error_handling import RetrievalError

class QueryEngine:
    """Main query processing engine"""
    
    def __init__(self, faiss_store, embedder, llm_client, metadata_store, config_manager, reranker=None, query_enhancer=None):
        self.faiss_store = faiss_store
        self.embedder = embedder
        self.llm_client = llm_client
        self.metadata_store = metadata_store
        self.config = config_manager.get_config()
        self.reranker = reranker
        self.query_enhancer = query_enhancer
        
        logging.info(f"Query engine initialized with reranker: {reranker is not None}, query enhancer: {query_enhancer is not None}")
    
    def process_query(self, query: str, filters: Dict[str, Any] = None, 
                     top_k: int = None) -> Dict[str, Any]:
        """Process a user query and return response with sources"""
        top_k = top_k or self.config.retrieval.top_k
        
        try:
            # Enhance query if enhancer is available
            enhanced_query = None
            query_variants = [(query, 1.0)]  # Default: original query with max confidence
            
            if self.query_enhancer:
                try:
                    enhanced_query = self.query_enhancer.enhance_query(query)
                    query_variants = self.query_enhancer.get_all_query_variants(enhanced_query)
                    logging.info(f"Query enhanced: {len(query_variants)} variants generated")
                except Exception as e:
                    logging.warning(f"Query enhancement failed, using original query: {e}")
            
            # Search with multiple query variants
            all_results = []
            for query_text, confidence in query_variants[:3]:  # Use top 3 variants
                # Generate query embedding
                query_embedding = self.embedder.embed_text(query_text)
                
                # Search for similar chunks
                search_results = self.faiss_store.search_with_metadata(
                    query_vector=query_embedding,
                    k=top_k
                )
                
                # Add confidence weighting to results
                for result in search_results:
                    result['query_confidence'] = confidence
                    result['query_variant'] = query_text
                    # Adjust similarity score by query confidence
                    original_score = result.get('similarity_score', 0)
                    result['weighted_score'] = original_score * confidence
                
                all_results.extend(search_results)
            
            # Deduplicate and merge results
            search_results = self._merge_search_results(all_results)
            
            if not search_results:
                return self._create_empty_response(query)
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in search_results
                if result.get('similarity_score', 0) >= self.config.retrieval.similarity_threshold
            ]
            
            if not filtered_results:
                return self._create_empty_response(query)
            
            # Apply reranking if enabled and available
            if self.reranker and self.config.retrieval.enable_reranking:
                logging.info(f"Applying reranking to {len(filtered_results)} results")
                reranked_results = self.reranker.rerank(
                    query=query, 
                    documents=filtered_results, 
                    top_k=self.config.retrieval.rerank_top_k
                )
                top_results = reranked_results
            else:
                # Take top k results without reranking
                top_results = filtered_results[:top_k]
            
            # Generate response using LLM
            response = self._generate_llm_response(query, top_results)
            
            # Prepare response with enhancement info
            response_data = {
                'query': query,
                'response': response,
                'sources': self._format_sources(top_results),
                'total_sources': len(top_results),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add query enhancement information if available
            if enhanced_query:
                response_data['query_enhancement'] = {
                    'intent_type': enhanced_query.intent.query_type.value,
                    'intent_confidence': enhanced_query.intent.confidence,
                    'keywords': enhanced_query.keywords,
                    'expanded_queries': enhanced_query.expanded_queries,
                    'reformulated_queries': enhanced_query.reformulated_queries,
                    'total_variants': len(query_variants)
                }
            
            return response_data
            
        except Exception as e:
            raise RetrievalError(f"Query processing failed: {e}", query=query)
    
    def _generate_llm_response(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Generate response using LLM with retrieved sources"""
        # Build context from sources
        context_parts = []
        for i, source in enumerate(sources[:5]):  # Use top 5 sources
            text = source.get('text', '')
            context_parts.append(f"Source {i+1}: {text}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following context, answer the user's question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            return self.llm_client.generate(prompt)
        except Exception as e:
            logging.error(f"LLM generation failed: {e}")
            return "I apologize, but I'm unable to generate a response at the moment due to a technical issue."
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for response"""
        formatted_sources = []
        
        for source in sources:
            formatted_source = {
                'text': source.get('text', '')[:200] + "..." if len(source.get('text', '')) > 200 else source.get('text', ''),
                'similarity_score': source.get('similarity_score', 0),
                'rerank_score': source.get('rerank_score'),
                'original_score': source.get('original_score'),
                'metadata': source.get('metadata', {}),
                'source_type': source.get('metadata', {}).get('source_type', 'unknown'),
                'doc_id': source.get('doc_id', 'unknown'),
                'chunk_id': source.get('chunk_id', 'unknown')
            }
            formatted_sources.append(formatted_source)
        
        return formatted_sources
    
    def _create_empty_response(self, query: str) -> Dict[str, Any]:
        """Create response when no relevant sources found"""
        return {
            'query': query,
            'response': "I couldn't find any relevant information to answer your question. Please try rephrasing your query or check if the information exists in the knowledge base.",
            'sources': [],
            'total_sources': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _merge_search_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge and deduplicate search results from multiple query variants"""
        if not all_results:
            return []
        
        # Group results by chunk_id to deduplicate
        result_groups = {}
        for result in all_results:
            chunk_id = result.get('chunk_id', 'unknown')
            
            if chunk_id not in result_groups:
                result_groups[chunk_id] = result
            else:
                # Keep result with higher weighted score
                existing = result_groups[chunk_id]
                if result.get('weighted_score', 0) > existing.get('weighted_score', 0):
                    result_groups[chunk_id] = result
        
        # Convert back to list and sort by weighted score
        merged_results = list(result_groups.values())
        merged_results.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
        
        logging.info(f"Merged {len(all_results)} results into {len(merged_results)} unique results")
        return merged_results
    
    def get_similar_queries(self, query: str, limit: int = 5) -> List[str]:
        """Get similar queries from query history"""
        # This would require storing query history
        # For now, return empty list
        return [] 