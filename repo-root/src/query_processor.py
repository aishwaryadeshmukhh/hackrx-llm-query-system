"""
Module: query_processor.py
Functionality: Complete query processing pipeline with LLM integration.
"""
import json
import re
import time
import warnings
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Suppress specific tokenizer warnings for BGE Reranker
warnings.filterwarnings(
    "ignore", 
    message=".*XLMRobertaTokenizerFast.*__call__.*method is faster.*", 
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*fast tokenizer.*__call__.*method is faster.*",
    category=UserWarning
)

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Pinecone embeddings integration
from .embed_and_index import generate_query_embedding_pinecone

class QueryProcessor:
    def __init__(self, pinecone_api_key: str, gemini_api_key: str, index_name: str = 'policy-index'):
        self.pinecone_api_key = pinecone_api_key
        self.gemini_api_key = gemini_api_key
        self.index_name = index_name
        self.quota_exceeded = False
        self.fallback_reason = None
        
        # Initialize Pinecone
        if PINECONE_AVAILABLE and pinecone_api_key and pinecone_api_key != 'dummy':
            try:
                from .embed_and_index import check_or_create_pinecone_index
                
                # Ensure index has correct dimensions for Pinecone inference
                print("🔍 Checking/creating Pinecone index with correct dimensions...")
                if check_or_create_pinecone_index(pinecone_api_key, index_name, 1024):
                    self.pc = Pinecone(api_key=pinecone_api_key)
                    self.index = self.pc.Index(index_name)
                    print("✅ Initialized Pinecone client and index")
                else:
                    print("❌ Failed to create/verify Pinecone index")
                    self.pc = None
                    self.index = None
            except Exception as e:
                print(f"Pinecone initialization error: {e}")
                self.pc = None
                self.index = None
        else:
            self.pc = None
            self.index = None
        
        # Use Pinecone embeddings - no fallbacks
        print("✅ Using Pinecone multilingual-e5-large embeddings")
        
        # Reranking disabled - using similarity scores only
        print("✅ Using similarity scores only (no reranking)")
        self.reranker = None
        self.reranker_type = "none"
        
        # Initialize Gemini
        if GENAI_AVAILABLE and gemini_api_key and gemini_api_key != 'dummy':
            try:
                genai.configure(api_key=gemini_api_key)
                # Try different models based on availability
                model_options = [
                    'gemini-2.5-flash',  # More available for students
                    'gemini-2.5-pro',        # Standard model
                    'gemini-2.5-pro'     # Premium model (might be limited)
                ]
                
                self.llm = None
                self.model_name = None
                
                # Try each model until one works
                for model_name in model_options:
                    try:
                        # Set generation config with temperature 0.7
                        # Gemini temperature ranges from 0-2, where 0 is deterministic and 2 is highly random
                        generation_config = {
                            #"temperature": 0.3,  # Medium-high creative temperature (default is 0.9)
                            "top_p": 0.95,
                            "top_k": 40
                        }
                        
                        test_model = genai.GenerativeModel(
                            model_name=model_name,
                            generation_config=generation_config
                        )
                        
                        # Test with a simple query to verify access
                        test_response = test_model.generate_content("Hello")
                        if test_response:
                            self.llm = test_model
                            self.model_name = model_name
                            print(f"Successfully initialized Gemini model: {model_name} with temperature=0.7")
                            break
                    except Exception as e:
                        print(f"Failed to initialize {model_name}: {str(e)}")
                        if 'quota' in str(e).lower() or 'limit' in str(e).lower():
                            self.quota_exceeded = True
                            self.fallback_reason = f"Quota exceeded for {model_name}"
                        continue
                
                if not self.llm:
                    print("Warning: No Gemini models available, using fallback methods")
                    if not self.quota_exceeded:
                        self.fallback_reason = "No models accessible"
                    
            except Exception as e:
                print(f"Gemini initialization error: {e}")
                self.llm = None
                self.model_name = None
                if 'quota' in str(e).lower() or 'limit' in str(e).lower():
                    self.quota_exceeded = True
                    self.fallback_reason = "API quota exceeded"
                else:
                    self.fallback_reason = f"Initialization error: {str(e)}"
        else:
            self.llm = None
            self.model_name = None
            if gemini_api_key == 'dummy':
                self.fallback_reason = "Using dummy API key for testing"
            else:
                self.fallback_reason = "API key not provided"
    
    def _encode_query(self, query: str) -> List[float]:
        """Encode query using Pinecone's embedding service."""
        try:
            # Use Pinecone embeddings with the same API key
            return generate_query_embedding_pinecone(query, self.pinecone_api_key)
                    
        except Exception as e:
            print(f"❌ Pinecone query encoding error: {e}")
            # Return zero vector as fallback (1024 dimensions for multilingual-e5-large)
            return [0.0] * 1024
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get current API status and recommendations."""
        status = {
            'gemini_available': self.llm is not None,
            'model_name': getattr(self, 'model_name', None),
            'quota_exceeded': getattr(self, 'quota_exceeded', False),
            'fallback_reason': getattr(self, 'fallback_reason', None),
            'recommendations': []
        }
        
        if self.quota_exceeded:
            status['recommendations'].extend([
                "🚨 Gemini API quota exceeded - system is using fallback methods",
                "💡 Wait for quota reset (usually 24 hours) or upgrade your API plan",
                "📚 Student tier: Check Google AI Studio for quota details",
                "⚡ Fallback methods provide ~70% accuracy vs LLM's 95%"
            ])
        elif not self.llm:
            status['recommendations'].extend([
                "⚠️ LLM not available - using rule-based analysis",
                "🔑 Check your Gemini API key is valid",
                "📱 Try different model tiers (flash, pro, pro-1.5)"
            ])
        else:
            status['recommendations'].append(f"✅ Using {self.model_name} for optimal results")
        
        
        return status
    
    def _extract_json_from_response(self, response_text: str, context: str = "response") -> Optional[Dict]:
        """Helper method to robustly extract JSON from LLM responses."""
        if not response_text or not response_text.strip():
            print(f"🔍 Empty {context}, skipping JSON extraction")
            return None
        
        response_text = response_text.strip()
        
        # Try parsing the whole response first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Look for JSON block in the response
        if '{' in response_text and '}' in response_text:
            try:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_str = response_text[start:end]
                return json.loads(json_str)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"🔍 JSON extraction failed from {context}: {str(e)[:50]}...")
                print(f"🔍 Raw response preview: '{response_text[:100]}...'")
        
        print(f"🔍 No valid JSON found in {context}")
        return None
    
    def _make_llm_request_with_retry(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """Make LLM request with retry logic and quota detection."""
        if not self.llm:
            return None
            
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.generate_content(prompt)
                if response and response.text:
                    # Reset quota exceeded flag on successful request
                    if self.quota_exceeded:
                        print("✅ Quota exceeded flag reset - LLM is working again")
                        self.quota_exceeded = False
                    return response.text.strip()
                else:
                    print(f"🔍 Empty response on attempt {attempt + 1}")
                    if attempt < max_retries:
                        time.sleep(1)  # Brief pause before retry
                        continue
                    return None
                    
            except Exception as e:
                error_msg = str(e).lower()
                if 'quota' in error_msg or 'limit' in error_msg or 'exceeded' in error_msg:
                    print(f"⚠️ Gemini API quota exceeded: {e}")
                    # Set quota exceeded flag but don't return None immediately
                    self.quota_exceeded = True
                    if attempt < max_retries:
                        # Wait longer for quota errors before retrying
                        wait_time = min(5 * (attempt + 1), 15)  # Wait 5s, 10s, 15s
                        print(f"🔄 Waiting {wait_time}s before retry due to quota error...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("❌ All retries exhausted due to quota error")
                        # Don't return None for quota errors - let the caller handle it
                        return None
                elif attempt < max_retries:
                    print(f"🔄 Retry {attempt + 1} after error: {str(e)[:50]}...")
                    time.sleep(1)
                    continue
                else:
                    print(f"🔍 Final attempt failed: {e}")
                    return None
        
        return None

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract structured entities from natural language query using Gemini only."""
        if self.llm:
            return self._llm_entity_extraction(query)
        else:
            print("❌ No LLM available for entity extraction")
            return {
                "age": None,
                "gender": None,
                "procedure": None,
                "location": None,
                "policy_duration": None,
                "policy_type": None,
                "amount": None
            }
    
    def _llm_entity_extraction(self, query: str) -> Dict[str, Any]:
        """Use LLM for entity extraction - no fallbacks."""
        if not self.llm:
            print("❌ LLM not available for entity extraction")
            return {
                "age": None,
                "gender": None,
                "procedure": None,
                "location": None,
                "policy_duration": None,
                "policy_type": None,
                "amount": None
            }
            
        prompt = f"""
        Extract the following entities from this insurance/medical query: "{query}"
        
        Return a JSON object with these fields (use null if not found):
        - age: integer (patient age)
        - gender: string (M/F/Male/Female)
        - procedure: string (medical procedure/surgery)
        - location: string (city/location)
        - policy_duration: string (how old is the policy)
        - policy_type: string (type of insurance policy)
        - amount: number (any monetary amount mentioned)
        
        Example: {{"age": 46, "gender": "M", "procedure": "knee surgery", "location": "Pune", "policy_duration": "3 months", "policy_type": null, "amount": null}}
        
        Only return the JSON, no other text.
        """
        
        # Use robust request method
        response_text = self._make_llm_request_with_retry(prompt)
        if not response_text:
            print("❌ No response from LLM for entity extraction")
            return {
                "age": None,
                "gender": None,
                "procedure": None,
                "location": None,
                "policy_duration": None,
                "policy_type": None,
                "amount": None
            }
        
        # Extract JSON from response
        entities = self._extract_json_from_response(response_text, "entity extraction")
        if entities:
            print("✅ Successfully extracted entities using LLM")
            return entities
        else:
            print("❌ JSON extraction failed from LLM response")
            return {
                "age": None,
                "gender": None,
                "procedure": None,
                "location": None,
                "policy_duration": None,
                "policy_type": None,
                "amount": None
            }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query for hybrid search."""
        # Remove stop words and special characters
        import re
        import string
        
        # Common stop words
        stop_words = {
            "a", "an", "the", "in", "on", "at", "by", "for", "with", "about", 
            "against", "between", "into", "through", "during", "before", "after",
            "above", "below", "to", "from", "up", "down", "is", "am", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "having", "do",
            "does", "did", "doing", "would", "should", "could", "ought", "i'm",
            "you're", "he's", "she's", "it's", "we're", "they're", "i've", "you've",
            "we've", "they've", "i'd", "you'd", "he'd", "she'd", "we'd", "they'd",
            "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "isn't", "aren't",
            "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't",
            "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot",
            "couldn't", "mustn't", "let's", "that's", "who's", "what's", "here's",
            "there's", "when's", "where's", "why's", "how's", "of", "this", "that",
            "these", "those", "is", "are", "will", "be"
        }
        
        # Clean the query
        query = query.lower()
        # Remove punctuation
        query = re.sub(f'[{string.punctuation}]', ' ', query)
        # Split into words
        words = query.split()
        # Filter out stop words and single-character words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add any numbers as they are likely important
        numbers = re.findall(r'\d+', query)
        keywords.extend(numbers)
        
        # Deduplicate
        keywords = list(set(keywords))
        
        # Take the most important keywords (limit to avoid too restrictive filtering)
        if len(keywords) > 5:
            keywords = keywords[:5]
            
        return keywords

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from the query for context-aware search."""
        # Simple key term extraction
        words = query.lower().split()
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'whose'}
        
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms

    def _generate_expanded_queries(self, query: str) -> List[str]:
        """Generate expanded queries using synonyms and related terms."""
        expanded_queries = [query]
        
        # Common synonyms and related terms - INSURANCE DOMAIN SPECIFIC
        synonyms = {
            'payment': ['payment', 'fee', 'cost', 'charge', 'premium'],
            'coverage': ['coverage', 'benefits', 'protection', 'insurance'],
            'policy': ['policy', 'plan', 'agreement', 'contract'],
            'treatment': ['treatment', 'medical', 'healthcare', 'care', 'procedure'],
            'claim': ['claim', 'request', 'application', 'submission'],
            'benefit': ['benefit', 'advantage', 'coverage', 'protection'],
            'exclusion': ['exclusion', 'limitation', 'restriction', 'exception'],
            'deductible': ['deductible', 'excess', 'co-payment', 'co-insurance'],
            'premium': ['premium', 'payment', 'fee', 'cost'],
            'hospital': ['hospital', 'medical center', 'clinic', 'facility'],
            'surgery': ['surgery', 'operation', 'procedure', 'treatment'],
            'diagnosis': ['diagnosis', 'condition', 'illness', 'disease']
        }
        
        # Check for known terms and add synonyms
        query_lower = query.lower()
        for term, related_terms in synonyms.items():
            if term in query_lower:
                for related_term in related_terms:
                    if related_term not in query_lower:
                        expanded_query = query.replace(term, related_term)
                        if expanded_query not in expanded_queries:
                            expanded_queries.append(expanded_query)
        
        return expanded_queries

    def _process_search_results(self, results: Dict, min_score: float) -> List[Dict[str, Any]]:
        """Process and format search results with content type awareness."""
        processed_results = []
        
        if not results or not hasattr(results, 'matches'):
            return processed_results
        
        matches = results.matches
        
        for match in matches:
            pinecone_score = match.score
            metadata = match.metadata or {}
            
            if pinecone_score >= min_score and metadata:
                # Extract content properly
                content = metadata.get('content', '')
                if not content:
                    # Try alternative content fields
                    content = metadata.get('text', '')
                if not content:
                    # For tables, use summary
                    if metadata.get('content_type') == 'table':
                        content = metadata.get('summary', '')
                
                result_item = {
                    "score": pinecone_score,
                    "content": content,
                    "text": content,  # Ensure 'text' field exists for compatibility
                    "content_type": metadata.get('content_type', 'text'),
                    "metadata": metadata,
                    "id": match.id,
                    "document_name": metadata.get('document_name', ''),
                    "page_number": metadata.get('page_number', 1),
                    "chunk_index": metadata.get('chunk_index', 0)
                }
                
                # Add table-specific info if available
                if metadata.get('content_type') == 'table':
                    result_item["table_info"] = {
                        "table_id": metadata.get('table_id', ''),
                        "num_rows": metadata.get('num_rows', 0),
                        "num_columns": metadata.get('num_columns', 0),
                        "columns": metadata.get('columns', [])
                    }
                
                processed_results.append(result_item)
        
        return processed_results

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate results based on content similarity."""
        seen_contents = set()
        deduplicated = []
        
        for result in results:
            content = result.get('content', '')[:100]  # Use first 100 chars for deduplication
            content_hash = hash(content)
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                deduplicated.append(result)
        
        return deduplicated

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize scores across different search methods."""
        if not results:
            return results
        
        # Get all scores for normalization
        scores = [r.get('score', 0) for r in results]
        if not scores:
            return results
        
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            normalized_results = results
        else:
            normalized_results = []
            for result in results:
                normalized_score = (result.get('score', 0) - min_score) / (max_score - min_score)
                result['normalized_score'] = normalized_score
                normalized_results.append(result)
        
        return normalized_results

    def _balance_content_types(self, results: List[Dict[str, Any]], max_text: int = 3, max_tables: int = 2) -> List[Dict[str, Any]]:
        """Balance results to ensure mix of text and table content."""
        text_results = [r for r in results if r.get('content_type') != 'table']
        table_results = [r for r in results if r.get('content_type') == 'table']
        
        balanced_results = []
        
        # Add top text results
        balanced_results.extend(text_results[:max_text])
        
        # Add top table results
        balanced_results.extend(table_results[:max_tables])
        
        # Sort by score for final ordering
        balanced_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return balanced_results

    async def _perform_context_search(self, query: str, top_k: int, min_score: float) -> List[Dict[str, Any]]:
        """Perform context-aware search using key terms."""
        try:
            # Extract key terms and concepts
            key_terms = self._extract_key_terms(query)
            
            if not key_terms:
                return []
            
            print(f"🔍 Context search using key terms: {key_terms}")
            
            all_results = []
            for term in key_terms:
                try:
                    # Generate embedding for the term
                    term_embedding = self._encode_query(term)
                    
                    # Search for this term
                    response = self.index.query(
                        vector=term_embedding,
                        top_k=top_k // len(key_terms) if len(key_terms) > 0 else top_k,
                        include_metadata=True
                    )
                    
                    # Process results with lower threshold for context
                    context_results = self._process_search_results(response, min_score * 0.8)
                    all_results.extend(context_results)
                    
                except Exception as e:
                    print(f"⚠️ Context search failed for term '{term}': {e}")
                    continue
            
            return all_results
            
        except Exception as e:
            print(f"❌ Error in context search: {str(e)}")
            return []

    async def _perform_expanded_search(self, query: str, top_k: int, min_score: float) -> List[Dict[str, Any]]:
        """Perform search with query expansion."""
        try:
            # Generate expanded queries
            expanded_queries = self._generate_expanded_queries(query)
            
            if len(expanded_queries) == 1:
                return []  # No expansion needed
            
            print(f"🔍 Expanded search using {len(expanded_queries)} queries")
            
            all_results = []
            for expanded_query in expanded_queries[1:]:  # Skip original query
                try:
                    # Generate embedding for expanded query
                    expanded_embedding = self._encode_query(expanded_query)
                    
                    # Search with expanded query
                    response = self.index.query(
                        vector=expanded_embedding,
                        top_k=top_k // len(expanded_queries),
                        include_metadata=True
                    )
                    
                    expanded_results = self._process_search_results(response, min_score)
                    all_results.extend(expanded_results)
                    
                except Exception as e:
                    print(f"⚠️ Expanded search failed for query '{expanded_query}': {e}")
                    continue
            
            return all_results
            
        except Exception as e:
            print(f"❌ Error in expanded search: {str(e)}")
            return []

    async def advanced_search_pinecone(self, query: str, top_k: int = 15, min_score: float = 0.05) -> List[Dict[str, Any]]:
        """
        Advanced multi-stage search with query expansion and semantic understanding.
        """
        try:
            print(f"🚀 Starting advanced search for: '{query}'")
            
            # Stage 1: Direct semantic search
            print(f"🔍 Stage 1: Direct semantic search for '{query}'")
            direct_embedding = self._encode_query(query)
            direct_response = self.index.query(
                vector=direct_embedding,
                top_k=top_k,
                include_metadata=True
            )
            direct_results = self._process_search_results(direct_response, min_score)
            
            # Stage 2: Query expansion and synonym search
            print(f"🔍 Stage 2: Query expansion for '{query}'")
            expanded_results = await self._perform_expanded_search(query, top_k, min_score)
            
            # Stage 3: Context-aware search
            print(f"🔍 Stage 3: Context-aware search for '{query}'")
            context_results = await self._perform_context_search(query, top_k, min_score)
            
            # Combine all results
            all_results = direct_results + expanded_results + context_results
            
            if not all_results:
                print("⚠️ No results found in advanced search")
                return []
            
            # Deduplicate results
            print(f"🔍 Deduplicating {len(all_results)} results...")
            deduplicated_results = self._deduplicate_results(all_results)
            
            # Normalize scores
            print(f"🔍 Normalizing scores for {len(deduplicated_results)} results...")
            normalized_results = self._normalize_scores(deduplicated_results)
            
            # Balance content types
            print(f"🔍 Balancing content types...")
            balanced_results = self._balance_content_types(normalized_results)
            
            # Sort by relevance score
            balanced_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            print(f"📊 Advanced search found {len(balanced_results)} unique results")
            return balanced_results[:top_k]
                    
        except Exception as e:
            print(f"❌ Error in advanced search: {str(e)}")
            # Fallback to basic search - use the synchronous version
            try:
                return self.semantic_search_with_similarity(query, top_k)
            except Exception as fallback_error:
                print(f"❌ Fallback search also failed: {fallback_error}")
                return []
    
    def _calculate_hybrid_score(self, candidate: Dict, keywords: List[str]) -> float:
        """Calculate a hybrid score based on vector similarity and keyword presence."""
        # Start with vector similarity score (typically 0-1)
        score = candidate.get("vector_score", 0.0)
        
        if not keywords:
            return score
            
        # Get text from candidate
        text = candidate.get("text", "").lower()
        
        # Count keywords present in the text
        keyword_count = sum(1 for kw in keywords if kw.lower() in text)
        
        # Boost score based on keyword matches (0.05 boost per keyword match)
        keyword_boost = min(0.3, keyword_count * 0.05)  # Cap at 0.3 to avoid dominating vector score
        
        # Combine scores
        hybrid_score = score + keyword_boost
        
        # Add debug info
        candidate["keyword_matches"] = keyword_count
        candidate["keyword_boost"] = keyword_boost
        
        return hybrid_score

    def _should_rerank(self, candidates: List[Dict], final_k: int) -> bool:
        """Decide if reranking is beneficial based on score distribution."""
        if len(candidates) <= final_k:
            return False
        
        # Get vector scores of top candidates
        top_scores = [c.get("vector_score", 0) for c in candidates[:final_k * 2]]
        
        if not top_scores:
            return True  # Always rerank if no scores
        
        # Calculate score variance - if scores are very similar, reranking helps
        import statistics
        try:
            score_std = statistics.stdev(top_scores) if len(top_scores) > 1 else 0
            # If standard deviation is low (scores are similar), reranking is beneficial
            return score_std < 0.1  # Threshold for "similar" scores
        except:
            return True  # Default to reranking if calculation fails

    def semantic_search_with_similarity(self, query: str, top_k: int = 3, query_embedding: Optional[list] = None) -> List[Dict]:
        """
        Enhanced hybrid search using vector similarity, keyword matching, and advanced search strategies:
        1. Retrieve top candidates using vector similarity 
        2. Apply advanced search techniques (context-aware, query expansion)
        3. Boost results that contain keywords from the query (post-retrieval)
        4. Return top results sorted by combined scores with context expansion
        """
        if not self.index:
            print("❌ Pinecone index not available")
            return []
        try:
            # Use provided query_embedding if available, else encode
            if query_embedding is not None:
                print("🔍 Using precomputed query embedding for search...")
                embedding = query_embedding
            else:
                embedding = self._encode_query(query)
                
            print(f"🔍 Retrieving top {top_k} candidates using enhanced search...")
            
            # Extract keywords for hybrid search (post-retrieval)
            keywords = self._extract_keywords(query)
            if keywords:
                print(f"🔍 Will apply keyword boosting after retrieval: {keywords}")
            
            # Get more candidates for post-filtering
            retrieve_k = min(top_k * 2, 20)  # Get more results but cap at 20
            
            # First retrieve with vector search only
            response = self.index.query(
                vector=embedding,
                top_k=retrieve_k,
                include_metadata=True
            )
            
            # Format results (already sorted by similarity score)
            candidates = []
            for match in response.matches:
                candidates.append({
                    "id": match.id,
                    "vector_score": match.score,
                    "text": match.metadata.get("text", ""),
                    "document_name": match.metadata.get("document_name", ""),
                    "page_number": match.metadata.get("page_number", 1),
                    "chunk_index": match.metadata.get("chunk_index", 0),
                    "content_type": match.metadata.get("content_type", "text")
                })
            
            if not candidates:
                print("⚠️ No candidates found")
                return []
                
            print(f"✅ Retrieved {len(candidates)} candidates with vector search")
            
            # Apply hybrid scoring that combines vector similarity with keyword matching
            if keywords:
                print("🔍 Applying hybrid scoring with keyword boosting...")
                for candidate in candidates:
                    hybrid_score = self._calculate_hybrid_score(candidate, keywords)
                    candidate["hybrid_score"] = hybrid_score
                    candidate["final_score"] = hybrid_score
                    
                # Re-sort based on hybrid scores
                candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
                print(f"✅ Re-ranked results using hybrid scoring (vector + keyword boost)")
            else:
                # Use similarity scores as final scores (no hybrid scoring)
                for candidate in candidates:
                    candidate["final_score"] = candidate["vector_score"]
                print("ℹ️ Using pure vector similarity scores (no keywords found)")
            
            # Take top-k after hybrid scoring
            candidates = candidates[:top_k]
            
            # Context expansion
            print(f"📄 Context expansion for {len(candidates)} chunks...")
            expanded_results = self._expand_context(candidates)
            
            # Add ranking metadata
            for i, result in enumerate(expanded_results):
                result["final_rank"] = i + 1
                result["ranking_method"] = "hybrid_post_retrieval" if keywords else "vector_similarity"
                
            self._print_ranking_summary(query, expanded_results)
            return expanded_results
        except Exception as e:
            print(f"❌ Search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _expand_context(self, candidates: List[Dict], context_chars: int = 500) -> List[Dict]:
        """Expand context around selected chunks by retrieving adjacent chunks."""
        expanded_results = []
        
        for candidate in candidates:
            try:
                doc_name = candidate["document_name"]
                chunk_index = candidate.get("chunk_index", 0)
                
                # Try to get adjacent chunks for context using the correct method name
                adjacent_chunks = self._get_adjacent_chunks_extended(doc_name, chunk_index, 25, 25)
                
                if adjacent_chunks:
                    # Combine with context
                    expanded_text = self._combine_chunks_with_context(
                        candidate.get("text", ""), 
                        adjacent_chunks,
                        context_chars
                    )
                    candidate["text"] = expanded_text
                    candidate["context_expanded"] = True
                    candidate["adjacent_chunks_count"] = len(adjacent_chunks)
                else:
                    candidate["context_expanded"] = False
                    candidate["adjacent_chunks_count"] = 0
                
                expanded_results.append(candidate)
                
            except Exception as e:
                print(f"⚠️ Context expansion failed for chunk: {e}")
                candidate["context_expanded"] = False
                expanded_results.append(candidate)
        
        return expanded_results

    def _combine_chunks_with_context(self, main_text: str, adjacent_chunks: List[Dict], context_chars: int = 500) -> str:
        """Combine main text with adjacent chunks for context expansion."""
        try:
            combined_text = main_text
            
            # Add adjacent chunks (limit to context_chars)
            for chunk in adjacent_chunks:
                chunk_text = chunk.get("text", "")
                if chunk_text and len(combined_text) < context_chars:
                    combined_text += f"\n{chunk_text}"
            
            # Truncate if too long
            if len(combined_text) > context_chars:
                combined_text = combined_text[:context_chars] + "..."
            
            return combined_text
            
        except Exception as e:
            print(f"⚠️ Error combining chunks: {e}")
            return main_text
    
    def _get_adjacent_chunks_extended(self, doc_name: str, chunk_index: int, chunks_before: int = 25, chunks_after: int = 25) -> List[Dict]:
        """Retrieve extended adjacent chunks (25 before + 25 after) from the same document."""
        try:
            if not self.index:
                return []
            
            # Create a dummy vector for metadata-only search (correct dimension)
            dummy_vector = [0.0] * 1024  # Fixed dimension for multilingual-e5-large
            
            # Query for all chunks from the same document
            response = self.index.query(
                vector=dummy_vector,
                top_k=1000,  # Get many chunks to find all adjacent ones
                include_metadata=True
            )
            
            # Filter and find adjacent chunks manually
            same_doc_chunks = []
            for match in response.matches:
                metadata = match.metadata or {}
                if metadata.get("document_name") == doc_name:
                    same_doc_chunks.append({
                        "text": metadata.get("text", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "chunk_id": match.id,
                        "score": match.score
                    })
            
            # Sort by chunk index to find adjacent chunks
            same_doc_chunks.sort(key=lambda x: x["chunk_index"])
            
            # Find chunks adjacent to our target
            adjacent = []
            target_found = False
            target_position = None
            
            for i, chunk in enumerate(same_doc_chunks):
                if chunk["chunk_index"] == chunk_index:
                    target_found = True
                    target_position = i
                    break
            
            if target_found:
                # Get 25 previous chunks
                start_idx = max(0, target_position - chunks_before)
                for j in range(start_idx, target_position):
                    prev_chunk = same_doc_chunks[j]
                    adjacent.append({
                        "text": prev_chunk["text"],
                        "chunk_index": prev_chunk["chunk_index"],
                        "position": "before",
                        "distance": target_position - j
                    })
                
                # Get 25 next chunks
                end_idx = min(len(same_doc_chunks), target_position + chunks_after + 1)
                for j in range(target_position + 1, end_idx):
                    next_chunk = same_doc_chunks[j]
                    adjacent.append({
                        "text": next_chunk["text"],
                        "chunk_index": next_chunk["chunk_index"],
                        "position": "after",
                        "distance": j - target_position
                    })
            else:
                print(f"⚠️ Target chunk {chunk_index} not found in document {doc_name}")
                return []
            
            print(f"🔍 Found {len(adjacent)} adjacent chunks for {doc_name} chunk {chunk_index} (target: 25 before + 25 after)")
            return adjacent
            
        except Exception as e:
            print(f"⚠️ Could not retrieve adjacent chunks: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_comprehensive_context(self, top_vectors: List[Dict]) -> str:
        """Create comprehensive context from top 5 vectors with their adjacent chunks."""
        context_sections = []
        
        for i, vector in enumerate(top_vectors, 1):
            doc_name = vector["document_name"]
            chunk_index = vector.get("chunk_index", 0)
            main_text = vector["text"]
            similarity_score = vector.get("score", 0.0)
            
            print(f"📄 Processing Vector {i}: Getting adjacent context for chunk {chunk_index} from {doc_name}")
            
            # Get 25 chunks before and 25 chunks after
            adjacent_chunks = self._get_adjacent_chunks_extended(doc_name, chunk_index, 25, 25)
            
            # Organize chunks
            before_chunks = [c for c in adjacent_chunks if c["position"] == "before"]
            after_chunks = [c for c in adjacent_chunks if c["position"] == "after"]
            
            # Sort by distance from main chunk
            before_chunks.sort(key=lambda x: x["distance"], reverse=True)  # Closest first
            after_chunks.sort(key=lambda x: x["distance"])  # Closest first
            
            # Build the section
            section_parts = []
            
            # Add header for this vector
            section_parts.append(f"=== VECTOR {i} (Similarity: {similarity_score:.3f}) ===")
            section_parts.append(f"Document: {doc_name}")
            section_parts.append(f"Main Chunk Index: {chunk_index}")
            section_parts.append("")
            
            # Add before context
            if before_chunks:
                section_parts.append(f"--- CONTEXT BEFORE (25 chunks) ---")
                for chunk in before_chunks:
                    section_parts.append(f"[Chunk {chunk['chunk_index']}] {chunk['text']}")
                section_parts.append("")
            
            # Add main chunk
            section_parts.append(f"--- MAIN CHUNK (Most Relevant) ---")
            section_parts.append(f"[Chunk {chunk_index}] {main_text}")
            section_parts.append("")
            
            # Add after context
            if after_chunks:
                section_parts.append(f"--- CONTEXT AFTER (25 chunks) ---")
                for chunk in after_chunks:
                    section_parts.append(f"[Chunk {chunk['chunk_index']}] {chunk['text']}")
                section_parts.append("")
            
            # Add summary for this vector
            total_context = len(before_chunks) + 1 + len(after_chunks)
            section_parts.append(f"--- END VECTOR {i} (Total chunks: {total_context}) ---")
            section_parts.append("")
            
            context_sections.append("\n".join(section_parts))
        
        # Combine all sections
        full_context = "\n".join(context_sections)
        
        print(f"📊 Created comprehensive context with {len(top_vectors)} vectors and their adjacent chunks")
        print(f"📊 Total context length: {len(full_context)} characters")
        
        return full_context
    
    def _print_ranking_summary(self, query: str, results: List[Dict]):
        """Print detailed ranking summary for debugging."""
        print(f"\n📊 Semantic Search Results Summary:")
        print(f"Query: '{query}'")
        print(f"Final results: {len(results)}")
        
        for i, result in enumerate(results, 1):
            similarity_score = result.get("vector_score", 0)
            final_score = result.get("final_score", similarity_score)
            doc_name = result.get("document_name", "Unknown")
            context_expanded = result.get("context_expanded", False)
            
            print(f"  {i}. Doc: {doc_name}")
            print(f"     Similarity: {similarity_score:.3f} | Final: {final_score:.3f} | Context: {'✅' if context_expanded else '❌'}")
            print(f"     Text: {result['text'][:100]}...")
            print()
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
         """Perform semantic search in Pinecone."""
         if not self.index:
             return []
         
         try:
             # Generate query embedding using helper method
             query_embedding = self._encode_query(query)
             
             # Search in Pinecone (returns a QueryResponse)
             response = self.index.query(
                 vector=query_embedding,
                 top_k=top_k,
                 include_metadata=True
             )
 
             # Format results from response.matches
             search_results = []
             for m in response.matches:
                 search_results.append({
                     "id": m.id,
                     "score": m.score,
                     "text": m.metadata.get("text", ""),
                     "document_name": m.metadata.get("document_name", ""),
                     "page_number": m.metadata.get("page_number", 1)
                 })
             
             return search_results
         except Exception as e:
             print(f"Search error: {e}")
             return []
    
    def evaluate_claim(self, query: str, entities: Dict, search_results: List[Dict]) -> Dict[str, Any]:
        """Use LLM to evaluate claim based on retrieved context - no fallbacks."""
        
        if self.llm:
            return self._llm_evaluation(query, entities, search_results)
        else:
            print("❌ No LLM available for claim evaluation")
            return {
                "decision": "error",
                "amount": None,
                "confidence": 0.0,
                "justification": "LLM not available for claim evaluation.",
                "relevant_clauses": [],
                "reasoning": "System requires LLM for claim evaluation"
            }
    
    def _check_llm_availability(self) -> bool:
        """Check if LLM is available and working."""
        if not self.llm:
            return False
        
        if self.quota_exceeded:
            return False
        
        # LLM object exists and quota not exceeded - assume it's available
        return True

    def _llm_evaluation_with_comprehensive_context(self, query: str, entities: Dict, comprehensive_context: str, top_vectors: List[Dict]) -> Dict[str, Any]:
        """Use LLM for evaluation with comprehensive adjacent context from 5 vectors."""
        # Only check if LLM object exists, don't check quota here
        if not self.llm:
            print("❌ LLM not available for evaluation - using fallback response")
            return self._generate_fallback_response(query, top_vectors, comprehensive_context)
        
        # Create a summary of the vectors for reference
        vector_summary = []
        for i, vector in enumerate(top_vectors, 1):
            # Handle different possible score field names
            score = vector.get('score', vector.get('vector_score', vector.get('final_score', 0.0)))
            document_name = vector.get('document_name', 'Unknown')
            vector_summary.append(f"Vector {i}: {document_name} (Similarity: {score:.3f})")
        
        prompt = f"""
        You are an insurance policy expert. Based on the comprehensive context from policy documents, provide a clear and concise answer in 2-3 sentences.

        QUERY: {query}

        EXTRACTED ENTITIES:
        {json.dumps(entities, indent=2)}

        TOP 5 VECTORS SUMMARY:
        {chr(10).join(vector_summary)}

        COMPREHENSIVE POLICY CONTEXT WITH ADJACENT CHUNKS:
        {comprehensive_context}

        Instructions:
        - Each vector section contains the most relevant chunk plus 25 chunks before and 25 chunks after it
        - Consider information from all 5 vector sections when forming your answer
        - Look for complementary information across different sections
        - If multiple sections discuss the same topic, synthesize the information
        - For coverage questions, check waiting periods, exclusions, and conditions across all sections
        - For amount/limit questions, look for specific numbers in any of the sections

        Return your answer as a JSON object with these fields:
        - answer: string (your concise answer in 2-3 sentences)
        - source_document: string (the primary document name you referenced)
        - relevant_sections: array of integers (which vector sections 1-5 were most relevant)

        For yes/no questions, format your answer as "Yes, [brief reason]" or "No, [brief reason]"
        Even if something is not explicitly mentioned, infer from the comprehensive context provided.
        Use information from multiple sections to provide a complete answer.

        Example response:
        {{
          "answer": "Yes, dental implants are covered up to ₹50,000 under your policy's dental care benefit. This is subject to a 6-month waiting period as mentioned in the policy terms.",
          "source_document": "policy.pdf",
          "relevant_sections": [1, 3, 4]
        }}

        Please provide accurate information based solely on the comprehensive policy context provided. Only return the JSON object, nothing else.
        """
        
        # Use robust request method
        response_text = self._make_llm_request_with_retry(prompt)
        if not response_text:
            # Check if quota is exceeded - if so, don't use fallback, just return a simple response
            if self.quota_exceeded:
                print("⚠️ LLM quota exceeded - returning simple response based on search results")
                return {
                    "answer": f"Based on the policy documents, I found relevant information about your query: '{query}'. However, I'm currently experiencing high demand and cannot provide a detailed analysis. Please refer to the policy documents for specific details.",
                    "source_document": top_vectors[0].get('document_name', 'Unknown') if top_vectors else "N/A",
                    "relevant_sections": [1] if top_vectors else []
                }
            else:
                print("❌ No response from LLM for comprehensive evaluation - using fallback")
                return self._generate_fallback_response(query, top_vectors, comprehensive_context)
        
        # Extract JSON from response
        evaluation = self._extract_json_from_response(response_text, "comprehensive evaluation")
        if evaluation:
            print("✅ Successfully completed comprehensive evaluation using LLM")
            # Add metadata about the comprehensive context
            evaluation['context_stats'] = {
                'total_vectors': len(top_vectors),
                'total_context_length': len(comprehensive_context),
                'chunks_per_vector': 51  # 25 before + 1 main + 25 after
            }
            return evaluation
        else:
            print("❌ JSON extraction failed from LLM comprehensive evaluation response - using fallback")
            return self._generate_fallback_response(query, top_vectors, comprehensive_context)

    def _generate_fallback_response(self, query: str, top_vectors: List[Dict], comprehensive_context: str) -> Dict[str, Any]:
        """Generate a fallback response when LLM is unavailable."""
        try:
            if not top_vectors:
                return {
                    "answer": "I apologize, but I couldn't find any relevant information in the policy documents to answer your question. Please refer to your full policy document for specific details.",
                    "source_document": "N/A",
                    "relevant_sections": []
                }
            
            # Extract key information from the top vectors
            best_vector = top_vectors[0] if top_vectors else {}
            document_name = best_vector.get('document_name', 'Unknown')
            content = best_vector.get('content', best_vector.get('text', ''))
            
            # Create a simple answer based on the content
            if content:
                # Truncate content for summary
                summary = content[:500] + "..." if len(content) > 500 else content
                
                # Try to extract a relevant answer from the content
                answer = self._extract_relevant_answer_from_content(query, content)
                
                return {
                    "answer": answer,
                    "source_document": document_name,
                    "relevant_sections": [1] if top_vectors else []
                }
            else:
                return {
                    "answer": "Based on the available policy information, I found some relevant content but cannot provide a specific answer without LLM processing. Please refer to your policy document for detailed information.",
                    "source_document": document_name,
                    "relevant_sections": [1] if top_vectors else []
                }
                
        except Exception as e:
            print(f"⚠️ Error generating fallback response: {e}")
            return {
                "answer": "I apologize, but I'm currently unable to process your query due to technical limitations. Please try again later or refer to your policy document for specific information.",
                "source_document": "N/A",
                "relevant_sections": []
            }

    def _extract_relevant_answer_from_content(self, query: str, content: str) -> str:
        """Extract a relevant answer from content when LLM is unavailable."""
        try:
            # Simple keyword-based answer extraction
            query_lower = query.lower()
            content_lower = content.lower()
            
            # Check for common insurance-related keywords
            if any(word in query_lower for word in ['covered', 'coverage', 'cover']):
                if any(word in content_lower for word in ['covered', 'coverage', 'cover', 'benefit']):
                    return f"Based on the policy information, this appears to be covered under your policy. Please refer to the specific terms and conditions in your policy document for complete details."
                else:
                    return f"Based on the policy information, this may not be covered under your policy. Please refer to the specific terms and conditions in your policy document for complete details."
            
            elif any(word in query_lower for word in ['waiting', 'period']):
                if any(word in content_lower for word in ['waiting', 'period', 'days', 'months', 'years']):
                    return f"The policy information indicates there may be waiting periods applicable. Please refer to the specific terms in your policy document for exact waiting period details."
                else:
                    return f"Based on the policy information, please refer to your policy document for specific waiting period details."
            
            elif any(word in query_lower for word in ['amount', 'limit', 'maximum', 'sum']):
                if any(word in content_lower for word in ['amount', 'limit', 'maximum', 'sum', 'rs', '₹', 'rupees']):
                    return f"The policy information contains details about amounts and limits. Please refer to the specific terms in your policy document for exact figures."
                else:
                    return f"Based on the policy information, please refer to your policy document for specific amount and limit details."
            
            else:
                return f"Based on the policy information found, please refer to your policy document for specific details about your query. The relevant information appears to be in the policy sections."
                
        except Exception as e:
            print(f"⚠️ Error extracting relevant answer: {e}")
            return "Based on the available policy information, please refer to your policy document for specific details about your query."
    
    async def process_query(self, query: str, query_embedding: Optional[list] = None) -> Dict[str, Any]:
        """Complete query processing pipeline with advanced search strategies and comprehensive adjacent context."""
        try:
            # Get API status for debugging
            api_status = self.get_api_status()
            
            # Step 1: Use advanced search for better results
            print("🔍 Step 1: Using advanced multi-stage search...")
            try:
                # Use provided query_embedding if available, else encode
                if query_embedding is not None:
                    print("🔍 Using precomputed query embedding for search...")
                    embedding = query_embedding
                else:
                    embedding = self._encode_query(query)
                
                # Use advanced search for better results
                search_results = await self.advanced_search_pinecone(query, top_k=15, min_score=0.05)
                
                if not search_results:
                    print("⚠️ Advanced search returned no results, falling back to basic search...")
                    # Fallback to basic search
                    search_results = self.semantic_search_with_similarity(query, top_k=5, query_embedding=embedding)
                
                # Take top 5 results for comprehensive context
                top_vectors = search_results[:5]
                
                print(f"✅ Retrieved {len(top_vectors)} top vectors using advanced search")
                
                # Step 2: Create comprehensive context with adjacent chunks for each vector
                print("🔍 Step 2: Creating comprehensive context with 25 before + 25 after chunks for each vector...")
                comprehensive_context = self._create_comprehensive_context(top_vectors)
                
            except Exception as e:
                print(f"❌ Advanced search error: {e}")
                # Fallback to basic search
                try:
                    search_results = self.semantic_search_with_similarity(query, top_k=5, query_embedding=query_embedding)
                    top_vectors = search_results[:5]
                    comprehensive_context = self._create_comprehensive_context(top_vectors)
                except Exception as fallback_error:
                    print(f"❌ Fallback search also failed: {fallback_error}")
                    top_vectors = []
                    comprehensive_context = ""
                    search_results = []
            
            # Step 3: Evaluate with comprehensive context
            print("🔍 Step 3: Evaluating with comprehensive context...")
            evaluation = self._llm_evaluation_with_comprehensive_context(query, {}, comprehensive_context, top_vectors)
            
            # Add search method information
            evaluation['search_method'] = 'advanced_multi_stage_search'
            evaluation['reranker_type'] = 'none'
            evaluation['reranker_available'] = False
            evaluation['hybrid_search_enabled'] = True
            evaluation['reranking_enabled'] = False
            evaluation['total_candidates_retrieved'] = len(search_results)
            evaluation['final_chunks_used'] = len(top_vectors)
            evaluation['adjacent_chunks_per_vector'] = 50  # 25 before + 25 after
            evaluation['advanced_search_features'] = {
                'context_aware_search': True,
                'query_expansion': True,
                'content_type_balancing': True,
                'result_deduplication': True,
                'score_normalization': True
            }
            
            # Add performance notes
            if not self.llm:
                evaluation['note'] = "❌ Analysis performed without LLM (required for full functionality)"
            evaluation['performance_note'] = "⚡ Using advanced multi-stage search with comprehensive adjacent context (25 before + 25 after) for each top vector"
            
            # Combine all results
            result = {
                "query": query,
                "search_results": search_results,
                "evaluation": evaluation,
                "api_status": api_status,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Query processing error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "query": query,
                "search_results": [],
                "evaluation": {
                    "decision": "error",
                    "amount": None,
                    "confidence": 0.0,
                    "justification": f"Processing error: {str(e)}",
                    "relevant_clauses": [],
                    "reasoning": "System error occurred"
                },
                "api_status": self.get_api_status(),
                "status": "error",
                "error": str(e)
            }

    def process_query_sync(self, query: str, query_embedding: Optional[list] = None) -> Dict[str, Any]:
        """Synchronous wrapper for process_query method."""
        import asyncio
        
        try:
            # Create a new event loop if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async method
            if loop.is_running():
                # If we're already in an event loop, use asyncio.run
                return asyncio.run(self.process_query(query, query_embedding))
            else:
                # Otherwise, use the existing loop
                return loop.run_until_complete(self.process_query(query, query_embedding))
                
        except Exception as e:
            print(f"❌ Error in synchronous wrapper: {e}")
            return {
                "query": query,
                "search_results": [],
                "evaluation": {
                    "decision": "error",
                    "amount": None,
                    "confidence": 0.0,
                    "justification": f"Processing error: {str(e)}",
                    "relevant_clauses": [],
                    "reasoning": "System error occurred"
                },
                "api_status": self.get_api_status(),
                "status": "error",
                "error": str(e)
            }
    
    async def process_queries_batch(self, queries: List[str], query_embeddings: Optional[List[List[float]]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple queries in parallel using asyncio.
        
        Args:
            queries: List of queries to process
            query_embeddings: Optional list of precomputed embeddings for each query
            
        Returns:
            List of results in the same order as input queries
        """
        import asyncio
        
        # Validate inputs
        if not queries:
            return []
        
        if query_embeddings and len(query_embeddings) != len(queries):
            raise ValueError("Number of embeddings must match number of queries")
        
        # Create semaphore to limit concurrent processing (prevent overwhelming the APIs)
        max_concurrent = min(10, len(queries))  # Limit to 10 concurrent queries
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_query_async(idx: int, query: str) -> Dict[str, Any]:
            """Process a single query with semaphore protection."""
            async with semaphore:
                try:
                    # Use precomputed embedding if available
                    embedding = query_embeddings[idx] if query_embeddings else None
                    
                    # Call the async process_query method directly
                    result = await self.process_query(query, embedding)
                    
                    # Add index to track original order
                    result["original_index"] = idx
                    return result
                    
                except Exception as e:
                    # Return error result in same format
                    return {
                        "query": query,
                        "search_results": [],
                        "evaluation": {
                            "decision": "error",
                            "amount": None,
                            "confidence": 0.0,
                            "justification": f"Batch processing error: {str(e)}",
                            "relevant_clauses": [],
                            "reasoning": "Batch processing system error"
                        },
                        "api_status": self.get_api_status(),
                        "status": "error",
                        "error": str(e),
                        "original_index": idx
                    }
        
        # Create tasks for all queries
        print(f"🚀 Starting batch processing of {len(queries)} queries with max {max_concurrent} concurrent...")
        tasks = []
        for idx, query in enumerate(queries):
            task = process_single_query_async(idx, query)
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Handle any exceptions that occurred
        processed_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result for failed queries
                error_result = {
                    "query": queries[idx],
                    "search_results": [],
                    "evaluation": {
                        "decision": "error",
                        "amount": None,
                        "confidence": 0.0,
                        "justification": f"Exception during processing: {str(result)}",
                        "relevant_clauses": [],
                        "reasoning": "Exception occurred"
                    },
                    "api_status": self.get_api_status(),
                    "status": "error",
                    "error": str(result),
                    "original_index": idx
                }
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        # Sort results by original index to maintain order
        processed_results.sort(key=lambda x: x.get("original_index", 0))
        
        # Remove the original_index field from final results
        final_results = []
        for result in processed_results:
            if "original_index" in result:
                del result["original_index"]
            final_results.append(result)
        
        print(f"✅ Batch processing completed in {end_time - start_time:.2f} seconds")
        print(f"📊 Processed {len(final_results)} queries successfully")
        
        return final_results