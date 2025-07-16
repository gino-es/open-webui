import logging
from typing import List, Dict, Optional, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from open_webui.models.chat_message import ChatMessages
from open_webui.retrieval.utils import get_embedding_function
from open_webui.config import RAG_EMBEDDING_ENGINE, RAG_EMBEDDING_MODEL, RAG_EMBEDDING_BATCH_SIZE

log = logging.getLogger(__name__)

class ChatAnalyticsRetriever(BaseRetriever):
    
    # Define Pydantic fields for the attributes you want to set
    top_k: int = Field(default=50, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold for fallback")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize embedding function lazily (don't set as instance attribute)
        self._embedding_function = None
    
    def _get_embedding_function(self):
        """Get or create embedding function (lazy initialization)"""
        if self._embedding_function is None:
            try:
                if RAG_EMBEDDING_ENGINE.value == "":
                    from sentence_transformers import SentenceTransformer
                    ef = SentenceTransformer(RAG_EMBEDDING_MODEL.value)
                else:
                    ef = None
                
                self._embedding_function = get_embedding_function(
                    embedding_engine=RAG_EMBEDDING_ENGINE.value,
                    embedding_model=RAG_EMBEDDING_MODEL.value,
                    embedding_function=ef,
                    url="",
                    key="",
                    embedding_batch_size=RAG_EMBEDDING_BATCH_SIZE.value
                )
                
            except Exception as e:
                log.error(f"Error loading embedding function: {e}")
                raise
        
        return self._embedding_function

    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        # Step 1: Analyze query to determine search strategy
        search_strategy = self._analyze_query_strategy(query)
        log.info(f"Query: '{query[:50]}...' -> Strategy: {search_strategy}")
        
        # Step 2: Try primary strategy first
        results = self._search_with_strategy(query, search_strategy)
        
        # Step 3: Fallback if results are poor
        if self._should_fallback(results):
            fallback_strategy = "chunks" if search_strategy == "messages" else "messages"
            log.info(f"Primary strategy {search_strategy} failed, trying {fallback_strategy}")
            results = self._search_with_strategy(query, fallback_strategy)
        
        return self._convert_to_documents(results)
    
    def _analyze_query_strategy(self, query: str) -> str:
        """Analyze query to determine whether to use chunks or full messages"""
        query_lower = query.lower()
        
        # Keywords that indicate specific/detailed search (use chunks)
        chunk_indicators = [
            # Technical/code related
            'code', 'function', 'class', 'method', 'api', 'endpoint', 'error', 'bug',
            'exception', 'stack trace', 'log', 'syntax', 'compile', 'runtime',
            
            # Specific search terms
            'explain', 'details', 'specific', 'exact', 'precise', 'particular',
            'how to', 'what is', 'where is', 'when did', 'show me',
            
            # Document/content specific
            'document', 'file', 'content', 'text', 'paragraph', 'section',
            'line', 'word', 'phrase', 'snippet', 'example',
            
            # Troubleshooting
            'fix', 'solve', 'resolve', 'issue', 'problem', 'troubleshoot',
            'debug', 'diagnose', 'error message'
        ]
        
        # Keywords that indicate broad/overview search (use full messages)
        message_indicators = [
            # General/overview terms
            'overview', 'summary', 'general', 'overall', 'in general',
            'what are', 'how do users', 'what problems', 'user feedback',
            'trends', 'patterns', 'common', 'typical', 'usually',
            
            # Sentiment/feeling related
            'feel', 'opinion', 'experience', 'satisfaction', 'preference',
            'like', 'dislike', 'happy', 'frustrated', 'satisfied'
        ]
        
        # Count indicators
        chunk_score = sum(1 for indicator in chunk_indicators if indicator in query_lower)
        message_score = sum(1 for indicator in message_indicators if indicator in query_lower)
        
        # Query length analysis
        word_count = len(query.split())
        if word_count <= 3:
            # Short queries are usually broad
            message_score += 1
        elif word_count > 8:
            # Long queries are usually specific
            chunk_score += 1
        
        # Decision logic
        if chunk_score > message_score:
            return "chunks"
        elif message_score > chunk_score:
            return "messages"
        else:
            # Tie breaker: default to chunks for code/document heavy environment
            return "chunks"
    
    def _search_with_strategy(self, query: str, strategy: str) -> List[Dict]:
        """Search using the specified strategy"""
        query_embedding = self._get_embedding_function()(query)
        
        return ChatMessages.search_similar_messages(
            query_embedding=query_embedding,
            limit=self.top_k,
            search_chunks=(strategy == "chunks"),
            filters=None
        )
    
    def _should_fallback(self, results: List[Dict]) -> bool:
        """Determine if we should try the fallback strategy"""
        if not results:
            return True
        
        # Check if top result has good similarity
        top_similarity = results[0].get('similarity', 0)
        return top_similarity < self.similarity_threshold
    
    def _convert_to_documents(self, results: List[Dict]) -> List[Document]:
        """Convert database results to LangChain Documents"""
        documents = []
        
        for result in results:
            metadata = {
                'id': result['id'],
                'chat_id': result['chat_id'],
                'user_id': result['user_id'],
                'role': result['role'],
                'turn_number': result['turn_number'],
                'intent': result['intent'],
                'topic': result['topic'],
                'sentiment': result['sentiment'],
                'message_id': result['message_id'],
                'similarity': result['similarity'],
                'created_at': result['created_at']
            }
            
            # Add chunk info if available
            if result.get('chunk_id'):
                metadata['chunk_id'] = result['chunk_id']
                metadata['chunk_no'] = result['chunk_no']
            
            doc = Document(
                page_content=result['content'],
                metadata=metadata
            )
            documents.append(doc)
        
        return documents 