import logging
from typing import List, Dict, Optional, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import Field

from open_webui.models.chat_embedding import ChatEmbeddings

log = logging.getLogger(__name__)

class ChatAnalyticsRetriever(BaseRetriever):
    """
    Custom retriever that works with your existing chat_embedding table
    """
    
    # Define Pydantic fields
    top_k: int = Field(default=10, description="Number of documents to retrieve")
    embeddings: Optional[HuggingFaceEmbeddings] = Field(default=None, description="Embedding model")
    
    def __init__(self, top_k: int = 10, **kwargs):
        super().__init__(top_k=top_k, **kwargs)
        
        # Initialize embeddings if not already set
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            log.info("Initialized local embedding model: all-MiniLM-L6-v2")
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant chat messages using vector similarity"""
        
        try:
            log.info(f"Searching for query: {query}")
            
            # Get query embedding using the same model as your worker
            query_embedding = self.embeddings.embed_query(query)
            log.info(f"Query embedding dimension: {len(query_embedding)}")
            
            # Use your existing search method
            results = ChatEmbeddings.search_similar_messages(
                query_embedding=query_embedding,
                limit=self.top_k
            )
            
            log.info(f"Found {len(results)} results from database")
            
            # Convert to LangChain Documents
            documents = []
            for result in results:
                doc = Document(
                    page_content=result['content'],
                    metadata={
                        'id': result['id'],
                        'chat_id': result['chat_id'],
                        'user_id': result['user_id'],
                        'role': result['role'],
                        'message_id': result['message_id'],
                        'similarity': result['similarity'],
                        'created_at': result['created_at']
                    }
                )
                documents.append(doc)
            
            log.info(f"Returning {len(documents)} documents")
            return documents
            
        except Exception as e:
            log.error(f"Error in vector search: {e}")
            log.error(f"Full error details: {str(e)}")
            raise 