import logging
from typing import Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from open_webui.retrieval.chat_analytics_retriever import ChatAnalyticsRetriever

log = logging.getLogger(__name__)

class ChatAnalyticsService:
    """
    Simple analytics service using your existing chat_embedding table.
    """
    
    def __init__(self):
        self.retriever = ChatAnalyticsRetriever(top_k=10)
    
    def analyze_chat_data(
        self, 
        query: str, 
        user_id: Optional[str] = None,
        model_id: str = "gpt-4"
    ) -> Dict[str, Any]:
        """
        Analyze chat data using natural language query.
        """
        try:
            log.info(f"Analyzing chat data for query: '{query}'")
            
            # Initialize LLM
            llm = ChatOpenAI(model=model_id)
            
            # Create RAG chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=self.retriever,
                return_source_documents=True
            )
            
            # Run analysis
            result = qa_chain({"query": query})
            
            return {
                "success": True,
                "analysis": result["result"],
                "query": query,
                "sources": [
                    {
                        "content": doc.page_content,
                        "role": doc.metadata.get("role"),
                        "user_id": doc.metadata.get("user_id"),
                        "created_at": doc.metadata.get("created_at")
                    }
                    for doc in result["source_documents"][:5]
                ]
            }
            
        except Exception as e:
            log.error(f"Error in chat analytics: {e}")
            return {
                "success": False,
                "message": f"Error analyzing chat data: {str(e)}",
                "query": query
            } 