import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from open_webui.services.chat_analytics_service import ChatAnalyticsService
from open_webui.utils.auth import get_verified_user
from open_webui.constants import ERROR_MESSAGES

log = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class AnalyticsRequest(BaseModel):
    query: str = Field(..., description="Natural language query for chat analysis")
    user_id: Optional[str] = Field(None, description="Filter to specific user")
    model_id: str = Field("gpt-3.5-turbo", description="LLM model to use")

class AnalyticsResponse(BaseModel):
    success: bool
    analysis: Optional[str] = None
    query: str
    sources: list
    message: Optional[str] = None

# Initialize service
analytics_service = ChatAnalyticsService()

@router.post("/analyze", response_model=AnalyticsResponse)
async def analyze_chat_data(
    request: AnalyticsRequest,
    user=Depends(get_verified_user)
):
    """
    Simple chat analytics endpoint using LangChain PGVector.
    
    Takes a natural language query and returns AI-generated insights
    based on vector similarity search of chat embeddings.
    """
    try:
        log.info(f"Received analytics request: {request.query}")
        
        result = analytics_service.analyze_chat_data(
            query=request.query,
            user_id=request.user_id,
            model_id=request.model_id
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
        
        log.info("Analytics completed successfully")
        return AnalyticsResponse(**result)
        
    except Exception as e:
        log.exception(f"Error in chat analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e)
        ) 