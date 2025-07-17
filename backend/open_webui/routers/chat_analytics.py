import logging
from typing import Optional, List
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
    model_id: Optional[str] = Field("gpt-4.1", description="LLM model to use")

class AnalyticsResponse(BaseModel):
    success: bool
    analysis: Optional[str] = None
    query: str
    tool_used: Optional[List[str]] = []  # ← Changed from str to List[str]
    sources: list = []
    sql_results: Optional[dict] = None
    message: Optional[str] = None

@router.post("/analyze", response_model=AnalyticsResponse)
async def analyze_chat_data(
    request: AnalyticsRequest,
    user=Depends(get_verified_user)
):
    try:
        service = ChatAnalyticsService(model_id=request.model_id)
        result = service.analyze_chat_data(query=request.query)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
        return AnalyticsResponse(**result)
        
    except Exception as e:
        log.exception(f"Error in chat analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e)
        )