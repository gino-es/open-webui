import logging
from typing import Optional, List
from datetime import date, datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from open_webui.models.daily_report import DailyReports, DailyReportModel
from open_webui.utils.auth import get_verified_user, get_admin_user
from open_webui.services.daily_report_worker import daily_report_worker
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter()

############################
# Response Models
############################

class DailyReportResponse(BaseModel):
    id: str
    chat_id: str
    user_id: str
    conversation_analysis: str
    created_at: int

class DailyReportSummaryResponse(BaseModel):
    total_reports: int
    reports: List[DailyReportResponse]

class ForceProcessResponse(BaseModel):
    message: str
    target_date: str
    success: bool

############################
# Get Current Day Reports
############################

@router.get("/today", response_model=DailyReportSummaryResponse)
async def get_today_reports(user=Depends(get_verified_user)):
    """Get all daily reports for the current day"""
    try:
        today = date.today()
        reports = DailyReports.get_daily_report_by_date(today, user_id=user.id)
        
        return DailyReportSummaryResponse(
            total_reports=len(reports),
            reports=reports
        )
    except Exception as e:
        log.error(f"Error fetching today's reports: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch today's reports"
        )

############################
# Get Reports by Date
############################

@router.get("/date/{report_date}", response_model=DailyReportSummaryResponse)
async def get_reports_by_date(
    report_date: str,
    user=Depends(get_verified_user)
):
    """Get daily reports for a specific date"""
    try:
        # Parse date string (expecting YYYY-MM-DD format)
        target_date = datetime.strptime(report_date, "%Y-%m-%d").date()
        reports = DailyReports.get_daily_report_by_date(target_date, user_id=user.id)
        
        return DailyReportSummaryResponse(
            total_reports=len(reports),
            reports=reports
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD"
        )
    except Exception as e:
        log.error(f"Error fetching reports for date {report_date}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch reports for the specified date"
        )

############################
# Get Reports by Date Range
############################

@router.get("/range/{start_date}/{end_date}", response_model=DailyReportSummaryResponse)
async def get_reports_by_date_range(
    start_date: str,
    end_date: str,
    user=Depends(get_verified_user)
):
    """Get daily reports for a date range"""
    try:
        # Parse date strings (expecting YYYY-MM-DD format)
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if start > end:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Start date must be before or equal to end date"
            )
        
        reports = DailyReports.get_daily_report_by_date_range(start, end, user_id=user.id)
        
        return DailyReportSummaryResponse(
            total_reports=len(reports),
            reports=reports
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD"
        )
    except Exception as e:
        log.error(f"Error fetching reports for date range {start_date} to {end_date}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch reports for the specified date range"
        )

############################
# Get Report by Chat ID
############################

@router.get("/chat/{chat_id}", response_model=Optional[DailyReportResponse])
async def get_report_by_chat_id(
    chat_id: str,
    user=Depends(get_verified_user)
):
    """Get daily report for a specific chat"""
    try:
        report = DailyReports.get_daily_report_by_chat_id(chat_id)
        
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No daily report found for this chat"
            )
        
        # Check if user has access to this chat's report
        if report.user_id != user.id and user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this report"
            )
        
        return report
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error fetching report for chat {chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch report for the specified chat"
        )

############################
# Force Process Daily Reports
############################

@router.post("/force-process", response_model=ForceProcessResponse)
async def force_process_daily_reports(
    target_date: str,
    user=Depends(get_admin_user)
):
    """Force process daily reports for a specific date (admin only)"""
    try:
        # Parse date string (expecting YYYY-MM-DD format)
        parsed_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        
        # Force process the daily reports
        daily_report_worker.force_process_daily_reports(parsed_date)
        
        return ForceProcessResponse(
            message=f"Successfully triggered daily report processing for {target_date}",
            target_date=target_date,
            success=True
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD"
        )
    except Exception as e:
        log.error(f"Error forcing daily report processing for {target_date}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to force process daily reports"
        )

############################
# Get Daily Report Worker Status
############################

@router.get("/worker/status")
async def get_worker_status(user=Depends(get_admin_user)):
    """Get the status of the daily report worker (admin only)"""
    try:
        status_info = {
            "worker_running": daily_report_worker.worker_thread and daily_report_worker.worker_thread.is_alive(),
            "last_processed_date": daily_report_worker.last_processed_date.isoformat() if daily_report_worker.last_processed_date else None,
            "last_processing_time": daily_report_worker.last_processing_time.isoformat() if hasattr(daily_report_worker, 'last_processing_time') and daily_report_worker.last_processing_time else None,
            "daily_processing_hour": daily_report_worker.daily_processing_hour,
            "daily_processing_minute": daily_report_worker.daily_processing_minute,
            "tolerance_minutes": daily_report_worker.tolerance_minutes
        }
        
        return status_info
    except Exception as e:
        log.error(f"Error getting worker status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get worker status"
        )
