import logging
import time
import uuid
from typing import Optional, Dict, Any
from datetime import date, datetime

from open_webui.internal.db import Base, get_db

from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, String, Text, JSON, Integer, Date
from sqlalchemy.sql import text

log = logging.getLogger(__name__)

####################
# Daily Report DB Schema
####################

class DailyReport(Base):
    __tablename__ = "daily_report"
    
    id = Column(String, primary_key=True)
    chat_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    conversation_analysis = Column(Text, nullable=False)
    created_at = Column(BigInteger, nullable=False)


class DailyReportModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    chat_id: str
    user_id: str
    conversation_analysis: str
    created_at: int


####################
# Forms
####################

class DailyReportForm(BaseModel):
    chat_id: str
    user_id: str
    conversation_analysis: str


####################
# Table Operations
####################

class DailyReportTable:
    def insert_new_daily_report(self, form_data: DailyReportForm) -> Optional[DailyReportModel]:
        """Insert a new daily report record"""
        try:
            with get_db() as db:
                daily_report = DailyReport(
                    id=str(uuid.uuid4()),
                    chat_id=form_data.chat_id,
                    user_id=form_data.user_id,
                    conversation_analysis=form_data.conversation_analysis,
                    created_at=int(time.time())
                )
                
                db.add(daily_report)
                db.commit()
                db.refresh(daily_report)
                
                return DailyReportModel.model_validate(daily_report)
                
        except Exception as e:
            log.error(f"Error inserting daily report: {e}")
            return None
    
    def get_daily_report_by_date(self, report_date: date, user_id: Optional[str] = None) -> list[DailyReportModel]:
        """Get daily reports for a specific date, optionally filtered by user"""
        try:
            with get_db() as db:
                # Convert date to timestamp range
                start_timestamp = int(datetime.combine(report_date, datetime.min.time()).timestamp())
                end_timestamp = int(datetime.combine(report_date, datetime.max.time()).timestamp())
                
                query = text("""
                    SELECT * FROM daily_report 
                    WHERE created_at BETWEEN :start_timestamp AND :end_timestamp
                    """ + (" AND user_id = :user_id" if user_id else "") + """
                    ORDER BY created_at DESC
                """)
                
                params = {"start_timestamp": start_timestamp, "end_timestamp": end_timestamp}
                if user_id:
                    params["user_id"] = user_id
                
                result = db.execute(query, params)
                reports = []
                
                for row in result:
                    reports.append(DailyReportModel.model_validate(dict(row._mapping)))
                
                return reports
                
        except Exception as e:
            log.error(f"Error getting daily reports: {e}")
            return []
    
    def get_daily_report_by_chat_id(self, chat_id: str) -> Optional[DailyReportModel]:
        """Get daily report for a specific chat"""
        try:
            with get_db() as db:
                query = text("SELECT * FROM daily_report WHERE chat_id = :chat_id")
                result = db.execute(query, {"chat_id": chat_id})
                row = result.fetchone()
                
                if row:
                    return DailyReportModel.model_validate(dict(row._mapping))
                
                return None
                
        except Exception as e:
            log.error(f"Error getting daily report by chat_id: {e}")
            return None
    
    def get_daily_reports_by_date_range(self, start_date: date, end_date: date, user_id: Optional[str] = None) -> list[DailyReportModel]:
        """Get daily reports for a date range, optionally filtered by user"""
        try:
            with get_db() as db:
                # Convert dates to timestamp range
                start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
                end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
                
                query = text("""
                    SELECT * FROM daily_report 
                    WHERE created_at BETWEEN :start_timestamp AND :end_timestamp
                    """ + (" AND user_id = :user_id" if user_id else "") + """
                    ORDER BY created_at DESC
                """)
                
                params = {"start_timestamp": start_timestamp, "end_timestamp": end_timestamp}
                if user_id:
                    params["user_id"] = user_id
                
                result = db.execute(query, params)
                reports = []
                
                for row in result:
                    reports.append(DailyReportModel.model_validate(dict(row._mapping)))
                
                return reports
                
        except Exception as e:
            log.error(f"Error getting daily reports by date range: {e}")
            return []
    
    def update_daily_report(self, id: str, updates: Dict[str, Any]) -> Optional[DailyReportModel]:
        """Update a daily report record"""
        try:
            with get_db() as db:
                query = text("""
                    UPDATE daily_report 
                    SET conversation_analysis = :conversation_analysis
                    WHERE id = :id
                """)
                
                result = db.execute(query, {
                    "id": id,
                    "conversation_analysis": updates.get("conversation_analysis")
                })
                db.commit()
                
                if result.rowcount > 0:
                    return self.get_daily_report_by_id(id)
                
                return None
                
        except Exception as e:
            log.error(f"Error updating daily report: {e}")
            return None
    
    def get_daily_report_by_id(self, id: str) -> Optional[DailyReportModel]:
        """Get daily report by ID"""
        try:
            with get_db() as db:
                query = text("SELECT * FROM daily_report WHERE id = :id")
                result = db.execute(query, {"id": id})
                row = result.fetchone()
                
                if row:
                    return DailyReportModel.model_validate(dict(row._mapping))
                
                return None
                
        except Exception as e:
            log.error(f"Error getting daily report by ID: {e}")
            return None
    
    def delete_daily_report_by_id(self, id: str) -> bool:
        """Delete a daily report record"""
        try:
            with get_db() as db:
                query = text("DELETE FROM daily_report WHERE id = :id")
                result = db.execute(query, {"id": id})
                db.commit()
                
                return result.rowcount > 0
                
        except Exception as e:
            log.error(f"Error deleting daily report: {e}")
            return False


# Global instance
DailyReports = DailyReportTable() 