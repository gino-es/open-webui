import logging
import time
import uuid
from typing import Optional

from open_webui.internal.db import Base, get_db
from open_webui.env import SRC_LOG_LEVELS

from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, String, Text, JSON

####################
# Admin Query Log DB Schema
####################

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])


class AdminQueryLog(Base):
    __tablename__ = "admin_query_log"

    id = Column(String, primary_key=True)
    chat_id = Column(String, nullable=False)
    user_msg_id = Column(String, nullable=True)
    ai_msg_id = Column(String, nullable=True)
    sql_text = Column(Text, nullable=True)
    sql_summary = Column(JSON, nullable=True)
    vector_summary = Column(JSON, nullable=True)
    created_at = Column(BigInteger, nullable=False)


class AdminQueryLogModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    chat_id: str
    user_msg_id: Optional[str] = None
    ai_msg_id: Optional[str] = None
    sql_text: Optional[str] = None
    sql_summary: Optional[dict] = None
    vector_summary: Optional[dict] = None
    created_at: int  # timestamp in epoch


####################
# Forms
####################


class AdminQueryLogForm(BaseModel):
    chat_id: str
    user_msg_id: Optional[str] = None
    ai_msg_id: Optional[str] = None
    sql_text: Optional[str] = None
    sql_summary: Optional[dict] = None
    vector_summary: Optional[dict] = None


class AdminQueryLogResponse(BaseModel):
    id: str
    chat_id: str
    user_msg_id: Optional[str] = None
    ai_msg_id: Optional[str] = None
    sql_text: Optional[str] = None
    sql_summary: Optional[dict] = None
    vector_summary: Optional[dict] = None
    created_at: int


####################
# Table Operations
####################


class AdminQueryLogTable:
    def insert_query_log(self, form_data: AdminQueryLogForm) -> Optional[AdminQueryLogModel]:
        """Insert a new query log entry"""
        with get_db() as db:
            id = str(uuid.uuid4())
            query_log = AdminQueryLogModel(
                **{
                    "id": id,
                    "chat_id": form_data.chat_id,
                    "user_msg_id": form_data.user_msg_id,
                    "ai_msg_id": form_data.ai_msg_id,
                    "sql_text": form_data.sql_text,
                    "sql_summary": form_data.sql_summary,
                    "vector_summary": form_data.vector_summary,
                    "created_at": int(time.time()),
                }
            )

            result = AdminQueryLog(**query_log.model_dump())
            db.add(result)
            db.commit()
            db.refresh(result)
            return AdminQueryLogModel.model_validate(result) if result else None

    def get_query_log_by_id(self, id: str) -> Optional[AdminQueryLogModel]:
        """Get a query log entry by ID"""
        try:
            with get_db() as db:
                query_log = db.get(AdminQueryLog, id)
                return AdminQueryLogModel.model_validate(query_log) if query_log else None
        except Exception:
            return None

    def get_query_logs_by_chat_id(
        self, chat_id: str, skip: int = 0, limit: int = 50
    ) -> list[AdminQueryLogModel]:
        """Get query logs for a specific chat"""
        with get_db() as db:
            query_logs = (
                db.query(AdminQueryLog)
                .filter_by(chat_id=chat_id)
                .order_by(AdminQueryLog.created_at.desc())
                .offset(skip)
                .limit(limit)
                .all()
            )
            return [AdminQueryLogModel.model_validate(log) for log in query_logs]

    def get_query_logs_by_user_id(
        self, user_id: str, skip: int = 0, limit: int = 50
    ) -> list[AdminQueryLogModel]:
        """Get query logs for a specific user (via chat_id lookup)"""
        with get_db() as db:
            # Join with chat table to get user's query logs
            from open_webui.models.chats import Chat
            
            query_logs = (
                db.query(AdminQueryLog)
                .join(Chat, AdminQueryLog.chat_id == Chat.id)
                .filter(Chat.user_id == user_id)
                .order_by(AdminQueryLog.created_at.desc())
                .offset(skip)
                .limit(limit)
                .all()
            )
            return [AdminQueryLogModel.model_validate(log) for log in query_logs]

    def get_all_query_logs(
        self, skip: int = 0, limit: int = 50
    ) -> list[AdminQueryLogModel]:
        """Get all query logs with pagination"""
        with get_db() as db:
            query_logs = (
                db.query(AdminQueryLog)
                .order_by(AdminQueryLog.created_at.desc())
                .offset(skip)
                .limit(limit)
                .all()
            )
            return [AdminQueryLogModel.model_validate(log) for log in query_logs]

    def delete_query_log_by_id(self, id: str) -> bool:
        """Delete a query log entry by ID"""
        try:
            with get_db() as db:
                db.query(AdminQueryLog).filter_by(id=id).delete()
                db.commit()
                return True
        except Exception:
            return False

    def delete_query_logs_by_chat_id(self, chat_id: str) -> bool:
        """Delete all query logs for a specific chat"""
        try:
            with get_db() as db:
                db.query(AdminQueryLog).filter_by(chat_id=chat_id).delete()
                db.commit()
                return True
        except Exception:
            return False

    def delete_query_logs_by_user_id(self, user_id: str) -> bool:
        """Delete all query logs for a specific user"""
        try:
            with get_db() as db:
                from open_webui.models.chats import Chat
                
                # Get chat IDs for the user
                chat_ids = (
                    db.query(Chat.id)
                    .filter(Chat.user_id == user_id)
                    .subquery()
                )
                
                # Delete query logs for those chat IDs
                db.query(AdminQueryLog).filter(
                    AdminQueryLog.chat_id.in_(chat_ids)
                ).delete(synchronize_session=False)
                db.commit()
                return True
        except Exception:
            return False


# Global instance
AdminQueryLogs = AdminQueryLogTable() 