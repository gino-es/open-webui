import json
import time
import uuid
from typing import Optional

from open_webui.internal.db import Base, get_db

from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, String, Text, JSON

####################
# Chat Embedding DB Schema
####################

class ChatEmbedding(Base):
    __tablename__ = "chat_embedding"
    
    id = Column(Text, primary_key=True)
    chat_id = Column(Text, nullable=False)
    user_id = Column(Text, nullable=False)
    role = Column(Text, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    message_id = Column(Text, nullable=True)  # Original message ID from chat JSON
    parent_id = Column(Text, nullable=True)
    data = Column(JSON, nullable=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)
    embedding = Column(Text, nullable=True)


class ChatEmbeddingModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    chat_id: str
    user_id: str
    role: str
    content: str
    message_id: Optional[str] = None
    parent_id: Optional[str] = None
    data: Optional[dict] = None
    meta: Optional[dict] = None
    created_at: int
    updated_at: int
    embedding: Optional[str] = None


####################
# Forms
####################

class ChatEmbeddingForm(BaseModel):
    chat_id: str
    user_id: str
    role: str
    content: str
    message_id: Optional[str] = None
    parent_id: Optional[str] = None
    data: Optional[dict] = None
    meta: Optional[dict] = None


####################
# Table Operations
####################

class ChatEmbeddingTable:
    def insert_new_chat_embedding(
        self, form_data: ChatEmbeddingForm
    ) -> Optional[ChatEmbeddingModel]:
        with get_db() as db:
            id = str(uuid.uuid4())
            ts = int(time.time_ns())
            
            chat_embedding = ChatEmbeddingModel(
                **{
                    "id": id,
                    "chat_id": form_data.chat_id,
                    "user_id": form_data.user_id,
                    "role": form_data.role,
                    "content": form_data.content,
                    "message_id": form_data.message_id,
                    "parent_id": form_data.parent_id,
                    "data": form_data.data,
                    "meta": form_data.meta,
                    "created_at": ts,
                    "updated_at": ts,
                }
            )

            result = ChatEmbedding(**chat_embedding.model_dump())
            db.add(result)
            db.commit()
            db.refresh(result)
            return ChatEmbeddingModel.model_validate(result) if result else None

    def get_chat_embeddings_by_chat_id(
        self, chat_id: str, skip: int = 0, limit: int = 50
    ) -> list[ChatEmbeddingModel]:
        with get_db() as db:
            messages = (
                db.query(ChatEmbedding)
                .filter_by(chat_id=chat_id)
                .order_by(ChatEmbedding.created_at.asc())
                .offset(skip)
                .limit(limit)
                .all()
            )
            return [ChatEmbeddingModel.model_validate(msg) for msg in messages]

    def get_messages_without_embeddings(
        self, limit: int = 100
    ) -> list[ChatEmbeddingModel]:
        with get_db() as db:
            messages = (
                db.query(ChatEmbedding)
                .filter(ChatEmbedding.embedding.is_(None))
                .order_by(ChatEmbedding.created_at.asc())
                .limit(limit)
                .all()
            )
            return [ChatEmbeddingModel.model_validate(msg) for msg in messages]

    def update_embedding_by_id(
        self, id: str, embedding: str
    ) -> Optional[ChatEmbeddingModel]:
        with get_db() as db:
            message = db.get(ChatEmbedding, id)
            if message:
                message.embedding = embedding
                message.updated_at = int(time.time_ns())
                db.commit()
                db.refresh(message)
                return ChatEmbeddingModel.model_validate(message)
            return None

    def delete_chat_embedding_by_id(self, id: str) -> bool:
        with get_db() as db:
            db.query(ChatEmbedding).filter_by(id=id).delete()
            db.commit()
            return True


ChatEmbeddings = ChatEmbeddingTable()