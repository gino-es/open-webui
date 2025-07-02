import json
import time
import uuid
import logging
from typing import Optional

from open_webui.internal.db import Base, get_db

from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, String, Text, JSON
from sqlalchemy.sql import text

log = logging.getLogger(__name__)

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
            ts = int(time.time())
            
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
                message.updated_at = int(time.time())
                db.commit()
                db.refresh(message)
                return ChatEmbeddingModel.model_validate(message)
            return None

    def delete_chat_embedding_by_id(self, id: str) -> bool:
        with get_db() as db:
            db.query(ChatEmbedding).filter_by(id=id).delete()
            db.commit()
            return True

    def search_similar_messages(
        self, 
        query_embedding: list[float], 
        limit: int = 10
    ) -> list[dict]:
        """Search for similar chat messages using vector similarity"""
        with get_db() as db:
            # DEBUG: Check if table has data
            count_query = text("SELECT COUNT(*) FROM chat_embedding")
            count_result = db.execute(count_query)
            total_count = count_result.scalar()
            log.info(f"Total records in chat_embedding table: {total_count}")
            
            # DEBUG: Check a few sample records
            sample_query = text("SELECT id, content, embedding FROM chat_embedding LIMIT 3")
            sample_result = db.execute(sample_query)
            for row in sample_result:
                log.info(f"Sample record - ID: {row.id}, Content: {row.content[:50]}...")
                log.info(f"Embedding type: {type(row.embedding)}, Length: {len(row.embedding) if hasattr(row.embedding, '__len__') else 'N/A'}")
            
            # Convert embedding list to PostgreSQL vector format
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            log.info(f"Query embedding length: {len(query_embedding)}")
            log.info(f"Query embedding sample: {query_embedding[:5]}...")
            
            # SQL query for vector similarity search
            # Use string formatting for the embedding parameter to avoid casting issues
            query = text(f"""
                SELECT 
                    id, chat_id, user_id, role, content, message_id, 
                    created_at, updated_at,
                    (embedding::vector) <=> ('{embedding_str}'::vector) as similarity
                FROM chat_embedding 
                WHERE embedding IS NOT NULL
                ORDER BY (embedding::vector) <=> ('{embedding_str}'::vector)
                LIMIT :limit
            """)
            
            result = db.execute(query, {
                'limit': limit
            })
            
            return [
                {
                    'id': row.id,
                    'chat_id': row.chat_id,
                    'user_id': row.user_id,
                    'role': row.role,
                    'content': row.content,
                    'message_id': row.message_id,
                    'created_at': row.created_at,
                    'updated_at': row.updated_at,
                    'similarity': float(row.similarity)
                }
                for row in result
            ]


async def save_chat_embedding_record(
    chat_id: str,
    user_id: str,
    role: str,
    content: str,
    message_id: str,
    parent_id: Optional[str] = None
):
    """Save a chat message to the embedding table immediately"""
    form_data = ChatEmbeddingForm(
        chat_id=chat_id,
        user_id=user_id,
        role=role,
        content=content,
        message_id=message_id,
        parent_id=parent_id,
    )
    
    result = ChatEmbeddings.insert_new_chat_embedding(form_data)
    return result


ChatEmbeddings = ChatEmbeddingTable()