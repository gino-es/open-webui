import json
import time
import uuid
import logging
from typing import Optional, List, Dict, Any

from open_webui.internal.db import Base, get_db

from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, String, Text, SmallInteger
from sqlalchemy.sql import text

log = logging.getLogger(__name__)

####################
# Chat Message Chunk DB Schema
####################

class ChatMessageChunk(Base):
    __tablename__ = "chat_message_chunk"
    
    chunk_id = Column(Text, primary_key=True)
    msg_id = Column(Text, nullable=False)  # Foreign key to chat_message.id
    chunk_no = Column(SmallInteger, nullable=False)  # Order of chunk within message
    content = Column(Text, nullable=False)  # The actual chunk content
    embedding = Column(Text, nullable=True)  # JSON string of embedding vector
    created_at = Column(BigInteger, nullable=False)


class ChatMessageChunkModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    chunk_id: str
    msg_id: str
    chunk_no: int
    content: str
    embedding: Optional[str] = None
    created_at: int


####################
# Forms
####################

class ChatMessageChunkForm(BaseModel):
    msg_id: str
    chunk_no: int
    content: str


####################
# Table Operations
####################

class ChatMessageChunkTable:
    def insert_new_chunk(
        self, form_data: ChatMessageChunkForm
    ) -> Optional[ChatMessageChunkModel]:
        with get_db() as db:
            chunk_id = str(uuid.uuid4())
            ts = int(time.time())
            
            chunk = ChatMessageChunkModel(
                **{
                    "chunk_id": chunk_id,
                    "msg_id": form_data.msg_id,
                    "chunk_no": form_data.chunk_no,
                    "content": form_data.content,
                    "created_at": ts,
                }
            )

            result = ChatMessageChunk(**chunk.model_dump())
            db.add(result)
            db.commit()
            db.refresh(result)
            return ChatMessageChunkModel.model_validate(result) if result else None

    def get_chunks_by_message_id(
        self, msg_id: str, order_by_chunk_no: bool = True
    ) -> List[ChatMessageChunkModel]:
        with get_db() as db:
            query = db.query(ChatMessageChunk).filter_by(msg_id=msg_id)
            
            if order_by_chunk_no:
                query = query.order_by(ChatMessageChunk.chunk_no.asc())
            
            chunks = query.all()
            return [ChatMessageChunkModel.model_validate(chunk) for chunk in chunks]

    def get_chunks_without_embeddings(
        self, limit: int = 100
    ) -> List[ChatMessageChunkModel]:
        with get_db() as db:
            chunks = (
                db.query(ChatMessageChunk)
                .filter(ChatMessageChunk.embedding.is_(None))
                .order_by(ChatMessageChunk.created_at.asc())
                .limit(limit)
                .all()
            )
            return [ChatMessageChunkModel.model_validate(chunk) for chunk in chunks]

    def update_chunk_embedding(
        self, chunk_id: str, embedding: str
    ) -> Optional[ChatMessageChunkModel]:
        with get_db() as db:
            chunk = db.get(ChatMessageChunk, chunk_id)
            if chunk:
                chunk.embedding = embedding
                db.commit()
                db.refresh(chunk)
                return ChatMessageChunkModel.model_validate(chunk)
            return None

    def delete_chunk_by_id(self, chunk_id: str) -> bool:
        with get_db() as db:
            db.query(ChatMessageChunk).filter_by(chunk_id=chunk_id).delete()
            db.commit()
            return True

    def delete_chunks_by_message_id(self, msg_id: str) -> bool:
        with get_db() as db:
            db.query(ChatMessageChunk).filter_by(msg_id=msg_id).delete()
            db.commit()
            return True

    def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        msg_id_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        with get_db() as db:
            # Convert embedding list to PostgreSQL vector format
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            # Build the base query
            base_query = """
                SELECT 
                    cmc.chunk_id, cmc.msg_id, cmc.chunk_no, cmc.content, cmc.created_at,
                    cm.chat_id, cm.user_id, cm.role, cm.turn_number, cm.intent, cm.topic, cm.sentiment,
                    (cmc.embedding::vector) <=> (:query_embedding::vector) as similarity
                FROM chat_message_chunk cmc
                JOIN chat_message cm ON cmc.msg_id = cm.id
                WHERE cmc.embedding IS NOT NULL
            """
            
            # Add filters
            filters = []
            params = {'query_embedding': embedding_str, 'limit': limit}
            
            if msg_id_filter:
                filters.append("cmc.msg_id = :msg_id_filter")
                params['msg_id_filter'] = msg_id_filter
            
            if filters:
                base_query += " AND " + " AND ".join(filters)
            
            # Add ordering and limit
            base_query += " ORDER BY similarity LIMIT :limit"
            
            query = text(base_query)
            result = db.execute(query, params)
            
            return [
                {
                    'chunk_id': row.chunk_id,
                    'msg_id': row.msg_id,
                    'chunk_no': row.chunk_no,
                    'content': row.content,
                    'created_at': row.created_at,
                    'chat_id': row.chat_id,
                    'user_id': row.user_id,
                    'role': row.role,
                    'turn_number': row.turn_number,
                    'intent': row.intent,
                    'topic': row.topic,
                    'sentiment': row.sentiment,
                    'similarity': float(row.similarity)
                }
                for row in result
            ]

    def get_chunk_statistics(self) -> Dict[str, Any]:
        """Get statistics about message chunks"""
        with get_db() as db:
            # Total chunks
            total_query = text("SELECT COUNT(*) FROM chat_message_chunk")
            total_count = db.execute(total_query).scalar()
            
            # Chunks with embeddings
            embedding_query = text("SELECT COUNT(*) FROM chat_message_chunk WHERE embedding IS NOT NULL")
            embedding_count = db.execute(embedding_query).scalar()
            
            # Average chunks per message
            avg_chunks_query = text("""
                SELECT AVG(chunk_count) as avg_chunks_per_message
                FROM (
                    SELECT msg_id, COUNT(*) as chunk_count
                    FROM chat_message_chunk
                    GROUP BY msg_id
                ) as chunk_counts
            """)
            avg_chunks = db.execute(avg_chunks_query).scalar()
            
            # Messages with chunks
            messages_with_chunks_query = text("""
                SELECT COUNT(DISTINCT msg_id) FROM chat_message_chunk
            """)
            messages_with_chunks = db.execute(messages_with_chunks_query).scalar()
            
            return {
                'total_chunks': total_count,
                'chunks_with_embeddings': embedding_count,
                'average_chunks_per_message': float(avg_chunks) if avg_chunks else 0.0,
                'messages_with_chunks': messages_with_chunks
            }


async def save_chat_message_chunk_record(
    msg_id: str,
    chunk_no: int,
    content: str
):
    """Save a message chunk to the chat_message_chunk table"""
    form_data = ChatMessageChunkForm(
        msg_id=msg_id,
        chunk_no=chunk_no,
        content=content,
    )
    
    result = ChatMessageChunks.insert_new_chunk(form_data)
    return result


ChatMessageChunks = ChatMessageChunkTable() 