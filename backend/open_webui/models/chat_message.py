import json
import time
import uuid
import logging
from typing import Optional, List, Dict, Any

from open_webui.internal.db import Base, get_db

from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, String, Text, JSON, Integer, Float, SmallInteger
from sqlalchemy.sql import text

log = logging.getLogger(__name__)

####################
# Chat Message DB Schema
####################

class ChatMessage(Base):
    __tablename__ = "chat_message"
    
    id = Column(Text, primary_key=True)
    chat_id = Column(Text, nullable=False)
    user_id = Column(Text, nullable=False)
    role = Column(Text, nullable=False)  # 'user' or 'assistant'
    turn_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    intent = Column(Text, nullable=True)
    topic = Column(Text, nullable=True)
    sentiment = Column(Float, nullable=True)
    message_id = Column(Text, nullable=True)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)
    embedding = Column(Text, nullable=True)


class ChatMessageChunk(Base):
    __tablename__ = "chat_message_chunk"
    
    chunk_id = Column(Text, primary_key=True)
    msg_id = Column(Text, nullable=False)
    chunk_no = Column(SmallInteger, nullable=False)
    embedding = Column(Text, nullable=True)


class ChatMessageModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    chat_id: str
    user_id: str
    role: str
    turn_number: int
    content: str
    intent: Optional[str] = None
    topic: Optional[str] = None
    sentiment: Optional[float] = None
    message_id: Optional[str] = None
    created_at: int
    updated_at: int
    embedding: Optional[str] = None


class ChatMessageChunkModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    chunk_id: str
    msg_id: str
    chunk_no: int
    embedding: Optional[str] = None


####################
# Forms
####################

class ChatMessageForm(BaseModel):
    chat_id: str
    user_id: str
    role: str
    turn_number: int
    content: str
    intent: Optional[str] = None
    topic: Optional[str] = None
    sentiment: Optional[float] = None
    message_id: Optional[str] = None


####################
# Table Operations
####################

class ChatMessageTable:
    def search_similar_messages(
        self, 
        query_embedding: list[float], 
        limit: int = 10,
        search_chunks: bool = False,
        filters: Optional[dict] = None
    ) -> list[dict]:
        """Search for similar chat messages using vector similarity"""
        with get_db() as db:
            # Convert embedding list to PostgreSQL vector format
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            # Build WHERE clause for filters
            where_conditions = ["embedding IS NOT NULL"]
            params = {'limit': limit}
            
            if filters:
                if filters.get('role'):
                    where_conditions.append("role = :role")
                    params['role'] = filters['role']
                if filters.get('intent'):
                    where_conditions.append("intent = :intent")
                    params['intent'] = filters['intent']
                if filters.get('topic'):
                    where_conditions.append("topic = :topic")
                    params['topic'] = filters['topic']
                if filters.get('user_id'):
                    where_conditions.append("user_id = :user_id")
                    params['user_id'] = filters['user_id']
                if filters.get('chat_id'):
                    where_conditions.append("chat_id = :chat_id")
                    params['chat_id'] = filters['chat_id']
            
            where_clause = " AND ".join(where_conditions)
            
            if search_chunks:
                # Search in chunks table
                query = text(f"""
                    SELECT 
                        cm.id, cm.chat_id, cm.user_id, cm.role, cm.turn_number,
                        cm.content, cm.intent, cm.topic, cm.sentiment, cm.message_id,
                        cm.created_at, cm.updated_at,
                        cmc.chunk_id, cmc.chunk_no,
                        (cmc.embedding::vector) <=> ('{embedding_str}'::vector) as similarity
                    FROM chat_message_chunk cmc
                    JOIN chat_message cm ON cmc.msg_id = cm.id
                    WHERE {where_clause.replace('embedding IS NOT NULL', 'cmc.embedding IS NOT NULL')}
                    ORDER BY (cmc.embedding::vector) <=> ('{embedding_str}'::vector)
                    LIMIT :limit
                """)
            else:
                # Search in main messages table
                query = text(f"""
                    SELECT 
                        id, chat_id, user_id, role, turn_number, content, intent, topic, sentiment,
                        message_id, created_at, updated_at,
                        (embedding::vector) <=> ('{embedding_str}'::vector) as similarity
                    FROM chat_message 
                    WHERE {where_clause}
                    ORDER BY (embedding::vector) <=> ('{embedding_str}'::vector)
                    LIMIT :limit
                """)
            
            result = db.execute(query, params)
            
            return [
                {
                    'id': row.id,
                    'chat_id': row.chat_id,
                    'user_id': row.user_id,
                    'role': row.role,
                    'turn_number': row.turn_number,
                    'content': row.content,
                    'intent': row.intent,
                    'topic': row.topic,
                    'sentiment': row.sentiment,
                    'message_id': row.message_id,
                    'created_at': row.created_at,
                    'updated_at': row.updated_at,
                    'chunk_id': getattr(row, 'chunk_id', None),
                    'chunk_no': getattr(row, 'chunk_no', None),
                    'similarity': float(row.similarity)
                }
                for row in result
            ]

    def search_by_intent_topic(
        self,
        intent: Optional[str] = None,
        topic: Optional[str] = None,
        limit: int = 50
    ) -> list[dict]:
        """Search messages by intent and/or topic"""
        with get_db() as db:
            where_conditions = []
            params = {'limit': limit}
            
            if intent:
                where_conditions.append("intent = :intent")
                params['intent'] = intent
            if topic:
                where_conditions.append("topic = :topic")
                params['topic'] = topic
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query = text(f"""
                SELECT 
                    id, chat_id, user_id, role, turn_number, content, intent, topic, sentiment,
                    message_id, created_at, updated_at
                FROM chat_message 
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            
            result = db.execute(query, params)
            
            return [
                {
                    'id': row.id,
                    'chat_id': row.chat_id,
                    'user_id': row.user_id,
                    'role': row.role,
                    'turn_number': row.turn_number,
                    'content': row.content,
                    'intent': row.intent,
                    'topic': row.topic,
                    'sentiment': row.sentiment,
                    'message_id': row.message_id,
                    'created_at': row.created_at,
                    'updated_at': row.updated_at
                }
                for row in result
            ]

    def insert_new_chat_message(
        self, form_data: ChatMessageForm
    ) -> Optional[ChatMessageModel]:
        with get_db() as db:
            id = str(uuid.uuid4())
            ts = int(time.time())
            
            chat_message = ChatMessageModel(
                **{
                    "id": id,
                    "chat_id": form_data.chat_id,
                    "user_id": form_data.user_id,
                    "role": form_data.role,
                    "turn_number": form_data.turn_number,
                    "content": form_data.content,
                    "message_id": form_data.message_id,
                    "created_at": ts,
                    "updated_at": ts,
                }
            )

            result = ChatMessage(**chat_message.model_dump())
            db.add(result)
            db.commit()
            db.refresh(result)
            return ChatMessageModel.model_validate(result) if result else None

    def get_chat_messages_by_chat_id(
        self, chat_id: str, skip: int = 0, limit: int = 50
    ) -> List[ChatMessageModel]:
        with get_db() as db:
            messages = (
                db.query(ChatMessage)
                .filter_by(chat_id=chat_id)
                .order_by(ChatMessage.turn_number.asc())
                .offset(skip)
                .limit(limit)
                .all()
            )
            return [ChatMessageModel.model_validate(msg) for msg in messages]

    def get_messages_without_embeddings(
        self, limit: int = 100
    ) -> List[ChatMessageModel]:
        with get_db() as db:
            messages = (
                db.query(ChatMessage)
                .filter(ChatMessage.embedding.is_(None))
                .order_by(ChatMessage.created_at.asc())
                .limit(limit)
                .all()
            )
            return [ChatMessageModel.model_validate(msg) for msg in messages]

    def get_messages_without_llm_enrichment(
        self, limit: int = 100
    ) -> List[ChatMessageModel]:
        with get_db() as db:
            messages = (
                db.query(ChatMessage)
                .filter(
                    (ChatMessage.intent.is_(None)) |
                    (ChatMessage.intent == '') |
                    (ChatMessage.topic.is_(None)) |
                    (ChatMessage.topic == '') |
                    (ChatMessage.sentiment.is_(None))
                )
                .order_by(ChatMessage.created_at.asc())
                .limit(limit)
                .all()
            )
            return [ChatMessageModel.model_validate(msg) for msg in messages]

    def update_message_enrichment(
        self, 
        id: str, 
        intent: Optional[str] = None,
        topic: Optional[str] = None,
        sentiment: Optional[float] = None,
        embedding: Optional[str] = None
    ) -> Optional[ChatMessageModel]:
        with get_db() as db:
            message = db.get(ChatMessage, id)
            if message:
                if intent is not None:
                    message.intent = intent
                if topic is not None:
                    message.topic = topic
                if sentiment is not None:
                    message.sentiment = sentiment
                if embedding is not None:
                    message.embedding = embedding
                
                message.updated_at = int(time.time())
                db.commit()
                db.refresh(message)
                return ChatMessageModel.model_validate(message)
            return None

    def delete_chat_message_by_id(self, id: str) -> bool:
        with get_db() as db:
            db.query(ChatMessage).filter_by(id=id).delete()
            db.commit()
            return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about chat messages"""
        with get_db() as db:
            # Total messages
            total_query = text("SELECT COUNT(*) FROM chat_message")
            total_count = db.execute(total_query).scalar()
            
            # Messages with embeddings
            embedding_query = text("SELECT COUNT(*) FROM chat_message WHERE embedding IS NOT NULL")
            embedding_count = db.execute(embedding_query).scalar()
            
            # Messages with LLM enrichment
            llm_query = text("""
                SELECT COUNT(*) FROM chat_message 
                WHERE intent IS NOT NULL AND topic IS NOT NULL AND sentiment IS NOT NULL
            """)
            llm_count = db.execute(llm_query).scalar()
            
            # Intent distribution
            intent_query = text("""
                SELECT intent, COUNT(*) as count 
                FROM chat_message 
                WHERE intent IS NOT NULL 
                GROUP BY intent 
                ORDER BY count DESC
            """)
            intent_distribution = [dict(row._mapping) for row in db.execute(intent_query)]
            
            # Topic distribution
            topic_query = text("""
                SELECT topic, COUNT(*) as count 
                FROM chat_message 
                WHERE topic IS NOT NULL 
                GROUP BY topic 
                ORDER BY count DESC
            """)
            topic_distribution = [dict(row._mapping) for row in db.execute(topic_query)]
            
            # Average sentiment
            sentiment_query = text("""
                SELECT AVG(sentiment) as avg_sentiment 
                FROM chat_message 
                WHERE sentiment IS NOT NULL
            """)
            avg_sentiment = db.execute(sentiment_query).scalar()
            
            return {
                'total_messages': total_count,
                'messages_with_embeddings': embedding_count,
                'messages_with_llm_enrichment': llm_count,
                'intent_distribution': intent_distribution,
                'topic_distribution': topic_distribution,
                'average_sentiment': float(avg_sentiment) if avg_sentiment else 0.0
            }


####################
# Utility Functions
####################

def calculate_turn_number(chat_data, message_id):
    """Calculate turn number based on message hierarchy"""
    if not chat_data or not message_id:
        return 1
    
    history = chat_data.chat.get("history", {})
    messages = history.get("messages", {})
    
    if message_id not in messages:
        return 1
    
    # Count all messages in the conversation up to this point
    turn_number = 1
    current_message = messages[message_id]
    
    # Traverse up the parent chain to count all messages
    while current_message and current_message.get("parentId"):
        parent_id = current_message["parentId"]
        if parent_id in messages:
            turn_number += 1
            current_message = messages[parent_id]
        else:
            break
    
    return turn_number


async def save_chat_message_record(
    chat_id: str,
    user_id: str,
    role: str,
    turn_number: int,
    content: str,
    message_id: str
):
    """Save a chat message to the chat_message table immediately"""
    form_data = ChatMessageForm(
        chat_id=chat_id,
        user_id=user_id,
        role=role,
        turn_number=turn_number,
        content=content,
        message_id=message_id,
    )
    
    result = ChatMessages.insert_new_chat_message(form_data)
    return result


ChatMessages = ChatMessageTable() 