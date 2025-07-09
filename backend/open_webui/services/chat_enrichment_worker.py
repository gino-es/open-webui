import threading
import time
import logging
import json
import re
import os
from typing import Optional, List, Dict, Any
from open_webui.retrieval.utils import get_embedding_function
from open_webui.config import RAG_EMBEDDING_ENGINE, RAG_EMBEDDING_MODEL
from open_webui.internal.db import get_db
from sqlalchemy import text
from groq import Groq

from open_webui.models.chat_message import ChatMessages, ChatMessageModel
from open_webui.models.chat_message_chunk import ChatMessageChunks, ChatMessageChunkModel

log = logging.getLogger(__name__)

class ChatEnrichmentWorker:
    def __init__(self):
        self.embedding_function = None
        self.llm_client = None
        
        # Thread management
        self.worker_thread = None
        self.stop_event = threading.Event()
        
        # Processing intervals
        self.llm_batch_size = 10
        self.embedding_batch_size = 50
        self.llm_sleep_interval = 300  
        self.embedding_sleep_interval = 300
        
        # Track last processing times
        self.last_llm_processing = 0
        self.last_embedding_processing = 0
        
        # Chunking configuration
        self.max_tokens_per_chunk = 512
        self.overlap_tokens = 50
    
    def get_embedding_function(self):
        if self.embedding_function is None:
            try:
                if RAG_EMBEDDING_ENGINE.value == "":
                    from sentence_transformers import SentenceTransformer
                    ef = SentenceTransformer(RAG_EMBEDDING_MODEL.value)
                else:
                    ef = None
                
                self.embedding_function = get_embedding_function(
                    embedding_engine=RAG_EMBEDDING_ENGINE.value,
                    embedding_model=RAG_EMBEDDING_MODEL.value,
                    embedding_function=ef,
                    url="",
                    key="",
                    embedding_batch_size=self.embedding_batch_size
                )
                log.info(f"Loaded embedding function: {RAG_EMBEDDING_MODEL.value}")
            except Exception as e:
                log.error(f"Error loading embedding function: {e}")
                raise
        return self.embedding_function
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks of approximately 512 tokens"""
        if not text.strip():
            return []
        
        # First, try to split by paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph is too long, split by sentences
            if self.estimate_tokens(paragraph) > self.max_tokens_per_chunk:
                sentences = re.split(r'[.!?]+', paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    # If sentence is too long, split by words
                    if self.estimate_tokens(sentence) > self.max_tokens_per_chunk:
                        words = sentence.split()
                        temp_chunk = ""
                        for word in words:
                            if self.estimate_tokens(temp_chunk + " " + word) <= self.max_tokens_per_chunk:
                                temp_chunk += " " + word if temp_chunk else word
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                temp_chunk = word
                        if temp_chunk:
                            current_chunk += " " + temp_chunk if current_chunk else temp_chunk
                    else:
                        # Add sentence to current chunk
                        if self.estimate_tokens(current_chunk + " " + sentence) <= self.max_tokens_per_chunk:
                            current_chunk += " " + sentence if current_chunk else sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
            else:
                # Add paragraph to current chunk
                if self.estimate_tokens(current_chunk + " " + paragraph) <= self.max_tokens_per_chunk:
                    current_chunk += " " + paragraph if current_chunk else paragraph
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If we still have chunks that are too long, force split them
        final_chunks = []
        for chunk in chunks:
            if self.estimate_tokens(chunk) <= self.max_tokens_per_chunk:
                final_chunks.append(chunk)
            else:
                # Force split by words
                words = chunk.split()
                temp_chunk = ""
                for word in words:
                    if self.estimate_tokens(temp_chunk + " " + word) <= self.max_tokens_per_chunk:
                        temp_chunk += " " + word if temp_chunk else word
                    else:
                        if temp_chunk:
                            final_chunks.append(temp_chunk.strip())
                        temp_chunk = word
                if temp_chunk:
                    final_chunks.append(temp_chunk.strip())
        
        return final_chunks
    
    def get_llm_client(self):
        """Lazy load LLM client for classification"""
        if self.llm_client is None:
            try:
                self.llm_client = Groq(
                    api_key=os.getenv("GROQ_API_KEY")
                )
                log.info("Loaded Groq LLM client for classification")
                
            except Exception as e:
                log.error(f"Error loading LLM client: {e}")
                self.llm_client = None
        return self.llm_client
    
    def get_messages_for_llm_enrichment(self, limit: int = 100) -> List[dict]:
        """Get messages that need LLM classification"""
        messages = ChatMessages.get_messages_without_llm_enrichment(limit=limit)
        return [message.model_dump() for message in messages]
    
    def get_messages_for_embedding(self, limit: int = 100) -> List[dict]:
        """Get messages that need embeddings"""
        messages = ChatMessages.get_messages_without_embeddings(limit=limit)
        return [message.model_dump() for message in messages]
    
    def classify_message(self, content: str, role: str) -> Dict[str, Any]:
        """Synchronous LLM classification"""
        try:
            llm_client = self.get_llm_client()
            if not llm_client:
                return {'intent': 'general', 'topic': 'general', 'sentiment': 0.0}
            
            classification_prompt = f"""
            Analyze this message and classify it. Return ONLY a JSON object.

            Message: "{content}"
            Role: {role}

            Return this JSON format:
            {{
              "intent": "describe what the person wants or is doing (clear and concise, only short words)",
              "topic": "what subject or area this is about (clear and concise, only short words)",
              "sentiment": -1.0 to 1.0 (-1.0 = very negative, 0.0 = neutral, 1.0 = very positive)
            }}

            Be natural and descriptive. Return ONLY the JSON.
            """
            
            # Synchronous Groq API call
            response = llm_client.chat.completions.create(
                model=os.getenv("GROQ_MODEL_ID") or "llama3-8b-8192",
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            classification = json.loads(response_text)
            
            return {
                'intent': str(classification.get('intent', 'general')).lower().strip(),
                'topic': str(classification.get('topic', 'general')).lower().strip(),
                'sentiment': max(-1.0, min(1.0, float(classification.get('sentiment', 0.0))))
            }
            
        except Exception as e:
            log.error(f"Error in LLM classification: {e}")
            return {'intent': 'general', 'topic': 'general', 'sentiment': 0.0}
    
    def update_message_llm_enrichment(self, message_id: str, intent: str, topic: str, sentiment: float) -> bool:
        try:
            result = ChatMessages.update_message_enrichment(
                id=message_id,
                intent=intent,
                topic=topic,
                sentiment=sentiment
            )
            return result is not None
        except Exception as e:
            log.error(f"Error updating LLM enrichment for message {message_id}: {e}")
            return False
    
    def update_message_embedding(self, message_id: str, embedding: List[float]) -> bool:
        try:
            embedding_json = json.dumps(embedding)
            result = ChatMessages.update_message_enrichment(
                id=message_id,
                embedding=embedding_json
            )
            return result is not None
        except Exception as e:
            log.error(f"Error updating embedding for message {message_id}: {e}")
            return False
    
    def create_message_chunks_with_embeddings(self, message_id: str, content: str) -> bool:
        try:
            chunks = self.split_text_into_chunks(content)
            
            if not chunks:
                return True
            
            log.info(f"Created {len(chunks)} chunks for message {message_id}")
            
            embedding_function = self.get_embedding_function()
            chunk_embeddings = embedding_function(chunks)
            
            with get_db() as db:
                for i, (chunk, chunk_embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    chunk_id = f"{message_id}_chunk_{i}"
                    
                    # Handle different embedding formats
                    if hasattr(chunk_embedding, 'tolist'):
                        chunk_embedding_json = json.dumps(chunk_embedding.tolist())
                    elif isinstance(chunk_embedding, list):
                        chunk_embedding_json = json.dumps(chunk_embedding)
                    else:
                        chunk_embedding_json = json.dumps(list(chunk_embedding))
                    
                    query = text("""
                        INSERT INTO chat_message_chunk (chunk_id, msg_id, chunk_no, embedding)
                        VALUES (:chunk_id, :msg_id, :chunk_no, :embedding)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            embedding = EXCLUDED.embedding
                    """)
                    
                    db.execute(query, {
                        'chunk_id': chunk_id,
                        'msg_id': message_id,
                        'chunk_no': i,
                        'embedding': chunk_embedding_json
                    })
                
                db.commit()
            
            return True
        except Exception as e:
            log.error(f"Error creating chunks for message {message_id}: {e}")
            return False
    
    def process_llm_enrichment(self):
        try:
            pending_messages = self.get_messages_for_llm_enrichment(limit=self.llm_batch_size)
            
            if not pending_messages:
                return
            
            log.info(f"Processing {len(pending_messages)} messages for LLM classification")
            
            for message in pending_messages:
                try:
                    classification = self.classify_message(
                        message['content'], 
                        message['role']
                    )
                    
                    # Update database
                    success = self.update_message_llm_enrichment(
                        message['id'],
                        classification['intent'],
                        classification['topic'],
                        classification['sentiment']
                    )
                    
                    if success:
                        log.info(f"Completed LLM classification for message {message['id']}")
                    
                except Exception as e:
                    log.error(f"Error processing LLM classification for message {message['id']}: {e}")
                    continue
                    
        except Exception as e:
            log.error(f"Error in LLM enrichment: {e}")
    
    def process_embedding_enrichment(self):
        """Process embeddings for messages"""
        try:
            pending_messages = self.get_messages_for_embedding(limit=self.embedding_batch_size)
            
            if not pending_messages:
                return
            
            log.info(f"Processing {len(pending_messages)} messages for embeddings")
            
            for message in pending_messages:
                try:
                    # Generate embedding
                    embedding_function = self.get_embedding_function()
                    embedding = embedding_function(message['content'])
                    
                    # Update message with embedding
                    success = self.update_message_embedding(
                        message['id'],
                        embedding
                    )
                    
                    if success:
                        # Create chunks
                        chunk_success = self.create_message_chunks_with_embeddings(
                            message['id'],
                            message['content']
                        )
                        
                        if chunk_success:
                            log.info(f"Completed embedding and chunking for message {message['id']}")
                    
                except Exception as e:
                    log.error(f"Error processing embedding for message {message['id']}: {e}")
                    continue
                    
        except Exception as e:
            log.error(f"Error in embedding enrichment: {e}")
    
    def worker_thread_loop(self):
        """Main worker loop running in separate thread"""
        log.info("Chat enrichment worker thread started")
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Process LLM classification
                if current_time - self.last_llm_processing >= self.llm_sleep_interval:
                    self.process_llm_enrichment()
                    self.last_llm_processing = current_time
                
                # Process embeddings
                if current_time - self.last_embedding_processing >= self.embedding_sleep_interval:
                    self.process_embedding_enrichment()
                    self.last_embedding_processing = current_time
                
                # Sleep for a short time
                self.stop_event.wait(30)  # Check every 30 seconds
                
            except Exception as e:
                log.error(f"Error in worker thread: {e}")
                time.sleep(10)
        
        log.info("Chat enrichment worker thread stopped")
    
    def start(self):
        """Start the worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.stop_event.clear()
            self.worker_thread = threading.Thread(target=self.worker_thread_loop, daemon=True)
            self.worker_thread.start()
            log.info("Chat enrichment worker thread started")
    
    def stop(self):
        """Stop the worker thread"""
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10)
        log.info("Chat enrichment worker thread stopped")
    
    def get_stats(self) -> dict:
        """Get worker statistics"""
        try:
            with get_db() as db:
                total_query = text("SELECT COUNT(*) FROM chat_message")
                total_count = db.execute(total_query).scalar()
                
                pending_llm_query = text("""
                    SELECT COUNT(*) FROM chat_message 
                    WHERE intent IS NULL OR intent = '' OR topic IS NULL OR topic = '' OR sentiment IS NULL
                """)
                pending_llm_count = db.execute(pending_llm_query).scalar()
                
                pending_embedding_query = text("""
                    SELECT COUNT(*) FROM chat_message 
                    WHERE embedding IS NULL OR embedding = ''
                """)
                pending_embedding_count = db.execute(pending_embedding_query).scalar()
                
                completed_query = text("""
                    SELECT COUNT(*) FROM chat_message 
                    WHERE intent IS NOT NULL AND topic IS NOT NULL AND sentiment IS NOT NULL 
                    AND embedding IS NOT NULL AND embedding != ''
                """)
                completed_count = db.execute(completed_query).scalar()
            
            return {
                "is_running": self.worker_thread.is_alive() if self.worker_thread else False,
                "thread_alive": self.worker_thread.is_alive() if self.worker_thread else False,
                "total_messages": total_count,
                "pending_llm_enrichment": pending_llm_count,
                "pending_embedding_enrichment": pending_embedding_count,
                "completed_enrichment": completed_count
            }
        except Exception as e:
            log.error(f"Error getting worker stats: {e}")
            return {"error": str(e)}

# Global worker instance
chat_enrichment_worker = ChatEnrichmentWorker()