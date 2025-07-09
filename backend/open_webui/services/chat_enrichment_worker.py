import asyncio
import logging
import time
import json
import re
from typing import Optional, List, Dict, Any
from open_webui.retrieval.utils import get_embedding_function
from open_webui.config import RAG_EMBEDDING_ENGINE, RAG_EMBEDDING_MODEL
from open_webui.internal.db import get_db
from sqlalchemy import text

from open_webui.models.chat_message import ChatMessages, ChatMessageModel
from open_webui.models.chat_message_chunk import ChatMessageChunks, ChatMessageChunkModel

log = logging.getLogger(__name__)

class ChatEnrichmentWorker:
    def __init__(self):
        self.embedding_function = None
        self.llm_client = None
        self.is_running = False
        self.worker_task = None
        
        # Processing intervals
        self.llm_batch_size = 10  # Smaller batch for LLM (API rate limits)
        self.embedding_batch_size = 50  # Larger batch for embeddings
        self.llm_sleep_interval = 300  # 5 minutes between LLM processing
        self.embedding_sleep_interval = 600  # 10 minutes between embedding processing
        
        # Track last processing times
        self.last_llm_processing = 0
        self.last_embedding_processing = 0
        
        # Chunking configuration
        self.max_tokens_per_chunk = 512
        self.overlap_tokens = 50 # Overlap between chunks for better continuity
    
    def get_embedding_function(self):
        """Lazy load embedding function"""
        if self.embedding_function is None:
            try:
                # For local models (default), create the SentenceTransformer
                if RAG_EMBEDDING_ENGINE.value == "":
                    from sentence_transformers import SentenceTransformer
                    ef = SentenceTransformer(RAG_EMBEDDING_MODEL.value)
                else:
                    ef = None  # For remote models
                
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
                # Import the LLM client based on your configuration
                from open_webui.models.models import get_model_by_id
                from open_webui.services.chat_service import ChatService
                
                # Get a suitable model for classification (you can configure this)
                model_id = "phi3:mini"  # Lightweight model for classifications
                model = get_model_by_id(model_id)
                
                if model:
                    self.llm_client = ChatService(model)
                    log.info(f"Loaded LLM client for classification: {model_id}")
                else:
                    log.warning(f"Model {model_id} not found, classification will be skipped")
                    
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
    
    async def classify_message(self, content: str, role: str) -> Dict[str, Any]:
        """Use LLM to classify message intent, topic, and sentiment"""
        try:
            llm_client = self.get_llm_client()
            if not llm_client:
                return {
                    'intent': 'general',
                    'topic': 'general',
                    'sentiment': 0.0
                }
            
            # Simple, open-ended classification prompt
            classification_prompt = f"""
            Analyze this message and classify it. Return ONLY a JSON object.

            Message: "{content}"
            Role: {role}

            Return this JSON format:
            {{
              "intent": "describe what the person wants or is doing (clear and concise, only short words)",
              "topic": "what subject or area this is about (clear and concise, only short words)",
              "sentiment": -1.0 to 1.0
            }}

            Guidelines:
            - intent: What is the person trying to achieve? (e.g., "asking for help", "sharing information", "expressing frustration")
            - topic: What is this about? (e.g., "python programming", "relationship advice", "work stress")
            - sentiment: How positive/negative is the tone? (-1.0 = very negative, 0.0 = neutral, 1.0 = very positive)

            Be natural and descriptive. Return ONLY the JSON.
            """
            
            # Get LLM response
            response = await llm_client.chat_completion(
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=200
            )
            
            # Parse the response
            try:
                # Extract JSON from response
                response_text = response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
                # Clean up the response to extract JSON
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                classification = json.loads(response_text)
                
                # Validate and set defaults
                intent = classification.get('intent', 'general')
                topic = classification.get('topic', 'general')
                sentiment = classification.get('sentiment', 0.0)
                
                # Normalize intent and topic (lowercase, trim whitespace)
                intent = str(intent).lower().strip() if intent else 'general'
                topic = str(topic).lower().strip() if topic else 'general'
                
                # Validate sentiment range
                try:
                    sentiment = float(sentiment)
                    sentiment = max(-1.0, min(1.0, sentiment))  # Clamp between -1 and 1
                except (ValueError, TypeError):
                    sentiment = 0.0
                
                return {
                    'intent': intent,
                    'topic': topic,
                    'sentiment': sentiment
                }
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                log.error(f"Error parsing LLM classification response: {e}")
                return {
                    'intent': 'general',
                    'topic': 'general',
                    'sentiment': 0.0
                }
                
        except Exception as e:
            log.error(f"Error in LLM classification: {e}")
            return {
                'intent': 'general',
                'topic': 'general',
                'sentiment': 0.0
            }
    
    def update_message_llm_enrichment(self, message_id: str, intent: str, topic: str, sentiment: float) -> bool:
        """Update the message with LLM classification data"""
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
        """Update the message with embedding"""
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
        """Create chunks for a message and store their individual embeddings"""
        try:
            # Split content into chunks
            chunks = self.split_text_into_chunks(content)
            
            if not chunks:
                return True
            
            log.info(f"Created {len(chunks)} chunks for message {message_id}")
            
            # Generate embeddings for each chunk
            embedding_function = self.get_embedding_function()
            chunk_embeddings = embedding_function(chunks)
            
            # Store chunks with their embeddings
            with get_db() as db:
                for i, (chunk, chunk_embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    chunk_id = f"{message_id}_chunk_{i}"
                    chunk_embedding_json = json.dumps(chunk_embedding.tolist())
                    
                    query = text("""
                        INSERT INTO chat_message_chunk (chunk_id, msg_id, chunk_no, content, embedding, created_at)
                        VALUES (:chunk_id, :msg_id, :chunk_no, :content, :embedding, :created_at)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            created_at = EXCLUDED.created_at
                    """)
                    
                    db.execute(query, {
                        'chunk_id': chunk_id,
                        'msg_id': message_id,
                        'chunk_no': i,
                        'content': chunk,
                        'embedding': chunk_embedding_json,
                        'created_at': int(time.time())
                    })
                
                db.commit()
            
            return True
        except Exception as e:
            log.error(f"Error creating chunks for message {message_id}: {e}")
            return False
    
    async def process_llm_enrichment(self):
        """Process LLM classification for messages"""
        try:
            current_time = time.time()
            
            # Check if enough time has passed since last LLM processing
            if current_time - self.last_llm_processing < self.llm_sleep_interval:
                return 0
            
            # Get messages that need LLM classification
            pending_messages = self.get_messages_for_llm_enrichment(limit=self.llm_batch_size)
            
            if not pending_messages:
                return 0
            
            log.info(f"Processing {len(pending_messages)} messages for LLM classification")
            
            processed_count = 0
            
            for message in pending_messages:
                try:
                    # Classify message using LLM
                    classification = await self.classify_message(
                        message['content'], 
                        message['role']
                    )
                    
                    # Update message with classification data
                    success = self.update_message_llm_enrichment(
                        message['id'],
                        classification['intent'],
                        classification['topic'],
                        classification['sentiment']
                    )
                    
                    if success:
                        log.info(f"Completed LLM classification for message {message['message_id']}")
                        processed_count += 1
                    else:
                        log.error(f"Failed to update LLM classification for message {message['id']}")
                        
                except Exception as e:
                    log.error(f"Error processing LLM classification for message {message['id']}: {e}")
                    continue
            
            if processed_count > 0:
                self.last_llm_processing = current_time
            
            return processed_count
            
        except Exception as e:
            log.error(f"Error processing LLM enrichment: {e}")
            return 0
    
    async def process_embedding_enrichment(self):
        """Process embeddings for messages with smart chunking"""
        try:
            current_time = time.time()
            
            # Check if enough time has passed since last embedding processing
            if current_time - self.last_embedding_processing < self.embedding_sleep_interval:
                return 0
            
            # Get messages that need embeddings
            pending_messages = self.get_messages_for_embedding(limit=self.embedding_batch_size)
            
            if not pending_messages:
                return 0
            
            log.info(f"Processing {len(pending_messages)} messages for embeddings (with chunking)")
            
            processed_count = 0
            
            for message in pending_messages:
                try:
                    # Generate embedding for the full message
                    embedding_function = self.get_embedding_function()
                    embedding = embedding_function(message['content'])
                    
                    # Update message with embedding
                    success = self.update_message_embedding(
                        message['id'],
                        embedding
                    )
                    
                    if success:
                        # Create chunks with embeddings for better search
                        chunk_success = self.create_message_chunks_with_embeddings(
                            message['id'],
                            message['content']
                        )
                        
                        if chunk_success:
                            log.info(f"Completed embedding and chunking for message {message['message_id']}")
                            processed_count += 1
                        else:
                            log.error(f"Failed to create chunks for message {message['id']}")
                    else:
                        log.error(f"Failed to update embedding for message {message['id']}")
                        
                except Exception as e:
                    log.error(f"Error processing embedding for message {message['id']}: {e}")
                    continue
            
            if processed_count > 0:
                self.last_embedding_processing = current_time
            
            return processed_count
            
        except Exception as e:
            log.error(f"Error processing embedding enrichment: {e}")
            return 0
    
    async def worker_loop(self):
        """Main worker loop with separate processing for LLM and embeddings"""
        log.info("Chat enrichment worker started")
        self.is_running = True
        
        while self.is_running:
            try:
                # Process LLM classification (frequent, small batches)
                llm_processed = await self.process_llm_enrichment()
                
                # Process embeddings (infrequent, large batches)
                embedding_processed = await self.process_embedding_enrichment()
                
                if llm_processed == 0 and embedding_processed == 0:
                    # No work to do, sleep
                    await asyncio.sleep(30)  # Check every 30 seconds
                else:
                    # Work was done, sleep shorter
                    await asyncio.sleep(5)
                    
            except Exception as e:
                log.error(f"Error in enrichment worker loop: {e}")
                await asyncio.sleep(10)
        
        log.info("Chat enrichment worker stopped")
    
    async def start(self):
        """Start the enrichment worker"""
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self.worker_loop())
            log.info("Chat enrichment worker task created")
    
    async def stop(self):
        """Stop the enrichment worker"""
        self.is_running = False
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        log.info("Chat enrichment worker stopped")
    
    def get_stats(self) -> dict:
        """Get worker statistics"""
        try:
            with get_db() as db:
                # Get counts from the new chat_message table
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
                "is_running": self.is_running,
                "total_messages": total_count,
                "pending_llm_enrichment": pending_llm_count,
                "pending_embedding_enrichment": pending_embedding_count,
                "completed_enrichment": completed_count,
                "worker_task_running": not self.worker_task.done() if self.worker_task else False
            }
        except Exception as e:
            log.error(f"Error getting worker stats: {e}")
            return {"error": str(e)}
        
    def search_similar_messages(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        search_chunks: bool = False,
        intent_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
        sentiment_min: Optional[float] = None,
        sentiment_max: Optional[float] = None
    ) -> List[dict]:
        """Search for similar messages with optional filters"""
        try:
            with get_db() as db:
                # Build the base query
                if search_chunks:
                    # Search in message chunks
                    base_query = """
                        SELECT 
                            cm.id, cm.chat_id, cm.user_id, cm.role, cm.content, 
                            cm.message_id, cm.created_at, cm.updated_at,
                            cmc.chunk_id, cmc.chunk_no, cmc.content as chunk_content,
                            (cmc.embedding::vector) <=> (:query_embedding::vector) as similarity
                        FROM chat_message_chunk cmc
                        JOIN chat_message cm ON cmc.msg_id = cm.id
                        WHERE cmc.embedding IS NOT NULL
                    """
                else:
                    # Search in full messages
                    base_query = """
                        SELECT 
                            id, chat_id, user_id, role, content, message_id, 
                            created_at, updated_at,
                            (embedding::vector) <=> (:query_embedding::vector) as similarity
                        FROM chat_message 
                        WHERE embedding IS NOT NULL
                    """
                
                # Add filters
                filters = []
                params = {'query_embedding': f"[{','.join(map(str, query_embedding))}]"}
                
                if intent_filter:
                    filters.append("intent = :intent_filter")
                    params['intent_filter'] = intent_filter
                
                if topic_filter:
                    filters.append("topic = :topic_filter")
                    params['topic_filter'] = topic_filter
                
                if sentiment_min is not None:
                    filters.append("sentiment >= :sentiment_min")
                    params['sentiment_min'] = sentiment_min
                
                if sentiment_max is not None:
                    filters.append("sentiment <= :sentiment_max")
                    params['sentiment_max'] = sentiment_max
                
                if filters:
                    base_query += " AND " + " AND ".join(filters)
                
                # Add ordering and limit
                base_query += " ORDER BY similarity LIMIT :limit"
                params['limit'] = limit
                
                query = text(base_query)
                result = db.execute(query, params)
                
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            log.error(f"Error searching similar messages: {e}")
            return []

# Global worker instance
chat_enrichment_worker = ChatEnrichmentWorker()