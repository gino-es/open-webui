import asyncio
import logging
import time
from typing import Optional
from open_webui.models.chat_embedding import ChatEmbeddings, ChatEmbeddingModel
from open_webui.retrieval.utils import get_embedding_function
from open_webui.config import RAG_EMBEDDING_ENGINE, RAG_EMBEDDING_MODEL

log = logging.getLogger(__name__)

class ChatEmbeddingWorker:
    def __init__(self):
        self.embedding_function = None
        self.is_running = False
        self.worker_task = None
        self.batch_size = 10
        self.sleep_interval = 5  # seconds between checks
    
    def get_embedding_function(self):
        """Lazy load embedding function"""
        if self.embedding_function is None:
            try:
                self.embedding_function = get_embedding_function(
                    embedding_engine=RAG_EMBEDDING_ENGINE.value,
                    embedding_model=RAG_EMBEDDING_MODEL.value,
                    embedding_function=None,
                    url="",
                    key="",
                    embedding_batch_size=self.batch_size
                )
                log.info(f"Loaded embedding function: {RAG_EMBEDDING_MODEL.value}")
            except Exception as e:
                log.error(f"Error loading embedding function: {e}")
                raise
        return self.embedding_function
    
    async def process_pending_embeddings(self):
        """Process all pending embeddings in batches"""
        try:
            # Get pending embeddings
            pending_messages = ChatEmbeddings.get_messages_without_embeddings(limit=self.batch_size)
            
            if not pending_messages:
                return 0
            
            log.info(f"Processing {len(pending_messages)} pending embeddings")
            
            # Update status to processing
            for message in pending_messages:
                ChatEmbeddings.update_data_by_id(
                    message.id,
                    {
                        "status": "processing",
                        "started_at": int(time.time())
                    }
                )
            
            # Generate embeddings in batch
            embedding_function = self.get_embedding_function()
            contents = [msg.content for msg in pending_messages]
            embeddings = embedding_function(contents)
            
            # Save embeddings
            for message, embedding in zip(pending_messages, embeddings):
                ChatEmbeddings.update_embedding_by_id(
                    message.id,
                    embedding=embedding,
                    data={
                        "status": "completed",
                        "completed_at": int(time.time()),
                        "model": RAG_EMBEDDING_MODEL.value
                    }
                )
                log.info(f"Completed embedding for message {message.message_id}")
            
            return len(pending_messages)
            
        except Exception as e:
            log.error(f"Error processing pending embeddings: {e}")
            # Mark failed messages
            for message in pending_messages:
                ChatEmbeddings.update_data_by_id(
                    message.id,
                    {
                        "status": "failed",
                        "error": str(e),
                        "failed_at": int(time.time())
                    }
                )
            return 0
    
    async def worker_loop(self):
        """Main worker loop - reads from table instead of queue"""
        log.info("Chat embedding worker started")
        self.is_running = True
        
        while self.is_running:
            try:
                # Process pending embeddings
                processed_count = await self.process_pending_embeddings()
                
                if processed_count == 0:
                    # No work to do, sleep longer
                    await asyncio.sleep(self.sleep_interval)
                else:
                    # Work was done, sleep shorter
                    await asyncio.sleep(1)
                    
            except Exception as e:
                log.error(f"Error in embedding worker loop: {e}")
                await asyncio.sleep(5)
        
        log.info("Chat embedding worker stopped")
    
    async def start(self):
        """Start the embedding worker"""
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self.worker_loop())
            log.info("Chat embedding worker task created")
    
    async def stop(self):
        """Stop the embedding worker"""
        self.is_running = False
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        log.info("Chat embedding worker stopped")
    
    def get_stats(self) -> dict:
        """Get worker statistics"""
        try:
            total_messages = ChatEmbeddings.get_total_count()
            pending_messages = ChatEmbeddings.get_pending_count()
            completed_messages = ChatEmbeddings.get_completed_count()
            failed_messages = ChatEmbeddings.get_failed_count()
            
            return {
                "is_running": self.is_running,
                "total_messages": total_messages,
                "pending_messages": pending_messages,
                "completed_messages": completed_messages,
                "failed_messages": failed_messages,
                "worker_task_running": not self.worker_task.done() if self.worker_task else False
            }
        except Exception as e:
            log.error(f"Error getting worker stats: {e}")
            return {"error": str(e)}
        
    def update_data_by_id(self, id: str, data: dict) -> Optional[ChatEmbeddingModel]:
      """Update the data field of a chat embedding record"""
      with get_db() as db:
          record = db.get(ChatEmbedding, id)
          if record:
              record.data = data
              record.updated_at = int(time.time())
              db.commit()
              db.refresh(record)
              return ChatEmbeddingModel.model_validate(record)
          return None

    def update_embedding_by_id(self, id: str, embedding: str, data: dict = None) -> Optional[ChatEmbeddingModel]:
        """Update the embedding and optionally the data field"""
        with get_db() as db:
            record = db.get(ChatEmbedding, id)
            if record:
                record.embedding = embedding
                if data:
                    record.data = data
                record.updated_at = int(time.time())
                db.commit()
                db.refresh(record)
                return ChatEmbeddingModel.model_validate(record)
            return None

    def get_total_count(self) -> int:
        """Get total number of chat embedding records"""
        with get_db() as db:
            return db.query(ChatEmbedding).count()

    def get_pending_count(self) -> int:
        """Get number of pending embedding records"""
        with get_db() as db:
            return db.query(ChatEmbedding).filter(
                ChatEmbedding.data.contains({"status": "pending"})
            ).count()

    def get_completed_count(self) -> int:
        """Get number of completed embedding records"""
        with get_db() as db:
            return db.query(ChatEmbedding).filter(
                ChatEmbedding.data.contains({"status": "completed"})
            ).count()

    def get_failed_count(self) -> int:
        """Get number of failed embedding records"""
        with get_db() as db:
            return db.query(ChatEmbedding).filter(
                ChatEmbedding.data.contains({"status": "failed"})
            ).count()

# Global worker instance
chat_embedding_worker = ChatEmbeddingWorker()