import threading
import time
import logging
import json
import uuid
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
from open_webui.internal.db import get_db
from sqlalchemy import text
from groq import Groq
import os

from open_webui.models.chat_message import ChatMessages
from open_webui.models.daily_report import DailyReports, DailyReportForm
from open_webui.services.prompts.chat_analytics_prompts import DAILY_REPORT_PROMPT

log = logging.getLogger(__name__)

class DailyReportWorker:
    def __init__(self):
        self.llm_client = None
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.processing_lock = threading.Lock()  
        
        self.daily_processing_hour = 1 # 1 AM
        self.daily_processing_minute = 0
        self.tolerance_minutes = 2
        self.sleep_interval = 30  

        self.last_processed_date = None
        self.max_tokens_per_chunk = 4000 
        self.process_interval = 60
        
    
    def get_llm_client(self):
        """Get or create Groq LLM client"""
        if self.llm_client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                log.warning("GROQ_API_KEY not set, daily report analysis will be limited")
                return None
            self.llm_client = Groq(api_key=api_key)
        return self.llm_client
    
    def get_yesterday_date(self) -> date:
        """Get yesterday's date in local time"""
        return (datetime.now() - timedelta(days=1)).date()
    
    def is_time_to_process(self) -> bool:
        """Check if it's time to process daily reports with tolerance"""
        now = datetime.now()
        
        target_time = now.replace(hour=self.daily_processing_hour, 
                                minute=self.daily_processing_minute, 
                                second=0, microsecond=0)
        
        # Allow 2 minutes tolerance 
        tolerance = timedelta(minutes=self.tolerance_minutes)
        time_diff = abs((now - target_time).total_seconds())
        
        return time_diff <= tolerance.total_seconds()
    
    def should_process_today(self) -> bool:
        """Check if we should process today (avoid duplicate processing)"""
        current_date = datetime.now().date()
        
        # If we haven't processed today yet, we can process
        if self.last_processed_date != current_date:
            return True
        
        # If we have processed today, check if it's been more than 23 hours
        if hasattr(self, 'last_processing_time'):
            time_since_last = datetime.now() - self.last_processing_time
            if time_since_last.total_seconds() > 23 * 3600:  # 23 hours
                return True
        
        return False
    
    def get_next_chat_needing_report(self, target_date: date) -> Optional[Dict[str, Any]]:
        """Get the next single chat that needs a daily report"""
        try:
            with get_db() as db:
                # Convert date to timestamp range (local time)
                start_timestamp = int(datetime.combine(target_date, datetime.min.time()).timestamp())
                end_timestamp = int(datetime.combine(target_date, datetime.max.time()).timestamp())
                
                query = text("""
                    SELECT DISTINCT 
                        cm.chat_id,
                        cm.user_id,
                        COUNT(cm.id) as message_count,
                        SUM(LENGTH(cm.content)) as total_content_length,
                        MIN(cm.created_at) as first_message_time,
                        MAX(cm.created_at) as last_message_time
                    FROM chat_message cm
                    WHERE cm.created_at BETWEEN :start_timestamp AND :end_timestamp
                    AND NOT EXISTS (
                        SELECT 1 FROM daily_report dr 
                        WHERE dr.chat_id = cm.chat_id 
                        AND DATE(to_timestamp(dr.created_at)) = CURRENT_DATE  
                    )
                    GROUP BY cm.chat_id, cm.user_id
                    ORDER BY total_content_length DESC
                    LIMIT 1
                """)
                
                result = db.execute(query, {
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                    "target_date": target_date,
                })
                
                row = result.fetchone()
                return dict(row._mapping) if row else None
                
        except Exception as e:
            log.error(f"Error getting next chat needing report for date {target_date}: {e}")
            return None

    def get_chat_messages_for_analysis(self, chat_id: str, target_date: date) -> Dict[str, Any]:
        """Get all messages from a specific chat for a specific date AND previous report context"""
        try:
            with get_db() as db:
                start_timestamp = int(datetime.combine(target_date, datetime.min.time()).timestamp())
                end_timestamp = int(datetime.combine(target_date, datetime.max.time()).timestamp())
                
                # Get messages for the target date
                messages_query = text("""
                    SELECT 
                        id, role, content, turn_number, created_at
                    FROM chat_message
                    WHERE chat_id = :chat_id 
                    AND created_at BETWEEN :start_timestamp AND :end_timestamp
                    ORDER BY turn_number ASC
                """)
                
                messages_result = db.execute(messages_query, {
                    "chat_id": chat_id,
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp
                })
                
                messages = []
                for row in messages_result:
                    messages.append(dict(row._mapping))
                
                # Get the most recent previous report for context
                previous_report_query = text("""
                    SELECT 
                        conversation_analysis,
                        created_at
                    FROM daily_report
                    WHERE chat_id = :chat_id 
                    AND DATE(to_timestamp(created_at)) < :target_date
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                
                previous_report_result = db.execute(previous_report_query, {
                    "chat_id": chat_id,
                    "target_date": target_date
                })
                
                previous_report = previous_report_result.fetchone()
                previous_context = None
                if previous_report:
                    previous_context = {
                        "analysis": previous_report.conversation_analysis,
                        "created_at": previous_report.created_at
                    }
                
                return {
                    "messages": messages,
                    "previous_report": previous_context
                }
                
        except Exception as e:
            log.error(f"Error getting messages for chat {chat_id}: {e}")
            return {"messages": [], "previous_report": None}
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        if not text:
            return 0
        
        # Rough estimation: 1 token ≈ 4 characters for code, 1 token ≈ 3 characters for text
        code_indicators = ['```', 'def ', 'class ', 'import ', 'function ', 'var ', 'let ', 'const ']
        is_code_heavy = any(indicator in text for indicator in code_indicators)
        
        if is_code_heavy:
            return len(text) // 4  # Code tokens
        else:
            return len(text) // 3  # Text tokens
    
    def chunk_conversation_for_analysis(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Split long conversations into manageable chunks for AI analysis"""
        if not messages:
            return []
        
        # Combine all messages into one conversation text
        full_conversation = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            full_conversation += f"{role.upper()}: {content}\n\n"
        
        total_tokens = self.estimate_tokens(full_conversation)
        log.info(f"Total conversation tokens: {total_tokens}, max per chunk: {self.max_tokens_per_chunk}")
        
        # If conversation is short enough, return as single chunk
        if total_tokens <= self.max_tokens_per_chunk:
            return [full_conversation]
        
        # Split into chunks based on token limits
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            message_text = f"{role.upper()}: {content}\n\n"
            message_tokens = self.estimate_tokens(message_text)
            
            # If this single message exceeds chunk limit, split the message itself
            if message_tokens > self.max_tokens_per_chunk:
                # First, save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                
                # Split the long message into smaller pieces
                message_chunks = self._split_long_message(message_text, role)
                for msg_chunk in message_chunks:
                    chunks.append(msg_chunk.strip())
                
            # If adding this message would exceed chunk limit, start new chunk
            elif current_tokens + message_tokens > self.max_tokens_per_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = message_text
                current_tokens = message_tokens
            else:
                current_chunk += message_text
                current_tokens += message_tokens
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_ai_analysis_with_fallback(self, conversation_chunks: List[str]) -> str:
        """Generate AI analysis using multiple models with fallback"""
        try:
            llm_client = self.get_llm_client()
            if not llm_client:
                return "AI analysis not available (LLM client not configured)"
            
            # Get fallback models from environment
            groq_models = os.getenv("GROQ_MODEL_IDS", "llama3.1-8b-instant").split(",")
            
            if len(conversation_chunks) == 1:
                # Single chunk - try multiple models
                return self._analyze_single_chunk_with_fallback(conversation_chunks[0], llm_client, groq_models)
            else:
                # Multiple chunks - analyze each and synthesize
                return self._analyze_multiple_chunks_with_fallback(conversation_chunks, llm_client, groq_models)
                
        except Exception as e:
            log.error(f"Error generating AI analysis: {e}")
            return f"AI analysis failed: {str(e)}"
    
    def _analyze_single_chunk_with_fallback(self, conversation: str, llm_client, groq_models: List[str], 
                                      previous_chunk_context: str = "", chunk_position: str = "") -> str:
        """Analyze a single conversation chunk with model fallback"""
        formatted_prompt = DAILY_REPORT_PROMPT.format(
            conversation=conversation,
            previous_context=previous_chunk_context,
            chunk_position=chunk_position
        )
        
        messages = [{"role": "user", "content": formatted_prompt}]
        
        # Try each model in order
        for model in groq_models:
            model = model.strip()
            if not model:
                continue
                
            try:
                log.debug(f"Trying Groq model '{model}' for conversation analysis")
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=800,
                    temperature=0.3
                )
                
                analysis = response.choices[0].message.content.strip()
                if analysis:
                    log.info(f"Groq model '{model}' analysis successful")
                    return analysis
                    
            except Exception as e:
                log.warning(f"Groq model '{model}' failed: {e}")
                continue
        
        # If all models fail, return default analysis
        log.warning("All Groq models failed, using default analysis")
        return "Analysis unavailable due to technical issues"
    
    def _analyze_multiple_chunks_with_fallback(self, conversation_chunks: List[str], llm_client, groq_models: List[str]) -> str:
        """Analyze multiple chunks with model fallback and synthesize"""
        current_analysis = ""
        
        for i, chunk in enumerate(conversation_chunks):
            # Build context from current analysis
            previous_context = ""
            if current_analysis:
                previous_context = f"""
                PREVIOUS ANALYSIS:
                {current_analysis}
                
                CURRENT CHUNK ({i+1}/{len(conversation_chunks)}):
                """
            
            # Send chunk with context of current analysis
            analysis = self._analyze_single_chunk_with_fallback(
                chunk, llm_client, groq_models, 
                previous_chunk_context=previous_context,
                chunk_position=f"{i+1}/{len(conversation_chunks)}"
            )
            
            # Update current analysis with new insights
            current_analysis = analysis
            
            if i < len(conversation_chunks) - 1:
                time.sleep(self.process_interval)
        
        # Return the final comprehensive analysis
        return current_analysis            
    
    def process_single_chat(self, chat: Dict[str, Any], target_date: date) -> bool:
        """Process a single chat and generate its daily report"""
        try:
            chat_id = chat['chat_id']
            user_id = chat['user_id']
            
            log.info(f"Processing daily report for chat {chat_id} ...")
            
            # Get all messages for analysis
            messages_data = self.get_chat_messages_for_analysis(chat_id, target_date)
            if not messages_data or not messages_data['messages']:
                log.warning(f"No messages found for chat {chat_id} on {target_date}")
                return False
            
            # Extract just the messages list
            messages = messages_data['messages']
            previous_report = messages_data['previous_report']
            
            # Chunk conversation if needed
            conversation_chunks = self.chunk_conversation_for_analysis(messages)
            log.info(f"Split conversation into {len(conversation_chunks)} chunks for analysis")
            
            # Generate AI-powered analysis with fallback
            conversation_analysis = self.generate_ai_analysis_with_fallback(conversation_chunks)
            
            # Create report form
            report_form = DailyReportForm(
                chat_id=chat_id,
                user_id=user_id,
                conversation_analysis=conversation_analysis
            )
            
            # Save report
            result = DailyReports.insert_new_daily_report(report_form)
            if result:
                log.info(f"Successfully created daily report for chat {chat_id}")
                return True
            else:
                log.error(f"Failed to save daily report for chat {chat_id}")
                return False
                
        except Exception as e:
            log.error(f"Error processing chat {chat['chat_id']}: {e}")
            return False
    
    def process_daily_reports(self, target_date: date):
        try:
            log.info(f"Starting daily report processing for {target_date}")
            processed_count = 0

            while True:
                # Add safety limit
                if processed_count >= 100:  # Maximum chats per session
                    log.warning(f"Reached maximum chats limit ({processed_count}) for {target_date}")
                    break

                chat = self.get_next_chat_needing_report(target_date)
                
                if not chat:
                    log.info(f"No more chats need reports for {target_date}")
                    break
                

                if self.process_single_chat(chat, target_date):
                    processed_count += 1
                    log.info(f"Processed chat {chat['chat_id']} for {target_date}")
                else:
                    chat_id = chat['chat_id']
                    log.error(f"Failed to process chat {chat_id}")
                
                time.sleep(self.process_interval)
            
            log.info(f"Daily report processing completed for {target_date}. Processed {processed_count} chats.")
            
        except Exception as e:
            log.error(f"Error in daily report processing: {e}")
    
    def _wait_for_database(self):
        log.info("Waiting for database connection...")
        delay = 2
        attempt = 0
        
        while True:  # Keep trying until database is available
            try:
                with get_db() as db:
                    db.execute(text("SELECT 1"))
                    log.info("Database connection established")
                    return
            except Exception as e:
                attempt += 1
                log.info(f"Database not ready (attempt {attempt}), waiting {delay}s... Error: {e}")
                time.sleep(delay)
                
                # Optional: increase delay gradually to avoid spam
                if attempt % 10 == 0:  # Every 10 attempts
                    delay = min(delay * 1.5, 30)  # Cap at 30 seconds
                    log.info(f"Increasing delay to {delay}s due to persistent connection issues")
    
    def worker_loop(self):
        log.info("Daily report worker started")
        self._wait_for_database()
        
        while not self.stop_event.is_set():
            try:
                if (self.is_time_to_process() and 
                    self.should_process_today() and
                    not self.processing_lock.locked()):
                    
                    with self.processing_lock:
                        log.info("Starting daily report processing...")
                        
                        self.last_processed_date = datetime.now().date()
                        self.last_processing_time = datetime.now()
                        
                        yesterday = self.get_yesterday_date()
                        self.process_daily_reports(yesterday)
                        
                        log.info(f"Daily report processing completed for {yesterday}")
                
                self.stop_event.wait(self.sleep_interval)
                
            except Exception as e:
                log.error(f"Worker error: {e}")
                time.sleep(10)
        
        log.info("Daily report worker stopped")
    
    def start(self):
        """Start the worker thread"""
        if not self.worker_thread or not self.worker_thread.is_alive():
            self.stop_event.clear()
            self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
            self.worker_thread.start()
            log.info("Daily report worker thread started")
    
    def stop(self):
        """Stop the worker thread"""
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10)
            log.info("Daily report worker thread stopped")
    
    def force_process_date(self, target_date: date):
        """Force process reports for a specific date (for testing or manual runs)"""
        log.info(f"Force processing daily reports for {target_date}")
        self.process_daily_reports(target_date)

    def _split_long_message(self, message_text: str, role: str) -> List[str]:
        """Split a single long message into smaller chunks"""
        chunks = []
        words = message_text.split()
        current_chunk = ""
        current_tokens = 0
        
        for word in words:
            word_with_space = word + " "
            word_tokens = self.estimate_tokens(word_with_space)
            
            if current_tokens + word_tokens > self.max_tokens_per_chunk and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = word_with_space
                current_tokens = word_tokens
            else:
                current_chunk += word_with_space
                current_tokens += word_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


# Global instance
daily_report_worker = DailyReportWorker() 