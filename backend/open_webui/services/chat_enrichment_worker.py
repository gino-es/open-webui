import threading
import time
import logging
import json
import re
import os
from typing import List, Dict, Any
from open_webui.retrieval.utils import get_embedding_function
from open_webui.config import RAG_EMBEDDING_ENGINE, RAG_EMBEDDING_MODEL
from open_webui.internal.db import get_db
from sqlalchemy import text
from groq import Groq

from open_webui.models.chat_message import ChatMessages
from open_webui.services.prompts.chat_analytics_prompts import CLASSIFICATION_PROMPT

log = logging.getLogger(__name__)

class ChatEnrichmentWorker:
    def __init__(self):
        self.embedding_function = None
        self.llm_client = None
        
        # Thread management
        self.worker_thread = None
        self.stop_event = threading.Event()
        
        # Processing config
        self.batch_size = 10
        self.sleep_interval = 60  # 1 minutes
        
        # Chunking config
        self.max_tokens = 512
        self.overlap_tokens = 50
    
    def get_embedding_function(self):
        if self.embedding_function is None:
            if RAG_EMBEDDING_ENGINE.value == "":
                from sentence_transformers import SentenceTransformer
                ef = SentenceTransformer(RAG_EMBEDDING_MODEL.value)
            else:
                ef = None
            
            self.embedding_function = get_embedding_function(
                embedding_engine=RAG_EMBEDDING_ENGINE.value,
                embedding_model=RAG_EMBEDDING_MODEL.value,
                embedding_function=ef,
                url="", key="", embedding_batch_size=self.batch_size
            )
        return self.embedding_function
    
    def get_llm_client(self):
        if self.llm_client is None:
            self.llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        return self.llm_client
    
    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        
        # More accurate estimation for different content types
        lines = text.split('\n')
        
        # Check if it's code-heavy
        code_indicators = ['def ', 'class ', 'import ', 'function ', 'var ', 'let ', 'const ']
        code_lines = sum(1 for line in lines if any(indicator in line for indicator in code_indicators))
        
        if code_lines > len(lines) * 0.2:  # 20% code lines
            # Code tokens: more tokens per character due to syntax
            return len(text) // 3
        else:
            # Text tokens: standard estimation
            return len(text) // 4
    
    def needs_chunking(self, content: str) -> bool:
        """Simple check if content needs chunking"""
        # Skip if too short
        if len(content) <= 1000 or self.estimate_tokens(content) <= self.max_tokens:
            return False
        
        # Chunk if has code or technical content
        technical_indicators = ['```', 'function', 'class', 'api', 'error', 'code']
        return any(indicator in content.lower() for indicator in technical_indicators)
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Smart chunking that handles both code and text content"""
        if not text.strip():
            return []
        
        # Detect content type for appropriate chunking
        content_type = self._detect_content_type(text)
        
        if content_type == "code":
            return self._split_code_aware(text)
        elif content_type == "mixed":
            return self._split_mixed_content(text)
        else:
            return self._split_text_aware(text)

    def _detect_content_type(self, text: str) -> str:
        """Detect if content is code, text, or mixed"""
        lines = text.split('\n')
        total_lines = len(lines)
        
        # Code indicators
        code_patterns = [
            r'^\s*(def|class|import|from|if|for|while|try|except|with|async|await)\s+',
            r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[=:]\s*',
            r'^\s*[{}[\]]\s*$',
            r'^\s*#\s+',
            r'^\s*//\s+',
            r'^\s*/\*',
            r'^\s*\*/',
            r'^\s*function\s+',
            r'^\s*var\s+|let\s+|const\s+',
            r'^\s*return\s+',
            r'^\s*console\.',
            r'^\s*print\s*\(',
        ]
        
        code_lines = 0
        for line in lines:
            for pattern in code_patterns:
                if re.match(pattern, line.strip()):
                    code_lines += 1
                    break
        
        # Check for code blocks
        code_blocks = text.count('```')
        
        # Determine content type
        code_ratio = code_lines / total_lines if total_lines > 0 else 0
        
        if code_blocks > 0 or code_ratio > 0.4:
            return "code"
        elif code_ratio > 0.1:
            return "mixed"
        else:
            return "text"

    def _split_code_aware(self, text: str) -> List[str]:
        """Split code while preserving logical structure"""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_tokens = 0
        
        # Track code block context
        in_code_block = False
        block_indent_level = 0
        
        for i, line in enumerate(lines):
            line_tokens = self.estimate_tokens(line)
            line_stripped = line.strip()
            
            # Check if we're in a code block
            if '```' in line:
                in_code_block = not in_code_block
                if in_code_block:
                    block_indent_level = len(line) - len(line.lstrip())
            
            # Calculate effective tokens (code lines might be longer)
            effective_tokens = line_tokens * (1.5 if in_code_block else 1.0)
            
            # Check if adding this line would exceed limit
            if current_tokens + effective_tokens > self.max_tokens and current_chunk:
                # Don't break in the middle of a code block
                if in_code_block and i < len(lines) - 1:
                    # Look ahead to find end of code block
                    end_block_idx = self._find_code_block_end(lines, i)
                    if end_block_idx > i:
                        # Include the entire code block
                        for j in range(i, end_block_idx + 1):
                            current_chunk.append(lines[j])
                            current_tokens += self.estimate_tokens(lines[j])
                        i = end_block_idx
                        in_code_block = False
                    else:
                        # Force break if we can't find end
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                else:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(line)
            current_tokens += effective_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return self._add_overlap(chunks) if len(chunks) > 1 else chunks

    def _find_code_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a code block starting from start_idx"""
        for i in range(start_idx + 1, len(lines)):
            if '```' in lines[i]:
                return i
        return start_idx

    def _split_mixed_content(self, text: str) -> List[str]:
        """Split content that has both code and text"""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.estimate_tokens(line)
            
            # Check if this line is code
            is_code_line = any(re.match(pattern, line.strip()) for pattern in [
                r'^\s*(def|class|import|from|if|for|while|try|except|with|async|await)\s+',
                r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[=:]\s*',
                r'^\s*[{}[\]]\s*$',
            ])
            
            # Code lines get more weight
            effective_tokens = line_tokens * (1.3 if is_code_line else 1.0)
            
            if current_tokens + effective_tokens > self.max_tokens and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(line)
            current_tokens += effective_tokens
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return self._add_overlap(chunks) if len(chunks) > 1 else chunks

    def _split_text_aware(self, text: str) -> List[str]:
        """Split text content by paragraphs and sentences"""
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph fits, add to current chunk
            if self.estimate_tokens(current_chunk + " " + paragraph) <= self.max_tokens:
                current_chunk += " " + paragraph if current_chunk else paragraph
            else:
                # Try to split paragraph by sentences
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split long paragraph by sentences
                sentences = re.split(r'[.!?]+', paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if self.estimate_tokens(current_chunk + " " + sentence) <= self.max_tokens:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return self._add_overlap(chunks) if len(chunks) > 1 else chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks"""
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: add start of next chunk
                if len(chunks) > 1:
                    next_start = " ".join(chunks[1].split()[:self.overlap_tokens])
                    overlapped.append(chunk + " " + next_start)
                else:
                    overlapped.append(chunk)
            elif i == len(chunks) - 1:
                # Last chunk: add end of previous chunk
                prev_end = " ".join(chunks[i-1].split()[-self.overlap_tokens:])
                overlapped.append(prev_end + " " + chunk)
            else:
                # Middle chunk: add both overlaps
                prev_end = " ".join(chunks[i-1].split()[-self.overlap_tokens//2:])
                next_start = " ".join(chunks[i+1].split()[:self.overlap_tokens//2])
                overlapped.append(prev_end + " " + chunk + " " + next_start)
        
        return overlapped
    
    def classify_message(self, content: str, role: str) -> Dict[str, Any]:
        """Simple LLM classification with Groq model fallback from env"""
        prompt = CLASSIFICATION_PROMPT.format(content=content, role=role)

        messages = [{"role": "user", "content": prompt}]
        groq_models = os.getenv("GROQ_MODEL_IDS").split(",")

        llm_client = self.get_llm_client()
        for model in groq_models:
            model = model.strip()
            if not model:
                continue
            try:
                log.debug(f"Trying Groq model '{model}' for content: {content[:100]}...")
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=200
                )
                response_text = response.choices[0].message.content.strip()
                log.debug(f"Groq model '{model}' response: {response_text}")

                # Remove markdown code blocks if present
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                if response_text:
                    classification = json.loads(response_text)
                    result = {
                        'intent': str(classification.get('intent', 'general')).lower().strip(),
                        'topic': str(classification.get('topic', 'general')).lower().strip(),
                        'sentiment': max(-1.0, min(1.0, float(classification.get('sentiment', 0.0))))
                    }
                    log.info(f"Groq model '{model}' classification successful: {result}")
                    return result
            except Exception as e:
                log.warning(f"Groq model '{model}' failed: {e}")

        # If all models fail, return default
        log.warning("All Groq models failed, using default classification")
        return {'intent': 'general', 'topic': 'general', 'sentiment': 0.0}
    
    def process_messages(self):
        """Main processing function"""
        try:
            # Get messages needing processing
            messages = ChatMessages.get_messages_without_embeddings(limit=self.batch_size)
            if not messages:
                return
            
            log.info(f"Processing {len(messages)} messages")
            
            for message in messages:
                try:
                    # Generate embedding
                    embedding = self.get_embedding_function()(message.content)
                    
                    # Update message with embedding
                    ChatMessages.update_message_enrichment(
                        id=message.id, embedding=json.dumps(embedding)
                    )
                    
                    # Create chunks if needed
                    if self.needs_chunking(message.content):
                        self._create_chunks(message.id, message.content)
                    
                    # Classify message
                    classification = self.classify_message(message.content, message.role)
                    ChatMessages.update_message_enrichment(
                        id=message.id,
                        intent=classification['intent'],
                        topic=classification['topic'],
                        sentiment=classification['sentiment']
                    )
                    
                except Exception as e:
                    log.error(f"Error processing message {message.id}: {e}")
                    continue
            
            log.info(f"Completed processing {len(messages)} messages")
            
        except Exception as e:
            log.error(f"Processing error: {e}")
    
    def _create_chunks(self, message_id: str, content: str):
        """Create chunks for a message"""
        try:
            chunks = self.split_text_into_chunks(content)
            if not chunks:
                return
            
            embeddings = self.get_embedding_function()(chunks)
            
            with get_db() as db:
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_id = f"{message_id}_chunk_{i}"
                    embedding_json = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)
                    
                    db.execute(text("""
                        INSERT INTO chat_message_chunk (chunk_id, msg_id, chunk_no, embedding)
                        VALUES (:chunk_id, :msg_id, :chunk_no, :embedding)
                        ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding
                    """), {
                        'chunk_id': chunk_id, 'msg_id': message_id,
                        'chunk_no': i, 'embedding': embedding_json
                    })
                
                db.commit()
            
            log.info(f"Created {len(chunks)} chunks for message {message_id}")
            
        except Exception as e:
            log.error(f"Error creating chunks for {message_id}: {e}")
    
    def _wait_for_database(self):
        """Simple wait loop until database is connected"""
        log.info("Waiting for database connection...")
        max_attempts = 30
        delay = 2
        
        for attempt in range(max_attempts):
            try:
                with get_db() as db:
                    db.execute(text("SELECT 1"))
                    log.info("Database connection established")
                    return
            except Exception as e:
                if attempt < max_attempts - 1:
                    log.info(f"Database not ready (attempt {attempt + 1}/{max_attempts}), waiting {delay}s...")
                    time.sleep(delay)
                else:
                    log.error(f"Database connection failed after {max_attempts} attempts")
                    raise

    def worker_loop(self):
        """Main worker loop"""
        log.info("Chat enrichment worker started")
        
        # Wait for database to be ready
        self._wait_for_database()
        
        while not self.stop_event.is_set():
            try:
                self.process_messages()
                self.stop_event.wait(self.sleep_interval)
            except Exception as e:
                log.error(f"Worker error: {e}")
                time.sleep(10)
        
        log.info("Chat enrichment worker stopped")
    
    def start(self):
        if not self.worker_thread or not self.worker_thread.is_alive():
            self.stop_event.clear()
            self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
            self.worker_thread.start()
    
    def stop(self):
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10)

# Global instance
chat_enrichment_worker = ChatEnrichmentWorker()