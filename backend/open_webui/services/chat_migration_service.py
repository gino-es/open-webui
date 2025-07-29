
import logging
from typing import List, Dict, Any
from sqlalchemy import text

from open_webui.internal.db import get_db
from open_webui.models.chats import Chats, ChatModel
from open_webui.models.chat_message import ChatMessages, ChatMessageForm

log = logging.getLogger(__name__)

class ChatMigrationService:
    
    def __init__(self):
        self.batch_size = 50
    
    def check_migration_status(self) -> Dict[str, Any]:
        with get_db() as db:
            chat_count = db.execute(text("SELECT COUNT(*) FROM chat")).scalar()
            chat_message_count = db.execute(text("SELECT COUNT(*) FROM chat_message")).scalar()
            
            processed_chats = db.execute(text("""
                SELECT COUNT(DISTINCT chat_id) FROM chat_message
            """)).scalar()
            
            chats_with_messages = db.execute(text("""
                SELECT COUNT(DISTINCT c.id) 
                FROM chat c 
                WHERE c.chat IS NOT NULL 
                AND c.chat::text != '{}' 
                AND c.chat::text != 'null'
                AND c.chat::text != '""'
            """)).scalar()
            
            completion_percentage = 0
            if chats_with_messages > 0:
                completion_percentage = (processed_chats / chats_with_messages) * 100
            
            return {
                "chat_table_count": chat_count,
                "chat_message_table_count": chat_message_count,
                "processed_chats": processed_chats,
                "chats_with_messages": chats_with_messages,
                "completion_percentage": round(completion_percentage, 2),
                "migration_needed": chat_count > 0 and chat_message_count == 0,
                "partially_migrated": chat_count > 0 and chat_message_count > 0 and processed_chats < chats_with_messages,
                "migration_complete": processed_chats >= chats_with_messages and chats_with_messages > 0
            }
    
    def _is_analytics_chat(self, chat: ChatModel) -> bool:
        """Check if a chat contains analytics content"""
        if not chat.chat or not isinstance(chat.chat, dict):
            return False
        
        history = chat.chat.get("history", {})
        messages = history.get("messages", {})
        
        for message_data in messages.values():
            if message_data.get("role") == "assistant":
                content = message_data.get("content", "")
                analytics_indicators = [
                    '<summary>Analytics Thought</summary>',
                    '**Enhanced Query:**',
                    '**Tools Used:**',
                    '**SQL Query Results:**',
                    '**Vector Search Results:**',
                    'analytics_reasoning_block'
                ]
                
                content_lower = content.lower()
                for indicator in analytics_indicators:
                    if indicator.lower() in content_lower:
                        return True
        return False
    
    def _extract_messages_from_chat(self, chat: ChatModel) -> List[Dict[str, Any]]:
        """Extract individual messages from a chat's JSON data"""
        messages = []
        
        if not chat.chat or not isinstance(chat.chat, dict):
            return messages
        
        history = chat.chat.get("history", {})
        chat_messages = history.get("messages", {})
        
        if not chat_messages:
            return messages
        
        # Sort messages by timestamp or order them sequentially
        sorted_messages = []
        for message_id, message_data in chat_messages.items():
            if message_data.get("content") and message_data.get("role") in ["user", "assistant"]:
                sorted_messages.append((message_id, message_data))
        
        # Sort by timestamp if available, otherwise keep original order
        try:
            sorted_messages.sort(key=lambda x: x[1].get("timestamp", 0))
        except (TypeError, KeyError):
            pass
        
        # Convert to list of message dictionaries
        for turn_number, (message_id, message_data) in enumerate(sorted_messages):
            messages.append({
                "message_id": message_id,
                "role": message_data.get("role", "user"),
                "content": message_data["content"],
                "turn_number": turn_number,
                "timestamp": message_data.get("timestamp", chat.created_at)
            })
        
        return messages
    
    def _migrate_single_chat(self, chat: ChatModel) -> int:
        """Migrate a single chat to chat_message records"""
        # Skip analytics chats
        if self._is_analytics_chat(chat):
            log.debug(f"Skipping analytics chat {chat.id}")
            return 0
        
        messages = self._extract_messages_from_chat(chat)
        migrated_count = 0
        
        for message_data in messages:
            try:
                # Check if message already exists
                existing_messages = ChatMessages.get_chat_messages_by_chat_id(chat.id)
                message_exists = any(
                    msg.message_id == message_data["message_id"] 
                    for msg in existing_messages
                )
                
                if message_exists:
                    continue
                
                # Create chat message record
                form_data = ChatMessageForm(
                    chat_id=chat.id,
                    user_id=chat.user_id,
                    role=message_data["role"],
                    turn_number=message_data["turn_number"],
                    content=message_data["content"],
                    message_id=message_data["message_id"]
                )
                
                result = ChatMessages.insert_new_chat_message(form_data)
                if result:
                    migrated_count += 1
                
            except Exception as e:
                log.error(f"Error migrating message {message_data['message_id']} from chat {chat.id}: {e}")
        
        return migrated_count
    
    def migrate_chats(self) -> Dict[str, Any]:
        """Main migration pipeline - migrates all chats in batches, skipping analytics content"""
        log.info("Starting chat to chat_message migration")
        
        # Get remaining chats that need migration
        with get_db() as db:
            processed_chat_ids = db.execute(text("""
                SELECT DISTINCT chat_id FROM chat_message
            """)).fetchall()
            processed_ids = {row[0] for row in processed_chat_ids}
        
        all_chats = Chats.get_chats()
        remaining_chats = []
        
        for chat in all_chats:
            if chat.id not in processed_ids and self._has_messages(chat):
                remaining_chats.append(chat)
        
        total_chats = len(remaining_chats)
        if total_chats == 0:
            log.info("No chats remaining to migrate")
            return {
                "total_chats": 0,
                "total_messages_migrated": 0,
                "skipped_analytics_chats": 0,
                "errors": 0,
                "success": True,
                "migration_complete": True
            }
        
        log.info(f"Found {total_chats} chats remaining to migrate")
        
        # Process in batches
        total_migrated = 0
        total_errors = 0
        skipped_analytics = 0
        
        for i in range(0, total_chats, self.batch_size):
            batch_chats = remaining_chats[i:i + self.batch_size]
            
            for chat in batch_chats:
                try:
                    if self._is_analytics_chat(chat):
                        skipped_analytics += 1
                        continue
                    
                    migrated = self._migrate_single_chat(chat)
                    total_migrated += migrated
                    
                except Exception as e:
                    log.error(f"Error migrating chat {chat.id}: {e}")
                    total_errors += 1
            
            # Progress logging
            processed = min(i + self.batch_size, total_chats)
            log.info(f"Progress: {processed}/{total_chats} chats processed, {total_migrated} messages migrated, {skipped_analytics} analytics chats skipped")
        
        # Check final status
        status = self.check_migration_status()
        
        result = {
            "total_chats": total_chats,
            "total_messages_migrated": total_migrated,
            "skipped_analytics_chats": skipped_analytics,
            "errors": total_errors,
            "success": total_errors == 0,
            "migration_complete": status["migration_complete"]
        }
        
        if status["migration_complete"]:
            log.info("🎉 Migration completed successfully!")
        else:
            log.info(f"Migration progress: {status['completion_percentage']}% complete")
        
        return result
    
    def _has_messages(self, chat: ChatModel) -> bool:
        """Check if a chat has messages to migrate"""
        if not chat.chat or not isinstance(chat.chat, dict):
            return False
        
        history = chat.chat.get("history", {})
        messages = history.get("messages", {})
        
        for message_data in messages.values():
            if message_data.get("content"):
                return True
        
        return False

# Global instance
chat_migration_service = ChatMigrationService() 