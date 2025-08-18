import logging
import json
import sqlglot
import time
from typing import Dict, Any, Optional, List
from enum import Enum
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from sqlalchemy import text, inspect
from sqlglot import parse_one, exp
from sqlglot.errors import ParseError
from datetime import date, datetime

from open_webui.retrieval.chat_analytics_retriever import ChatAnalyticsRetriever
from open_webui.internal.db import get_db, engine
from open_webui.services.prompts.chat_analytics_prompts import (
    ENHANCED_SQL_GENERATION_PROMPT,
    FINAL_ANALYSIS_PROMPT,
)
from open_webui.models.admin_query_log import AdminQueryLogs

log = logging.getLogger(__name__)
    
class ChatAnalyticsService:
    
    def __init__(self):
        self.retriever = ChatAnalyticsRetriever()
        self.allowed_tables = ["user", "chat", "daily_report"]
        self.tool_types = ["SQL_QUERY", "VECTOR_SEARCH"]

        self.control_llm = ChatOpenAI(model="gpt-4.1", temperature=0)
        # self.answer_llm = ChatOpenAI(model=model_id, temperature=0.1)

    def _truncate_conversation_context(self, conversation_context: List[Dict[str, Any]], max_chars_per_message: int = 500) -> List[Dict[str, Any]]:
        """
        Truncate conversation context to prevent token limit exceeded errors.
        
        Args:
            conversation_context: List of message dictionaries
            max_chars_per_message: Maximum characters to keep per message content
            
        Returns:
            Truncated conversation context
        """
        truncated_context = []
        total_original_chars = 0
        total_truncated_chars = 0
        
        for i, msg in enumerate(conversation_context):
            content = msg.get('content', '')
            total_original_chars += len(content)
            
            # Truncate content if it's too long
            if len(content) > max_chars_per_message:
                # Try to truncate at a word boundary if possible
                truncated_content = content[:max_chars_per_message]
                if ' ' in truncated_content:
                    last_space = truncated_content.rfind(' ')
                    if last_space > max_chars_per_message * 0.8:  # Only if we don't lose too much
                        truncated_content = truncated_content[:last_space]
                truncated_content += "..."
                log.info(f"Truncated message {i} from {len(content)} to {len(truncated_content)} characters")
            else:
                truncated_content = content
            
            total_truncated_chars += len(truncated_content)
            
            truncated_msg = {
                "idx": i,
                "role": msg.get('role', 'unknown'),
                "text": truncated_content
            }
            truncated_context.append(truncated_msg)
        
        # Log truncation summary
        if total_original_chars != total_truncated_chars:
            reduction_percent = ((total_original_chars - total_truncated_chars) / total_original_chars) * 100
            log.info(f"Conversation context truncated: {total_original_chars} -> {total_truncated_chars} chars ({reduction_percent:.1f}% reduction)")
        
        return truncated_context

    #########################################################################################
    # 1. Determine appropriate tools for the query
    #########################################################################################

    #async def determine_required_tools(self, query: str, conversation_context: List[Dict[str, Any]]) -> Dict[str, Any]:
    #    # Truncate conversation context to prevent token limit exceeded errors
    #    truncated_context = self._truncate_conversation_context(conversation_context)
    #    
    #    # Convert conversation context to JSON string format
    #    context_lines = []
    #    for msg in truncated_context:
    #        context_lines.append(json.dumps(msg)) 
    #    
    #    conversation_context_str = "[\n  " + ",\n  ".join(context_lines) + "\n]"
    #    structured_prompt = ""
#
    #    try:
    #        structured_prompt = ENHANCED_TOOL_SELECTION_PROMPT.format(
    #            question=query,
    #            conversation_context=conversation_context_str
    #        )
    #        
    #        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
    #        estimated_tokens = len(structured_prompt) // 4
    #        log.info(f"Estimated prompt tokens: {estimated_tokens}")
    #        
    #        # If still too long, use more aggressive truncation
    #        if estimated_tokens > 150000:  # Leave some buffer for response
    #            log.warning(f"Prompt still too long ({estimated_tokens} tokens), using aggressive truncation")
    #            truncated_context = self._truncate_conversation_context(conversation_context, max_chars_per_message=200)
    #            
    #            context_lines = []
    #            for msg in truncated_context:
    #                context_lines.append(json.dumps(msg)) 
    #            
    #            conversation_context_str = "[\n  " + ",\n  ".join(context_lines) + "\n]"
    #            structured_prompt = ENHANCED_TOOL_SELECTION_PROMPT.format(
    #                question=query,
    #                conversation_context=conversation_context_str
    #            )
    #            
    #            estimated_tokens = len(structured_prompt) // 4
    #            log.info(f"After aggressive truncation, estimated tokens: {estimated_tokens}")
    #            
    #    except Exception as e:
    #        log.error(f"Error formatting prompt: {e}")
    #        log.error(f"ENHANCED_TOOL_SELECTION_PROMPT: {ENHANCED_TOOL_SELECTION_PROMPT}")
    #        raise
#
    #    try:
    #        response = self.control_llm.invoke(structured_prompt)
    #        content = response.content.strip()
    #        result = json.loads(content)
    #        
    #        return result
    #    except Exception as e:
    #        if "maximum context length" in str(e) or "tokens" in str(e):
    #            log.error(f"Token limit exceeded even after truncation: {e}")
    #            # Fallback: return a simple response without context analysis
    #            return {
    #                "enhanced_query": query,
    #                "tools": ["SQL_QUERY", "VECTOR_SEARCH"],  # Default to both tools
    #                "prev_context": [],
    #                "reasoning": "Fallback due to token limit exceeded"
    #            }
    #        else:
    #            raise
    
    
    #########################################################################################
    # 2.1 Execute a SQL query
    #########################################################################################

    def execute_sql_query(self, conversation_context: str) -> Dict[str, Any]:
        try:
            available_tables = self.get_available_tables()
            schema = self.get_database_schema(available_tables)
            sql_prompt = ENHANCED_SQL_GENERATION_PROMPT.format(
                schema=schema,
                conversation_context=conversation_context
            )
            
            # Get LLM response
            llm_response = self.control_llm.invoke(sql_prompt).content.strip()
            log.info(f"LLM response: {llm_response}")
            
            # Parse JSON response to extract all fields
            try:
                parsed_response = json.loads(llm_response)
                sql_query = parsed_response.get("sql_query", "")
                enhanced_query = parsed_response.get("enhanced_query", "")
                reasoning = parsed_response.get("reasoning", "")
                
                if not sql_query:
                    log.error("No SQL query found in LLM response")
                    return {
                        "success": False,
                        "error": "No SQL query generated by LLM",
                        "llm_response": parsed_response  # Include the full response for debugging
                    }
                
                log.info(f"Enhanced query: {enhanced_query}")
                log.info(f"Reasoning: {reasoning}")
                log.info(f"Generated SQL: {sql_query}")
                
            except json.JSONDecodeError as e:
                log.error(f"Failed to parse LLM response as JSON: {e}")
                log.error(f"Raw response: {llm_response}")
                return {
                    "success": False,
                    "error": f"Invalid JSON response from LLM: {str(e)}",
                    "raw_response": llm_response
                }
            
            # Validate and execute SQL
            validation = self.validate_sql_query(sql_query)
            log.info(f"SQL validation: {validation}")
            
            if not validation.get("valid", False):
                log.error(f"SQL validation failed: {validation.get('error')}")
                return {
                    "success": False,
                    "error": f"SQL validation failed: {validation.get('error')}",
                    "llm_response": parsed_response,  # Include the full LLM response
                    "sql_query": sql_query
                }
            
            with get_db() as db:
                result = db.execute(text(validation["parsed_query"]))
                rows = result.fetchall()
                columns = result.keys()
                
                # ✅ FIX: Convert date objects to strings for JSON serialization
                serializable_rows = []
                for row in rows:
                    serializable_row = []
                    for value in row:
                        if isinstance(value, (date, datetime)):
                            serializable_row.append(value.isoformat())
                        else:
                            serializable_row.append(value)
                    serializable_rows.append(serializable_row)
                
                log.info(f"SQL execution completed. Rows: {len(serializable_rows)}, Columns: {list(columns)}")
                
                # ✅ FIX: Convert back to dictionary format for compatibility
                results = []
                for row in serializable_rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        row_dict[col] = row[i]
                    results.append(row_dict)
                
                return {
                    "success": True,
                    "enhanced_query": parsed_response.get("enhanced_query", ""),
                    "reasoning": parsed_response.get("reasoning", ""),
                    "sql_query": validation["parsed_query"],
                    "results": results  # Use dictionary format
                }
                
        except Exception as e:
            log.error(f"Error executing SQL query: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_available_tables(self) -> List[str]:
        try:
            inspector = inspect(engine)
            all_tables = inspector.get_table_names()
            
            if self.allowed_tables:
                return [table for table in all_tables if table in self.allowed_tables]
            return all_tables
        except Exception as e:
            log.error(f"Error getting available tables: {e}")
            return []
        
    def get_database_schema(self, available_tables: Optional[List[str]] = None) -> str:
        try:
            inspector = inspect(engine)
            schema_info = []
            all_tables = inspector.get_table_names()
            
            # Filter tables if available_tables is specified
            tables_to_process = all_tables
            if available_tables:
                tables_to_process = [table for table in all_tables if table in available_tables]
            
            for table_name in tables_to_process:
                # Get column information directly from database to avoid SQLAlchemy model mismatch
                with get_db() as db:
                    columns_result = db.execute(text(f"""
                        SELECT 
                            column_name,
                            data_type,
                            CASE 
                                WHEN data_type = 'USER-DEFINED' AND udt_name = 'vector' 
                                THEN 'vector(384)'
                                ELSE data_type
                            END as display_type
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}' 
                        AND table_schema = 'public'
                        ORDER BY ordinal_position
                    """))
                    
                    column_info = []
                    for row in columns_result:
                        column_info.append(f"  - {row.column_name}: {row.display_type}")
                
                # Add example data for timestamp columns
                if table_name == "user":
                    column_info.append("  # Example: created_at = 1750745189 (Unix timestamp in seconds)")
                    column_info.append("  # Example: TO_TIMESTAMP(1750745189) = '2025-07-24 12:33:09'")
                
                schema_info.append(f"Table: {table_name}")
                schema_info.extend(column_info)
                schema_info.append("")
            
            return "\n".join(schema_info)
        except Exception as e:
            log.error(f"Error getting database schema: {e}")
            return "Error retrieving schema"

    def validate_sql_query(self, sql_query: str) -> Dict[str, Any]:
        try:
            parsed = parse_one(sql_query, dialect="postgres")
            
            # Security guardrails
            guardrail_checks = self._check_sql_guardrails(parsed)
            if not guardrail_checks["valid"]:
                return {
                    "valid": False,
                    "error": f"SQL guardrail violation: {guardrail_checks['error']}",
                    "parsed_query": parsed
                }
            
            # Apply fixes if needed (like adding LIMIT)
            parsed = self._apply_sql_fixes(sql_query, parsed)
            
            return {
                "valid": True,
                "error": None,
                "parsed_query": parsed
            }
            
        except ParseError as e:
            return {
                "valid": False,
                "error": f"SQL parsing error: {str(e)}",
                "parsed_query": None
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}",
                "parsed_query": None
            }
    
    def _check_sql_guardrails(self, parsed_query) -> Dict[str, Any]:
        """Check SQL query against security guardrails"""
        try:
            # 1. No SELECT * (wildcard selects)
            for select in parsed_query.find_all(exp.Select):
                for col in select.find_all(exp.Column):
                    if col.name == "*":
                        return {
                            "valid": False,
                            "error": "SELECT * is not allowed for security reasons. Please specify exact columns.",
                            "violation": "wildcard_select"
                        }
            
            # 2. No UPDATE/DELETE operations
            if parsed_query.find(exp.Update):
                return {
                    "valid": False,
                    "error": "UPDATE operations are not allowed for security reasons.",
                    "violation": "update_operation"
                }
            
            if parsed_query.find(exp.Delete):
                return {
                    "valid": False,
                    "error": "DELETE operations are not allowed for security reasons.",
                    "violation": "delete_operation"
                }
            
            # 3. Check for LIMIT clause and enforce maximum
            limit_found = False
            max_limit = 1000
            
            for select in parsed_query.find_all(exp.Select):
                if select.args.get("limit"):
                    limit_value = select.args["limit"]
                    if hasattr(limit_value, 'expression'):
                        try:
                            limit_num = int(limit_value.expression)
                            if limit_num > max_limit:
                                return {
                                    "valid": False,
                                    "error": f"LIMIT {limit_num} exceeds maximum allowed limit of {max_limit} rows.",
                                    "violation": "limit_exceeded",
                                    "current_limit": limit_num,
                                    "max_limit": max_limit
                                }
                            limit_found = True
                        except (ValueError, TypeError):
                            # If limit is not a simple number, we'll add a LIMIT clause
                            pass
            
            # 4. Add LIMIT if not present (for SELECT queries)
            if parsed_query.find(exp.Select) and not limit_found:
                # We'll add the limit in the fix_sql_query method
                log.info("No LIMIT clause found, will add LIMIT 1000")
            
            # 5. No DROP, CREATE operations
            dangerous_operations = [exp.Drop, exp.Create]
            for op_class in dangerous_operations:
                if parsed_query.find(op_class):
                    return {
                        "valid": False,
                        "error": f"{op_class.__name__.upper()} operations are not allowed for security reasons.",
                        "violation": "ddl_operation"
                    }
            
            # 6. No INSERT operations (read-only access)
            if parsed_query.find(exp.Insert):
                return {
                    "valid": False,
                    "error": "INSERT operations are not allowed for read-only analytics.",
                    "violation": "insert_operation"
                }
            
            return {
                "valid": True,
                "guardrails_passed": True
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Error checking guardrails: {str(e)}",
                "violation": "guardrail_error"
            }
        
    def _apply_sql_fixes(self, sql_query: str, parsed_query) -> str:
        try:
            fixed_query = sql_query
            
            # Add LIMIT if missing (for SELECT queries)
            if parsed_query.find(exp.Select) and not parsed_query.find(exp.Limit):
                if fixed_query.strip().endswith(';'):
                    fixed_query = fixed_query.rstrip(';') + " LIMIT 1000;"
                else:
                    fixed_query = fixed_query + " LIMIT 1000"
                # log.info(f"Added LIMIT clause: {fixed_query}")
            
            # Common timestamp column fixes for your schema
            timestamp_fixes = {
                "created_at": "created_at",
                "updated_at": "updated_at", 
                "last_active_at": "last_active_at",
            }
            
            # Fix timestamp operations
            for old_col, new_col in timestamp_fixes.items():
                # Fix DATE() wrapper on timestamp columns
                fixed_query = fixed_query.replace(f"DATE({old_col})", old_col)
                fixed_query = fixed_query.replace(f"DATE({new_col})", new_col)
                
                # Fix timestamp conversion for epoch timestamps
                if f"TO_TIMESTAMP({old_col})" not in fixed_query and old_col in fixed_query:
                    # Add TO_TIMESTAMP for epoch timestamps
                    fixed_query = fixed_query.replace(f"WHERE {old_col}", f"WHERE TO_TIMESTAMP({old_col})")
                    fixed_query = fixed_query.replace(f"AND {old_col}", f"AND TO_TIMESTAMP({old_col})")
                    fixed_query = fixed_query.replace(f"OR {old_col}", f"OR TO_TIMESTAMP({old_col})")
            
            # Fix common table name issues
            table_fixes = {
                "users": "user",
                "messages": "chat_message",
                "chats": "chat",
            }
            
            for old_table, new_table in table_fixes.items():
                fixed_query = fixed_query.replace(f"FROM {old_table}", f"FROM {new_table}")
                fixed_query = fixed_query.replace(f"JOIN {old_table}", f"JOIN {new_table}")
            
            return fixed_query
            
        except Exception as e:
            log.error(f"Error applying SQL fixes: {e}")
            return sql_query
        
        
    #########################################################################################
    # 2.2 Execute a vector search
    #########################################################################################
            
    def perform_vector_search(self, query: str) -> Dict[str, Any]:
        try:
            # Get relevant documents
            documents = self.retriever._get_relevant_documents(query, run_manager=None)
            
            # Extract content and metadata
            sources = []
            for doc in documents:
                sources.append({
                    "content": doc.page_content,
                    "role": doc.metadata.get("role"),
                    "user_id": doc.metadata.get("user_id"),
                    "created_at": doc.metadata.get("created_at"),
                    "similarity": doc.metadata.get("similarity")
                })
            
            return {
                "success": True,
                "sources": sources,
                "count": len(sources)
            }
            
        except Exception as e:
            log.error(f"Error in vector search: {e}")
            return {
                "success": False,
                "error": str(e),
                "sources": []
            }
    

    #########################################################################################
    # Main pipeline
    #########################################################################################

    async def analyze_chat_data(self, conversation_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_time = time.time()

        try:
            query = ""
            context_lines = []

            last_message = conversation_context[-1]
            if last_message.get("role") == "user":
                query = last_message.get("content", "")
            
            log.info(f"Analyzing chat data for query: '{query}' from {len(conversation_context)} messages")

            # Step 1: Prepare conversation context for SQL generation
            context_prep_start = time.time()
            truncated_context = self._truncate_conversation_context(conversation_context)
            
            # Convert conversation context to a simple string format for SQL generation
            context_lines = []
            for i, msg in enumerate(truncated_context):
                role = msg.get('role', 'unknown')
                content = msg.get('text', '')[:200]  # Limit content length
                context_lines.append(f"{i}. {role}: {content}")
            
            conversation_context_str = "\n".join(context_lines)
            context_prep_time = time.time() - context_prep_start

            # Step 2: Execute SQL query directly
            pipeline_start = time.time()
            sql_results = self.execute_sql_query(conversation_context_str)
            pipeline_time = time.time() - pipeline_start
            log.info(f"SQL pipeline execution took {pipeline_time:.3f}s")
            
            # Step 3: Extract data for frontend consumption
            sources = []
            message = None
            
            # Check for SQL success
            sql_success = sql_results is not None and sql_results.get("success") if sql_results else False
            
            # Extract enhanced query from SQL results
            enhanced_query = query  # Default to original query
            if sql_success and sql_results.get("llm_response"):
                enhanced_query = sql_results["llm_response"].get("enhanced_query", query)
            
            if not sql_success:
                message = "No relevant data was found for this query"
            
            total_time = time.time() - start_time
            log.info(f"Total analysis completed in {total_time:.3f}s")
            
            return {
                "success": True,
                "enriched_context": {
                    "sql_results": sql_results,
                    "vector_results": None,  # Always None in SQL-only mode
                },
                "query": query,
                "enhanced_query": enhanced_query,
                "tool_used": ["SQL_QUERY"],  # Always SQL only
                "sources": sources,
                "message": message,
                "timing": {
                    "total_time": round(total_time, 3),
                    "context_preparation": round(context_prep_time, 3),
                    "pipeline_execution": round(pipeline_time, 3),
                }
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            log.error(f"Error in chat analytics after {total_time:.3f}s: {e}")
            return {
                "success": False,
                "message": f"Error analyzing chat data: {str(e)}",
                "query": query if 'query' in locals() else "",
                "enhanced_query": enhanced_query if 'enhanced_query' in locals() else query if 'query' in locals() else "",
                "tool_used": ["SQL_QUERY"], 
                "sources": [],
                "sql_results": None,
                "timing": {
                    "total_time": round(total_time, 3),
                    "error_occurred_at": round(total_time, 3)
                }
            }

    def _execute_pipeline(self, query: str, required_tools: List[str]) -> Dict[str, Any]:
        """Execute the pipeline with the required data retrieval tools"""
        results = {
            "sql_results": None,
            "vector_results": None
        }
        
        for tool in required_tools:
            try:
                # 2.1 Execute a SQL query
                if tool == "SQL_QUERY":
                    sql_results = self.execute_sql_query(query)
                    results["sql_results"] = sql_results
                    log.info(f"SQL_QUERY pipeline completed: {sql_results.get('success', False)}")
                
                # 2.1 Execute a SQL query
                elif tool == "VECTOR_SEARCH":
                    vector_results = self.perform_vector_search(query)
                    results["vector_results"] = vector_results
                    log.info(f"VECTOR_SEARCH pipeline completed: {vector_results.get('success', False)}")
                    
            except Exception as e:
                log.error(f"Error in {tool} pipeline: {e}")
        
        return results
    
    def _generate_final_analysis(self, query: str, pipeline_results: Dict[str, Any]) -> str:
        """Generate final analysis using all collected data and the answer LLM"""
        try:
            # Prepare context for the answer LLM
            context_parts = []
            
            # Add SQL results if available
            if pipeline_results.get("sql_results") and pipeline_results["sql_results"].get("success"):
                sql_data = pipeline_results["sql_results"]
                sql_context = f"Database Query Results:\n"
                sql_context += f"SQL: {sql_data.get('sql_query', 'N/A')}\n"
                
                # Convert datetime objects to strings before JSON serialization
                results = sql_data.get('results', [])
                serializable_results = []
                for row in results:
                    serializable_row = {}
                    for key, value in row.items():
                        if hasattr(value, 'isoformat'):  # datetime objects
                            serializable_row[key] = value.isoformat()
                        else:
                            serializable_row[key] = value
                    serializable_results.append(serializable_row)
                
                sql_context += f"Results: {json.dumps(serializable_results, indent=2)}\n"
                context_parts.append(sql_context)
            
            # Add vector search results if available
            if pipeline_results.get("vector_results") and pipeline_results["vector_results"].get("success"):
                vector_data = pipeline_results["vector_results"]
                if vector_data.get("sources"):
                    vector_context = "Semantic Search Results:\n"
                    for i, source in enumerate(vector_data["sources"][:3], 1):
                        vector_context += f"{i}. {source.get('content', '')[:300]}...\n"
                    context_parts.append(vector_context)
            
            synthesis_prompt = FINAL_ANALYSIS_PROMPT.format(
                question=query,
                context_parts=chr(10).join(context_parts)
            )
            
            response = self.answer_llm.invoke(synthesis_prompt)
            return response.content
                
        except Exception as e:
            log.error(f"Error generating final analysis: {e}")
            return f"Error generating analysis: {str(e)}" 

    def get_prev_tool_results(self, prev_context_indices: list, conversation_context: list, chat_id: str) -> list:
        results = []
        for idx in prev_context_indices:
            if 0 <= idx < len(conversation_context):
                msg = conversation_context[idx]
                # Try to get the assistant message id (or user message id if needed)
                ai_msg_id = msg.get("id") or msg.get("ai_msg_id")
                if ai_msg_id:
                    log_entry = AdminQueryLogs.get_query_log_by_id(ai_msg_id)
                    if log_entry:
                        results.append(log_entry)
                    else:
                        # Optionally, try to fetch by chat_id and message content if id is missing
                        # Or just append None to keep the order
                        results.append(None)
                else:
                    results.append(None)
            else:
                results.append(None)
        return results 

#def build_conversation_context(conversation_context: List[Dict[str, Any]]) -> str:
#    """Build conversation context without truncation."""
#    context_lines = []
#    
#    for i, msg in enumerate(conversation_context):
#        # Don't truncate, or use a much higher limit
#        content = msg['content'][:1000] if len(msg['content']) > 1000 else msg['content']
#        context_lines.append(f"{i}. {msg['role']}: {content}")
#    
#    return "\n".join(context_lines) 