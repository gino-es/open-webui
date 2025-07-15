from langchain.prompts import PromptTemplate

# Tool selection prompt
TOOL_SELECTION_PROMPT = """
You are an intelligent router that analyzes a user's question and determines which tools are needed to provide a comprehensive answer.

Available tools:
1. SQL_QUERY - For questions about metrics, KPIs, counts, statistics, user activity, or any quantitative data analysis
2. VECTOR_SEARCH - For questions about specific conversations, chat content, user interactions, or semantic search

Question: {question}

Analyze the question and respond with ONLY a JSON object in this exact format:
{{ "tools": ["TOOL1", "TOOL2", ...] }}

Where tools is an array of required tools. Use:
- ["SQL_QUERY"] - for questions about numbers, counts, metrics
- ["VECTOR_SEARCH"] - for questions about content, conversations, what was said
- ["SQL_QUERY", "VECTOR_SEARCH"] - for questions that need both data and content
- [] - for general questions that don't need specific data or don't need any tools

Examples:
- "How many users registered last month?" → ["SQL"]
- "What did users say about the new feature?" → ["VECTOR"] 
- "What are the most common topics discussed and how many messages are about each?" → ["SQL", "VECTOR"]
- "Explain how machine learning works" → []

Response:"""

# SQL generation prompt with better guidance
SQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question", "schema"],
    template="""
You are a SQL expert. Given a question and database schema, write a SQL query to answer it.

IMPORTANT RULES:
1. Only use tables and columns that exist in the schema
2. Use PostgreSQL syntax
3. For date/time operations, use PostgreSQL functions
4. Be careful with column names - use exact names from schema
5. If a column doesn't exist, don't use it
6. Return ONLY the SQL query, no markdown formatting, no code blocks, no explanations

Database Schema:
{schema}

Question: {question}

Write a SQL query that answers this question. Return ONLY the SQL query, nothing else.

SQL Query: """
)

# RAG response prompt
RAG_RESPONSE_PROMPT = PromptTemplate(
    input_variables=["question", "context", "sql_results"],
    template="""
You are an AI assistant that provides comprehensive answers based on multiple sources of information.

Question: {question}

Context from vector search (if any):
{context}

SQL query results (if any):
{sql_results}

Provide a comprehensive answer that combines all available information. Be specific and detailed.

Answer: """
)

# Combined analysis prompt
COMBINED_ANALYSIS_PROMPT = """
You are an AI analyst that provides comprehensive insights by combining quantitative data and qualitative content analysis.

Question: {question}

Quantitative Data (SQL Results):
{sql_results_str}

Qualitative Content (Vector Search Results):
{context}

Provide a comprehensive analysis that:
1. Summarizes the quantitative findings
2. Analyzes the qualitative content and themes
3. Connects the data with the content insights
4. Provides actionable insights and conclusions

Make sure to reference specific numbers from the SQL results and specific themes from the content analysis.

Analysis: """ 