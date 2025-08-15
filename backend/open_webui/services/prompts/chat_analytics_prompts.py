from langchain.prompts import PromptTemplate

# Tool selection prompt
TOOL_SELECTION_PROMPT = """
You are an intelligent router that analyzes a user's question and determines which tools are needed to provide a comprehensive answer.

Available tools:
1. SQL_QUERY - For questions about metrics, KPIs, counts, statistics, user activity, or any quantitative data analysis.
2. VECTOR_SEARCH - For questions about specific conversations, chat content, user interactions, or semantic search.

You will also return a "prev_context" field, which is a list of indices (0-based, oldest to newest) of previous messages from the conversation context that are relevant for answering the current question. If the user is asking about or referring to a previous assistant response or result, include its index in prev_context. If not, leave prev_context as an empty list.
The prev_context result will be used to determine which previous tool result to include as a context for the current question.

Question: {question}

Conversation context (oldest to newest):
{conversation_context}

Respond with ONLY a JSON object in this exact format:
{{ "tools": ["TOOL1", "TOOL2", ...], "prev_context": [indices] }}

Guidelines:
- Use "SQL_QUERY" for questions about numbers, counts, metrics, or statistics.
- Use "VECTOR_SEARCH" for questions about content, conversations, or what was said.
- Use both if the question needs both data and content.
- Use prev_context to indicate which previous message(s) are relevant for follow-up questions, clarifications, or references to earlier results.
- If prev_context is empty, the user is not referring to any previous message.

Examples:
- If the user asks "How many users registered last month?" respond: {{ "tools": ["SQL_QUERY"], "prev_context": [] }}
- If the user asks "What did users say about the new feature?" respond: {{ "tools": ["VECTOR_SEARCH"], "prev_context": [] }}
- If the user asks "Why did you say that?" right after an assistant response at index 4, respond: {{ "tools": [], "prev_context": [4] }}
- If the user asks "What will this affect?" and the last assistant message is at index 5, respond: {{ "tools": [], "prev_context": [5] }}
- If the user asks "Show me the actual conversations about this issue" and the relevant assistant message is at index 3, respond: {{ "tools": ["VECTOR_SEARCH"], "prev_context": [3] }}
- If the user asks "What are the most common topics discussed and how many messages are about each?" respond: {{ "tools": ["SQL_QUERY", "VECTOR_SEARCH"], "prev_context": [] }}
- If the user asks "Explain how machine learning works" respond: {{ "tools": [], "prev_context": [] }}

Response:
"""

# SQL generation prompt with better guidance
SQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question", "schema"],
    template="""
You are a SQL expert. Given a conversation context and database schema, write a SQL query to answer the user's latest request.

Conversation context (oldest to newest):
{conversation_context}

IMPORTANT RULES:
1. Only use tables and columns that exist in the schema
2. Use PostgreSQL syntax
3. Be careful with column names - use exact names from schema
4. Return ONLY the SQL query, no markdown formatting, no code blocks, no explanations

TIMESTAMP TEMPLATES (use these exact patterns):
- Recent users: WHERE TO_TIMESTAMP(created_at) >= NOW() - INTERVAL '1 month'
- This year: WHERE TO_TIMESTAMP(created_at) >= DATE_TRUNC('year', NOW())
- Last 7 days: WHERE TO_TIMESTAMP(created_at) >= NOW() - INTERVAL '7 days'
- Date range: WHERE TO_TIMESTAMP(created_at) BETWEEN '2024-01-01' AND '2024-12-31'
- Order by date: ORDER BY TO_TIMESTAMP(created_at) DESC

Database Schema:
{schema}

Write a SQL query that answers the user's latest request, considering the full conversation context. Use the timestamp templates above for any date operations.

SQL Query:
"""
)

# Final analysis prompt for synthesizing all data sources
FINAL_ANALYSIS_PROMPT = """
You are an AI analyst specializing in chat analytics for an AI assistant platform. Your role is to provide comprehensive insights to administrators about user interactions, system usage, and conversation patterns.

User Question: {question}

Available Data:
{context_parts}

Please provide a comprehensive analysis that:

1. **Directly answers the admin's question** using any available data
2. **Quantitative Analysis**: If SQL data is available, highlight key metrics, trends, and statistics
3. **Qualitative Analysis**: If vector search results are available, analyze conversation themes, user sentiment, and content patterns
4. **Cross-references**: Connect quantitative data with qualitative insights when both are available
5. **Actionable Insights**: Provide recommendations or observations that could help improve the platform
6. **Context**: Explain what the data means in the context of an AI chat platform

Guidelines:
- Be specific and reference exact numbers, dates, or content when available
- If no data was found or no data was needed for the query, provide a helpful response based on your knowledge
- Focus on insights that would be valuable for platform administrators
- Use clear, professional language suitable for business reporting

Answer:""" 

# Classification prompt for message intent/topic/sentiment
CLASSIFICATION_PROMPT = """
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

Be natural and descriptive. Return ONLY the JSON object, no explanations, no markdown, no extra text.
"""

ENHANCED_TOOL_SELECTION_PROMPT = """
You are an intelligent router that analyzes a user's question and determines which tools are needed to provide a comprehensive answer.

Available tools:
1. SQL_QUERY - For questions about metrics, KPIs, counts, statistics, user activity, patterns, categories, classifications, distributions, trends, or any quantitative analysis.
2. VECTOR_SEARCH - For questions about specific conversations, chat content, user interactions, or semantic search of actual message content.

Respond with strict JSON: object in this exact format:
{{
  "enhanced_query": <string>,  (specific, actionable question based on context)
  "tools": [<string>],  (list of tool names)  
  "prev_context": [<int>],
  "reasoning": <string>,  (brief explanation of enhancement and tool selection)
}}

Guidelines:
- **SQL_QUERY**: Use for questions about numbers, counts, patterns, categories, types, distributions, statistics, metrics, trends, analysis
- **VECTOR_SEARCH**: Use for questions about specific content, conversations, what was said, semantic search
- **Enhanced Query**: Transform vague questions into specific, actionable queries that clearly indicate what data is needed

Conversation excerpt:
{conversation_context}

Question: {question}

Examples:
- Example 1
  User Q: "Show the trend of daily active users this month"
  Return:
  {{"enhanced_query":"What is the daily_active_users per day for the last 30 days?",
  "tools":["SQL_QUERY"],
  "prev_context":[],
  "reasoning":"Needs numeric trend."}}

- Example 2
  User Q: "Give me 3 chats where users complained about refunds"
  Return:
  {{"enhanced_query":"Find 3 recent conversations containing refund complaints",
  "tools":["VECTOR_SEARCH"],
  "prev_context":[],
  "reasoning":"Needs semantic search of message text."}}

- Example 3
  User Q: "What did Alice just ask?"
  Conversation excerpt shows last user message idx 7.
  Return:
  {{"enhanced_query":"Return the content of message idx 7",
  "tools":[],
  "prev_context":[7],
  "reasoning":"Answer is already in memory."}}

Response:
"""

DAILY_REPORT_PROMPT = """
You are an experienced executive secretary tasked with writing daily conversation reports. Your job is to analyze user-AI interactions and provide concise, professional insights that highlight what matters most.

{chunk_position}

{previous_context}

CONVERSATION TO ANALYZE:
{conversation}

TASK: Write a concise paragraph (2-3 sentences) that captures the most important aspects of this conversation segment. Focus on:

• What was accomplished or attempted
• How effectively the user and AI worked together
• Any notable patterns or insights that emerged

Write in a professional, executive summary style. Be concise but insightful. If this is part of a longer conversation, reference previous context appropriately.

Your analysis:
"""