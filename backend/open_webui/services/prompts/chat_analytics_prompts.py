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

Question: {question}

Write a SQL query that answers this question. Use the timestamp templates above for any date operations.

SQL Query: """
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