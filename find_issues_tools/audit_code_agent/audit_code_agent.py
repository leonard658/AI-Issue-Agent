# audit_code_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from find_issues_tools.audit_tools.semantic_query_vdb_tools import query_documents_tool
from find_issues_tools.audit_tools.id_query_vdb_tool import fetch_documents_tool
from pydantic_types.document_schema import DocumentList

prompt = '''
You are an assistant that retrieves code from a vector database and prepares it for the find_issues_agent. 
Your job is not just to fetch chunks, but to carefully curate only the code that is relevant.

Follow these rules strictly:
1. Filter for Relevance
Only consider code chunks that are directly relevant to the query/task given by the find_issues_agent.
If a code chunk is unrelated or irrelevant, discard it.
You are welcome to alter the top_k in the query_documents_tool input. 
You should return somewhere between 1 and 8 documents back.

2. Handle Incomplete Chunks
If a relevant code chunk appears incomplete, you can find other chunks from the same file to reconstruct the full logic.
Use the id field: all chunks from the same file share a common prefix (e.g., filename.ts-0, filename.ts-1).
You also have access to a tool where you can get specific file chunks by id from the vector database.

3. Respect Token Limits
You may only return up to 5000 tokens of code in total, leaving headroom for LLM reasoning.
If the combined result would exceed the token limit:
Prioritize chunks nearest the relevant chunk’s chunk_index.
Optionally skip low-value chunks (e.g., comments, boilerplate) or summarize them.

4. Iterate and Refine
If the first retrieved set of chunks does not meet these standards, you can continue to query the vector DB for better results.
DO NOT QUERY THE VECTOR DATABASE MORE THAN A TOTAL OF 5 TIMES.
'''

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[query_documents_tool, fetch_documents_tool],
    prompt=prompt,
    response_format=DocumentList
)

class AuditCodeAgentToolInput(BaseModel):
    query: str = Field(description="Query to find relevant code chunks for")
@tool("audit_code_agent_tool", args_schema=AuditCodeAgentToolInput, return_direct=False)
def audit_code_agent_tool(query: str) -> str:
    """
    Retrieve and curate the most relevant code chunks from the codebase for a given issue or query. 
    This agent filters out unrelated code, reconstructs incomplete logic by fetching additional chunks from the same file, and ensures the total code returned stays within a 5000-token limit. 
    This provides concise, high-quality code context to help analyze or resolve the specified issue.
    """
    response = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    }, debug=False)
    return response['structured_response']

#if __name__ == "__main__":
    #example_find_issues_msg = "We are working through an issue related to auth, please find me the relevant code documents."
    #print(audit_code_agent_tool(example_find_issues_msg))


