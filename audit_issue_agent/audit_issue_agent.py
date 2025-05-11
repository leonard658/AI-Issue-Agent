# audit_issues_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from audit_tools.semantic_query_vdb_tools import query_issues_tool
from audit_tools.id_query_vdb_tool import fetch_issues_tool

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

class IssueSummary(BaseModel):
    issue_summary: str  = Field(..., description="Summary of the issues that have been looked into")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=.8)

agent = create_react_agent(
    model=llm,
    tools=[query_issues_tool, fetch_issues_tool],
    prompt=prompt,
    response_format=IssueSummary
)

if __name__ == "__main__":
    example_find_issues_msg = "We are working through an issue related to azure."
    response = agent.invoke({
        "messages": [{"role": "user", "content": example_find_issues_msg}]
    }, debug=True)
    print(response["structured_response"])
