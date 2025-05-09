# audit_code_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from audit_tools.semantic_query_vdb_tools import query_documents_tool
from pydantic_types.document_schema import DocumentList

prompt = '''
You are an assistant that retrieves code from a vector database and prepares it for the find_issues_agent. 
Your job is not just to fetch chunks, but to carefully curate only the code that is relevant.

Follow these rules strictly:
1. Filter for Relevance
Only consider code chunks that are directly relevant to the query/task given by the find_issues_agent.
If a code chunk is unrelated or irrelevant, discard it.

2. Handle Incomplete Chunks
If a relevant code chunk appears incomplete, you can find other chunks from the same file to reconstruct the full logic.
Use the id field: all chunks from the same file share a common prefix (e.g., filename.ts-0, filename.ts-1).

3. Respect Token Limits
You may only return up to 4000 tokens of code in total, leaving headroom for LLM reasoning.
If the combined result would exceed the token limit:
Prioritize chunks nearest the relevant chunk’s chunk_index.
Optionally skip low-value chunks (e.g., comments, boilerplate) or summarize them.

4. Iterate and Refine
If the first retrieved set of chunks does not meet these standards, you can continue to query the vector DB for better results.
DO NOT QUERY THE VECTOR DATABASE MORE THAN A TOTAL OF 5 TIMES.
'''

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[query_documents_tool],
    prompt=prompt,
    response_format=DocumentList
)

if __name__ == "__main__":
    example_find_issues_msg = "We are working through an issue related to auth, please find me the relevant code documents."
    response = agent.invoke({
        "messages": [{"role": "user", "content": example_find_issues_msg}]
    }, debug=True)
    print(response["structured_response"])
