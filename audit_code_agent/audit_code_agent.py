# audit_code_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.schema import Document
from typing import List, Dict
from pydantic import BaseModel, Field
from audit_tools.query_pinecone_tools import query_documents_tool
from audit_tools.combine_chunks import combine_chunks_tool

prompt = '''
You are an assistant that retrieves code from a vector database and prepares it for the find_issues_agent. 
Your job is not just to fetch chunks, but to carefully curate and reconstruct only the code that is relevant and complete.

Follow these rules strictly:
1. Filter for Relevance
Only consider code chunks that are directly relevant to the query/task given by the find_issues_agent.
If a code chunk is unrelated or irrelevant, discard it and continue searching.

2. Handle Incomplete Chunks
If a relevant code chunk appears incomplete, you can find other chunks from the same file to reconstruct the full logic.
Use the id field: all chunks from the same file share a common prefix (e.g., filename.ts-0, filename.ts-1).
Sort these by their chunk_index to reassemble the full block.

3. Reconstruct Logically Complete Code
When combining chunks, ensure that the final result represents a complete unit of code (such as a full function, class, or component).
Only report code to find_issues_agent when it is both relevant and logically complete.

4. Respect Token Limits
You may only return up to 3000 tokens of code in total, leaving headroom for LLM reasoning.
Use a tokenizer to estimate token count before combining chunks.
If the combined result would exceed the token limit:
Prioritize chunks nearest the relevant chunk’s chunk_index.
Optionally skip low-value chunks (e.g., comments, boilerplate) or summarize them.
Stop once the reconstructed block is coherent and under budget.

5. Iterate and Refine
If the first retrieved set of chunks does not meet these standards, you can continue to query the vector DB for better results.    
    
 6. Once you have the relevant code, return it as a list of Documents with the following fields. :
    - .metadata: all your stored metadata
        - .metadata.file_path: the path to the file in the original repo
        - .metadata.language: the language of the chunk (e.g. "python")
    - .page_content: the text of the chunk
'''

llm = ChatOpenAI(model="gpt-4o-mini", temperature=.3)

class DocumentSchema(BaseModel):
    page_content: str = Field(description="The text of the code chunk.")
    #metadata: Dict[str, str] = Field(description="Metadata for the document such as file_path and language.")

class DocumentList(BaseModel):
    documents: List[DocumentSchema] = Field(description="List of documents containing code chunks.")

agent = create_react_agent(
    model=llm,
    tools=[query_documents_tool, combine_chunks_tool],
    prompt=prompt,
    response_format=(DocumentSchema)
)

if __name__ == "__main__":
    example_find_issues_msg = "We are working through an issue related to auth, please find me the relevant code documents."
    response = agent.invoke({
        "messages": [{"role": "user", "content": example_find_issues_msg}]
    }, debug=True)
    print(response["structured_response"])
