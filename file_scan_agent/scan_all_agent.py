# file_scan_agent.py
import json
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from file_scan_agent.tools.fetch_chunks_by_prefix import fetch_chunks_by_prefix
from file_scan_agent.tools.pull_all_index_ids import pull_all_index_ids, pull_all_index_prefixes
from file_scan_agent.tools.fetch_chunk import fetch_next_chunk_tool, fetch_first_chunk
import os
from dotenv import load_dotenv
from find_issues_agent.broad_find_issues_agent import broad_find_issues_agent
from pydantic_types.document_schema import DocumentList
from pydantic_types.to_json_str import to_json_str

load_dotenv()

class ShouldScan(BaseModel):
    should_scan: bool = Field(description="Boolean value indicating if the file should be scanned (True - should be scanned, False - should not be scanned)")

class FileScanState(BaseModel):
    file_prefixes_to_explore: list[str]
    files_current_prefix: DocumentList
    scanned_prefixes: list[str]
    scanned_summaries: list[str]
    
prompt = '''
1. Input: You’ll receive the first chunk of a file.
2. Goal: Decide whether this file needs an issue scan.
3. Skip files that clearly do not affect system behavior (unless specifically asked for by the user), such as:
   Dependency‐list or lock files (e.g. `package.json`, `requirements.txt`, `Pipfile.lock`).
   Files containing only comments, headers, or documentation.
   Pure configuration or metadata files (e.g. `.gitignore`, `README.md`).
4. Uncertain?
   Use the available tool to fetch the next chunk and re-evaluate.
5. Output: Return `True` if the file should be scanned; otherwise `False`.

Make sure that if you call the get next chunk tool, that the current file has more than 1 total_chunks
'''

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[fetch_next_chunk_tool],
    prompt=prompt,
    response_format=ShouldScan
)

def scan_specific_files():
    """
    Scan all files in the index
    """
    docs_index = os.getenv("DOCUMENTS_VDB_INDEX")
    # 1) Get starting state:
    state = FileScanState(
        file_prefixes_to_explore = pull_all_index_prefixes(docs_index),
        files_current_prefix = DocumentList(documents=[]),
        scanned_prefixes =  [],
        scanned_summaries = []
    )
    #print(state)
    
    # 2) Loop through all prefixes
    while len(state.file_prefixes_to_explore) > 0:
        # Grab the next prefix
        cur_prefix = state.file_prefixes_to_explore.pop()
        first_chunk = fetch_first_chunk(cur_prefix)
        if first_chunk is None:
            print(f"Error: No first chunk found for prefix {cur_prefix}")
            continue
    
        chunk_str = to_json_str(first_chunk)
        # build a single content string that includes instruction + metadata
        message_content = f"""
        Here is the first chunk of a file and its info:
        {chunk_str}
        """

        # Check if the file should be scanned
        response = agent.invoke({
            "messages": [{"role": "user", "content": message_content}]
        }, debug=False)

        # If it shouldnt be scanned, move on
        if not response['structured_response'].should_scan:
            continue
        
        chunks = fetch_chunks_by_prefix(docs_index,cur_prefix)
        state.files_current_prefix.documents.extend(chunks)
        print(f"scanning file with prefix: {cur_prefix}")
        # For each chunk, scan with the find_issues_agent
        while len(state.files_current_prefix.documents) > 0:
            cur_file = state.files_current_prefix.documents.pop()
            response = broad_find_issues_agent(cur_file)
            state.scanned_summaries.append(response)
            
        return state.scanned_summaries
