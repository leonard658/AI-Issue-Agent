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
1. Setup: Before scanning begins, you will receive a user‐provided filter or descriptor that defines which files to consider.
    a. You should STRICTLY follow any user provided filter or description of what they want scanned. 
    b. IF they ask only for a specific file be scanned. ONLY scan that specific file.
2. Input: You’ll receive:
   a. A user’s filter or topic description (e.g., “scan files related to X,” “only include files matching Y,” etc.).
   b. The first chunk of a file along with its metadata.

3. Goal: Decide whether this file should be scanned based on:
   a. Whether it matches the user’s filter or descriptor (inspect filename, path, or content for relevance).
   b. General rules to skip files that do not affect system behavior (unless specifically requested by the user), such as:
      • Dependency or lock files (e.g., package manifests, lockfiles).  
      • Files containing only comments, headers, or documentation.  
      • Pure configuration or metadata files.

4. If uncertain whether the file matches the user’s filter or affects behavior, use the provided tool to fetch the next chunk and re‐evaluate.

5. Output: Return `True` if the file meets the user’s criteria and should be scanned; otherwise return `False`.

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
    Scan files based on a user‐provided filter or descriptor
    """
    docs_index = os.getenv("DOCUMENTS_VDB_INDEX")

    # 1) Ask the user for a filter/descriptor before scanning begins
    filter_criteria = input("Enter a filter or descriptor for files to scan (e.g., 'only include files related to authentication'): ").strip()

    # 2) Initialize state
    state = FileScanState(
        file_prefixes_to_explore=pull_all_index_prefixes(docs_index),
        files_current_prefix=DocumentList(documents=[]),
        scanned_prefixes=[],
        scanned_summaries=[]
    )

    # 3) Loop through all prefixes
    while state.file_prefixes_to_explore:
        cur_prefix = state.file_prefixes_to_explore.pop()
        first_chunk = fetch_first_chunk(cur_prefix)
        if first_chunk is None:
            print(f"Error: No first chunk found for prefix {cur_prefix}")
            continue

        chunk_str = to_json_str(first_chunk)
        # Include the user‐provided filter in the agent prompt
        message_content = f"""
        User filter/descriptor:
        {filter_criteria}

        Here is the first chunk of a file and its metadata:
        {chunk_str}
        """

        # 4) Ask the agent whether to scan this file
        response = agent.invoke(
            { "messages": [{"role": "user", "content": message_content}] },
            debug=False
        )

        if not response["structured_response"].should_scan:
            continue  # skip files that don’t match the criteria

        # 5) Fetch all chunks for this prefix and scan them
        chunks = fetch_chunks_by_prefix(docs_index, cur_prefix)
        state.files_current_prefix.documents.extend(chunks)
        print(f"Scanning file with prefix: {cur_prefix}")

        while state.files_current_prefix.documents:
            cur_file = state.files_current_prefix.documents.pop()
            scan_result = "files have been scanned"#broad_find_issues_agent(cur_file)
            state.scanned_summaries.append(scan_result)

    return state.scanned_summaries

if __name__ == "__main__":
    print(scan_specific_files())

