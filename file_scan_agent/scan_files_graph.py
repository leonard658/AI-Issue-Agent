# file_scan_agent.py
import json
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from file_scan_agent.tools.fetch_chunks_by_prefix import fetch_chunks_by_prefix
from file_scan_agent.tools.pull_all_index_ids import pull_all_index_prefixes
from file_scan_agent.tools.fetch_chunk import fetch_next_chunk_tool, fetch_first_chunk
import os
from dotenv import load_dotenv
from find_issues_agent.broad_find_issues_agent import broad_find_issues_agent
from langgraph.graph import StateGraph, START, END
from pydantic_types.to_json_str import to_json_str

load_dotenv()

DOCS_INDEX = os.getenv("DOCUMENTS_VDB_INDEX")

class ShouldScan(BaseModel):
    should_scan: bool = Field(description="Boolean value indicating if the file should be scanned (True - should be scanned, False - should not be scanned)")

# ── ScanState ───────────────────────────────────────────────────
class ScanState(BaseModel):
    filter_criteria: str | None = None

    todo_prefixes: list[str] = []
    cur_prefix: str | None = None
    scanned_prefixes: list[str] = []
    scanned_summaries: list[str] = []

    should_scan: bool | None = None      # ← used in router
    _event: str | None = None            # ← used in router

    
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

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[fetch_next_chunk_tool],
    prompt=prompt,
    response_format=ShouldScan
)

def load_prefixes_and_filter(state: ScanState) -> ScanState:
    filt = input("Enter a filter or descriptor for files to scan: ").strip()
    prefixes = list(pull_all_index_prefixes(DOCS_INDEX))   # convert set → list
    return state.model_copy(
        update={"filter_criteria": filt, "todo_prefixes": prefixes}
    )

def decision_node(state: ScanState):
    # 1) Handle empty list → END
    if not state.todo_prefixes:
        return {"_event": "done"}

    # 2) Pop next prefix & fetch first chunk
    cur_prefix = state.todo_prefixes.pop()
    first_chunk = fetch_first_chunk(cur_prefix)
    if first_chunk is None:
        # Skip corrupt / empty file.  Return no updates → loop again.
        return {}

    # 3) Ask the LLM agent
    message_content = (
        f"User filter/descriptor:\n{state.filter_criteria}\n\n"
        f"Here is the first chunk:\n{to_json_str(first_chunk)}"
    )
    res = agent.invoke({"messages": [{"role": "user", "content": message_content}]})
    should_scan: bool = res["structured_response"].should_scan

    # 4) If skip → loop; if scan → gather chunks & update
    if not should_scan:
        return state.model_copy(update={"should_scan": False})

    chunks = fetch_chunks_by_prefix(os.getenv("DOCUMENTS_VDB_INDEX"), cur_prefix)
    # TODO: run your scanning logic on `chunks` here
    summary = f"Scanned {len(chunks)} chunks in {cur_prefix}"

    return state.model_copy(
        update={
            "cur_prefix": cur_prefix,
            "should_scan": True,
        }
    )

def scan_node(state: ScanState) -> ScanState:
    chunks = fetch_chunks_by_prefix(DOCS_INDEX, state.cur_prefix)
    summary = f"Scanned {len(chunks)} chunks in {state.cur_prefix}" # broad_find_issues_agent(cur_file)
    return state.model_copy(
        update={
            "scanned_prefixes": state.scanned_prefixes + [state.cur_prefix],
            "scanned_summaries":  state.scanned_summaries  + [summary],
        }
    )

# ---- single conditional router --------------------------------
def _route(state: ScanState) -> str:
    # safe attribute lookup
    if getattr(state, "_event", None) == "done":
        return "done"
    return "scan" if getattr(state, "should_scan", False) else "loop"


# ── graph wiring with a scan step ───────────────────────────────
graph = StateGraph(ScanState)
graph.add_node("load",   load_prefixes_and_filter)
graph.add_node("decide", decision_node)
graph.add_node("scan",   scan_node)

graph.add_edge(START, "load")
graph.add_edge("load", "decide")
graph.add_edge("scan", "decide")      # after scanning, check next prefix

graph.add_conditional_edges(
    "decide",
    _route,
    {"done": END, "scan": "scan", "loop": "decide"},
)

scan_files_graph = graph.compile()

# Provide an initial state (filter_criteria can be empty string)
if __name__ == "__main__":
    final_state = scan_files_graph.invoke(ScanState(filter_criteria=""), config={"recursion_limit": 100})
    print("Scanned:", final_state["scanned_prefixes"])
