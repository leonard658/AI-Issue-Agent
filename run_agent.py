# main_graph.py

from pydantic import BaseModel
from typing import Optional, Tuple

from langgraph.graph import StateGraph, START, END
from ingestion_agent import ingestion_agent
from file_scan_agent.file_scan_agent import scan_all_files
import graphviz

# assume you’ve already built & compiled your graph:
#   graph = builder.compile()


class MainState(BaseModel):
    # inputs
    owner: str
    repo: str
    branch: str

    # populated by ingest_node
    ingestion_result: Optional[Tuple[str, str]] = None


def ingest_node(state: MainState) -> dict:
    """
    1) Ingest code and issues via your existing agent.
    2) Store the tuple of (repo_message, issues_message) in state.
    """
    repo_msg, issues_msg = ingestion_agent(state.owner, state.repo, state.branch)
    return {"ingestion_result": (repo_msg, issues_msg)}


def scan_node(state: MainState) -> dict:
    """
    1) Kick off your file-scan routine.
    2) (You could return a summary here if you modify scan_all_files to return one.)
    """
    scan_all_files()
    return {}


# Build the graph
builder = StateGraph(MainState)
builder.add_node(ingest_node)
builder.add_node(scan_node)

# Define control flow: START → ingest_node → scan_node → END
builder.add_edge(START, ingest_node.__name__)
builder.add_edge(ingest_node.__name__, scan_node.__name__)
builder.add_edge(scan_node.__name__, END)

graph = builder.compile()

if __name__ == "__main__":
    # Kick it off—you only need to supply owner, repo, and branch.
    output = graph.invoke({
        "owner": "leonard658",
        "repo": "CustomLearnAi",
        "branch": "main"
    })
    print(output)
