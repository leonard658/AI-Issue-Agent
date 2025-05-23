# ingestion_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from ingestion_agent.github_repo_fetch import clone_repo, delete_repo, load_text_documents
from ingestion_agent.pinecone_stuff import add_to_index_for_code, add_to_index_for_issues, clear_index
from ingestion_agent.github_issues_fetch import fetch_github_issues
import os
from dotenv import load_dotenv

load_dotenv()

class IngestRepoToolInput(BaseModel):
    owner: str = Field(description="Owner of the GitHub repo to clone")
    repo: str = Field(description="Name of the GitHub repo to clone")
    branch: str = Field(description="Branch name of repo to clone")
#@tool("ingest_repo_tool", args_schema=IngestRepoToolInput, return_direct=False)
def ingest_repo_tool(owner: str, repo: str, branch: str) -> str:
    """
    Tool entrypoint: expects repo_url and branch
    Description: Cleans local target folder, Clones repo, loads & splits code, then puts code chunks into an index.
    """
    target_dir = "./tmp"
    index = os.getenv("DOCUMENTS_VDB_INDEX")

     # 0) Make sure temp repo is clean
    delete_repo(target_dir)

    # 1) Clone
    clone_repo(owner, repo, branch, target_dir)

    # 2) Load into Document chunks
    docs = load_text_documents(target_dir)

    # 3) Clear index and save vector store here
    clear_index(index)
    docsAdded: str = add_to_index_for_code(index, docs)

    # 4) Get rid of temp repo
    delete_repo(target_dir)

    return f"Ingested {len(docs)} files from https://github.com/{owner}/{repo}@{branch}\n" + docsAdded


class IngestIssuesToolInput(BaseModel):
    owner: str = Field(description="Owner of the GitHub repo to get issues from")
    repo: str = Field(description="Name of the GitHub repo to get issues from")
#@tool("ingest_issues_tool", args_schema=IngestIssuesToolInput, return_direct=False)
def ingest_issues_tool(owner: str, repo: str) -> str:
    """
    Tool entrypoint: expects owner and repo
    Gets issues, chunks them, then loads chunks into an index.
    """

    index = os.getenv("ISSUES_VDB_INDEX")

    # 1) Get issues
    issues = fetch_github_issues(owner, repo)

    # 2) Clear index and save vector store here
    clear_index(index)
    docsAdded = add_to_index_for_issues(index, issues)

    return f"Ingested issues from https://api.github.com/repos/{owner}/{repo}/issues\nAnd" + docsAdded

def ingestion_agent(owner: str, repo: str, branch: str) -> str:
    """
    Ingestion agent that ingests code and issues from GitHub repositories.
    """
    repo_response = ingest_repo_tool(owner, repo, branch)
    issues_response = ingest_issues_tool(owner, repo)
    
    return repo_response, issues_response

if __name__ == "__main__":
   ingestion_agent('leonard658', 'CustomLearnAi', 'main')