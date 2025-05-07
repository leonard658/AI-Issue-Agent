# ingestion_agent.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from ingestion_agent.clone_repo_load_docs import clone_repo, load_code_documents, load_text_documents
from ingestion_agent.pinecone_stuff import add_to_index

class IngestToolInput(BaseModel):
    repo_url: str = Field(description="Url of the GitHub repo")
    branch: str = Field(description="Branch name of repo to clone")
#@tool("ingest_repo_tool", args_schema=IngestToolInput, return_direct=False)
def ingest_repo_tool(repo_url: str, branch: str) -> str:
    """
    Tool entrypoint: expects repo_url and branch
    Clones, loads & splits code, then returns a human-friendly summary.
    """
    target_dir = "./tmp_repo"
    # 1) Clone
    clone_repo(repo_url, branch, target_dir)

    # 2) Load & split into Document chunks
    docs = load_code_documents(target_dir)
    #print(docs)

    # 3) (Optional) Save to disk or vector store here...
    add_to_index("documents", docs)

    return f"✅ Ingested {len(docs)} code chunks from {repo_url}@{branch}"




llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[ingest_repo_tool],
    prompt="You are a helpful code-ingestion assistant."
)

if __name__ == "__main__":
    ingest_repo_tool("https://github.com/leonard658/mlb-predictions-frontend", "main")
    #user_msg = "Ingest repo https://github.com/leonard658/mlb-predictions-frontend main"
    #response = agent.invoke({
    #    "messages": [{"role": "user", "content": user_msg}]
    #})
    #print(response)
    