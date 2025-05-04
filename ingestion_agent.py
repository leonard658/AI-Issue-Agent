# ingestion_agent.py
import os
import shutil
from git import Repo
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

def clone_repo(repo_url: str, branch: str, target_dir: str):
    """
    Clone (or re-clone) the given repo + branch into target_dir.
    If GITHUB_TOKEN is set and the URL is HTTPS, injects the token
    so that private repos can be cloned.
    """
    # if you've set GITHUB_TOKEN in your env, and URL is HTTPS,
    # embed it so GitPython can authenticate
    token = os.getenv("GITHUB_TOKEN")
    if token and repo_url.startswith("https://"):
        # note: keep this out of logs!
        auth_url = repo_url.replace("https://", f"https://{token}@")
    else:
        auth_url = repo_url

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    Repo.clone_from(auth_url, target_dir, branch=branch, depth=1)
    return target_dir

# Example usage:
# clone_repo("https://github.com/leonard658/mlb-predictions-frontend", "main", "mlb-predictions-frontend")


# May need to add token length checking to make sure that the LLM can handle all of the documents
def load_code_documents(repo_path: str) -> list[Document]:
    """
    Walks the cloned repo and loads each code file into a LangChain Document.
    """
    # Only pick common code file extensions; adjust as needed.
    # Currently has Python, JavaScript (and Jsx), TypeScript (and Tsx), Java, Go, C/C++, C#, HTML, CSS, Markdown, and plain text.
    FILE_GLOBS = ["**/*.py", "**/*.js",  "**/*.jsx", "**/*.ts",  "**/*.tsx", "**/*.java", "**/*.go",  "**/*.c",  "**/*.cpp",  "**/*.cs", "**/*.html", "**/*.css", "**/*.md", "**/*.txt"]

    loader = DirectoryLoader(
        repo_path,
        glob=FILE_GLOBS,
        loader_cls=TextLoader,           # simply reads raw text
        show_progress=True,              # tqdm progress bar
        recursive=True
    )
    docs = loader.load()  # → list[Document]

    # Add metadata: file path and deduced language
    for doc in docs:
        path = Path(doc.metadata["source"])
        doc.metadata["file_path"] = path.relative_to(repo_path).as_posix()
        doc.metadata["language"] = path.suffix.lstrip(".")
    return docs

# Example usage:
# print(load_code_documents("mlb-predictions-frontend"))


def ingest_repo_tool(input_str: str) -> str:
    """
    Tool entrypoint: expects "repo_url branch_name"
    Clones, loads & splits code, then returns a human-friendly summary.
    """
    try:
        repo_url, branch = input_str.split()
    except ValueError:
        return "❌ Input must be: <repo_url> <branch>"

    target_dir = "./tmp_repo"
    # 1) Clone
    clone_repo(repo_url, branch, target_dir)

    # 2) Load & split into Document chunks
    docs = load_code_documents(target_dir)

    # 3) (Optional) Save to disk or vector store here...

    return docs#f"✅ Ingested {len(docs)} code chunks from {repo_url}@{branch}"

ingest_tool = Tool(
    name="ingest_repo",
    func=ingest_repo_tool,
    description=(
        "Clones a Git repo and breaks its files into Document chunks. "
        "Input: '<repo_url> <branch>'."
    )
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=[ingest_tool],
    prompt="You are a helpful code-ingestion assistant."
)

if __name__ == "__main__":
    user_msg = "Ingest repo https://github.com/leonard658/mlb-predictions-frontend main"
    response = agent.invoke({
        "messages": [{"role": "user", "content": user_msg}]
    })
    print(response)