# ingestion_agent.py
import os
import shutil
from git import Repo
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document

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

print(load_code_documents("mlb-predictions-frontend"))
