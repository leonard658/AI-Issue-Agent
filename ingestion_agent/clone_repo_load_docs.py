import os
import shutil
from git import Repo
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
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

def load_text_documents(repo_path: str) -> list[Document]:
    """
    Recursively load *all* files in `repo_path`, but only keep
    the ones we can successfully decode as UTF-8 text.
    """
    docs: list[Document] = []
    root = Path(repo_path)

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        # read raw bytes
        raw = file_path.read_bytes()

        # simple binary check: if you see a NULL byte, skip it
        if b"\x00" in raw:
            continue

        # try to decode as UTF-8 (you can try other encodings or chardet here)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue

        # build metadata
        rel = file_path.relative_to(root).as_posix()
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_path": rel,
                    "language": file_path.suffix.lstrip("."),  # e.g. "py", "cfg", "toml", "env"
                },
            )
        )

    return docs

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

