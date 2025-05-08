import os
import shutil
from git import Repo
import stat
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
from dotenv import load_dotenv
import string


load_dotenv()

# Example usage:
# clone_repo("leonard658", "mlb-predictions-frontend", "main", "./temp") 
def clone_repo(owner: str, repo: str, branch: str, target_dir: str):
    """
    Clone (or re-clone) the given repo + branch into target_dir.
    If GITHUB_TOKEN is set and the URL is HTTPS, injects the token
    so that private repos can be cloned.
    After cloning, recursively clears the read-only bit on every file and folder.
    """
    repo_url = f"https://github.com/{owner}/{repo}"
    token = os.getenv("GITHUB_TOKEN")
    if token and repo_url.startswith("https://"):
        auth_url = repo_url.replace("https://", f"https://{token}@")
    else:
        auth_url = repo_url

    # blow away any previous clone
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # perform the shallow clone
    Repo.clone_from(auth_url, target_dir, branch=branch, depth=1)

    # --- now clear read-only on everything under target_dir ---
    for root, dirs, files in os.walk(target_dir):
        # clear on directories
        for d in dirs:
            full_dir = os.path.join(root, d)
            os.chmod(full_dir, stat.S_IWRITE | stat.S_IREAD)
        # clear on files
        for f in files:
            full_file = os.path.join(root, f)
            os.chmod(full_file, stat.S_IWRITE | stat.S_IREAD)

    return target_dir

# Example usage:
# delete_repo("./temp_repo")
def delete_repo(target_dir: str) -> None:
    """
    Deletes the directory at `target_dir` and all of its contents.
    If the directory does not exist, no action is taken.
    """
    repo_path = Path(target_dir)
    if repo_path.is_dir():
        shutil.rmtree(repo_path)
        print(f"Removed repository directory: {repo_path}")
    else:
        print(f"No repository found at: {repo_path}")


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

# ---------- extensions that are *always* considered binary ----------
BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".icns", ".svg",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".gz", ".bz2", ".xz", ".tar",
    ".exe", ".dll", ".so", ".dylib", ".class", ".jar", ".war",
    ".ttf", ".otf", ".woff", ".woff2",
    ".mp3", ".wav", ".flac", ".ogg",
    ".mp4", ".mkv", ".avi", ".mov",
    ".psd", ".ai", ".sketch",
}

_PRINTABLE = set(bytes(string.printable, "ascii")) | {0x09, 0x0A, 0x0D}  # tab/lf/cr

def _looks_textual(raw: bytes, sample_len: int = 4096, pct_printable: float = 0.70) -> bool:
    """
    Heuristic: treat as text if the first `sample_len` bytes contain
    *no* NULs and >= `pct_printable` of bytes are printable ASCII.
    """
    if b"\x00" in raw:
        return False
    sample = raw[:sample_len]
    if not sample:  # empty file
        return False
    printable = sum(b in _PRINTABLE for b in sample)
    return printable / len(sample) >= pct_printable

# Example usage:
# print(load_code_documents("mlb-predictions-frontend"))
def load_text_documents(repo_path: str) -> list[Document]:
    """
    Walk `repo_path`, skipping hidden paths and binary files, and return
    LangChain `Document`s containing UTF‑8 text.
    """
    docs: list[Document] = []
    root = Path(repo_path).expanduser().resolve()

    for file_path in root.rglob("*"):
        # -- 1. skip directories & hidden segments (.git, .vscode, etc.)
        if file_path.is_dir():
            continue
        if any(part.startswith(".") for part in file_path.relative_to(root).parts):
            continue

        # -- 2. quick binary veto by extension
        if file_path.suffix.lower() in BINARY_EXTS:
            continue

        raw = file_path.read_bytes()

        # -- 3. quick heuristic for text‑ish content
        if not _looks_textual(raw):
            continue

        # -- 4. UTF‑8 decode (skip if it fails)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue

        # -- 5. build the doc
        inside = file_path.relative_to(root)             
        rel_path = str(inside)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "file_path": rel_path,
                    "language": file_path.suffix.lstrip(".") or "text",
                },
            )
        )

    return docs


