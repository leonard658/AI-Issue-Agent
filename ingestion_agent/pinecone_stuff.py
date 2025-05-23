from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from llama_index.core.node_parser import CodeSplitter
from langchain.schema import Document
import os
from openai import OpenAI
import tiktoken

# Link to the supported languages pack: https://github.com/Goldziher/tree-sitter-language-pack?tab=readme-ov-file#available-languages
# Languages that are currently supported for this: 
# C, C++, C#, CSS, Go, HTML, Java, JavaScript, Perl, PHP, Python, Ruby, Rust, Swift, TypeScript, and miscellaneous file types like JSON, CSV, YAML, XML, and shell scripts.
suported_languages_mapped = {
     # Python
    'py':     'python',
    'pyw':    'python',
    'pyi':    'python',
    'ipynb':  'python',
    'pyx':    'python',
    'pxd':    'python',

    # TypeScript
    'ts':     'typescript',
    'tsx':    'typescript',

    # JavaScript
    'js':     'javascript',
    'mjs':    'javascript',
    'cjs':    'javascript',
    'jsx':    'javascript',

    # Java
    'java':   'java',

    # Go
    'go':     'go',

    # C
    'c':      'c',
    'h':      'c',

    # C++
    'cpp':    'cpp',
    'cc':     'cpp',
    'cxx':    'cpp',
    'hpp':    'cpp',

    # C#
    'cs':     'csharp',

    # HTML
    'html':   'html',
    'htm':    'html',
    'xhtml':  'html',
    'shtml':  'html',

    # CSS
    'css':    'css',
    'scss':   'css',
    'sass':   'css',
    'less':   'css',
    'styl':   'css',

    # PHP
    'php':    'php',
    'phtml':  'php',

    # Ruby
    'rb':     'ruby',
    'erb':    'ruby',

    # Rust
    'rs':     'rust',

    # Perl
    'pl':     'perl',
    'pm':     'perl',

    # Swift
    'swift':  'swift',

    # Miscellaneous file types
    'json':  'json',

    'csv': 'csv',

    'yaml': 'yaml',
    'yml':  'yaml',

    'xml': 'xml',

    'sh':    'bash',
    'bash':  'bash',

    'ps1':   'powershell',
    'psm1':  'powershell',
    'psd1':  'powershell',
}

load_dotenv()

pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---  token helpers - to make sure that the chunks of each doc aren't too big  ---------------------------------------------------------
_MODEL            = "text-embedding-3-small"
_MAX_MODEL_TOKENS = 8192
_TARGET_TOKENS    = _MAX_MODEL_TOKENS // 2      # 4 096
_enc              = tiktoken.encoding_for_model(_MODEL)

def n_tokens(text: str) -> int:
    """Exact token count for this model."""
    return len(_enc.encode(text))

def split_by_tokens(text: str, limit: int = _TARGET_TOKENS) -> list[str]:
    """
    Split a long string into pieces, each ≤ limit tokens.
    Keeps line boundaries so code stays readable.
    """
    lines, out, cur, cur_tok = text.splitlines(), [], [], 0
    for ln in lines:
        ln_tok = n_tokens(ln) + 1          # +1 for the newline we lost
        if cur_tok + ln_tok > limit and cur:
            out.append("\n".join(cur))
            cur, cur_tok = [ln], ln_tok
        else:
            cur.append(ln)
            cur_tok += ln_tok
    if cur:
        out.append("\n".join(cur))
    return out


# ---  Pinecone db management  ---------------------------------------------------------
def create_index(index_name: str):
    if not pc_client.has_index(index_name):
        pc_client.create_index(
            name=index_name,
            dimension=1536, # Replace with your model dimensions
            metric="cosine", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )

def delete_index(index_name: str):
    pc_client.delete_index(index_name)

def clear_index(index_name: str):
    try:
        pc_client.Index(index_name).delete(delete_all=True)
    except Exception as e:
        print(f"Error clearing index {index_name}: {e}")
     

# ---  embedding functions  ----------------------------------------------------
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    batch_size = 100  # Gemini’s current max batch size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        resp = openai_client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )

        # Collect each vector, not flatten one of them
        embeddings.extend([d.embedding for d in resp.data])

    return embeddings
        
def chunkify(
    doc: Document,
    *,
    min_small_lines: int = 300,
    code_chunk_lines: int = 500,
    code_overlap: int = 100,
    text_chunk_lines: int = 300,
    text_overlap: int = 40,
    token_limit: int = _TARGET_TOKENS,     # ≈ 4 096 by default
) -> list[str]:
    """
    Split one Document into token‑safe chunks ready for embedding.

    • Uses an AST‑aware splitter for known code languages.
    • Falls back to simple line windows for everything else.
    • Guarantees every returned chunk is ≤ token_limit tokens
      (recursively re‑splits anything that slips through).
    """
    # ────────────────────────────────────────────────────────────────
    # 0) Detect extension / language
    # ────────────────────────────────────────────────────────────────
    ext  = doc.metadata.get("language", "").lower()
    lang = suported_languages_mapped.get(ext)
    text = doc.page_content
    lines = text.splitlines()

    # 1) Tiny files → single chunk
    if len(lines) <= min_small_lines:
        chunks = [text]

    # 2) Code files → AST splitter + merge very small pieces
    elif lang:
        splitter = CodeSplitter(
            language            = lang,
            chunk_lines         = code_chunk_lines,
            chunk_lines_overlap = code_overlap,
        )
        raw_chunks = splitter.split_text(text)

        # merge dangling tiny chunks (<50 lines) into their predecessor
        chunks = []
        for ch in raw_chunks:
            if ch.count("\n") + 1 < 50 and chunks:
                chunks[-1] = chunks[-1] + "\n" + ch
            else:
                chunks.append(ch)

    # 3) Plain‑text fallback
    else:
        chunks, start, total = [], 0, len(lines)
        while start < total:
            end = min(start + text_chunk_lines, total)
            chunks.append("\n".join(lines[start:end]))
            start += text_chunk_lines - text_overlap

    # 4) FINAL PASS → enforce token budget on every chunk
    safe_chunks: list[str] = []
    for ch in chunks:
        if n_tokens(ch) <= token_limit:
            safe_chunks.append(ch)
        else:
            safe_chunks.extend(split_by_tokens(ch, token_limit))

    return safe_chunks

# ---  adding docs to indexes  -------------------------------------------------------
def add_to_index_for_code(index_name: str, docs: list[Document], name_space: str | None = None):
    vectors = []
    addedFilesAndChunks = "Files and chunks added:\n"
    for doc in docs:
        prefix = doc.metadata.get("file_path", doc.metadata.get("source", "doc")).replace("/", "_")
        
        chunks = chunkify(doc)
        addedFilesAndChunks += f"{prefix}({len(chunks)} chunk(s)).\n"

        # generate embeddings in bulk
        embs = embed_chunks(chunks)
        for i, (text, vector) in enumerate(zip(chunks, embs)):
            #print(text, vector)
            chunk_id = f"{prefix}-{i}"
            # build upsert payload
            vectors.append({
                "id": chunk_id,              # 1) unique ID
                "values": vector,            # 2) embedding vector
                "metadata": {
                    **doc.metadata,          # original metadata: file_path, source, language, …
                    "chunk_index": i,        # which chunk number
                    "total_chunks": len(chunks),
                    "text": text,            # (optional) raw code snippet
                },
            })

    # only upsert if there’s something to send
    if vectors:
        pc_client.Index(index_name).upsert(vectors, namespace=name_space)  
    
    return f"Added:\n {addedFilesAndChunks} chunks to index {index_name}"
    

# ---  adding issues to index  -------------------------------------------------------
def add_to_index_for_issues(index_name: str, docs: list[Document], name_space: str | None = None):
    vectors = []
    added_issues = ""
    
    for doc in docs:
        # build a “safe” prefix from author and timestamp
        author     = doc.metadata.get("author", "unknown")
        created_at = doc.metadata.get("created_at", "").replace(":", "-")
        prefix     = f"{author}_{created_at}"
        
        # split the issue body into chunks
        chunks = chunkify(doc)
        
        added_issues += f"\n{prefix} ({len(chunks)} chunk(s))"
        
        # embed in bulk
        embs = embed_chunks(chunks)
        for i, (text, vector) in enumerate(zip(chunks, embs)):
            chunk_id = f"{prefix}-{i}"
            vectors.append({
                "id": chunk_id,
                "values": vector,
                "metadata": {
                    "author":         doc.metadata.get("author"),
                    "created_at":     doc.metadata.get("created_at"),
                    "updated_at":     doc.metadata.get("updated_at"),
                    "closed_at":      doc.metadata.get("closed_at"),
                    "state":          doc.metadata.get("state"),
                    "number":         doc.metadata.get("number"),
                    "slug":           doc.metadata.get("slug"),
                    "title":          doc.metadata.get("title"),
                    "labels":         doc.metadata.get("labels"),
                    "chunk_index":    i,
                    "total_chunks":   len(chunks),
                    "embedding_text": text,
                },
            })
    
    # only upsert if there’s something to send
    if vectors:
        pc_client.Index(index_name).upsert(vectors, namespace=name_space)
    
    # build and return a summary
    return f" added {len(vectors)} chunks across {len(docs)} issues to index '{index_name}'.\nAdded following issues: {added_issues}"

    
