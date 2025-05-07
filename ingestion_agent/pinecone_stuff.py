from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from llama_index.core.node_parser import CodeSplitter
from langchain.schema import Document
from google import genai
from google.genai import types
import time
from google.genai.errors import ClientError
import os

# Link to the supported languages pack: https://github.com/Goldziher/tree-sitter-language-pack?tab=readme-ov-file#available-languages
# Languages that are currently supported for this: 
# C, C++, C#, CSS, Go, HTML, Java, JavaScript, Perl, PHP, Python, Ruby, Rust, Swift, TypeScript
suported_languages_mapped = {
    # Python
    'py':   'python',
    'ipynb':'python',
    'pyi':  'python',
    'pyc':  'python',
    'pyo':  'python',
    'pyw':  'python',
    'pyx':  'python',
    'pxd':  'python',
    'pyd':  'python',
    'pyz':  'python',
    # Typescript
    'ts':           'typescript',
    'tsx':          'typescript',
    'd.ts':         'typescript',
    'mts':          'typescript',
    'cts':          'typescript',
    'tsbuildinfo':  'typescript',
    # JavaScript
    'js':      'javascript',
    'mjs':     'javascript',
    'cjs':     'javascript',
    'jsx':     'javascript',
    'json':    'javascript',
    'min.js':  'javascript',
    'node':    'javascript',
    'map':     'javascript',
    'es6':     'javascript',
    'es':      'javascript',
    # Java
    'java':  'java',
    'class': 'java',
    'jar':   'java',
    'war':   'java',
    'ear':   'java',
    'jmod':  'java',
    'jsp':   'java',
    'jspx':  'java',
    'jnlp':  'java',
    'jad':   'java',
    # Go
    'go':      'go',
    'mod':     'go',
    'sum':     'go',
    'test.go': 'go',
    'tmpl':    'go',
    'tpl':     'go',
    'gohtml':  'go',
    # C
    'c':   'c',
    'h':   'c',
    'i':   'c',
    'o':   'c',
    'a':   'c',
    'so':  'c',
    'dll': 'c',
    # C++
    'cpp':  'cpp',
    'cc':   'cpp',
    'cxx':  'cpp',
    'C':    'cpp',
    'h':    'cpp',
    'hpp':  'cpp',
    'hh':   'cpp',
    'hxx':  'cpp',
    'ii':   'cpp',
    'o':    'cpp',
    'a':    'cpp',
    'so':   'cpp',
    'dll':  'cpp',
    # C#
    'cs':       'csharp',
    'csproj':   'csharp',
    'sln':      'csharp',
    'dll':      'csharp',
    'exe':      'csharp',
    'config':   'csharp',
    'resx':     'csharp',
    'cshtml':   'csharp',
    'razor':    'csharp',
    'xaml':     'csharp',
    'csx':      'csharp',
    'nuspec':   'csharp',
    # HTML
    'html':  'html',
    'htm':   'html',
    'xhtml': 'html',
    'shtml': 'html',
    'mhtml': 'html',
    'phtml': 'html',
    'dhtml': 'html',
    # CSS
    'css':     'css',
    'scss':    'css',
    'sass':    'css',
    'less':    'css',
    'styl':    'css',
    'pcss':    'css',
    'postcss': 'css',
    # PHP
    'php':   'php',
    'phtml': 'php',
    'php3':  'php',
    'php4':  'php',
    'php5':  'php',
    'inc':   'php',
    'phar':  'php',
    'phps':  'php',
    'phpt':  'php',
    # Ruby
    'rb':       'ruby',
    'erb':      'ruby',
    'rake':     'ruby',
    'gemspec':  'ruby',
    'ru':       'ruby',
    'gem':      'ruby',
    'rbw':      'ruby',
    # Rust
    'rs':           'rust',
    'rlib':         'rust',
    'rmeta':        'rust',
    'toml':         'rust',
    'lock':         'rust',
    'ron':          'rust',
    # Perl
    'pl':    'perl',
    'pm':    'perl',
    't':     'perl',
    'pod':   'perl',
    'cgi':   'perl',
    'psgi':  'perl',
    'xs':    'perl',
    'plx':   'perl',
    # Swift
    'swift':        'swift',
    'swiftmodule':  'swift',
    'swiftinterface':'swift',
    'playground':   'swift',
    'swiftpm':      'swift',
    'xcodeproj':    'swift',
    'xcworkspace':  'swift',
}

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def create_index(index_name: str):
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=2, # Replace with your model dimensions
            metric="cosine", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
   
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    batch_size = 100  # Gemini’s current max batch size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        resp = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        # extend our master list with this batch’s embeddings
        embeddings.extend(resp.embeddings)

    return embeddings

def embed_chunks_with_api_throttle_handling(chunks: list[str], max_retries: int = 8) -> list[list[float]]:
    embeddings: list[list[float]] = []
    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        attempt = 0

        while True:
            try:
                resp = client.models.embed_content(
                    model="gemini-embedding-exp-03-07",
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )
                embeddings.extend(resp.embeddings)
                break

            except ClientError as e:
                # only back off on rate-limit errors
                if e.code == 429 and attempt < max_retries:
                    backoff = 2 ** attempt
                    print(f"[embed_chunks] rate limited; retrying in {backoff}s…")
                    time.sleep(backoff)
                    attempt += 1
                else:
                    # re-raise all other errors (or if we exhaust retries)
                    raise

        # optional micro-throttle so you don’t hammer the API
        time.sleep(0.1)

    return embeddings

        
def chunkify(doc: Document):
    # 1) Grab the extension (e.g. "py", "java", etc.)
    ext = doc.metadata.get("language", "").lower()
    lang = suported_languages_mapped.get(ext)

    # 2) If it's a supported code language, use the CodeSplitter
    if lang:
        splitter = CodeSplitter(
            language=lang,           # your tree-sitter language
            chunk_lines=200,
            chunk_lines_overlap=40,
        )
        return splitter.split_text(doc.page_content)

    # 3) Otherwise, fall back to plain-text splitting by lines
    text = doc.page_content
    lines = text.splitlines()
    chunk_size = 200
    overlap = 40

    chunks = []
    start = 0
    total_lines = len(lines)
    while start < total_lines:
        # grab up to chunk_size lines
        end = min(start + chunk_size, total_lines)
        chunk = "\n".join(lines[start:end])
        chunks.append(chunk)

        # advance by chunk_size minus overlap
        start += chunk_size - overlap

    return chunks

def chunkify2(doc: Document):
    ext  = doc.metadata.get("language", "").lower()
    lang = suported_languages_mapped.get(ext)
    text = doc.page_content
    lines = text.splitlines()

    # 1) If it’s a small file, just keep it whole  
    if len(lines) <= 300:
        return [text]

    # 2) If it’s a supported code language, use the AST splitter  
    if lang:
        splitter = CodeSplitter(
            language            = lang,
            chunk_lines         = 500,   # up to 500 lines per chunk
            chunk_lines_overlap = 100,   # 100 lines of overlap
        )
        raw_chunks = splitter.split_text(text)

        # 3) Merge any very small chunks into their predecessor
        merged = []
        for chunk in raw_chunks:
            # count lines in that chunk
            n = chunk.count("\n") + 1
            if n < 50 and merged:
                # tack it onto the previous chunk
                merged[-1] = merged[-1] + "\n" + chunk
            else:
                merged.append(chunk)
        return merged

    # 4) Plain-text fallback (same as before)
    chunk_size = 300
    overlap    =  40
    chunks     = []
    start      = 0
    total      = len(lines)

    while start < total:
        end   = min(start + chunk_size, total)
        chunk = "\n".join(lines[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def add_to_index(index_name: str, docs: list[Document]):
    vectors = []
    for doc in docs:
        #print(doc)
        prefix = doc.metadata.get("file_path", doc.metadata.get("source", "doc")).replace("/", "_")
        #print(prefix)
        
        chunks = chunkify2(doc)
        print(f"{prefix}({len(chunks)}):")
        #print(chunks)
        # generate embeddings in bulk
        embs = embed_chunks_with_api_throttle_handling(chunks)
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
                    "text": text,            # (optional) raw code snippet
                },
            })

    # send a single upsert with all vectors
    print(vectors)
    pc.Index(index_name).upsert(vectors)  
    
    