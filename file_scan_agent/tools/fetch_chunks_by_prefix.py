import os
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import  Optional
from langchain.schema import Document
from pydantic_types.document_schema import DocumentsChunkSchema

load_dotenv()

pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    
def fetch_documents(
    ids: list[str],
    include_values: bool = False,
    name_space: str | None = None
) -> list[DocumentsChunkSchema]:
    """
    Fetch one or more chunks *by ID* from the given Pinecone index.

    Metadata returned for each chunk:
      .metadata.chunk_index   – chunk number inside original file
      .metadata.file_path     – path of the source file
      .metadata.language      – language of that chunk
      .metadata.total_chunks  – total chunks in the parent file
      .metadata.id            – vector ID (echoed back for convenience)
      .metadata.values        – raw embedding vector (only if include_values=True)

    .page_content holds the actual chunk text.
    """
    # ------------------------------------------------------------------
    # 1) call Pinecone
    # ------------------------------------------------------------------
    index_name = os.getenv("DOCUMENTS_VDB_INDEX")
    idx = pc_client.Index(index_name)
    fetch_resp = idx.fetch(ids=ids, namespace=name_space)

    # fetch_resp is a dict: {'vectors': {id -> Vector}, 'namespace': ...}
    vectors = fetch_resp.vectors


    # ------------------------------------------------------------------
    # 2) build LangChain Documents
    # ------------------------------------------------------------------
    docs: list[Document] = []
    for vec in vectors.values():                 # vec is a pinecone.Vector
        md = vec.metadata.copy() if vec.metadata else {}

        # Pop out the text for page_content
        text = md.pop("text", md.pop("embedding_text", ""))

        # Bring required extra fields in line with the query tool
        md["id"]    = vec.id
        md["score"] = None                       # no similarity score on a fetch

        if include_values:
            md["values"] = vec.values            # raw embedding
        else:
            # ensure we don't leak the embedding if it was saved in metadata
            md.pop("values", None)

        docs.append(Document(page_content=text, metadata=md))

    # ------------------------------------------------------------------
    # 3) coerce to DocumentsChunkSchema and return
    # ------------------------------------------------------------------
    return [
        DocumentsChunkSchema(page_content=d.page_content, metadata=d.metadata)
        for d in docs
    ]

def fetch_chunks_by_prefix(
    index_name: str,
    prefix: str,
    name_space: Optional[str] = None,
) -> list[DocumentsChunkSchema]:
    """
    Retrieves every ID from a Pinecone index.
    
    Args:
        index_name:     The name for your index (e.g. "abc123-us-west1-gcp").
        name_space:      (Optional) the namespace to target. If omitted, default namespace is used.

    Returns:
        A list of ids in the index
    """
    # Initialize client and target the index
    index = pc_client.Index(index_name)
    
    ids = index.list(namespace=name_space, prefix=prefix)
    # Gets list of lists of IDs by page
    ids = list(ids)
    # Flatten the list of lists
    ids = [item for sublist in ids for item in sublist]
    
    return fetch_documents(ids, name_space=name_space)
