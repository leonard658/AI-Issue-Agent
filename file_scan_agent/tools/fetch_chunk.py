# ---- pinecone_fetch_tools.py -----------------------------------------------
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from pydantic_types.document_schema import DocumentsChunkSchema
from pinecone import Vector
import os

load_dotenv()

pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# ---------- generic fetch for “documents” chunks ----------------------------
class FetchNextChunkToolInput(BaseModel):
    id:        str   = Field(...,  description="Vector ID to fetch next chunk for") 
    name_space: str | None = Field(default=None, description="Subgroup of the index to target")
@tool("fetch_next_chunk_tool", args_schema=FetchNextChunkToolInput, return_direct=False)
def fetch_next_chunk_tool(
    id: str,
    name_space: str | None = None
) -> DocumentsChunkSchema | None:
    """
    Fetch next chunk *by current ID* from the given Pinecone index.

    Metadata returned for each chunk:
      .metadata.chunk_index   – chunk number inside original file
      .metadata.file_path     – path of the source file
      .metadata.language      – language of that chunk
      .metadata.total_chunks  – total chunks in the parent file
      .metadata.id            – vector ID (echoed back for convenience)
      .metadata.values        – raw embedding vector (only if include_values=True)

    .page_content holds the actual chunk text.
    """
    include_values = False
    try:
        # ------------------------------------------------------------------
        # 1) call Pinecone
        # ------------------------------------------------------------------
        prefix, _idnum = id.rsplit('-', 1)
        _idnum = int(_idnum) + 1
        id = f"{prefix}-{_idnum}"
        
        index_name = os.getenv("DOCUMENTS_VDB_INDEX")
        idx = pc_client.Index(index_name)
        fetch_resp = idx.fetch(ids=[id], namespace=name_space)

        # fetch_resp is a dict: {'vectors': {id -> Vector}, 'namespace': ...}
        vectors = fetch_resp.vectors
        
        # ------------------------------------------------------------------
        # 2) build LangChain Documents
        # ------------------------------------------------------------------
        vec: Vector = next(iter(vectors.values()))

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

        doc = (Document(page_content=text, metadata=md))

        # ------------------------------------------------------------------
        # 3) coerce to DocumentsChunkSchema and return
        # ------------------------------------------------------------------
        return DocumentsChunkSchema(page_content=doc.page_content, metadata=doc.metadata)
    except Exception as e:
        print(f"Error fetching next chunk: {e}")
        return None



def fetch_first_chunk(
    prefix: str,
    include_values: bool = False,
    name_space: str | None = None
) -> DocumentsChunkSchema | None:
    """
    Fetch next chunk *by current ID* from the given Pinecone index.

    Metadata returned for each chunk:
      .metadata.chunk_index   – chunk number inside original file
      .metadata.file_path     – path of the source file
      .metadata.language      – language of that chunk
      .metadata.total_chunks  – total chunks in the parent file
      .metadata.id            – vector ID (echoed back for convenience)
      .metadata.values        – raw embedding vector (only if include_values=True)

    .page_content holds the actual chunk text.
    """
    try:
        # ------------------------------------------------------------------
        # 1) call Pinecone
        # ------------------------------------------------------------------
        id = f"{prefix}-0"
        
        index_name = os.getenv("DOCUMENTS_VDB_INDEX")
        idx = pc_client.Index(index_name)
        fetch_resp = idx.fetch(ids=[id], namespace=name_space)

        # fetch_resp is a dict: {'vectors': {id -> Vector}, 'namespace': ...}
        vectors = fetch_resp.vectors
        
        # ------------------------------------------------------------------
        # 2) build LangChain Documents
        # ------------------------------------------------------------------
        vec: Vector = next(iter(vectors.values()))

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

        doc = (Document(page_content=text, metadata=md))

        # ------------------------------------------------------------------
        # 3) coerce to DocumentsChunkSchema and return
        # ------------------------------------------------------------------
        return DocumentsChunkSchema(page_content=doc.page_content, metadata=doc.metadata)
    except Exception as e:
        print(f"Error fetching first chunk: {e}")
        return None

