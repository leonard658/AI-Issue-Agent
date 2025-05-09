# ---- pinecone_fetch_tools.py -----------------------------------------------
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from pydantic_types.document_schema import DocumentsChunkSchema
import os

load_dotenv()

pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# ---------- generic fetch for “documents” chunks ----------------------------
class FetchDocumentsToolInput(BaseModel):
    index_name: str           = Field(...,  description="Name of the Pinecone index (e.g. 'documents')")
    ids:        list[str]     = Field(...,  description="Vector IDs to fetch")
    include_values: bool      = Field(False, description="If True, embed-values are returned in metadata")

@tool("fetch_documents_tool", args_schema=FetchDocumentsToolInput, return_direct=False)
def fetch_documents_tool(
    index_name: str,
    ids: list[str],
    include_values: bool = False,
) -> list[DocumentsChunkSchema]:
    """
    Fetch one or more vectors *by ID* from the given Pinecone index.

    Metadata returned for each chunk:
      .metadata.chunk_index   – chunk number inside original file
      .metadata.file_path     – path of the source file
      .metadata.language      – language of that chunk
      .metadata.total_chunks  – total chunks in the parent file
      .metadata.id            – vector ID (echoed back for convenience)
      .metadata.values        – raw embedding vector (only if include_values=True)

    .page_content holds the actual chunk text.
    """
    idx   = pc_client.Index(index_name)
    resp  = idx.fetch(ids=ids).to_dict()          # ‹fetch› always returns metadata and max 1000 IDs/req :contentReference[oaicite:0]{index=0}
    vecs  = resp.get("vectors", {})

    docs: list[Document] = []
    for vid, vdata in vecs.items():
        md = vdata.get("metadata", {}).copy()

        # pull the stored text (your up-sert uses "text" or "embedding_text")
        text = md.pop("text", md.pop("embedding_text", ""))

        md["id"] = vid
        if include_values:
            md["values"] = vdata.get("values", [])

        docs.append(Document(page_content=text, metadata=md))

    return [DocumentsChunkSchema(page_content=d.page_content, metadata=d.metadata) for d in docs]


# ---------- identical pattern for “issues” chunks ---------------------------
class FetchIssuesToolInput(BaseModel):
    index_name: str           = Field(...,  description="Name of the Pinecone index (e.g. 'issues')")
    ids:        list[str]     = Field(...,  description="Vector IDs to fetch")
    include_values: bool      = Field(False, description="If True, embed-values are returned in metadata")

@tool("fetch_issues_tool", args_schema=FetchIssuesToolInput, return_direct=False)
def fetch_issues_tool(
    index_name: str,
    ids: list[str],
    include_values: bool = False,
) -> list[DocumentsChunkSchema]:
    """
    Fetch issue chunks by vector ID.

    Metadata returned for each chunk:
      .metadata.author        – GitHub issue author
      .metadata.chunk_index   – chunk number inside the issue
      .metadata.created_at    – ISO timestamp
      .metadata.labels        – list of labels
      .metadata.title         – issue title
      .metadata.total_chunks  – total chunks in that issue
      .metadata.id            – vector ID
      .metadata.values        – raw embedding vector (only if include_values=True)
    """
    idx   = pc_client.Index(index_name)
    resp  = idx.fetch(ids=ids).to_dict()
    vecs  = resp.get("vectors", {})

    docs: list[Document] = []
    for vid, vdata in vecs.items():
        md = vdata.get("metadata", {}).copy()
        page_text = md.pop("embedding_text", md.pop("text", ""))

        md["id"] = vid
        if include_values:
            md["values"] = vdata.get("values", [])

        docs.append(Document(page_content=page_text, metadata=md))

    return [DocumentsChunkSchema(page_content=d.page_content, metadata=d.metadata) for d in docs]
# ---------------------------------------------------------------------------

