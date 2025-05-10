# ---- pinecone_fetch_tools.py -----------------------------------------------
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from pydantic_types.document_schema import DocumentsChunkSchema
from pydantic_types.issue_schema import IssueChunkSchema
import os

load_dotenv()

pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# ---------- generic fetch for “documents” chunks ----------------------------
class FetchDocumentsToolInput(BaseModel):
    index_name: str           = Field(...,  description="Name of the Pinecone index (e.g. 'documents')")
    ids:        list[str]     = Field(...,  description="Vector IDs to fetch")
    include_values: bool      = Field(False, description="If True, embed-values are returned in metadata")
    name_space: str | None = Field(default=None, description="Subgroup of the index to target")
@tool("fetch_documents_tool", args_schema=FetchDocumentsToolInput, return_direct=False)
def fetch_documents_tool(
    index_name: str,
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

# ---------- identical pattern for “issues” chunks ---------------------------
class FetchIssuesToolInput(BaseModel):
    index_name: str           = Field(...,  description="Name of the Pinecone index (e.g. 'issues')")
    ids:        list[str]     = Field(...,  description="Vector IDs to fetch")
    include_values: bool      = Field(False, description="If True, embed-values are returned in metadata")    
    name_space: str | None = Field(default=None, description="Subgroup of the index to target")
@tool("fetch_issues_tool", args_schema=FetchIssuesToolInput, return_direct=False)
def fetch_issues_tool(
    index_name: str,
    ids: list[str],
    include_values: bool = False,
    name_space: str | None = None
) -> list[IssueChunkSchema]:
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
    # ------------------------------------------------------------------
    # 1) call Pinecone
    # ------------------------------------------------------------------
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
        IssueChunkSchema(page_content=d.page_content, metadata=d.metadata)
        for d in docs
    ]
# ---------------------------------------------------------------------------

#print(fetch_documents_tool("documents", ['custom-learn-ai\README.md-0']))
#print(fetch_issues_tool("issues", ['leonard658_2025-05-08T05-21-20Z-0']))
