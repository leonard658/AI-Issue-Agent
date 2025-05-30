from pinecone import Pinecone
from dotenv import load_dotenv
from langchain.schema import Document
import os
from openai import OpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from pydantic_types.document_schema import DocumentsChunkSchema
from pydantic_types.issue_schema import IssueChunkSchema

load_dotenv()

pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL            = "text-embedding-3-small"

class QueryDocumentsToolInput(BaseModel):
    query: str = Field(description="The natural‐language or code snippet you want to find similar docs for")
    top_k: int = Field(default=3, description="How many results to return")
    include_values: bool = Field(default=False, description="If True, returns the raw embedding values in metadata")
    name_space: str | None = Field(default=None, description="Subgroup of the index to target")
@tool("query_documents_tool", args_schema=QueryDocumentsToolInput, return_direct=False)
def query_documents_tool(
    query: str,
    top_k: int = 3,
    include_values: bool = False,
    name_space: str | None = None
) -> list[DocumentsChunkSchema]:
    """
    Semantic‐search your Pinecone index of code/text chunks.

    Returns:
      A list of LangChain Documents, each with:
        - .metadata: all your stored metadata
            - .metadata.chunk_index: the chunk number in the original document (0-based)
            - .metadata.file_path: the path to the file in the original repo
            - .metadata.language: the language of the chunk (e.g. "python")
            - .metadata.id: the id of the chunk in the index
            - .metadata.score: the similarity score of this chunk to the query
            - .metadata.values: the raw embedding values (if include_values=True) This should rarely to never be used since it's not needed for most use cases.
            - .metadata.total_chunks: total number of chunks the parent file has been split into
        - .page_content: the text of the chunk

    """ 
    # 1) embed the query
    q_resp = openai_client.embeddings.create(
        input=[query],
        model=_MODEL
    )
    q_vec = q_resp.data[0].embedding

    # 2) ask Pinecone for the closest vectors
    index_name = os.getenv("DOCUMENTS_VDB_INDEX")
    idx = pc_client.Index(index_name)
    query_response = idx.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
        include_values=include_values,
        namespace=name_space
    )

    # 3) build Documents
    results = []
    for match in query_response.get("matches", []):
        # copy so we can pop
        md = match.get("metadata", {}).copy()

        # pull out the text for page_content…
        text = md.pop("text", md.pop("embedding_text", ""))

        # …and drop it from metadata
        md["id"]    = match.get("id")
        md["score"] = match.get("score")
        if include_values:
            md["values"] = match.get("values", [])

        results.append(Document(page_content=text, metadata=md))

    return [DocumentsChunkSchema(page_content=d.page_content, metadata=d.metadata)
            for d in results]


class QueryIssuesToolInput(BaseModel):
    query: str = Field(description="The natural‐language or code snippet you want to find similar docs for")
    top_k: int = Field(default=3, description="How many results to return")
    include_values: bool = Field(default=False, description="If True, returns the raw embedding values in metadata")
    name_space: str | None = Field(default=None, description="Subgroup of the index to target")
@tool("query_issues_tool", args_schema=QueryIssuesToolInput, return_direct=False)
def query_issues_tool(
    query: str,
    top_k: int = 3,
    include_values: bool = False,
    name_space: str | None = None
) -> list[IssueChunkSchema]:
    """
    Semantic‐search your Pinecone index of issue‐chunks.

    Returns a list of LangChain Documents with:
    - .metadata: all your stored metadata
        - .metadat.author : author of the issue
        - .metadata.chunk_index: the chunk number in the original issue (0-based)
        - .metadata.created_at: the timestamp of the issue
        - .metadata.labels: the labels of the issue
        - .metadata.title: the title of the issue
        - .metadata.id: the id of the issue in the index
        - .metadata.score: the similarity score of this chunk to the query
        - .metadata.total_chunks: total number of chunks the parent issue has been split into
    - .page_content : the text body of the issue
    """
    # 1) embed the query
    q_resp = openai_client.embeddings.create(
        input=[query],
        model=_MODEL
    )
    q_vec = q_resp.data[0].embedding

    # 2) query Pinecone
    index_name = os.getenv("ISSUES_VDB_INDEX")
    idx = pc_client.Index(index_name)
    resp = idx.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
        include_values=include_values,
        namespace=name_space
    ).to_dict()

    # 3) build Documents
    results = []
    for match in resp.get("matches", []):
        md = match.get("metadata", {}).copy()

        # pull out the issue text
        page_text = md.pop("embedding_text", "")

        # inject vector info
        md["id"]    = match.get("id")
        md["score"] = match.get("score")
        if include_values:
            md["values"] = match.get("values", [])

        results.append(Document(page_content=page_text, metadata=md))

    return [IssueChunkSchema(page_content=d.page_content, metadata=d.metadata)
            for d in results]
