from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

# --- 1️⃣  define a closed schema for metadata -------------
class IssueMetadataSchema(BaseModel):
    title: str  = Field(..., description="Title of the issue")
    author: str   = Field(..., description="Author of the issue")
    chunk_index: Optional[int] = Field(None, description="Index of the chunk in its file")
    total_chunks: Optional[int] = Field(None, description="Number of chunks in parent file")
    id: Optional[str] = Field(None, description="Id of the current chunk in the vector database")
    score:       Optional[float] = Field(None, description="Similarity score returned by the retriever")
    created_at: Optional[str] = Field(None, description="Datetime of what the parent issue was created in ISO format")
    labels: Optional[list[str]] = Field(None, description="labels attached to the parent issue from github")

    model_config = ConfigDict(extra="forbid")  # <- generates additionalProperties: false

# --- 2️⃣  describe one chunk --------------------------------
class IssueChunkSchema(BaseModel):
    page_content: str          = Field(..., description="The text of the chunk")
    metadata:      IssueMetadataSchema

    model_config = ConfigDict(extra="forbid")

# --- 3️⃣  top-level list wrapper ----------------------------
class IssueList(BaseModel):
    documents: List[IssueChunkSchema]

    model_config = ConfigDict(extra="forbid")
