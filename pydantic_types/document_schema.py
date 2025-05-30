from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

# --- 1️⃣  define a closed schema for metadata -------------
class DocumentsMetadataSchema(BaseModel):
    file_path: str  = Field(..., description="Path to the file in the repo")
    language: str   = Field(..., description="Programming language of the chunk")
    chunk_index: Optional[int] = Field(None, description="Index of the chunk in its file")
    total_chunks: Optional[int] = Field(None, description="Number of chunks in parent file")
    id: Optional[str] = Field(None, description="Id of the current chunk in the vector database")
    score:       Optional[float] = Field(None, description="Similarity score returned by the retriever")

    model_config = ConfigDict(extra="forbid")  # <- generates additionalProperties: false

# --- 2️⃣  describe one chunk --------------------------------
class DocumentsChunkSchema(BaseModel):
    page_content: str          = Field(..., description="The text of the chunk")
    metadata:      DocumentsMetadataSchema

    model_config = ConfigDict(extra="forbid")

# --- 3️⃣  top-level list wrapper ----------------------------
class DocumentList(BaseModel):
    documents: list[DocumentsChunkSchema]

    model_config = ConfigDict(extra="forbid")
