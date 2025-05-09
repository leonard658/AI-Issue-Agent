from pydantic import BaseModel, Field
import tiktoken
from langchain.schema import Document
from langchain_core.tools import tool

class CombineChunksToolInput(BaseModel):
    chunks: str = Field(description='Documents with .page_content and metadata["chunk_index"].')
    max_tokens: int = Field(description="Maximum tokens allowed (default 3000).")
    model: str = Field(description='Model name for tokenization (default "gpt-4").')
@tool("combine_chunks_tool", args_schema=CombineChunksToolInput, return_direct=False)
def combine_chunks_tool(
    chunks: list[Document],
    max_tokens: int = 3000,
    model: str = "gpt-4"
) -> str | None:
    """
    Merge a list of code chunks into one block, up to a max token budget.

    Args:
        chunks:      Documents with .page_content and metadata["chunk_index"].
        max_tokens:  Maximum tokens allowed (default 3000).
        model:       Model name for tokenization (default "gpt-4").

    Returns:
        The merged code string if under the token limit, otherwise returns None.
    """
    # 1) Sort by chunk_index
    sorted_chunks = sorted(chunks, key=lambda c: c.metadata["chunk_index"])
    
    # 2) Concatenate
    combined = "\n".join(chunk.page_content for chunk in sorted_chunks)
    
    # 3) Tokenize and check length
    enc = tiktoken.encoding_for_model(model)
    token_count = len(enc.encode(combined))
    
    if token_count > max_tokens:
        # Handle oversize: log and return None
        print(f"[Error] Combined code is {token_count} tokens, exceeds limit of {max_tokens}.")
        return None
    
    return combined
