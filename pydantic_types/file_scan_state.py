from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from pydantic_types.document_schema import DocumentsChunkSchema, DocumentList


class FileScanState(TypedDict):
    file_prefixes_to_explore: list[str]
    files_current_prefix: DocumentList
    scanned_prefixes: list[str]
    
    