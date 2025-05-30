import json
from pydantic import BaseModel
from typing import Any

def to_json_str(obj: Any, indent: int = 2) -> str:
    """
    Serialize a Pydantic model (or any object with .model_dump() / .dict())
    to a pretty-printed JSON string.

    :param obj: A Pydantic BaseModel instance (or list/dict of them).
    :param indent: Number of spaces to use for indentation.
    :return: A JSON-formatted string.
    """
    # If it’s a BaseModel, use model_dump(); if v1, fallback to dict()
    if isinstance(obj, BaseModel):
        data = obj.model_dump()

    else:
        # If it’s already a dict or list, assume it’s serializable
        data = obj

    return json.dumps(data, indent=indent)
